"""
Qwen3 decoder-only stack for MLX (weight layout compatible with ``mlx_lm`` Qwen3).

Attention is split into ``project_qkv`` → RoPE → ``dot_product_attention`` so Phase 3
can supply externally gathered ``keys`` / ``values`` before SDPA without changing
projection / output layers.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from mlx_lm.models.rope_utils import initialize_rope


@dataclass
class Qwen3MLXModelArgs(BaseModelArgs):
    """Same fields as ``mlx_lm.models.qwen3.ModelArgs`` (config.json compatible)."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    rms_norm_eps: float
    vocab_size: int
    num_key_value_heads: int
    max_position_embeddings: int
    rope_theta: float
    head_dim: int
    tie_word_embeddings: bool
    rope_scaling: Optional[dict[str, Any]] = None


class Qwen3MLXAttention(nn.Module):
    def __init__(self, args: Qwen3MLXModelArgs):
        super().__init__()
        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        assert args.num_key_value_heads is not None
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        head_dim = args.head_dim
        self.scale = head_dim**-0.5

        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def project_qkv(self, x: mx.array) -> tuple[mx.array, mx.array, mx.array]:
        """``x`` (B, L, D) → ``queries``, ``keys``, ``values`` (B, n_h, L, Dh) / (B, n_kv, L, Dh)."""
        B, L, _D = x.shape
        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(0, 2, 1, 3)
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        return queries, keys, values

    def apply_rope(
        self,
        queries: mx.array,
        keys: mx.array,
        cache: Any | None,
    ) -> tuple[mx.array, mx.array]:
        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)
        return queries, keys

    def dot_product_attention(
        self,
        queries: mx.array,
        keys: mx.array,
        values: mx.array,
        mask: mx.array | str | None,
        cache: Any | None,
    ) -> mx.array:
        """Phase 3: pass gathered ``keys``/``values`` (same layout) instead of projected cache."""
        return scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

    def output_proj(self, attn_out: mx.array) -> mx.array:
        """``attn_out`` (B, L, n_heads * Dh) → (B, L, D)."""
        return self.o_proj(attn_out)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
        *,
        keys_override: mx.array | None = None,
        values_override: mx.array | None = None,
    ) -> mx.array:
        B, L, _ = x.shape
        queries, keys, values = self.project_qkv(x)
        queries, keys = self.apply_rope(queries, keys, cache)
        if keys_override is not None:
            keys = keys_override
        if values_override is not None:
            values = values_override
        output = self.dot_product_attention(queries, keys, values, mask, cache)
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.output_proj(output)


class Qwen3MLXMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class Qwen3MLXTransformerBlock(nn.Module):
    def __init__(self, args: Qwen3MLXModelArgs):
        super().__init__()
        self.self_attn = Qwen3MLXAttention(args)
        self.mlp = Qwen3MLXMLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | str | None = None,
        cache: Any | None = None,
        *,
        keys_override: mx.array | None = None,
        values_override: mx.array | None = None,
    ) -> mx.array:
        r = self.self_attn(
            self.input_layernorm(x),
            mask,
            cache,
            keys_override=keys_override,
            values_override=values_override,
        )
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3MLXInnerModel(nn.Module):
    def __init__(self, args: Qwen3MLXModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        self.num_hidden_layers = args.num_hidden_layers
        assert self.vocab_size > 0
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Qwen3MLXTransformerBlock(args=args) for _ in range(args.num_hidden_layers)]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
        *,
        attn_keys_override: list[mx.array | None] | None = None,
        attn_values_override: list[mx.array | None] | None = None,
    ):
        if input_embeddings is not None:
            h = input_embeddings
        else:
            h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)
        if attn_keys_override is None:
            attn_keys_override = [None] * len(self.layers)
        if attn_values_override is None:
            attn_values_override = [None] * len(self.layers)

        mask = create_attention_mask(h, cache[0])

        for layer, c, ko, vo in zip(self.layers, cache, attn_keys_override, attn_values_override):
            h = layer(h, mask, c, keys_override=ko, values_override=vo)

        return self.norm(h)


class Qwen3MLXForCausalLM(nn.Module):
    """Top-level module: attribute names match ``mlx_lm.models.qwen3.Model`` for weight loading."""

    def __init__(self, args: Qwen3MLXModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3MLXInnerModel(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache: list | None = None,
        input_embeddings: mx.array | None = None,
        *,
        attn_keys_override: list[mx.array | None] | None = None,
        attn_values_override: list[mx.array | None] | None = None,
    ):
        out = self.model(
            inputs,
            cache,
            input_embeddings,
            attn_keys_override=attn_keys_override,
            attn_values_override=attn_values_override,
        )
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def sanitize(self, weights: dict):
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    def compute_logits(self, hidden: mx.array) -> mx.array:
        """Project hidden states (…, ``hidden_size``) to logits; not the output of ``__call__``."""
        if self.args.tie_word_embeddings:
            return self.model.embed_tokens.as_linear(hidden)
        return self.lm_head(hidden)

    @property
    def layers(self):
        return self.model.layers


def _load_safetensors(model_path: Path) -> dict[str, mx.array]:
    weights: dict[str, mx.array] = {}
    for wf in sorted(glob.glob(str(model_path / "model*.safetensors"))):
        weights.update(mx.load(wf))
    return weights


def load_qwen3_mlx_from_path(
    model_path: str | Path,
    *,
    lazy: bool = False,
    strict: bool = True,
) -> tuple[Qwen3MLXForCausalLM, dict[str, Any]]:
    """
    Load ``Qwen3MLXForCausalLM`` from a local MLX model directory (``config.json`` + ``model*.safetensors``).

    Quantized checkpoints are not supported in this loader yet; use ``mlx_lm.utils.load``
    and migrate weights if you need quantized models.
    """
    from mlx_lm.utils import load_config as mlx_load_config

    model_path = Path(model_path)
    config = mlx_load_config(model_path)
    if config.get("quantization") is not None or config.get("quantization_config"):
        raise NotImplementedError(
            "Quantized MLX weights are not supported by load_qwen3_mlx_from_path yet; "
            "use a bf16/fp16 converted model or extend this loader."
        )

    weight_files = glob.glob(str(model_path / "model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No model*.safetensors under {model_path}")

    weights = _load_safetensors(model_path)
    model_args = Qwen3MLXModelArgs.from_dict(config)
    model = Qwen3MLXForCausalLM(model_args)
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)
    model.eval()
    model.load_weights(list(weights.items()), strict=strict)
    if not lazy:
        mx.eval(model.parameters())

    return model, config


def qwen3_mlx_model_args_from_config(config: dict) -> Qwen3MLXModelArgs:
    """Build args from a raw ``config.json`` dict (e.g. to fill ``MLXConfig`` geometry)."""
    return Qwen3MLXModelArgs.from_dict(config)
