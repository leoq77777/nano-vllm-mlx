"""MLX model runner: paged KV, prefill/decode batches (subset of CUDA ``ModelRunner``)."""

from __future__ import annotations

from pathlib import Path

import mlx.core as mx

from nanovllm.mlx.config import MLXConfig
from nanovllm.mlx.context import MLXRunnerContext, reset_mlx_runner_context, set_mlx_runner_context
from nanovllm.mlx.layers.attention import (
    block_table_prefix_slots,
    gather_kv_from_slots,
    scatter_kv_tokens,
    scatter_kv_tokens_batched,
)
from nanovllm.mlx.layers.rope_flat import apply_rope_segmented
from nanovllm.mlx.models.qwen3 import Qwen3MLXForCausalLM, Qwen3MLXModelArgs
from nanovllm.mlx.sampler import mlx_sample_tokens
from nanovllm.mlx.sequence import Sequence


def build_prefill_mask(ctx: MLXRunnerContext, total_q: int, total_k: int) -> mx.array:
    """(1,1,total_q,total_k) boolean: same sequence segment and key position <= query position."""
    q_pos_np = ctx.positions.tolist()
    seg_q_l = [0] * total_q
    seg_k_l = [0] * total_k
    k_pos_l = [0] * total_k
    for s in range(len(ctx.cu_seqlens_q) - 1):
        q0, q1 = ctx.cu_seqlens_q[s], ctx.cu_seqlens_q[s + 1]
        k0, k1 = ctx.cu_seqlens_k[s], ctx.cu_seqlens_k[s + 1]
        for i in range(q0, q1):
            seg_q_l[i] = s
        for j in range(k0, k1):
            seg_k_l[j] = s
            k_pos_l[j] = j - k0
    seg_q = mx.array(seg_q_l, dtype=mx.int32)
    seg_k = mx.array(seg_k_l, dtype=mx.int32)
    k_pos = mx.array(k_pos_l, dtype=mx.int32)
    q_pos = mx.array(q_pos_np, dtype=mx.int32)
    same = seg_q[:, None] == seg_k[None, :]
    causal = k_pos[None, :] <= q_pos[:, None]
    return (same & causal)[None, None, :, :]


class MLXModelRunner:
    def __init__(self, config: MLXConfig, model: Qwen3MLXForCausalLM, arch: dict):
        self.config = config
        self.model = model
        self.args = Qwen3MLXModelArgs.from_dict(arch)
        self.block_size = config.kvcache_block_size
        self.dtype = mx.float16
        nb = config.resolve_num_kv_cache_blocks()
        self.k_caches: list[mx.array] = []
        self.v_caches: list[mx.array] = []
        from nanovllm.mlx.layers.attention import kv_cache_slots

        for _ in range(self.args.num_hidden_layers):
            k, v = kv_cache_slots(nb, self.block_size, self.args.num_key_value_heads, self.args.head_dim, self.dtype)
            self.k_caches.append(k)
            self.v_caches.append(v)
        self._scale = float(self.args.head_dim) ** -0.5
        self._slot_buf = mx.zeros((self.config.max_num_batched_tokens,), dtype=mx.int32)
        # ``mlx_compile_decode``: MLX ``mx.compile`` currently requires array-only args; runner
        # context is a Python dataclass — compile is deferred until we pack ctx into flat buffers.

    @staticmethod
    def _use_batched_scatter(slot_mapping: mx.array) -> bool:
        return int(slot_mapping.size) > 0 and not bool(mx.any(slot_mapping < 0).item())

    def prepare_block_tables(self, seqs: list[Sequence]) -> mx.array:
        max_len = max(len(seq.block_table) for seq in seqs)
        rows = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        return mx.array(rows, dtype=mx.int32)

    def prepare_prefill(self, seqs: list[Sequence]) -> tuple[mx.array, mx.array, MLXRunnerContext]:
        input_ids: list[int] = []
        positions: list[int] = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping: list[int] = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            start = min(seq.num_cached_tokens, seqlen - 1)
            seqlen_q = seq.num_scheduled_tokens
            seqlen_k = seqlen
            end = start + seqlen_q
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:
            block_tables = self.prepare_block_tables(seqs)
        tq = len(input_ids)
        ids = mx.array(input_ids, dtype=mx.int32)[None, :]
        pos = mx.array(positions, dtype=mx.int32)
        if slot_mapping:
            sm = mx.array(slot_mapping, dtype=mx.int32)
        else:
            sm = mx.full((tq,), -1, dtype=mx.int32)
        ctx = MLXRunnerContext(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=sm,
            context_lens=None,
            block_tables=block_tables,
            positions=pos,
            seqs=seqs,
            slot_buffer=self._slot_buf,
        )
        return ids, pos, ctx

    def prepare_decode(self, seqs: list[Sequence]) -> tuple[mx.array, mx.array, MLXRunnerContext]:
        input_ids: list[int] = []
        positions: list[int] = []
        slot_mapping: list[int] = []
        context_lens: list[int] = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens - 1)
        b = len(seqs)
        ids = mx.array(input_ids, dtype=mx.int32)[None, :]
        pos = mx.array(positions, dtype=mx.int32)
        sm = mx.array(slot_mapping, dtype=mx.int32)
        cl = mx.array(context_lens, dtype=mx.int32)
        bt = self.prepare_block_tables(seqs)
        ctx = MLXRunnerContext(
            is_prefill=False,
            cu_seqlens_q=[0, b],
            cu_seqlens_k=[0, b],
            max_seqlen_q=1,
            max_seqlen_k=max(context_lens),
            slot_mapping=sm,
            context_lens=cl,
            block_tables=bt,
            positions=pos,
            seqs=seqs,
            slot_buffer=self._slot_buf,
        )
        return ids, pos, ctx

    def _gather_kv_concat(self, layer_idx: int, ctx: MLXRunnerContext, total_k: int) -> tuple[mx.array, mx.array]:
        """Stack keys/values in concat-key order (length ``total_k``)."""
        k_buf = self.k_caches[layer_idx]
        v_buf = self.v_caches[layer_idx]
        parts_k: list[mx.array] = []
        parts_v: list[mx.array] = []
        off = 0
        for s, seq in enumerate(ctx.seqs):
            k0, k1 = ctx.cu_seqlens_k[s], ctx.cu_seqlens_k[s + 1]
            tk = k1 - k0
            slots = block_table_prefix_slots(seq.block_table, self.block_size, 0, tk)
            kg, vg = gather_kv_from_slots(k_buf, v_buf, slots)
            parts_k.append(kg)
            parts_v.append(vg)
            off += tk
        assert off == total_k, (off, total_k)
        k_cat = mx.concatenate(parts_k, axis=0)
        v_cat = mx.concatenate(parts_v, axis=0)
        nkv, dh = k_cat.shape[-2], k_cat.shape[-1]
        k_cat = k_cat.transpose(1, 0, 2)[None, :, :, :]
        v_cat = v_cat.transpose(1, 0, 2)[None, :, :, :]
        return k_cat, v_cat

    def _decoder_block_prefill(self, layer, layer_idx: int, h: mx.array, ctx: MLXRunnerContext) -> mx.array:
        x = layer.input_layernorm(h)
        attn = layer.self_attn
        q, k, v = attn.project_qkv(x)
        q, k = apply_rope_segmented(attn.rope, q, k, ctx.positions, is_prefill=True, cu_seqlens_q=ctx.cu_seqlens_q)
        t = h.shape[1]
        nh, nkv, dh = attn.n_heads, attn.n_kv_heads, self.args.head_dim
        kf = k.transpose(0, 2, 1, 3).reshape(t, nkv, dh)
        vf = v.transpose(0, 2, 1, 3).reshape(t, nkv, dh)
        k_buf, v_buf = self.k_caches[layer_idx], self.v_caches[layer_idx]
        if self._use_batched_scatter(ctx.slot_mapping):
            nk, nv = scatter_kv_tokens_batched(kf, vf, k_buf, v_buf, ctx.slot_mapping)
        else:
            nk, nv = scatter_kv_tokens(kf, vf, k_buf, v_buf, ctx.slot_mapping)
        self.k_caches[layer_idx], self.v_caches[layer_idx] = nk, nv

        total_k = ctx.cu_seqlens_k[-1]
        total_q = t
        kg, vg = self._gather_kv_concat(layer_idx, ctx, total_k)
        mask = build_prefill_mask(ctx, total_q, total_k)
        o = mx.fast.scaled_dot_product_attention(q, kg, vg, scale=self._scale, mask=mask)
        out = o.transpose(0, 2, 1, 3).reshape(1, t, nh * dh)
        out = attn.output_proj(out)
        h = h + out
        out2 = layer.mlp(layer.post_attention_layernorm(h))
        return h + out2

    def _decoder_block_decode(self, layer, layer_idx: int, h: mx.array, ctx: MLXRunnerContext) -> mx.array:
        x = layer.input_layernorm(h)
        attn = layer.self_attn
        q, k, v = attn.project_qkv(x)
        q, k = apply_rope_segmented(attn.rope, q, k, ctx.positions, is_prefill=False, cu_seqlens_q=ctx.cu_seqlens_q)
        b = h.shape[1]
        nh, nkv, dh = attn.n_heads, attn.n_kv_heads, self.args.head_dim
        kf = k.transpose(0, 2, 1, 3).reshape(b, nkv, dh)
        vf = v.transpose(0, 2, 1, 3).reshape(b, nkv, dh)
        k_buf, v_buf = self.k_caches[layer_idx], self.v_caches[layer_idx]
        if self._use_batched_scatter(ctx.slot_mapping):
            nk, nv = scatter_kv_tokens_batched(kf, vf, k_buf, v_buf, ctx.slot_mapping)
        else:
            nk, nv = scatter_kv_tokens(kf, vf, k_buf, v_buf, ctx.slot_mapping)
        self.k_caches[layer_idx], self.v_caches[layer_idx] = nk, nv
        k_buf, v_buf = self.k_caches[layer_idx], self.v_caches[layer_idx]

        o = mx.zeros((1, nh, b, dh), dtype=h.dtype)
        for bi, seq in enumerate(ctx.seqs):
            L = len(seq)
            slots = block_table_prefix_slots(seq.block_table, self.block_size, 0, L)
            kg, vg = gather_kv_from_slots(k_buf, v_buf, slots)
            kg = kg.transpose(1, 0, 2)[None, :, :, :]
            vg = vg.transpose(1, 0, 2)[None, :, :, :]
            qb = q[:, :, bi : bi + 1, :]
            seg = mx.fast.scaled_dot_product_attention(qb, kg, vg, scale=self._scale, mask=None)
            o[:, :, bi : bi + 1, :] = seg
        out = o.transpose(0, 2, 1, 3).reshape(1, b, nh * dh)
        out = attn.output_proj(out)
        h = h + out
        out2 = layer.mlp(layer.post_attention_layernorm(h))
        return h + out2

    def forward(self, input_ids: mx.array, positions: mx.array, ctx: MLXRunnerContext) -> mx.array:
        h = self.model.model.embed_tokens(input_ids)
        for li, layer in enumerate(self.model.model.layers):
            if ctx.is_prefill:
                h = self._decoder_block_prefill(layer, li, h, ctx)
            else:
                h = self._decoder_block_decode(layer, li, h, ctx)
        return self.model.model.norm(h)

    def run_model(self, input_ids: mx.array, positions: mx.array, ctx: MLXRunnerContext) -> mx.array:
        set_mlx_runner_context(ctx)
        try:
            h = self.forward(input_ids, positions, ctx)
            logits = self.model.compute_logits(h)
            return logits
        finally:
            reset_mlx_runner_context()

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        if is_prefill:
            ids, pos, ctx = self.prepare_prefill(seqs)
        else:
            ids, pos, ctx = self.prepare_decode(seqs)
        logits = self.run_model(ids, pos, ctx)
        temps = mx.array([seq.temperature for seq in seqs], dtype=mx.float32)
        if is_prefill:
            ends = mx.array([ctx.cu_seqlens_q[i + 1] - 1 for i in range(len(seqs))], dtype=mx.int32)
            rows = logits[0, ends, :]
        else:
            rows = logits[0]
        return [int(x) for x in mlx_sample_tokens(rows, temps).tolist()]


def load_mlx_arch(model_path: str | Path) -> dict:
    from mlx_lm.utils import load_config

    return load_config(Path(model_path))
