"""MLX runtime configuration (scheduling + KV cache sizing)."""

from __future__ import annotations

from dataclasses import dataclass


def kv_cache_bytes_per_block(
    *,
    num_hidden_layers: int,
    num_kv_heads: int,
    head_dim: int,
    kvcache_block_size: int,
    dtype_size: int = 2,
) -> int:
    """Bytes per physical KV block (K + V), row-major logical layout."""
    per_layer = 2 * kvcache_block_size * num_kv_heads * head_dim * dtype_size
    return num_hidden_layers * per_layer


def metal_kv_budget_bytes(*, memory_fraction: float = 0.9, subtract_active_cache: bool = True) -> int:
    """Rough Metal memory budget for KV pools (uses MLX device info + active/cache)."""
    try:
        import mlx.core as mx  # type: ignore
    except Exception:
        pool = 8 << 30
        return max(int(pool * memory_fraction), 1 << 20)

    try:
        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    except Exception:
        info = {"memory_size": 8 << 30, "max_recommended_working_set_size": 6 << 30}

    raw = info.get("max_recommended_working_set_size") or info.get("memory_size") or (8 << 30)
    pool = max(int(raw * memory_fraction), 1 << 20)

    if not subtract_active_cache:
        return pool

    try:
        active_fn = getattr(mx, "get_active_memory", None) or getattr(mx.metal, "get_active_memory", None)
        cache_fn = getattr(mx, "get_cache_memory", None) or getattr(mx.metal, "get_cache_memory", None)
        active = int(active_fn()) if active_fn else 0
        cache = int(cache_fn()) if cache_fn else 0
    except Exception:
        active, cache = 0, 0

    return max(pool - active - cache, 1 << 20)


def suggest_num_kv_cache_blocks(
    bytes_per_block: int,
    *,
    memory_fraction: float = 0.5,
    reserved_bytes: int = 0,
    subtract_active_cache: bool = True,
) -> int:
    """How many KV blocks likely fit, given per-block footprint and a conservative budget."""
    if bytes_per_block <= 0:
        return 1
    budget = metal_kv_budget_bytes(memory_fraction=memory_fraction, subtract_active_cache=subtract_active_cache)
    usable = max(budget - reserved_bytes, bytes_per_block)
    return max(usable // bytes_per_block, 1)


@dataclass(slots=True)
class MLXConfig:
    model: str
    cache_limit_gb: int = 20
    tensor_parallel_size: int = 1
    progress: bool = False

    # Scheduling / length limits (aligned with CUDA engine knobs)
    max_num_batched_tokens: int = 8192
    max_num_seqs: int = 256
    max_model_len: int = 4096
    kvcache_block_size: int = 256

    # Paged KV pool: None → auto from Metal budget when geometry is known
    num_kvcache_blocks: int | None = None

    # Optional model geometry for KV byte accounting (Phase 2+ will populate from weights)
    num_hidden_layers: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None
    kv_dtype_size: int = 2  # fp16/bf16 style element size for budgeting

    # Heuristic sizing when geometry unknown (Phase 1 default pool)
    kv_memory_fraction: float = 0.5
    kv_reserved_bytes: int = 0
    fallback_num_kvcache_blocks: int = 512

    def kv_bytes_per_block(self) -> int | None:
        if (
            self.num_hidden_layers is None
            or self.num_kv_heads is None
            or self.head_dim is None
        ):
            return None
        return kv_cache_bytes_per_block(
            num_hidden_layers=self.num_hidden_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            kvcache_block_size=self.kvcache_block_size,
            dtype_size=self.kv_dtype_size,
        )

    def resolve_num_kv_cache_blocks(self) -> int:
        """Pinned count for BlockManager: explicit, auto from geometry, or heuristic fallback."""
        if self.num_kvcache_blocks is not None:
            return max(self.num_kvcache_blocks, 1)
        bpb = self.kv_bytes_per_block()
        if bpb is not None:
            return suggest_num_kv_cache_blocks(
                bpb,
                memory_fraction=self.kv_memory_fraction,
                reserved_bytes=self.kv_reserved_bytes,
            )
        return max(self.fallback_num_kvcache_blocks, 1)

    def apply_sequence_block_size(self) -> None:
        """Keep ``Sequence`` token blocking in sync with ``kvcache_block_size`` (same as CUDA LLMEngine)."""
        from nanovllm.mlx.sequence import Sequence

        Sequence.block_size = self.kvcache_block_size


def apply_qwen_mlx_arch_to_mlx_config(mlx_cfg: MLXConfig, arch: dict) -> None:
    """Fill KV geometry fields on ``MLXConfig`` from a Qwen3 ``config.json`` (MLX or HF-style)."""
    from nanovllm.mlx.models.qwen3 import Qwen3MLXModelArgs

    args = Qwen3MLXModelArgs.from_dict(arch)
    mlx_cfg.num_hidden_layers = args.num_hidden_layers
    mlx_cfg.num_kv_heads = args.num_key_value_heads
    mlx_cfg.head_dim = args.head_dim
    if getattr(args, "max_position_embeddings", None):
        mlx_cfg.max_model_len = min(mlx_cfg.max_model_len, int(args.max_position_embeddings))
