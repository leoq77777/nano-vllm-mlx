"""Runner context for MLX continuous batching (replaces CUDA thread-local context)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import mlx.core as mx


@dataclass(slots=True)
class MLXRunnerContext:
    is_prefill: bool
    cu_seqlens_q: list[int]
    cu_seqlens_k: list[int]
    max_seqlen_q: int
    max_seqlen_k: int
    slot_mapping: mx.array
    context_lens: mx.array | None
    block_tables: mx.array | None
    positions: mx.array
    seqs: list[Any]  # Sequence objects for this step (slot / block metadata)
    # Preallocated buffers (Phase 7); optional until runner sets them
    slot_buffer: mx.array | None = None


_CTX: MLXRunnerContext | None = None


def set_mlx_runner_context(ctx: MLXRunnerContext | None) -> None:
    global _CTX
    _CTX = ctx


def get_mlx_runner_context() -> MLXRunnerContext | None:
    return _CTX


def reset_mlx_runner_context() -> None:
    set_mlx_runner_context(None)
