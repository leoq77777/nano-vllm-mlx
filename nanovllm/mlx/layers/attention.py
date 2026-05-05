"""Paged KV scatter / gather helpers for MLX (Phase 3 + Phase 7 batch paths)."""

from __future__ import annotations

import mlx.core as mx


def kv_cache_slots(num_blocks: int, block_size: int, num_kv: int, head_dim: int, dtype) -> tuple[mx.array, mx.array]:
    """Per-layer K/V buffers shaped like CUDA: (num_blocks, block_size, num_kv, head_dim)."""
    shape = (num_blocks, block_size, num_kv, head_dim)
    k = mx.zeros(shape, dtype=dtype)
    v = mx.zeros(shape, dtype=dtype)
    return k, v


def scatter_kv_tokens(
    k_states: mx.array,
    v_states: mx.array,
    k_buf: mx.array,
    v_buf: mx.array,
    slot_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """
    Write token-aligned K/V into paged buffers.

    ``k_states``/``v_states``: (T, n_kv, d); ``slot_mapping``: (T,) int32; skip slots ``< 0``.
    """
    t = k_states.shape[0]
    _, block_size, _, _ = k_buf.shape
    k_out, v_out = k_buf, v_buf
    for i in range(t):
        s = int(slot_mapping[i])
        if s < 0:
            continue
        blk = s // block_size
        off = s % block_size
        k_out[blk, off] = k_states[i]
        v_out[blk, off] = v_states[i]
    return k_out, v_out


def scatter_kv_tokens_batched(
    k_states: mx.array,
    v_states: mx.array,
    k_buf: mx.array,
    v_buf: mx.array,
    slot_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """Vectorized scatter when indices are valid (Phase 7)."""
    flat_k = k_buf.reshape(-1, k_buf.shape[-2], k_buf.shape[-1])
    flat_v = v_buf.reshape(-1, v_buf.shape[-2], v_buf.shape[-1])
    idx = slot_mapping.astype(mx.int32)
    flat_k[idx] = k_states
    flat_v[idx] = v_states
    return flat_k.reshape(k_buf.shape), flat_v.reshape(v_buf.shape)


def gather_kv_from_slots(
    k_buf: mx.array,
    v_buf: mx.array,
    slot_indices: list[int],
) -> tuple[mx.array, mx.array]:
    """Gather (L, n_kv, d) for physical slot indices (Route A)."""
    flat_k = k_buf.reshape(-1, k_buf.shape[-2], k_buf.shape[-1])
    flat_v = v_buf.reshape(-1, v_buf.shape[-2], v_buf.shape[-1])
    if not slot_indices:
        d = int(flat_k.shape[-1])
        nkv = int(flat_k.shape[-2])
        z = flat_k.dtype
        return mx.zeros((0, nkv, d), dtype=z), mx.zeros((0, nkv, d), dtype=z)
    idx = mx.array(slot_indices, dtype=mx.int32)
    return flat_k[idx], flat_v[idx]


def block_table_prefix_slots(
    block_table: list[int],
    block_size: int,
    start_token: int,
    end_token: int,
) -> list[int]:
    """Flat slots for token indices ``[start_token, end_token)``."""
    slots: list[int] = []
    for t in range(start_token, end_token):
        b = t // block_size
        o = t % block_size
        slots.append(block_table[b] * block_size + o)
    return slots
