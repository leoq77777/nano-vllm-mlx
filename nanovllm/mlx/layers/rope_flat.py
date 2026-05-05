"""RoPE for flattened prefill/decode batches (offsets match ``mlx_lm`` / ``nn.RoPE``)."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def apply_rope_segmented(
    rope_module: Any,
    q: mx.array,
    k: mx.array,
    positions: mx.array,
    *,
    is_prefill: bool,
    cu_seqlens_q: list[int],
) -> tuple[mx.array, mx.array]:
    """
    Apply the model's RoPE module with **per-segment** offsets (chunked prefill + multi-seq).

    - Prefill: ``q``/``k`` are (1, heads, T, d); ``positions`` length ``T``; each segment
      ``[a, b)`` from ``cu_seqlens_q`` uses ``offset = int(positions[a])``.
    - Decode: ``q``/``k`` are (1, heads, B, d); ``positions`` length ``B``; one token per
      row with ``offset = int(positions[bi])``.
    """
    if is_prefill:
        qs: list[mx.array] = []
        ks: list[mx.array] = []
        for s in range(len(cu_seqlens_q) - 1):
            a, b = cu_seqlens_q[s], cu_seqlens_q[s + 1]
            off = int(positions[a].item())
            qs.append(rope_module(q[:, :, a:b, :], offset=off))
            ks.append(rope_module(k[:, :, a:b, :], offset=off))
        return mx.concatenate(qs, axis=2), mx.concatenate(ks, axis=2)

    B = q.shape[2]
    qs = []
    ks = []
    for bi in range(B):
        off = int(positions[bi].item())
        qs.append(rope_module(q[:, :, bi : bi + 1, :], offset=off))
        ks.append(rope_module(k[:, :, bi : bi + 1, :], offset=off))
    return mx.concatenate(qs, axis=2), mx.concatenate(ks, axis=2)
