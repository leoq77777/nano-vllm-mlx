"""MLX sampling (Gumbel-max / softmax-div-exponential), ported from ``layers/sampler.py``."""

from __future__ import annotations

import mlx.core as mx


def mlx_sample_tokens(logits: mx.array, temperatures: mx.array) -> mx.array:
    """
    ``logits`` (B, V), ``temperatures`` (B,) — same contract as CUDA ``Sampler``.
    Returns (B,) int32 token ids.
    """
    t = mx.maximum(temperatures.astype(mx.float32), mx.array(1e-10, dtype=mx.float32))
    t = t[:, None]
    scaled = logits.astype(mx.float32) / t
    probs = mx.softmax(scaled, axis=-1)
    u = mx.maximum(mx.random.uniform(shape=probs.shape), mx.array(1e-10, dtype=probs.dtype))
    exp = -mx.log(u)
    exp = mx.maximum(exp, mx.array(1e-10, dtype=probs.dtype))
    return mx.argmax(probs / exp, axis=-1).astype(mx.int32)
