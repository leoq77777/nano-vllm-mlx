#!/usr/bin/env python3
"""Verify MLX continuous-batching path (Phases 1–7): multi-prompt + decode steps."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "model_path",
        nargs="?",
        default=os.environ.get("NANO_MLX_CB_MODEL", ""),
        help="Local MLX model dir (bf16/fp16). Default: $NANO_MLX_CB_MODEL",
    )
    p.add_argument("--max-tokens", type=int, default=6)
    args = p.parse_args()
    if not args.model_path:
        print("SKIP: pass model_path or set NANO_MLX_CB_MODEL", file=sys.stderr)
        return 0

    import mlx.core as mx

    from nanovllm.mlx.engine import MLXEngine
    from nanovllm.sampling_params import SamplingParams

    mx.random.seed(42)
    path = os.path.expanduser(args.model_path)
    eng = MLXEngine(
        path,
        progress=False,
        legacy_mlx_lm=False,
        num_kvcache_blocks=384,
        max_num_batched_tokens=2048,
    )
    sp = SamplingParams(temperature=0.8, max_tokens=args.max_tokens, ignore_eos=True)
    prompts = ["The capital of France is", "1+1 equals"]
    outs = eng.generate_cb(prompts, [sp, sp], use_tqdm=False)
    assert len(outs) == 2
    for i, o in enumerate(outs):
        t = o.get("text", "")
        assert len(t) > 0, f"empty output for prompt {i}"
        print(f"--- prompt {i} ---\n{t[:400]!r}\n")
    print("OK: continuous batching completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
