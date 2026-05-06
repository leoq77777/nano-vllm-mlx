#!/usr/bin/env python3
"""Replicate nano-vllm CUDA ``bench.py`` workload on the MLX continuous-batching engine.

Throughput definition matches upstream:

    throughput = sum(sampling_params.max_tokens) / wall_time_seconds

Upstream uses synthetic random prompt token ids in ``[0, 10000]``. To stay valid for
embedding tables, ids are clipped to ``[0, min(10000, vocab_size - 1)]``.

By default this script calls ``MLXEngine.generate_cb(..., decode_text=False)`` so the
timed region does **not** include decoding every completion back to UTF-8 text — closer
to what ``bench.py`` measures (CUDA path never materializes decoded strings).
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from random import randint, seed

from nanovllm.mlx.llm import MLXLLM
from nanovllm.mlx.model_runner import load_mlx_arch
from nanovllm.sampling_params import SamplingParams


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _mlx_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"mlx": None, "mlx_lm": None}
    try:
        import importlib.metadata as md

        out["mlx"] = md.version("mlx")
    except Exception:
        pass
    try:
        import importlib.metadata as md

        out["mlx_lm"] = md.version("mlx-lm")
    except Exception:
        pass
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=os.path.expanduser("~/huggingface/Qwen3-0.6B-mlx-v3"),
        help="Local MLX model directory (non-legacy continuous batching path).",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-seqs", type=int, default=256)
    parser.add_argument("--max-input-len", type=int, default=1024)
    parser.add_argument("--max-output-len", type=int, default=1024)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--cache-limit-gb", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument(
        "--decode-text",
        action="store_true",
        help="Include tokenizer.decode of every completion in the timed region (not bench.py parity).",
    )
    parser.add_argument("--json-out", default="", help="Optional path to write a JSON summary line.")
    args = parser.parse_args()

    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        print(f"Model directory not found: {model}", file=sys.stderr)
        return 2

    arch = load_mlx_arch(model)
    vocab = int(arch.get("vocab_size") or 0)
    if vocab <= 0:
        print("Could not read vocab_size from mlx arch; refusing random token ids.", file=sys.stderr)
        return 2
    hi = min(10000, vocab - 1)

    seed(args.seed)
    prompt_token_ids = [
        [randint(0, hi) for _ in range(randint(100, args.max_input_len))] for _ in range(args.num_seqs)
    ]
    sampling_params = [
        SamplingParams(temperature=0.6, ignore_eos=True, max_tokens=randint(100, args.max_output_len))
        for _ in range(args.num_seqs)
    ]

    llm = MLXLLM(
        model,
        cache_limit_gb=args.cache_limit_gb,
        tensor_parallel_size=1,
        progress=args.progress,
        legacy_mlx_lm=False,
        max_model_len=args.max_model_len,
    )

    llm.generate_cb(["Benchmark: "], SamplingParams(), use_tqdm=False, decode_text=False)

    t0 = time.perf_counter()
    llm.generate_cb(
        prompt_token_ids,
        sampling_params,
        use_tqdm=False,
        decode_text=args.decode_text,
    )
    elapsed = time.perf_counter() - t0

    total_tokens = sum(sp.max_tokens for sp in sampling_params)
    throughput = total_tokens / elapsed if elapsed > 0 else 0.0

    summary = {
        "bench": "nano-vllm bench.py parity (MLX)",
        "model": model,
        "git_head": _git_head(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "mlx": _mlx_versions(),
        "seed": args.seed,
        "num_seqs": args.num_seqs,
        "max_input_len": args.max_input_len,
        "max_output_len": args.max_output_len,
        "max_model_len": args.max_model_len,
        "decode_text_in_timed_region": bool(args.decode_text),
        "total_planned_output_tokens": total_tokens,
        "time_s": elapsed,
        "throughput_tok_s": throughput,
        "random_token_id_range": [0, hi],
    }

    print(
        f"Total: {total_tokens}tok, Time: {elapsed:.2f}s, Throughput: {throughput:.2f}tok/s",
        flush=True,
    )

    if args.json_out:
        path = os.path.expanduser(args.json_out)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, sort_keys=True)
            f.write("\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
