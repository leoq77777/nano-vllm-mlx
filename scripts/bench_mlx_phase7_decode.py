#!/usr/bin/env python3
"""A/B benchmark for Phase 7 decode batch-gather optimization."""

from __future__ import annotations

import argparse
import os
import statistics
import sys
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import mlx.core as mx

from nanovllm.mlx.engine import MLXEngine
from nanovllm.sampling_params import SamplingParams


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    if len(values) == 1:
        return values[0]
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def run_case(
    model: str,
    *,
    decode_batch_gather: bool,
    prompts: list[str],
    max_tokens: int,
    temperature: float,
    runs: int,
) -> tuple[list[float], list[float]]:
    lat_ms: list[float] = []
    chars_ps: list[float] = []
    for i in range(runs):
        mx.random.seed(1234 + i)
        llm = MLXEngine(
            model,
            progress=False,
            legacy_mlx_lm=False,
            decode_batch_gather=decode_batch_gather,
            num_kvcache_blocks=512,
            max_num_batched_tokens=4096,
        )
        sp = SamplingParams(temperature=temperature, max_tokens=max_tokens, ignore_eos=True)
        t0 = perf_counter()
        outs = llm.generate_cb(prompts, [sp] * len(prompts), use_tqdm=False)
        dt = perf_counter() - t0
        lat_ms.append(dt * 1000.0)
        total_chars = sum(len(o["text"]) for o in outs)
        chars_ps.append(total_chars / dt if dt > 0 else 0.0)
    return lat_ms, chars_ps


def summarize(name: str, lat_ms: list[float], chars_ps: list[float]) -> None:
    print(f"\n[{name}]")
    print(f"p50_latency_ms={percentile(lat_ms, 0.5):.2f}")
    print(f"p95_latency_ms={percentile(lat_ms, 0.95):.2f}")
    print(f"median_chars_per_s={statistics.median(chars_ps):.2f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B-mlx-v3"))
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=24)
    p.add_argument("--temperature", type=float, default=0.8)
    args = p.parse_args()

    prompts = [f"Give one short fact about number {i}." for i in range(args.batch_size)]

    lat_old, cps_old = run_case(
        args.model,
        decode_batch_gather=False,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        runs=args.runs,
    )
    lat_new, cps_new = run_case(
        args.model,
        decode_batch_gather=True,
        prompts=prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        runs=args.runs,
    )

    summarize("decode_loop_per_seq", lat_old, cps_old)
    summarize("decode_batch_gather", lat_new, cps_new)

    p50_old = percentile(lat_old, 0.5)
    p50_new = percentile(lat_new, 0.5)
    cps_med_old = statistics.median(cps_old)
    cps_med_new = statistics.median(cps_new)
    lat_gain = (p50_old - p50_new) / p50_old * 100.0 if p50_old > 0 else 0.0
    th_gain = (cps_med_new - cps_med_old) / cps_med_old * 100.0 if cps_med_old > 0 else 0.0
    print("\n[delta]")
    print(f"p50_latency_change={lat_gain:+.2f}%")
    print(f"median_chars_per_s_change={th_gain:+.2f}%")


if __name__ == "__main__":
    main()

