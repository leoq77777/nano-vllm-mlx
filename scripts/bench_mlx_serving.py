#!/usr/bin/env python3
"""Serving-oriented benchmark for nano-vLLM MLX.

This script complements ``bench_mlx_nano_vllm_parity.py`` with service metrics:

- TTFT (time to first token)
- TPOT (time per output token after first token)
- end-to-end latency
- output throughput (tokens/s)

It can compare:

1) nano-vLLM MLX continuous batching path (CB)
2) legacy ``mlx_lm.generate`` path (baseline)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from time import perf_counter

from nanovllm.mlx.llm import MLXLLM
from nanovllm.mlx.sequence import Sequence
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


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", os.path.dirname(os.path.dirname(__file__)), "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return None


def _hardware_chip() -> str:
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        return out or platform.processor() or "unknown"
    except Exception:
        return platform.processor() or "unknown"


def _build_prompt(tokenizer, target_tokens: int) -> str:
    seed = (
        "Explain KV cache and continuous batching in concise engineering terms. "
        "Mention prefill and decode trade-offs. "
    )
    ids = tokenizer.encode(seed)
    while len(ids) < target_tokens:
        ids.extend(tokenizer.encode(seed))
    ids = ids[:target_tokens]
    return tokenizer.decode(ids)


@dataclass
class RunMetrics:
    latency_ms: float
    throughput_tps: float
    ttft_ms_mean: float | None = None
    tpot_ms_mean: float | None = None


def _run_cb_once(engine: MLXLLM, prompts: list[str], sp_list: list[SamplingParams]) -> RunMetrics:
    seqs: list[Sequence] = []
    for prompt, sp in zip(prompts, sp_list):
        seq = Sequence(engine.tokenizer.encode(prompt), sp)
        engine.scheduler.add(seq)
        seqs.append(seq)

    started = perf_counter()
    first_token_ts: dict[int, float] = {}

    while not engine.scheduler.is_finished():
        before = {seq.seq_id: seq.num_completion_tokens for seq in seqs}
        engine.step()
        now = perf_counter()
        for seq in seqs:
            if before[seq.seq_id] == 0 and seq.num_completion_tokens > 0 and seq.seq_id not in first_token_ts:
                first_token_ts[seq.seq_id] = now

    ended = perf_counter()

    output_tokens = sum(seq.num_completion_tokens for seq in seqs)
    elapsed = max(ended - started, 1e-9)
    throughput = output_tokens / elapsed

    ttfts_ms = []
    tpots_ms = []
    for seq in seqs:
        ft = first_token_ts.get(seq.seq_id, ended)
        ttfts_ms.append((ft - started) * 1000.0)
        # TPOT: average decode latency per token after first token.
        rem = max(seq.num_completion_tokens - 1, 1)
        tpots_ms.append((ended - ft) * 1000.0 / rem)

    return RunMetrics(
        latency_ms=elapsed * 1000.0,
        throughput_tps=throughput,
        ttft_ms_mean=statistics.mean(ttfts_ms) if ttfts_ms else None,
        tpot_ms_mean=statistics.mean(tpots_ms) if tpots_ms else None,
    )


def _run_legacy_once(engine: MLXLLM, prompts: list[str], max_tokens: int, temperature: float) -> RunMetrics:
    started = perf_counter()
    outputs = engine.generate(prompts, max_tokens=max_tokens, temperature=temperature)
    ended = perf_counter()
    elapsed = max(ended - started, 1e-9)

    output_tokens = 0
    for out in outputs:
        text = out.get("text", "")
        output_tokens += len(engine.tokenizer.encode(text))

    throughput = output_tokens / elapsed
    return RunMetrics(latency_ms=elapsed * 1000.0, throughput_tps=throughput)


def _summarize(runs: list[RunMetrics]) -> dict:
    lat = [x.latency_ms for x in runs]
    tps = [x.throughput_tps for x in runs]
    ttft = [x.ttft_ms_mean for x in runs if x.ttft_ms_mean is not None]
    tpot = [x.tpot_ms_mean for x in runs if x.tpot_ms_mean is not None]
    out = {
        "latency_ms": {
            "mean": statistics.mean(lat),
            "p50": percentile(lat, 0.5),
            "p95": percentile(lat, 0.95),
            "p99": percentile(lat, 0.99),
        },
        "throughput_tps": {
            "mean": statistics.mean(tps),
            "p50": percentile(tps, 0.5),
            "p95": percentile(tps, 0.95),
        },
    }
    if ttft:
        out["ttft_ms"] = {
            "mean": statistics.mean(ttft),
            "p50": percentile(ttft, 0.5),
            "p95": percentile(ttft, 0.95),
            "p99": percentile(ttft, 0.99),
        }
    if tpot:
        out["tpot_ms"] = {
            "mean": statistics.mean(tpot),
            "p50": percentile(tpot, 0.5),
            "p95": percentile(tpot, 0.95),
            "p99": percentile(tpot, 0.99),
        }
    return out


def _pct_improve(base: float, new: float) -> float:
    if base == 0:
        return 0.0
    return (base - new) / base * 100.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B-mlx-v3"))
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--concurrency", type=str, default="1,2,4")
    parser.add_argument("--prompt-len", type=int, default=128, help="Prompt length in tokens.")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--cache-limit-gb", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--compare-legacy", action="store_true")
    parser.add_argument("--json-out", default="")
    args = parser.parse_args()

    model = os.path.expanduser(args.model)
    if not os.path.isdir(model):
        print(f"Model directory not found: {model}", file=sys.stderr)
        return 2

    concurrencies = [int(x.strip()) for x in args.concurrency.split(",") if x.strip()]
    if not concurrencies:
        print("No valid --concurrency values.", file=sys.stderr)
        return 2

    cb = MLXLLM(
        model,
        cache_limit_gb=args.cache_limit_gb,
        tensor_parallel_size=1,
        progress=args.progress,
        legacy_mlx_lm=False,
    )

    prompt = _build_prompt(cb.tokenizer, args.prompt_len)
    sp_template = SamplingParams(temperature=args.temperature, ignore_eos=True, max_tokens=args.max_tokens)

    legacy = None
    if args.compare_legacy:
        legacy = MLXLLM(
            model,
            cache_limit_gb=args.cache_limit_gb,
            tensor_parallel_size=1,
            progress=args.progress,
            legacy_mlx_lm=True,
        )

    results = {
        "meta": {
            "bench": "nano-vllm-mlx serving benchmark",
            "git_head": _git_head(),
            "model": model,
            "python": sys.version.split()[0],
            "platform": platform.platform(),
            "chip": _hardware_chip(),
            "runs": args.runs,
            "warmup_runs": args.warmup_runs,
            "prompt_len_tokens": args.prompt_len,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "concurrency": concurrencies,
            "compare_legacy": bool(args.compare_legacy),
        },
        "cb": {},
    }
    if legacy is not None:
        results["legacy"] = {}
        results["comparison"] = {}

    for c in concurrencies:
        prompts = [prompt for _ in range(c)]
        sps = [sp_template for _ in range(c)]

        for _ in range(args.warmup_runs):
            _run_cb_once(cb, prompts, sps)
            if legacy is not None:
                _run_legacy_once(legacy, prompts, args.max_tokens, args.temperature)

        cb_runs = [_run_cb_once(cb, prompts, sps) for _ in range(args.runs)]
        cb_summary = _summarize(cb_runs)
        results["cb"][str(c)] = {"runs": [asdict(x) for x in cb_runs], "summary": cb_summary}

        print(
            f"[CB][c={c}] latency_mean={cb_summary['latency_ms']['mean']:.2f}ms "
            f"ttft_mean={cb_summary.get('ttft_ms', {}).get('mean', 0.0):.2f}ms "
            f"tpot_mean={cb_summary.get('tpot_ms', {}).get('mean', 0.0):.2f}ms "
            f"throughput_mean={cb_summary['throughput_tps']['mean']:.2f}tok/s"
        )

        if legacy is not None:
            lg_runs = [
                _run_legacy_once(legacy, prompts, args.max_tokens, args.temperature)
                for _ in range(args.runs)
            ]
            lg_summary = _summarize(lg_runs)
            results["legacy"][str(c)] = {"runs": [asdict(x) for x in lg_runs], "summary": lg_summary}
            cmp_row = {
                "latency_mean_improve_pct": _pct_improve(
                    lg_summary["latency_ms"]["mean"],
                    cb_summary["latency_ms"]["mean"],
                ),
                "throughput_mean_gain_pct": (
                    (cb_summary["throughput_tps"]["mean"] - lg_summary["throughput_tps"]["mean"])
                    / max(lg_summary["throughput_tps"]["mean"], 1e-9)
                    * 100.0
                ),
            }
            results["comparison"][str(c)] = cmp_row
            print(
                f"[vs legacy][c={c}] latency_improve={cmp_row['latency_mean_improve_pct']:+.2f}% "
                f"throughput_gain={cmp_row['throughput_mean_gain_pct']:+.2f}%"
            )

    if args.json_out:
        out = os.path.expanduser(args.json_out)
        os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, sort_keys=True)
            f.write("\n")
        print(f"\nWrote: {out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
