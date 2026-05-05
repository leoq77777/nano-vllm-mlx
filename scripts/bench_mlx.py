import argparse
import os
import statistics
from time import perf_counter

from nanovllm.mlx.llm import MLXLLM


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


def main():
    parser = argparse.ArgumentParser(description="Simple benchmark for MLX runtime.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B-mlx"))
    parser.add_argument("--prompt", default="Explain what KV cache is in one sentence.")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--cache-limit-gb", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise FileNotFoundError(
            f"Model directory not found: {args.model}\n"
            "Run: ./scripts/convert_to_mlx.sh --hf-path Qwen/Qwen3-0.6B --mlx-path ~/huggingface/Qwen3-0.6B-mlx --quantize --q-bits 4 --q-mode nvfp4"
        )

    llm = MLXLLM(
        args.model,
        cache_limit_gb=args.cache_limit_gb,
        tensor_parallel_size=1,
        progress=args.progress,
    )

    latencies = []
    token_throughputs = []
    for i in range(args.runs):
        t0 = perf_counter()
        outputs = llm.generate([args.prompt], max_tokens=args.max_tokens, temperature=args.temperature)
        elapsed = perf_counter() - t0
        latencies.append(elapsed * 1000.0)
        text = outputs[0]["text"]
        # rough throughput proxy (chars/s) for quick local iteration
        token_throughputs.append(len(text) / elapsed if elapsed > 0 else 0.0)
        print(f"[run {i+1}] latency={elapsed*1000.0:.2f} ms, chars/s={token_throughputs[-1]:.2f}")

    print("\n[summary]")
    print(f"p50_latency_ms={percentile(latencies, 0.5):.2f}")
    print(f"p95_latency_ms={percentile(latencies, 0.95):.2f}")
    print(f"median_chars_per_s={statistics.median(token_throughputs):.2f}")


if __name__ == "__main__":
    main()

