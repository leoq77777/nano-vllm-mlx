import argparse
import os

from nanovllm.mlx.llm import MLXLLM


def main():
    parser = argparse.ArgumentParser(description="Smoke test for MLX runtime.")
    parser.add_argument("--model", default=os.path.expanduser("~/huggingface/Qwen3-0.6B-mlx"))
    parser.add_argument("--prompt", default="Hello")
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--cache-limit-gb", type=int, default=20)
    parser.add_argument("--progress", action="store_true")
    parser.add_argument("--chat", action="store_true", help="Use simple chat-style prompt template.")
    args = parser.parse_args()

    if not os.path.isdir(args.model):
        raise FileNotFoundError(
            f"Model directory not found: {args.model}\n"
            "Convert first using mlx-lm:\n"
            "mlx_lm.convert --hf-path Qwen/Qwen3-0.6B --mlx-path ~/huggingface/Qwen3-0.6B-mlx -q --q-bits 4 --q-mode nvfp4\n"
            "or use: ./scripts/convert_to_mlx.sh --quantize --q-bits 4 --q-mode nvfp4"
        )

    llm = MLXLLM(
        args.model,
        cache_limit_gb=args.cache_limit_gb,
        tensor_parallel_size=1,
        progress=args.progress,
    )
    prompt = args.prompt
    if args.chat:
        prompt = (
            "<|im_start|>system\n"
            "You are a concise and helpful assistant.\n"
            "<|im_end|>\n"
            "<|im_start|>user\n"
            f"{args.prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    outputs = llm.generate([prompt], max_tokens=args.max_tokens, temperature=args.temperature)
    print("=== prompt ===")
    print(args.prompt if not args.chat else f"[chat]\n{args.prompt}")
    print("=== output ===")
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()

