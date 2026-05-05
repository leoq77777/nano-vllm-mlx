# macOS MLX Runtime (WIP)

This branch introduces a separate MLX runtime path for Apple Silicon devices.

## What is implemented

- New runtime entry: `MLXLLM` (`nanovllm.mlx.llm`)
- MLX model loader wrapper: `MLXModelLoader`
- Metal cache limit configuration:
  - `mx.metal.set_cache_limit(cache_limit_gb * 1024**3)`
- Single-device-only runtime (`tensor_parallel_size=1`)
- Smoke test script: `scripts/smoke_test_mlx.py`

## Why this path

The original `nano-vllm` path is CUDA/Triton/FlashAttention oriented.  
MLX runtime is a dedicated Apple Silicon path and does not use CUDA graph or Triton kernels.

## Model conversion

Convert HF weights to MLX format first:

```bash
pip install mlx-lm
mlx_lm.convert \
  --hf-path Qwen/Qwen3-0.6B \
  --mlx-path ~/huggingface/Qwen3-0.6B-mlx \
  -q --q-bits 4 --q-mode nvfp4
```

Or use helper script:

```bash
chmod +x scripts/convert_to_mlx.sh
./scripts/convert_to_mlx.sh \
  --hf-path Qwen/Qwen3-0.6B \
  --mlx-path ~/huggingface/Qwen3-0.6B-mlx \
  --quantize \
  --q-bits 4 \
  --q-mode nvfp4
```

## Smoke test

```bash
PYTHONPATH=/Users/leo/code/nano-vllm \
python /Users/leo/code/nano-vllm/scripts/smoke_test_mlx.py \
  --model ~/huggingface/Qwen3-0.6B-mlx \
  --prompt "Hello" \
  --max-tokens 16 \
  --progress
```

## Quick benchmark

```bash
PYTHONPATH=/Users/leo/code/nano-vllm \
python /Users/leo/code/nano-vllm/scripts/bench_mlx.py \
  --model ~/huggingface/Qwen3-0.6B-mlx \
  --runs 3 \
  --max-tokens 64
```

Note: this benchmark prints latency and chars/s as a lightweight local indicator.

## Known limitations

- Runtime API currently targets basic single-prompt generation.
- Continuous batching and block-wise attention scheduling are not yet implemented in the MLX path.
