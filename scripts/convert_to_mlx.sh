#!/usr/bin/env bash
set -euo pipefail

# Convert HF model to MLX format.
# Example:
#   ./scripts/convert_to_mlx.sh \
#     --hf-path Qwen/Qwen3-0.6B \
#     --mlx-path ~/huggingface/Qwen3-0.6B-mlx \
#     --quantize \
#     --q-bits 4

HF_PATH="Qwen/Qwen3-0.6B"
MLX_PATH="$HOME/huggingface/Qwen3-0.6B-mlx"
USE_QUANTIZE="1"
Q_BITS="4"
Q_MODE="nvfp4"
PYTHON_BIN="${PYTHON_BIN:-python3}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --hf-path)
      HF_PATH="$2"
      shift 2
      ;;
    --mlx-path)
      MLX_PATH="$2"
      shift 2
      ;;
    --quantize)
      USE_QUANTIZE="1"
      shift 1
      ;;
    --no-quantize)
      USE_QUANTIZE="0"
      shift 1
      ;;
    --q-bits)
      Q_BITS="$2"
      shift 2
      ;;
    --q-mode)
      Q_MODE="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1"
      exit 1
      ;;
  esac
done

echo "[info] python=${PYTHON_BIN}"
echo "[info] hf-path=${HF_PATH}"
echo "[info] mlx-path=${MLX_PATH}"
echo "[info] quantize=${USE_QUANTIZE}"
echo "[info] q-bits=${Q_BITS}"
echo "[info] q-mode=${Q_MODE}"

${PYTHON_BIN} -m pip install -U mlx mlx-lm
if [[ -e "${MLX_PATH}" ]]; then
  echo "[error] target path already exists: ${MLX_PATH}"
  echo "Please remove it first, or choose a new --mlx-path."
  exit 1
fi
if [[ "${USE_QUANTIZE}" == "1" ]]; then
  mlx_lm.convert --hf-path "${HF_PATH}" --mlx-path "${MLX_PATH}" -q --q-bits "${Q_BITS}" --q-mode "${Q_MODE}"
else
  mlx_lm.convert --hf-path "${HF_PATH}" --mlx-path "${MLX_PATH}"
fi

echo "[done] converted model saved to ${MLX_PATH}"
