#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   MODEL_PATH=~/huggingface/Qwen3-0.6B ./scripts/run_p0_perf_eval.sh
#
# Optional env vars:
#   PYTHON_BIN=python3
#   RUNS=5
#   NUM_SEQS=256

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNS="${RUNS:-5}"
NUM_SEQS="${NUM_SEQS:-256}"
MODEL_PATH="${MODEL_PATH:-$HOME/huggingface/Qwen3-0.6B}"
RESULT_DIR="${ROOT_DIR}/results/p0-eval"
TMP_DIR="${ROOT_DIR}/results/.tmp"

mkdir -p "${RESULT_DIR}"
mkdir -p "${TMP_DIR}"

echo "[info] root=${ROOT_DIR}"
echo "[info] python=${PYTHON_BIN}"
echo "[info] model=${MODEL_PATH}"
echo "[info] results=${RESULT_DIR}"

cd "${ROOT_DIR}"

CURRENT_BRANCH="$(git branch --show-current)"
trap 'git checkout "${CURRENT_BRANCH}" >/dev/null 2>&1 || true' EXIT

# Materialize bench_metrics script once so it can run on branches
# that do not contain this file.
git show feat/p0-benchmark-metrics:scripts/bench_metrics.py > "${TMP_DIR}/bench_metrics.py"

# 1) Baseline on main (requires benchmark script from feature branch path or direct bench.py fallback)
echo "[step] checkout main"
git checkout main

echo "[step] run baseline bench.py on main"
${PYTHON_BIN} bench.py | tee "${RESULT_DIR}/main_bench.log"

# 2) Feature: benchmark metrics
echo "[step] checkout feat/p0-benchmark-metrics"
git checkout feat/p0-benchmark-metrics
${PYTHON_BIN} scripts/bench_metrics.py \
  --model "${MODEL_PATH}" \
  --runs "${RUNS}" \
  --num-seqs "${NUM_SEQS}" \
  --min-input-len 100 --max-input-len 1024 \
  --min-output-len 100 --max-output-len 1024 \
  --csv "${RESULT_DIR}/p0_benchmark_metrics.csv" \
  | tee "${RESULT_DIR}/p0_benchmark_metrics.log"

# 3) Feature: prepare path optimization
echo "[step] checkout feat/p0-prepare-path-optimization"
git checkout feat/p0-prepare-path-optimization
${PYTHON_BIN} "${TMP_DIR}/bench_metrics.py" \
  --model "${MODEL_PATH}" \
  --runs "${RUNS}" \
  --num-seqs "${NUM_SEQS}" \
  --min-input-len 100 --max-input-len 1024 \
  --min-output-len 100 --max-output-len 1024 \
  --csv "${RESULT_DIR}/p0_prepare_opt_metrics.csv" \
  | tee "${RESULT_DIR}/p0_prepare_opt_metrics.log"

# 4) Feature: parameter sweep defaults
echo "[step] checkout feat/p0-parameter-sweep-defaults"
git checkout feat/p0-parameter-sweep-defaults
${PYTHON_BIN} scripts/bench_sweep.py \
  --model "${MODEL_PATH}" \
  --repeat 3 \
  --csv "${RESULT_DIR}/p0_sweep.csv" \
  | tee "${RESULT_DIR}/p0_sweep.log"

echo "[done] evaluation complete. See ${RESULT_DIR}"
