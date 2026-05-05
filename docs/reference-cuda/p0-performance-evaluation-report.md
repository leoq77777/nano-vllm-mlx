# P0 三个 Feature 性能收益测算报告

## 执行摘要

目标：完成以下三个分支的性能收益测算并出报告。

- `feat/p0-benchmark-metrics`
- `feat/p0-prepare-path-optimization`
- `feat/p0-parameter-sweep-defaults`

当前状态：**在当前运行环境无法完成真实测算**（见“执行阻塞”），已补齐一键测评脚本，支持在具备模型与依赖的本机直接生成最终结果。

---

## 执行阻塞（本次会话实测）

1. 缺少模型目录  
命令：

```bash
ls -la "$HOME/huggingface/Qwen3-0.6B"
```

输出：

```text
No such file or directory
```

2. 缺少 PyTorch 依赖  
命令：

```bash
python3 -c "import torch; print(torch.__version__)"
```

输出：

```text
ModuleNotFoundError: No module named 'torch'
```

---

## 已交付自动化测评能力

新增脚本：`scripts/run_p0_perf_eval.sh`

用途：
- 自动切换 `main` 与三个 feature 分支；
- 运行 baseline / metrics / sweep；
- 将日志与 CSV 输出到 `results/p0-eval/`；
- 执行完毕后自动切回原分支。

---

## 复现实跑方式（在本机准备好环境后）

### 1) 环境准备

- 安装 `torch`（与 CUDA 匹配版本）
- 准备模型目录：`~/huggingface/Qwen3-0.6B/`

### 2) 执行测评

```bash
chmod +x scripts/run_p0_perf_eval.sh
MODEL_PATH=~/huggingface/Qwen3-0.6B ./scripts/run_p0_perf_eval.sh
```

### 3) 产物位置

- `results/p0-eval/main_bench.log`
- `results/p0-eval/p0_benchmark_metrics.csv`
- `results/p0-eval/p0_prepare_opt_metrics.csv`
- `results/p0-eval/p0_sweep.csv`
- 对应 `.log` 文件

---

## 报告填写模板（跑完后补数）

### A. `feat/p0-benchmark-metrics`

- 性质：可观测性增强，不改变引擎执行语义  
- 期望：主路径性能基本持平，新增可比较指标（TTFT/TPOT/P95等）

### B. `feat/p0-prepare-path-optimization`

- 关注指标：`ttft_ms`, `tpot_ms`, `decode_tok_s`, `p95_ms`
- 对比方式：`p0_prepare_opt_metrics.csv` vs `p0_benchmark_metrics.csv`
- 收益计算：
  - 吞吐提升：`(new-old)/old`
  - 延迟改善：`(old-new)/old`

### C. `feat/p0-parameter-sweep-defaults`

- 关注指标：`throughput_tok_s`, `peak_mem_mb`, `oom`
- 产出：最优参数组合 + 稳定性边界（OOM 临界）

---

## 结论（当前）

本次会话内已完成：
- 三个 feature 的自动化测评流程搭建；
- 报告框架与结果采集路径固定；
- 阻塞项定位（模型目录、torch 依赖）。

待完成：
- 在具备推理环境的机器执行脚本；
- 回填 CSV 数据并生成最终数值结论。
