# nano-vllm-mlx 项目总结

## 1. 项目背景

`nano-vllm` 原始实现主要面向 CUDA 生态（FlashAttention、Triton、Torch/CUDA 路径），在 Apple Silicon（macOS + Metal）环境下难以直接运行。  

本项目（`nano-vllm-mlx`）的目标是：在不破坏原有 CUDA 主路径设计思想的前提下，新增一条 **MLX 本地推理路径**，让项目在 Mac 上具备“可运行、可测试、可评估”的能力。

---

## 2. 项目目标

1. 在 macOS 上打通端到端推理链路（模型转换 -> 加载 -> 生成）。
2. 提供可复用的脚本工具（转换、smoke test、benchmark）。
3. 给出基础性能观测结果，形成工程化闭环。
4. 明确能力边界：该路径是可用性优先，不等价替代原 CUDA 高性能路径。

---

## 3. 主要实现内容

### 3.1 新增 MLX 运行时模块

新增目录：`nanovllm/mlx/`

- `config.py`：`MLXConfig`（模型路径、缓存上限、单设备约束等）
- `loader.py`：`MLXModelLoader`（封装 `mlx_lm.load`）
- `engine.py`：`MLXEngine`
  - 统一加载与生成接口
  - 默认单设备运行（`tensor_parallel_size=1`）
  - 缓存上限控制（`mx.set_cache_limit`，兼容旧 API）
- `llm.py`：`MLXLLM` 对外入口

### 3.2 包入口兼容处理

- `nanovllm/__init__.py`：对 `LLM` 做可选导入，并暴露 `MLXLLM`，避免 MLX 环境因 CUDA 依赖缺失直接 import 失败。

### 3.3 工具链脚本

- `scripts/convert_to_mlx.sh`
  - 一键安装 `mlx/mlx-lm`
  - 执行 HF -> MLX 模型转换
  - 支持量化参数（`-q --q-bits --q-mode`）
  - 避免“目标目录已存在导致转换失败”的常见坑
- `scripts/smoke_test_mlx.py`
  - 快速验证端到端推理
  - 支持 `--chat` 形式提示词
  - 支持进度输出
- `scripts/bench_mlx.py`
  - 轻量基准测试
  - 输出 run-by-run 延迟与吞吐近似（chars/s）
  - 汇总 P50/P95

### 3.4 文档化

- `docs/mac-mlx-runtime.md`：运行说明与限制
- `docs/feature-mac-mlx-runtime-report.md`：功能与测试报告
- 本文档：项目总体总结

---

## 4. 测试与结果

测试模型：`Qwen3-0.6B`（转换为 `...-mlx-v3`）  
测试脚本：`scripts/bench_mlx.py`  
重复次数：每组 3 次

### 对照结果

- `max_tokens=16`
  - p50 latency: **338.43 ms**
  - p95 latency: **357.18 ms**
  - median chars/s: **218.66**
- `max_tokens=64`
  - p50 latency: **797.46 ms**
  - p95 latency: **812.77 ms**
  - median chars/s: **366.16**
- `max_tokens=128`
  - p50 latency: **1488.13 ms**
  - p95 latency: **1839.77 ms**
  - median chars/s: **385.05**

结论：MLX 路径已完成功能打通并具备可观测性能表现，可作为本地开发与实验验证版本使用。

---

## 5. 当前能力边界（重要）

`nano-vllm-mlx` 当前定位是“可运行 + 可评估”的 MLX 适配版本，而非原 CUDA 高性能能力的完全迁移。

与原始 CUDA 路径相比，目前缺失/未覆盖：

- FlashAttention 高性能内核路径
- Triton 自定义 kernel 路径
- CUDA Graph / Torch CUDA 侧优化路径
- Tensor Parallel 多设备路径（MLX 分支固定单设备）
- vLLM 风格完整 continuous batching 与调度能力迁移

---

## 6. 项目价值

### 工程价值

- 将原本强绑定 CUDA 的项目扩展到 Apple Silicon 生态
- 建立了完整工具链（转换、测试、基准、文档）
- 形成了“可实现 -> 可运行 -> 可评估 -> 可说明”的交付闭环

### 学习价值

- 理解了不同推理后端（CUDA vs MLX）的能力差异与取舍
- 处理了真实依赖与 API 兼容问题（CLI 参数变化、生成参数兼容、缓存 API 变更）

---

## 7. 后续优化方向

1. 将 benchmark 从 chars/s 扩展到 token 级统计（TTFT、tokens/s、P95 token latency）。
2. 在 MLX 路径增加并发场景测评（多 prompt、多轮工具调用）。
3. 增强 `mlx_lm` 版本兼容层（自动适配不同 generate 参数签名）。
4. 评估更大模型（如 9B）及量化策略（Q4/NVFP4）对延迟和内存的影响。

---

## 8. 简历项目式描述（可直接使用）

在 `nano-vllm` 基础上实现了面向 Apple Silicon 的 `MLX` 推理分支（`nano-vllm-mlx`），完成从 Hugging Face 权重到 MLX 本地推理的端到端打通，并交付模型转换、功能 smoke test 与性能 benchmark 工具链。通过 `max_tokens=16/64/128` 三组对照实验验证了运行稳定性与延迟表现（p50 分别为 338ms / 797ms / 1488ms），同时明确了与原 CUDA 高性能路径的能力边界与后续迁移路线。