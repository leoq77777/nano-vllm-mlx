# nano-vLLM-MLX：面向 Apple Silicon 的本地 LLM 推理适配项目

## 项目简介

`nano-vLLM-MLX` 是在轻量级推理引擎 `nano-vllm` 基础上扩展出的 Apple Silicon 本地推理分支。原项目主要面向 Linux + NVIDIA CUDA 生态，依赖 PyTorch、FlashAttention、Triton 与 CUDA Graph 等高性能后端；本项目新增独立的 MLX runtime，使其能够在 macOS / Apple Silicon 上完成从模型转换、加载、生成到基准测试的端到端闭环。

项目目标不是简单“能跑”，而是在保留原 CUDA 路径设计边界的同时，构建一条清晰、可测试、可评估的 Metal/MLX 推理路径，为本地开发、Demo 和后续 Apple Silicon 优化提供基础。

---

## 技术栈

### 推理与模型运行

- **MLX / MLX-LM**：Apple 官方面向 Apple Silicon 的数组计算与大模型推理栈。
- **Metal 后端**：通过 MLX 使用 Apple GPU；支持本地统一内存架构。
- **Hugging Face 模型格式**：以 HF 模型为输入，转换为 MLX 可加载格式。
- **Qwen3-0.6B**：用于功能验证和性能基准的轻量模型。

### 原项目相关技术

- **PyTorch / CUDA**：原始 nano-vllm 的主要执行路径。
- **FlashAttention / Triton**：原始项目依赖的 CUDA 高性能 attention 与自定义 kernel 路径。
- **Paged KV Cache / Continuous Batching**：原始项目的核心推理引擎思想，本分支保留接口与后续迁移空间。

### 工程工具

- **Python**：主开发语言。
- **Shell 脚本**：模型转换自动化。
- **Markdown 文档**：沉淀设计、测试、限制与项目成果。
- **Benchmark 脚本**：测量本地 MLX 运行时延迟与吞吐近似指标。

---

## 架构设计

本项目采用“新增后端，不破坏主路径”的设计方式：

- 原始 `nanovllm.LLM` 仍代表 CUDA/PyTorch 路径。
- 新增 `nanovllm.mlx.MLXLLM`，作为 Apple Silicon 专用入口。
- `MLXLLM` 内部通过 `MLXEngine` 封装模型加载、缓存配置和生成逻辑。
- `MLXModelLoader` 负责隔离 `mlx_lm.load()`，避免上层直接依赖 MLX-LM 细节。

核心新增模块：

```text
nanovllm/mlx/
  config.py      # MLXConfig：运行时配置
  loader.py      # MLXModelLoader：模型和 tokenizer 加载
  engine.py      # MLXEngine：MLX 推理主流程
  llm.py         # MLXLLM：对外入口
```

辅助工具：

```text
scripts/
  convert_to_mlx.sh     # HF -> MLX 模型转换
  smoke_test_mlx.py     # 端到端功能测试
  bench_mlx.py          # 本地轻量 benchmark
```

---

## 核心实现点

### 1. 独立 MLX Runtime

新增 `MLXEngine`，负责：

- 校验运行约束：`tensor_parallel_size=1`
- 加载 MLX 模型与 tokenizer
- 配置 MLX/Metal 缓存上限
- 提供与原项目风格接近的 `generate()` 接口

该设计使 MLX 路径独立于原 CUDA 路径，避免在 macOS 环境中导入 FlashAttention/Triton 时直接失败。

### 2. 模型转换工具链

新增 `scripts/convert_to_mlx.sh`，解决从 HF 权重到 MLX 格式的转换问题：

- 支持 `--hf-path` / `--mlx-path`
- 支持量化参数：`--quantize --q-bits 4 --q-mode nvfp4`
- 修复 `mlx_lm.convert` 新版 CLI 参数变化带来的兼容问题
- 避免提前创建目标目录导致 `mlx_lm.convert` 失败

### 3. MLX 版本兼容

项目中处理了 MLX API 变更：

- 优先使用新版 `mx.set_cache_limit`
- 兼容旧版 `mx.metal.set_cache_limit`

这避免了 MLX 升级后因 deprecated API 造成长期维护风险。

### 4. 基准测试闭环

新增 `scripts/bench_mlx.py`，用于快速评估本地 MLX runtime：

- 多次运行
- 统计 P50 / P95 latency
- 统计 chars/s 作为本地快速迭代的吞吐近似指标

虽然当前还不是严格 token-level benchmark，但已经能支撑早期性能验证和回归比较。

---

## 工程难点

### 难点 1：原项目强依赖 CUDA 生态

原始 nano-vllm 假设运行环境具备 CUDA、FlashAttention、Triton 和 NCCL 等组件，而 Apple Silicon 不具备这些后端。因此不能简单“改 device”，而需要新增一条独立后端路径。

解决方式：

- 不直接替换原 PyTorch/CUDA 路径
- 新增 `nanovllm.mlx` 子模块
- 保持原路径边界清晰，降低对原项目的侵入性

### 难点 2：模型格式不兼容

原项目直接加载 HF safetensors 权重，而 MLX runtime 需要 MLX 格式模型。

解决方式：

- 新增转换脚本 `convert_to_mlx.sh`
- 支持本地 HF 目录和远程 HF repo
- 支持 4-bit / NVFP4 量化路径

### 难点 3：MLX-LM API 版本差异

实际开发中遇到 `mlx_lm.convert` 参数变化、`mlx_lm.generate` 采样参数不兼容等问题。

解决方式：

- 及时修正转换 CLI 参数：使用 `-q --q-bits --q-mode`
- 在 `MLXEngine` 中对 temperature 参数做兼容保留，避免因版本差异阻塞基本生成
- 将运行方式和限制写入文档，降低复现成本

### 难点 4：评估指标设计

原 vLLM 类项目常用 token/s、TTFT、TPOT 等指标，但 MLX-LM 封装层早期不直接暴露细粒度 token 时间。

解决方式：

- 先落地轻量 benchmark，统计端到端 latency 和 chars/s
- 明确该指标是早期近似指标
- 后续计划补充 token-level 统计

---

## 测试结果

测试模型：`Qwen3-0.6B` 转换后的 MLX 模型  
测试脚本：`scripts/bench_mlx.py`  
每组运行：3 次

| max_tokens | P50 Latency | P95 Latency | Median chars/s |
|---:|---:|---:|---:|
| 16  | 338.43 ms  | 357.18 ms  | 218.66 |
| 64  | 797.46 ms  | 812.77 ms  | 366.16 |
| 128 | 1488.13 ms | 1839.77 ms | 385.05 |

结果说明：

- MLX runtime 已完成端到端推理验证。
- 随输出长度增加，端到端延迟线性上升，符合预期。
- 较长输出场景下 chars/s 更稳定，说明启动开销被更好摊薄。

---

## 与原 nano-vllm 的能力边界

当前分支不是原 CUDA 路径的等价替代，而是 Apple Silicon 运行路径的 MVP。

当前未覆盖：

- FlashAttention 高性能 CUDA attention kernel
- Triton 自定义 CUDA kernel
- CUDA Graph
- Tensor Parallel 多卡路径
- 完整 Paged KV Cache / Continuous Batching 调度迁移
- token-level 细粒度性能采样

当前新增能力：

- Apple Silicon 本地推理路径
- MLX 模型加载与生成
- HF -> MLX 转换工具链
- 本地 smoke test 和 benchmark
- 面向 macOS 的工程化文档

---

## 项目亮点

- **跨后端迁移能力**：从 CUDA/PyTorch 推理栈扩展到 Apple MLX/Metal 生态。
- **工程闭环完整**：包含模型转换、运行时、测试、benchmark、文档。
- **边界清晰**：不伪装成完整 vLLM 替代，而是明确定位为 Apple Silicon MVP runtime。
- **真实问题驱动**：处理了 CLI 变更、API deprecated、模型目录冲突、依赖缺失等实际工程问题。
- **可量化成果**：提供多组延迟与吞吐近似测试数据。

---

## 可用于简历的项目描述

**nano-vLLM-MLX：Apple Silicon 本地 LLM 推理适配**

在轻量级推理引擎 `nano-vllm` 基础上设计并实现 Apple Silicon 专用 MLX runtime，新增 `MLXLLM` 推理入口、模型加载器、HF 到 MLX 模型转换脚本、端到端 smoke test 与 benchmark 工具链。项目解决了原引擎强绑定 CUDA/FlashAttention/Triton 导致 macOS 不可运行的问题，并在 Qwen3-0.6B 上完成本地推理验证。通过 `max_tokens=16/64/128` 三组 benchmark，获得 P50 延迟 338ms / 797ms / 1488ms 的测试结果，形成从后端适配、模型转换到性能评估的完整工程闭环。

---

## 后续可扩展方向

1. 将 benchmark 升级为 token-level 统计（TTFT、TPOT、tokens/s）。
2. 实现更接近 vLLM 的 MLX KV cache 与 block-wise attention。
3. 增加多 prompt / 多轮工具调用场景下的并发 benchmark。
4. 支持 Qwen3 更大模型（如 9B）和不同量化策略对比。
5. 增加 MLX-LM 版本探测与 generate 参数自动适配层。

