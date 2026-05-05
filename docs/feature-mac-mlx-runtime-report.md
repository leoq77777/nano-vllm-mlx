# Feature Report: mac-mlx-runtime

## 1. 目标

在不改变原有 CUDA 主路径的前提下，为 Apple Silicon 新增一条独立 MLX 推理运行时，满足：

- 在 macOS 本地可运行（功能可用）
- 提供基础 smoke test 与 benchmark 工具
- 保持与现有项目结构兼容（新增路径而非破坏原路径）

---

## 2. 本分支做了什么

分支：`feat/mac-mlx-runtime`

### 2.1 新增 MLX Runtime 路径

- `nanovllm/mlx/config.py`
  - `MLXConfig`：模型路径、`cache_limit_gb`、`tensor_parallel_size`、进度开关等。
- `nanovllm/mlx/loader.py`
  - `MLXModelLoader`：封装 `mlx_lm.load()`，加载 MLX 格式模型与 tokenizer。
- `nanovllm/mlx/engine.py`
  - `MLXEngine`：MLX 路径运行时主类。
  - 强制 `tensor_parallel_size=1`（符合 M 系列单 GPU 目标）。
  - Metal 缓存上限控制：
    - 优先 `mx.set_cache_limit`
    - 兼容旧版 `mx.metal.set_cache_limit`
  - 批量 prompts 的基础 `generate` 接口。
- `nanovllm/mlx/llm.py`
  - `MLXLLM` 入口类。
- `nanovllm/mlx/__init__.py`
  - 暴露 MLX 子模块入口。

### 2.2 包入口与依赖

- `nanovllm/__init__.py`
  - `LLM` 改为可选导入（避免缺 CUDA 依赖时阻塞 MLX 路径）。
  - 暴露 `MLXLLM`。
- `pyproject.toml`
  - 新增 optional dependencies：
    - `mlx`
    - `mlx-lm`

### 2.3 工具脚本与文档

- `scripts/smoke_test_mlx.py`
  - 端到端功能测试脚本（含 `--chat` 模式）。
- `scripts/bench_mlx.py`
  - 轻量 benchmark 脚本：
    - 输出 run-by-run latency 与 chars/s
    - 汇总 P50/P95 latency、median chars/s
- `scripts/convert_to_mlx.sh`
  - 一键转换脚本（修复了 `mlx_lm.convert` 参数兼容问题）：
    - 使用 `-q --q-bits --q-mode` 而非旧的 `--quantize 4`
    - 目标目录已存在时提前报错，避免误导
- `docs/mac-mlx-runtime.md`
  - 记录转换、smoke、benchmark 用法与限制。

---

## 3. 测试方法

测试模型：`~/huggingface/Qwen3-0.6B-mlx-v3`  
测试脚本：`scripts/bench_mlx.py`  
测试环境：本机 `.venv-mac` + MLX runtime  
每组重复：3 次

对照维度：

- `max_tokens = 16`
- `max_tokens = 64`
- `max_tokens = 128`

---

## 4. 测试结果

### 4.1 `max_tokens=16`

- p50 latency: **338.43 ms**
- p95 latency: **357.18 ms**
- median chars/s: **218.66**

### 4.2 `max_tokens=64`

- p50 latency: **797.46 ms**
- p95 latency: **812.77 ms**
- median chars/s: **366.16**

### 4.3 `max_tokens=128`

- p50 latency: **1488.13 ms**
- p95 latency: **1839.77 ms**
- median chars/s: **385.05**

### 4.4 结果解读

- 分支已完成“可用性验证”：MLX 路径可加载、可生成、可观测。
- 随 `max_tokens` 增大，端到端延迟上升符合预期。
- chars/s 在较长生成下更稳定，说明 warmup 与一次性启动成本被摊薄。

---

## 5. 已知限制

- 当前是“可运行优先”的 MLX 路径，不是 vLLM 全量能力移植。
- 暂未实现 continuous batching / block manager 级别调度迁移。
- 当前 benchmark 为轻量指标（latency + chars/s），后续可扩展为 token 级统计与并发压测。

---

## 6. 下一步建议

1. 在 `MLXEngine` 增加版本探测与采样参数自动适配（减少 `mlx_lm` API 差异影响）。
2. 补充 `bench_mlx.py` 的 token 统计（而非 chars/s 近似）。
3. 引入小规模并发测试（多 prompt）评估 MLX 路径在工具调用场景的收益。

---

## 7. 简历项目式介绍（可直接使用）

在 `nano-vllm` 中主导实现 Apple Silicon 的 MLX 独立推理路径（`feat/mac-mlx-runtime`），新增模型加载、运行时引擎、转换脚本与基准测试工具，完成从 Hugging Face 权重到 MLX 本地推理的端到端打通。通过 `max_tokens=16/64/128` 三组对照实验验证了运行稳定性与延迟表现（p50 分别为 338ms / 797ms / 1488ms），并沉淀可复用的文档化流程与调试工具链。