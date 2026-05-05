# nano-vllm-mlx Dev V2 实现总结

## 背景

`nano-vllm` 原项目主要面向 CUDA / PyTorch 生态，依赖 FlashAttention、Triton、CUDA Graph、NCCL 等能力。`dev_v2_prompt.md` 的目标是把此前只包装 `mlx_lm.generate()` 的 Mac MLX 路径，继续向原 CUDA 路径的核心推理能力靠拢，形成一条更完整的 Apple Silicon 本地推理运行时。

本阶段工作的核心目标不是简单“能跑”，而是让 MLX 分支具备以下能力：

- Paged KV Cache 与 prefix cache 数据结构。
- Qwen3 MLX 模型实现，权重布局兼容 `mlx_lm`。
- MLX 原生 gather / scatter 形式的 paged attention。
- Continuous batching、chunked prefill、decode 调度与采样闭环。
- 可验证、可 benchmark、可回退的 Phase7 优化实验。

说明：GitNexus 已索引独立项目 `nano-vllm-mlx`，索引路径为 `/Users/leo/code/nano-vllm-mlx`，远程为 `https://github.com/leoq77777/nano-vllm-mlx`。当前 GitNexus 索引停在提交 `fb926b0`（基础 MLX runtime 版本），而本文覆盖的 Dev V2 实现包含后续本地提交 `f2d5d08`、`5f0439f`、`cffdd43` 以及未提交的 Phase7 实验改动。因此本文会同时引用 GitNexus 对旧版 MLX 入口链路的结构分析，并以当前代码为准描述 Dev V2 的新增能力。

## 实现概览

当前 MLX 路径主要位于 `nanovllm/mlx/`：

```text
nanovllm/mlx/
  config.py
  sequence.py
  block_manager.py
  models/qwen3.py
  layers/attention.py
  layers/rope_flat.py
  scheduler.py
  model_runner.py
  sampler.py
  engine.py
```

最近关键提交：

- `f2d5d08`：Phase 1 数据层，新增 `Sequence`、`BlockManager`、扩展 `MLXConfig`。
- `5f0439f`：Phase 2 Qwen3 MLX 模型，权重布局兼容 `mlx_lm`。
- `cffdd43`：Phase 3–7 主链路接入，修复 RoPE 对齐，加入 continuous batching 验证脚本。

## GitNexus 索引视角

GitNexus 当前索引的 `nano-vllm-mlx` 版本是 `fb926b0`，对应 Dev V2 之前的基础 MLX wrapper 形态。该索引结果对项目演进很有帮助：

- `MLXEngine` 位于 `nanovllm/mlx/engine.py`，在索引版本中包含 `__init__`、`_configure_metal`、`generate_one`、`generate` 方法。
- `MLXLLM` 位于 `nanovllm/mlx/llm.py`，继承自 `MLXEngine`，是外部用户调用 MLX 路径的主要入口。
- GitNexus 显示 `MLXLLM` 被 `scripts/smoke_test_mlx.py`、`scripts/bench_mlx.py`、`nanovllm/__init__.py`、`nanovllm/mlx/__init__.py` 引用，说明基础版本的公开接口集中在 `MLXLLM`。
- GitNexus 对 `MLXEngine` / `MLXLLM` 未识别到复杂 execution process，符合当时实现：`MLXEngine` 主要是对 `mlx_lm.generate()` 的轻量封装，没有自有 scheduler / model runner / paged attention 执行链。

因此，Dev V2 的核心变化可以概括为：保留 GitNexus 识别出的 `MLXLLM -> MLXEngine` 入口形态，但把 `MLXEngine` 内部从 `mlx_lm.generate()` wrapper 升级为自有 continuous batching 推理引擎。

## Phase 1：数据层

新增 `nanovllm/mlx/sequence.py` 与 `nanovllm/mlx/block_manager.py`，基本对齐 CUDA 路径的 `engine/sequence.py` 和 `engine/block_manager.py`。

主要内容：

- `Sequence` / `SequenceStatus`：记录 token、prompt 长度、completion 长度、block table、cached token、scheduled token 等调度状态。
- `BlockManager`：实现 KV block 分配、释放、引用计数、prefix cache hash lookup。
- Prefix hash：保留 `xxhash` 逻辑，但移除 NumPy 依赖，用 `struct.pack(..., "q")` 生成与 `np.array(token_ids).tobytes()` 一致的 int64 字节。
- `MLXConfig`：增加 `max_num_batched_tokens`、`max_num_seqs`、`max_model_len`、`kvcache_block_size`、`num_kvcache_blocks` 等调度与 KV cache 参数。
- KV cache 预算：通过 MLX device info、active memory、cache memory 估算可用 KV block 数量。

验证：

- 已对拍 CUDA 与 MLX 的 block hash，一致。
- 已验证 `BlockManager.allocate()` / `deallocate()` 能正常工作。

## Phase 2：Qwen3 MLX 模型

新增 `nanovllm/mlx/models/qwen3.py`，实现可被自定义 paged attention 驱动的 Qwen3 MLX 模型。

主要内容：

- `Qwen3MLXModelArgs`：与 `mlx_lm.models.qwen3.ModelArgs` 配置字段对齐。
- `Qwen3MLXAttention`：实现 Q/K/V projection、Q/K RMSNorm、RoPE、SDPA、output projection。
- `Qwen3MLXMLP`：实现 gate / up / down projection 和 SwiGLU。
- `Qwen3MLXForCausalLM`：顶层模型结构，属性名与 `mlx_lm` 权重布局对齐，支持 `load_weights`。
- `load_qwen3_mlx_from_path()`：从本地 MLX 模型目录读取 `config.json` 与 `model*.safetensors`。

关键设计：

- Attention 被拆成 `project_qkv()`、`apply_rope()`、`dot_product_attention()`、`output_proj()`。
- 这样 Phase 3 可以在 `dot_product_attention()` 前注入外部 gather 出来的 K/V，而不需要侵入模型权重结构。

当前边界：

- `load_qwen3_mlx_from_path()` 当前面向非量化 safetensors；量化权重可继续接入 `mlx_lm` 的 quantization 逻辑。

## Phase 3：MLX Paged Attention

新增 `nanovllm/mlx/layers/attention.py`，实现 MLX 侧 paged KV 的 scatter / gather 工具。

主要内容：

- `kv_cache_slots()`：为每层预分配 `(num_blocks, block_size, num_kv_heads, head_dim)` 形状的 K/V cache。
- `scatter_kv_tokens()`：逐 token 将新 K/V 写入物理 KV slot。
- `scatter_kv_tokens_batched()`：当 slot 全部有效时使用批量 scatter。
- `gather_kv_from_slots()`：按物理 slot 列表 gather 历史 K/V。
- `block_table_prefix_slots()`：根据 `block_table`、`block_size`、token 范围构建物理 slot 列表。

这部分替代了 CUDA 路径中 FlashAttention `block_table` 参数的能力：MLX 的 `mx.fast.scaled_dot_product_attention` 不理解 block table，因此需要由 `model_runner` 手动完成 gather / scatter。

## Phase 4：Scheduler

新增 `nanovllm/mlx/scheduler.py`，移植 CUDA 路径的 continuous batching 调度逻辑。

主要内容：

- `waiting` / `running` 双队列。
- Prefill 阶段按 `max_num_batched_tokens` 控制 token budget。
- 支持 chunked prefill：当一个 prompt 太长时，可以只调度部分 token。
- Decode 阶段轮转调度 running sequences。
- KV block 不足时执行 preemption，将序列放回 waiting 队列。
- `postprocess()` 更新 cached token、追加采样 token、判断 EOS / max token 结束。

## Phase 5：ModelRunner

新增 `nanovllm/mlx/model_runner.py`，对应 CUDA 路径中的 `engine/model_runner.py`，负责把 scheduler 输出的 `Sequence` 批次转换成模型执行输入。

主要内容：

- `prepare_prefill()`：
  - 构建 `input_ids`、`positions`、`cu_seqlens_q`、`cu_seqlens_k`、`slot_mapping`、`block_tables`。
  - 支持 prefix cache 场景下从 KV cache gather 历史 K/V。
- `prepare_decode()`：
  - 每个序列取最后一个 token。
  - 构建 context length、block table、slot mapping。
- Prefill 执行：
  - 对当前输入 token 计算 Q/K/V。
  - 将新 K/V scatter 到 paged cache。
  - 从 cache gather 完整 K/V。
  - 构建变长 causal mask，调用 `mx.fast.scaled_dot_product_attention`。
- Decode 执行：
  - 新 token 的 K/V scatter 到 cache。
  - 根据每个序列的 block table gather 历史 K/V。
  - 对每个序列执行单 token attention。
- `mlx_sample_tokens()`：移植 Gumbel-max 采样。

关键修复：

- 原先手写 RoPE 与 `mlx.nn.RoPE` 数值不一致。现改为 `apply_rope_segmented()`，对 prefill / decode 都调用模型层自己的 RoPE module，并按 segment offset 应用，已与 `nn.RoPE` 对拍误差为 0。

## Phase 6：Engine 集成

重写 `nanovllm/mlx/engine.py` 的主路径，让 `MLXEngine` 可以运行完整 continuous batching 流程。

主要内容：

- `legacy_mlx_lm=True`：保留旧路径，继续使用 `mlx_lm.generate()`。
- 默认 continuous batching 路径：
  - 读取模型 config。
  - 加载 `Qwen3MLXForCausalLM` 权重。
  - 加载 tokenizer。
  - 构造 `MLXScheduler` 与 `MLXModelRunner`。
- `add_request()`：将 prompt 转为 `Sequence` 并加入 scheduler。
- `step()`：执行一次 schedule -> model_runner -> postprocess。
- `generate_cb()`：多 prompt continuous batching 生成。
- `generate()`：在默认情况下走 `generate_cb()`，旧模式下走 `mlx_lm.generate()`。

## Phase 7：优化实验

Phase 7 做了两类评估：

1. `mx.compile()`：
   - MLX 的 `mx.compile` 要求函数参数是 array 或常量树。
   - 当前 `decode` 需要 `MLXRunnerContext`（Python dataclass，包含 seq 对象与列表元数据），直接 compile 会报错。
   - 因此保留 `mlx_compile_decode` 配置位，但标记为 reserved；后续若要启用，需要先把 runner context 打包成纯 `mx.array` 参数。

2. Batch gather：
   - 新增 `build_decode_slot_matrix()` 与 `gather_kv_from_slot_matrix()`，支持将 decode 中逐序列 gather 改成批量 gather + batched SDPA。
   - 增加 `decode_batch_gather` 和 `decode_batch_gather_min_work` 配置。
   - 实测该优化收益不稳定，在多个负载下会变慢，因此默认值设为 `False`，作为实验开关保留。

A/B benchmark 脚本：

- `scripts/bench_mlx_phase7_decode.py`

实测样例：

- `batch=12, max_tokens=32, runs=3`
  - 逐序列 decode：p50 latency `4230.81 ms`，median chars/s `345.32`
  - batch gather：p50 latency `5262.01 ms`，median chars/s `303.50`
  - 结论：当前 batch gather 版本在该负载下更慢，默认关闭。

## 验证结果

本地模型：

- `~/huggingface/Qwen3-0.6B-mlx-v3`

已跑通过的验证：

```bash
python scripts/verify_mlx_cb.py "$HOME/huggingface/Qwen3-0.6B-mlx-v3" --max-tokens 8
```

结果：

```text
OK: continuous batching completed
```

直接调用 `MLXEngine(..., decode_batch_gather=False)` 双 prompt 生成也已通过，说明 Phase 1–6 主链路在不启用 Phase 7 实验优化时可正常运行。

## 当前能力边界

当前 `nano-vllm-mlx` 已经从“包装 `mlx_lm.generate()`”推进到“自有 continuous batching 推理栈”，但仍不等价 CUDA 高性能路径：

- 没有 FlashAttention / Triton kernel。
- 没有 CUDA Graph 等同级别 capture。
- 没有 Tensor Parallel / NCCL；MLX distributed 仍是后续可选方向。
- Paged attention 目前基于 MLX 原生 gather / scatter，性能上仍需要进一步优化。
- Token 级 benchmark（TTFT、decode tokens/s、P95 token latency）仍可继续补齐。

## 项目价值

这个项目的价值不在于“把 CUDA 代码机械搬到 Mac”，而在于完成了一次跨后端推理系统适配：

- 保留原项目 CUDA 主路径，不破坏已有设计。
- 为 Apple Silicon 增加独立 MLX 后端。
- 把模型加载、KV cache、调度、paged attention、采样、engine 集成到一条可运行链路。
- 对优化做实测，不把不稳定优化默认打开。
- 将实现、验证、benchmark、文档整理成可复现的工程闭环。

## 简历 Bullet Points

- 在 `nano-vllm` 基础上实现 Apple Silicon / MLX 独立推理后端，新增 `Sequence`、`BlockManager`、`Scheduler`、`ModelRunner` 与 `MLXEngine.generate_cb()`，将原先逐 prompt 的 `mlx_lm.generate()` 包装升级为支持 continuous batching 的自有推理链路。
- 自研 Qwen3 MLX 模型结构并对齐 `mlx_lm` 权重布局，拆分 QKV projection、RoPE、SDPA 与 output projection，使 paged KV gather / scatter 能在 attention 外部注入，支持后续替换和优化。
- 在 MLX 上实现 paged KV cache 与 block-table attention：通过 KV slot scatter/gather、prefix cache hash、chunked prefill 与 decode 调度，复刻 CUDA 路径中核心推理引擎思想，并完成本地 Qwen3-0.6B 端到端验证。
- 建立可复现验证与性能评估闭环，包含模型转换、smoke test、continuous batching 验证和 Phase7 A/B benchmark；对 batch gather / `mx.compile` 优化做实测评估，并将收益不稳定的实验优化默认关闭以保证主链路稳定。

