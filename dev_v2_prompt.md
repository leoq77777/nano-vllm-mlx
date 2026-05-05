MLX 路径对齐核心功能开发计划                   ───────────────────────────────

  现状总览        ────────────────────────────────────────────                ──

  ┌─────────────────┬────────────────────────────┬──────────────────────────┐
  │      功能       │      CUDA 路径已实现       │       MLX 路径现状       │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Paged KV Cache  │    ✅ block_manager.py     │            ❌            │
  │                 │          (112行)           │                          │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Continuous      │   ✅ scheduler.py (84行)   │       ❌ 逐个串行        │
  │ Batching        │                            │                          │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Prefix Caching  │  ✅ hash-to-block lookup   │            ❌            │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Chunked Prefill │     ✅ scheduler 内置      │            ❌            │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Paged Attention │ ✅ FlashAttention + Triton │    ❌ 无 block 级别      │
  │                 │           kernel           │      gather/scatter      │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ Tensor          │     ✅ NCCL + 共享内存     │   ❌ MLX distributed     │
  │ Parallelism     │                            │        可用但未用        │
  ├─────────────────┼────────────────────────────┼──────────────────────────┤
  │ CUDA Graph /    │   ✅ CUDA Graph capture    │  ❌ 可用 mx.compile()    │
  │ 等价            │                            │           替代           │
  └─────────────────┴────────────────────────────┴──────────────────────────┘

  预计新增代码总量：~1,200-1,500 行 MLX 代码。

  ---
  Phase 1: 数据层 — Paged KV Cache + Sequence（~200 行）

  文件：nanovllm/mlx/sequence.py, nanovllm/mlx/block_manager.py,
  nanovllm/mlx/config.py

  纯 Python 逻辑，几乎可以从 CUDA 路径直接移植：

  - Sequence — 直接复用 engine/sequence.py（83行，无 PyTorch 依赖）
  - BlockManager — 移植 engine/block_manager.py（112行），xxHash 前缀 hash
  计算可直接复用，仅需移除 numpy 依赖或保留
  - Config — 从简单的 4 字段 MLXConfig 扩展到支持 max_num_batched_tokens,
  max_num_seqs, max_model_len, kvcache_block_size, num_kvcache_blocks 等，并根据
   mx.metal.get_active_memory() + get_cache_memory() 自动计算可分配的 KV cache
  block 数量

  风险：无。这是最安全的阶段。

  ---
  Phase 2: 模型层 — Qwen3 MLX 实现（~300 行）

  文件：nanovllm/mlx/models/qwen3.py

  mlx_lm 没有暴露可修改 attention 层的 Qwen3 模型类，必须自己实现：

  - Embedding — mx.take(weight, input_ids, axis=0)
  - RMSNorm — 直接用 mx.fast.rms_norm
  - RoPE — 直接用 mx.fast.rope
  - QKV Linear — mx.matmul + fused Q/K/V projection + split
  - MLP — mx.matmul + SiLU gating + down projection
  - Attention（关键差异）— 不直接 compute attention，而是接收外部传入的
  pre-gathered K/V（见 Phase 3），或通过 hook 注入自定义 attention forward

  关键架构决策：Attention 层的设计有两种路线：
  1. 侵入式：修改 attention forward，在内部做 KV gather/scatter → Phase 3
  耦合度高
  2. 非侵入式：attention 层只负责给定 Q/K/V 做计算，gather/scatter 由外部
  model_runner 完成 → 推荐此路线，CUDA 路径也是这么做的

  风险：中。需要正确实现 Qwen3 的 Q/K normalization、attention bias 等细节。可用
   mlx_lm.load() 加载权重 + 自己定义的模型结构。

  ---
  Phase 3: Paged Attention（MLX 原生实现，~300 行）

  文件：nanovllm/mlx/layers/attention.py

  这是整个项目中最核心的新代码。CUDA 路径依赖 FlashAttention 的 block_table
  参数直接做 block-aware attention。MLX 的 mx.fast.scaled_dot_product_attention
  不支持 block_table，需要手动 gather/scatter。

  Prefill 路径（一次处理多个序列，变长）：
  1. 对每个序列，根据 block_table + slot_mapping 从 k_cache/v_cache gather
  出完整的 K/V
  2. 对变长序列做 padding + causal mask，或分别调用
  sdpa（序列数多时分多次调用开销大）
  3. mx.fast.scaled_dot_product_attention(q, gathered_k, gathered_v, scale,
  mask)
  4. 将新的 K/V scatter 回 k_cache/v_cache 对应 slot

  Decode 路径（每序列 1 token）：
  1. 根据 block_table + context_lens 从 k_cache/v_cache gather
  出每序列的完整历史 K/V
  2. Q 是单 token，直接 sdpa
  3. scatter 新 KV 到 block_table 最后一个 slot

  两种实现路线：
  - 路线 A：用 mx.take + 循环逐序列 gather，简单但可能慢
  - 路线 B：用 mx.gather 批量 gather（需要对 block_table 做 index
  变换），更复杂但更快

  建议先实现路线 A，后用路线 B 优化。

  风险：高。性能关键路径，需要 benchmark 对比直接 sdpa 和 gather+sdpa+scatter 的
   overhead。

  ---
  Phase 4: 调度层 — Continuous Batching + Chunked Prefill（~100 行）

  文件：nanovllm/mlx/scheduler.py

  几乎可以直接从 engine/scheduler.py 移植：
  - waiting/running 双 deque
  - Prefill 阶段：token budget 管理 + 块分配 + chunked prefill（部分 prefill）
  - Decode 阶段：轮转调度 + preemption（KV block 不足时抢占）
  - postprocess() → 更新 num_cached_tokens / append_token / 判断 finished

  唯一差异：BlockManager 的 API 不变，所以同 Phase 1 衔接即可。

  风险：低。纯 Python 逻辑。

  ---
  Phase 5: 模型执行器 — MLX ModelRunner（~250 行）

  文件：nanovllm/mlx/model_runner.py

  对应 CUDA engine/model_runner.py（258行），去除 CUDA Graph 和 NCCL
  逻辑后的等价实现：

  prepare_prefill(seqs):
    - 构建 input_ids, positions, cu_seqlens, slot_mapping, block_tables
    - 处理 prefix cache 场景（block_tables 非 None 时用缓存的 K/V）

  prepare_decode(seqs):
    - 构建 input_ids (每序列最后一个 token), positions, slot_mapping
    - context_lens, block_tables

  run_model(input_ids, positions, is_prefill):
    - model.forward(input_ids, positions)  → hidden_states
    - model.compute_logits(hidden_states)  → logits (需要 lm_head)

  采样: Sampler (Gumbel-max, ~15行，直接移植)

  关键设计：Context 传递机制。CUDA 路径用全局 set_context/get_context
  在线程间传递 block_tables/slot_mapping。MLX 单线程可直接传参或保留类似机制。

  风险：中。需要正确处理 prefill/decode 两套 batch 构建逻辑，尤其是 cu_seqlens
  的等价物。

  ---
  Phase 6: 引擎集成 — MLX LLMEngine + generate()（~120 行）

  文件：nanovllm/mlx/engine.py（重写）、nanovllm/mlx/llm.py

  将现有 MLXEngine（69行，只包装 mlx_lm.generate）重写为完整的 continuous
  batching 引擎：

  class MLXEngine:
      def __init__(model, **kwargs):
          config = MLXConfig(model, **kwargs)
          self.tokenizer = ...       # AutoTokenizer 或 mlx_lm tokenizer
          self.scheduler = Scheduler(config)
          self.model_runner = MLXModelRunner(config)

      def step():
          seqs, is_prefill = self.scheduler.schedule()
          token_ids = self.model_runner.run(seqs, is_prefill)
          self.scheduler.postprocess(seqs, token_ids, is_prefill)
          return outputs, num_tokens

      def generate(prompts, sampling_params):
          while not is_finished():
              output, _ = self.step()
              ...

  以及批量 generate（Phase 5 之前 MLX 路径只能逐个 prompt 调用
  mlx_lm.generate）。

  风险：中。需要确保 tokenizer（HuggingFace AutoTokenizer 或 mlx_lm
  tokenizer）的正确对接。

  ---
  Phase 7: 优化（~80 行）

  - mx.compile() 替代 CUDA Graph — 对 decode 路径使用 MLX 的 JIT 编译，减少
  Metal kernel launch 开销（类似于 torch.compile 而非 CUDA Graph 的全量
  capture，但效果类似）
  - Batch gather — 将 Phase 3 的逐序列 gather 优化为批量操作
  - 内存预分配 — 预分配 attention mask、slot_mapping 等缓冲区，避免每次 step
  重新分配

  风险：低。mx.compile 是 MLX 的标准优化手段。

  ---
  Phase 8（可选）: Tensor Parallelism（~250 行）

  MLX 已有 mx.distributed 模块，包含 all_sum, all_gather, send, recv,
  init。理论上可以实现 Row/Column parallel linear：

  - ColumnParallelLinear：输入相同，权重按列切分，输出按列拼接 → 本地计算即可
  - RowParallelLinear：输入按列切分，权重按行切分 → 本地计算后 dist.all_sum
  - QKV 融合并行：逻辑同 CUDA 路径

  但有几个限制：
  - MLX distributed 的后端可能只有 localhost（多进程 Metal）
  - 无 NCCL 级别的跨机通信
  - 优先级建议放最后，单卡 Metal 已经能跑大部分模型了

  ---
  推荐执行顺序 & 里程碑

  ┌─────────────────────────────────┬─────────────┬──────┬─────────┐
  │              阶段               │    依赖     │ 难度 │ 代码量  │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M1: Phase 1 (数据层)            │ 无          │ 低   │ ~200 行 │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M2: Phase 2 (Qwen3 MLX)         │ Phase 1     │ 中   │ ~300 行 │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M3: Phase 3 (Paged Attention)   │ Phase 2     │ 高   │ ~300 行 │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M4: Phase 4 (Scheduler)         │ Phase 1     │ 低   │ ~100 行 │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M5: Phase 5 (ModelRunner)       │ Phase 2,3,4 │ 中   │ ~250 行 │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M6: Phase 6 (Engine + generate) │ Phase 5     │ 中   │ ~120 行 │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M7: Phase 7 (优化)              │ Phase 6     │ 低   │ ~80 行  │
  ├─────────────────────────────────┼─────────────┼──────┼─────────┤
  │ M8: Phase 8 (TP)                │ Phase 6     │ 高   │ ~250 行 │
  └─────────────────────────────────┴─────────────┴──────┴─────────┘

  总计 ~1,500 行增量代码。最小可行版本（M1-M6）约 ~1,200 行。

  核心难点是 Phase 3 的 paged attention —
  这是唯一没有现成库支持、必须从零实现的部分。建议先做 M1 → M2 → M3（验证
  attention 性能），再推进调度和引擎集成。
