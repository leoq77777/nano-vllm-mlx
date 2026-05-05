# 文档索引

本仓库在 **CUDA / PyTorch**（上游 Nano-vLLM）之外，增加了 **Apple Silicon + MLX** 运行时。文档按用途分开，避免和 MLX 功能说明混在一起。

## MLX（本仓库扩展，推荐阅读）

| 文档 | 说明 |
|------|------|
| [mac-mlx-runtime.md](mac-mlx-runtime.md) | Mac 上安装、转换权重、`MLXLLM` 使用方式 |
| [feature-mac-mlx-runtime-report.md](feature-mac-mlx-runtime-report.md) | MLX 运行时实现与基准摘要 |
| [nano-vllm-mlx-summary.md](nano-vllm-mlx-summary.md) | MLX 方向技术小结 |
| [nano-vllm-mlx-resume-project.md](nano-vllm-mlx-resume-project.md) | 简历/项目叙述用整理（与实现一致处为准） |

## 参考：上游 CUDA 路线（非 MLX）

| 文档 | 说明 |
|------|------|
| [reference-cuda/p0-performance-evaluation-report.md](reference-cuda/p0-performance-evaluation-report.md) | 针对上游 P0 分支（benchmark / prepare / sweep）的测算说明与脚本入口；**与 MLX 路径无关** |

根目录 [README.md](../README.md) 仍以 **上游 CUDA 版** 的安装与 Quick Start 为主；MLX 读者请先看本文档中的 **MLX** 一节。
