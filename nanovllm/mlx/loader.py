from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class LoadedMLXModel:
    model: object
    tokenizer: object


class MLXModelLoader:
    def __init__(self):
        try:
            from mlx_lm import load  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "mlx_lm is required for MLX runtime. "
                "Install with: pip install mlx-lm"
            ) from e
        self._load = load

    def load(self, model_path: str) -> LoadedMLXModel:
        model, tokenizer = self._load(model_path)
        return LoadedMLXModel(model=model, tokenizer=tokenizer)

