from __future__ import annotations

import os
from typing import Iterable

from nanovllm.mlx.config import MLXConfig
from nanovllm.mlx.loader import MLXModelLoader


class MLXEngine:
    def __init__(self, model: str, **kwargs):
        self.config = MLXConfig(model=model, **{k: v for k, v in kwargs.items() if k in MLXConfig.__slots__})
        self.progress = self.config.progress or os.getenv("NANO_VLLM_PROGRESS", "0") == "1"
        if self.config.tensor_parallel_size != 1:
            raise ValueError("MLX runtime only supports tensor_parallel_size=1.")
        try:
            import mlx.core as mx  # type: ignore
        except Exception as e:
            raise RuntimeError("MLX runtime requires mlx. Install with: pip install mlx") from e
        self.mx = mx
        self._configure_metal()
        if self.progress:
            print("[nano-vllm][mlx] loading model...", flush=True)
        loaded = MLXModelLoader().load(self.config.model)
        self.model = loaded.model
        self.tokenizer = loaded.tokenizer
        try:
            from mlx_lm import generate  # type: ignore
        except Exception as e:
            raise RuntimeError("mlx_lm.generate is required. Install with: pip install mlx-lm") from e
        self._generate = generate
        if self.progress:
            print("[nano-vllm][mlx] model ready", flush=True)

    def _configure_metal(self):
        limit_bytes = self.config.cache_limit_gb * 1024**3
        set_cache_limit = getattr(self.mx, "set_cache_limit", None)
        if set_cache_limit is not None:
            set_cache_limit(limit_bytes)
            return
        metal = getattr(self.mx, "metal", None)
        if metal is None:
            return
        legacy_set_cache_limit = getattr(metal, "set_cache_limit", None)
        if legacy_set_cache_limit is not None:
            legacy_set_cache_limit(limit_bytes)

    def generate_one(self, prompt: str, max_tokens: int, temperature: float) -> str:
        _ = temperature  # kept for API compatibility; mlx_lm version differences handle sampling internally
        out = self._generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=self.progress,
        )
        if isinstance(out, str):
            return out
        return str(out)

    def generate(self, prompts: Iterable[str], max_tokens: int = 128, temperature: float = 0.7):
        outputs = []
        for idx, prompt in enumerate(prompts):
            if self.progress:
                print(f"[nano-vllm][mlx] decoding prompt {idx+1}", flush=True)
            text = self.generate_one(prompt, max_tokens=max_tokens, temperature=temperature)
            outputs.append({"text": text})
        return outputs

