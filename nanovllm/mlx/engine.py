from __future__ import annotations

import os
from dataclasses import fields
from pathlib import Path
from time import perf_counter
from typing import Iterable

from tqdm.auto import tqdm

from nanovllm.mlx.config import MLXConfig, apply_qwen_mlx_arch_to_mlx_config
from nanovllm.mlx.loader import MLXModelLoader
from nanovllm.mlx.model_runner import MLXModelRunner, load_mlx_arch
from nanovllm.mlx.models.qwen3 import load_qwen3_mlx_from_path
from nanovllm.mlx.scheduler import MLXScheduler
from nanovllm.mlx.sequence import Sequence
from nanovllm.sampling_params import SamplingParams


class MLXEngine:
    def __init__(self, model: str, **kwargs):
        slots = {f.name for f in fields(MLXConfig)}
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in slots}
        self.config = MLXConfig(model=model, **cfg_kwargs)
        self.config.apply_sequence_block_size()
        self.progress = self.config.progress or os.getenv("NANO_VLLM_PROGRESS", "0") == "1"
        self._legacy = bool(self.config.legacy_mlx_lm)

        if self._legacy:
            self._init_legacy_mlx_lm()
            return

        try:
            import mlx.core as mx  # type: ignore
        except Exception as e:
            raise RuntimeError("MLX runtime requires mlx. Install with: pip install mlx") from e
        self.mx = mx
        self._configure_metal()

        model_path = Path(model)
        arch = load_mlx_arch(model_path)
        apply_qwen_mlx_arch_to_mlx_config(self.config, arch)

        if self.config.mlx_compile_decode and self.progress:
            print(
                "[nano-vllm][mlx] note: mlx_compile_decode is reserved (mx.compile needs array-only args); ignored.",
                flush=True,
            )
        if self.progress:
            print("[nano-vllm][mlx] loading weights (continuous batching path)...", flush=True)
        self._cb_model, _cfgd = load_qwen3_mlx_from_path(model_path, lazy=False, strict=True)
        from mlx_lm.utils import load_tokenizer  # type: ignore

        self.tokenizer = load_tokenizer(model_path)
        eos = getattr(self.tokenizer, "eos_token_id", None)
        if eos is None:
            eos = arch.get("eos_token_id", 2)
        self.config.eos = int(eos)

        self.scheduler = MLXScheduler(self.config)
        self.model_runner = MLXModelRunner(self.config, self._cb_model, arch)
        if self.progress:
            print("[nano-vllm][mlx] continuous batching engine ready", flush=True)

    def _init_legacy_mlx_lm(self) -> None:
        try:
            import mlx.core as mx  # type: ignore
        except Exception as e:
            raise RuntimeError("MLX runtime requires mlx.") from e
        self.mx = mx
        self._configure_metal()
        if self.config.tensor_parallel_size != 1:
            raise ValueError("MLX runtime only supports tensor_parallel_size=1.")
        if self.progress:
            print("[nano-vllm][mlx] loading model (legacy mlx_lm)...", flush=True)
        loaded = MLXModelLoader().load(self.config.model)
        self._legacy_model = loaded.model
        self.tokenizer = loaded.tokenizer
        try:
            from mlx_lm import generate  # type: ignore
        except Exception as e:
            raise RuntimeError("mlx_lm.generate is required.") from e
        self._generate = generate
        self.scheduler = None
        self.model_runner = None
        self._cb_model = None
        if self.progress:
            print("[nano-vllm][mlx] legacy path ready", flush=True)

    def _configure_metal(self) -> None:
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

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams) -> None:
        if self._legacy:
            raise RuntimeError("add_request is not used in legacy_mlx_lm mode.")
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        if self._legacy:
            raise RuntimeError("step() is not used in legacy_mlx_lm mode.")
        seqs, is_prefill = self.scheduler.schedule()
        num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
        token_ids = self.model_runner.run(seqs, is_prefill)
        self.scheduler.postprocess(seqs, token_ids, is_prefill)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def is_finished(self) -> bool:
        if self._legacy:
            return True
        return self.scheduler.is_finished()

    def generate_cb(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        decode_text: bool = True,
    ) -> list[dict]:
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        outputs: dict[int, list[int]] = {}
        prefill_throughput = decode_throughput = 0.0
        while not self.scheduler.is_finished():
            t0 = perf_counter()
            finished, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t0)
            else:
                decode_throughput = (-num_tokens) / max(perf_counter() - t0, 1e-9)
            pbar.set_postfix(
                Prefill=f"{int(prefill_throughput)}tok/s",
                Decode=f"{int(decode_throughput)}tok/s",
            )
            for seq_id, token_ids in finished:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        ordered = [outputs[sid] for sid in sorted(outputs.keys())]
        if decode_text:
            return [
                {"text": self.tokenizer.decode(toks), "token_ids": toks}
                for toks in ordered
            ]
        return [{"text": "", "token_ids": toks} for toks in ordered]

    def generate_one(self, prompt: str, max_tokens: int, temperature: float) -> str:
        _ = temperature
        out = self._generate(
            self._legacy_model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=self.progress,
        )
        if isinstance(out, str):
            return out
        return str(out)

    def generate(self, prompts: Iterable[str], max_tokens: int = 128, temperature: float = 0.7):
        prompts = list(prompts)
        if self._legacy:
            outputs = []
            for idx, prompt in enumerate(prompts):
                if self.progress:
                    print(f"[nano-vllm][mlx] decoding prompt {idx + 1}", flush=True)
                text = self.generate_one(prompt, max_tokens=max_tokens, temperature=temperature)
                outputs.append({"text": text})
            return outputs
        sps = [
            SamplingParams(temperature=max(temperature, 1.01e-5), max_tokens=max_tokens, ignore_eos=False)
            for _ in prompts
        ]
        return self.generate_cb(prompts, sps, use_tqdm=self.progress)
