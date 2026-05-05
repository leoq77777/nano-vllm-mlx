from dataclasses import dataclass


@dataclass(slots=True)
class MLXConfig:
    model: str
    cache_limit_gb: int = 20
    tensor_parallel_size: int = 1
    progress: bool = False

