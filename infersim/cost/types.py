from dataclasses import dataclass


@dataclass(frozen=True)
class KernelCost:
    useful_ops: int
    aligned_ops: int
    compute_seconds: float
    memory_bytes: float
    memory_seconds: float
    launch_seconds: float
    seconds: float
    bottleneck: str
