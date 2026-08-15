from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from infersim.cost.memory import MemoryBreakdown
from infersim.schema.parallel import ParallelPlan


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


@dataclass(frozen=True)
class StageMetrics:
    stage: str
    scenario_name: str
    plan: ParallelPlan
    latency_seconds: float
    tpot_seconds: float | None
    prompt_token_capacity: float | None
    output_token_capacity: float | None
    request_capacity: float
    average_context_length: float
    gemm_seconds: float
    vector_seconds: float
    tp_seconds: float
    ep_seconds: float
    useful_gemm_ops: int
    aligned_gemm_ops: int
    useful_vector_ops: int
    aligned_vector_ops: int
    memory: MemoryBreakdown
    component_seconds: Mapping[str, float]
    max_supported_batch: int | None = None
    max_supported_concurrency: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_seconds",
            MappingProxyType(dict(self.component_seconds)),
        )
