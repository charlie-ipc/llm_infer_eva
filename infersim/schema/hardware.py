from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from types import MappingProxyType
from typing import Any

from infersim.errors import InputValidationError


def _required(data: Mapping[str, Any], field: str, prefix: str = "") -> Any:
    path = f"{prefix}.{field}" if prefix else field
    if field not in data:
        raise InputValidationError(path, "field is required")
    return data[field]


def _mapping(value: Any, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise InputValidationError(path, "expected a mapping")
    return value


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    normalized = float(value)
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    return normalized


def _positive_number(value: Any, path: str) -> float:
    value = _number(value, path)
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _nonnegative_number(value: Any, path: str) -> float:
    value = _number(value, path)
    if value < 0:
        raise InputValidationError(path, "must be nonnegative")
    return value


def _positive_integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(path, "must be a non-empty string")
    return value


def _performance_mapping(value: Any, path: str) -> Mapping[str, float]:
    values = _mapping(value, path)
    if not values:
        raise InputValidationError(path, "must not be empty")
    normalized = {}
    for mode, throughput in values.items():
        if not isinstance(mode, str) or not mode:
            raise InputValidationError(path, "mode names must be non-empty strings")
        normalized[mode] = _positive_number(throughput, f"{path}.{mode}")
    return MappingProxyType(normalized)


@dataclass(frozen=True)
class InterconnectSpec:
    bandwidth_gbps: float
    latency_us: float


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    memory_capacity_gb: float
    memory_bandwidth_gbps: float
    cards_per_node: int
    gemm_tflops: Mapping[str, float]
    vector_tflops: Mapping[str, float]
    gemm_tile: tuple[int, int, int]
    gemm_engines: int
    vector_width: int
    vector_units: int
    gemm_launch_latency_us: float
    vector_launch_latency_us: float
    collective_launch_latency_us: float
    intra_node: InterconnectSpec
    inter_node: InterconnectSpec
    memory_reserve_fraction: float
    runtime_workspace_gb: float
    cost_per_card_hour: float | None

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "HardwareSpec":
        data = _mapping(data, "$")

        compute = _mapping(
            _required(data, "compute_tflops"), "compute_tflops"
        )
        gemm_tflops = _performance_mapping(
            _required(compute, "gemm", "compute_tflops"),
            "compute_tflops.gemm",
        )
        vector_tflops = _performance_mapping(
            _required(compute, "vector", "compute_tflops"),
            "compute_tflops.vector",
        )

        tile = _mapping(_required(data, "gemm_tile"), "gemm_tile")
        gemm_tile = tuple(
            _positive_integer(
                _required(tile, dimension, "gemm_tile"),
                f"gemm_tile.{dimension}",
            )
            for dimension in ("m", "n", "k")
        )

        launch = _mapping(
            _required(data, "kernel_launch_latency_us"),
            "kernel_launch_latency_us",
        )
        launch_latencies = {
            kind: _nonnegative_number(
                _required(launch, kind, "kernel_launch_latency_us"),
                f"kernel_launch_latency_us.{kind}",
            )
            for kind in ("gemm", "vector", "collective")
        }

        interconnect = _mapping(
            _required(data, "interconnect"), "interconnect"
        )
        intra_node = InterconnectSpec(
            bandwidth_gbps=_positive_number(
                _required(interconnect, "intra_node_gbps", "interconnect"),
                "interconnect.intra_node_gbps",
            ),
            latency_us=_nonnegative_number(
                _required(
                    interconnect, "intra_node_latency_us", "interconnect"
                ),
                "interconnect.intra_node_latency_us",
            ),
        )
        inter_node = InterconnectSpec(
            bandwidth_gbps=_positive_number(
                _required(interconnect, "inter_node_gbps", "interconnect"),
                "interconnect.inter_node_gbps",
            ),
            latency_us=_nonnegative_number(
                _required(
                    interconnect, "inter_node_latency_us", "interconnect"
                ),
                "interconnect.inter_node_latency_us",
            ),
        )

        reserve = _number(
            data.get("memory_reserve_fraction", 0.1),
            "memory_reserve_fraction",
        )
        if not 0 < reserve <= 1:
            raise InputValidationError(
                "memory_reserve_fraction", "must be in the range (0, 1]"
            )
        cost = None
        if "cost_per_card_hour" in data:
            cost = _nonnegative_number(
                data["cost_per_card_hour"], "cost_per_card_hour"
            )

        return cls(
            name=_nonempty_string(_required(data, "name"), "name"),
            memory_capacity_gb=_positive_number(
                _required(data, "memory_capacity_gb"), "memory_capacity_gb"
            ),
            memory_bandwidth_gbps=_positive_number(
                _required(data, "memory_bandwidth_gbps"),
                "memory_bandwidth_gbps",
            ),
            cards_per_node=_positive_integer(
                _required(data, "cards_per_node"), "cards_per_node"
            ),
            gemm_tflops=gemm_tflops,
            vector_tflops=vector_tflops,
            gemm_tile=gemm_tile,
            gemm_engines=_positive_integer(
                _required(data, "gemm_engines"), "gemm_engines"
            ),
            vector_width=_positive_integer(
                _required(data, "vector_width"), "vector_width"
            ),
            vector_units=_positive_integer(
                _required(data, "vector_units"), "vector_units"
            ),
            gemm_launch_latency_us=launch_latencies["gemm"],
            vector_launch_latency_us=launch_latencies["vector"],
            collective_launch_latency_us=launch_latencies["collective"],
            intra_node=intra_node,
            inter_node=inter_node,
            memory_reserve_fraction=reserve,
            runtime_workspace_gb=_nonnegative_number(
                data.get("runtime_workspace_gb", 0), "runtime_workspace_gb"
            ),
            cost_per_card_hour=cost,
        )
