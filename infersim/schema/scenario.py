from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from numbers import Real
from typing import Any

from infersim.errors import InputValidationError


def _required(data: Mapping[str, Any], field: str, prefix: str = "") -> Any:
    path = f"{prefix}.{field}" if prefix else field
    if field not in data:
        raise InputValidationError(path, "field is required")
    return data[field]


def _number(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be finite") from None
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


@dataclass(frozen=True)
class WorkloadScenario:
    name: str
    input_length: int
    output_length: int
    request_rate: float
    concurrency: int
    ttft_limit_ms: float
    tpot_limit_ms: float
    weight: float = 1.0

    @classmethod
    def from_dict(
        cls, data: Mapping[str, Any], path: str = ""
    ) -> "WorkloadScenario":
        if not isinstance(data, Mapping):
            raise InputValidationError(path or "$", "expected a mapping")

        def field_path(field: str) -> str:
            return f"{path}.{field}" if path else field

        name = _required(data, "name", path)
        if not isinstance(name, str) or not name:
            raise InputValidationError(
                field_path("name"), "must be a non-empty string"
            )
        return cls(
            name=name,
            input_length=_positive_integer(
                _required(data, "input_length", path),
                field_path("input_length"),
            ),
            output_length=_positive_integer(
                _required(data, "output_length", path),
                field_path("output_length"),
            ),
            request_rate=_positive_number(
                _required(data, "request_rate", path),
                field_path("request_rate"),
            ),
            concurrency=_positive_integer(
                _required(data, "concurrency", path),
                field_path("concurrency"),
            ),
            ttft_limit_ms=_positive_number(
                _required(data, "ttft_limit_ms", path),
                field_path("ttft_limit_ms"),
            ),
            tpot_limit_ms=_positive_number(
                _required(data, "tpot_limit_ms", path),
                field_path("tpot_limit_ms"),
            ),
            weight=_positive_number(
                data.get("weight", 1.0), field_path("weight")
            ),
        )


@dataclass(frozen=True)
class ScenarioSet:
    policy: str
    scenarios: tuple[WorkloadScenario, ...]

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ScenarioSet":
        if not isinstance(data, Mapping):
            raise InputValidationError("$", "expected a mapping")
        policy = _required(data, "policy")
        if policy not in ("all", "weighted"):
            raise InputValidationError("policy", "must be 'all' or 'weighted'")
        raw_scenarios = _required(data, "scenarios")
        if (
            isinstance(raw_scenarios, (str, bytes, bytearray))
            or not isinstance(raw_scenarios, Sequence)
        ):
            raise InputValidationError("scenarios", "must be a sequence")
        if not raw_scenarios:
            raise InputValidationError("scenarios", "must not be empty")

        scenarios = []
        names = set()
        for index, raw_scenario in enumerate(raw_scenarios):
            scenario = WorkloadScenario.from_dict(
                raw_scenario, f"scenarios[{index}]"
            )
            if scenario.name in names:
                raise InputValidationError(
                    f"scenarios[{index}].name", "must be unique"
                )
            names.add(scenario.name)
            scenarios.append(scenario)
        return cls(policy=policy, scenarios=tuple(scenarios))


@dataclass(frozen=True)
class PDLinkSpec:
    bandwidth_gbps: float
    latency_us: float
    efficiency: float
    max_concurrent_transfers: int

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PDLinkSpec":
        if not isinstance(data, Mapping):
            raise InputValidationError("$", "expected a mapping")
        efficiency = _number(_required(data, "efficiency"), "efficiency")
        if not 0 < efficiency <= 1:
            raise InputValidationError(
                "efficiency", "must be in the range (0, 1]"
            )
        return cls(
            bandwidth_gbps=_positive_number(
                _required(data, "bandwidth_gbps"), "bandwidth_gbps"
            ),
            latency_us=_nonnegative_number(
                _required(data, "latency_us"), "latency_us"
            ),
            efficiency=efficiency,
            max_concurrent_transfers=_positive_integer(
                _required(data, "max_concurrent_transfers"),
                "max_concurrent_transfers",
            ),
        )
