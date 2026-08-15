from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import isfinite
from numbers import Real
from typing import TYPE_CHECKING

from infersim.errors import InputValidationError
from infersim.schema.parallel import ParallelPlan
from infersim.schema.scenario import ScenarioSet, WorkloadScenario

if TYPE_CHECKING:
    from infersim.cost.memory import MemoryBreakdown
    from infersim.cost.types import StageMetrics


def _record_tuple(value, path: str, item_type: type) -> tuple:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    normalized = tuple(value)
    for index, item in enumerate(normalized):
        if not isinstance(item, item_type):
            raise InputValidationError(
                f"{path}[{index}]", f"must be a {item_type.__name__}"
            )
    return normalized


def _string_tuple(value, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    normalized = tuple(value)
    for index, item in enumerate(normalized):
        if not isinstance(item, str) or not item:
            raise InputValidationError(
                f"{path}[{index}]", "must be a non-empty string"
            )
    return normalized


def _nonnegative_number(value, path: str, *, optional: bool = False):
    if value is None and optional:
        return value
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        finite = isfinite(float(value))
    except (OverflowError, ValueError):
        finite = False
    if not finite:
        raise InputValidationError(path, "must be finite")
    if value < 0:
        raise InputValidationError(path, "must be nonnegative")
    return value


@dataclass(frozen=True)
class StageCandidate:
    candidate_id: str
    plan: ParallelPlan
    metrics: tuple[StageMetrics, ...]
    feasible: bool
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    total_cards: int
    hourly_cost: float | None
    request_capacity: float
    request_capacity_per_card: float
    ttft_ms: float | None
    tpot_ms: float | None
    scenarios: tuple[WorkloadScenario, ...] = ()

    def __post_init__(self) -> None:
        from infersim.cost.types import StageMetrics

        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise InputValidationError(
                "candidate_id", "must be a non-empty string"
            )
        if not isinstance(self.plan, ParallelPlan):
            raise InputValidationError("plan", "must be a ParallelPlan")
        metrics = _record_tuple(self.metrics, "metrics", StageMetrics)
        if not metrics:
            raise InputValidationError("metrics", "must not be empty")
        if type(self.feasible) is not bool:
            raise InputValidationError("feasible", "must be a boolean")
        reason_codes = _string_tuple(self.reason_codes, "reason_codes")
        warnings = _string_tuple(self.warnings, "warnings")
        if type(self.total_cards) is not int or self.total_cards <= 0:
            raise InputValidationError(
                "total_cards", "must be a positive integer"
            )
        if self.total_cards != self.plan.total_cards:
            raise InputValidationError(
                "total_cards", "must equal plan.total_cards"
            )
        _nonnegative_number(self.hourly_cost, "hourly_cost", optional=True)
        _nonnegative_number(self.request_capacity, "request_capacity")
        _nonnegative_number(
            self.request_capacity_per_card, "request_capacity_per_card"
        )
        _nonnegative_number(self.ttft_ms, "ttft_ms", optional=True)
        _nonnegative_number(self.tpot_ms, "tpot_ms", optional=True)
        scenario_values = _record_tuple(
            self.scenarios, "scenarios", WorkloadScenario
        )
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "reason_codes", reason_codes)
        object.__setattr__(self, "warnings", warnings)
        object.__setattr__(self, "scenarios", scenario_values)


def _stable_unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _number(value, path: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    if positive and normalized <= 0:
        raise InputValidationError(path, "must be positive")
    if not positive and normalized < 0:
        raise InputValidationError(path, "must be nonnegative")
    return normalized


def _scenario_values(
    candidate: StageCandidate,
    policy: str,
    scenarios: tuple[WorkloadScenario, ...] | ScenarioSet | None,
) -> tuple[WorkloadScenario, ...]:
    if isinstance(scenarios, ScenarioSet):
        if scenarios.policy != policy:
            raise InputValidationError(
                "policy", "must match scenarios.policy"
            )
        values = scenarios.scenarios
    elif scenarios is None:
        values = candidate.scenarios
    elif type(scenarios) is tuple:
        values = scenarios
    else:
        raise InputValidationError(
            "scenarios", "must be a tuple or ScenarioSet"
        )
    return _record_tuple(values, "scenarios", WorkloadScenario)


def _match_scenarios(
    metrics: tuple[StageMetrics, ...],
    scenarios: tuple[WorkloadScenario, ...],
) -> tuple[WorkloadScenario, ...]:
    if len(metrics) != len(scenarios):
        raise InputValidationError(
            "scenarios", "must contain one scenario for each metric"
        )

    metric_names = set()
    for index, metric in enumerate(metrics):
        name = metric.scenario_name
        if not isinstance(name, str) or not name:
            raise InputValidationError(
                f"metrics[{index}].scenario_name",
                "must be a non-empty string",
            )
        if name in metric_names:
            raise InputValidationError(
                f"metrics[{index}].scenario_name", "must be unique"
            )
        metric_names.add(name)

    by_name = {}
    for index, scenario in enumerate(scenarios):
        if not isinstance(scenario.name, str) or not scenario.name:
            raise InputValidationError(
                f"scenarios[{index}].name", "must be a non-empty string"
            )
        if scenario.name in by_name:
            raise InputValidationError(
                f"scenarios[{index}].name", "must be unique"
            )
        by_name[scenario.name] = scenario

    missing = [name for name in metric_names if name not in by_name]
    if missing:
        raise InputValidationError(
            "scenarios", f"missing scenario named '{missing[0]}'"
        )
    return tuple(by_name[metric.scenario_name] for metric in metrics)


def _stage(metrics: tuple[StageMetrics, ...]) -> str:
    stages = {metric.stage for metric in metrics}
    if len(stages) != 1:
        raise InputValidationError("metrics", "must all use the same stage")
    stage = next(iter(stages))
    if stage not in ("prefill", "decode"):
        raise InputValidationError(
            "metrics[0].stage", "must be 'prefill' or 'decode'"
        )
    return stage


def _weighted_mean(values: tuple[float, ...], weights: tuple[float, ...]) -> float:
    total_weight = sum(weights)
    if not isfinite(total_weight):
        raise InputValidationError("scenarios", "weights must have a finite sum")
    return sum(value * weight for value, weight in zip(values, weights)) / total_weight


def evaluate_stage_constraints(
    candidate: StageCandidate,
    policy: str,
    scenarios: tuple[WorkloadScenario, ...] | ScenarioSet | None = None,
) -> StageCandidate:
    from infersim.cost.memory import MemoryBreakdown

    if not isinstance(candidate, StageCandidate):
        raise InputValidationError("candidate", "must be a StageCandidate")
    if policy not in ("all", "weighted"):
        raise InputValidationError("policy", "must be 'all' or 'weighted'")
    if not candidate.feasible and not candidate.reason_codes:
        raise InputValidationError(
            "candidate.feasible",
            "cannot be false when reason_codes is empty",
        )

    scenario_values = _scenario_values(candidate, policy, scenarios)
    matched = _match_scenarios(candidate.metrics, scenario_values)
    stage = _stage(candidate.metrics)
    reasons = list(candidate.reason_codes)
    warnings = list(candidate.warnings)

    capacities = []
    latencies_ms = []
    weights = []
    for index, (metric, scenario) in enumerate(zip(candidate.metrics, matched)):
        capacity = _number(
            metric.request_capacity,
            f"metrics[{index}].request_capacity",
        )
        request_rate = _number(
            scenario.request_rate,
            f"scenarios[{index}].request_rate",
            positive=True,
        )
        if type(scenario.concurrency) is not int or scenario.concurrency <= 0:
            raise InputValidationError(
                f"scenarios[{index}].concurrency",
                "must be a positive integer",
            )
        if not isinstance(metric.memory, MemoryBreakdown):
            raise InputValidationError(
                f"metrics[{index}].memory", "must be a MemoryBreakdown"
            )

        if stage == "prefill":
            latency_ms = _number(
                metric.latency_seconds,
                f"metrics[{index}].latency_seconds",
            ) * 1000
            limit_ms = _number(
                scenario.ttft_limit_ms,
                f"scenarios[{index}].ttft_limit_ms",
                positive=True,
            )
            latency_code = "TTFT_SLO"
        else:
            if metric.tpot_seconds is None:
                raise InputValidationError(
                    f"metrics[{index}].tpot_seconds",
                    "must be present for decode metrics",
                )
            latency_ms = _number(
                metric.tpot_seconds,
                f"metrics[{index}].tpot_seconds",
            ) * 1000
            limit_ms = _number(
                scenario.tpot_limit_ms,
                f"scenarios[{index}].tpot_limit_ms",
                positive=True,
            )
            latency_code = "TPOT_SLO"

        supported_concurrency = metric.max_supported_concurrency
        if supported_concurrency is None:
            supported_concurrency = (
                candidate.plan.replicas * candidate.plan.batch_size
            )
        if type(supported_concurrency) is not int or supported_concurrency < 0:
            raise InputValidationError(
                f"metrics[{index}].max_supported_concurrency",
                "must be a nonnegative integer or None",
            )

        hard_codes = []
        soft_codes = []
        prefix = f"{scenario.name}:"
        if not metric.memory.feasible:
            hard_codes.append(prefix + "MEMORY_CAPACITY")
        if latency_ms > limit_ms:
            soft_codes.append(prefix + latency_code)
        if capacity < request_rate:
            soft_codes.append(prefix + "REQUEST_RATE")
        if supported_concurrency < scenario.concurrency:
            soft_codes.append(prefix + "CONCURRENCY")
        reasons.extend(hard_codes)
        if policy == "all":
            reasons.extend(soft_codes)
        else:
            warnings.extend(soft_codes)

        capacities.append(capacity)
        latencies_ms.append(latency_ms)
        if policy == "weighted":
            weights.append(
                _number(
                    scenario.weight,
                    f"scenarios[{index}].weight",
                    positive=True,
                )
            )

    if policy == "all":
        request_capacity = min(capacities)
        latency_ms = max(latencies_ms)
    else:
        normalized_weights = tuple(weights)
        request_capacity = _weighted_mean(
            tuple(capacities), normalized_weights
        )
        latency_ms = _weighted_mean(
            tuple(latencies_ms), normalized_weights
        )

    reason_codes = _stable_unique(reasons)
    warning_codes = _stable_unique(warnings)
    return replace(
        candidate,
        feasible=not reason_codes,
        reason_codes=reason_codes,
        warnings=warning_codes,
        request_capacity=request_capacity,
        request_capacity_per_card=request_capacity / candidate.total_cards,
        ttft_ms=latency_ms if stage == "prefill" else None,
        tpot_ms=latency_ms if stage == "decode" else None,
    )
