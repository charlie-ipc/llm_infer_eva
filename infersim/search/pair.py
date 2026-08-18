from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import inf, isfinite
from numbers import Real

from infersim.cost.pd import (
    PDMetrics,
    _validated_link_values,
    evaluate_pd_pair,
)
from infersim.errors import InputValidationError
from infersim.schema.scenario import PDLinkSpec, ScenarioSet, WorkloadScenario
from infersim.search.constraints import StageCandidate
from infersim.search.runner import SearchResult


_WEIGHTED_SOFT_CODES = {
    "PREFILL_RATE",
    "DECODE_RATE",
    "TTFT_SLO",
    "TPOT_SLO",
}


def _number(value, path: str, *, optional: bool = False) -> float | None:
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    if normalized < 0:
        raise InputValidationError(path, "must be nonnegative")
    return normalized


def _string_tuple(value, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    values = tuple(value)
    for index, item in enumerate(values):
        if not isinstance(item, str) or not item:
            raise InputValidationError(
                f"{path}[{index}]", "must be a non-empty string"
            )
    if len(set(values)) != len(values):
        raise InputValidationError(path, "must be unique")
    return values


def _metric_tuple(value, path: str) -> tuple[PDMetrics, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    values = tuple(value)
    if not values:
        raise InputValidationError(path, "must not be empty")
    for index, metric in enumerate(values):
        if not isinstance(metric, PDMetrics):
            raise InputValidationError(
                f"{path}[{index}]", "must be a PDMetrics"
            )
    return values


def _candidate_tuple(value, path: str) -> tuple[PDCandidate, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    values = tuple(value)
    for index, candidate in enumerate(values):
        if not isinstance(candidate, PDCandidate):
            raise InputValidationError(
                f"{path}[{index}]", "must be a PDCandidate"
            )
    return values


def _stable_unique(values) -> tuple[str, ...]:
    return tuple(dict.fromkeys(values))


def _base_reason(reason: str) -> str:
    return reason.rsplit(":", 1)[-1]


@dataclass(frozen=True)
class PDCandidate:
    candidate_id: str
    prefill_candidate_id: str
    decode_candidate_id: str
    prefill_candidate: StageCandidate
    decode_candidate: StageCandidate
    metrics: tuple[PDMetrics, ...]
    feasible: bool
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    total_cards: int
    hourly_cost: float | None
    request_capacity: float
    request_capacity_per_card: float
    ttft_ms: float
    tpot_ms: float

    def __post_init__(self) -> None:
        for path, value in (
            ("candidate_id", self.candidate_id),
            ("prefill_candidate_id", self.prefill_candidate_id),
            ("decode_candidate_id", self.decode_candidate_id),
        ):
            if not isinstance(value, str) or not value:
                raise InputValidationError(path, "must be a non-empty string")
        if not isinstance(self.prefill_candidate, StageCandidate):
            raise InputValidationError(
                "prefill_candidate", "must be a StageCandidate"
            )
        if not isinstance(self.decode_candidate, StageCandidate):
            raise InputValidationError(
                "decode_candidate", "must be a StageCandidate"
            )
        if self.prefill_candidate_id != self.prefill_candidate.candidate_id:
            raise InputValidationError(
                "prefill_candidate_id", "must equal prefill candidate ID"
            )
        if self.decode_candidate_id != self.decode_candidate.candidate_id:
            raise InputValidationError(
                "decode_candidate_id", "must equal decode candidate ID"
            )
        expected_id = f"{self.prefill_candidate_id}::{self.decode_candidate_id}"
        if self.candidate_id != expected_id:
            raise InputValidationError(
                "candidate_id", "must be derived from the phase candidate IDs"
            )
        metrics = _metric_tuple(self.metrics, "metrics")
        scenario_names = set()
        for index, metric in enumerate(metrics):
            if metric.prefill_candidate_id != self.prefill_candidate_id:
                raise InputValidationError(
                    f"metrics[{index}].prefill_candidate_id",
                    "must equal prefill_candidate_id",
                )
            if metric.decode_candidate_id != self.decode_candidate_id:
                raise InputValidationError(
                    f"metrics[{index}].decode_candidate_id",
                    "must equal decode_candidate_id",
                )
            if metric.scenario_name in scenario_names:
                raise InputValidationError(
                    f"metrics[{index}].scenario_name", "must be unique"
                )
            scenario_names.add(metric.scenario_name)
        if type(self.feasible) is not bool:
            raise InputValidationError("feasible", "must be a boolean")
        reasons = _string_tuple(self.reason_codes, "reason_codes")
        warnings = _string_tuple(self.warnings, "warnings")
        if self.feasible != (not reasons):
            raise InputValidationError(
                "feasible", "must be true exactly when reason_codes is empty"
            )
        if type(self.total_cards) is not int or self.total_cards <= 0:
            raise InputValidationError(
                "total_cards", "must be a positive integer"
            )
        expected_cards = (
            self.prefill_candidate.total_cards
            + self.decode_candidate.total_cards
        )
        if self.total_cards != expected_cards:
            raise InputValidationError(
                "total_cards", "must equal the sum of phase card counts"
            )
        expected_cost = None
        if (
            self.prefill_candidate.hourly_cost is not None
            and self.decode_candidate.hourly_cost is not None
        ):
            expected_cost = (
                self.prefill_candidate.hourly_cost
                + self.decode_candidate.hourly_cost
            )
        _number(self.hourly_cost, "hourly_cost", optional=True)
        if self.hourly_cost != expected_cost:
            raise InputValidationError(
                "hourly_cost", "must equal the sum of known phase costs"
            )
        _number(self.request_capacity, "request_capacity")
        _number(self.request_capacity_per_card, "request_capacity_per_card")
        _number(self.ttft_ms, "ttft_ms")
        _number(self.tpot_ms, "tpot_ms")
        if self.request_capacity_per_card != (
            self.request_capacity / self.total_cards
        ):
            raise InputValidationError(
                "request_capacity_per_card",
                "must equal request_capacity divided by total_cards",
            )
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "warnings", warnings)


def _pd_semantic_key(candidate: PDCandidate) -> tuple:
    return (
        _stage_semantic_key(candidate.prefill_candidate),
        _stage_semantic_key(candidate.decode_candidate),
        candidate.total_cards,
        candidate.hourly_cost,
        candidate.request_capacity,
        candidate.request_capacity_per_card,
        candidate.ttft_ms,
        candidate.tpot_ms,
    )


def _recommendation(candidates: Sequence[PDCandidate]) -> PDCandidate | None:
    feasible = tuple(candidate for candidate in candidates if candidate.feasible)
    if not feasible:
        return None
    use_cost = all(candidate.hourly_cost is not None for candidate in feasible)

    def key(candidate: PDCandidate) -> tuple:
        value = (candidate.total_cards,)
        if use_cost:
            value += (candidate.hourly_cost,)
        return value + (
            -candidate.request_capacity_per_card,
            candidate.prefill_candidate_id,
            candidate.decode_candidate_id,
        )

    return min(feasible, key=key)


def _dominates(
    left: PDCandidate, right: PDCandidate, *, use_cost: bool
) -> bool:
    comparisons = [
        (left.total_cards, right.total_cards, True),
        (left.request_capacity, right.request_capacity, False),
        (left.ttft_ms, right.ttft_ms, True),
        (left.tpot_ms, right.tpot_ms, True),
    ]
    if use_cost:
        comparisons.append((left.hourly_cost, right.hourly_cost, True))
    no_worse = all(
        a <= b if minimize else a >= b
        for a, b, minimize in comparisons
    )
    better = any(
        a < b if minimize else a > b
        for a, b, minimize in comparisons
    )
    return no_worse and better


def _pareto(candidates: Sequence[PDCandidate]) -> tuple[PDCandidate, ...]:
    semantic = {}
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        if candidate.feasible:
            semantic.setdefault(_pd_semantic_key(candidate), candidate)
    feasible = tuple(semantic.values())
    use_cost = all(candidate.hourly_cost is not None for candidate in feasible)
    frontier = tuple(
        candidate
        for candidate in feasible
        if not any(
            other is not candidate
            and _dominates(other, candidate, use_cost=use_cost)
            for other in feasible
        )
    )
    def key(candidate: PDCandidate) -> tuple:
        value = (candidate.total_cards,)
        if use_cost:
            value += (candidate.hourly_cost,)
        return value + (
            -candidate.request_capacity_per_card,
            candidate.ttft_ms,
            candidate.tpot_ms,
            candidate.prefill_candidate_id,
            candidate.decode_candidate_id,
        )

    return tuple(sorted(frontier, key=key))


def _dominant_rejection(candidates: Sequence[PDCandidate]) -> str | None:
    counts = Counter(
        _base_reason(reason)
        for candidate in candidates
        if not candidate.feasible
        for reason in candidate.reason_codes
    )
    if not counts:
        return None
    priority = {
        "PD_TRANSFER_CONCURRENCY": 0,
        "PD_LINK_RATE": 1,
        "PREFILL_RATE": 2,
        "DECODE_RATE": 2,
        "TTFT_SLO": 3,
        "TPOT_SLO": 3,
    }
    return min(
        counts,
        key=lambda code: (
            -counts[code],
            priority.get(code, len(priority)),
            code,
        ),
    )


@dataclass(frozen=True)
class PDSearchResult:
    candidates: tuple[PDCandidate, ...]
    feasible_candidates: tuple[PDCandidate, ...]
    pareto_frontier: tuple[PDCandidate, ...]
    recommendation: PDCandidate | None
    dominant_rejection: str | None
    scenario_set: ScenarioSet
    pd_link: PDLinkSpec

    def __post_init__(self) -> None:
        candidates = _candidate_tuple(self.candidates, "candidates")
        feasible = _candidate_tuple(
            self.feasible_candidates, "feasible_candidates"
        )
        frontier = _candidate_tuple(self.pareto_frontier, "pareto_frontier")
        if candidates != tuple(
            sorted(candidates, key=lambda item: item.candidate_id)
        ):
            raise InputValidationError(
                "candidates", "must be sorted by candidate_id"
            )
        ids = set()
        for index, candidate in enumerate(candidates):
            if candidate.candidate_id in ids:
                raise InputValidationError(
                    f"candidates[{index}].candidate_id", "must be unique"
                )
            ids.add(candidate.candidate_id)
        expected_feasible = tuple(
            candidate for candidate in candidates if candidate.feasible
        )
        if feasible != expected_feasible:
            raise InputValidationError(
                "feasible_candidates",
                "must exactly contain all feasible candidates",
            )
        expected_frontier = _pareto(candidates)
        if frontier != expected_frontier:
            raise InputValidationError(
                "pareto_frontier", "must equal the deterministic Pareto frontier"
            )
        expected_recommendation = _recommendation(candidates)
        if self.recommendation != expected_recommendation:
            raise InputValidationError(
                "recommendation", "must equal the deterministic recommendation"
            )
        expected_rejection = _dominant_rejection(candidates)
        if self.dominant_rejection != expected_rejection:
            raise InputValidationError(
                "dominant_rejection",
                "must equal the dominant candidate rejection",
            )
        if not isinstance(self.scenario_set, ScenarioSet):
            raise InputValidationError(
                "scenario_set", "must be a ScenarioSet"
            )
        if not isinstance(self.pd_link, PDLinkSpec):
            raise InputValidationError("pd_link", "must be a PDLinkSpec")
        scenarios = _validate_scenarios(self.scenario_set)
        _validated_link_values(self.pd_link)
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "feasible_candidates", feasible)
        object.__setattr__(self, "pareto_frontier", frontier)
        object.__setattr__(
            self,
            "scenario_set",
            ScenarioSet(self.scenario_set.policy, scenarios),
        )


def _stage_semantic_key(candidate: StageCandidate) -> tuple:
    return (
        candidate.plan,
        candidate.total_cards,
        candidate.hourly_cost,
        candidate.request_capacity,
        candidate.request_capacity_per_card,
        candidate.ttft_ms,
        candidate.tpot_ms,
    )


def _pruned_stage_candidates(result: SearchResult) -> tuple[StageCandidate, ...]:
    values = list(result.pareto_frontier)
    if result.recommendation is not None:
        values.append(result.recommendation)
    semantic = {}
    for candidate in sorted(values, key=lambda item: item.candidate_id):
        semantic.setdefault(_stage_semantic_key(candidate), candidate)
    return tuple(sorted(semantic.values(), key=lambda item: item.candidate_id))


def _validate_scenarios(scenario_set: ScenarioSet) -> tuple[WorkloadScenario, ...]:
    if not isinstance(scenario_set, ScenarioSet):
        raise InputValidationError("scenario_set", "must be a ScenarioSet")
    if scenario_set.policy not in ("all", "weighted"):
        raise InputValidationError(
            "scenario_set.policy", "must be 'all' or 'weighted'"
        )
    if isinstance(scenario_set.scenarios, (str, bytes, bytearray)) or not isinstance(
        scenario_set.scenarios, Sequence
    ):
        raise InputValidationError(
            "scenario_set.scenarios", "must be a sequence"
        )
    scenarios = tuple(scenario_set.scenarios)
    if not scenarios:
        raise InputValidationError(
            "scenario_set.scenarios", "must not be empty"
        )
    names = set()
    for index, scenario in enumerate(scenarios):
        path = f"scenario_set.scenarios[{index}]"
        if not isinstance(scenario, WorkloadScenario):
            raise InputValidationError(path, "must be a WorkloadScenario")
        if not isinstance(scenario.name, str) or not scenario.name:
            raise InputValidationError(
                path + ".name", "must be a non-empty string"
            )
        if scenario.name in names:
            raise InputValidationError(path + ".name", "must be unique")
        names.add(scenario.name)
        for field in ("input_length", "output_length", "concurrency"):
            value = getattr(scenario, field)
            if type(value) is not int:
                raise InputValidationError(path + "." + field, "must be an integer")
            if value <= 0:
                raise InputValidationError(path + "." + field, "must be positive")
        request_rate = _number(scenario.request_rate, path + ".request_rate")
        ttft_limit = _number(
            scenario.ttft_limit_ms, path + ".ttft_limit_ms"
        )
        tpot_limit = _number(
            scenario.tpot_limit_ms, path + ".tpot_limit_ms"
        )
        weight = _number(scenario.weight, path + ".weight")
        if request_rate <= 0:
            raise InputValidationError(path + ".request_rate", "must be positive")
        if ttft_limit <= 0:
            raise InputValidationError(path + ".ttft_limit_ms", "must be positive")
        if tpot_limit <= 0:
            raise InputValidationError(path + ".tpot_limit_ms", "must be positive")
        if weight <= 0:
            raise InputValidationError(path + ".weight", "must be positive")
    return scenarios


def _validate_stage_result(
    result: SearchResult,
    expected_stage: str,
    scenarios: tuple[WorkloadScenario, ...],
) -> tuple[StageCandidate, ...]:
    path = f"{expected_stage}_result"
    if not isinstance(result, SearchResult):
        raise InputValidationError(path, "must be a SearchResult")
    if result.stage != expected_stage:
        raise InputValidationError(
            path + ".stage", f"must be '{expected_stage}'"
        )
    expected_names = {scenario.name for scenario in scenarios}
    for candidate_index, candidate in enumerate(result.candidates):
        if not candidate.metrics:
            continue
        metric_names = []
        for metric_index, metric in enumerate(candidate.metrics):
            if metric.scenario_name in metric_names:
                raise InputValidationError(
                    f"{path}.candidates[{candidate_index}].metrics[{metric_index}].scenario_name",
                    "must be unique",
                )
            metric_names.append(metric.scenario_name)
        if set(metric_names) != expected_names or len(metric_names) != len(
            expected_names
        ):
            raise InputValidationError(
                f"{path}.candidates[{candidate_index}].metrics",
                "scenario names must exactly match scenario_set",
            )
    return _pruned_stage_candidates(result)


def _payloads(mapping, scenarios: tuple[WorkloadScenario, ...]) -> dict[str, int]:
    if not isinstance(mapping, Mapping):
        raise InputValidationError(
            "kv_state_bytes_by_scenario", "must be a mapping"
        )
    for index, key in enumerate(mapping):
        if not isinstance(key, str) or not key:
            raise InputValidationError(
                f"kv_state_bytes_by_scenario.keys[{index}]",
                "must be a non-empty string",
            )
    expected = {scenario.name for scenario in scenarios}
    values = {}
    for name in sorted(expected):
        path = f"kv_state_bytes_by_scenario.{name}"
        if name not in mapping:
            raise InputValidationError(path, "field is required")
        value = mapping[name]
        if type(value) is not int:
            raise InputValidationError(path, "must be an integer")
        if value <= 0:
            raise InputValidationError(path, "must be positive")
        values[name] = value
    extra = sorted(key for key in mapping if key not in expected)
    if extra:
        raise InputValidationError(
            f"kv_state_bytes_by_scenario.{extra[0]}", "is not a known scenario"
        )
    return values


def _weighted_metric(metric: PDMetrics) -> PDMetrics:
    hard = tuple(
        reason
        for reason in metric.reason_codes
        if _base_reason(reason) not in _WEIGHTED_SOFT_CODES
    )
    soft = tuple(
        reason
        for reason in metric.reason_codes
        if _base_reason(reason) in _WEIGHTED_SOFT_CODES
    )
    return replace(
        metric,
        feasible=not hard,
        reason_codes=hard,
        warnings=_stable_unique(metric.warnings + soft),
    )


def _build_candidate(
    prefill: StageCandidate,
    decode: StageCandidate,
    metrics: tuple[PDMetrics, ...],
    scenario_set: ScenarioSet,
) -> PDCandidate:
    if scenario_set.policy == "weighted":
        metrics = tuple(_weighted_metric(metric) for metric in metrics)
    reasons = _stable_unique(
        reason for metric in metrics for reason in metric.reason_codes
    )
    warnings = _stable_unique(
        tuple(prefill.warnings)
        + tuple(decode.warnings)
        + tuple(
            warning for metric in metrics for warning in metric.warnings
        )
    )
    if scenario_set.policy == "all":
        capacity = min(metric.system_request_capacity for metric in metrics)
        ttft_ms = max(metric.ttft_ms for metric in metrics)
        tpot_ms = max(metric.tpot_ms for metric in metrics)
    else:
        total_weight = sum(scenario.weight for scenario in scenario_set.scenarios)
        if not isfinite(total_weight):
            raise InputValidationError(
                "scenario_set.scenarios", "weights must have a finite sum"
            )
        by_name = {scenario.name: scenario for scenario in scenario_set.scenarios}

        def mean(field: str) -> float:
            return sum(
                getattr(metric, field) * by_name[metric.scenario_name].weight
                for metric in metrics
            ) / total_weight

        capacity = mean("system_request_capacity")
        ttft_ms = mean("ttft_ms")
        tpot_ms = mean("tpot_ms")
    total_cards = prefill.total_cards + decode.total_cards
    hourly_cost = None
    if prefill.hourly_cost is not None and decode.hourly_cost is not None:
        hourly_cost = prefill.hourly_cost + decode.hourly_cost
    return PDCandidate(
        candidate_id=f"{prefill.candidate_id}::{decode.candidate_id}",
        prefill_candidate_id=prefill.candidate_id,
        decode_candidate_id=decode.candidate_id,
        prefill_candidate=prefill,
        decode_candidate=decode,
        metrics=metrics,
        feasible=not reasons,
        reason_codes=reasons,
        warnings=warnings,
        total_cards=total_cards,
        hourly_cost=hourly_cost,
        request_capacity=capacity,
        request_capacity_per_card=capacity / total_cards,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
    )


def pair_stage_results(
    prefill_result: SearchResult,
    decode_result: SearchResult,
    pd_link: PDLinkSpec,
    scenario_set: ScenarioSet,
    kv_state_bytes_by_scenario,
) -> PDSearchResult:
    scenarios = _validate_scenarios(scenario_set)
    _validated_link_values(pd_link)
    normalized_scenario_set = ScenarioSet(scenario_set.policy, scenarios)
    prefill_candidates = _validate_stage_result(
        prefill_result, "prefill", scenarios
    )
    decode_candidates = _validate_stage_result(
        decode_result, "decode", scenarios
    )
    payloads = _payloads(kv_state_bytes_by_scenario, scenarios)

    candidates = []
    for prefill in prefill_candidates:
        for decode in decode_candidates:
            metrics = tuple(
                evaluate_pd_pair(
                    prefill,
                    decode,
                    pd_link,
                    payloads[scenario.name],
                    scenario,
                )
                for scenario in scenarios
            )
            candidates.append(
                _build_candidate(
                    prefill, decode, metrics, normalized_scenario_set
                )
            )
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    feasible = tuple(candidate for candidate in ordered if candidate.feasible)
    return PDSearchResult(
        candidates=ordered,
        feasible_candidates=feasible,
        pareto_frontier=_pareto(ordered),
        recommendation=_recommendation(ordered),
        dominant_rejection=_dominant_rejection(ordered),
        scenario_set=normalized_scenario_set,
        pd_link=pd_link,
    )
