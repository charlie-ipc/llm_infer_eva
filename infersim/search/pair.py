from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import inf, isclose, isfinite
from numbers import Real

from infersim.cost.pd import (
    PDMetrics,
    _validated_link_values,
    evaluate_pd_pair,
)
from infersim.errors import InputValidationError
from infersim.schema.scenario import PDLinkSpec, ScenarioSet, WorkloadScenario
from infersim.search.constraints import StageCandidate
from infersim.search.runner import SearchContext, SearchResult


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
        if not self.prefill_candidate.feasible:
            raise InputValidationError(
                "prefill_candidate.feasible", "must be true"
            )
        if self.prefill_candidate.reason_codes:
            raise InputValidationError(
                "prefill_candidate.reason_codes", "must be empty"
            )
        if not self.decode_candidate.feasible:
            raise InputValidationError(
                "decode_candidate.feasible", "must be true"
            )
        if self.decode_candidate.reason_codes:
            raise InputValidationError(
                "decode_candidate.reason_codes", "must be empty"
            )
        if self.prefill_candidate_id != self.prefill_candidate.candidate_id:
            raise InputValidationError(
                "prefill_candidate_id", "must equal prefill candidate ID"
            )
        if self.decode_candidate_id != self.decode_candidate.candidate_id:
            raise InputValidationError(
                "decode_candidate_id", "must equal decode candidate ID"
            )
        expected_id = _pair_candidate_id(
            self.prefill_candidate_id, self.decode_candidate_id
        )
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
        expected_reasons = tuple(
            dict.fromkeys(
                reason
                for metric in metrics
                for reason in metric.reason_codes
            )
        )
        if reasons != expected_reasons:
            raise InputValidationError(
                "reason_codes", "must equal the metric rejection reasons"
            )
        expected_warnings = tuple(
            dict.fromkeys(
                self.prefill_candidate.warnings
                + self.decode_candidate.warnings
                + tuple(
                    warning
                    for metric in metrics
                    for warning in metric.warnings
                )
            )
        )
        if warnings != expected_warnings:
            raise InputValidationError(
                "warnings", "must equal the phase and metric warnings"
            )
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
    use_cost = any(candidate.hourly_cost is not None for candidate in feasible)

    def key(candidate: PDCandidate) -> tuple:
        value = (candidate.total_cards,)
        if use_cost:
            value += (
                inf if candidate.hourly_cost is None else candidate.hourly_cost,
            )
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
    prefill_context: SearchContext | None = None
    decode_context: SearchContext | None = None
    prefill_result: SearchResult | None = None
    decode_result: SearchResult | None = None

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
        if (self.prefill_context is None) != (self.decode_context is None):
            missing = (
                "prefill_context"
                if self.prefill_context is None
                else "decode_context"
            )
            raise InputValidationError(
                missing, "must be present when the other phase has context"
            )
        for path, context in (
            ("prefill_context", self.prefill_context),
            ("decode_context", self.decode_context),
        ):
            if context is not None and not isinstance(context, SearchContext):
                raise InputValidationError(
                    path, "must be a SearchContext or None"
                )
        if (self.prefill_result is None) != (self.decode_result is None):
            missing = (
                "prefill_result"
                if self.prefill_result is None
                else "decode_result"
            )
            raise InputValidationError(
                missing, "must be present when the other phase result is bound"
            )
        for path, result in (
            ("prefill_result", self.prefill_result),
            ("decode_result", self.decode_result),
        ):
            if result is not None and not isinstance(result, SearchResult):
                raise InputValidationError(
                    path, "must be a SearchResult or None"
                )
        if self.prefill_context is not None and self.prefill_result is None:
            raise InputValidationError(
                "prefill_result", "must bind results when contexts are present"
            )
        if self.prefill_result is not None and self.prefill_context is None:
            raise InputValidationError(
                "prefill_context", "must be present when phase results are bound"
            )
        scenarios = _validate_scenarios(self.scenario_set)
        _validated_link_values(self.pd_link)
        normalized_scenario_set = ScenarioSet(
            self.scenario_set.policy, scenarios
        )
        allowed_prefill = None
        allowed_decode = None
        if self.prefill_result is not None:
            allowed_prefill = _validate_stage_result(
                self.prefill_result, "prefill", normalized_scenario_set
            )
            allowed_decode = _validate_stage_result(
                self.decode_result, "decode", normalized_scenario_set
            )
            _validate_context_compatibility(
                self.prefill_result, self.decode_result
            )
            if self.prefill_result.context is None:
                raise InputValidationError(
                    "prefill_result.context",
                    "must be present when the result is bound",
                )
            if self.decode_result.context is None:
                raise InputValidationError(
                    "decode_result.context",
                    "must be present when the result is bound",
                )
            if self.prefill_context != self.prefill_result.context:
                raise InputValidationError(
                    "prefill_context", "must equal prefill_result.context"
                )
            if self.decode_context != self.decode_result.context:
                raise InputValidationError(
                    "decode_context", "must equal decode_result.context"
                )
        for index, candidate in enumerate(candidates):
            _validate_pd_candidate_context(
                candidate,
                index,
                normalized_scenario_set,
                self.pd_link,
            )
            if allowed_prefill is not None and _stage_semantic_key(
                candidate.prefill_candidate
            ) not in {
                _stage_semantic_key(value) for value in allowed_prefill
            }:
                raise InputValidationError(
                    f"candidates[{index}].prefill_candidate",
                    "must come from the bound prefill result",
                )
            if allowed_decode is not None and _stage_semantic_key(
                candidate.decode_candidate
            ) not in {
                _stage_semantic_key(value) for value in allowed_decode
            }:
                raise InputValidationError(
                    f"candidates[{index}].decode_candidate",
                    "must come from the bound decode result",
                )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "feasible_candidates", feasible)
        object.__setattr__(self, "pareto_frontier", frontier)
        object.__setattr__(
            self,
            "scenario_set",
            normalized_scenario_set,
        )
        object.__setattr__(self, "pd_link", replace(self.pd_link))


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


def _pair_candidate_id(prefill_id: str, decode_id: str) -> str:
    return (
        f"pd:{len(prefill_id)}:{prefill_id}:"
        f"{len(decode_id)}:{decode_id}"
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


def _require_close(actual: float, expected: float, path: str) -> None:
    if not isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15):
        raise InputValidationError(path, "does not match the derived value")


def _stage_scenarios_match(
    candidate: StageCandidate,
    expected_by_name: dict[str, WorkloadScenario],
    path: str,
) -> None:
    by_name = {}
    for index, scenario in enumerate(candidate.scenarios):
        if scenario.name in by_name:
            raise InputValidationError(
                f"{path}[{index}].name", "must be unique"
            )
        by_name[scenario.name] = scenario
    if set(by_name) != set(expected_by_name) or any(
        by_name[name] != expected_by_name[name] for name in expected_by_name
    ):
        raise InputValidationError(
            path, "must exactly match the result scenarios"
        )


def _validate_pd_candidate_context(
    candidate: PDCandidate,
    candidate_index: int,
    scenario_set: ScenarioSet,
    pd_link: PDLinkSpec,
) -> None:
    from infersim.cost.pd import evaluate_pd_pair as derive_pd_pair

    base = f"candidates[{candidate_index}]"
    expected_by_name = {
        scenario.name: scenario for scenario in scenario_set.scenarios
    }
    metric_by_name = {metric.scenario_name: metric for metric in candidate.metrics}
    if set(metric_by_name) != set(expected_by_name) or len(metric_by_name) != len(
        candidate.metrics
    ):
        raise InputValidationError(
            base + ".metrics", "scenario names must exactly match scenario_set"
        )
    _stage_scenarios_match(
        candidate.prefill_candidate,
        expected_by_name,
        base + ".prefill_candidate.scenarios",
    )
    _stage_scenarios_match(
        candidate.decode_candidate,
        expected_by_name,
        base + ".decode_candidate.scenarios",
    )
    expected_metrics = []
    for metric_index, metric in enumerate(candidate.metrics):
        metric_base = f"{base}.metrics[{metric_index}]"
        scenario = expected_by_name[metric.scenario_name]
        expected = derive_pd_pair(
            candidate.prefill_candidate,
            candidate.decode_candidate,
            pd_link,
            metric.transfer.payload_bytes,
            scenario,
        )
        if scenario_set.policy == "weighted":
            expected = _weighted_metric(expected)
        expected_metrics.append(expected)
        for field in (
            "payload_bytes",
            "effective_bandwidth_bytes_per_second",
            "transfer_seconds",
            "link_request_capacity",
        ):
            _require_close(
                getattr(metric.transfer, field),
                getattr(expected.transfer, field),
                f"{metric_base}.transfer.{field}",
            )
        for field in (
            "concurrent_transfers_required",
            "concurrency_feasible",
        ):
            if getattr(metric.transfer, field) != getattr(
                expected.transfer, field
            ):
                raise InputValidationError(
                    f"{metric_base}.transfer.{field}",
                    "does not match the derived value",
                )
        for field in (
            "prefill_request_capacity",
            "decode_request_capacity",
            "system_request_capacity",
            "ttft_ms",
            "tpot_ms",
        ):
            _require_close(
                getattr(metric, field),
                getattr(expected, field),
                f"{metric_base}.{field}",
            )
        for field in (
            "bottleneck",
            "feasible",
            "reason_codes",
            "warnings",
        ):
            if getattr(metric, field) != getattr(expected, field):
                raise InputValidationError(
                    f"{metric_base}.{field}",
                    "does not match the derived value",
                )

    metrics = tuple(expected_metrics)
    if scenario_set.policy == "all":
        expected_capacity = min(
            metric.system_request_capacity for metric in metrics
        )
        expected_ttft = max(metric.ttft_ms for metric in metrics)
        expected_tpot = max(metric.tpot_ms for metric in metrics)
    else:
        total_weight = sum(
            scenario.weight for scenario in scenario_set.scenarios
        )

        def weighted(field: str) -> float:
            return sum(
                getattr(
                    next(
                        metric
                        for metric in metrics
                        if metric.scenario_name == scenario.name
                    ),
                    field,
                )
                * scenario.weight
                for scenario in scenario_set.scenarios
            ) / total_weight

        expected_capacity = weighted("system_request_capacity")
        expected_ttft = weighted("ttft_ms")
        expected_tpot = weighted("tpot_ms")
    _require_close(
        candidate.request_capacity,
        expected_capacity,
        base + ".request_capacity",
    )
    _require_close(candidate.ttft_ms, expected_ttft, base + ".ttft_ms")
    _require_close(candidate.tpot_ms, expected_tpot, base + ".tpot_ms")


def _validate_stage_result(
    result: SearchResult,
    expected_stage: str,
    scenario_set: ScenarioSet,
) -> tuple[StageCandidate, ...]:
    path = f"{expected_stage}_result"
    if not isinstance(result, SearchResult):
        raise InputValidationError(path, "must be a SearchResult")
    if result.stage != expected_stage:
        raise InputValidationError(
            path + ".stage", f"must be '{expected_stage}'"
        )
    scenarios = scenario_set.scenarios
    expected_by_name = {scenario.name: scenario for scenario in scenarios}
    expected_names = set(expected_by_name)
    if result.context is not None:
        if result.context.scenario_set != scenario_set:
            raise InputValidationError(
                path + ".context.scenario_set",
                "must equal the PD pairing scenario_set",
            )
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
        if result.context is None:
            candidate_by_name = {}
            for scenario_index, scenario in enumerate(candidate.scenarios):
                if scenario.name in candidate_by_name:
                    raise InputValidationError(
                        f"{path}.candidates[{candidate_index}].scenarios[{scenario_index}].name",
                        "must be unique",
                    )
                candidate_by_name[scenario.name] = scenario
            if set(candidate_by_name) != expected_names or any(
                candidate_by_name[name] != expected_by_name[name]
                for name in expected_names
            ):
                raise InputValidationError(
                    f"{path}.candidates[{candidate_index}].scenarios",
                    "must exactly match the PD pairing scenarios",
                )
    return _pruned_stage_candidates(result)


def _validate_context_compatibility(
    prefill_result: SearchResult,
    decode_result: SearchResult,
) -> None:
    prefill_context = prefill_result.context
    decode_context = decode_result.context
    if (prefill_context is None) != (decode_context is None):
        missing = (
            "prefill_result.context"
            if prefill_context is None
            else "decode_result.context"
        )
        raise InputValidationError(
            missing, "must be present when the other phase has context"
        )
    if prefill_context is None:
        return
    if prefill_context.model != decode_context.model:
        raise InputValidationError(
            "decode_result.context.model",
            "must equal prefill_result.context.model",
        )
    if prefill_context.precision != decode_context.precision:
        raise InputValidationError(
            "decode_result.context.precision",
            "must equal prefill_result.context.precision",
        )


def _payloads(
    mapping, scenarios: tuple[WorkloadScenario, ...]
) -> dict[str, float]:
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
        value = _number(mapping[name], path)
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
        candidate_id=_pair_candidate_id(
            prefill.candidate_id, decode.candidate_id
        ),
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
        prefill_result, "prefill", normalized_scenario_set
    )
    decode_candidates = _validate_stage_result(
        decode_result, "decode", normalized_scenario_set
    )
    _validate_context_compatibility(prefill_result, decode_result)
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
        prefill_context=prefill_result.context,
        decode_context=decode_result.context,
        prefill_result=(
            prefill_result if prefill_result.context is not None else None
        ),
        decode_result=(
            decode_result if decode_result.context is not None else None
        ),
    )
