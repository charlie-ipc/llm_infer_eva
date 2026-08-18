from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
from math import isfinite
from numbers import Real
import re
from types import MappingProxyType

from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import PlanValidation, SearchSpace
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import ScenarioSet, WorkloadScenario
from infersim.search.constraints import (
    StageCandidate,
    evaluate_stage_constraints,
)
from infersim.search.enumerate import enumerate_plans
from infersim.search.pareto import pareto_frontier as find_pareto_frontier
from infersim.search.pareto import recommend


_ASSUMPTIONS = (
    "Analytical grid search; no P99 queueing model.",
    "Prefill and decode stages are evaluated independently.",
    "Kernel time uses tiled roofline compute and memory costs.",
    "Collective time uses configured topology bandwidth and latency.",
)


@dataclass(frozen=True)
class CandidateDiagnostic:
    candidate_id: str
    reason_code: str
    detail: str

    def __post_init__(self) -> None:
        for field, value in (
            ("candidate_id", self.candidate_id),
            ("reason_code", self.reason_code),
            ("detail", self.detail),
        ):
            if not isinstance(value, str) or not value:
                raise InputValidationError(field, "must be a non-empty string")


def _snapshot_scenarios(value: ScenarioSet) -> ScenarioSet:
    path = "context.scenario_set"
    if value.policy not in ("all", "weighted"):
        raise InputValidationError(
            f"{path}.policy", "must be 'all' or 'weighted'"
        )
    scenarios = value.scenarios
    if isinstance(scenarios, (str, bytes, bytearray)) or not isinstance(
        scenarios, Sequence
    ):
        raise InputValidationError(f"{path}.scenarios", "must be a sequence")
    if not scenarios:
        raise InputValidationError(f"{path}.scenarios", "must not be empty")
    snapshot = tuple(scenarios)
    names = set()
    for index, scenario in enumerate(snapshot):
        if not isinstance(scenario, WorkloadScenario):
            raise InputValidationError(
                f"{path}.scenarios[{index}]", "must be a WorkloadScenario"
            )
        if scenario.name in names:
            raise InputValidationError(
                f"{path}.scenarios[{index}].name", "must be unique"
            )
        names.add(scenario.name)
    return replace(value, scenarios=snapshot)


def _snapshot_search_space(value: SearchSpace) -> SearchSpace:
    axes = {}
    for axis in (
        "total_cards",
        "replicas",
        "attention_tp",
        "attention_dp",
        "moe_tp",
        "expert_parallel",
        "batch_sizes",
    ):
        path = f"context.search_space.{axis}"
        raw_values = getattr(value, axis)
        if isinstance(raw_values, (str, bytes, bytearray)) or not isinstance(
            raw_values, Sequence
        ):
            raise InputValidationError(path, "must be a sequence")
        if not raw_values:
            raise InputValidationError(path, "must not be empty")
        seen = set()
        normalized = []
        for index, item in enumerate(raw_values):
            item_path = f"{path}[{index}]"
            if type(item) is not int:
                raise InputValidationError(item_path, "must be an integer")
            if item <= 0:
                raise InputValidationError(item_path, "must be positive")
            if item in seen:
                raise InputValidationError(
                    item_path, "must not contain duplicates"
                )
            seen.add(item)
            normalized.append(item)
        axes[axis] = tuple(sorted(normalized))
    return replace(value, **axes)


def _snapshot_performance(value, path: str):
    if not isinstance(value, Mapping):
        raise InputValidationError(path, "must be a mapping")
    if not value:
        raise InputValidationError(path, "must not be empty")
    normalized = {}
    for mode, throughput in value.items():
        if not isinstance(mode, str) or not mode:
            raise InputValidationError(path, "mode names must be non-empty strings")
        value_path = f"{path}.{mode}"
        if isinstance(throughput, bool) or not isinstance(throughput, Real):
            raise InputValidationError(value_path, "must be a number")
        normalized_throughput = float(throughput)
        if not isfinite(normalized_throughput):
            raise InputValidationError(value_path, "must be finite")
        if normalized_throughput <= 0:
            raise InputValidationError(value_path, "must be positive")
        normalized[mode] = normalized_throughput
    return MappingProxyType(dict(sorted(normalized.items())))


def _snapshot_hardware(value: HardwareSpec) -> HardwareSpec:
    tile = value.gemm_tile
    path = "context.hardware.gemm_tile"
    if isinstance(tile, (str, bytes, bytearray)) or not isinstance(
        tile, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    if len(tile) != 3:
        raise InputValidationError(path, "must contain exactly three dimensions")
    normalized_tile = []
    for index, dimension in enumerate(tile):
        dimension_path = f"{path}[{index}]"
        if type(dimension) is not int:
            raise InputValidationError(dimension_path, "must be an integer")
        if dimension <= 0:
            raise InputValidationError(dimension_path, "must be positive")
        normalized_tile.append(dimension)
    return replace(
        value,
        gemm_tflops=_snapshot_performance(
            value.gemm_tflops, "context.hardware.gemm_tflops"
        ),
        vector_tflops=_snapshot_performance(
            value.vector_tflops, "context.hardware.vector_tflops"
        ),
        gemm_tile=tuple(normalized_tile),
    )


@dataclass(frozen=True)
class SearchContext:
    model: ModelSpec
    hardware: HardwareSpec
    precision: PrecisionSpec
    scenario_set: ScenarioSet
    search_space: SearchSpace
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        for path, value, expected in (
            ("context.model", self.model, ModelSpec),
            ("context.hardware", self.hardware, HardwareSpec),
            ("context.precision", self.precision, PrecisionSpec),
            ("context.scenario_set", self.scenario_set, ScenarioSet),
            ("context.search_space", self.search_space, SearchSpace),
        ):
            if not isinstance(value, expected):
                raise InputValidationError(
                    path, f"must be a {expected.__name__}"
                )
        if isinstance(self.assumptions, (str, bytes, bytearray)) or not isinstance(
            self.assumptions, Sequence
        ):
            raise InputValidationError(
                "context.assumptions", "must be a sequence"
            )
        assumptions = tuple(self.assumptions)
        for index, assumption in enumerate(assumptions):
            if not isinstance(assumption, str) or not assumption:
                raise InputValidationError(
                    f"context.assumptions[{index}]",
                    "must be a non-empty string",
                )
        object.__setattr__(self, "hardware", _snapshot_hardware(self.hardware))
        object.__setattr__(
            self, "scenario_set", _snapshot_scenarios(self.scenario_set)
        )
        object.__setattr__(
            self, "search_space", _snapshot_search_space(self.search_space)
        )
        object.__setattr__(self, "assumptions", assumptions)


def evaluate_prefill(*args, **kwargs):
    from infersim.cost.stage import evaluate_prefill as implementation

    return implementation(*args, **kwargs)


def evaluate_decode(*args, **kwargs):
    from infersim.cost.stage import evaluate_decode as implementation

    return implementation(*args, **kwargs)


def _candidate_tuple(value, path: str) -> tuple[StageCandidate, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    values = tuple(value)
    for index, candidate in enumerate(values):
        if not isinstance(candidate, StageCandidate):
            raise InputValidationError(
                f"{path}[{index}]", "must be a StageCandidate"
            )
    return values


def _contains(values: tuple[StageCandidate, ...], candidate: StageCandidate) -> bool:
    return any(value == candidate for value in values)


@dataclass(frozen=True)
class SearchResult:
    stage: str
    candidates: tuple[StageCandidate, ...]
    feasible_candidates: tuple[StageCandidate, ...]
    pareto_frontier: tuple[StageCandidate, ...]
    recommendation: StageCandidate | None
    dominant_rejection: str | None
    diagnostics: tuple[CandidateDiagnostic, ...] = ()
    context: SearchContext | None = None

    def __post_init__(self) -> None:
        if self.stage not in ("prefill", "decode"):
            raise InputValidationError(
                "stage", "must be 'prefill' or 'decode'"
            )
        candidates = _candidate_tuple(self.candidates, "candidates")
        feasible = _candidate_tuple(
            self.feasible_candidates, "feasible_candidates"
        )
        frontier = _candidate_tuple(self.pareto_frontier, "pareto_frontier")

        candidate_ids = set()
        for index, candidate in enumerate(candidates):
            if candidate.candidate_id in candidate_ids:
                raise InputValidationError(
                    f"candidates[{index}].candidate_id", "must be unique"
                )
            candidate_ids.add(candidate.candidate_id)
            for metric_index, metric in enumerate(candidate.metrics):
                if metric.stage != self.stage:
                    raise InputValidationError(
                        f"candidates[{index}].metrics[{metric_index}].stage",
                        "must equal result stage",
                    )
        if candidates != tuple(
            sorted(candidates, key=lambda item: item.candidate_id)
        ):
            raise InputValidationError(
                "candidates", "must be sorted by candidate_id"
            )

        for path, values in (
            ("feasible_candidates", feasible),
            ("pareto_frontier", frontier),
        ):
            for index, candidate in enumerate(values):
                if not candidate.feasible:
                    raise InputValidationError(
                        f"{path}[{index}].feasible", "must be true"
                    )
                if not _contains(candidates, candidate):
                    raise InputValidationError(
                        f"{path}[{index}]", "must also be in candidates"
                    )

        expected_feasible = tuple(
            candidate for candidate in candidates if candidate.feasible
        )
        if feasible != expected_feasible:
            raise InputValidationError(
                "feasible_candidates",
                "must exactly contain all feasible candidates",
            )
        expected_frontier = tuple(
            sorted(
                find_pareto_frontier(candidates),
                key=lambda item: item.candidate_id,
            )
        )
        if frontier != expected_frontier:
            raise InputValidationError(
                "pareto_frontier",
                "must equal the deterministic candidate Pareto frontier",
            )

        recommendation = self.recommendation
        if recommendation is not None:
            if not isinstance(recommendation, StageCandidate):
                raise InputValidationError(
                    "recommendation", "must be a StageCandidate or None"
                )
            if not recommendation.feasible or not _contains(
                candidates, recommendation
            ):
                raise InputValidationError(
                    "recommendation", "must be a feasible candidate"
                )
        expected_recommendation = recommend(candidates)
        if recommendation != expected_recommendation:
            raise InputValidationError(
                "recommendation", "must equal the deterministic recommendation"
            )
        rejection = self.dominant_rejection
        if rejection is not None and (
            not isinstance(rejection, str) or not rejection
        ):
            raise InputValidationError(
                "dominant_rejection", "must be a non-empty string or None"
            )
        ranked_rejections = _rank_rejections(candidates)
        expected_rejection = (
            ranked_rejections[0][0] if ranked_rejections else None
        )
        if rejection != expected_rejection:
            raise InputValidationError(
                "dominant_rejection",
                "must equal the dominant candidate rejection",
            )
        if isinstance(self.diagnostics, (str, bytes, bytearray)) or not isinstance(
            self.diagnostics, Sequence
        ):
            raise InputValidationError("diagnostics", "must be a sequence")
        diagnostics = tuple(self.diagnostics)
        for index, diagnostic in enumerate(diagnostics):
            if not isinstance(diagnostic, CandidateDiagnostic):
                raise InputValidationError(
                    f"diagnostics[{index}]", "must be a CandidateDiagnostic"
                )
        if diagnostics != tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.candidate_id,
                    item.reason_code,
                    item.detail,
                ),
            )
        ):
            raise InputValidationError(
                "diagnostics", "must be sorted deterministically"
            )
        candidate_reasons = {
            candidate.candidate_id: set(candidate.reason_codes)
            for candidate in candidates
            if not candidate.feasible
        }
        seen_diagnostic_reasons = set()
        for index, diagnostic in enumerate(diagnostics):
            if diagnostic.candidate_id not in candidate_reasons:
                raise InputValidationError(
                    f"diagnostics[{index}].candidate_id",
                    "must identify an infeasible candidate",
                )
            if diagnostic.reason_code not in candidate_reasons[
                diagnostic.candidate_id
            ]:
                raise InputValidationError(
                    f"diagnostics[{index}].reason_code",
                    "must identify a candidate rejection reason",
                )
            key = (diagnostic.candidate_id, diagnostic.reason_code)
            if key in seen_diagnostic_reasons:
                raise InputValidationError(
                    f"diagnostics[{index}]", "must be unique"
                )
            seen_diagnostic_reasons.add(key)
        expected_diagnostic_reasons = {
            (candidate_id, reason_code)
            for candidate_id, reason_codes in candidate_reasons.items()
            for reason_code in reason_codes
        }
        if seen_diagnostic_reasons != expected_diagnostic_reasons:
            raise InputValidationError(
                "diagnostics", "must describe every candidate rejection"
            )
        if self.context is not None and not isinstance(self.context, SearchContext):
            raise InputValidationError(
                "context", "must be a SearchContext or None"
            )

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "feasible_candidates", feasible)
        object.__setattr__(self, "pareto_frontier", frontier)
        object.__setattr__(self, "diagnostics", diagnostics)


def _require_type(value, expected: type, path: str) -> None:
    if not isinstance(value, expected):
        raise InputValidationError(path, f"must be a {expected.__name__}")


def _plan_key(validation: PlanValidation) -> tuple:
    plan = validation.plan
    return (
        plan.replicas,
        plan.attention_tp,
        plan.attention_dp,
        plan.moe_tp,
        plan.expert_parallel,
        plan.batch_size,
        not validation.feasible,
        validation.reason_code or "",
        validation.reason or "",
    )


def _candidate_id(validation: PlanValidation) -> str:
    plan = validation.plan
    identity = (
        f"r{plan.replicas}-atp{plan.attention_tp}-adp{plan.attention_dp}"
        f"-mtp{plan.moe_tp}-ep{plan.expert_parallel}-b{plan.batch_size}"
    )
    if validation.feasible:
        readable_status = "valid"
    else:
        code = validation.reason_code or "INVALID_PLAN"
        normalized_code = re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")
        readable_status = "invalid-" + (normalized_code or "invalid-plan")
    canonical = json.dumps(_plan_key(validation), separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]
    return f"{identity}-{readable_status}-{digest}"


def _base_reason(code: str) -> str:
    return code.rsplit(":", 1)[-1]


def _constraint_detail(reason_code: str) -> str:
    scenario_name, separator, base_code = reason_code.rpartition(":")
    if not separator:
        scenario_name = "unknown"
        base_code = reason_code
    descriptions = {
        "MEMORY_CAPACITY": "memory capacity exceeded",
        "TTFT_SLO": "TTFT limit exceeded",
        "TPOT_SLO": "TPOT limit exceeded",
        "REQUEST_RATE": "request-rate capacity is insufficient",
        "CONCURRENCY": "supported concurrency is insufficient",
    }
    description = descriptions.get(base_code, "constraint rejected")
    return f"scenario '{scenario_name}': {description}"


def _rank_rejections(
    candidates: Sequence[StageCandidate],
) -> tuple[tuple[str, int], ...]:
    counts = Counter(
        _base_reason(reason)
        for candidate in candidates
        if not candidate.feasible
        for reason in candidate.reason_codes
    )
    priority = {
        "MEMORY_CAPACITY": 0,
        "TTFT_SLO": 1,
        "TPOT_SLO": 1,
        "REQUEST_RATE": 2,
        "CONCURRENCY": 3,
    }
    return tuple(
        sorted(
            counts.items(),
            key=lambda item: (
                -item[1],
                priority.get(item[0], len(priority)),
                item[0],
            ),
        )
    )


def run_stage_search(
    stage: str,
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    scenario_set: ScenarioSet,
    search_space: SearchSpace,
) -> SearchResult:
    if stage not in ("prefill", "decode"):
        raise InputValidationError("stage", "must be 'prefill' or 'decode'")
    _require_type(model, ModelSpec, "model")
    _require_type(hardware, HardwareSpec, "hardware")
    _require_type(precision, PrecisionSpec, "precision")
    _require_type(scenario_set, ScenarioSet, "scenario_set")
    _require_type(search_space, SearchSpace, "search_space")
    context = SearchContext(
        model=model,
        hardware=hardware,
        precision=precision,
        scenario_set=scenario_set,
        search_space=search_space,
        assumptions=_ASSUMPTIONS,
    )
    model = context.model
    hardware = context.hardware
    precision = context.precision
    scenario_set = context.scenario_set
    search_space = context.search_space
    precision.validate_hardware(hardware)

    validations = tuple(enumerate_plans(model, search_space))
    for index, validation in enumerate(validations):
        if not isinstance(validation, PlanValidation):
            raise InputValidationError(
                f"validations[{index}]", "must be a PlanValidation"
            )
    sorted_validations = sorted(validations, key=_plan_key)
    occupied_ids = set()
    evaluator = evaluate_prefill if stage == "prefill" else evaluate_decode
    candidates = []
    diagnostics = []

    for validation in sorted_validations:
        base_id = _candidate_id(validation)
        candidate_id = base_id
        suffix = 1
        while candidate_id in occupied_ids:
            candidate_id = f"{base_id}-n{suffix}"
            suffix += 1
        occupied_ids.add(candidate_id)
        plan = validation.plan
        hourly_cost = (
            None
            if hardware.cost_per_card_hour is None
            else plan.total_cards * hardware.cost_per_card_hour
        )
        if not validation.feasible:
            candidate = StageCandidate(
                candidate_id=candidate_id,
                plan=plan,
                metrics=(),
                feasible=False,
                reason_codes=(validation.reason_code or "INVALID_PLAN",),
                warnings=(),
                total_cards=plan.total_cards,
                hourly_cost=hourly_cost,
                request_capacity=0,
                request_capacity_per_card=0,
                ttft_ms=None,
                tpot_ms=None,
                scenarios=(),
            )
            diagnostics.append(
                CandidateDiagnostic(
                    candidate_id,
                    validation.reason_code or "INVALID_PLAN",
                    validation.reason
                    or f"plan rejected: {validation.reason_code or 'INVALID_PLAN'}",
                )
            )
        else:
            metrics = tuple(
                evaluator(model, hardware, precision, plan, scenario)
                for scenario in scenario_set.scenarios
            )
            raw_candidate = StageCandidate(
                candidate_id=candidate_id,
                plan=plan,
                metrics=metrics,
                feasible=True,
                reason_codes=(),
                warnings=(),
                total_cards=plan.total_cards,
                hourly_cost=hourly_cost,
                request_capacity=0,
                request_capacity_per_card=0,
                ttft_ms=None,
                tpot_ms=None,
                scenarios=scenario_set.scenarios,
            )
            candidate = evaluate_stage_constraints(
                raw_candidate, scenario_set.policy
            )
            diagnostics.extend(
                CandidateDiagnostic(
                    candidate_id,
                    reason_code,
                    _constraint_detail(reason_code),
                )
                for reason_code in candidate.reason_codes
            )
        candidates.append(candidate)

    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    feasible = tuple(item for item in ordered if item.feasible)
    frontier = tuple(
        sorted(find_pareto_frontier(ordered), key=lambda item: item.candidate_id)
    )
    selected = recommend(ordered)
    ranked_rejections = _rank_rejections(ordered)
    dominant = ranked_rejections[0][0] if ranked_rejections else None
    return SearchResult(
        stage=stage,
        candidates=ordered,
        feasible_candidates=feasible,
        pareto_frontier=frontier,
        recommendation=selected,
        dominant_rejection=dominant,
        diagnostics=tuple(
            sorted(
                diagnostics,
                key=lambda item: (
                    item.candidate_id,
                    item.reason_code,
                    item.detail,
                ),
            )
        ),
        context=context,
    )
