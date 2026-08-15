from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import re

from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import PlanValidation, SearchSpace
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import ScenarioSet
from infersim.search.constraints import (
    StageCandidate,
    evaluate_stage_constraints,
)
from infersim.search.enumerate import enumerate_plans
from infersim.search.pareto import pareto_frontier as find_pareto_frontier
from infersim.search.pareto import recommend


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
        rejection = self.dominant_rejection
        if rejection is not None and (
            not isinstance(rejection, str) or not rejection
        ):
            raise InputValidationError(
                "dominant_rejection", "must be a non-empty string or None"
            )

        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "feasible_candidates", feasible)
        object.__setattr__(self, "pareto_frontier", frontier)


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
        return identity + "-valid"
    code = validation.reason_code or "INVALID_PLAN"
    normalized_code = re.sub(r"[^a-z0-9]+", "-", code.lower()).strip("-")
    return identity + "-invalid-" + (normalized_code or "invalid-plan")


def _base_reason(code: str) -> str:
    return code.rsplit(":", 1)[-1]


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
    precision.validate_hardware(hardware)

    validations = tuple(enumerate_plans(model, search_space))
    for index, validation in enumerate(validations):
        if not isinstance(validation, PlanValidation):
            raise InputValidationError(
                f"validations[{index}]", "must be a PlanValidation"
            )
    sorted_validations = sorted(validations, key=_plan_key)
    base_counts = Counter(_candidate_id(item) for item in sorted_validations)
    seen = Counter()
    evaluator = evaluate_prefill if stage == "prefill" else evaluate_decode
    candidates = []

    for validation in sorted_validations:
        base_id = _candidate_id(validation)
        seen[base_id] += 1
        candidate_id = base_id
        if base_counts[base_id] > 1:
            candidate_id += f"-n{seen[base_id]}"
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
    )
