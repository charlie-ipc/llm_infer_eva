from collections.abc import Callable, Iterable
from math import inf

from infersim.errors import InputValidationError
from infersim.search.constraints import StageCandidate


def _candidate_tuple(
    candidates: Iterable[StageCandidate],
) -> tuple[StageCandidate, ...]:
    try:
        values = tuple(candidates)
    except TypeError:
        raise InputValidationError("candidates", "must be iterable") from None
    for index, candidate in enumerate(values):
        if not isinstance(candidate, StageCandidate):
            raise InputValidationError(
                f"candidates[{index}]", "must be a StageCandidate"
            )
    return values


def recommendation_sort_key(
    candidates: Iterable[StageCandidate],
) -> Callable[[StageCandidate], tuple]:
    values = _candidate_tuple(candidates)
    use_cost = any(
        candidate.hourly_cost is not None
        for candidate in values
        if candidate.feasible
    )

    def key(candidate: StageCandidate) -> tuple:
        if not isinstance(candidate, StageCandidate):
            raise InputValidationError("candidate", "must be a StageCandidate")
        base = (candidate.total_cards,)
        if use_cost:
            base += (
                inf if candidate.hourly_cost is None else candidate.hourly_cost,
            )
        plan = candidate.plan
        return base + (
            -candidate.request_capacity_per_card,
            plan.replicas,
            plan.attention_tp,
            plan.attention_dp,
            plan.moe_tp,
            plan.expert_parallel,
            plan.batch_size,
        )

    return key


def recommend(candidates: Iterable[StageCandidate]) -> StageCandidate | None:
    values = _candidate_tuple(candidates)
    feasible = tuple(candidate for candidate in values if candidate.feasible)
    if not feasible:
        return None
    return min(feasible, key=recommendation_sort_key(feasible))


def _dominates(left: StageCandidate, right: StageCandidate) -> bool:
    no_worse = (
        left.total_cards <= right.total_cards
        and left.request_capacity >= right.request_capacity
    )
    strictly_better = (
        left.total_cards < right.total_cards
        or left.request_capacity > right.request_capacity
    )

    optional_minimums = (
        (left.hourly_cost, right.hourly_cost),
        (left.ttft_ms, right.ttft_ms),
        (left.tpot_ms, right.tpot_ms),
    )
    for left_value, right_value in optional_minimums:
        if left_value is None or right_value is None:
            continue
        no_worse = no_worse and left_value <= right_value
        strictly_better = strictly_better or left_value < right_value
    return no_worse and strictly_better


def pareto_frontier(
    candidates: Iterable[StageCandidate],
) -> list[StageCandidate]:
    values = _candidate_tuple(candidates)
    feasible = []
    seen_ids = set()
    for candidate in values:
        identity = id(candidate)
        if candidate.feasible and identity not in seen_ids:
            feasible.append(candidate)
            seen_ids.add(identity)

    frontier = [
        candidate
        for candidate in feasible
        if not any(
            other is not candidate and _dominates(other, candidate)
            for other in feasible
        )
    ]
    return sorted(frontier, key=recommendation_sort_key(frontier))
