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


def _semantic_key(candidate: StageCandidate) -> tuple:
    return (
        candidate.plan,
        candidate.total_cards,
        candidate.hourly_cost,
        candidate.request_capacity,
        candidate.request_capacity_per_card,
        candidate.ttft_ms,
        candidate.tpot_ms,
    )


def _dominates(
    left: StageCandidate,
    right: StageCandidate,
    *,
    use_cost: bool,
    use_ttft: bool,
    use_tpot: bool,
) -> bool:
    comparisons = [
        (left.total_cards, right.total_cards, True),
        (left.request_capacity, right.request_capacity, False),
    ]
    if use_cost:
        comparisons.append((left.hourly_cost, right.hourly_cost, True))
    if use_ttft:
        comparisons.append((left.ttft_ms, right.ttft_ms, True))
    if use_tpot:
        comparisons.append((left.tpot_ms, right.tpot_ms, True))

    no_worse = all(
        left_value <= right_value
        if minimize
        else left_value >= right_value
        for left_value, right_value, minimize in comparisons
    )
    strictly_better = any(
        left_value < right_value
        if minimize
        else left_value > right_value
        for left_value, right_value, minimize in comparisons
    )
    return no_worse and strictly_better


def pareto_frontier(
    candidates: Iterable[StageCandidate],
) -> list[StageCandidate]:
    values = _candidate_tuple(candidates)
    semantic_candidates = {}
    for candidate in sorted(values, key=lambda value: value.candidate_id):
        if candidate.feasible:
            semantic_candidates.setdefault(_semantic_key(candidate), candidate)
    feasible = list(semantic_candidates.values())

    use_cost = all(candidate.hourly_cost is not None for candidate in feasible)
    use_ttft = all(candidate.ttft_ms is not None for candidate in feasible)
    use_tpot = all(candidate.tpot_ms is not None for candidate in feasible)

    frontier = [
        candidate
        for candidate in feasible
        if not any(
            other is not candidate
            and _dominates(
                other,
                candidate,
                use_cost=use_cost,
                use_ttft=use_ttft,
                use_tpot=use_tpot,
            )
            for other in feasible
        )
    ]
    primary_key = recommendation_sort_key(frontier)

    def frontier_key(candidate: StageCandidate) -> tuple:
        return primary_key(candidate) + (
            inf if candidate.ttft_ms is None else candidate.ttft_ms,
            inf if candidate.tpot_ms is None else candidate.tpot_ms,
            candidate.candidate_id,
        )

    return sorted(frontier, key=frontier_key)
