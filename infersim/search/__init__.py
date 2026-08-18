from infersim.search.enumerate import enumerate_plans, validate_plan
from infersim.search.constraints import (
    StageCandidate,
    evaluate_stage_constraints,
)
from infersim.search.pareto import (
    pareto_frontier,
    recommend,
    recommendation_sort_key,
)
from infersim.search.runner import (
    CandidateDiagnostic,
    SearchContext,
    SearchResult,
    run_stage_search,
)

__all__ = [
    "CandidateDiagnostic",
    "SearchContext",
    "PDCandidate",
    "PDSearchResult",
    "StageCandidate",
    "SearchResult",
    "enumerate_plans",
    "evaluate_stage_constraints",
    "pareto_frontier",
    "pair_stage_results",
    "recommend",
    "recommendation_sort_key",
    "run_stage_search",
    "validate_plan",
]


def __getattr__(name):
    if name in {"PDCandidate", "PDSearchResult", "pair_stage_results"}:
        from infersim.search import pair

        return getattr(pair, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
