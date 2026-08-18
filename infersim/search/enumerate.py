from collections.abc import Iterable, Iterator
from itertools import product
from typing import Any

from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan, PlanValidation, SearchSpace


def _invalid(plan: ParallelPlan, code: str, reason: str) -> PlanValidation:
    return PlanValidation(
        plan=plan,
        feasible=False,
        reason_code=code,
        reason=reason,
    )


def validate_plan(
    model: ModelSpec,
    plan: ParallelPlan,
    selected_total_cards: int | None = None,
) -> PlanValidation:
    values = (
        plan.replicas,
        plan.attention_tp,
        plan.attention_dp,
        plan.moe_tp,
        plan.expert_parallel,
        plan.batch_size,
    )
    if any(type(value) is not int or value <= 0 for value in values):
        return _invalid(
            plan,
            "NONPOSITIVE_PARALLELISM",
            "parallelism and batch size values must be positive integers",
        )

    if selected_total_cards is not None and (
        type(selected_total_cards) is not int
        or selected_total_cards <= 0
        or selected_total_cards != plan.total_cards
    ):
        return _invalid(
            plan,
            "TOTAL_CARDS_MISMATCH",
            "selected total cards must be a positive integer matching the plan",
        )

    if not model.is_moe and (
        plan.attention_dp != 1
        or plan.expert_parallel != 1
        or plan.moe_tp != plan.attention_tp
    ):
        return _invalid(
            plan,
            "DENSE_PARALLELISM_INVALID",
            "dense models require attention_dp=1, expert_parallel=1, and moe_tp=attention_tp",
        )

    if model.is_moe and (
        plan.attention_tp * plan.attention_dp
        != plan.moe_tp * plan.expert_parallel
    ):
        return _invalid(
            plan,
            "MOE_WIDTH_MISMATCH",
            "attention and MoE parallel widths must match",
        )

    if model.num_attention_heads % plan.attention_tp != 0:
        return _invalid(
            plan,
            "ATTENTION_HEADS_NOT_DIVISIBLE",
            "attention heads must be divisible by attention_tp",
        )

    if (
        model.attention_kind != "mla"
        and model.num_key_value_heads % plan.attention_tp != 0
    ):
        return _invalid(
            plan,
            "KV_HEADS_NOT_DIVISIBLE",
            "key-value heads must be divisible by attention_tp",
        )

    if (
        model.num_linear_attention_layers > 0
        and model.linear_num_key_heads % plan.attention_tp != 0
    ):
        return _invalid(
            plan,
            "LINEAR_KEY_HEADS_NOT_DIVISIBLE",
            "linear attention key heads must be divisible by attention_tp",
        )

    if (
        model.num_linear_attention_layers > 0
        and model.linear_num_value_heads % plan.attention_tp != 0
    ):
        return _invalid(
            plan,
            "LINEAR_VALUE_HEADS_NOT_DIVISIBLE",
            "linear attention value heads must be divisible by attention_tp",
        )

    ffn_tp = plan.moe_tp if model.is_moe else plan.attention_tp
    if model.intermediate_size % ffn_tp != 0:
        return _invalid(
            plan,
            "INTERMEDIATE_NOT_DIVISIBLE",
            "intermediate size must be divisible by its tensor parallel width",
        )

    if (
        model.is_moe
        and model.num_shared_experts
        and model.shared_expert_intermediate_size % plan.moe_tp != 0
    ):
        return _invalid(
            plan,
            "SHARED_INTERMEDIATE_NOT_DIVISIBLE",
            "shared expert intermediate size must be divisible by moe_tp",
        )

    if model.is_moe and model.num_routed_experts % plan.expert_parallel != 0:
        return _invalid(
            plan,
            "EXPERTS_NOT_DIVISIBLE",
            "routed experts must be divisible by expert_parallel",
        )

    return PlanValidation(plan=plan, feasible=True)


def _sort_key(value: Any) -> tuple[Any, ...]:
    if type(value) is int:
        return (0, value)
    return (1, type(value).__module__, type(value).__qualname__, repr(value))


def _sorted_unique(values: Iterable[Any]) -> tuple[Any, ...]:
    unique = {
        (type(value).__module__, type(value).__qualname__, repr(value)): value
        for value in values
    }
    return tuple(sorted(unique.values(), key=_sort_key))


def enumerate_plans(
    model: ModelSpec,
    search_space: SearchSpace,
) -> Iterator[PlanValidation]:
    axes = (
        search_space.total_cards,
        search_space.replicas,
        search_space.attention_tp,
        search_space.attention_dp,
        search_space.moe_tp,
        search_space.expert_parallel,
        search_space.batch_sizes,
    )
    for candidate in product(*(_sorted_unique(axis) for axis in axes)):
        selected_total_cards, *plan_values = candidate
        plan = ParallelPlan(*plan_values)
        yield validate_plan(model, plan, selected_total_cards)
