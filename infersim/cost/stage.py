from math import isfinite

from infersim.cost.collective import (
    activation_payload_bytes,
    all_reduce_cost,
    all_to_all_cost,
)
from infersim.cost.kernels import (
    gemm_cost,
    vector_cost,
    vector_mode_for_bits,
)
from infersim.cost.memory import memory_breakdown
from infersim.cost.operations import stage_operations
from infersim.cost.types import StageMetrics
from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import ScenarioSet, WorkloadScenario
from infersim.search import validate_plan


def _finite_float(value: int | float, path: str) -> float:
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(
            path, "derived value must be finite"
        ) from None
    if not isfinite(normalized):
        raise InputValidationError(path, "derived value must be finite")
    return normalized


def _finite_product(
    left: int | float, right: int | float, path: str
) -> float:
    try:
        value = left * right
    except OverflowError:
        raise InputValidationError(
            path, "derived value must be finite"
        ) from None
    return _finite_float(value, path)


def _finite_sum(values: tuple[float, ...], path: str) -> float:
    total = 0.0
    for value in values:
        total = _finite_float(total + value, path)
    return total


def _finite_divide(
    numerator: int | float, denominator: int | float, path: str
) -> float:
    try:
        value = numerator / denominator
    except (OverflowError, ZeroDivisionError):
        raise InputValidationError(
            path, "derived value must be finite"
        ) from None
    return _finite_float(value, path)


def _validate_inputs(
    model: object,
    hardware: object,
    precision: object,
    plan: object,
    scenario: object,
) -> tuple[
    ModelSpec,
    HardwareSpec,
    PrecisionSpec,
    ParallelPlan,
    WorkloadScenario,
]:
    expected = (
        ("model", model, ModelSpec),
        ("hardware", hardware, HardwareSpec),
        ("precision", precision, PrecisionSpec),
        ("plan", plan, ParallelPlan),
        ("scenario", scenario, WorkloadScenario),
    )
    for path, value, expected_type in expected:
        if not isinstance(value, expected_type):
            raise InputValidationError(
                path, f"must be a {expected_type.__name__}"
            )

    precision.validate_hardware(hardware)
    validation = validate_plan(model, plan)
    if not validation.feasible:
        raise InputValidationError(
            "plan", f"{validation.reason_code}: {validation.reason}"
        )
    return model, hardware, precision, plan, scenario


def evaluate_prefill(
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    plan: ParallelPlan,
    scenario: WorkloadScenario,
) -> StageMetrics:
    """Evaluate one prefill plan without applying workload policy or SLOs."""

    model, hardware, precision, plan, scenario = _validate_inputs(
        model, hardware, precision, plan, scenario
    )
    operations = stage_operations(
        model,
        stage="prefill",
        batch_size=plan.batch_size,
        input_length=scenario.input_length,
        average_context=scenario.input_length,
        plan=plan,
    )

    gemm_seconds = 0.0
    useful_gemm_ops = 0
    aligned_gemm_ops = 0
    for shape in operations.gemms:
        cost = gemm_cost(
            shape.m,
            shape.k,
            shape.n,
            hardware,
            precision,
            repeats=shape.batch_repeats,
        )
        gemm_seconds = _finite_sum(
            (
                gemm_seconds,
                _finite_product(
                    cost.seconds, shape.repeats, "latency_seconds"
                ),
            ),
            "latency_seconds",
        )
        useful_gemm_ops += cost.useful_ops * shape.repeats
        aligned_gemm_ops += cost.aligned_ops * shape.repeats

    vector_seconds = 0.0
    useful_vector_ops = 0
    aligned_vector_ops = 0
    vector_mode = vector_mode_for_bits(precision.vector_bits)
    for shape in operations.vectors:
        cost = vector_cost(
            shape.elements,
            shape.ops_per_element,
            vector_mode,
            hardware,
            repeats=1,
        )
        vector_seconds = _finite_sum(
            (
                vector_seconds,
                _finite_product(
                    cost.seconds, shape.repeats, "latency_seconds"
                ),
            ),
            "latency_seconds",
        )
        useful_vector_ops += cost.useful_ops * shape.repeats
        aligned_vector_ops += cost.aligned_ops * shape.repeats

    local_tokens = (
        plan.batch_size * scenario.input_length + plan.attention_dp - 1
    ) // plan.attention_dp
    activation_payload = activation_payload_bytes(
        local_tokens * model.hidden_size, precision
    )
    attention_tp_seconds = _finite_product(
        all_reduce_cost(
            activation_payload, plan.attention_tp, hardware
        ).seconds,
        model.num_full_attention_layers,
        "latency_seconds",
    )
    ffn_tp = plan.moe_tp if model.is_moe else plan.attention_tp
    ffn_tp_seconds = _finite_product(
        all_reduce_cost(activation_payload, ffn_tp, hardware).seconds,
        model.num_hidden_layers,
        "latency_seconds",
    )
    tp_seconds = _finite_sum(
        (attention_tp_seconds, ffn_tp_seconds), "latency_seconds"
    )

    ep_seconds = 0.0
    if model.is_moe and plan.expert_parallel > 1:
        expert_payload = activation_payload_bytes(
            local_tokens * model.experts_per_token * model.hidden_size,
            precision,
        )
        ep_seconds = _finite_product(
            all_to_all_cost(
                expert_payload, plan.expert_parallel, hardware
            ).seconds,
            2 * model.num_hidden_layers,
            "latency_seconds",
        )

    memory = memory_breakdown(
        model,
        hardware,
        precision,
        plan,
        stage="prefill",
        batch_size=plan.batch_size,
        input_length=scenario.input_length,
        output_length=scenario.output_length,
    )
    component_seconds = {
        "gemm": gemm_seconds,
        "vector": vector_seconds,
        "tp": tp_seconds,
        "ep": ep_seconds,
    }
    latency_seconds = _finite_sum(
        tuple(component_seconds.values()), "latency_seconds"
    )
    if latency_seconds <= 0:
        raise InputValidationError(
            "latency_seconds", "derived value must be positive"
        )

    request_capacity = _finite_divide(
        plan.replicas * plan.batch_size,
        latency_seconds,
        "request_capacity",
    )
    prompt_token_capacity = _finite_divide(
        plan.replicas * plan.batch_size * scenario.input_length,
        latency_seconds,
        "prompt_token_capacity",
    )
    return StageMetrics(
        stage="prefill",
        scenario_name=scenario.name,
        plan=plan,
        latency_seconds=latency_seconds,
        tpot_seconds=None,
        prompt_token_capacity=prompt_token_capacity,
        output_token_capacity=None,
        request_capacity=request_capacity,
        average_context_length=_finite_float(
            scenario.input_length, "average_context_length"
        ),
        gemm_seconds=gemm_seconds,
        vector_seconds=vector_seconds,
        tp_seconds=tp_seconds,
        ep_seconds=ep_seconds,
        useful_gemm_ops=useful_gemm_ops,
        aligned_gemm_ops=aligned_gemm_ops,
        useful_vector_ops=useful_vector_ops,
        aligned_vector_ops=aligned_vector_ops,
        memory=memory,
        component_seconds=component_seconds,
    )


def evaluate_prefill_scenarios(
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    plan: ParallelPlan,
    scenario_set: ScenarioSet,
) -> tuple[StageMetrics, ...]:
    """Evaluate scenarios independently in their input order."""

    if not isinstance(scenario_set, ScenarioSet):
        raise InputValidationError(
            "scenario_set", "must be a ScenarioSet"
        )
    return tuple(
        evaluate_prefill(model, hardware, precision, plan, scenario)
        for scenario in scenario_set.scenarios
    )
