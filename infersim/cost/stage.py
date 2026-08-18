from math import ceil, floor, isfinite

from infersim.cost.collective import (
    activation_payload_bytes,
    all_reduce_cost,
    all_to_all_cost,
)
from infersim.cost.kernels import (
    gemm_cost,
    kernel_cost,
    vector_cost,
    vector_mode_for_bits,
)
from infersim.cost.memory import kv_bytes_per_request, memory_breakdown
from infersim.cost.operations import (
    recurrent_state_bytes_per_request,
    stage_operations,
)
from infersim.cost.types import StageMetrics
from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import ScenarioSet, WorkloadScenario
from infersim.search.enumerate import validate_plan


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


def _kernel_totals(
    operations,
    hardware,
    precision,
    *,
    decode_model=None,
    attention_tp=None,
    local_requests=None,
    operation_context=None,
):
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
        if (
            decode_model is not None
            and shape.name in ("attention.qk", "attention.pv")
        ):
            path = f"{shape.name}.memory_bytes"
            non_cache_bytes = _finite_divide(
                shape.batch_repeats
                * (shape.m * shape.k + shape.m * shape.n)
                * precision.activation_bits,
                8,
                path,
            )
            if decode_model.attention_kind == "mla":
                cache_width = (
                    decode_model.kv_lora_rank
                    + decode_model.qk_rope_head_dim
                    if shape.name == "attention.qk"
                    else decode_model.kv_lora_rank
                )
            else:
                local_kv_heads = (
                    decode_model.num_key_value_heads
                    // attention_tp
                )
                cache_width = local_kv_heads * decode_model.head_dim
            cache_bytes = _finite_divide(
                local_requests
                * operation_context
                * cache_width
                * precision.kv_cache_bits,
                8,
                path,
            )
            cost = kernel_cost(
                useful_ops=cost.useful_ops,
                aligned_ops=cost.aligned_ops,
                compute_ops_per_second=(
                    hardware.gemm_tflops[precision.gemm_mode] * 1e12
                ),
                memory_bytes=_finite_sum(
                    (non_cache_bytes, cache_bytes), path
                ),
                memory_bandwidth_bytes_s=(
                    hardware.memory_bandwidth_gbps * 1e9
                ),
                launch_seconds=hardware.gemm_launch_latency_us * 1e-6,
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
        memory_bytes = None
        if (
            decode_model is not None
            and shape.name == "linear_attention.core"
        ):
            path = f"{shape.name}.memory_bytes"
            default_memory_bytes = _finite_divide(
                shape.elements * 2 * precision.vector_bits,
                8,
                path,
            )
            state_read_bytes = _finite_divide(
                recurrent_state_bytes_per_request(decode_model)
                * local_requests,
                decode_model.num_linear_attention_layers * attention_tp,
                path,
            )
            memory_bytes = _finite_sum(
                (default_memory_bytes, state_read_bytes), path
            )
        cost = vector_cost(
            shape.elements,
            shape.ops_per_element,
            vector_mode,
            hardware,
            repeats=1,
            memory_bytes=memory_bytes,
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
    return (
        gemm_seconds,
        vector_seconds,
        useful_gemm_ops,
        aligned_gemm_ops,
        useful_vector_ops,
        aligned_vector_ops,
    )


def _communication_seconds(
    model,
    hardware,
    precision,
    plan,
    *,
    replica_tokens,
    local_attention_tokens,
):
    attention_payload = activation_payload_bytes(
        local_attention_tokens * model.hidden_size, precision
    )
    attention_tp_seconds = _finite_product(
        all_reduce_cost(
            attention_payload, plan.attention_tp, hardware
        ).seconds,
        model.num_full_attention_layers
        + model.num_linear_attention_layers,
        "latency_seconds",
    )
    if model.is_moe:
        routed_assignments = (
            replica_tokens * model.experts_per_token
            + plan.expert_parallel
            - 1
        ) // plan.expert_parallel
        routed_payload = activation_payload_bytes(
            routed_assignments * model.hidden_size, precision
        )
        routed_tp_seconds = _finite_product(
            all_reduce_cost(
                routed_payload, plan.moe_tp, hardware
            ).seconds,
            model.num_hidden_layers,
            "latency_seconds",
        )
        shared_tp_seconds = 0.0
        if model.num_shared_experts:
            shared_tp_seconds = _finite_product(
                all_reduce_cost(
                    attention_payload, plan.moe_tp, hardware
                ).seconds,
                model.num_hidden_layers,
                "latency_seconds",
            )
        tp_seconds = _finite_sum(
            (
                attention_tp_seconds,
                routed_tp_seconds,
                shared_tp_seconds,
            ),
            "latency_seconds",
        )
    else:
        dense_ffn_tp_seconds = _finite_product(
            all_reduce_cost(
                attention_payload, plan.attention_tp, hardware
            ).seconds,
            model.num_hidden_layers,
            "latency_seconds",
        )
        tp_seconds = _finite_sum(
            (attention_tp_seconds, dense_ffn_tp_seconds),
            "latency_seconds",
        )

    ep_seconds = 0.0
    if model.is_moe and plan.expert_parallel > 1:
        expert_payload = activation_payload_bytes(
            local_attention_tokens
            * model.experts_per_token
            * model.hidden_size,
            precision,
        )
        ep_seconds = _finite_product(
            all_to_all_cost(
                expert_payload, plan.expert_parallel, hardware
            ).seconds,
            2 * model.num_hidden_layers,
            "latency_seconds",
        )
    return tp_seconds, ep_seconds


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
    (
        gemm_seconds,
        vector_seconds,
        useful_gemm_ops,
        aligned_gemm_ops,
        useful_vector_ops,
        aligned_vector_ops,
    ) = _kernel_totals(operations, hardware, precision)

    replica_tokens = plan.batch_size * scenario.input_length
    local_requests = (
        plan.batch_size + plan.attention_dp - 1
    ) // plan.attention_dp
    local_attention_tokens = local_requests * scenario.input_length
    tp_seconds, ep_seconds = _communication_seconds(
        model,
        hardware,
        precision,
        plan,
        replica_tokens=replica_tokens,
        local_attention_tokens=local_attention_tokens,
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


def evaluate_decode(
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    plan: ParallelPlan,
    scenario: WorkloadScenario,
) -> StageMetrics:
    """Evaluate one full decode iteration independently of prefill.

    The reported average context remains exact. Odd output lengths are
    conservatively rounded up only for the integer GEMM operation shapes.
    """

    model, hardware, precision, plan, scenario = _validate_inputs(
        model, hardware, precision, plan, scenario
    )
    average_context_length = _finite_sum(
        (
            _finite_float(scenario.input_length, "average_context_length"),
            _finite_divide(
                scenario.output_length, 2, "average_context_length"
            ),
        ),
        "average_context_length",
    )
    operations = stage_operations(
        model,
        stage="decode",
        batch_size=plan.batch_size,
        input_length=scenario.input_length,
        average_context=ceil(average_context_length),
        plan=plan,
    )
    local_requests = (
        plan.batch_size + plan.attention_dp - 1
    ) // plan.attention_dp
    (
        gemm_seconds,
        vector_seconds,
        useful_gemm_ops,
        aligned_gemm_ops,
        useful_vector_ops,
        aligned_vector_ops,
    ) = _kernel_totals(
        operations,
        hardware,
        precision,
        decode_model=model,
        attention_tp=plan.attention_tp,
        local_requests=local_requests,
        operation_context=ceil(average_context_length),
    )
    tp_seconds, ep_seconds = _communication_seconds(
        model,
        hardware,
        precision,
        plan,
        replica_tokens=plan.batch_size,
        local_attention_tokens=local_requests,
    )
    memory = memory_breakdown(
        model,
        hardware,
        precision,
        plan,
        stage="decode",
        batch_size=plan.batch_size,
        input_length=scenario.input_length,
        output_length=scenario.output_length,
    )
    kv_shards = 1 if model.attention_kind == "mla" else plan.attention_tp
    variable_bytes = _finite_sum(
        (
            _finite_divide(
                kv_bytes_per_request(
                    model,
                    precision,
                    scenario.input_length + scenario.output_length,
                ),
                kv_shards,
                "max_supported_batch",
            ),
            _finite_float(
                recurrent_state_bytes_per_request(model),
                "max_supported_batch",
            ),
            _finite_divide(
                2 * model.hidden_size * precision.activation_bits,
                8,
                "max_supported_batch",
            ),
        ),
        "max_supported_batch",
    )
    if variable_bytes <= 0:
        raise InputValidationError(
            "max_supported_batch",
            "derived variable memory must be positive",
        )
    available_variable_bytes = _finite_float(
        max(0.0, memory.usable_bytes - memory.total_weight_bytes),
        "max_supported_batch",
    )
    max_local_requests = floor(
        _finite_divide(
            available_variable_bytes,
            variable_bytes,
            "max_supported_batch",
        )
    )
    max_supported_batch = max_local_requests * plan.attention_dp
    max_supported_concurrency = plan.replicas * max_supported_batch
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
    output_token_capacity = _finite_divide(
        plan.replicas * plan.batch_size,
        latency_seconds,
        "output_token_capacity",
    )
    request_capacity = _finite_divide(
        output_token_capacity,
        scenario.output_length,
        "request_capacity",
    )
    return StageMetrics(
        stage="decode",
        scenario_name=scenario.name,
        plan=plan,
        latency_seconds=latency_seconds,
        tpot_seconds=latency_seconds,
        prompt_token_capacity=None,
        output_token_capacity=output_token_capacity,
        request_capacity=request_capacity,
        average_context_length=average_context_length,
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
        max_supported_batch=max_supported_batch,
        max_supported_concurrency=max_supported_concurrency,
    )


def evaluate_decode_scenarios(
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    plan: ParallelPlan,
    scenario_set: ScenarioSet,
) -> tuple[StageMetrics, ...]:
    """Evaluate decode scenarios independently in their input order."""

    if not isinstance(scenario_set, ScenarioSet):
        raise InputValidationError(
            "scenario_set", "must be a ScenarioSet"
        )
    return tuple(
        evaluate_decode(model, hardware, precision, plan, scenario)
        for scenario in scenario_set.scenarios
    )
