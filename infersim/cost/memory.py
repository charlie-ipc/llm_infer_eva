from dataclasses import dataclass
from math import isfinite

from infersim.cost.operations import (
    kv_elements_per_token,
    model_counts,
    recurrent_state_bytes_per_request,
)
from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan
from infersim.schema.precision import PrecisionSpec
from infersim.search import validate_plan


_BIT_WIDTHS = frozenset({4, 8, 16, 32})


def _positive_integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _precision_bit_width(precision: PrecisionSpec, field: str) -> int:
    value = getattr(precision, field)
    if type(value) is not int or value not in _BIT_WIDTHS:
        raise InputValidationError(
            field,
            "must be one of 4, 8, 16, or 32",
        )
    return value


def _finite_float(value: int | float, path: str) -> float:
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "derived value must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "derived value must be finite")
    return normalized


def _finite_divide(
    numerator: int | float,
    denominator: int | float,
    path: str,
) -> float:
    try:
        value = numerator / denominator
    except OverflowError:
        raise InputValidationError(path, "derived value must be finite") from None
    return _finite_float(value, path)


def _finite_product(left: float, right: int | float, path: str) -> float:
    try:
        value = left * right
    except OverflowError:
        raise InputValidationError(path, "derived value must be finite") from None
    return _finite_float(value, path)


def _finite_sum(values: tuple[float, ...], path: str) -> float:
    total = 0.0
    for value in values:
        total = _finite_float(total + value, path)
    return total


@dataclass(frozen=True)
class MemoryBreakdown:
    stage: str
    embedding_weight_bytes: float
    attention_weight_bytes: float
    linear_attention_weight_bytes: float
    dense_ffn_weight_bytes: float
    routed_expert_weight_bytes: float
    shared_expert_weight_bytes: float
    total_weight_bytes: float
    kv_bytes_per_card: float
    recurrent_state_bytes_per_card: float
    activation_bytes_per_card: float
    workspace_bytes: float
    reserved_bytes: float
    resident_bytes: float
    total_required_bytes: float
    capacity_bytes: float
    usable_bytes: float
    capacity_margin_bytes: float
    feasible: bool


def kv_bytes_per_request(
    model: ModelSpec,
    precision: PrecisionSpec,
    context_length: int,
) -> float:
    """Return the full, unsharded KV cache bytes for one request."""

    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")
    if not isinstance(precision, PrecisionSpec):
        raise InputValidationError("precision", "must be a PrecisionSpec")
    context_length = _positive_integer(context_length, "context_length")
    kv_cache_bits = _precision_bit_width(precision, "kv_cache_bits")
    return _finite_divide(
        kv_elements_per_token(model)
        * context_length
        * kv_cache_bits,
        8,
        "kv_bytes_per_request",
    )


def memory_breakdown(
    model: ModelSpec,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    plan: ParallelPlan,
    *,
    stage: str,
    batch_size: int,
    input_length: int,
    output_length: int,
) -> MemoryBreakdown:
    """Estimate the peak memory working set resident on one card.

    Activation memory is a first-version two-buffer approximation for local
    residual/input-output ping-pong storage. It deliberately excludes a
    materialized attention-score tensor; kernel traffic is accounted for by
    the kernel cost model instead.
    """

    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")
    if not isinstance(hardware, HardwareSpec):
        raise InputValidationError("hardware", "must be a HardwareSpec")
    if not isinstance(precision, PrecisionSpec):
        raise InputValidationError("precision", "must be a PrecisionSpec")
    weight_bits = _precision_bit_width(precision, "weight_bits")
    activation_bits = _precision_bit_width(precision, "activation_bits")
    _precision_bit_width(precision, "kv_cache_bits")
    if not isinstance(plan, ParallelPlan):
        raise InputValidationError("plan", "must be a ParallelPlan")
    plan_validation = validate_plan(model, plan)
    if not plan_validation.feasible:
        raise InputValidationError(
            "plan",
            f"{plan_validation.reason_code}: {plan_validation.reason}",
        )
    if stage not in ("prefill", "decode"):
        raise InputValidationError("stage", "must be 'prefill' or 'decode'")
    batch_size = _positive_integer(batch_size, "batch_size")
    input_length = _positive_integer(input_length, "input_length")
    output_length = _positive_integer(output_length, "output_length")

    counts = model_counts(model)
    weight_denominator = 8
    embedding_weight_bytes = _finite_divide(
        counts.embedding_weight_elements * weight_bits,
        weight_denominator,
        "embedding_weight_bytes",
    )
    attention_weight_bytes = _finite_divide(
        counts.attention_weight_elements * weight_bits,
        weight_denominator * plan.attention_tp,
        "attention_weight_bytes",
    )
    linear_attention_weight_bytes = _finite_divide(
        counts.linear_attention_weight_elements * weight_bits,
        weight_denominator * plan.attention_tp,
        "linear_attention_weight_bytes",
    )
    dense_ffn_weight_bytes = _finite_divide(
        counts.dense_ffn_weight_elements * weight_bits,
        weight_denominator * plan.attention_tp,
        "dense_ffn_weight_bytes",
    )
    routed_expert_weight_bytes = _finite_divide(
        counts.routed_expert_weight_elements * weight_bits,
        weight_denominator * plan.moe_tp * plan.expert_parallel,
        "routed_expert_weight_bytes",
    )
    shared_expert_weight_bytes = _finite_divide(
        counts.shared_expert_weight_elements * weight_bits,
        weight_denominator * plan.moe_tp,
        "shared_expert_weight_bytes",
    )
    total_weight_bytes = _finite_sum(
        (
            embedding_weight_bytes,
            attention_weight_bytes,
            linear_attention_weight_bytes,
            dense_ffn_weight_bytes,
            routed_expert_weight_bytes,
            shared_expert_weight_bytes,
        ),
        "total_weight_bytes",
    )

    context_length = (
        input_length if stage == "prefill" else input_length + output_length
    )
    request_kv_bytes = kv_bytes_per_request(
        model,
        precision,
        context_length,
    )
    local_requests = (
        batch_size + plan.attention_dp - 1
    ) // plan.attention_dp
    kv_shards = 1 if model.attention_kind == "mla" else plan.attention_tp
    kv_bytes_per_card = _finite_divide(
        _finite_product(
            request_kv_bytes,
            local_requests,
            "kv_bytes_per_card",
        ),
        kv_shards,
        "kv_bytes_per_card",
    )

    recurrent_state_bytes_per_card = _finite_divide(
        recurrent_state_bytes_per_request(model) * local_requests,
        1,
        "recurrent_state_bytes_per_card",
    )

    local_tokens = (
        local_requests * input_length
        if stage == "prefill"
        else local_requests
    )
    activation_bytes_per_card = _finite_divide(
        2 * local_tokens * model.hidden_size * activation_bits,
        8,
        "activation_bytes_per_card",
    )

    capacity_bytes = _finite_product(
        hardware.memory_capacity_gb,
        1e9,
        "capacity_bytes",
    )
    reserved_bytes = _finite_product(
        capacity_bytes,
        hardware.memory_reserve_fraction,
        "reserved_bytes",
    )
    workspace_bytes = _finite_product(
        hardware.runtime_workspace_gb,
        1e9,
        "workspace_bytes",
    )
    usable_bytes = _finite_float(
        capacity_bytes - reserved_bytes - workspace_bytes,
        "usable_bytes",
    )
    resident_bytes = _finite_sum(
        (
            total_weight_bytes,
            kv_bytes_per_card,
            recurrent_state_bytes_per_card,
            activation_bytes_per_card,
        ),
        "resident_bytes",
    )
    total_required_bytes = _finite_sum(
        (resident_bytes, workspace_bytes, reserved_bytes),
        "total_required_bytes",
    )
    capacity_margin_bytes = _finite_float(
        usable_bytes - resident_bytes,
        "capacity_margin_bytes",
    )

    return MemoryBreakdown(
        stage=stage,
        embedding_weight_bytes=embedding_weight_bytes,
        attention_weight_bytes=attention_weight_bytes,
        linear_attention_weight_bytes=linear_attention_weight_bytes,
        dense_ffn_weight_bytes=dense_ffn_weight_bytes,
        routed_expert_weight_bytes=routed_expert_weight_bytes,
        shared_expert_weight_bytes=shared_expert_weight_bytes,
        total_weight_bytes=total_weight_bytes,
        kv_bytes_per_card=kv_bytes_per_card,
        recurrent_state_bytes_per_card=recurrent_state_bytes_per_card,
        activation_bytes_per_card=activation_bytes_per_card,
        workspace_bytes=workspace_bytes,
        reserved_bytes=reserved_bytes,
        resident_bytes=resident_bytes,
        total_required_bytes=total_required_bytes,
        capacity_bytes=capacity_bytes,
        usable_bytes=usable_bytes,
        capacity_margin_bytes=capacity_margin_bytes,
        feasible=capacity_margin_bytes >= 0,
    )
