from dataclasses import dataclass, fields
from math import isfinite
from numbers import Real

from infersim.errors import InputValidationError
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan
from infersim.search.enumerate import validate_plan


def _name(value: object, path: str) -> str:
    if not isinstance(value, str) or not value:
        raise InputValidationError(path, "must be a non-empty string")
    return value


def _positive_integral(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be an integer")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be a finite integer") from None
    if not isfinite(normalized) or not normalized.is_integer():
        raise InputValidationError(path, "must be a finite integer")
    integer = int(value)
    if integer <= 0:
        raise InputValidationError(path, "must be positive")
    return integer


def _positive_integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


@dataclass(frozen=True)
class GemmShape:
    """One GEMM shape with independent and in-launch repetition counts.

    ``repeats`` counts independent kernel launches. ``batch_repeats`` counts
    equal-shape work performed within each batched or grouped launch.
    """

    name: str
    m: int
    k: int
    n: int
    repeats: int = 1
    batch_repeats: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        for field in ("m", "k", "n", "repeats", "batch_repeats"):
            object.__setattr__(
                self,
                field,
                _positive_integral(getattr(self, field), field),
            )


@dataclass(frozen=True)
class VectorShape:
    """One vector shape whose ``repeats`` count independent launches."""

    name: str
    elements: int
    ops_per_element: int
    repeats: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _name(self.name, "name"))
        for field in ("elements", "ops_per_element", "repeats"):
            object.__setattr__(
                self,
                field,
                _positive_integral(getattr(self, field), field),
            )


@dataclass(frozen=True)
class StageOperations:
    gemms: tuple[GemmShape, ...]
    vectors: tuple[VectorShape, ...]

    def __post_init__(self) -> None:
        if type(self.gemms) is not tuple:
            raise InputValidationError("gemms", "must be a tuple")
        if type(self.vectors) is not tuple:
            raise InputValidationError("vectors", "must be a tuple")
        for index, shape in enumerate(self.gemms):
            if not isinstance(shape, GemmShape):
                raise InputValidationError(
                    f"gemms[{index}]", "must be a GemmShape"
                )
        for index, shape in enumerate(self.vectors):
            if not isinstance(shape, VectorShape):
                raise InputValidationError(
                    f"vectors[{index}]", "must be a VectorShape"
                )


@dataclass(frozen=True)
class ModelCounts:
    embedding_weight_elements: int
    attention_weight_elements: int
    linear_attention_weight_elements: int
    dense_ffn_weight_elements: int
    routed_expert_weight_elements: int
    shared_expert_weight_elements: int

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            if type(value) is not int or value < 0:
                raise InputValidationError(
                    field.name, "must be a nonnegative integer"
                )

    @property
    def total_weight_elements(self) -> int:
        return sum(getattr(self, field.name) for field in fields(self))


def model_counts(model: ModelSpec) -> ModelCounts:
    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")

    hidden = model.hidden_size
    embedding = model.vocab_size * hidden
    if not model.tie_word_embeddings:
        embedding *= 2

    if model.attention_kind == "mla":
        qk_dim = model.qk_nope_head_dim + model.qk_rope_head_dim
        if model.q_lora_rank is None:
            q_elements = hidden * model.num_attention_heads * qk_dim
        else:
            q_elements = (
                hidden * model.q_lora_rank
                + model.q_lora_rank * model.num_attention_heads * qk_dim
            )
        per_attention_layer = (
            q_elements
            + hidden * (model.kv_lora_rank + model.qk_rope_head_dim)
            + model.kv_lora_rank
            * model.num_attention_heads
            * (model.qk_nope_head_dim + model.v_head_dim)
            + hidden * model.num_attention_heads * model.v_head_dim
        )
    else:
        q_multiplier = 2 if model.attention_output_gate else 1
        per_attention_layer = (
            hidden
            * q_multiplier
            * model.num_attention_heads
            * model.head_dim
            + 2
            * hidden
            * model.num_key_value_heads
            * model.head_dim
            + hidden * model.num_attention_heads * model.head_dim
        )
    attention = model.num_full_attention_layers * per_attention_layer

    linear_attention = 0
    if model.num_linear_attention_layers:
        key_dim = model.linear_num_key_heads * model.linear_key_head_dim
        value_dim = (
            model.linear_num_value_heads * model.linear_value_head_dim
        )
        per_linear_layer = (
            2 * hidden * key_dim
            + 2 * hidden * value_dim
            + 2 * hidden * model.linear_num_value_heads
            + (2 * key_dim + value_dim) * model.linear_conv_kernel_dim
            + value_dim * hidden
        )
        linear_attention = (
            model.num_linear_attention_layers * per_linear_layer
        )

    dense_ffn = 0
    routed = 0
    shared = 0
    if model.is_moe:
        routed = (
            model.num_hidden_layers
            * model.num_routed_experts
            * 3
            * hidden
            * model.intermediate_size
        )
        shared = (
            model.num_hidden_layers
            * model.num_shared_experts
            * 3
            * hidden
            * model.shared_expert_intermediate_size
        )
    else:
        dense_ffn = (
            model.num_hidden_layers * 3 * hidden * model.intermediate_size
        )

    return ModelCounts(
        embedding_weight_elements=embedding,
        attention_weight_elements=attention,
        linear_attention_weight_elements=linear_attention,
        dense_ffn_weight_elements=dense_ffn,
        routed_expert_weight_elements=routed,
        shared_expert_weight_elements=shared,
    )


def kv_elements_per_token(model: ModelSpec) -> int:
    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")
    if model.attention_kind == "mla":
        per_layer = model.kv_lora_rank + model.qk_rope_head_dim
    else:
        per_layer = 2 * model.num_key_value_heads * model.head_dim
    return model.num_full_attention_layers * per_layer


def recurrent_state_bytes_per_request(model: ModelSpec) -> int:
    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")
    if not model.num_linear_attention_layers:
        return 0

    conv_elements = (
        model.linear_num_value_heads * model.linear_value_head_dim
        + 2 * model.linear_num_key_heads * model.linear_key_head_dim
    ) * (model.linear_conv_kernel_dim - 1)
    ssm_elements = (
        model.linear_num_value_heads
        * model.linear_key_head_dim
        * model.linear_value_head_dim
    )
    return model.num_linear_attention_layers * (
        conv_elements * 2 + ssm_elements * 4
    )


recurrent_state_bytes = recurrent_state_bytes_per_request


def _mha_operations(
    model: ModelSpec,
    plan: ParallelPlan,
    m: int,
    context: int,
) -> tuple[list[GemmShape], list[VectorShape]]:
    local_q_heads = model.num_attention_heads // plan.attention_tp
    local_kv_heads = model.num_key_value_heads // plan.attention_tp
    q_width = local_q_heads * model.head_dim
    kv_width = local_kv_heads * model.head_dim
    repeats = model.num_full_attention_layers
    q_multiplier = 2 if model.attention_output_gate else 1

    gemms = [
        GemmShape(
            "attention.q_proj",
            m,
            model.hidden_size,
            q_multiplier * q_width,
            repeats,
        ),
        GemmShape(
            "attention.k_proj", m, model.hidden_size, kv_width, repeats
        ),
        GemmShape(
            "attention.v_proj", m, model.hidden_size, kv_width, repeats
        ),
        GemmShape(
            "attention.o_proj", m, q_width, model.hidden_size, repeats
        ),
        GemmShape(
            "attention.qk",
            m,
            model.head_dim,
            context,
            repeats,
            batch_repeats=local_q_heads,
        ),
        GemmShape(
            "attention.pv",
            m,
            context,
            model.head_dim,
            repeats,
            batch_repeats=local_q_heads,
        ),
    ]
    vectors = [
        VectorShape(
            "attention.rope",
            m * (local_q_heads + local_kv_heads) * model.head_dim,
            6,
            repeats,
        ),
        VectorShape(
            "attention.softmax", m * local_q_heads * context, 5, repeats
        ),
    ]
    if model.attention_output_gate:
        vectors.append(
            VectorShape("attention.output_gate", m * q_width, 6, repeats)
        )
    return gemms, vectors


def _mla_operations(
    model: ModelSpec,
    plan: ParallelPlan,
    stage: str,
    m: int,
    context: int,
) -> tuple[list[GemmShape], list[VectorShape]]:
    local_heads = model.num_attention_heads // plan.attention_tp
    qk_dim = model.qk_nope_head_dim + model.qk_rope_head_dim
    local_qk_width = local_heads * qk_dim
    local_v_width = local_heads * model.v_head_dim
    repeats = model.num_full_attention_layers

    if model.q_lora_rank is None:
        gemms = [
            GemmShape(
                "attention.q_proj",
                m,
                model.hidden_size,
                local_qk_width,
                repeats,
            )
        ]
    else:
        gemms = [
            GemmShape(
                "attention.q_down_proj",
                m,
                model.hidden_size,
                model.q_lora_rank,
                repeats,
            ),
            GemmShape(
                "attention.q_up_proj",
                m,
                model.q_lora_rank,
                local_qk_width,
                repeats,
            ),
        ]
    gemms.append(
        GemmShape(
            "attention.kv_down_proj",
            m,
            model.hidden_size,
            model.kv_lora_rank + model.qk_rope_head_dim,
            repeats,
        )
    )

    if stage == "prefill":
        gemms.extend(
            (
                GemmShape(
                    "attention.kv_up_proj",
                    m,
                    model.kv_lora_rank,
                    local_heads
                    * (model.qk_nope_head_dim + model.v_head_dim),
                    repeats,
                ),
                GemmShape(
                    "attention.qk",
                    m,
                    qk_dim,
                    context,
                    repeats,
                    batch_repeats=local_heads,
                ),
                GemmShape(
                    "attention.pv",
                    m,
                    context,
                    model.v_head_dim,
                    repeats,
                    batch_repeats=local_heads,
                ),
                GemmShape(
                    "attention.o_proj",
                    m,
                    local_v_width,
                    model.hidden_size,
                    repeats,
                ),
            )
        )
    else:
        gemms.extend(
            (
                GemmShape(
                    "attention.q_wk",
                    m,
                    model.qk_nope_head_dim,
                    model.kv_lora_rank,
                    repeats,
                    batch_repeats=local_heads,
                ),
                GemmShape(
                    "attention.o_wv",
                    m,
                    model.kv_lora_rank,
                    model.v_head_dim,
                    repeats,
                    batch_repeats=local_heads,
                ),
                GemmShape(
                    "attention.qk",
                    m,
                    model.kv_lora_rank + model.qk_rope_head_dim,
                    context,
                    repeats,
                    batch_repeats=local_heads,
                ),
                GemmShape(
                    "attention.pv",
                    m,
                    context,
                    model.kv_lora_rank,
                    repeats,
                    batch_repeats=local_heads,
                ),
                GemmShape(
                    "attention.o_proj",
                    m,
                    local_v_width,
                    model.hidden_size,
                    repeats,
                ),
            )
        )

    vectors = [
        VectorShape(
            "attention.rope",
            m * (local_heads + 1) * model.qk_rope_head_dim,
            6,
            repeats,
        ),
        VectorShape(
            "attention.softmax", m * local_heads * context, 5, repeats
        ),
    ]
    return gemms, vectors


def _linear_attention_operations(
    model: ModelSpec, plan: ParallelPlan, m: int
) -> tuple[list[GemmShape], list[VectorShape]]:
    if not model.num_linear_attention_layers:
        return [], []
    local_key_heads = model.linear_num_key_heads // plan.attention_tp
    local_value_heads = model.linear_num_value_heads // plan.attention_tp
    local_key_dim = local_key_heads * model.linear_key_head_dim
    local_value_dim = (
        local_value_heads * model.linear_value_head_dim
    )
    output_width = (
        2 * local_key_dim + 2 * local_value_dim + 2 * local_value_heads
    )
    repeats = model.num_linear_attention_layers
    return (
        [
            GemmShape(
                "linear_attention.qkvzba_proj",
                m,
                model.hidden_size,
                output_width,
                repeats,
            ),
            GemmShape(
                "linear_attention.o_proj",
                m,
                local_value_dim,
                model.hidden_size,
                repeats,
            ),
        ],
        [
            VectorShape(
                "linear_attention.core",
                m * (2 * local_key_dim + 2 * local_value_dim),
                6,
                repeats,
            )
        ],
    )


def _ffn_operations(
    model: ModelSpec,
    plan: ParallelPlan,
    replica_tokens: int,
    attention_tokens: int,
) -> tuple[list[GemmShape], list[VectorShape]]:
    layers = model.num_hidden_layers
    if not model.is_moe:
        local_intermediate = model.intermediate_size // plan.attention_tp
        return (
            [
                GemmShape(
                    "ffn.gate_proj",
                    attention_tokens,
                    model.hidden_size,
                    local_intermediate,
                    layers,
                ),
                GemmShape(
                    "ffn.up_proj",
                    attention_tokens,
                    model.hidden_size,
                    local_intermediate,
                    layers,
                ),
                GemmShape(
                    "ffn.down_proj",
                    attention_tokens,
                    local_intermediate,
                    model.hidden_size,
                    layers,
                ),
            ],
            [
                VectorShape(
                    "ffn.silu_gate",
                    attention_tokens * local_intermediate,
                    6,
                    layers,
                )
            ],
        )

    local_experts = model.num_routed_experts // plan.expert_parallel
    local_assignments = _ceil_div(
        replica_tokens * model.experts_per_token, plan.expert_parallel
    )
    active_local = min(local_experts, local_assignments)
    tokens_per_active = _ceil_div(local_assignments, active_local)
    local_intermediate = model.intermediate_size // plan.moe_tp
    gemms = [
        GemmShape(
            "moe.routed_gate_proj",
            tokens_per_active,
            model.hidden_size,
            local_intermediate,
            layers,
            batch_repeats=active_local,
        ),
        GemmShape(
            "moe.routed_up_proj",
            tokens_per_active,
            model.hidden_size,
            local_intermediate,
            layers,
            batch_repeats=active_local,
        ),
        GemmShape(
            "moe.routed_down_proj",
            tokens_per_active,
            local_intermediate,
            model.hidden_size,
            layers,
            batch_repeats=active_local,
        ),
    ]
    vectors = [
        VectorShape(
            "moe.routing",
            attention_tokens * model.num_routed_experts,
            4,
            layers,
        ),
        VectorShape(
            "moe.routed_silu_gate",
            tokens_per_active * local_intermediate * active_local,
            6,
            layers,
        ),
    ]

    if model.num_shared_experts:
        if model.shared_expert_intermediate_size % plan.moe_tp:
            raise InputValidationError(
                "plan",
                "SHARED_INTERMEDIATE_NOT_DIVISIBLE: shared expert "
                "intermediate size must be divisible by moe_tp",
            )
        shared_m = attention_tokens
        local_shared_intermediate = (
            model.shared_expert_intermediate_size // plan.moe_tp
        )
        shared_width = model.num_shared_experts * local_shared_intermediate
        gemms.extend(
            (
                GemmShape(
                    "ffn.shared_gate_up",
                    shared_m,
                    model.hidden_size,
                    2 * shared_width,
                    layers,
                ),
                GemmShape(
                    "ffn.shared_down",
                    shared_m,
                    shared_width,
                    model.hidden_size,
                    layers,
                ),
            )
        )
        vectors.append(
            VectorShape(
                "ffn.shared_silu_gate",
                shared_m * shared_width,
                6,
                layers,
            )
        )
    return gemms, vectors


def stage_operations(
    model: ModelSpec,
    *,
    stage: str,
    batch_size: int,
    input_length: int,
    average_context: int,
    plan: ParallelPlan,
) -> StageOperations:
    """Describe per-card work while preserving independent launch counts."""

    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")
    if stage not in ("prefill", "decode"):
        raise InputValidationError("stage", "must be 'prefill' or 'decode'")
    batch_size = _positive_integer(batch_size, "batch_size")
    input_length = _positive_integer(input_length, "input_length")
    average_context = _positive_integral(
        average_context, "average_context"
    )
    if not isinstance(plan, ParallelPlan):
        raise InputValidationError("plan", "must be a ParallelPlan")
    validation = validate_plan(model, plan)
    if not validation.feasible:
        raise InputValidationError(
            "plan", f"{validation.reason_code}: {validation.reason}"
        )

    replica_tokens = (
        batch_size * input_length if stage == "prefill" else batch_size
    )
    local_requests = _ceil_div(batch_size, plan.attention_dp)
    attention_tokens = (
        local_requests * input_length
        if stage == "prefill"
        else local_requests
    )
    gemms: list[GemmShape] = []
    vectors: list[VectorShape] = []

    if model.num_full_attention_layers:
        if model.attention_kind == "mla":
            attention_gemms, attention_vectors = _mla_operations(
                model, plan, stage, attention_tokens, average_context
            )
        else:
            attention_gemms, attention_vectors = _mha_operations(
                model, plan, attention_tokens, average_context
            )
        gemms.extend(attention_gemms)
        vectors.extend(attention_vectors)

    linear_gemms, linear_vectors = _linear_attention_operations(
        model, plan, attention_tokens
    )
    gemms.extend(linear_gemms)
    vectors.extend(linear_vectors)

    ffn_gemms, ffn_vectors = _ffn_operations(
        model, plan, replica_tokens, attention_tokens
    )
    gemms.extend(ffn_gemms)
    vectors.extend(
        (
            VectorShape(
                "norm.input",
                attention_tokens * model.hidden_size,
                5,
                model.num_hidden_layers,
            ),
            VectorShape(
                "norm.post_attention",
                attention_tokens * model.hidden_size,
                5,
                model.num_hidden_layers,
            ),
            VectorShape(
                "residual.attention",
                attention_tokens * model.hidden_size,
                1,
                model.num_hidden_layers,
            ),
            VectorShape(
                "residual.ffn",
                attention_tokens * model.hidden_size,
                1,
                model.num_hidden_layers,
            ),
        )
    )
    vectors.extend(ffn_vectors)
    return StageOperations(gemms=tuple(gemms), vectors=tuple(vectors))
