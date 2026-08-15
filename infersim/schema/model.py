from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from infersim.errors import InputValidationError, UnsupportedModelError


def _field_path(prefix: str, field: str) -> str:
    return f"{prefix}.{field}" if prefix else field


def _required(config: Mapping[str, Any], field: str, prefix: str) -> Any:
    if field not in config:
        raise InputValidationError(_field_path(prefix, field), "field is required")
    return config[field]


def _string(config: Mapping[str, Any], field: str, prefix: str) -> str:
    value = _required(config, field, prefix)
    if not isinstance(value, str) or not value:
        raise InputValidationError(
            _field_path(prefix, field), "must be a non-empty string"
        )
    return value


def _integer(value: Any, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    return value


def _positive_integer(value: Any, path: str) -> int:
    value = _integer(value, path)
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _nonnegative_integer(value: Any, path: str) -> int:
    value = _integer(value, path)
    if value < 0:
        raise InputValidationError(path, "must be nonnegative")
    return value


def _required_positive(
    config: Mapping[str, Any], field: str, prefix: str
) -> int:
    return _positive_integer(
        _required(config, field, prefix), _field_path(prefix, field)
    )


def _optional_positive(
    config: Mapping[str, Any], field: str, prefix: str
) -> int | None:
    if field not in config:
        return None
    return _positive_integer(config[field], _field_path(prefix, field))


def _aliased_value(
    config: Mapping[str, Any], fields: tuple[str, ...], default: Any
) -> tuple[str, Any]:
    for field in fields:
        if field in config:
            return field, config[field]
    return fields[0], default


def _boolean(
    config: Mapping[str, Any], fields: tuple[str, ...], prefix: str
) -> bool:
    field, value = _aliased_value(config, fields, False)
    if type(value) is not bool:
        raise InputValidationError(_field_path(prefix, field), "must be a boolean")
    return value


@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    attention_kind: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_routed_experts: int
    experts_per_token: int
    num_shared_experts: int
    shared_expert_intermediate_size: int
    tie_word_embeddings: bool
    attention_output_gate: bool
    num_full_attention_layers: int
    num_linear_attention_layers: int
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    linear_key_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_value_heads: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_routed_experts > 1

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ModelSpec":
        if not isinstance(data, Mapping):
            raise InputValidationError("$", "expected a mapping")

        if data.get("is_encoder_decoder") is True:
            raise UnsupportedModelError(
                "is_encoder_decoder", "encoder-decoder models are unsupported"
            )
        for vision_field in ("vision_config", "vision_config_dict"):
            if vision_field in data:
                raise UnsupportedModelError(
                    vision_field, "multimodal models are unsupported"
                )

        config = data
        prefix = ""
        if "text_config" in data:
            text_config = data["text_config"]
            if not isinstance(text_config, Mapping):
                raise InputValidationError("text_config", "expected a mapping")
            config = text_config
            prefix = "text_config"

        model_type = _string(config, "model_type", prefix)
        hidden_size = _required_positive(config, "hidden_size", prefix)
        num_hidden_layers = _required_positive(
            config, "num_hidden_layers", prefix
        )
        vocab_size = _required_positive(config, "vocab_size", prefix)
        num_attention_heads = _required_positive(
            config, "num_attention_heads", prefix
        )

        if "head_dim" in config:
            head_dim = _positive_integer(
                config["head_dim"], _field_path(prefix, "head_dim")
            )
        elif hidden_size % num_attention_heads == 0:
            head_dim = hidden_size // num_attention_heads
        else:
            raise InputValidationError(
                _field_path(prefix, "hidden_size"),
                "must be divisible by num_attention_heads to infer head_dim",
            )

        num_key_value_heads = _positive_integer(
            config.get("num_key_value_heads", num_attention_heads),
            _field_path(prefix, "num_key_value_heads"),
        )
        if num_attention_heads % num_key_value_heads != 0:
            raise InputValidationError(
                _field_path(prefix, "num_key_value_heads"),
                "must divide num_attention_heads",
            )

        q_lora_rank = _optional_positive(config, "q_lora_rank", prefix)
        kv_lora_rank = _optional_positive(config, "kv_lora_rank", prefix)
        qk_nope_head_dim = _optional_positive(
            config, "qk_nope_head_dim", prefix
        )
        qk_rope_head_dim = _optional_positive(
            config, "qk_rope_head_dim", prefix
        )
        v_head_dim = _optional_positive(config, "v_head_dim", prefix)

        if kv_lora_rank is not None:
            attention_kind = "mla"
        elif num_key_value_heads == num_attention_heads:
            attention_kind = "mha"
        elif num_key_value_heads == 1:
            attention_kind = "mqa"
        else:
            attention_kind = "gqa"

        routed_fields = ("num_routed_experts", "num_experts")
        routed_field, routed_value = _aliased_value(config, routed_fields, 0)
        num_routed_experts = _nonnegative_integer(
            routed_value, _field_path(prefix, routed_field)
        )
        if num_routed_experts == 1:
            raise InputValidationError(
                _field_path(prefix, routed_field),
                "must be zero for dense models or greater than one for MoE",
            )

        selected_fields = ("num_experts_per_tok", "num_experts_per_token")
        selected_present = any(field in config for field in selected_fields)
        selected_field, selected_value = _aliased_value(
            config, selected_fields, 0
        )
        experts_per_token = _nonnegative_integer(
            selected_value, _field_path(prefix, selected_field)
        )
        is_moe = num_routed_experts > 1
        if is_moe:
            if experts_per_token == 0:
                raise InputValidationError(
                    _field_path(prefix, selected_field),
                    "must be positive when routed experts are configured",
                )
            if experts_per_token > num_routed_experts:
                raise InputValidationError(
                    _field_path(prefix, selected_field),
                    "must not exceed num_routed_experts",
                )
        elif selected_present:
            raise InputValidationError(
                _field_path(prefix, selected_field),
                "is only valid for MoE models",
            )

        if not is_moe and "moe_intermediate_size" in config:
            raise InputValidationError(
                _field_path(prefix, "moe_intermediate_size"),
                "is only valid for MoE models",
            )
        if is_moe and "moe_intermediate_size" in config:
            intermediate_field = "moe_intermediate_size"
        else:
            intermediate_field = "intermediate_size"
        intermediate_size = _required_positive(
            config, intermediate_field, prefix
        )

        shared_count_present = "num_shared_experts" in config
        shared_size_present = "shared_expert_intermediate_size" in config
        total_shared_size = None
        if shared_size_present:
            total_shared_size = _positive_integer(
                config["shared_expert_intermediate_size"],
                _field_path(prefix, "shared_expert_intermediate_size"),
            )

        if shared_count_present:
            num_shared_experts = _positive_integer(
                config["num_shared_experts"],
                _field_path(prefix, "num_shared_experts"),
            )
        elif shared_size_present:
            num_shared_experts = 1
        else:
            num_shared_experts = 0

        if num_shared_experts:
            if not is_moe:
                shared_path = (
                    "num_shared_experts"
                    if shared_count_present
                    else "shared_expert_intermediate_size"
                )
                raise InputValidationError(
                    _field_path(prefix, shared_path),
                    "shared experts are only valid for MoE models",
                )
            if total_shared_size is None:
                total_shared_size = num_shared_experts * intermediate_size
            if total_shared_size % num_shared_experts != 0:
                raise InputValidationError(
                    _field_path(prefix, "shared_expert_intermediate_size"),
                    "must be divisible by num_shared_experts",
                )
            shared_expert_intermediate_size = (
                total_shared_size // num_shared_experts
            )
        else:
            shared_expert_intermediate_size = 0

        full_present = "num_full_attention_layers" in config
        linear_present = "num_linear_attention_layers" in config
        if not full_present and not linear_present:
            num_full_attention_layers = num_hidden_layers
            num_linear_attention_layers = 0
        else:
            if full_present:
                num_full_attention_layers = _nonnegative_integer(
                    config["num_full_attention_layers"],
                    _field_path(prefix, "num_full_attention_layers"),
                )
            else:
                num_full_attention_layers = num_hidden_layers - _nonnegative_integer(
                    config["num_linear_attention_layers"],
                    _field_path(prefix, "num_linear_attention_layers"),
                )
            if linear_present:
                num_linear_attention_layers = _nonnegative_integer(
                    config["num_linear_attention_layers"],
                    _field_path(prefix, "num_linear_attention_layers"),
                )
            else:
                num_linear_attention_layers = (
                    num_hidden_layers - num_full_attention_layers
                )
            if (
                num_full_attention_layers < 0
                or num_linear_attention_layers < 0
                or num_full_attention_layers + num_linear_attention_layers
                != num_hidden_layers
            ):
                raise InputValidationError(
                    _field_path(prefix, "num_hidden_layers"),
                    "must equal full plus linear attention layer counts",
                )

        linear_fields = {
            field: _optional_positive(config, field, prefix)
            for field in (
                "linear_conv_kernel_dim",
                "linear_key_head_dim",
                "linear_num_key_heads",
                "linear_value_head_dim",
                "linear_num_value_heads",
            )
        }

        return cls(
            model_type=model_type,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            vocab_size=vocab_size,
            attention_kind=attention_kind,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            head_dim=head_dim,
            intermediate_size=intermediate_size,
            num_routed_experts=num_routed_experts,
            experts_per_token=experts_per_token,
            num_shared_experts=num_shared_experts,
            shared_expert_intermediate_size=shared_expert_intermediate_size,
            tie_word_embeddings=_boolean(
                config, ("tie_word_embeddings",), prefix
            ),
            attention_output_gate=_boolean(
                config, ("attention_output_gate", "attn_output_gate"), prefix
            ),
            num_full_attention_layers=num_full_attention_layers,
            num_linear_attention_layers=num_linear_attention_layers,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            **linear_fields,
        )
