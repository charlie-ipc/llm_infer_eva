from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec


_BIT_WIDTHS = frozenset({4, 8, 16, 32})
_VECTOR_MODES = {4: "fp4", 8: "int8", 16: "bf16", 32: "fp32"}


def _required(data: Mapping[str, Any], field: str) -> Any:
    if field not in data:
        raise InputValidationError(field, "field is required")
    return data[field]


def _bit_width(data: Mapping[str, Any], field: str) -> int:
    value = _required(data, field)
    if type(value) is not int or value not in _BIT_WIDTHS:
        raise InputValidationError(field, "must be one of 4, 8, 16, or 32")
    return value


def _optional_bit_width(
    data: Mapping[str, Any], field: str, default: int
) -> int:
    if field not in data:
        return default
    return _bit_width(data, field)


@dataclass(frozen=True)
class PrecisionSpec:
    gemm_mode: str
    weight_bits: int
    activation_bits: int
    vector_bits: int
    accumulator_bits: int
    kv_cache_bits: int
    tp_reduce_bits: int = 32
    ep_dispatch_bits: int = 8
    ep_combine_bits: int = 16

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PrecisionSpec":
        if not isinstance(data, Mapping):
            raise InputValidationError("$", "expected a mapping")
        gemm_mode = _required(data, "gemm_mode")
        if not isinstance(gemm_mode, str) or not gemm_mode:
            raise InputValidationError("gemm_mode", "must be a non-empty string")
        activation_bits = _bit_width(data, "activation_bits")
        vector_bits = _bit_width(data, "vector_bits")
        accumulator_bits = _bit_width(data, "accumulator_bits")
        kv_cache_bits = _bit_width(data, "kv_cache_bits")
        return cls(
            gemm_mode=gemm_mode,
            weight_bits=_bit_width(data, "weight_bits"),
            activation_bits=activation_bits,
            vector_bits=vector_bits,
            accumulator_bits=accumulator_bits,
            kv_cache_bits=kv_cache_bits,
            tp_reduce_bits=_optional_bit_width(
                data, "tp_reduce_bits", accumulator_bits
            ),
            ep_dispatch_bits=_optional_bit_width(
                data, "ep_dispatch_bits", activation_bits
            ),
            ep_combine_bits=_optional_bit_width(data, "ep_combine_bits", 16),
        )

    def validate_hardware(self, hardware: HardwareSpec) -> None:
        if self.gemm_mode not in hardware.gemm_tflops:
            raise InputValidationError(
                f"compute_tflops.gemm.{self.gemm_mode}",
                "required precision mode is not available",
            )
        vector_mode = _VECTOR_MODES[self.vector_bits]
        if vector_mode not in hardware.vector_tflops:
            raise InputValidationError(
                f"compute_tflops.vector.{vector_mode}",
                "required precision mode is not available",
            )
