from math import isfinite
from numbers import Real

from infersim.cost.types import KernelCost
from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.precision import PrecisionSpec


_VECTOR_MODES = {4: "fp4", 8: "int8", 16: "bf16", 32: "fp32"}
_VECTOR_BITS = {mode: bits for bits, mode in _VECTOR_MODES.items()}


def _positive_integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value < 0:
        raise InputValidationError(path, "must be nonnegative")
    return value


def _number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    return normalized


def _positive_number(value: object, path: str) -> float:
    normalized = _number(value, path)
    if normalized <= 0:
        raise InputValidationError(path, "must be positive")
    return normalized


def _nonnegative_number(value: object, path: str) -> float:
    normalized = _number(value, path)
    if normalized < 0:
        raise InputValidationError(path, "must be nonnegative")
    return normalized


def _ceil_div(value: int, divisor: int) -> int:
    return (value + divisor - 1) // divisor


def _finite_divide(
    numerator: int | float, denominator: int | float, path: str
) -> float:
    try:
        value = numerator / denominator
    except OverflowError:
        raise InputValidationError(path, "derived value must be finite") from None
    if not isfinite(value):
        raise InputValidationError(path, "derived value must be finite")
    return value


def _finite_add(left: float, right: float, path: str) -> float:
    value = left + right
    if not isfinite(value):
        raise InputValidationError(path, "derived value must be finite")
    return value


def kernel_cost(
    *,
    useful_ops: int,
    aligned_ops: int,
    compute_ops_per_second: float,
    memory_bytes: float,
    memory_bandwidth_bytes_s: float,
    launch_seconds: float,
) -> KernelCost:
    """Combine compute, memory, and launch costs with a roofline model.

    ``compute_ops_per_second`` is measured in ops/s. Kernel-specific wrappers
    convert advertised TFLOPS to ops/s before calling this function.
    """
    useful_ops = _nonnegative_integer(useful_ops, "useful_ops")
    aligned_ops = _nonnegative_integer(aligned_ops, "aligned_ops")
    compute_ops_per_second = _positive_number(
        compute_ops_per_second, "compute_ops_per_second"
    )
    memory_bytes = _nonnegative_number(memory_bytes, "memory_bytes")
    memory_bandwidth_bytes_s = _positive_number(
        memory_bandwidth_bytes_s, "memory_bandwidth_bytes_s"
    )
    launch_seconds = _nonnegative_number(launch_seconds, "launch_seconds")

    compute_seconds = _finite_divide(
        aligned_ops, compute_ops_per_second, "aligned_ops"
    )
    memory_seconds = _finite_divide(
        memory_bytes, memory_bandwidth_bytes_s, "memory_bytes"
    )
    bottleneck = (
        "compute" if compute_seconds >= memory_seconds else "memory"
    )
    return KernelCost(
        useful_ops=useful_ops,
        aligned_ops=aligned_ops,
        compute_seconds=compute_seconds,
        memory_bytes=memory_bytes,
        memory_seconds=memory_seconds,
        launch_seconds=launch_seconds,
        seconds=_finite_add(
            max(compute_seconds, memory_seconds), launch_seconds, "seconds"
        ),
        bottleneck=bottleneck,
    )


def gemm_cost(
    m: int,
    k: int,
    n: int,
    hardware: HardwareSpec,
    precision: PrecisionSpec,
    repeats: int = 1,
) -> KernelCost:
    """Model merged, equal-shape GEMM work with one kernel launch.

    Repeats share tile scheduling and pay launch latency once. Callers that
    represent repeated layer invocations should accumulate separate costs.
    """
    m = _positive_integer(m, "m")
    k = _positive_integer(k, "k")
    n = _positive_integer(n, "n")
    repeats = _positive_integer(repeats, "repeats")

    mode = precision.gemm_mode
    if mode not in hardware.gemm_tflops:
        raise InputValidationError(
            f"compute_tflops.gemm.{mode}",
            "required precision mode is not available",
        )

    tile_m, tile_n, tile_k = hardware.gemm_tile
    tiles_per_repeat = (
        _ceil_div(m, tile_m)
        * _ceil_div(n, tile_n)
        * _ceil_div(k, tile_k)
    )
    total_tiles = tiles_per_repeat * repeats
    aligned_tiles = (
        _ceil_div(total_tiles, hardware.gemm_engines)
        * hardware.gemm_engines
    )
    work_per_tile = 2 * tile_m * tile_n * tile_k

    memory_bits = repeats * (
        m * k * precision.activation_bits
        + k * n * precision.weight_bits
        + m * n * precision.activation_bits
    )
    memory_bytes = _finite_divide(memory_bits, 8, "memory_bytes")
    return kernel_cost(
        useful_ops=2 * m * k * n * repeats,
        aligned_ops=aligned_tiles * work_per_tile,
        compute_ops_per_second=hardware.gemm_tflops[mode] * 1e12,
        memory_bytes=memory_bytes,
        memory_bandwidth_bytes_s=hardware.memory_bandwidth_gbps * 1e9,
        launch_seconds=hardware.gemm_launch_latency_us * 1e-6,
    )


def vector_mode_for_bits(bits: int) -> str:
    if type(bits) is not int or bits not in _VECTOR_MODES:
        raise InputValidationError(
            "bits", "must be one of 4, 8, 16, or 32"
        )
    return _VECTOR_MODES[bits]


def vector_cost(
    elements: int,
    ops_per_element: int,
    vector_mode: str,
    hardware: HardwareSpec,
    repeats: int = 1,
    memory_bytes: float | None = None,
) -> KernelCost:
    """Model vector work, aligning every repeat to a full vector wave.

    Repeats pay alignment independently but share one kernel launch.
    """
    elements = _positive_integer(elements, "elements")
    ops_per_element = _positive_integer(ops_per_element, "ops_per_element")
    repeats = _positive_integer(repeats, "repeats")
    if not isinstance(vector_mode, str) or not vector_mode:
        raise InputValidationError(
            "vector_mode", "must be a non-empty string"
        )
    if vector_mode not in hardware.vector_tflops:
        raise InputValidationError(
            f"compute_tflops.vector.{vector_mode}",
            "required precision mode is not available",
        )

    if memory_bytes is None:
        if vector_mode not in _VECTOR_BITS:
            raise InputValidationError(
                "vector_mode",
                "memory bytes require a canonical precision mode",
            )
        memory_bytes = _finite_divide(
            elements * repeats * 2 * _VECTOR_BITS[vector_mode],
            8,
            "memory_bytes",
        )
    else:
        memory_bytes = _nonnegative_number(memory_bytes, "memory_bytes")

    vectors_per_repeat = _ceil_div(elements, hardware.vector_width)
    aligned_elements_per_repeat = (
        _ceil_div(vectors_per_repeat, hardware.vector_units)
        * hardware.vector_units
        * hardware.vector_width
    )
    return kernel_cost(
        useful_ops=elements * ops_per_element * repeats,
        aligned_ops=(
            aligned_elements_per_repeat * ops_per_element * repeats
        ),
        compute_ops_per_second=hardware.vector_tflops[vector_mode] * 1e12,
        memory_bytes=memory_bytes,
        memory_bandwidth_bytes_s=hardware.memory_bandwidth_gbps * 1e9,
        launch_seconds=hardware.vector_launch_latency_us * 1e-6,
    )
