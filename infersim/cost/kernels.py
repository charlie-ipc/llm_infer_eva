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


def kernel_cost(
    *,
    useful_ops: int,
    aligned_ops: int,
    compute_tops: float,
    memory_bytes: float,
    memory_bandwidth_bytes_s: float,
    launch_seconds: float,
) -> KernelCost:
    """Combine compute, memory, and launch costs with a roofline model.

    Despite its historical name, ``compute_tops`` is an ops/second value.
    Kernel-specific wrappers convert advertised TFLOPS to ops/second.
    """
    useful_ops = _nonnegative_integer(useful_ops, "useful_ops")
    aligned_ops = _nonnegative_integer(aligned_ops, "aligned_ops")
    compute_ops_s = _positive_number(compute_tops, "compute_tops")
    memory_bytes = _nonnegative_number(memory_bytes, "memory_bytes")
    memory_bandwidth_bytes_s = _positive_number(
        memory_bandwidth_bytes_s, "memory_bandwidth_bytes_s"
    )
    launch_seconds = _nonnegative_number(launch_seconds, "launch_seconds")

    compute_seconds = aligned_ops / compute_ops_s
    memory_seconds = memory_bytes / memory_bandwidth_bytes_s
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
        seconds=max(compute_seconds, memory_seconds) + launch_seconds,
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

    bytes_per_repeat = (
        m * k * precision.activation_bits / 8
        + k * n * precision.weight_bits / 8
        + m * n * precision.activation_bits / 8
    )
    return kernel_cost(
        useful_ops=2 * m * k * n * repeats,
        aligned_ops=aligned_tiles * work_per_tile,
        compute_tops=hardware.gemm_tflops[mode] * 1e12,
        memory_bytes=bytes_per_repeat * repeats,
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
        memory_bytes = (
            elements * repeats * 2 * _VECTOR_BITS[vector_mode] / 8
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
        compute_tops=hardware.vector_tflops[vector_mode] * 1e12,
        memory_bytes=memory_bytes,
        memory_bandwidth_bytes_s=hardware.memory_bandwidth_gbps * 1e9,
        launch_seconds=hardware.vector_launch_latency_us * 1e-6,
    )
