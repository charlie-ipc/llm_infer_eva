from dataclasses import dataclass
from math import isfinite
from numbers import Real

from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.precision import PrecisionSpec


_BIT_WIDTHS = frozenset({4, 8, 16, 32})


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


def _nonnegative_number(value: object, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    if normalized < 0:
        raise InputValidationError(path, "must be nonnegative")
    return normalized


def _finite_float(value: int | float, path: str) -> float:
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "derived value must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "derived value must be finite")
    return normalized


def _finite_product(
    left: int | float, right: int | float, path: str
) -> float:
    try:
        value = left * right
    except OverflowError:
        raise InputValidationError(path, "derived value must be finite") from None
    return _finite_float(value, path)


def _finite_divide(
    numerator: int | float, denominator: int | float, path: str
) -> float:
    try:
        value = numerator / denominator
    except (OverflowError, ZeroDivisionError):
        raise InputValidationError(path, "derived value must be finite") from None
    return _finite_float(value, path)


def _finite_sum(values: tuple[float, ...], path: str) -> float:
    total = 0.0
    for value in values:
        total = _finite_float(total + value, path)
    return total


@dataclass(frozen=True)
class CollectiveCost:
    kind: str
    payload_bytes: float
    group_size: int
    path: str
    transfer_bytes: float
    bandwidth_seconds: float
    latency_seconds: float
    launch_seconds: float
    seconds: float


def activation_payload_bytes(
    elements: int, precision: PrecisionSpec
) -> float:
    """Return the activation payload size without rounding sub-byte values."""
    elements = _nonnegative_integer(elements, "elements")
    if not isinstance(precision, PrecisionSpec):
        raise InputValidationError("precision", "must be a PrecisionSpec")
    activation_bits = precision.activation_bits
    if type(activation_bits) is not int or activation_bits not in _BIT_WIDTHS:
        raise InputValidationError(
            "activation_bits", "must be one of 4, 8, 16, or 32"
        )
    return _finite_divide(
        elements * activation_bits,
        8,
        "activation_payload_bytes",
    )


def _collective_cost(
    kind: str,
    payload_bytes: float,
    group_size: int,
    hardware: HardwareSpec,
    *,
    transfer_factor: int,
    latency_factor: int,
) -> CollectiveCost:
    payload_bytes = _nonnegative_number(payload_bytes, "payload_bytes")
    group_size = _positive_integer(group_size, "group_size")
    if not isinstance(hardware, HardwareSpec):
        raise InputValidationError("hardware", "must be a HardwareSpec")

    if group_size == 1:
        return CollectiveCost(
            kind=kind,
            payload_bytes=payload_bytes,
            group_size=group_size,
            path="none",
            transfer_bytes=0.0,
            bandwidth_seconds=0.0,
            latency_seconds=0.0,
            launch_seconds=0.0,
            seconds=0.0,
        )

    if group_size <= hardware.cards_per_node:
        path = "intra_node"
        interconnect = hardware.intra_node
    else:
        path = "inter_node"
        interconnect = hardware.inter_node

    transfer_ratio = transfer_factor * (group_size - 1) / group_size
    transfer_bytes = _finite_product(
        payload_bytes, transfer_ratio, "transfer_bytes"
    )
    bandwidth_bytes_per_second = _finite_product(
        interconnect.bandwidth_gbps, 1e9, "bandwidth_seconds"
    )
    bandwidth_seconds = _finite_divide(
        transfer_bytes,
        bandwidth_bytes_per_second,
        "bandwidth_seconds",
    )
    latency_steps = latency_factor * (group_size - 1)
    latency_seconds = _finite_product(
        _finite_product(
            latency_steps, interconnect.latency_us, "latency_seconds"
        ),
        1e-6,
        "latency_seconds",
    )
    launch_seconds = _finite_product(
        hardware.collective_launch_latency_us,
        1e-6,
        "launch_seconds",
    )
    seconds = _finite_sum(
        (bandwidth_seconds, latency_seconds, launch_seconds), "seconds"
    )
    return CollectiveCost(
        kind=kind,
        payload_bytes=payload_bytes,
        group_size=group_size,
        path=path,
        transfer_bytes=transfer_bytes,
        bandwidth_seconds=bandwidth_seconds,
        latency_seconds=latency_seconds,
        launch_seconds=launch_seconds,
        seconds=seconds,
    )


def all_reduce_cost(
    payload_bytes: float, group_size: int, hardware: HardwareSpec
) -> CollectiveCost:
    return _collective_cost(
        "all_reduce",
        payload_bytes,
        group_size,
        hardware,
        transfer_factor=2,
        latency_factor=2,
    )


def all_to_all_cost(
    payload_bytes: float, group_size: int, hardware: HardwareSpec
) -> CollectiveCost:
    return _collective_cost(
        "all_to_all",
        payload_bytes,
        group_size,
        hardware,
        transfer_factor=1,
        latency_factor=1,
    )
