from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from infersim.errors import InputValidationError


_AXES = (
    "total_cards",
    "replicas",
    "attention_tp",
    "attention_dp",
    "moe_tp",
    "expert_parallel",
    "batch_sizes",
)


def _powers_of_two(maximum: int) -> tuple[int, ...]:
    values = []
    value = 1
    while value <= maximum:
        values.append(value)
        value *= 2
    return tuple(values)


def _axis(value: Any, path: str, max_cards: int | None) -> tuple[int, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise InputValidationError(path, "must be a sequence")
    if not value:
        raise InputValidationError(path, "must not be empty")
    normalized = []
    seen = set()
    for index, item in enumerate(value):
        item_path = f"{path}[{index}]"
        if type(item) is not int:
            raise InputValidationError(item_path, "must be an integer")
        if item <= 0:
            raise InputValidationError(item_path, "must be positive")
        if max_cards is not None and item > max_cards:
            raise InputValidationError(
                item_path, f"must not exceed max_cards ({max_cards})"
            )
        if item in seen:
            raise InputValidationError(item_path, "must not contain duplicates")
        seen.add(item)
        normalized.append(item)
    return tuple(sorted(normalized))


@dataclass(frozen=True)
class SearchSpace:
    total_cards: tuple[int, ...]
    replicas: tuple[int, ...]
    attention_tp: tuple[int, ...]
    attention_dp: tuple[int, ...]
    moe_tp: tuple[int, ...]
    expert_parallel: tuple[int, ...]
    batch_sizes: tuple[int, ...]

    @classmethod
    def from_dict(
        cls,
        data: Mapping[str, Any] | None = None,
        max_cards: int = 64,
    ) -> "SearchSpace":
        if type(max_cards) is not int:
            raise InputValidationError("max_cards", "must be an integer")
        if max_cards <= 0:
            raise InputValidationError("max_cards", "must be positive")
        if data is None:
            data = {}
        elif not isinstance(data, Mapping):
            raise InputValidationError("$", "expected a mapping")

        defaults = _powers_of_two(max_cards)
        values = {
            axis: _axis(
                data[axis],
                axis,
                None if axis == "batch_sizes" else max_cards,
            )
            if axis in data else defaults
            for axis in _AXES
        }
        return cls(**values)
