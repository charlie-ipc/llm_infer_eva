from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import ceil, isfinite
from numbers import Real
from typing import TYPE_CHECKING

from infersim.cost.memory import kv_bytes_per_request
from infersim.cost.operations import recurrent_state_bytes_per_request
from infersim.errors import InputValidationError
from infersim.schema.model import ModelSpec
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import PDLinkSpec, WorkloadScenario

if TYPE_CHECKING:
    from infersim.search.constraints import StageCandidate


def _number(
    value,
    path: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (OverflowError, ValueError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    if positive and normalized <= 0:
        raise InputValidationError(path, "must be positive")
    if nonnegative and normalized < 0:
        raise InputValidationError(path, "must be nonnegative")
    return normalized


def _positive_integer(value, path: str) -> int:
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value <= 0:
        raise InputValidationError(path, "must be positive")
    return value


def _string_tuple(value, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(
        value, Sequence
    ):
        raise InputValidationError(path, "must be a sequence")
    normalized = tuple(value)
    for index, item in enumerate(normalized):
        if not isinstance(item, str) or not item:
            raise InputValidationError(
                f"{path}[{index}]", "must be a non-empty string"
            )
    return normalized


def _validate_scenario(scenario: WorkloadScenario) -> None:
    if not isinstance(scenario, WorkloadScenario):
        raise InputValidationError("scenario", "must be a WorkloadScenario")
    if not isinstance(scenario.name, str) or not scenario.name:
        raise InputValidationError(
            "scenario.name", "must be a non-empty string"
        )
    _positive_integer(scenario.input_length, "scenario.input_length")
    _positive_integer(scenario.output_length, "scenario.output_length")
    _number(scenario.request_rate, "scenario.request_rate", positive=True)
    _positive_integer(scenario.concurrency, "scenario.concurrency")
    _number(scenario.ttft_limit_ms, "scenario.ttft_limit_ms", positive=True)
    _number(scenario.tpot_limit_ms, "scenario.tpot_limit_ms", positive=True)
    _number(scenario.weight, "scenario.weight", positive=True)


def _validated_link_values(pd_link: PDLinkSpec) -> tuple[float, float, float, int]:
    if not isinstance(pd_link, PDLinkSpec):
        raise InputValidationError("pd_link", "must be a PDLinkSpec")
    bandwidth = _number(
        pd_link.bandwidth_gbps, "pd_link.bandwidth_gbps", positive=True
    )
    latency = _number(
        pd_link.latency_us, "pd_link.latency_us", nonnegative=True
    )
    efficiency = _number(
        pd_link.efficiency, "pd_link.efficiency", positive=True
    )
    if efficiency > 1:
        raise InputValidationError(
            "pd_link.efficiency", "must be in the range (0, 1]"
        )
    concurrency = _positive_integer(
        pd_link.max_concurrent_transfers,
        "pd_link.max_concurrent_transfers",
    )
    return bandwidth, latency, efficiency, concurrency


@dataclass(frozen=True)
class PDTransferMetrics:
    scenario_name: str
    payload_bytes: int
    effective_bandwidth_bytes_per_second: float
    transfer_seconds: float
    link_request_capacity: float
    concurrent_transfers_required: int
    concurrency_feasible: bool

    def __post_init__(self) -> None:
        if not isinstance(self.scenario_name, str) or not self.scenario_name:
            raise InputValidationError(
                "scenario_name", "must be a non-empty string"
            )
        _positive_integer(self.payload_bytes, "payload_bytes")
        _number(
            self.effective_bandwidth_bytes_per_second,
            "effective_bandwidth_bytes_per_second",
            positive=True,
        )
        _number(self.transfer_seconds, "transfer_seconds", positive=True)
        _number(
            self.link_request_capacity,
            "link_request_capacity",
            positive=True,
        )
        _positive_integer(
            self.concurrent_transfers_required,
            "concurrent_transfers_required",
        )
        if type(self.concurrency_feasible) is not bool:
            raise InputValidationError(
                "concurrency_feasible", "must be a boolean"
            )


@dataclass(frozen=True)
class PDMetrics:
    scenario_name: str
    prefill_candidate_id: str
    decode_candidate_id: str
    transfer: PDTransferMetrics
    prefill_request_capacity: float
    decode_request_capacity: float
    system_request_capacity: float
    ttft_ms: float
    tpot_ms: float
    bottleneck: str
    feasible: bool
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]

    def __post_init__(self) -> None:
        for path, value in (
            ("scenario_name", self.scenario_name),
            ("prefill_candidate_id", self.prefill_candidate_id),
            ("decode_candidate_id", self.decode_candidate_id),
        ):
            if not isinstance(value, str) or not value:
                raise InputValidationError(path, "must be a non-empty string")
        if not isinstance(self.transfer, PDTransferMetrics):
            raise InputValidationError(
                "transfer", "must be a PDTransferMetrics"
            )
        if self.transfer.scenario_name != self.scenario_name:
            raise InputValidationError(
                "transfer.scenario_name", "must equal scenario_name"
            )
        for path, value in (
            ("prefill_request_capacity", self.prefill_request_capacity),
            ("decode_request_capacity", self.decode_request_capacity),
            ("system_request_capacity", self.system_request_capacity),
            ("ttft_ms", self.ttft_ms),
            ("tpot_ms", self.tpot_ms),
        ):
            _number(value, path, nonnegative=True)
        if self.bottleneck not in ("prefill", "decode", "pd_link"):
            raise InputValidationError(
                "bottleneck", "must be 'prefill', 'decode', or 'pd_link'"
            )
        if type(self.feasible) is not bool:
            raise InputValidationError("feasible", "must be a boolean")
        reasons = _string_tuple(self.reason_codes, "reason_codes")
        warnings = _string_tuple(self.warnings, "warnings")
        if len(set(reasons)) != len(reasons):
            raise InputValidationError("reason_codes", "must be unique")
        if len(set(warnings)) != len(warnings):
            raise InputValidationError("warnings", "must be unique")
        if self.feasible != (not reasons):
            raise InputValidationError(
                "feasible", "must be true exactly when reason_codes is empty"
            )
        expected_capacity = min(
            self.prefill_request_capacity,
            self.decode_request_capacity,
            self.transfer.link_request_capacity,
        )
        if self.system_request_capacity != expected_capacity:
            raise InputValidationError(
                "system_request_capacity",
                "must equal the minimum component capacity",
            )
        expected_bottleneck = min(
            (
                (self.prefill_request_capacity, 0, "prefill"),
                (self.decode_request_capacity, 1, "decode"),
                (self.transfer.link_request_capacity, 2, "pd_link"),
            )
        )[2]
        if self.bottleneck != expected_bottleneck:
            raise InputValidationError(
                "bottleneck", "must identify the minimum component capacity"
            )
        object.__setattr__(self, "reason_codes", reasons)
        object.__setattr__(self, "warnings", warnings)


def pd_payload_bytes(
    model: ModelSpec,
    precision: PrecisionSpec,
    scenario: WorkloadScenario,
) -> int:
    """Return full prompt KV plus one terminal recurrent state."""

    if not isinstance(model, ModelSpec):
        raise InputValidationError("model", "must be a ModelSpec")
    if not isinstance(precision, PrecisionSpec):
        raise InputValidationError("precision", "must be a PrecisionSpec")
    _validate_scenario(scenario)
    payload = kv_bytes_per_request(model, precision, scenario.input_length)
    payload += recurrent_state_bytes_per_request(model)
    try:
        normalized = int(payload)
    except (OverflowError, ValueError):
        raise InputValidationError("payload_bytes", "derived value must be finite") from None
    if not isfinite(float(payload)):
        raise InputValidationError("payload_bytes", "derived value must be finite")
    if normalized <= 0 or normalized != payload:
        raise InputValidationError(
            "payload_bytes", "derived value must be a positive integer"
        )
    return normalized


def _find_metric(candidate: StageCandidate, stage: str, scenario_name: str):
    from infersim.search.constraints import StageCandidate

    path = f"{stage}_candidate"
    if not isinstance(candidate, StageCandidate):
        raise InputValidationError(path, "must be a StageCandidate")
    if not candidate.feasible:
        raise InputValidationError(f"{path}.feasible", "must be true")
    matches = []
    names = set()
    for index, metric in enumerate(candidate.metrics):
        metric_path = f"{path}.metrics[{index}]"
        if metric.stage != stage:
            raise InputValidationError(
                f"{metric_path}.stage", f"must be '{stage}'"
            )
        if not isinstance(metric.scenario_name, str) or not metric.scenario_name:
            raise InputValidationError(
                f"{metric_path}.scenario_name", "must be a non-empty string"
            )
        if metric.scenario_name in names:
            raise InputValidationError(
                f"{metric_path}.scenario_name", "must be unique"
            )
        names.add(metric.scenario_name)
        if metric.scenario_name == scenario_name:
            matches.append(metric)
    if len(matches) != 1:
        raise InputValidationError(
            f"{path}.metrics",
            f"must contain exactly one metric named '{scenario_name}'",
        )
    return matches[0]


def evaluate_pd_pair(
    prefill_candidate: StageCandidate,
    decode_candidate: StageCandidate,
    pd_link: PDLinkSpec,
    kv_state_bytes: int,
    scenario: WorkloadScenario,
) -> PDMetrics:
    _validate_scenario(scenario)
    payload = _positive_integer(kv_state_bytes, "kv_state_bytes")
    bandwidth, latency_us, efficiency, max_concurrent = _validated_link_values(
        pd_link
    )
    prefill = _find_metric(prefill_candidate, "prefill", scenario.name)
    decode = _find_metric(decode_candidate, "decode", scenario.name)

    prefill_capacity = _number(
        prefill.request_capacity,
        "prefill_candidate.metric.request_capacity",
        nonnegative=True,
    )
    decode_capacity = _number(
        decode.request_capacity,
        "decode_candidate.metric.request_capacity",
        nonnegative=True,
    )
    prefill_seconds = _number(
        prefill.latency_seconds,
        "prefill_candidate.metric.latency_seconds",
        nonnegative=True,
    )
    if decode.tpot_seconds is None:
        raise InputValidationError(
            "decode_candidate.metric.tpot_seconds", "must be present"
        )
    decode_seconds = _number(
        decode.tpot_seconds,
        "decode_candidate.metric.tpot_seconds",
        nonnegative=True,
    )

    effective_bandwidth = bandwidth * 1e9 * efficiency
    if not isfinite(effective_bandwidth):
        raise InputValidationError(
            "pd_link.effective_bandwidth", "derived value must be finite"
        )
    transfer_seconds = latency_us * 1e-6 + payload / effective_bandwidth
    link_capacity = effective_bandwidth / payload
    if not isfinite(transfer_seconds):
        raise InputValidationError(
            "transfer_seconds", "derived value must be finite"
        )
    if not isfinite(link_capacity):
        raise InputValidationError(
            "link_request_capacity", "derived value must be finite"
        )
    concurrent_work = scenario.request_rate * transfer_seconds
    if not isfinite(concurrent_work):
        raise InputValidationError(
            "concurrent_transfers_required", "derived value must be finite"
        )
    required = ceil(concurrent_work)
    required = max(1, required)
    concurrency_feasible = required <= max_concurrent
    transfer = PDTransferMetrics(
        scenario_name=scenario.name,
        payload_bytes=payload,
        effective_bandwidth_bytes_per_second=effective_bandwidth,
        transfer_seconds=transfer_seconds,
        link_request_capacity=link_capacity,
        concurrent_transfers_required=required,
        concurrency_feasible=concurrency_feasible,
    )

    reasons = []
    prefix = scenario.name + ":"
    if prefill_capacity < scenario.request_rate:
        reasons.append(prefix + "PREFILL_RATE")
    if decode_capacity < scenario.request_rate:
        reasons.append(prefix + "DECODE_RATE")
    if link_capacity < scenario.request_rate:
        reasons.append(prefix + "PD_LINK_RATE")
    ttft_ms = (prefill_seconds + transfer_seconds + decode_seconds) * 1000
    tpot_ms = decode_seconds * 1000
    if ttft_ms > scenario.ttft_limit_ms:
        reasons.append(prefix + "TTFT_SLO")
    if tpot_ms > scenario.tpot_limit_ms:
        reasons.append(prefix + "TPOT_SLO")
    if not concurrency_feasible:
        reasons.append(prefix + "PD_TRANSFER_CONCURRENCY")

    components = (
        (prefill_capacity, 0, "prefill"),
        (decode_capacity, 1, "decode"),
        (link_capacity, 2, "pd_link"),
    )
    system_capacity, _, bottleneck = min(components)
    reason_codes = tuple(reasons)
    return PDMetrics(
        scenario_name=scenario.name,
        prefill_candidate_id=prefill_candidate.candidate_id,
        decode_candidate_id=decode_candidate.candidate_id,
        transfer=transfer,
        prefill_request_capacity=prefill_capacity,
        decode_request_capacity=decode_capacity,
        system_request_capacity=system_capacity,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        bottleneck=bottleneck,
        feasible=not reason_codes,
        reason_codes=reason_codes,
        warnings=tuple(
            dict.fromkeys(prefill_candidate.warnings + decode_candidate.warnings)
        ),
    )
