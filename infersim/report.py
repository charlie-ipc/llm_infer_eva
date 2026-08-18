from __future__ import annotations

import csv
from collections.abc import Mapping
from dataclasses import fields, is_dataclass
import io
import json
from math import isfinite
from numbers import Real
import os
from pathlib import Path
import tempfile

from infersim.errors import InputValidationError
from infersim.search.constraints import StageCandidate
from infersim.search.pair import PDCandidate, PDSearchResult
from infersim.search.runner import SearchResult, _base_reason, _rank_rejections


CSV_FIELDS = (
    "candidate_id",
    "stage",
    "feasible",
    "reason_codes",
    "reason_details",
    "warning_codes",
    "total_cards",
    "hourly_cost",
    "request_capacity",
    "request_capacity_per_card",
    "ttft_ms",
    "tpot_ms",
    "replicas",
    "attention_tp",
    "attention_dp",
    "moe_tp",
    "expert_parallel",
    "batch_size",
    "scenario_count",
    "worst_latency_seconds",
    "worst_gemm_seconds",
    "worst_vector_seconds",
    "worst_tp_seconds",
    "worst_ep_seconds",
    "prompt_token_capacity",
    "output_token_capacity",
    "worst_memory_required_bytes",
    "worst_memory_margin_bytes",
    "component_bottleneck",
)

PD_CSV_FIELDS = (
    "candidate_id",
    "prefill_candidate_id",
    "decode_candidate_id",
    "prefill_plan",
    "decode_plan",
    "feasible",
    "reason_codes",
    "warning_codes",
    "total_cards",
    "hourly_cost",
    "request_capacity",
    "request_capacity_per_card",
    "ttft_ms",
    "tpot_ms",
    "bottleneck",
    "scenario_count",
    "transfer_summary",
)

_PD_ASSUMPTIONS = (
    "Prefill and decode stage candidates are evaluated independently before pairing.",
    "TTFT includes prefill latency and one KV/state transfer to decode.",
    "The PD link is modeled analytically from payload, effective bandwidth, latency, and concurrency.",
)


def _finite(value: int | float, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise InputValidationError(path, "must be a number")
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    return normalized


def _nonnegative_integer(value, path: str, *, optional: bool = False):
    if optional and value is None:
        return None
    if type(value) is not int:
        raise InputValidationError(path, "must be an integer")
    if value < 0:
        raise InputValidationError(path, "must be nonnegative")
    return value


def _seconds(value: int | float, path: str) -> str:
    return f"{_finite(value, path):.12f}"


def _six(value: int | float | None, path: str) -> str:
    if value is None:
        return ""
    return f"{_finite(value, path):.6f}"


def _bytes(value: int | float, path: str) -> str:
    return str(int(round(_finite(value, path))))


def _rounded(value: int | float | None, digits: int, path: str):
    if value is None:
        return None
    return round(_finite(value, path), digits)


def _component_bottleneck(candidate: StageCandidate) -> str | None:
    component_totals = {}
    for metric_index, metric in enumerate(candidate.metrics):
        for name, seconds in metric.component_seconds.items():
            value = _finite(
                seconds,
                f"metrics[{metric_index}].component_seconds.{name}",
            )
            component_totals[name] = max(component_totals.get(name, 0.0), value)
    if not component_totals:
        return None
    return min(component_totals, key=lambda name: (-component_totals[name], name))


def _candidate_row(
    stage: str,
    candidate: StageCandidate,
    reason_details: tuple[str, ...] = (),
) -> dict[str, str | int]:
    plan = candidate.plan
    metrics = candidate.metrics
    worst_latency = max(
        (metric.latency_seconds for metric in metrics), default=None
    )
    worst_required = max(
        (metric.memory.total_required_bytes for metric in metrics), default=None
    )
    worst_margin = min(
        (metric.memory.capacity_margin_bytes for metric in metrics), default=None
    )
    component_values = {
        name: max((getattr(metric, name) for metric in metrics), default=None)
        for name in (
            "gemm_seconds",
            "vector_seconds",
            "tp_seconds",
            "ep_seconds",
        )
    }
    prompt_capacity = min(
        (
            metric.prompt_token_capacity
            for metric in metrics
            if metric.prompt_token_capacity is not None
        ),
        default=None,
    )
    output_capacity = min(
        (
            metric.output_token_capacity
            for metric in metrics
            if metric.output_token_capacity is not None
        ),
        default=None,
    )
    return {
        "candidate_id": candidate.candidate_id,
        "stage": stage,
        "feasible": "true" if candidate.feasible else "false",
        "reason_codes": ";".join(candidate.reason_codes),
        "reason_details": ";".join(reason_details),
        "warning_codes": ";".join(candidate.warnings),
        "total_cards": candidate.total_cards,
        "hourly_cost": _six(candidate.hourly_cost, "hourly_cost"),
        "request_capacity": _six(candidate.request_capacity, "request_capacity"),
        "request_capacity_per_card": _six(
            candidate.request_capacity_per_card, "request_capacity_per_card"
        ),
        "ttft_ms": _six(candidate.ttft_ms, "ttft_ms"),
        "tpot_ms": _six(candidate.tpot_ms, "tpot_ms"),
        "replicas": plan.replicas,
        "attention_tp": plan.attention_tp,
        "attention_dp": plan.attention_dp,
        "moe_tp": plan.moe_tp,
        "expert_parallel": plan.expert_parallel,
        "batch_size": plan.batch_size,
        "scenario_count": len(metrics),
        "worst_latency_seconds": ""
        if worst_latency is None
        else _seconds(worst_latency, "worst_latency_seconds"),
        "worst_gemm_seconds": ""
        if component_values["gemm_seconds"] is None
        else _seconds(component_values["gemm_seconds"], "worst_gemm_seconds"),
        "worst_vector_seconds": ""
        if component_values["vector_seconds"] is None
        else _seconds(
            component_values["vector_seconds"], "worst_vector_seconds"
        ),
        "worst_tp_seconds": ""
        if component_values["tp_seconds"] is None
        else _seconds(component_values["tp_seconds"], "worst_tp_seconds"),
        "worst_ep_seconds": ""
        if component_values["ep_seconds"] is None
        else _seconds(component_values["ep_seconds"], "worst_ep_seconds"),
        "prompt_token_capacity": _six(
            prompt_capacity, "prompt_token_capacity"
        ),
        "output_token_capacity": _six(
            output_capacity, "output_token_capacity"
        ),
        "worst_memory_required_bytes": ""
        if worst_required is None
        else _bytes(worst_required, "worst_memory_required_bytes"),
        "worst_memory_margin_bytes": ""
        if worst_margin is None
        else _bytes(worst_margin, "worst_memory_margin_bytes"),
        "component_bottleneck": _component_bottleneck(candidate) or "",
    }


def _memory_payload(memory, path: str) -> dict:
    payload = {"stage": memory.stage, "feasible": memory.feasible}
    for field in fields(memory):
        if field.name in ("stage", "feasible"):
            continue
        payload[field.name] = int(
            round(_finite(getattr(memory, field.name), f"{path}.{field.name}"))
        )
    return payload


def _validate_metric(metric, path: str) -> None:
    for field_name in (
        "latency_seconds",
        "average_context_length",
        "gemm_seconds",
        "vector_seconds",
        "tp_seconds",
        "ep_seconds",
        "request_capacity",
    ):
        _finite(getattr(metric, field_name), f"{path}.{field_name}")
    for field_name in (
        "tpot_seconds",
        "prompt_token_capacity",
        "output_token_capacity",
    ):
        value = getattr(metric, field_name)
        if value is not None:
            _finite(value, f"{path}.{field_name}")
    for field_name in (
        "useful_gemm_ops",
        "aligned_gemm_ops",
        "useful_vector_ops",
        "aligned_vector_ops",
    ):
        _nonnegative_integer(
            getattr(metric, field_name), f"{path}.{field_name}"
        )
    for field_name in ("max_supported_batch", "max_supported_concurrency"):
        _nonnegative_integer(
            getattr(metric, field_name),
            f"{path}.{field_name}",
            optional=True,
        )
    for component_name, seconds in metric.component_seconds.items():
        if not isinstance(component_name, str) or not component_name:
            raise InputValidationError(
                f"{path}.component_seconds", "keys must be non-empty strings"
            )
        _finite(seconds, f"{path}.component_seconds.{component_name}")
    if type(metric.memory.feasible) is not bool:
        raise InputValidationError(
            f"{path}.memory.feasible", "must be a boolean"
        )
    for field in fields(metric.memory):
        if field.name in ("stage", "feasible"):
            continue
        value = _finite(
            getattr(metric.memory, field.name),
            f"{path}.memory.{field.name}",
        )
        if field.name != "capacity_margin_bytes" and value < 0:
            raise InputValidationError(
                f"{path}.memory.{field.name}", "must be nonnegative"
            )


def _validate_result_numbers(result: SearchResult) -> None:
    for candidate_index, candidate in enumerate(result.candidates):
        path = f"candidates[{candidate_index}]"
        for field_name in (
            "replicas",
            "attention_tp",
            "attention_dp",
            "moe_tp",
            "expert_parallel",
            "batch_size",
        ):
            value = _nonnegative_integer(
                getattr(candidate.plan, field_name), f"{path}.plan.{field_name}"
            )
            if value == 0:
                raise InputValidationError(
                    f"{path}.plan.{field_name}", "must be positive"
                )
        _nonnegative_integer(candidate.total_cards, f"{path}.total_cards")
        for field_name in (
            "request_capacity",
            "request_capacity_per_card",
            "ttft_ms",
            "tpot_ms",
            "hourly_cost",
        ):
            value = getattr(candidate, field_name)
            if value is not None:
                _finite(value, f"{path}.{field_name}")
        for metric_index, metric in enumerate(candidate.metrics):
            _validate_metric(metric, f"{path}.metrics[{metric_index}]")


def _metric_payload(metric, index: int) -> dict:
    path = f"metrics[{index}]"
    return {
        "aligned_gemm_ops": metric.aligned_gemm_ops,
        "aligned_vector_ops": metric.aligned_vector_ops,
        "average_context_length": _rounded(
            metric.average_context_length, 6, f"{path}.average_context_length"
        ),
        "component_seconds": {
            name: _rounded(value, 12, f"{path}.component_seconds.{name}")
            for name, value in sorted(metric.component_seconds.items())
        },
        "ep_seconds": _rounded(metric.ep_seconds, 12, f"{path}.ep_seconds"),
        "gemm_seconds": _rounded(
            metric.gemm_seconds, 12, f"{path}.gemm_seconds"
        ),
        "latency_seconds": _rounded(
            metric.latency_seconds, 12, f"{path}.latency_seconds"
        ),
        "max_supported_batch": metric.max_supported_batch,
        "max_supported_concurrency": metric.max_supported_concurrency,
        "memory": _memory_payload(metric.memory, f"{path}.memory"),
        "output_token_capacity": _rounded(
            metric.output_token_capacity, 6, f"{path}.output_token_capacity"
        ),
        "prompt_token_capacity": _rounded(
            metric.prompt_token_capacity, 6, f"{path}.prompt_token_capacity"
        ),
        "request_capacity": _rounded(
            metric.request_capacity, 6, f"{path}.request_capacity"
        ),
        "scenario_name": metric.scenario_name,
        "stage": metric.stage,
        "tpot_seconds": _rounded(
            metric.tpot_seconds, 12, f"{path}.tpot_seconds"
        ),
        "tp_seconds": _rounded(metric.tp_seconds, 12, f"{path}.tp_seconds"),
        "useful_gemm_ops": metric.useful_gemm_ops,
        "useful_vector_ops": metric.useful_vector_ops,
        "vector_seconds": _rounded(
            metric.vector_seconds, 12, f"{path}.vector_seconds"
        ),
    }


def _candidate_payload(stage: str, candidate: StageCandidate) -> dict:
    plan = candidate.plan
    return {
        "bottleneck": _component_bottleneck(candidate),
        "candidate_id": candidate.candidate_id,
        "feasible": candidate.feasible,
        "hourly_cost": _rounded(candidate.hourly_cost, 6, "hourly_cost"),
        "plan": {
            "attention_dp": plan.attention_dp,
            "attention_tp": plan.attention_tp,
            "batch_size": plan.batch_size,
            "expert_parallel": plan.expert_parallel,
            "moe_tp": plan.moe_tp,
            "replicas": plan.replicas,
        },
        "reason_codes": list(candidate.reason_codes),
        "scenarios": [
            _metric_payload(metric, index)
            for index, metric in enumerate(candidate.metrics)
        ],
        "stage": stage,
        "summary": {
            "request_capacity": _rounded(
                candidate.request_capacity, 6, "request_capacity"
            ),
            "request_capacity_per_card": _rounded(
                candidate.request_capacity_per_card,
                6,
                "request_capacity_per_card",
            ),
            "total_cards": candidate.total_cards,
            "tpot_ms": _rounded(candidate.tpot_ms, 6, "tpot_ms"),
            "ttft_ms": _rounded(candidate.ttft_ms, 6, "ttft_ms"),
        },
        "warning_codes": list(candidate.warnings),
    }


def _csv_content(stage: str, candidates, diagnostic_details) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
    writer.writeheader()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        writer.writerow(
            _candidate_row(
                stage,
                candidate,
                diagnostic_details.get(candidate.candidate_id, ()),
            )
        )
    return handle.getvalue()


def _normalized_value(value, path: str):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _normalized_value(
                getattr(value, field.name), f"{path}.{field.name}"
            )
            for field in fields(value)
        }
    if isinstance(value, Mapping):
        for key in value:
            if not isinstance(key, str) or not key:
                raise InputValidationError(path, "keys must be non-empty strings")
        return {
            key: _normalized_value(item, f"{path}.{key}")
            for key, item in sorted(value.items())
        }
    if isinstance(value, tuple):
        return [
            _normalized_value(item, f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, float):
        return _finite(value, path)
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return value
    if type(value) is int:
        return value
    raise InputValidationError(path, "contains an unsupported value")
    return value


def _normalized_input_summary(result: SearchResult):
    context = result.context
    if context is None:
        return None
    return {
        "hardware": _normalized_value(context.hardware, "context.hardware"),
        "model": _normalized_value(context.model, "context.model"),
        "precision": _normalized_value(context.precision, "context.precision"),
        "scenario_set": _normalized_value(
            context.scenario_set, "context.scenario_set"
        ),
        "search_space": _normalized_value(
            context.search_space, "context.search_space"
        ),
    }


def _diagnostic_details(result: SearchResult):
    values = {}
    for diagnostic in result.diagnostics:
        values.setdefault(diagnostic.candidate_id, []).append(diagnostic.detail)
    return {
        candidate_id: tuple(details)
        for candidate_id, details in sorted(values.items())
    }


def _rejection_details(result: SearchResult, base_reason: str) -> list[str]:
    return sorted(
        {
            diagnostic.detail
            for diagnostic in result.diagnostics
            if _base_reason(diagnostic.reason_code) == base_reason
        }
    )


def _summary(result: SearchResult, rejection_ranking) -> str:
    selected = result.recommendation
    lines = [f"Stage: {result.stage}"]
    if selected is None:
        lines.extend(
            (
                "Selected plan: none",
                "Total cards: N/A",
                "SLO status: no feasible plan",
                "Component bottleneck: none",
                "No feasible plan satisfied the requested constraints.",
            )
        )
    else:
        plan = selected.plan
        lines.extend(
            (
                "Selected plan: "
                f"replicas={plan.replicas}, attention_tp={plan.attention_tp}, "
                f"attention_dp={plan.attention_dp}, moe_tp={plan.moe_tp}, "
                f"expert_parallel={plan.expert_parallel}, batch_size={plan.batch_size}",
                f"Total cards: {selected.total_cards}",
                "SLO status: feasible",
                f"Component bottleneck: {_component_bottleneck(selected) or 'none'}",
            )
        )
    lines.append("Top rejection reasons:")
    if rejection_ranking:
        lines.extend(
            f"{reason}: {count} | "
            + "; ".join(_rejection_details(result, reason))
            for reason, count in rejection_ranking[:3]
        )
    else:
        lines.append("none")
    return "\n".join(lines) + "\n"


def _report_contents(result: SearchResult) -> dict[str, str]:
    _validate_result_numbers(result)
    diagnostic_details = _diagnostic_details(result)
    rejection_ranking = _rank_rejections(result.candidates)
    payload = {
        "assumptions": []
        if result.context is None
        else list(result.context.assumptions),
        "dominant_rejection": result.dominant_rejection,
        "normalized_input_summary": _normalized_input_summary(result),
        "recommendation": None
        if result.recommendation is None
        else _candidate_payload(result.stage, result.recommendation),
        "stage": result.stage,
        "top_rejection_reasons": [
            {
                "count": count,
                "details": _rejection_details(result, reason),
                "reason": reason,
            }
            for reason, count in rejection_ranking[:3]
        ],
    }
    json_content = json.dumps(
        payload,
        sort_keys=True,
        indent=2,
        allow_nan=False,
    ) + "\n"
    return {
        "all_candidates.csv": _csv_content(
            result.stage, result.candidates, diagnostic_details
        ),
        "feasible_candidates.csv": _csv_content(
            result.stage, result.feasible_candidates, diagnostic_details
        ),
        "pareto_frontier.csv": _csv_content(
            result.stage, result.pareto_frontier, diagnostic_details
        ),
        "recommendation.json": json_content,
        "summary.txt": _summary(result, rejection_ranking),
    }


def _plan_summary(candidate: StageCandidate) -> str:
    plan = candidate.plan
    return (
        f"replicas={plan.replicas},attention_tp={plan.attention_tp},"
        f"attention_dp={plan.attention_dp},moe_tp={plan.moe_tp},"
        f"expert_parallel={plan.expert_parallel},batch_size={plan.batch_size}"
    )


def _pd_metric_payload(metric, index: int) -> dict:
    path = f"metrics[{index}]"
    transfer = metric.transfer
    return {
        "bottleneck": metric.bottleneck,
        "decode_candidate_id": metric.decode_candidate_id,
        "decode_request_capacity": _rounded(
            metric.decode_request_capacity,
            6,
            f"{path}.decode_request_capacity",
        ),
        "feasible": metric.feasible,
        "payload_bytes": _rounded(
            transfer.payload_bytes, 6, f"{path}.transfer.payload_bytes"
        ),
        "prefill_candidate_id": metric.prefill_candidate_id,
        "prefill_request_capacity": _rounded(
            metric.prefill_request_capacity,
            6,
            f"{path}.prefill_request_capacity",
        ),
        "reason_codes": list(metric.reason_codes),
        "scenario_name": metric.scenario_name,
        "system_request_capacity": _rounded(
            metric.system_request_capacity,
            6,
            f"{path}.system_request_capacity",
        ),
        "tpot_ms": _rounded(metric.tpot_ms, 6, f"{path}.tpot_ms"),
        "transfer": {
            "concurrency_feasible": transfer.concurrency_feasible,
            "concurrent_transfers_required": transfer.concurrent_transfers_required,
            "effective_bandwidth_bytes_per_second": _rounded(
                transfer.effective_bandwidth_bytes_per_second,
                6,
                f"{path}.transfer.effective_bandwidth_bytes_per_second",
            ),
            "link_request_capacity": _rounded(
                transfer.link_request_capacity,
                6,
                f"{path}.transfer.link_request_capacity",
            ),
            "payload_bytes": _rounded(
                transfer.payload_bytes,
                6,
                f"{path}.transfer.payload_bytes",
            ),
            "scenario_name": transfer.scenario_name,
            "transfer_seconds": _rounded(
                transfer.transfer_seconds,
                12,
                f"{path}.transfer.transfer_seconds",
            ),
        },
        "ttft_ms": _rounded(metric.ttft_ms, 6, f"{path}.ttft_ms"),
        "warning_codes": list(metric.warnings),
    }


def _pd_candidate_payload(candidate: PDCandidate) -> dict:
    metrics = tuple(sorted(candidate.metrics, key=lambda item: item.scenario_name))
    return {
        "candidate_id": candidate.candidate_id,
        "decode_candidate": _candidate_payload(
            "decode", candidate.decode_candidate
        ),
        "decode_candidate_id": candidate.decode_candidate_id,
        "feasible": candidate.feasible,
        "hourly_cost": _rounded(candidate.hourly_cost, 6, "hourly_cost"),
        "prefill_candidate": _candidate_payload(
            "prefill", candidate.prefill_candidate
        ),
        "prefill_candidate_id": candidate.prefill_candidate_id,
        "reason_codes": list(candidate.reason_codes),
        "request_capacity": _rounded(
            candidate.request_capacity, 6, "request_capacity"
        ),
        "request_capacity_per_card": _rounded(
            candidate.request_capacity_per_card,
            6,
            "request_capacity_per_card",
        ),
        "scenarios": [
            _pd_metric_payload(metric, index)
            for index, metric in enumerate(metrics)
        ],
        "total_cards": candidate.total_cards,
        "tpot_ms": _rounded(candidate.tpot_ms, 6, "tpot_ms"),
        "ttft_ms": _rounded(candidate.ttft_ms, 6, "ttft_ms"),
        "warning_codes": list(candidate.warnings),
    }


def _pd_transfer_summary(candidate: PDCandidate) -> str:
    parts = []
    for metric in sorted(candidate.metrics, key=lambda item: item.scenario_name):
        transfer = metric.transfer
        parts.append(
            f"{metric.scenario_name}:payload_bytes={transfer.payload_bytes:.6f},"
            f"effective_bw_Bps={transfer.effective_bandwidth_bytes_per_second:.6f},"
            f"transfer_seconds={transfer.transfer_seconds:.12f},"
            f"concurrency={transfer.concurrent_transfers_required},"
            f"feasible={'true' if transfer.concurrency_feasible else 'false'}"
        )
    return ";".join(parts)


def _pd_candidate_row(candidate: PDCandidate) -> dict:
    bottlenecks = sorted({metric.bottleneck for metric in candidate.metrics})
    return {
        "candidate_id": candidate.candidate_id,
        "prefill_candidate_id": candidate.prefill_candidate_id,
        "decode_candidate_id": candidate.decode_candidate_id,
        "prefill_plan": _plan_summary(candidate.prefill_candidate),
        "decode_plan": _plan_summary(candidate.decode_candidate),
        "feasible": "true" if candidate.feasible else "false",
        "reason_codes": ";".join(candidate.reason_codes),
        "warning_codes": ";".join(candidate.warnings),
        "total_cards": candidate.total_cards,
        "hourly_cost": _six(candidate.hourly_cost, "hourly_cost"),
        "request_capacity": _six(
            candidate.request_capacity, "request_capacity"
        ),
        "request_capacity_per_card": _six(
            candidate.request_capacity_per_card, "request_capacity_per_card"
        ),
        "ttft_ms": _six(candidate.ttft_ms, "ttft_ms"),
        "tpot_ms": _six(candidate.tpot_ms, "tpot_ms"),
        "bottleneck": ";".join(bottlenecks),
        "scenario_count": len(candidate.metrics),
        "transfer_summary": _pd_transfer_summary(candidate),
    }


def _pd_csv_content(candidates) -> str:
    handle = io.StringIO(newline="")
    writer = csv.DictWriter(handle, fieldnames=PD_CSV_FIELDS)
    writer.writeheader()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        writer.writerow(_pd_candidate_row(candidate))
    return handle.getvalue()


def _pd_summary(result: PDSearchResult) -> str:
    selected = result.recommendation
    lines = ["System: prefill/decode disaggregated"]
    if selected is None:
        lines.extend(
            (
                "Selected pair: none",
                "Total cards: N/A",
                "SLO status: no feasible pair",
                f"Dominant rejection: {result.dominant_rejection or 'none'}",
                "No feasible PD pair satisfied the requested constraints.",
            )
        )
    else:
        bottlenecks = sorted({metric.bottleneck for metric in selected.metrics})
        lines.extend(
            (
                f"Selected pair: {selected.candidate_id}",
                f"Prefill candidate: {selected.prefill_candidate_id}",
                f"Decode candidate: {selected.decode_candidate_id}",
                f"Total cards: {selected.total_cards}",
                f"Request capacity: {selected.request_capacity:.6f} req/s",
                f"TTFT: {selected.ttft_ms:.6f} ms",
                f"TPOT: {selected.tpot_ms:.6f} ms",
                f"Bottleneck: {';'.join(bottlenecks)}",
                "SLO status: feasible",
            )
        )
    return "\n".join(lines) + "\n"


def _pd_report_contents(result: PDSearchResult) -> dict[str, str]:
    payload = {
        "assumptions": list(_PD_ASSUMPTIONS),
        "dominant_rejection": result.dominant_rejection,
        "normalized_input_summary": {
            "pd_link": _normalized_value(result.pd_link, "pd_link"),
            "scenario_set": _normalized_value(
                result.scenario_set, "scenario_set"
            ),
        },
        "recommendation": None
        if result.recommendation is None
        else _pd_candidate_payload(result.recommendation),
    }
    return {
        "all_pairs.csv": _pd_csv_content(result.candidates),
        "feasible_pairs.csv": _pd_csv_content(result.feasible_candidates),
        "pareto_frontier.csv": _pd_csv_content(result.pareto_frontier),
        "recommendation.json": json.dumps(
            payload, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        "summary.txt": _pd_summary(result),
    }


def _temporary_file(output_dir: Path, name: str, content: str | bytes) -> Path:
    descriptor, raw_path = tempfile.mkstemp(
        dir=output_dir,
        prefix=f".{name}.",
        suffix=".tmp",
    )
    path = Path(raw_path)
    try:
        if isinstance(content, bytes):
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(content)
        else:
            with os.fdopen(
                descriptor, "w", encoding="utf-8", newline=""
            ) as handle:
                handle.write(content)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _replace_reports(output_dir: Path, contents: Mapping[str, str]) -> None:
    originals = {
        name: (output_dir / name).read_bytes()
        if (output_dir / name).exists()
        else None
        for name in contents
    }
    temporary = {}
    replaced = []
    try:
        for name, content in contents.items():
            temporary[name] = _temporary_file(output_dir, name, content)
        for name in contents:
            os.replace(temporary[name], output_dir / name)
            replaced.append(name)
        temporary.clear()
    except BaseException:
        for name in reversed(replaced):
            target = output_dir / name
            original = originals[name]
            if original is None:
                target.unlink(missing_ok=True)
            else:
                restore = _temporary_file(output_dir, name, original)
                try:
                    os.replace(restore, target)
                finally:
                    restore.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def write_stage_reports(output_dir: Path, result: SearchResult) -> None:
    if not isinstance(output_dir, Path):
        raise InputValidationError("output_dir", "must be a Path")
    if not isinstance(result, SearchResult):
        raise InputValidationError("result", "must be a SearchResult")
    contents = _report_contents(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    _replace_reports(output_dir, contents)


def write_pd_reports(output_dir: Path, result: PDSearchResult) -> None:
    if not isinstance(output_dir, Path):
        raise InputValidationError("output_dir", "must be a Path")
    if not isinstance(result, PDSearchResult):
        raise InputValidationError("result", "must be a PDSearchResult")
    contents = _pd_report_contents(result)
    output_dir.mkdir(parents=True, exist_ok=True)
    _replace_reports(output_dir, contents)
