from __future__ import annotations

import csv
from dataclasses import fields
import json
from math import isfinite
from pathlib import Path

from infersim.errors import InputValidationError
from infersim.search.constraints import StageCandidate
from infersim.search.runner import SearchResult, _rank_rejections


CSV_FIELDS = (
    "candidate_id",
    "stage",
    "feasible",
    "reason_codes",
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
    "worst_memory_required_bytes",
    "worst_memory_margin_bytes",
    "component_bottleneck",
)


def _finite(value: int | float, path: str) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError, OverflowError):
        raise InputValidationError(path, "must be finite") from None
    if not isfinite(normalized):
        raise InputValidationError(path, "must be finite")
    return normalized


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


def _candidate_row(stage: str, candidate: StageCandidate) -> dict[str, str | int]:
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
    return {
        "candidate_id": candidate.candidate_id,
        "stage": stage,
        "feasible": "true" if candidate.feasible else "false",
        "reason_codes": ";".join(candidate.reason_codes),
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


def _write_csv(path: Path, stage: str, candidates) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for candidate in sorted(candidates, key=lambda item: item.candidate_id):
            writer.writerow(_candidate_row(stage, candidate))


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
            f"{reason}: {count}" for reason, count in rejection_ranking[:3]
        )
    else:
        lines.append("none")
    return "\n".join(lines) + "\n"


def write_stage_reports(output_dir: Path, result: SearchResult) -> None:
    if not isinstance(output_dir, Path):
        raise InputValidationError("output_dir", "must be a Path")
    if not isinstance(result, SearchResult):
        raise InputValidationError("result", "must be a SearchResult")
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "all_candidates.csv", result.stage, result.candidates)
    _write_csv(
        output_dir / "feasible_candidates.csv",
        result.stage,
        result.feasible_candidates,
    )
    _write_csv(
        output_dir / "pareto_frontier.csv", result.stage, result.pareto_frontier
    )

    rejection_ranking = _rank_rejections(result.candidates)
    payload = {
        "dominant_rejection": result.dominant_rejection,
        "recommendation": None
        if result.recommendation is None
        else _candidate_payload(result.stage, result.recommendation),
        "stage": result.stage,
        "top_rejection_reasons": [
            {"count": count, "reason": reason}
            for reason, count in rejection_ranking[:3]
        ],
    }
    with (output_dir / "recommendation.json").open(
        "w", encoding="utf-8", newline="\n"
    ) as handle:
        json.dump(payload, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    (output_dir / "summary.txt").write_text(
        _summary(result, rejection_ranking), encoding="utf-8", newline="\n"
    )
