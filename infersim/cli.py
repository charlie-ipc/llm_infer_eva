from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

from infersim.cost import pd_payload_bytes
from infersim.errors import InputValidationError
from infersim.report import write_pd_reports, write_stage_reports
from infersim.schema import (
    HardwareSpec,
    ModelSpec,
    PDLinkSpec,
    PrecisionSpec,
    ScenarioSet,
    SearchSpace,
)
from infersim.search import pair_stage_results, run_stage_search


class CliInputError(ValueError):
    pass


_DEFAULT_SEARCH_SPACE = {
    "total_cards": (1, 2, 4, 8),
    "replicas": (1, 2),
    "attention_tp": (1, 2, 4),
    "attention_dp": (1, 2),
    "moe_tp": (1, 2, 4),
    "expert_parallel": (1, 2),
    "batch_sizes": (1, 4, 16),
}


def _json(path: Path):
    def reject_duplicate_keys(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise CliInputError(f"{path}: duplicate key {key!r}")
            value[key] = item
        return value

    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle, object_pairs_hook=reject_duplicate_keys)
    except UnicodeError:
        raise CliInputError(f"{path}: invalid UTF-8") from None
    except json.JSONDecodeError as error:
        raise CliInputError(
            f"{path}: invalid JSON at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from None
    except OSError as error:
        raise CliInputError(f"{path}: cannot read file: {error.strerror or error}") from None


def _search_space(path: Path | None) -> SearchSpace:
    return SearchSpace.from_dict(
        _DEFAULT_SEARCH_SPACE if path is None else _json(path)
    )


def _safe_io_detail(error: OSError) -> str:
    detail = str(error)
    return detail if detail.isascii() else type(error).__name__


def _pair_output_preflight(output: Path) -> None:
    if output.exists() and not output.is_dir():
        raise CliInputError(f"{output}: output must be a directory")


def _write_pair_reports(output, prefill, decode, paired) -> None:
    _pair_output_preflight(output)
    parent = output.parent
    staging = None
    backup = None
    old_moved = False
    published = False
    prefix = f".{output.name or 'infersim'}."
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(
            tempfile.mkdtemp(prefix=prefix + "tmp-", dir=parent)
        )
        write_stage_reports(staging / "prefill", prefill)
        write_stage_reports(staging / "decode", decode)
        write_pd_reports(staging / "pd", paired)

        if output.exists():
            backup = Path(
                tempfile.mkdtemp(prefix=prefix + "backup-", dir=parent)
            )
            backup.rmdir()
            os.replace(output, backup)
            old_moved = True
        os.replace(staging, output)
        staging = None
        published = True
        if backup is not None:
            shutil.rmtree(backup)
            backup = None
    except OSError as error:
        if old_moved and not published and backup is not None:
            try:
                os.replace(backup, output)
                backup = None
            except OSError as restore_error:
                raise CliInputError(
                    f"{output}: report publish and rollback failed: "
                    f"{_safe_io_detail(restore_error)}"
                ) from error
        raise CliInputError(
            f"{output}: cannot write reports: {_safe_io_detail(error)}"
        ) from None
    finally:
        if staging is not None and staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
        if backup is not None and backup.exists() and (published or not old_moved):
            shutil.rmtree(backup, ignore_errors=True)


def _run_search(args) -> int:
    model = ModelSpec.from_dict(_json(args.model))
    hardware = HardwareSpec.from_dict(_json(args.hardware))
    precision = PrecisionSpec.from_dict(_json(args.precision))
    scenarios = ScenarioSet.from_dict(_json(args.scenarios))
    result = run_stage_search(
        args.stage,
        model,
        hardware,
        precision,
        scenarios,
        _search_space(args.search_space),
    )
    try:
        write_stage_reports(args.output, result)
    except OSError as error:
        raise CliInputError(
            f"{args.output}: cannot write reports: {_safe_io_detail(error)}"
        ) from None
    selected = result.recommendation
    if selected is None:
        print(
            f"stage: {args.stage}; no feasible recommendation; output: {args.output}"
        )
        return 1
    print(
        f"stage: {args.stage}; candidate: {selected.candidate_id}; "
        f"total cards: {selected.total_cards}; capacity: "
        f"{selected.request_capacity:.6f} req/s; output: {args.output}"
    )
    return 0


def _run_pair_pd(args) -> int:
    _pair_output_preflight(args.output)
    model = ModelSpec.from_dict(_json(args.model))
    prefill_hardware = HardwareSpec.from_dict(_json(args.prefill_hardware))
    decode_hardware = HardwareSpec.from_dict(_json(args.decode_hardware))
    precision = PrecisionSpec.from_dict(_json(args.precision))
    scenarios = ScenarioSet.from_dict(_json(args.scenarios))
    pd_link = PDLinkSpec.from_dict(_json(args.pd_link))

    prefill = run_stage_search(
        "prefill",
        model,
        prefill_hardware,
        precision,
        scenarios,
        _search_space(args.prefill_search_space),
    )
    decode = run_stage_search(
        "decode",
        model,
        decode_hardware,
        precision,
        scenarios,
        _search_space(args.decode_search_space),
    )
    payloads = {
        scenario.name: pd_payload_bytes(model, precision, scenario)
        for scenario in scenarios.scenarios
    }
    paired = pair_stage_results(prefill, decode, pd_link, scenarios, payloads)
    _write_pair_reports(args.output, prefill, decode, paired)

    selected = paired.recommendation
    if selected is None:
        print(f"PD pair: no feasible recommendation; output: {args.output}")
        return 1
    print(
        f"PD pair: {selected.candidate_id}; total cards: {selected.total_cards}; "
        f"capacity: {selected.request_capacity:.6f} req/s; output: {args.output}"
    )
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="infersim",
        description="PD-aware analytical LLM deployment search",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    search = subparsers.add_parser("search", help="search one inference stage")
    search.add_argument("--model", type=Path, required=True)
    search.add_argument("--hardware", type=Path, required=True)
    search.add_argument("--precision", type=Path, required=True)
    search.add_argument("--scenarios", type=Path, required=True)
    search.add_argument("--search-space", type=Path)
    search.add_argument("--stage", choices=("prefill", "decode"), required=True)
    search.add_argument("--output", type=Path, required=True)
    search.set_defaults(handler=_run_search)

    pair = subparsers.add_parser("pair-pd", help="pair prefill and decode searches")
    pair.add_argument("--model", type=Path, required=True)
    pair.add_argument("--prefill-hardware", type=Path, required=True)
    pair.add_argument("--decode-hardware", type=Path, required=True)
    pair.add_argument("--pd-link", type=Path, required=True)
    pair.add_argument("--precision", type=Path, required=True)
    pair.add_argument("--scenarios", type=Path, required=True)
    pair.add_argument("--prefill-search-space", type=Path)
    pair.add_argument("--decode-search-space", type=Path)
    pair.add_argument("--output", type=Path, required=True)
    pair.set_defaults(handler=_run_pair_pd)
    return parser


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    try:
        return args.handler(args)
    except (CliInputError, InputValidationError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
