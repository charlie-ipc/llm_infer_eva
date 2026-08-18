from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

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
_PAIR_OUTPUT_MARKER = ".infersim-pd-output.json"
_PAIR_OUTPUT_MARKER_BYTES = (
    b'{\n'
    b'  "format": "infersim-pd-output",\n'
    b'  "version": 1\n'
    b'}\n'
)
_STAGE_REPORT_NAMES = frozenset(
    {
        "all_candidates.csv",
        "feasible_candidates.csv",
        "pareto_frontier.csv",
        "recommendation.json",
        "summary.txt",
    }
)
_PD_REPORT_NAMES = frozenset(
    {
        "all_pairs.csv",
        "feasible_pairs.csv",
        "pareto_frontier.csv",
        "recommendation.json",
        "summary.txt",
    }
)


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
    try:
        if output.is_symlink():
            raise CliInputError(f"{output}: output must not be a symlink")
        if not output.exists():
            return
        if not output.is_dir():
            raise CliInputError(f"{output}: output must be a directory")
        entries = tuple(output.iterdir())
        if not entries:
            return
        expected_root = {
            _PAIR_OUTPUT_MARKER,
            "prefill",
            "decode",
            "pd",
        }
        if {entry.name for entry in entries} != expected_root:
            raise CliInputError(
                f"{output}: not an owned InferSim PD output; "
                "refusing to replace a nonempty directory"
            )
        marker = output / _PAIR_OUTPUT_MARKER
        if (
            marker.is_symlink()
            or not marker.is_file()
            or marker.read_bytes() != _PAIR_OUTPUT_MARKER_BYTES
        ):
            raise CliInputError(
                f"{output}: not an owned InferSim PD output; invalid marker"
            )
        for directory, expected_files in (
            ("prefill", _STAGE_REPORT_NAMES),
            ("decode", _STAGE_REPORT_NAMES),
            ("pd", _PD_REPORT_NAMES),
        ):
            report_root = output / directory
            if report_root.is_symlink() or not report_root.is_dir():
                raise CliInputError(
                    f"{output}: not an owned InferSim PD output; "
                    f"{directory} must be a regular directory"
                )
            reports = tuple(report_root.iterdir())
            if {report.name for report in reports} != expected_files or any(
                report.is_symlink() or not report.is_file()
                for report in reports
            ):
                raise CliInputError(
                    f"{output}: not an owned InferSim PD output; "
                    f"unexpected {directory} report files"
                )
    except CliInputError:
        raise
    except OSError as error:
        raise CliInputError(
            f"{output}: cannot inspect output: {_safe_io_detail(error)}"
        ) from None


def _generated_child(path: Path, parent: Path, label: str) -> Path:
    path = path.absolute()
    if not path.is_absolute() or path.parent != parent:
        raise RuntimeError(
            f"refusing to clean unsafe {label} path outside {parent}"
        )
    return path


def _remove_generated_tree(path: Path, parent: Path, label: str) -> None:
    path = _generated_child(path, parent, label)
    if not path.exists():
        return
    if path.is_symlink() or not path.is_dir():
        raise RuntimeError(f"refusing to clean invalid {label} path {path}")
    shutil.rmtree(path)


def _reserve_generated_path(parent: Path, prefix: str, label: str) -> Path:
    path = Path(tempfile.mkdtemp(prefix=prefix, dir=parent)).absolute()
    path = _generated_child(path, parent, label)
    path.rmdir()
    return path


def _restore_backup(backup: Path, output: Path, parent: Path) -> None:
    backup = _generated_child(backup, parent, "backup")
    if output.exists() or output.is_symlink():
        raise RuntimeError(f"refusing to restore backup over {output}")
    for attempt in range(3):
        try:
            os.rename(backup, output)
            return
        except PermissionError:
            if attempt == 2:
                raise
            # Windows can briefly retain handles after a populated directory
            # rename. The backup remains the sole old copy while retrying.
            time.sleep(0.05)


def _write_pair_reports(output, prefill, decode, paired) -> None:
    _pair_output_preflight(output)
    output = Path(output).absolute()
    parent = output.parent
    staging = None
    backup = None
    old_moved = False
    published = False
    prefix = f".{output.name or 'infersim'}."
    try:
        parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(prefix=prefix + "tmp-", dir=parent)).absolute()
        _generated_child(staging, parent, "staging")
        write_stage_reports(staging / "prefill", prefill)
        write_stage_reports(staging / "decode", decode)
        write_pd_reports(staging / "pd", paired)
        (staging / _PAIR_OUTPUT_MARKER).write_bytes(_PAIR_OUTPUT_MARKER_BYTES)

        if output.exists():
            backup = _reserve_generated_path(
                parent,
                prefix + "backup-",
                "backup",
            )
            os.replace(output, backup)
            old_moved = True
        os.replace(staging, output)
        published = True
        if backup is not None:
            _remove_generated_tree(backup, parent, "backup")
            backup = None
        staging = None
    except BaseException as error:
        try:
            if old_moved and backup is not None and backup.exists():
                if published and output.exists():
                    if staging is None:
                        staging = _reserve_generated_path(
                            parent,
                            prefix + "rollback-",
                            "rollback staging",
                        )
                    os.replace(output, staging)
                    published = False
                _restore_backup(backup, output, parent)
                backup = None
                old_moved = False
            if staging is not None:
                _remove_generated_tree(staging, parent, "staging")
                staging = None
        except BaseException as restore_error:
            raise restore_error from error
        if isinstance(error, OSError):
            raise CliInputError(
                f"{output}: cannot write reports: {_safe_io_detail(error)}"
            ) from None
        raise
    finally:
        if staging is not None:
            try:
                _remove_generated_tree(staging, parent, "staging")
            except OSError:
                pass
        if backup is not None and not old_moved:
            try:
                _remove_generated_tree(backup, parent, "backup")
            except OSError:
                pass


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
