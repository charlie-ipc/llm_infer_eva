from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

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


def _json(path: Path):
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except json.JSONDecodeError as error:
        raise CliInputError(
            f"{path}: invalid JSON at line {error.lineno}, column {error.colno}: "
            f"{error.msg}"
        ) from None
    except OSError as error:
        raise CliInputError(f"{path}: cannot read file: {error.strerror or error}") from None


def _search_space(path: Path | None) -> SearchSpace:
    return SearchSpace.from_dict(None if path is None else _json(path))


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
    write_stage_reports(args.output, result)
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
    write_stage_reports(args.output / "prefill", prefill)
    write_stage_reports(args.output / "decode", decode)
    payloads = {
        scenario.name: pd_payload_bytes(model, precision, scenario)
        for scenario in scenarios.scenarios
    }
    paired = pair_stage_results(prefill, decode, pd_link, scenarios, payloads)
    write_pd_reports(args.output / "pd", paired)

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
