import csv
from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import infersim.cli as cli


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples" / "search"
MODEL = ROOT / "hf_configs" / "qwen3-8B_config.json"
STAGE_REPORTS = {
    "all_candidates.csv",
    "feasible_candidates.csv",
    "pareto_frontier.csv",
    "recommendation.json",
    "summary.txt",
}
PD_REPORTS = {
    "all_pairs.csv",
    "feasible_pairs.csv",
    "pareto_frontier.csv",
    "recommendation.json",
    "summary.txt",
}
PAIR_MARKER = ".infersim-pd-output.json"
PAIR_MARKER_BYTES = (
    b'{\n'
    b'  "format": "infersim-pd-output",\n'
    b'  "version": 1\n'
    b'}\n'
)


def hardware_fixture():
    return {
        "name": "CLI Test NPU",
        "memory_capacity_gb": 128,
        "memory_bandwidth_gbps": 4000,
        "cards_per_node": 8,
        "compute_tflops": {
            "gemm": {"w4a4": 1200, "w4a8": 900},
            "vector": {"fp4": 160, "int8": 120, "bf16": 60, "fp32": 30},
        },
        "gemm_tile": {"m": 128, "n": 128, "k": 64},
        "gemm_engines": 4,
        "vector_width": 32,
        "vector_units": 16,
        "kernel_launch_latency_us": {"gemm": 5, "vector": 3, "collective": 8},
        "interconnect": {
            "intra_node_gbps": 900,
            "intra_node_latency_us": 1,
            "inter_node_gbps": 400,
            "inter_node_latency_us": 5,
        },
        "cost_per_card_hour": 2.5,
    }


def run_cli(*arguments):
    return subprocess.run(
        [sys.executable, "-m", "infersim", *map(str, arguments)],
        cwd=ROOT,
        text=True,
        capture_output=True,
        encoding="utf-8",
        timeout=20,
    )


def search_arguments(stage, output):
    return (
        "search",
        "--model",
        MODEL,
        "--hardware",
        EXAMPLES / "custom_npu.json",
        "--precision",
        EXAMPLES / "w4a8.json",
        "--scenarios",
        EXAMPLES / "scenarios.json",
        "--search-space",
        EXAMPLES / "search_space.json",
        "--stage",
        stage,
        "--output",
        output,
    )


def without_option(arguments, option):
    values = list(arguments)
    index = values.index(option)
    del values[index : index + 2]
    return tuple(values)


def pair_arguments(output):
    return (
        "pair-pd",
        "--model",
        MODEL,
        "--prefill-hardware",
        EXAMPLES / "custom_npu.json",
        "--decode-hardware",
        EXAMPLES / "custom_npu.json",
        "--pd-link",
        EXAMPLES / "pd_link.json",
        "--precision",
        EXAMPLES / "w4a8.json",
        "--scenarios",
        EXAMPLES / "scenarios.json",
        "--prefill-search-space",
        EXAMPLES / "search_space.json",
        "--decode-search-space",
        EXAMPLES / "search_space.json",
        "--output",
        output,
    )


def tree_snapshot(root):
    return tuple(
        (
            "directory" if path.is_dir() else "file",
            path.relative_to(root).as_posix(),
            None if path.is_dir() else path.read_bytes(),
        )
        for path in sorted(root.rglob("*"))
    )


def make_owned_pair_output(root):
    root.mkdir()
    (root / PAIR_MARKER).write_bytes(PAIR_MARKER_BYTES)
    for directory, report_names in (
        ("prefill", STAGE_REPORTS),
        ("decode", STAGE_REPORTS),
        ("pd", PD_REPORTS),
    ):
        report_root = root / directory
        report_root.mkdir()
        for name in report_names:
            (report_root / name).write_bytes(
                f"old:{directory}/{name}".encode("ascii")
            )


def make_junction(link, target):
    if not hasattr(Path, "is_junction"):
        return False, "Path.is_junction is unavailable"
    try:
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(target)],
            text=True,
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
    except OSError as error:
        return False, str(error)
    if result.returncode != 0 or not link.is_junction():
        detail = result.stderr.strip() or result.stdout.strip()
        return False, detail or "junction creation failed"
    return True, ""


def run_main(*arguments):
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        code = cli.main(list(map(str, arguments)))
    return code, stdout.getvalue(), stderr.getvalue()


class SearchCliTests(unittest.TestCase):
    def test_prefill_and_decode_write_deterministic_reports(self):
        with tempfile.TemporaryDirectory(prefix="infersim cli ") as temporary:
            root = Path(temporary)
            for stage in ("prefill", "decode"):
                first = root / f"{stage} output one"
                second = root / f"{stage} output two"

                first_run = run_cli(*search_arguments(stage, first))
                second_run = run_cli(*search_arguments(stage, second))

                self.assertEqual(first_run.returncode, 0, first_run.stderr)
                self.assertEqual(second_run.returncode, 0, second_run.stderr)
                self.assertEqual(first_run.stderr, "")
                self.assertIn(f"stage: {stage}", first_run.stdout)
                self.assertIn("total cards:", first_run.stdout)
                self.assertIn("capacity:", first_run.stdout)
                self.assertEqual({path.name for path in first.iterdir()}, STAGE_REPORTS)
                self.assertEqual({path.name for path in second.iterdir()}, STAGE_REPORTS)
                for name in STAGE_REPORTS:
                    self.assertEqual(
                        (first / name).read_bytes(),
                        (second / name).read_bytes(),
                        name,
                    )

    def test_pair_pd_writes_three_report_directories(self):
        with tempfile.TemporaryDirectory(prefix="infersim pair ") as temporary:
            output = Path(temporary) / "pd output with spaces"
            command = pair_arguments(output)
            result = run_cli(*command)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertIn("PD pair:", result.stdout)
            self.assertIn("total cards:", result.stdout)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {PAIR_MARKER, "prefill", "decode", "pd"},
            )
            self.assertEqual((output / PAIR_MARKER).read_bytes(), PAIR_MARKER_BYTES)
            self.assertEqual(
                {path.name for path in (output / "prefill").iterdir()},
                STAGE_REPORTS,
            )
            self.assertEqual(
                {path.name for path in (output / "decode").iterdir()},
                STAGE_REPORTS,
            )
            self.assertEqual(
                {path.name for path in (output / "pd").iterdir()},
                PD_REPORTS,
            )
            recommendation = json.loads(
                (output / "pd" / "recommendation.json").read_text(encoding="utf-8")
            )
            self.assertIsNotNone(recommendation["recommendation"])
            inputs = recommendation["normalized_input_summary"]
            self.assertEqual(
                set(inputs),
                {
                    "model",
                    "precision",
                    "prefill_hardware",
                    "decode_hardware",
                    "prefill_search_space",
                    "decode_search_space",
                    "scenario_set",
                    "pd_link",
                },
            )
            self.assertEqual(inputs["model"]["model_type"], "qwen3")
            self.assertEqual(inputs["precision"]["gemm_mode"], "w4a8")
            self.assertEqual(
                inputs["prefill_hardware"]["name"], "Example 128GB NPU"
            )
            self.assertEqual(
                inputs["decode_hardware"]["name"], "Example 128GB NPU"
            )
            metric = recommendation["recommendation"]["scenarios"][0]
            self.assertIsInstance(metric["payload_bytes"], float)
            self.assertIn("effective_bandwidth_bytes_per_second", metric["transfer"])
            with (output / "pd" / "all_pairs.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                pair_row = next(csv.DictReader(handle))
            self.assertIn("minimum_link_request_capacity", pair_row)
            self.assertEqual(
                pair_row["minimum_link_request_capacity"],
                f'{metric["transfer"]["link_request_capacity"]:.6f}',
            )
            self.assertIn("link_request_capacity=", pair_row["transfer_summary"])

            second_output = Path(temporary) / "second pd output"
            second_command = list(command)
            second_command[second_command.index("--output") + 1] = second_output
            second = run_cli(*second_command)
            self.assertEqual(second.returncode, 0, second.stderr)
            for directory, names in (
                ("prefill", STAGE_REPORTS),
                ("decode", STAGE_REPORTS),
                ("pd", PD_REPORTS),
            ):
                for name in names:
                    self.assertEqual(
                        (output / directory / name).read_bytes(),
                        (second_output / directory / name).read_bytes(),
                        f"{directory}/{name}",
                    )

    def test_invalid_json_and_schema_errors_exit_two_on_stderr(self):
        with tempfile.TemporaryDirectory(prefix="infersim invalid ") as temporary:
            root = Path(temporary)
            malformed = root / "bad hardware.json"
            malformed.write_text('{"name": ', encoding="utf-8")
            malformed_args = list(
                search_arguments("prefill", root / "malformed output")
            )
            malformed_args[malformed_args.index("--hardware") + 1] = malformed
            malformed_result = run_cli(*malformed_args)
            self.assertEqual(malformed_result.returncode, 2)
            self.assertEqual(malformed_result.stdout, "")
            self.assertIn(str(malformed), malformed_result.stderr)
            self.assertIn("invalid JSON", malformed_result.stderr)

            hardware = hardware_fixture()
            del hardware["compute_tflops"]["vector"]
            invalid_schema = root / "invalid schema.json"
            invalid_schema.write_text(
                json.dumps(hardware), encoding="utf-8"
            )
            schema_args = list(
                search_arguments("decode", root / "schema output")
            )
            schema_args[schema_args.index("--hardware") + 1] = invalid_schema
            schema_result = run_cli(*schema_args)
            self.assertEqual(schema_result.returncode, 2)
            self.assertEqual(schema_result.stdout, "")
            self.assertIn(
                "compute_tflops.vector: field is required",
                schema_result.stderr,
            )

            missing_args = list(
                search_arguments("decode", root / "missing output")
            )
            missing_path = root / "does not exist.json"
            missing_args[missing_args.index("--precision") + 1] = missing_path
            missing_result = run_cli(*missing_args)
            self.assertEqual(missing_result.returncode, 2)
            self.assertEqual(missing_result.stdout, "")
            self.assertIn(str(missing_path), missing_result.stderr)
            self.assertIn("cannot read file", missing_result.stderr)

    def test_invalid_utf8_and_duplicate_keys_are_input_errors(self):
        with tempfile.TemporaryDirectory(prefix="infersim strict json ") as temporary:
            root = Path(temporary)
            cases = []

            invalid_utf8 = root / "invalid utf8.json"
            invalid_utf8.write_bytes(b"\xff")
            cases.append((invalid_utf8, "invalid UTF-8"))

            hardware = json.dumps(hardware_fixture())
            top_duplicate = root / "top duplicate.json"
            top_duplicate.write_text(
                '{"name":"duplicate",' + hardware[1:], encoding="utf-8"
            )
            cases.append((top_duplicate, "duplicate key 'name'"))

            nested_duplicate = root / "nested duplicate.json"
            nested_duplicate.write_text(
                hardware.replace(
                    '"compute_tflops": {',
                    '"compute_tflops": {"gemm":{"w4a8":1},',
                    1,
                ),
                encoding="utf-8",
            )
            cases.append((nested_duplicate, "duplicate key 'gemm'"))

            for path, message in cases:
                with self.subTest(path=path.name):
                    args = list(
                        search_arguments("prefill", root / (path.stem + " output"))
                    )
                    args[args.index("--hardware") + 1] = path
                    result = run_cli(*args)

                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertIn(str(path), result.stderr)
                    self.assertIn(message, result.stderr)
                    self.assertNotIn("Traceback", result.stderr)

    def test_default_search_space_is_bounded_and_pair_pd_completes(self):
        with tempfile.TemporaryDirectory(prefix="infersim defaults ") as temporary:
            root = Path(temporary)
            search_output = root / "search"
            search_result = run_cli(
                *without_option(
                    search_arguments("prefill", search_output),
                    "--search-space",
                )
            )

            self.assertEqual(search_result.returncode, 0, search_result.stderr)
            with (search_output / "all_candidates.csv").open(
                encoding="utf-8", newline=""
            ) as handle:
                candidates = list(csv.DictReader(handle))
            self.assertEqual(len(candidates), 864)
            self.assertLessEqual(len(candidates), 1000)

            pair_output = root / "pair"
            pair_result = run_cli(
                "pair-pd",
                "--model",
                MODEL,
                "--prefill-hardware",
                EXAMPLES / "custom_npu.json",
                "--decode-hardware",
                EXAMPLES / "custom_npu.json",
                "--pd-link",
                EXAMPLES / "pd_link.json",
                "--precision",
                EXAMPLES / "w4a8.json",
                "--scenarios",
                EXAMPLES / "scenarios.json",
                "--output",
                pair_output,
            )
            self.assertEqual(pair_result.returncode, 0, pair_result.stderr)
            for stage in ("prefill", "decode"):
                with (pair_output / stage / "all_candidates.csv").open(
                    encoding="utf-8", newline=""
                ) as handle:
                    self.assertEqual(sum(1 for _ in csv.DictReader(handle)), 864)

    def test_no_feasible_plan_still_writes_all_stage_reports(self):
        with tempfile.TemporaryDirectory(prefix="infersim infeasible ") as temporary:
            root = Path(temporary)
            hardware = hardware_fixture()
            hardware["memory_capacity_gb"] = 0.001
            impossible = root / "too small.json"
            impossible.write_text(json.dumps(hardware), encoding="utf-8")
            output = root / "diagnostic output"
            args = list(search_arguments("prefill", output))
            args[args.index("--hardware") + 1] = impossible

            result = run_cli(*args)

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertIn("no feasible recommendation", result.stdout)
            self.assertEqual({path.name for path in output.iterdir()}, STAGE_REPORTS)
            payload = json.loads(
                (output / "recommendation.json").read_text(encoding="utf-8")
            )
            self.assertIsNone(payload["recommendation"])

    def test_pair_pd_with_infeasible_stage_still_writes_diagnostics(self):
        with tempfile.TemporaryDirectory(prefix="infersim pd infeasible ") as temporary:
            root = Path(temporary)
            hardware = hardware_fixture()
            hardware["memory_capacity_gb"] = 0.001
            impossible = root / "impossible hardware.json"
            impossible.write_text(json.dumps(hardware), encoding="utf-8")
            output = root / "pd diagnostics"

            result = run_cli(
                "pair-pd",
                "--model",
                MODEL,
                "--prefill-hardware",
                impossible,
                "--decode-hardware",
                EXAMPLES / "custom_npu.json",
                "--pd-link",
                EXAMPLES / "pd_link.json",
                "--precision",
                EXAMPLES / "w4a8.json",
                "--scenarios",
                EXAMPLES / "scenarios.json",
                "--prefill-search-space",
                EXAMPLES / "search_space.json",
                "--decode-search-space",
                EXAMPLES / "search_space.json",
                "--output",
                output,
            )

            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertIn("no feasible recommendation", result.stdout)
            self.assertEqual(
                {path.name for path in (output / "prefill").iterdir()},
                STAGE_REPORTS,
            )
            self.assertEqual(
                {path.name for path in (output / "decode").iterdir()},
                STAGE_REPORTS,
            )
            self.assertEqual(
                {path.name for path in (output / "pd").iterdir()},
                PD_REPORTS,
            )
            pd_payload = json.loads(
                (output / "pd" / "recommendation.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIsNone(pd_payload["recommendation"])

    def test_pair_pd_rejects_a_non_directory_output_without_modifying_it(self):
        with tempfile.TemporaryDirectory(prefix="infersim output file ") as temporary:
            output = Path(temporary) / "existing result"
            output.write_bytes(b"owned by caller")

            result = run_cli(*pair_arguments(output))

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn(str(output), result.stderr)
            self.assertIn("must be a directory", result.stderr)
            self.assertEqual(output.read_bytes(), b"owned by caller")

    def test_search_output_io_error_is_reported_without_a_traceback(self):
        with tempfile.TemporaryDirectory(prefix="infersim search output ") as temporary:
            output = Path(temporary) / "existing result"
            output.write_bytes(b"owned by caller")

            result = run_cli(*search_arguments("prefill", output))

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertNotIn("Traceback", result.stderr)
            self.assertIn(str(output), result.stderr)
            self.assertEqual(output.read_bytes(), b"owned by caller")

    def test_pair_pd_report_failures_preserve_the_old_result_tree(self):
        with tempfile.TemporaryDirectory(prefix="infersim report rollback ") as temporary:
            parent = Path(temporary)
            real_stage_writer = cli.write_stage_reports

            def fail_second_stage():
                calls = 0

                def writer(*args):
                    nonlocal calls
                    calls += 1
                    if calls == 2:
                        raise OSError("injected second stage failure")
                    return real_stage_writer(*args)

                return patch("infersim.cli.write_stage_reports", side_effect=writer)

            failures = (
                fail_second_stage,
                lambda: patch(
                    "infersim.cli.write_pd_reports",
                    side_effect=OSError("injected PD report failure"),
                ),
            )
            for index, failure in enumerate(failures):
                with self.subTest(failure=failure):
                    output = parent / f"results-{index}"
                    make_owned_pair_output(output)
                    before = tree_snapshot(output)

                    with failure():
                        code, stdout, stderr = run_main(*pair_arguments(output))

                    self.assertEqual(code, 2)
                    self.assertEqual(stdout, "")
                    self.assertIn("error:", stderr)
                    self.assertEqual(tree_snapshot(output), before)
                    self.assertEqual(
                        {path.name for path in parent.iterdir()},
                        {f"results-{value}" for value in range(index + 1)},
                    )

    def test_pair_pd_publish_failure_restores_old_tree_and_cleans_staging(self):
        with tempfile.TemporaryDirectory(prefix="infersim publish rollback ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            make_owned_pair_output(output)
            before = tree_snapshot(output)
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path == output or destination_path == output:
                    calls += 1
                    if calls == 2:
                        raise OSError("injected publish failure")
                return real_replace(source, destination)

            with patch("infersim.cli.os.replace", side_effect=fail_second_replace):
                code, stdout, stderr = run_main(*pair_arguments(output))

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertIn("injected publish failure", stderr)
            self.assertEqual(tree_snapshot(output), before)
            self.assertEqual({path.name for path in parent.iterdir()}, {"results"})

    def test_pair_pd_backup_cleanup_failure_keeps_committed_new_reports(self):
        with tempfile.TemporaryDirectory(prefix="infersim cleanup commit ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            make_owned_pair_output(output)
            real_rmtree = cli.shutil.rmtree

            def partially_remove_backup(path):
                path = Path(path)
                if ".results.backup-" in path.name:
                    (path / "prefill" / "summary.txt").unlink()
                    raise PermissionError("injected partial backup cleanup")
                return real_rmtree(path)

            with patch(
                "infersim.cli.shutil.rmtree", side_effect=partially_remove_backup
            ):
                code, stdout, stderr = run_main(*pair_arguments(output))

            self.assertEqual(code, 2)
            self.assertEqual(stdout, "")
            self.assertNotIn("Traceback", stderr)
            self.assertIn(f"new reports published at {output.absolute()}", stderr)
            self.assertIn("backup cleanup failed at", stderr)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {PAIR_MARKER, "prefill", "decode", "pd"},
            )
            self.assertEqual((output / PAIR_MARKER).read_bytes(), PAIR_MARKER_BYTES)
            for directory, report_names in (
                ("prefill", STAGE_REPORTS),
                ("decode", STAGE_REPORTS),
                ("pd", PD_REPORTS),
            ):
                report_root = output / directory
                self.assertEqual(
                    {path.name for path in report_root.iterdir()}, report_names
                )
                self.assertTrue(
                    all(
                        not path.read_bytes().startswith(b"old:")
                        for path in report_root.iterdir()
                    )
                )
            residual = tuple(
                path for path in parent.iterdir() if ".results.backup-" in path.name
            )
            self.assertEqual(len(residual), 1)
            self.assertFalse(
                (residual[0] / "prefill" / "summary.txt").exists()
            )
            self.assertFalse(
                any(".tmp-" in path.name for path in parent.iterdir())
            )

    def test_pair_pd_non_io_backup_cleanup_error_keeps_committed_output(self):
        with tempfile.TemporaryDirectory(prefix="infersim cleanup runtime ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            make_owned_pair_output(output)
            real_rmtree = cli.shutil.rmtree

            def fail_backup_cleanup(path):
                if ".results.backup-" in Path(path).name:
                    raise RuntimeError("injected cleanup programming error")
                return real_rmtree(path)

            with patch(
                "infersim.cli.shutil.rmtree", side_effect=fail_backup_cleanup
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "injected cleanup programming error"
                ):
                    run_main(*pair_arguments(output))

            self.assertEqual((output / PAIR_MARKER).read_bytes(), PAIR_MARKER_BYTES)
            self.assertFalse(
                (output / "prefill" / "summary.txt").read_bytes().startswith(
                    b"old:"
                )
            )
            residual = tuple(
                path for path in parent.iterdir() if ".results.backup-" in path.name
            )
            self.assertEqual(len(residual), 1)

    def test_pair_pd_rejects_an_unowned_nonempty_result_root(self):
        with tempfile.TemporaryDirectory(prefix="infersim publish success ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            output.mkdir()
            (output / "old.txt").write_bytes(b"old generation")
            (output / ".git").mkdir()
            before = tree_snapshot(output)

            result = run_cli(*pair_arguments(output))

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("not an owned InferSim PD output", result.stderr)
            self.assertEqual(tree_snapshot(output), before)
            self.assertEqual({path.name for path in parent.iterdir()}, {"results"})

    def test_pair_pd_allows_empty_directory_and_exact_owned_rerun(self):
        with tempfile.TemporaryDirectory(prefix="infersim publish rerun ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            output.mkdir()

            first = run_cli(*pair_arguments(output))
            first_snapshot = tree_snapshot(output)
            second = run_cli(*pair_arguments(output))

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(tree_snapshot(output), first_snapshot)
            self.assertEqual(
                {path.name for path in output.iterdir()},
                {PAIR_MARKER, "prefill", "decode", "pd"},
            )
            self.assertEqual({path.name for path in parent.iterdir()}, {"results"})

    def test_pair_pd_rejects_corrupt_or_extended_owned_trees(self):
        with tempfile.TemporaryDirectory(prefix="infersim ownership ") as temporary:
            parent = Path(temporary)
            cases = {
                "extra-root": lambda root: (root / "notes.txt").write_text("keep"),
                "extra-report": lambda root: (
                    root / "prefill" / "private.txt"
                ).write_text("keep"),
                "corrupt-marker": lambda root: (root / PAIR_MARKER).write_text(
                    '{"format":"wrong","version":1}', encoding="utf-8"
                ),
            }
            for name, mutate in cases.items():
                with self.subTest(name=name):
                    output = parent / name
                    make_owned_pair_output(output)
                    mutate(output)
                    before = tree_snapshot(output)

                    result = run_cli(*pair_arguments(output))

                    self.assertEqual(result.returncode, 2)
                    self.assertEqual(result.stdout, "")
                    self.assertIn("not an owned InferSim PD output", result.stderr)
                    self.assertEqual(tree_snapshot(output), before)

    def test_pair_pd_rejects_symlink_output_when_supported(self):
        with tempfile.TemporaryDirectory(prefix="infersim ownership link ") as temporary:
            parent = Path(temporary)
            target = parent / "target"
            output = parent / "results"
            make_owned_pair_output(target)
            before = tree_snapshot(target)
            try:
                os.symlink(target, output, target_is_directory=True)
            except (OSError, NotImplementedError) as error:
                self.skipTest(f"directory symlinks unavailable: {error}")

            result = run_cli(*pair_arguments(output))

            self.assertEqual(result.returncode, 2)
            self.assertEqual(result.stdout, "")
            self.assertIn("symlink", result.stderr)
            self.assertTrue(output.is_symlink())
            self.assertEqual(tree_snapshot(target), before)

    def test_pair_pd_rejects_junction_output_when_supported(self):
        with tempfile.TemporaryDirectory(prefix="infersim ownership junction ") as temporary:
            parent = Path(temporary)
            target = parent / "target"
            output = parent / "results"
            make_owned_pair_output(target)
            before = tree_snapshot(target)
            created, detail = make_junction(output, target)
            if not created:
                self.skipTest(f"directory junctions unavailable: {detail}")
            try:
                result = run_cli(*pair_arguments(output))

                self.assertEqual(result.returncode, 2)
                self.assertEqual(result.stdout, "")
                self.assertIn(
                    "output must not be a symlink or junction", result.stderr
                )
                self.assertTrue(output.is_junction())
                self.assertEqual(tree_snapshot(target), before)
            finally:
                if output.is_junction():
                    output.rmdir()

    def test_generated_junction_is_never_traversed_during_cleanup(self):
        with tempfile.TemporaryDirectory(prefix="infersim cleanup junction ") as temporary:
            parent = Path(temporary).absolute()
            target = parent / "target"
            generated = parent / ".results.tmp-junction"
            target.mkdir()
            (target / "keep.txt").write_bytes(b"keep")
            created, detail = make_junction(generated, target)
            if not created:
                self.skipTest(f"directory junctions unavailable: {detail}")
            try:
                with self.assertRaisesRegex(RuntimeError, "link or junction"):
                    cli._remove_generated_tree(generated, parent, "staging")
                self.assertTrue(generated.is_junction())
                self.assertEqual((target / "keep.txt").read_bytes(), b"keep")
            finally:
                if generated.is_junction():
                    generated.rmdir()

    def test_pair_pd_runtime_publish_failure_restores_old_tree_and_reraises(self):
        with tempfile.TemporaryDirectory(prefix="infersim runtime rollback ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            make_owned_pair_output(output)
            before = tree_snapshot(output)
            real_replace = os.replace
            calls = 0

            def fail_second_replace(source, destination):
                nonlocal calls
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path == output or destination_path == output:
                    calls += 1
                    if calls == 2:
                        raise RuntimeError("injected runtime publish failure")
                return real_replace(source, destination)

            with patch("infersim.cli.os.replace", side_effect=fail_second_replace):
                with self.assertRaisesRegex(
                    RuntimeError, "injected runtime publish failure"
                ):
                    run_main(*pair_arguments(output))

            self.assertEqual(tree_snapshot(output), before)
            self.assertEqual({path.name for path in parent.iterdir()}, {"results"})

    def test_pair_pd_preserves_backup_when_rollback_itself_fails(self):
        with tempfile.TemporaryDirectory(prefix="infersim failed rollback ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            make_owned_pair_output(output)
            before = tree_snapshot(output)
            real_replace = os.replace
            publish_calls = 0

            def fail_publish(source, destination):
                nonlocal publish_calls
                source_path = Path(source)
                destination_path = Path(destination)
                if source_path == output or destination_path == output:
                    publish_calls += 1
                    if publish_calls == 2:
                        raise RuntimeError("injected runtime publish failure")
                return real_replace(source, destination)

            with (
                patch("infersim.cli.os.replace", side_effect=fail_publish),
                patch(
                    "infersim.cli.os.rename",
                    side_effect=PermissionError("injected rollback failure"),
                ),
            ):
                with self.assertRaisesRegex(
                    PermissionError, "injected rollback failure"
                ) as caught:
                    run_main(*pair_arguments(output))

            self.assertIsInstance(caught.exception.__cause__, RuntimeError)
            self.assertFalse(output.exists())
            entries = tuple(parent.iterdir())
            self.assertEqual(len(entries), 1)
            self.assertIn(".results.backup-", entries[0].name)
            self.assertEqual(tree_snapshot(entries[0]), before)

    def test_pair_pd_does_not_swallow_programming_errors(self):
        with tempfile.TemporaryDirectory(prefix="infersim programming error ") as temporary:
            parent = Path(temporary)
            output = parent / "results"
            make_owned_pair_output(output)
            before = tree_snapshot(output)

            with patch(
                "infersim.cli.write_pd_reports",
                side_effect=RuntimeError("injected programming error"),
            ):
                with self.assertRaisesRegex(
                    RuntimeError, "injected programming error"
                ):
                    run_main(*pair_arguments(output))

            self.assertEqual(tree_snapshot(output), before)
            self.assertEqual({path.name for path in parent.iterdir()}, {"results"})


if __name__ == "__main__":
    unittest.main()
