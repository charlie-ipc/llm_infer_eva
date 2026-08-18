import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


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
            command = (
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
            result = run_cli(*command)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertIn("PD pair:", result.stdout)
            self.assertIn("total cards:", result.stdout)
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
            metric = recommendation["recommendation"]["scenarios"][0]
            self.assertIsInstance(metric["payload_bytes"], float)
            self.assertIn("effective_bandwidth_bytes_per_second", metric["transfer"])

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


if __name__ == "__main__":
    unittest.main()
