import csv
import json
import math
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import patch

from infersim.errors import InputValidationError
from infersim.report import CSV_FIELDS, write_stage_reports
from infersim.schema.parallel import PlanValidation
from infersim.search import SearchResult, run_stage_search

from tests.helpers import (
    make_candidate,
    make_dense_model,
    make_dense_plan,
    make_hardware,
    make_metrics,
    make_scenario,
    make_scenario_set,
    make_search_space,
    make_w4a8_precision,
)


class SearchResultTests(unittest.TestCase):
    def test_public_record_has_exact_fields_and_normalizes_tuples(self):
        candidate = make_candidate()
        result = SearchResult(
            stage="prefill",
            candidates=[candidate],
            feasible_candidates=[candidate],
            pareto_frontier=[candidate],
            recommendation=candidate,
            dominant_rejection=None,
        )

        self.assertEqual(
            tuple(field.name for field in fields(SearchResult)),
            (
                "stage",
                "candidates",
                "feasible_candidates",
                "pareto_frontier",
                "recommendation",
                "dominant_rejection",
            ),
        )
        self.assertIsInstance(result.candidates, tuple)
        self.assertIsInstance(result.feasible_candidates, tuple)
        self.assertIsInstance(result.pareto_frontier, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.stage = "decode"

    def test_rejects_wrong_types_stages_and_inconsistent_subsets(self):
        prefill = make_candidate(candidate_id="prefill")
        decode_metric = make_metrics(stage="decode")
        decode = make_candidate(
            candidate_id="decode",
            metrics=(decode_metric,),
            scenarios=(make_scenario(name=decode_metric.scenario_name),),
            ttft_ms=None,
            tpot_ms=10,
        )
        infeasible = make_candidate(
            candidate_id="bad",
            feasible=False,
            reason_codes=("FAILED",),
        )
        cases = (
            ({"stage": "other"}, "stage"),
            ({"candidates": (object(),)}, "candidates[0]"),
            ({"candidates": (decode,)}, "candidates[0].metrics[0].stage"),
            (
                {"candidates": (prefill,), "feasible_candidates": (decode,)},
                "feasible_candidates[0]",
            ),
            (
                {
                    "candidates": (infeasible,),
                    "feasible_candidates": (infeasible,),
                },
                "feasible_candidates[0].feasible",
            ),
            (
                {"candidates": (prefill,), "pareto_frontier": (decode,)},
                "pareto_frontier[0]",
            ),
            (
                {"candidates": (prefill,), "recommendation": decode},
                "recommendation",
            ),
            ({"dominant_rejection": 1}, "dominant_rejection"),
        )
        defaults = {
            "stage": "prefill",
            "candidates": (prefill,),
            "feasible_candidates": (),
            "pareto_frontier": (),
            "recommendation": None,
            "dominant_rejection": None,
        }
        for override, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    SearchResult(**(defaults | override))
                self.assertEqual(caught.exception.path, path)


class RunnerTests(unittest.TestCase):
    def inputs(self, **hardware_overrides):
        return {
            "stage": "prefill",
            "model": make_dense_model(),
            "hardware": make_hardware(**hardware_overrides),
            "precision": make_w4a8_precision(),
            "scenario_set": make_scenario_set((make_scenario(concurrency=2),)),
            "search_space": make_search_space(),
        }

    def test_invalid_plan_is_recorded_without_calling_evaluator(self):
        validation = PlanValidation(
            plan=make_dense_plan(attention_dp=2),
            feasible=False,
            reason_code="DENSE_PARALLELISM_INVALID",
            reason="dense mismatch",
        )
        with patch(
            "infersim.search.runner.enumerate_plans",
            return_value=iter((validation,)),
        ), patch("infersim.search.runner.evaluate_prefill") as evaluator:
            result = run_stage_search(**self.inputs())

        evaluator.assert_not_called()
        self.assertEqual(len(result.candidates), 1)
        candidate = result.candidates[0]
        self.assertFalse(candidate.feasible)
        self.assertEqual(candidate.metrics, ())
        self.assertEqual(candidate.scenarios, ())
        self.assertEqual(
            candidate.reason_codes, ("DENSE_PARALLELISM_INVALID",)
        )

    def test_candidate_ids_and_result_order_ignore_enumerator_order(self):
        plans = (
            make_dense_plan(batch_size=4),
            make_dense_plan(batch_size=2),
        )
        validations = tuple(PlanValidation(plan, True) for plan in plans)

        def evaluate(model, hardware, precision, plan, scenario):
            return make_metrics(
                name=scenario.name,
                plan=plan,
                request_capacity=plan.batch_size * 10,
            )

        outputs = []
        for order in (validations, tuple(reversed(validations))):
            with patch(
                "infersim.search.runner.enumerate_plans",
                return_value=iter(order),
            ), patch(
                "infersim.search.runner.evaluate_prefill",
                side_effect=evaluate,
            ):
                outputs.append(run_stage_search(**self.inputs()))

        forward, reverse = outputs
        self.assertEqual(
            [item.candidate_id for item in forward.candidates],
            [item.candidate_id for item in reverse.candidates],
        )
        self.assertEqual(
            [item.candidate_id for item in forward.candidates],
            sorted(item.candidate_id for item in forward.candidates),
        )
        self.assertEqual(
            len({item.candidate_id for item in forward.candidates}), 2
        )

    def test_search_does_not_mutate_inputs_and_computes_hourly_cost(self):
        inputs = self.inputs(cost_per_card_hour=1.25)
        before = {name: repr(value) for name, value in inputs.items()}

        result = run_stage_search(**inputs)

        self.assertEqual(before, {name: repr(value) for name, value in inputs.items()})
        self.assertEqual(result.candidates[0].hourly_cost, 1.25)

    def test_oom_has_no_feasible_plan_and_stable_dominant_reason(self):
        inputs = self.inputs(
            memory_capacity_gb=1e-9,
            memory_reserve_fraction=1,
        )
        inputs["stage"] = "decode"

        result = run_stage_search(**inputs)

        self.assertIsNone(result.recommendation)
        self.assertEqual(result.feasible_candidates, ())
        self.assertEqual(result.pareto_frontier, ())
        self.assertEqual(result.dominant_rejection, "MEMORY_CAPACITY")

    def test_reason_dominance_strips_scenario_and_breaks_ties_lexically(self):
        validations = (
            PlanValidation(make_dense_plan(batch_size=2), False, "Z_REASON"),
            PlanValidation(make_dense_plan(batch_size=3), False, "A_REASON"),
        )
        with patch(
            "infersim.search.runner.enumerate_plans",
            return_value=iter(validations),
        ):
            result = run_stage_search(**self.inputs())
        self.assertEqual(result.dominant_rejection, "A_REASON")

    def test_rejects_unormalized_inputs_and_wrong_stage(self):
        inputs = self.inputs()
        cases = (
            ("stage", "both"),
            ("model", {}),
            ("hardware", {}),
            ("precision", {}),
            ("scenario_set", {}),
            ("search_space", {}),
        )
        for field, value in cases:
            with self.subTest(field=field):
                bad = dict(inputs)
                bad[field] = value
                with self.assertRaises(InputValidationError) as caught:
                    run_stage_search(**bad)
                self.assertEqual(caught.exception.path, field)


class ReportingTests(unittest.TestCase):
    def result(self, **hardware_overrides):
        return run_stage_search(
            stage="prefill",
            model=make_dense_model(),
            hardware=make_hardware(**hardware_overrides),
            precision=make_w4a8_precision(),
            scenario_set=make_scenario_set((make_scenario(concurrency=2),)),
            search_space=make_search_space(),
        )

    def test_stage_search_writes_exact_required_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            write_stage_reports(Path(tmp), self.result())
            self.assertEqual(
                {path.name for path in Path(tmp).iterdir()},
                {
                    "all_candidates.csv",
                    "feasible_candidates.csv",
                    "pareto_frontier.csv",
                    "recommendation.json",
                    "summary.txt",
                },
            )

    def test_csv_has_fixed_header_sorted_rows_and_fixed_formats(self):
        first = make_candidate(
            candidate_id="z-last",
            hourly_cost=1.23456789,
            request_capacity=12.3456789,
            request_capacity_per_card=12.3456789,
            ttft_ms=50.123456789,
        )
        second = replace(first, candidate_id="a-first")
        result = SearchResult(
            "prefill",
            (first, second),
            (first, second),
            (first,),
            first,
            None,
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_stage_reports(Path(tmp), result)
            with (Path(tmp) / "all_candidates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(tuple(rows[0]), CSV_FIELDS)
        self.assertEqual([row["candidate_id"] for row in rows], ["a-first", "z-last"])
        self.assertEqual(rows[0]["hourly_cost"], "1.234568")
        self.assertEqual(rows[0]["request_capacity"], "12.345679")
        self.assertEqual(rows[0]["ttft_ms"], "50.123457")
        self.assertEqual(rows[0]["worst_latency_seconds"], "0.050000000000")
        self.assertEqual(rows[0]["worst_memory_required_bytes"], "7")

    def test_recommendation_json_is_sorted_complete_and_finite(self):
        result = self.result(cost_per_card_hour=2.0)
        with tempfile.TemporaryDirectory() as tmp:
            write_stage_reports(Path(tmp), result)
            raw = (Path(tmp) / "recommendation.json").read_text(encoding="utf-8")
            pairs = json.loads(raw, object_pairs_hook=list)
            payload = json.loads(raw)

        self.assertEqual([key for key, _ in pairs], sorted(key for key, _ in pairs))
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        selected = payload["recommendation"]
        self.assertEqual(selected["stage"], "prefill")
        self.assertIn("plan", selected)
        self.assertIn("summary", selected)
        self.assertIn("bottleneck", selected)
        self.assertEqual(len(selected["scenarios"]), 1)
        metric = selected["scenarios"][0]
        self.assertIn("component_seconds", metric)
        self.assertIn("memory", metric)
        self.assertEqual(
            metric["memory"]["total_required_bytes"],
            round(result.recommendation.metrics[0].memory.total_required_bytes),
        )

    def test_empty_results_still_write_headers_and_diagnostics(self):
        candidate = make_candidate(
            candidate_id="invalid",
            metrics=(),
            scenarios=(),
            feasible=False,
            reason_codes=("INVALID_PLAN",),
        )
        result = SearchResult(
            "prefill", (candidate,), (), (), None, "INVALID_PLAN"
        )
        with tempfile.TemporaryDirectory() as tmp:
            write_stage_reports(Path(tmp), result)
            feasible = (Path(tmp) / "feasible_candidates.csv").read_text(
                encoding="utf-8"
            )
            frontier = (Path(tmp) / "pareto_frontier.csv").read_text(
                encoding="utf-8"
            )
            diagnostic = json.loads(
                (Path(tmp) / "recommendation.json").read_text(encoding="utf-8")
            )
            summary = (Path(tmp) / "summary.txt").read_text(encoding="utf-8")

        self.assertEqual(feasible, frontier)
        self.assertEqual(feasible.splitlines()[0].split(","), list(CSV_FIELDS))
        self.assertIsNone(diagnostic["recommendation"])
        self.assertEqual(diagnostic["dominant_rejection"], "INVALID_PLAN")
        self.assertIn("No feasible plan", summary)
        self.assertIn("INVALID_PLAN", summary)

    def test_summary_contains_selected_plan_slo_bottleneck_and_top_rejections(self):
        result = self.result(cost_per_card_hour=1)
        with tempfile.TemporaryDirectory() as tmp:
            write_stage_reports(Path(tmp), result)
            summary = (Path(tmp) / "summary.txt").read_text(encoding="utf-8")

        self.assertIn("Stage: prefill", summary)
        self.assertIn("Selected plan:", summary)
        self.assertIn("Total cards: 1", summary)
        self.assertIn("SLO status: feasible", summary)
        self.assertIn("Component bottleneck:", summary)
        self.assertIn("Top rejection reasons:", summary)

    def test_two_writes_are_byte_for_byte_identical(self):
        result = self.result(cost_per_card_hour=1.25)
        with (
            tempfile.TemporaryDirectory() as left,
            tempfile.TemporaryDirectory() as right,
        ):
            write_stage_reports(Path(left), result)
            write_stage_reports(Path(right), result)
            left_bytes = {
                path.name: path.read_bytes() for path in Path(left).iterdir()
            }
            right_bytes = {
                path.name: path.read_bytes() for path in Path(right).iterdir()
            }
        self.assertEqual(left_bytes, right_bytes)

    def test_nonfinite_json_values_and_wrong_arguments_are_rejected(self):
        metric = make_metrics()
        bad_metric = replace(metric, component_seconds={"gemm": math.nan})
        candidate = make_candidate(metrics=(bad_metric,))
        result = SearchResult(
            "prefill", (candidate,), (candidate,), (candidate,), candidate, None
        )
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(InputValidationError):
                write_stage_reports(Path(tmp), result)
            with self.assertRaises(InputValidationError) as caught:
                write_stage_reports(Path(tmp), object())
            self.assertEqual(caught.exception.path, "result")
            with self.assertRaises(InputValidationError) as caught:
                write_stage_reports("not-a-path", self.result())
            self.assertEqual(caught.exception.path, "output_dir")


if __name__ == "__main__":
    unittest.main()
