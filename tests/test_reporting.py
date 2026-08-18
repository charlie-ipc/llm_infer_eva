import csv
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from pathlib import Path
from unittest.mock import patch

import infersim.report as report_module
from infersim.errors import InputValidationError
from infersim.report import CSV_FIELDS, write_stage_reports
from infersim.schema.hardware import HardwareSpec
from infersim.schema.parallel import PlanValidation, SearchSpace
from infersim.schema.scenario import ScenarioSet, WorkloadScenario
from infersim.search import (
    CandidateDiagnostic,
    SearchContext,
    SearchResult,
    run_stage_search,
)

from tests.helpers import (
    make_candidate,
    make_dense_model,
    make_dense_plan,
    make_hardware,
    make_mla_moe_model,
    make_moe_plan,
    make_metrics,
    make_scenario,
    make_scenario_set,
    make_search_space,
    make_w4a8_precision,
)


REPORT_FILES = {
    "all_candidates.csv",
    "feasible_candidates.csv",
    "pareto_frontier.csv",
    "recommendation.json",
    "summary.txt",
}


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
                "diagnostics",
                "context",
            ),
        )
        self.assertIsInstance(result.candidates, tuple)
        self.assertIsInstance(result.feasible_candidates, tuple)
        self.assertIsInstance(result.pareto_frontier, tuple)
        with self.assertRaises(FrozenInstanceError):
            result.stage = "decode"

    def test_diagnostic_and_context_records_are_frozen_and_normalized(self):
        diagnostic = CandidateDiagnostic("candidate", "FAILED", "detail")
        context = SearchContext(
            model=make_dense_model(),
            hardware=make_hardware(),
            precision=make_w4a8_precision(),
            scenario_set=make_scenario_set(),
            search_space=make_search_space(),
            assumptions=["one", "two"],
        )

        self.assertEqual(
            tuple(field.name for field in fields(CandidateDiagnostic)),
            ("candidate_id", "reason_code", "detail"),
        )
        self.assertEqual(
            tuple(field.name for field in fields(SearchContext)),
            (
                "model",
                "hardware",
                "precision",
                "scenario_set",
                "search_space",
                "assumptions",
            ),
        )
        self.assertEqual(context.assumptions, ("one", "two"))
        with self.assertRaises(FrozenInstanceError):
            diagnostic.detail = "changed"
        with self.assertRaises(FrozenInstanceError):
            context.model = make_dense_model()

    def test_context_rejects_invalid_nested_values_with_exact_paths(self):
        defaults = {
            "model": make_dense_model(),
            "hardware": make_hardware(),
            "precision": make_w4a8_precision(),
            "scenario_set": make_scenario_set(),
            "search_space": make_search_space(),
            "assumptions": ("assumption",),
        }
        base_scenario = make_scenario()
        invalid_name_scenario = WorkloadScenario(
            **{
                item.name: []
                if item.name == "name"
                else getattr(base_scenario, item.name)
                for item in fields(WorkloadScenario)
            }
        )
        cases = (
            (
                {"scenario_set": ScenarioSet("all", [object()])},
                "context.scenario_set.scenarios[0]",
            ),
            (
                {
                    "scenario_set": ScenarioSet(
                        "all", [invalid_name_scenario]
                    )
                },
                "context.scenario_set.scenarios[0].name",
            ),
            (
                {
                    "search_space": replace(
                        make_search_space(), attention_tp=[True]
                    )
                },
                "context.search_space.attention_tp[0]",
            ),
            (
                {
                    "search_space": replace(
                        make_search_space(), attention_tp=[1, 1]
                    )
                },
                "context.search_space.attention_tp[1]",
            ),
            (
                {
                    "hardware": replace(
                        make_hardware(), gemm_tflops={"w4a8": "bad"}
                    )
                },
                "context.hardware.gemm_tflops.w4a8",
            ),
            (
                {
                    "hardware": replace(
                        make_hardware(), gemm_tile=[128, object(), 64]
                    )
                },
                "context.hardware.gemm_tile[1]",
            ),
        )
        for override, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    SearchContext(**(defaults | override))
                self.assertEqual(caught.exception.path, path)

    def test_context_normalizes_performance_overflow_errors_with_exact_paths(self):
        defaults = {
            "model": make_dense_model(),
            "precision": make_w4a8_precision(),
            "scenario_set": make_scenario_set(),
            "search_space": make_search_space(),
            "assumptions": ("assumption",),
        }
        cases = (
            ("gemm_tflops", "w4a8"),
            ("vector_tflops", "int8"),
        )
        for field, mode in cases:
            with self.subTest(field=field):
                base_hardware = make_hardware()
                values = {
                    item.name: getattr(base_hardware, item.name)
                    for item in fields(HardwareSpec)
                }
                values[field] = {mode: 10**10000}
                hardware = HardwareSpec(**values)
                with self.assertRaises(InputValidationError) as caught:
                    SearchContext(hardware=hardware, **defaults)
                self.assertEqual(
                    caught.exception.path,
                    f"context.hardware.{field}.{mode}",
                )
                self.assertEqual(caught.exception.message, "must be finite")

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
            (
                {
                    "candidates": (decode,),
                    "feasible_candidates": (decode,),
                    "pareto_frontier": (decode,),
                    "recommendation": decode,
                },
                "candidates[0].metrics[0].stage",
            ),
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
            "feasible_candidates": (prefill,),
            "pareto_frontier": (prefill,),
            "recommendation": prefill,
            "dominant_rejection": None,
        }
        for override, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    SearchResult(**(defaults | override))
                self.assertEqual(caught.exception.path, path)

    def test_requires_exact_derived_collections_and_diagnostics(self):
        first = make_candidate(candidate_id="a")
        second = make_candidate(candidate_id="b", plan=make_dense_plan(batch_size=4))
        failed = make_candidate(
            candidate_id="failed",
            metrics=(),
            scenarios=(),
            feasible=False,
            reason_codes=("FAILED",),
        )
        failed_diagnostic = CandidateDiagnostic("failed", "FAILED", "detail")
        valid = {
            "stage": "prefill",
            "candidates": (first, second, failed),
            "feasible_candidates": (first, second),
            "pareto_frontier": (first, second),
            "recommendation": first,
            "dominant_rejection": "FAILED",
            "diagnostics": (failed_diagnostic,),
        }
        cases = (
            ({"candidates": (second, first, failed)}, "candidates"),
            ({"feasible_candidates": (first,)}, "feasible_candidates"),
            ({"pareto_frontier": (first,)}, "pareto_frontier"),
            ({"recommendation": None}, "recommendation"),
            ({"dominant_rejection": "OTHER"}, "dominant_rejection"),
            ({"diagnostics": ()}, "diagnostics"),
            (
                {
                    "diagnostics": (
                        CandidateDiagnostic("other", "FAILED", "detail"),
                    )
                },
                "diagnostics[0].candidate_id",
            ),
            (
                {
                    "diagnostics": (
                        CandidateDiagnostic("failed", "OTHER", "detail"),
                    )
                },
                "diagnostics[0].reason_code",
            ),
        )
        for override, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    SearchResult(**(valid | override))
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
        self.assertEqual(
            result.diagnostics,
            (
                CandidateDiagnostic(
                    candidate.candidate_id,
                    "DENSE_PARALLELISM_INVALID",
                    "dense mismatch",
                ),
            ),
        )

    def test_shared_expert_invalid_plan_does_not_abort_valid_plan_search(self):
        model = make_mla_moe_model(
            num_shared_experts=1,
            shared_expert_intermediate_size=5,
        )
        search_space = SearchSpace(
            total_cards=(1, 2),
            replicas=(1,),
            attention_tp=(1, 2),
            attention_dp=(1,),
            moe_tp=(1, 2),
            expert_parallel=(1,),
            batch_sizes=(2,),
        )
        inputs = self.inputs()
        inputs["model"] = model
        inputs["search_space"] = search_space

        result = run_stage_search(**inputs)

        invalid = next(
            candidate
            for candidate in result.candidates
            if candidate.plan == make_moe_plan(
                attention_tp=2,
                attention_dp=1,
                moe_tp=2,
                expert_parallel=1,
            )
            and candidate.reason_codes
            == ("SHARED_INTERMEDIATE_NOT_DIVISIBLE",)
        )
        valid = next(
            candidate
            for candidate in result.candidates
            if candidate.plan == make_moe_plan(
                attention_tp=1,
                attention_dp=1,
                moe_tp=1,
                expert_parallel=1,
            )
            and candidate.metrics
        )
        diagnostic = next(
            item
            for item in result.diagnostics
            if item.candidate_id == invalid.candidate_id
        )

        self.assertEqual(invalid.metrics, ())
        self.assertTrue(valid.metrics)
        self.assertEqual(
            diagnostic.detail,
            "shared expert intermediate size must be divisible by moe_tp",
        )

    def test_runner_preserves_normalized_context_and_assumptions(self):
        inputs = self.inputs()

        result = run_stage_search(**inputs)

        self.assertIs(result.context.model, inputs["model"])
        self.assertIs(result.context.precision, inputs["precision"])
        self.assertEqual(result.context.hardware, inputs["hardware"])
        self.assertEqual(result.context.scenario_set, inputs["scenario_set"])
        self.assertEqual(result.context.search_space, inputs["search_space"])
        self.assertIsNot(result.context.hardware, inputs["hardware"])
        self.assertIsNot(result.context.scenario_set, inputs["scenario_set"])
        self.assertIsNot(result.context.search_space, inputs["search_space"])
        self.assertTrue(
            any("no P99 queueing" in item for item in result.context.assumptions)
        )
        self.assertTrue(
            any("independently" in item for item in result.context.assumptions)
        )
        self.assertTrue(
            any("roofline" in item for item in result.context.assumptions)
        )
        self.assertTrue(
            any("Collective" in item for item in result.context.assumptions)
        )

    def test_scenario_rejection_diagnostic_keeps_name_and_base_detail(self):
        inputs = self.inputs(
            memory_capacity_gb=1e-9,
            memory_reserve_fraction=1,
        )
        inputs["scenario_set"] = make_scenario_set(
            (make_scenario(name="large-input", concurrency=2),)
        )

        result = run_stage_search(**inputs)

        memory_diagnostic = next(
            item
            for item in result.diagnostics
            if item.reason_code == "large-input:MEMORY_CAPACITY"
        )
        self.assertIn("large-input", memory_diagnostic.detail)
        self.assertIn("memory capacity", memory_diagnostic.detail.lower())

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

    def test_candidate_ids_resist_slug_suffix_collisions(self):
        plan = make_dense_plan()
        validations = (
            PlanValidation(plan, False, "FOO", "first foo"),
            PlanValidation(plan, False, "FOO", "second foo"),
            PlanValidation(plan, False, "FOO-N1", "looks like a suffix"),
        )
        outputs = []
        for order in (validations, tuple(reversed(validations))):
            with patch(
                "infersim.search.runner.enumerate_plans",
                return_value=iter(order),
            ):
                outputs.append(run_stage_search(**self.inputs()))

        id_sets = [
            tuple(candidate.candidate_id for candidate in result.candidates)
            for result in outputs
        ]
        self.assertEqual(id_sets[0], id_sets[1])
        self.assertEqual(len(set(id_sets[0])), len(validations))

    def test_search_does_not_mutate_inputs_and_computes_hourly_cost(self):
        inputs = self.inputs(cost_per_card_hour=1.25)
        before = {name: repr(value) for name, value in inputs.items()}

        result = run_stage_search(**inputs)

        self.assertEqual(before, {name: repr(value) for name, value in inputs.items()})
        self.assertEqual(result.candidates[0].hourly_cost, 1.25)

    def test_search_context_snapshots_mutable_normalized_containers(self):
        scenario_values = [make_scenario(concurrency=2)]
        scenario_set = ScenarioSet("all", scenario_values)
        axes = {
            "total_cards": [1],
            "replicas": [1],
            "attention_tp": [1],
            "attention_dp": [1],
            "moe_tp": [1],
            "expert_parallel": [1],
            "batch_sizes": [2],
        }
        search_space = SearchSpace(**axes)
        base_hardware = make_hardware()
        gemm_tflops = dict(base_hardware.gemm_tflops)
        vector_tflops = dict(base_hardware.vector_tflops)
        gemm_tile = list(base_hardware.gemm_tile)
        hardware = replace(
            base_hardware,
            gemm_tflops=gemm_tflops,
            vector_tflops=vector_tflops,
            gemm_tile=gemm_tile,
        )

        result = run_stage_search(
            "prefill",
            make_dense_model(),
            hardware,
            make_w4a8_precision(),
            scenario_set,
            search_space,
        )
        with tempfile.TemporaryDirectory() as before_dir:
            write_stage_reports(Path(before_dir), result)
            before = {
                path.name: path.read_bytes() for path in Path(before_dir).iterdir()
            }

        scenario_values.append(make_scenario(name="later", concurrency=2))
        axes["batch_sizes"].append(4)
        gemm_tflops["w4a8"] = 1
        vector_tflops["int8"] = 1
        gemm_tile[0] = 1

        context = result.context
        self.assertEqual(context.scenario_set.scenarios, (scenario_values[0],))
        self.assertEqual(context.search_space.batch_sizes, (2,))
        self.assertEqual(context.hardware.gemm_tflops["w4a8"], 900.0)
        self.assertEqual(context.hardware.vector_tflops["int8"], 120.0)
        self.assertEqual(context.hardware.gemm_tile, (128, 128, 64))
        with self.assertRaises(TypeError):
            context.hardware.gemm_tflops["w4a8"] = 2

        with tempfile.TemporaryDirectory() as after_dir:
            write_stage_reports(Path(after_dir), result)
            after = {
                path.name: path.read_bytes() for path in Path(after_dir).iterdir()
            }
        self.assertEqual(after, before)

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
                REPORT_FILES,
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
            (second, first),
            (second, first),
            (second,),
            second,
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
        self.assertEqual(rows[0]["worst_gemm_seconds"], "0.050000000000")
        self.assertEqual(rows[0]["worst_vector_seconds"], "0.000000000000")
        self.assertEqual(rows[0]["worst_tp_seconds"], "0.000000000000")
        self.assertEqual(rows[0]["worst_ep_seconds"], "0.000000000000")
        self.assertEqual(rows[0]["prompt_token_capacity"], "1280.000000")
        self.assertEqual(rows[0]["output_token_capacity"], "")
        self.assertEqual(rows[0]["reason_details"], "")

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
        self.assertEqual(payload["assumptions"], list(result.context.assumptions))
        normalized = payload["normalized_input_summary"]
        self.assertEqual(
            set(normalized["model"]),
            {field.name for field in fields(result.context.model)},
        )
        self.assertEqual(
            set(normalized["hardware"]),
            {field.name for field in fields(result.context.hardware)},
        )
        self.assertEqual(
            set(normalized["precision"]),
            {field.name for field in fields(result.context.precision)},
        )
        self.assertEqual(
            set(normalized["search_space"]),
            {field.name for field in fields(result.context.search_space)},
        )
        scenario = normalized["scenario_set"]["scenarios"][0]
        self.assertEqual(
            set(scenario),
            {field.name for field in fields(result.context.scenario_set.scenarios[0])},
        )
        self.assertEqual(
            normalized["hardware"]["gemm_tflops"],
            dict(sorted(result.context.hardware.gemm_tflops.items())),
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
            "prefill",
            (candidate,),
            (),
            (),
            None,
            "INVALID_PLAN",
            (
                CandidateDiagnostic(
                    "invalid", "INVALID_PLAN", "invalid plan"
                ),
            ),
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
            with (Path(tmp) / "all_candidates.csv").open(
                newline="", encoding="utf-8"
            ) as handle:
                all_rows = list(csv.DictReader(handle))

        self.assertEqual(feasible, frontier)
        self.assertEqual(feasible.splitlines()[0].split(","), list(CSV_FIELDS))
        self.assertIsNone(diagnostic["recommendation"])
        self.assertEqual(diagnostic["dominant_rejection"], "INVALID_PLAN")
        self.assertEqual(diagnostic["assumptions"], [])
        self.assertIsNone(diagnostic["normalized_input_summary"])
        self.assertEqual(all_rows[0]["reason_details"], "invalid plan")
        self.assertEqual(
            diagnostic["top_rejection_reasons"][0]["details"],
            ["invalid plan"],
        )
        self.assertIn("No feasible plan", summary)
        self.assertIn("INVALID_PLAN", summary)
        self.assertIn("invalid plan", summary)

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

    def test_preflight_rejects_bad_numbers_without_changing_old_reports(self):
        corruptions = (
            (
                "useful_gemm_ops",
                1.5,
                "candidates[0].metrics[0].useful_gemm_ops",
            ),
            (
                "max_supported_batch",
                -1,
                "candidates[0].metrics[0].max_supported_batch",
            ),
            (
                "latency_seconds",
                math.nan,
                "candidates[0].metrics[0].latency_seconds",
            ),
        )
        for field, value, expected_path in corruptions:
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                output = Path(tmp)
                old = {
                    name: f"old:{name}".encode("utf-8")
                    for name in REPORT_FILES
                }
                for name, content in old.items():
                    (output / name).write_bytes(content)
                metric = replace(make_metrics(), **{field: value})
                candidate = make_candidate(metrics=(metric,))
                result = SearchResult(
                    "prefill",
                    (candidate,),
                    (candidate,),
                    (candidate,),
                    candidate,
                    None,
                )

                with self.assertRaises(InputValidationError) as caught:
                    write_stage_reports(output, result)

                self.assertEqual(caught.exception.path, expected_path)
                self.assertEqual(
                    {name: (output / name).read_bytes() for name in REPORT_FILES},
                    old,
                )
                self.assertEqual({path.name for path in output.iterdir()}, REPORT_FILES)

    def test_replace_failure_rolls_back_and_cleans_temporary_files(self):
        result = self.result(cost_per_card_hour=1.0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            old = {
                name: f"old:{name}".encode("utf-8")
                for name in REPORT_FILES
            }
            for name, content in old.items():
                (output / name).write_bytes(content)
            real_replace = os.replace
            calls = 0

            def fail_third(source, destination):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise OSError("injected replace failure")
                return real_replace(source, destination)

            with patch("infersim.report.os.replace", side_effect=fail_third):
                with self.assertRaises(OSError):
                    write_stage_reports(output, result)

            self.assertEqual(
                {name: (output / name).read_bytes() for name in REPORT_FILES},
                old,
            )
            self.assertEqual({path.name for path in output.iterdir()}, REPORT_FILES)

    def test_temporary_creation_failure_cleans_prior_temporary_files(self):
        result = self.result()
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            real_temporary_file = report_module._temporary_file
            calls = 0

            def fail_second(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected temporary creation failure")
                return real_temporary_file(*args, **kwargs)

            with patch(
                "infersim.report._temporary_file", side_effect=fail_second
            ):
                with self.assertRaises(OSError):
                    write_stage_reports(output, result)

            self.assertEqual(tuple(output.iterdir()), ())

    def test_import_order_is_stable(self):
        orders = (
            "import infersim.cost; import infersim.search; import infersim.report",
            "import infersim.report; import infersim.search; import infersim.cost",
        )
        for statement in orders:
            with self.subTest(statement=statement):
                completed = subprocess.run(
                    [sys.executable, "-c", statement],
                    cwd=Path(__file__).resolve().parents[1],
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(completed.returncode, 0, completed.stderr)


if __name__ == "__main__":
    unittest.main()
