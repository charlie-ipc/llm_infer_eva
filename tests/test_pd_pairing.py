import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace
from unittest.mock import patch

from infersim.cost import (
    PDMetrics,
    PDTransferMetrics,
    evaluate_pd_pair,
    kv_bytes_per_request,
    pd_payload_bytes,
    recurrent_state_bytes_per_request,
)
from infersim.errors import InputValidationError
from infersim.schema.scenario import ScenarioSet
from infersim.search import (
    PDCandidate,
    PDSearchResult,
    SearchContext,
    SearchResult,
    pair_stage_results,
)
from tests.helpers import (
    make_decode_candidate,
    make_dense_model,
    make_hybrid_model,
    make_hardware,
    make_mla_moe_model,
    make_metrics,
    make_pd_link,
    make_prefill_candidate,
    make_scenario,
    make_scenario_set,
    make_search_result,
    make_search_space,
    make_w4a4_precision,
    make_w4a8_precision,
)


def with_context(
    result,
    scenario_set,
    *,
    model=None,
    precision=None,
):
    return replace(
        result,
        context=SearchContext(
            model=model or make_dense_model(),
            hardware=make_hardware(),
            precision=precision or make_w4a8_precision(),
            scenario_set=scenario_set,
            search_space=make_search_space(),
            assumptions=("test",),
        ),
    )


def with_only_pd_candidate(result, candidate):
    return replace(
        result,
        candidates=(candidate,),
        feasible_candidates=(candidate,) if candidate.feasible else (),
        pareto_frontier=(candidate,) if candidate.feasible else (),
        recommendation=candidate if candidate.feasible else None,
        dominant_rejection=(
            None
            if candidate.feasible
            else candidate.reason_codes[0].rsplit(":", 1)[-1]
        ),
    )


class PDMetricTests(unittest.TestCase):
    def test_transfer_and_first_decode_step_contribute_to_ttft(self):
        scenario = make_scenario(request_rate=50, ttft_limit_ms=100)

        metrics = evaluate_pd_pair(
            make_prefill_candidate(latency_ms=20, request_capacity=100),
            make_decode_candidate(tpot_ms=5, request_capacity=80),
            make_pd_link(bandwidth_gbps=10, latency_us=100),
            kv_state_bytes=1_000_000,
            scenario=scenario,
        )

        transfer_seconds = 100e-6 + 1_000_000 / 10e9
        self.assertAlmostEqual(metrics.transfer.transfer_seconds, transfer_seconds)
        self.assertAlmostEqual(metrics.ttft_ms, 20 + transfer_seconds * 1000 + 5)
        self.assertEqual(metrics.tpot_ms, 5)
        self.assertEqual(metrics.bottleneck, "decode")
        self.assertEqual(metrics.system_request_capacity, 80)
        self.assertTrue(metrics.feasible)

    def test_link_efficiency_controls_time_capacity_and_bottleneck(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(latency_ms=5, request_capacity=1000),
            make_decode_candidate(tpot_ms=5, request_capacity=1000),
            make_pd_link(
                bandwidth_gbps=0.01, latency_us=10, efficiency=0.5
            ),
            kv_state_bytes=1_000_000,
            scenario=make_scenario(request_rate=1, ttft_limit_ms=1000),
        )

        self.assertEqual(metrics.transfer.effective_bandwidth_bytes_per_second, 5e6)
        self.assertAlmostEqual(metrics.transfer.transfer_seconds, 0.20001)
        self.assertEqual(metrics.transfer.link_request_capacity, 5)
        self.assertEqual(metrics.system_request_capacity, 5)
        self.assertEqual(metrics.bottleneck, "pd_link")

    def test_bottleneck_tie_uses_prefill_decode_link_order(self):
        common = dict(
            pd_link=make_pd_link(bandwidth_gbps=1),
            kv_state_bytes=10_000_000,
            scenario=make_scenario(request_rate=1, ttft_limit_ms=1000),
        )
        both_stage_tie = evaluate_pd_pair(
            make_prefill_candidate(request_capacity=100),
            make_decode_candidate(request_capacity=100),
            **common,
        )
        all_tie = evaluate_pd_pair(
            make_prefill_candidate(request_capacity=100),
            make_decode_candidate(request_capacity=1000),
            **common,
        )

        self.assertEqual(both_stage_tie.bottleneck, "prefill")
        self.assertEqual(all_tie.bottleneck, "prefill")

    def test_transfer_concurrency_changes_feasibility_not_payload(self):
        scenario = make_scenario(request_rate=15, ttft_limit_ms=1000)
        args = (
            make_prefill_candidate(request_capacity=100),
            make_decode_candidate(request_capacity=100),
        )
        limited = evaluate_pd_pair(
            *args,
            make_pd_link(
                bandwidth_gbps=1000,
                latency_us=100_000,
                max_concurrent_transfers=1,
            ),
            kv_state_bytes=1_000_000,
            scenario=scenario,
        )
        wide = evaluate_pd_pair(
            *args,
            make_pd_link(
                bandwidth_gbps=1000,
                latency_us=100_000,
                max_concurrent_transfers=2,
            ),
            kv_state_bytes=1_000_000,
            scenario=scenario,
        )

        self.assertEqual(limited.transfer.payload_bytes, wide.transfer.payload_bytes)
        self.assertEqual(limited.transfer.concurrent_transfers_required, 2)
        self.assertIn("interactive:PD_TRANSFER_CONCURRENCY", limited.reason_codes)
        self.assertFalse(limited.feasible)
        self.assertTrue(wide.feasible)

    def test_rate_and_slo_failures_have_phase_specific_codes(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(latency_ms=80, request_capacity=2),
            make_decode_candidate(tpot_ms=30, request_capacity=3),
            make_pd_link(bandwidth_gbps=0.001),
            kv_state_bytes=1_000_000,
            scenario=make_scenario(
                request_rate=5, ttft_limit_ms=50, tpot_limit_ms=20
            ),
        )

        self.assertEqual(
            metrics.reason_codes,
            (
                "interactive:PREFILL_RATE",
                "interactive:DECODE_RATE",
                "interactive:PD_LINK_RATE",
                "interactive:TTFT_SLO",
                "interactive:TPOT_SLO",
            ),
        )

    def test_metrics_are_frozen_and_deeply_immutable(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(),
            make_decode_candidate(),
            make_pd_link(),
            kv_state_bytes=1_000_000,
            scenario=make_scenario(),
        )

        self.assertIsInstance(metrics, PDMetrics)
        self.assertIsInstance(metrics.transfer, PDTransferMetrics)
        self.assertIsInstance(metrics.reason_codes, tuple)
        self.assertIsInstance(metrics.warnings, tuple)
        with self.assertRaises(FrozenInstanceError):
            metrics.ttft_ms = 0
        with self.assertRaises(FrozenInstanceError):
            metrics.transfer.payload_bytes = 0

    def test_metric_records_reject_inconsistent_derived_fields(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(),
            make_decode_candidate(),
            make_pd_link(),
            1_000_000,
            make_scenario(),
        )
        cases = (
            ("system_request_capacity", 999, "system_request_capacity"),
            ("bottleneck", "prefill", "bottleneck"),
            ("feasible", False, "feasible"),
            ("ttft_ms", -1, "ttft_ms"),
        )
        for field, value, path in cases:
            with self.subTest(field=field):
                with self.assertRaises(InputValidationError) as caught:
                    replace(metrics, **{field: value})
                self.assertEqual(caught.exception.path, path)
        with self.assertRaises(InputValidationError) as caught:
            replace(metrics.transfer, payload_bytes=0)
        self.assertEqual(caught.exception.path, "payload_bytes")

    def test_transfer_record_rejects_inconsistent_link_capacity(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(),
            make_decode_candidate(),
            make_pd_link(),
            1_000_000,
            make_scenario(),
        )
        with self.assertRaises(InputValidationError) as caught:
            replace(
                metrics.transfer,
                link_request_capacity=(
                    metrics.transfer.link_request_capacity + 1
                ),
            )
        self.assertEqual(caught.exception.path, "link_request_capacity")

    def test_concurrent_work_overflow_has_a_stable_path(self):
        with self.assertRaises(InputValidationError) as caught:
            evaluate_pd_pair(
                make_prefill_candidate(),
                make_decode_candidate(),
                make_pd_link(latency_us=1e307),
                1,
                replace(make_scenario(), request_rate=1e308),
            )
        self.assertEqual(caught.exception.path, "concurrent_transfers_required")

    def test_pair_evaluator_rejects_wrong_stage_infeasible_and_bad_payload(self):
        prefill = make_prefill_candidate()
        decode = make_decode_candidate()
        cases = (
            (
                (decode, decode, make_pd_link(), 1, make_scenario()),
                "prefill_candidate.metrics[0].stage",
            ),
            (
                (
                    replace(prefill, feasible=False, reason_codes=("X",)),
                    decode,
                    make_pd_link(),
                    1,
                    make_scenario(),
                ),
                "prefill_candidate.feasible",
            ),
            ((prefill, decode, make_pd_link(), 0, make_scenario()), "kv_state_bytes"),
            ((prefill, decode, make_pd_link(), True, make_scenario()), "kv_state_bytes"),
        )
        for args, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    evaluate_pd_pair(*args[:-1], scenario=args[-1])
                self.assertEqual(caught.exception.path, path)

    def test_fractional_payload_flows_through_transfer_formula(self):
        scenario = make_scenario(input_length=1)
        metrics = evaluate_pd_pair(
            make_prefill_candidate(scenarios=(scenario,)),
            make_decode_candidate(scenarios=(scenario,)),
            make_pd_link(bandwidth_gbps=1, latency_us=0),
            kv_state_bytes=1.5,
            scenario=scenario,
        )

        self.assertEqual(metrics.transfer.payload_bytes, 1.5)
        self.assertAlmostEqual(metrics.transfer.transfer_seconds, 1.5 / 1e9)
        self.assertAlmostEqual(
            metrics.transfer.link_request_capacity, 1e9 / 1.5
        )
        paired = pair_stage_results(
            make_search_result(
                (make_prefill_candidate(scenarios=(scenario,)),)
            ),
            make_search_result(
                (make_decode_candidate(scenarios=(scenario,)),)
            ),
            make_pd_link(),
            make_scenario_set((scenario,)),
            {"interactive": 1.5},
        )
        self.assertEqual(
            paired.candidates[0].metrics[0].transfer.payload_bytes, 1.5
        )

    def test_huge_direct_payload_is_a_validation_error(self):
        with self.assertRaises(InputValidationError) as caught:
            evaluate_pd_pair(
                make_prefill_candidate(),
                make_decode_candidate(),
                make_pd_link(),
                10**10000,
                make_scenario(),
            )
        self.assertEqual(caught.exception.path, "kv_state_bytes")

    def test_pair_evaluator_requires_one_exact_scenario_metric(self):
        scenario = make_scenario(name="target")
        missing = make_prefill_candidate(scenarios=(make_scenario(name="other"),))
        duplicate = make_prefill_candidate(
            scenarios=(scenario, replace(scenario, weight=2))
        )
        for candidate, path in (
            (missing, "prefill_candidate.metrics"),
            (duplicate, "prefill_candidate.metrics[1].scenario_name"),
        ):
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    evaluate_pd_pair(
                        candidate,
                        make_decode_candidate(scenarios=(scenario,)),
                        make_pd_link(),
                        1,
                        scenario,
                    )
                self.assertEqual(caught.exception.path, path)

    def test_numeric_overflow_and_nan_use_exact_paths(self):
        prefill = make_prefill_candidate()
        decode = make_decode_candidate()
        cases = (
            (replace(make_pd_link(), bandwidth_gbps=math.nan), "pd_link.bandwidth_gbps"),
            (replace(make_pd_link(), latency_us=math.inf), "pd_link.latency_us"),
            (replace(make_pd_link(), efficiency=math.nan), "pd_link.efficiency"),
        )
        for link, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    evaluate_pd_pair(
                        prefill, decode, link, 1_000_000, make_scenario()
                    )
                self.assertEqual(caught.exception.path, path)


class PDPayloadTests(unittest.TestCase):
    def test_dense_payload_is_full_prompt_kv(self):
        model = make_dense_model()
        precision = make_w4a8_precision()
        scenario = make_scenario(input_length=17)

        payload = pd_payload_bytes(model, precision, scenario)

        self.assertEqual(
            payload, kv_bytes_per_request(model, precision, scenario.input_length)
        )

    def test_hybrid_payload_adds_terminal_recurrent_state_once(self):
        model = make_hybrid_model()
        precision = make_w4a8_precision()
        scenario = make_scenario(input_length=17)

        payload = pd_payload_bytes(model, precision, scenario)

        self.assertEqual(
            payload,
            kv_bytes_per_request(model, precision, 17)
            + recurrent_state_bytes_per_request(model),
        )

    def test_mla_payload_uses_compressed_prompt_kv(self):
        model = make_mla_moe_model()
        precision = make_w4a8_precision()
        scenario = make_scenario(input_length=19)

        self.assertEqual(
            pd_payload_bytes(model, precision, scenario),
            kv_bytes_per_request(model, precision, 19),
        )

    def test_mla_fp4_payload_preserves_half_bytes(self):
        model = make_mla_moe_model(
            num_hidden_layers=1,
            kv_lora_rank=2,
            qk_nope_head_dim=3,
            qk_rope_head_dim=1,
        )
        payload = pd_payload_bytes(
            model,
            make_w4a8_precision(kv_cache_bits=4),
            make_scenario(input_length=1),
        )

        self.assertEqual(payload, 1.5)
        self.assertIsInstance(payload, float)

    def test_huge_recurrent_state_is_a_validation_error(self):
        model = replace(
            make_hybrid_model(), linear_key_head_dim=10**10000
        )
        with self.assertRaises(InputValidationError) as caught:
            pd_payload_bytes(
                model, make_w4a8_precision(), make_scenario(input_length=1)
            )
        self.assertEqual(caught.exception.path, "payload_bytes")

    def test_kv_precision_not_activation_precision_controls_payload(self):
        model = make_dense_model()
        scenario = make_scenario(input_length=16)
        w4a4 = pd_payload_bytes(model, make_w4a4_precision(), scenario)
        w4a8 = pd_payload_bytes(model, make_w4a8_precision(), scenario)
        kv4 = pd_payload_bytes(
            model, make_w4a8_precision(kv_cache_bits=4), scenario
        )

        self.assertEqual(w4a4, w4a8)
        self.assertEqual(kv4 * 2, w4a8)

    def test_payload_validates_inputs(self):
        cases = (
            ((object(), make_w4a8_precision(), make_scenario()), "model"),
            ((make_dense_model(), object(), make_scenario()), "precision"),
            ((make_dense_model(), make_w4a8_precision(), object()), "scenario"),
        )
        for args, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    pd_payload_bytes(*args)
                self.assertEqual(caught.exception.path, path)


class PDPairSearchTests(unittest.TestCase):
    def test_pair_search_minimizes_total_cards(self):
        scenario = make_scenario()
        prefill = make_search_result(
            (
                make_prefill_candidate(
                    candidate_id="p2", total_cards=2, request_capacity=100
                ),
                make_prefill_candidate(
                    candidate_id="p4", total_cards=4, request_capacity=200
                ),
            )
        )
        decode = make_search_result(
            (
                make_decode_candidate(
                    candidate_id="d4", total_cards=4, request_capacity=100
                ),
                make_decode_candidate(
                    candidate_id="d8", total_cards=8, request_capacity=200
                ),
            )
        )

        result = pair_stage_results(
            prefill,
            decode,
            make_pd_link(),
            make_scenario_set((scenario,)),
            kv_state_bytes_by_scenario={scenario.name: 1_000_000},
        )

        self.assertEqual(result.recommendation.total_cards, 6)
        self.assertEqual(result.recommendation.prefill_candidate_id, "p2")
        self.assertEqual(result.recommendation.decode_candidate_id, "d4")

    def test_pair_ids_use_unambiguous_length_prefixed_encoding(self):
        prefill_candidates = (
            make_prefill_candidate(
                candidate_id="a::b", total_cards=1, request_capacity=50
            ),
            make_prefill_candidate(
                candidate_id="a", total_cards=2, request_capacity=100
            ),
        )
        decode_candidates = (
            make_decode_candidate(
                candidate_id="c", total_cards=1, request_capacity=50
            ),
            make_decode_candidate(
                candidate_id="b::c", total_cards=2, request_capacity=100
            ),
        )

        forward = pair_stage_results(
            make_search_result(prefill_candidates),
            make_search_result(decode_candidates),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1},
        )
        reverse = pair_stage_results(
            make_search_result(tuple(reversed(prefill_candidates))),
            make_search_result(tuple(reversed(decode_candidates))),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1},
        )
        ids = tuple(candidate.candidate_id for candidate in forward.candidates)

        self.assertEqual(len(ids), len(set(ids)))
        self.assertIn("pd:4:a::b:1:c", ids)
        self.assertIn("pd:1:a:4:b::c", ids)
        self.assertEqual(forward, reverse)

    def test_cost_is_sum_only_when_both_phase_costs_are_known(self):
        scenario_set = make_scenario_set()
        known = pair_stage_results(
            make_search_result((make_prefill_candidate(hourly_cost=2),)),
            make_search_result((make_decode_candidate(hourly_cost=3),)),
            make_pd_link(),
            scenario_set,
            {"interactive": 1_000_000},
        )
        unknown = pair_stage_results(
            make_search_result((make_prefill_candidate(hourly_cost=2),)),
            make_search_result((make_decode_candidate(hourly_cost=None),)),
            make_pd_link(),
            scenario_set,
            {"interactive": 1_000_000},
        )

        self.assertEqual(known.recommendation.hourly_cost, 5)
        self.assertIsNone(unknown.recommendation.hourly_cost)

    def test_mixed_unknown_pair_cost_sorts_unknown_after_known(self):
        prefill = make_search_result(
            (
                make_prefill_candidate(
                    candidate_id="known", hourly_cost=1, request_capacity=50
                ),
                make_prefill_candidate(
                    candidate_id="unknown",
                    hourly_cost=None,
                    request_capacity=100,
                ),
            )
        )
        result = pair_stage_results(
            prefill,
            make_search_result((make_decode_candidate(hourly_cost=1),)),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1_000_000},
        )

        self.assertEqual(result.recommendation.prefill_candidate_id, "known")

    def test_same_name_different_scenario_fields_are_rejected_without_context(self):
        stage_scenario = make_scenario(input_length=128)
        pair_scenario = make_scenario(input_length=4096)
        prefill = make_search_result(
            (make_prefill_candidate(scenarios=(stage_scenario,)),)
        )
        decode = make_search_result(
            (make_decode_candidate(scenarios=(stage_scenario,)),)
        )

        with self.assertRaises(InputValidationError) as caught:
            pair_stage_results(
                prefill,
                decode,
                make_pd_link(),
                make_scenario_set((pair_scenario,)),
                {"interactive": 1},
            )
        self.assertEqual(
            caught.exception.path,
            "prefill_result.candidates[0].scenarios",
        )

    def test_context_policy_and_scenarios_must_match_pair_context(self):
        stage_set = make_scenario_set(policy="all")
        prefill = with_context(
            make_search_result((make_prefill_candidate(),)), stage_set
        )
        decode = with_context(
            make_search_result((make_decode_candidate(),)), stage_set
        )

        with self.assertRaises(InputValidationError) as caught:
            pair_stage_results(
                prefill,
                decode,
                make_pd_link(),
                make_scenario_set(policy="weighted"),
                {"interactive": 1},
            )
        self.assertEqual(
            caught.exception.path,
            "prefill_result.context.scenario_set",
        )

    def test_pair_result_preserves_both_phase_contexts_and_validates_them(self):
        scenario_set = make_scenario_set()
        prefill = with_context(
            make_search_result((make_prefill_candidate(),)), scenario_set
        )
        decode = with_context(
            make_search_result((make_decode_candidate(),)), scenario_set
        )

        result = pair_stage_results(
            prefill,
            decode,
            make_pd_link(),
            scenario_set,
            {"interactive": 1_000_000},
        )

        self.assertIs(result.prefill_context, prefill.context)
        self.assertIs(result.decode_context, decode.context)
        context_free = replace(
            result, prefill_context=None, decode_context=None
        )
        self.assertIsNone(context_free.prefill_context)
        for changes, path in (
            ({"decode_context": None}, "decode_context"),
            ({"prefill_context": "bad"}, "prefill_context"),
        ):
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    replace(result, **changes)
                self.assertEqual(caught.exception.path, path)

    def test_phase_context_presence_model_and_precision_must_match(self):
        scenario_set = make_scenario_set()
        base_prefill = make_search_result((make_prefill_candidate(),))
        base_decode = make_search_result((make_decode_candidate(),))
        prefill = with_context(base_prefill, scenario_set)
        cases = (
            (
                base_decode,
                "decode_result.context",
            ),
            (
                with_context(
                    base_decode,
                    scenario_set,
                    model=make_dense_model(hidden_size=16, head_dim=8),
                ),
                "decode_result.context.model",
            ),
            (
                with_context(
                    base_decode,
                    scenario_set,
                    precision=make_w4a8_precision(kv_cache_bits=4),
                ),
                "decode_result.context.precision",
            ),
        )
        for decode, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    pair_stage_results(
                        prefill,
                        decode,
                        make_pd_link(),
                        scenario_set,
                        {"interactive": 1},
                    )
                self.assertEqual(caught.exception.path, path)

    def test_all_policy_rejects_slo_but_weighted_retains_warning(self):
        slow = make_scenario(
            name="slow", request_rate=1, ttft_limit_ms=10, weight=1
        )
        normal = make_scenario(
            name="normal", request_rate=1, ttft_limit_ms=100, weight=3
        )
        prefill = make_search_result(
            (make_prefill_candidate(latency_ms=20, scenarios=(slow, normal)),)
        )
        decode = make_search_result(
            (make_decode_candidate(tpot_ms=5, scenarios=(slow, normal)),)
        )
        payloads = {"slow": 1_000, "normal": 1_000}

        all_result = pair_stage_results(
            prefill,
            decode,
            make_pd_link(),
            make_scenario_set((slow, normal), policy="all"),
            payloads,
        )
        weighted_result = pair_stage_results(
            prefill,
            decode,
            make_pd_link(),
            make_scenario_set((slow, normal), policy="weighted"),
            payloads,
        )

        self.assertIsNone(all_result.recommendation)
        self.assertIsNotNone(weighted_result.recommendation)
        self.assertIn(
            "slow:TTFT_SLO", weighted_result.recommendation.warnings
        )
        self.assertAlmostEqual(weighted_result.recommendation.ttft_ms, 25.01001)

    def test_weighted_policy_keeps_transfer_concurrency_hard(self):
        scenario = make_scenario(
            request_rate=15, ttft_limit_ms=1000, weight=1
        )
        result = pair_stage_results(
            make_search_result(
                (make_prefill_candidate(scenarios=(scenario,)),)
            ),
            make_search_result(
                (make_decode_candidate(scenarios=(scenario,)),)
            ),
            make_pd_link(
                bandwidth_gbps=1000,
                latency_us=100_000,
                max_concurrent_transfers=1,
            ),
            make_scenario_set((scenario,), policy="weighted"),
            {"interactive": 1_000_000},
        )

        self.assertIsNone(result.recommendation)
        self.assertIn(
            "interactive:PD_TRANSFER_CONCURRENCY",
            result.candidates[0].reason_codes,
        )

    def test_weighted_policy_aggregates_each_system_metric(self):
        light = make_scenario(
            name="light",
            request_rate=1,
            ttft_limit_ms=100,
            tpot_limit_ms=20,
            weight=1,
        )
        heavy = make_scenario(
            name="heavy",
            request_rate=1,
            ttft_limit_ms=100,
            tpot_limit_ms=20,
            weight=3,
        )
        scenarios = (light, heavy)
        prefill = make_prefill_candidate(scenarios=scenarios)
        decode = make_decode_candidate(scenarios=scenarios)
        prefill = replace(
            prefill,
            metrics=(
                make_metrics(
                    name="light",
                    stage="prefill",
                    ttft_ms=10,
                    request_capacity=10,
                    plan=prefill.plan,
                ),
                make_metrics(
                    name="heavy",
                    stage="prefill",
                    ttft_ms=30,
                    request_capacity=100,
                    plan=prefill.plan,
                ),
            ),
        )
        decode = replace(
            decode,
            metrics=tuple(
                make_metrics(
                    name=scenario.name,
                    stage="decode",
                    tpot_ms=5,
                    request_capacity=50,
                    plan=decode.plan,
                )
                for scenario in scenarios
            ),
        )

        result = pair_stage_results(
            make_search_result((prefill,)),
            make_search_result((decode,)),
            make_pd_link(),
            make_scenario_set(scenarios, policy="weighted"),
            {"light": 1_000, "heavy": 1_000},
        )

        self.assertAlmostEqual(result.recommendation.request_capacity, 40)
        self.assertAlmostEqual(result.recommendation.ttft_ms, 30.01001)
        self.assertAlmostEqual(result.recommendation.tpot_ms, 5)

    def test_only_frontier_or_recommendation_candidates_are_paired(self):
        dominated = make_prefill_candidate(
            candidate_id="p-dominated", total_cards=2, request_capacity=50
        )
        best = make_prefill_candidate(
            candidate_id="p-best", total_cards=1, request_capacity=100
        )
        prefill = make_search_result((dominated, best))
        decode = make_search_result((make_decode_candidate(),))

        with patch(
            "infersim.search.pair.evaluate_pd_pair",
            wraps=evaluate_pd_pair,
        ) as evaluator:
            result = pair_stage_results(
                prefill,
                decode,
                make_pd_link(),
                make_scenario_set(),
                {"interactive": 1_000_000},
            )

        self.assertEqual(evaluator.call_count, 1)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(result.candidates[0].prefill_candidate_id, "p-best")

    def test_input_order_does_not_change_deterministic_output(self):
        prefill_candidates = (
            make_prefill_candidate(candidate_id="p2", total_cards=2),
            make_prefill_candidate(candidate_id="p1", total_cards=1),
        )
        decode_candidates = (
            make_decode_candidate(candidate_id="d2", total_cards=2),
            make_decode_candidate(candidate_id="d1", total_cards=1),
        )
        args = (make_pd_link(), make_scenario_set(), {"interactive": 1_000_000})

        forward = pair_stage_results(
            make_search_result(prefill_candidates),
            make_search_result(decode_candidates),
            *args,
        )
        reverse = pair_stage_results(
            make_search_result(tuple(reversed(prefill_candidates))),
            make_search_result(tuple(reversed(decode_candidates))),
            *args,
        )

        self.assertEqual(forward, reverse)
        self.assertEqual(
            tuple(candidate.candidate_id for candidate in forward.candidates),
            tuple(sorted(candidate.candidate_id for candidate in forward.candidates)),
        )

    def test_pairing_does_not_mutate_phase_results(self):
        prefill = make_search_result((make_prefill_candidate(),))
        decode = make_search_result((make_decode_candidate(),))
        original_prefill = prefill
        original_decode = decode

        pair_stage_results(
            prefill,
            decode,
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1_000_000},
        )

        self.assertEqual(prefill, original_prefill)
        self.assertEqual(decode, original_decode)
        self.assertNotIn("pd", prefill.candidates[0].metrics[0].component_seconds)

    def test_stage_scenario_and_payload_mapping_validation_paths(self):
        prefill = make_search_result((make_prefill_candidate(),))
        decode = make_search_result((make_decode_candidate(),))
        cases = (
            (
                decode,
                decode,
                make_scenario_set(),
                {"interactive": 1},
                "prefill_result.stage",
            ),
            (
                prefill,
                prefill,
                make_scenario_set(),
                {"interactive": 1},
                "decode_result.stage",
            ),
            (
                prefill,
                decode,
                make_scenario_set((make_scenario(name="missing"),)),
                {"missing": 1},
                "prefill_result.candidates[0].metrics",
            ),
            (
                prefill,
                decode,
                make_scenario_set(),
                {},
                "kv_state_bytes_by_scenario.interactive",
            ),
            (
                prefill,
                decode,
                make_scenario_set(),
                {"interactive": math.nan},
                "kv_state_bytes_by_scenario.interactive",
            ),
            (
                prefill,
                decode,
                make_scenario_set(),
                {"interactive": 0},
                "kv_state_bytes_by_scenario.interactive",
            ),
            (
                prefill,
                decode,
                make_scenario_set(),
                {"interactive": 10**10000},
                "kv_state_bytes_by_scenario.interactive",
            ),
            (
                prefill,
                decode,
                make_scenario_set(),
                {"interactive": 1, "extra": 1},
                "kv_state_bytes_by_scenario.extra",
            ),
        )
        for prefill_value, decode_value, scenarios, payloads, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    pair_stage_results(
                        prefill_value,
                        decode_value,
                        make_pd_link(),
                        scenarios,
                        payloads,
                    )
                self.assertEqual(caught.exception.path, path)

    def test_empty_pairing_still_validates_link_and_freezes_scenarios(self):
        empty_prefill = SearchResult(
            stage="prefill",
            candidates=(),
            feasible_candidates=(),
            pareto_frontier=(),
            recommendation=None,
            dominant_rejection=None,
        )
        empty_decode = SearchResult(
            stage="decode",
            candidates=(),
            feasible_candidates=(),
            pareto_frontier=(),
            recommendation=None,
            dominant_rejection=None,
        )
        scenarios = [make_scenario()]
        result = pair_stage_results(
            empty_prefill,
            empty_decode,
            make_pd_link(),
            ScenarioSet("all", scenarios),
            {"interactive": 1},
        )
        scenarios.append(make_scenario(name="later"))

        self.assertEqual(len(result.scenario_set.scenarios), 1)
        self.assertIsInstance(result.scenario_set.scenarios, tuple)
        with self.assertRaises(InputValidationError) as caught:
            pair_stage_results(
                empty_prefill,
                empty_decode,
                replace(make_pd_link(), bandwidth_gbps=math.nan),
                make_scenario_set(),
                {"interactive": 1},
            )
        self.assertEqual(caught.exception.path, "pd_link.bandwidth_gbps")

    def test_dominated_candidate_cannot_hide_scenario_mismatch(self):
        correct = make_prefill_candidate(
            candidate_id="best", request_capacity=100
        )
        mismatched = make_prefill_candidate(
            candidate_id="dominated",
            request_capacity=50,
            total_cards=2,
            scenarios=(make_scenario(name="other"),),
        )
        with self.assertRaises(InputValidationError) as caught:
            pair_stage_results(
                make_search_result((correct, mismatched)),
                make_search_result((make_decode_candidate(),)),
                make_pd_link(),
                make_scenario_set(),
                {"interactive": 1},
            )
        self.assertEqual(
            caught.exception.path,
            "prefill_result.candidates[1].metrics",
        )

    def test_pd_records_are_frozen_normalized_and_enforce_result_invariants(self):
        result = pair_stage_results(
            make_search_result((make_prefill_candidate(),)),
            make_search_result((make_decode_candidate(),)),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1_000_000},
        )
        candidate = result.candidates[0]

        self.assertIsInstance(candidate, PDCandidate)
        self.assertIsInstance(result, PDSearchResult)
        self.assertIsInstance(candidate.metrics, tuple)
        self.assertIsInstance(candidate.reason_codes, tuple)
        self.assertIsInstance(candidate.warnings, tuple)
        self.assertIsInstance(result.candidates, tuple)
        self.assertIsInstance(result.feasible_candidates, tuple)
        self.assertIsInstance(result.pareto_frontier, tuple)
        with self.assertRaises(FrozenInstanceError):
            candidate.total_cards = 1
        with self.assertRaises(InputValidationError) as caught:
            replace(candidate, total_cards=candidate.total_cards + 1)
        self.assertEqual(caught.exception.path, "total_cards")
        with self.assertRaises(InputValidationError) as caught:
            replace(result, recommendation=None)
        self.assertEqual(caught.exception.path, "recommendation")

    def test_pd_candidate_rejects_infeasible_phase_and_derived_reason_mismatch(self):
        result = pair_stage_results(
            make_search_result((make_prefill_candidate(),)),
            make_search_result((make_decode_candidate(),)),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1},
        )
        candidate = result.candidates[0]
        infeasible_prefill = replace(
            candidate.prefill_candidate,
            feasible=False,
            reason_codes=("LOCAL_FAILURE",),
        )
        with self.assertRaises(InputValidationError) as caught:
            replace(candidate, prefill_candidate=infeasible_prefill)
        self.assertEqual(caught.exception.path, "prefill_candidate.feasible")

        rejected_metric = replace(
            candidate.metrics[0],
            feasible=False,
            reason_codes=("interactive:TTFT_SLO",),
        )
        with self.assertRaises(InputValidationError) as caught:
            replace(candidate, metrics=(rejected_metric,))
        self.assertEqual(caught.exception.path, "reason_codes")

    def test_pd_candidate_rejects_feasible_phase_with_rejection_reasons(self):
        result = pair_stage_results(
            make_search_result((make_prefill_candidate(),)),
            make_search_result((make_decode_candidate(),)),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1},
        )
        candidate = result.candidates[0]
        for field, path in (
            ("prefill_candidate", "prefill_candidate.reason_codes"),
            ("decode_candidate", "decode_candidate.reason_codes"),
        ):
            phase = replace(
                getattr(candidate, field),
                feasible=True,
                reason_codes=("interactive:MEMORY_CAPACITY",),
            )
            with self.subTest(field=field):
                with self.assertRaises(InputValidationError) as caught:
                    replace(candidate, **{field: phase})
                self.assertEqual(caught.exception.path, path)

    def test_pd_candidate_warnings_are_exact_stable_phase_metric_union(self):
        prefill = replace(
            make_prefill_candidate(), warnings=("prefill", "shared")
        )
        decode = replace(
            make_decode_candidate(), warnings=("decode", "shared")
        )
        result = pair_stage_results(
            make_search_result((prefill,)),
            make_search_result((decode,)),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1},
        )
        candidate = result.candidates[0]
        self.assertEqual(
            candidate.warnings, ("prefill", "shared", "decode")
        )

        for warnings in (
            ("prefill", "decode"),
            ("decode", "shared", "prefill"),
            ("prefill", "shared", "decode", "fabricated"),
        ):
            with self.subTest(warnings=warnings):
                with self.assertRaises(InputValidationError) as caught:
                    replace(candidate, warnings=warnings)
                self.assertEqual(caught.exception.path, "warnings")

    def test_pd_result_rederives_link_scenario_and_metric_formulas(self):
        scenario = make_scenario(ttft_limit_ms=1000)
        result = pair_stage_results(
            make_search_result(
                (make_prefill_candidate(scenarios=(scenario,)),)
            ),
            make_search_result(
                (make_decode_candidate(scenarios=(scenario,)),)
            ),
            make_pd_link(),
            make_scenario_set((scenario,)),
            {"interactive": 1_000_000},
        )
        cases = (
            (
                {"pd_link": make_pd_link(bandwidth_gbps=50)},
                "candidates[0].metrics[0].transfer.effective_bandwidth_bytes_per_second",
            ),
            (
                {
                    "scenario_set": make_scenario_set(
                        (replace(scenario, input_length=4096),)
                    )
                },
                "candidates[0].prefill_candidate.scenarios",
            ),
            (
                {
                    "scenario_set": make_scenario_set(
                        (replace(scenario, name="other"),)
                    )
                },
                "candidates[0].metrics",
            ),
        )
        for changes, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    replace(result, **changes)
                self.assertEqual(caught.exception.path, path)

        candidate = result.candidates[0]
        metric = candidate.metrics[0]
        forged_transfer = replace(
            metric.transfer,
            transfer_seconds=metric.transfer.transfer_seconds + 1,
        )
        forged_transfer_metric = replace(
            metric,
            transfer=forged_transfer,
            ttft_ms=metric.ttft_ms + 1000,
        )
        forged_transfer_candidate = replace(
            candidate,
            metrics=(forged_transfer_metric,),
            ttft_ms=candidate.ttft_ms + 1000,
        )
        with self.assertRaises(InputValidationError) as caught:
            with_only_pd_candidate(result, forged_transfer_candidate)
        self.assertEqual(
            caught.exception.path,
            "candidates[0].metrics[0].transfer.transfer_seconds",
        )

        forged_metric = replace(metric, ttft_ms=metric.ttft_ms + 1)
        forged_candidate = replace(
            candidate,
            metrics=(forged_metric,),
            ttft_ms=candidate.ttft_ms + 1,
        )
        with self.assertRaises(InputValidationError) as caught:
            with_only_pd_candidate(result, forged_candidate)
        self.assertEqual(caught.exception.path, "candidates[0].metrics[0].ttft_ms")

    def test_pd_result_rederives_candidate_aggregates(self):
        result = pair_stage_results(
            make_search_result((make_prefill_candidate(),)),
            make_search_result((make_decode_candidate(),)),
            make_pd_link(),
            make_scenario_set(),
            {"interactive": 1},
        )
        candidate = result.candidates[0]
        forged_capacity = candidate.request_capacity + 1
        forged = replace(
            candidate,
            request_capacity=forged_capacity,
            request_capacity_per_card=(
                forged_capacity / candidate.total_cards
            ),
        )

        with self.assertRaises(InputValidationError) as caught:
            with_only_pd_candidate(result, forged)
        self.assertEqual(caught.exception.path, "candidates[0].request_capacity")


if __name__ == "__main__":
    unittest.main()
