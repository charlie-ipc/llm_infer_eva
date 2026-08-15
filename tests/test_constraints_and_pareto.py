import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace

from infersim.errors import InputValidationError
from infersim.schema.scenario import ScenarioSet
from infersim.search import (
    StageCandidate,
    evaluate_stage_constraints,
    pareto_frontier,
    recommend,
    recommendation_sort_key,
)
from tests.helpers import (
    make_candidate,
    make_dense_plan,
    make_metrics,
    make_scenario,
    make_stage_candidate,
)


class StageCandidateTests(unittest.TestCase):
    def test_public_record_has_exact_fields_and_is_deeply_immutable(self):
        candidate = make_candidate(
            metrics=[make_metrics()],
            reason_codes=["PLAN"],
            warnings=["note"],
            scenarios=[make_scenario()],
        )

        self.assertEqual(
            tuple(field.name for field in fields(StageCandidate)),
            (
                "candidate_id",
                "plan",
                "metrics",
                "feasible",
                "reason_codes",
                "warnings",
                "total_cards",
                "hourly_cost",
                "request_capacity",
                "request_capacity_per_card",
                "ttft_ms",
                "tpot_ms",
                "scenarios",
            ),
        )
        self.assertIsInstance(candidate.metrics, tuple)
        self.assertIsInstance(candidate.reason_codes, tuple)
        self.assertIsInstance(candidate.warnings, tuple)
        self.assertIsInstance(candidate.scenarios, tuple)
        with self.assertRaises(FrozenInstanceError):
            candidate.candidate_id = "changed"

    def test_accepts_zero_raw_summary_placeholders(self):
        candidate = make_candidate()
        self.assertEqual(candidate.request_capacity, 0.0)
        self.assertEqual(candidate.request_capacity_per_card, 0.0)

    def test_rejects_invalid_identity_types_numbers_and_card_count(self):
        invalid = (
            ("candidate_id", "", "candidate_id"),
            ("plan", object(), "plan"),
            ("metrics", [object()], "metrics[0]"),
            ("feasible", 1, "feasible"),
            ("reason_codes", [1], "reason_codes[0]"),
            ("warnings", [1], "warnings[0]"),
            ("total_cards", 2, "total_cards"),
            ("hourly_cost", math.inf, "hourly_cost"),
            ("request_capacity", -1, "request_capacity"),
            ("request_capacity_per_card", math.nan, "request_capacity_per_card"),
            ("ttft_ms", -1, "ttft_ms"),
            ("tpot_ms", math.inf, "tpot_ms"),
            ("scenarios", [object()], "scenarios[0]"),
        )
        for field, value, path in invalid:
            with self.subTest(field=field):
                with self.assertRaises(InputValidationError) as caught:
                    replace(make_candidate(), **{field: value})
                self.assertEqual(caught.exception.path, path)


class ConstraintTests(unittest.TestCase):
    def test_all_policy_makes_long_prefill_ttft_infeasible(self):
        scenario = make_scenario(name="long", ttft_limit_ms=500)
        candidate = make_candidate(
            metrics=[make_metrics(name="long", ttft_ms=600)],
            scenarios=[scenario],
        )

        result = evaluate_stage_constraints(candidate, "all")

        self.assertFalse(result.feasible)
        self.assertEqual(result.reason_codes, ("long:TTFT_SLO",))
        self.assertEqual(result.ttft_ms, 600)
        self.assertIsNone(result.tpot_ms)

    def test_all_policy_reports_codes_in_metric_and_rule_order(self):
        first = make_metrics(
            name="first",
            ttft_ms=120,
            request_capacity=2,
            memory_feasible=False,
            max_supported_concurrency=2,
        )
        second = make_metrics(name="second", ttft_ms=20, request_capacity=20)
        scenarios = (
            make_scenario(
                name="second", request_rate=10, concurrency=1, ttft_limit_ms=50
            ),
            make_scenario(
                name="first", request_rate=3, concurrency=3, ttft_limit_ms=100
            ),
        )
        candidate = make_candidate(
            metrics=[first, second],
            scenarios=scenarios,
            reason_codes=("PLAN_RULE", "PLAN_RULE"),
            warnings=("existing", "existing"),
        )

        result = evaluate_stage_constraints(candidate, "all")

        self.assertEqual(
            result.reason_codes,
            (
                "PLAN_RULE",
                "first:MEMORY_CAPACITY",
                "first:TTFT_SLO",
                "first:REQUEST_RATE",
                "first:CONCURRENCY",
            ),
        )
        self.assertEqual(result.warnings, ("existing",))
        self.assertEqual(result.request_capacity, 2)
        self.assertEqual(result.request_capacity_per_card, 2)

    def test_concurrency_falls_back_to_replicas_times_batch_size(self):
        plan = make_dense_plan(replicas=2, batch_size=3)
        metric = make_metrics(plan=plan, max_supported_concurrency=None)
        candidate = make_candidate(
            plan=plan,
            metrics=[metric],
            scenarios=[make_scenario(concurrency=7)],
        )

        result = evaluate_stage_constraints(candidate, "all")

        self.assertIn("interactive:CONCURRENCY", result.reason_codes)

    def test_decode_uses_tpot_limit_and_summary(self):
        metric = make_metrics(stage="decode", tpot_ms=25)
        candidate = make_candidate(
            metrics=[metric],
            scenarios=[make_scenario(tpot_limit_ms=20)],
        )

        result = evaluate_stage_constraints(candidate, "all")

        self.assertEqual(result.reason_codes, ("interactive:TPOT_SLO",))
        self.assertEqual(result.tpot_ms, 25)
        self.assertIsNone(result.ttft_ms)

    def test_weighted_policy_keeps_memory_and_plan_hard_but_warns_for_slos(self):
        metric = make_metrics(
            ttft_ms=120,
            request_capacity=1,
            memory_feasible=False,
            max_supported_concurrency=1,
        )
        scenario = make_scenario(
            ttft_limit_ms=100, request_rate=2, concurrency=2
        )
        candidate = make_candidate(
            metrics=[metric],
            scenarios=[scenario],
            reason_codes=("PLAN_RULE",),
            warnings=("existing",),
        )

        result = evaluate_stage_constraints(candidate, "weighted")

        self.assertFalse(result.feasible)
        self.assertEqual(
            result.reason_codes,
            ("PLAN_RULE", "interactive:MEMORY_CAPACITY"),
        )
        self.assertEqual(
            result.warnings,
            (
                "existing",
                "interactive:TTFT_SLO",
                "interactive:REQUEST_RATE",
                "interactive:CONCURRENCY",
            ),
        )

    def test_weighted_summary_normalizes_positive_weights(self):
        metrics = (
            make_metrics(name="small", ttft_ms=10, request_capacity=20),
            make_metrics(name="large", ttft_ms=30, request_capacity=80),
        )
        scenarios = (
            make_scenario(name="small", weight=1, request_rate=1),
            make_scenario(name="large", weight=3, request_rate=1),
        )

        result = evaluate_stage_constraints(
            make_candidate(metrics=metrics, scenarios=scenarios), "weighted"
        )

        self.assertTrue(result.feasible)
        self.assertEqual(result.request_capacity, 65)
        self.assertEqual(result.ttft_ms, 25)

    def test_scenario_set_supplies_scenarios_and_requires_matching_policy(self):
        candidate = make_candidate(scenarios=())
        scenarios = ScenarioSet("all", (make_scenario(),))

        result = evaluate_stage_constraints(candidate, "all", scenarios)
        self.assertTrue(result.feasible)
        with self.assertRaises(InputValidationError) as caught:
            evaluate_stage_constraints(candidate, "weighted", scenarios)
        self.assertEqual(caught.exception.path, "policy")

    def test_rejects_mismatched_duplicate_and_mixed_stage_inputs(self):
        cases = (
            (
                make_candidate(scenarios=()),
                "scenarios",
            ),
            (
                make_candidate(
                    metrics=[make_metrics(name="a")],
                    scenarios=[make_scenario(name="b")],
                ),
                "scenarios",
            ),
            (
                make_candidate(
                    metrics=[make_metrics(name="a"), make_metrics(name="a")],
                    scenarios=[make_scenario(name="a"), make_scenario(name="b")],
                ),
                "metrics[1].scenario_name",
            ),
            (
                make_candidate(
                    metrics=[
                        make_metrics(name="a", stage="prefill"),
                        make_metrics(name="b", stage="decode"),
                    ],
                    scenarios=[make_scenario(name="a"), make_scenario(name="b")],
                ),
                "metrics",
            ),
        )
        for candidate, path in cases:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    evaluate_stage_constraints(candidate, "all")
                self.assertEqual(caught.exception.path, path)

    def test_rejects_bad_policy_candidate_state_and_nonpositive_weight(self):
        invalid_calls = (
            (lambda: evaluate_stage_constraints(object(), "all"), "candidate"),
            (lambda: evaluate_stage_constraints(make_candidate(), "some"), "policy"),
            (
                lambda: evaluate_stage_constraints(
                    make_candidate(feasible=False, reason_codes=()), "all"
                ),
                "candidate.feasible",
            ),
            (
                lambda: evaluate_stage_constraints(
                    make_candidate(scenarios=[make_scenario(weight=0)]),
                    "weighted",
                ),
                "scenarios[0].weight",
            ),
        )
        for call, path in invalid_calls:
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    call()
                self.assertEqual(caught.exception.path, path)


class RecommendationTests(unittest.TestCase):
    def candidate(
        self,
        candidate_id,
        *,
        cards=1,
        capacity=10,
        cost=None,
        ttft=10,
        tpot=None,
        feasible=True,
        plan_overrides=None,
    ):
        plan_overrides = dict(plan_overrides or {})
        plan = make_dense_plan(replicas=cards, **plan_overrides)
        return make_stage_candidate(
            candidate_id=candidate_id,
            plan=plan,
            metrics=[make_metrics(plan=plan)],
            feasible=feasible,
            reason_codes=() if feasible else ("FAILED",),
            hourly_cost=cost,
            request_capacity=capacity,
            request_capacity_per_card=capacity / cards,
            ttft_ms=ttft,
            tpot_ms=tpot,
        )

    def test_recommend_prefers_cards_before_throughput_and_excludes_infeasible(self):
        fast_large = self.candidate("fast-large", cards=8, capacity=100)
        small = self.candidate("small", cards=4, capacity=10)
        failed = self.candidate("failed", cards=1, capacity=1000, feasible=False)

        self.assertIs(recommend([failed, fast_large, small]), small)
        self.assertIsNone(recommend([failed]))

    def test_known_cost_is_used_and_unknown_cost_sorts_last(self):
        expensive = self.candidate("expensive", cost=5, capacity=100)
        cheap = self.candidate("cheap", cost=2, capacity=10)
        unknown = self.candidate("unknown", cost=None, capacity=1000)

        self.assertIs(recommend([expensive, unknown, cheap]), cheap)

    def test_all_unknown_cost_omits_cost_and_uses_throughput_per_card(self):
        low = self.candidate("low", capacity=10)
        high = self.candidate("high", capacity=20)
        key = recommendation_sort_key([low, high])

        self.assertIs(recommend([low, high]), high)
        self.assertLess(key(high), key(low))

    def test_plan_dimensions_break_ties_without_candidate_id(self):
        first = self.candidate("z", plan_overrides={"attention_tp": 2})
        second = self.candidate("a", plan_overrides={"attention_tp": 2})

        self.assertIs(recommend([first, second]), first)

    def test_pareto_removes_dominated_candidates_and_sorts_deterministically(self):
        best = self.candidate("best", cards=4, capacity=100, ttft=10)
        dominated = self.candidate("dominated", cards=8, capacity=90, ttft=20)
        tradeoff = self.candidate("tradeoff", cards=2, capacity=20, ttft=30)
        failed = self.candidate("failed", cards=1, capacity=1000, feasible=False)

        result = pareto_frontier([dominated, failed, best, best, tradeoff])

        self.assertEqual(result, [tradeoff, best])

    def test_pareto_ignores_optional_objectives_pairwise_when_missing(self):
        missing = self.candidate(
            "missing", cards=2, capacity=50, cost=None, ttft=None
        )
        known = self.candidate("known", cards=2, capacity=50, cost=1, ttft=10)

        self.assertEqual(pareto_frontier([known, missing]), [known, missing])

    def test_pareto_uses_cost_and_stage_latency_when_both_are_known(self):
        better = self.candidate("better", cost=1, ttft=10, capacity=20)
        worse = self.candidate("worse", cost=2, ttft=20, capacity=10)

        self.assertEqual(pareto_frontier([worse, better]), [better])


if __name__ == "__main__":
    unittest.main()
