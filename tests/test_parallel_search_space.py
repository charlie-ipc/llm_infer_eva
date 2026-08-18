import unittest

from infersim.schema.model import ModelSpec
from infersim.schema.parallel import (
    ParallelPlan,
    PlanValidation,
    SearchSpace,
)
from infersim.search import enumerate_plans, validate_plan
from tests.helpers import make_hybrid_model


def dense_model(**overrides):
    config = {
        "model_type": "test-dense",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "vocab_size": 32000,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "intermediate_size": 3072,
    }
    config.update(overrides)
    return ModelSpec.from_dict(config)


def moe_model(**overrides):
    config = {
        "model_type": "test-moe",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "vocab_size": 32000,
        "num_attention_heads": 12,
        "num_key_value_heads": 12,
        "num_routed_experts": 8,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 3072,
    }
    config.update(overrides)
    return ModelSpec.from_dict(config)


class ParallelPlanTests(unittest.TestCase):
    def test_card_properties_and_value_semantics(self):
        plan = ParallelPlan(2, 2, 2, 1, 4, 16)

        self.assertEqual(plan.cards_per_replica, 4)
        self.assertEqual(plan.total_cards, 8)
        self.assertEqual(
            sorted([plan, ParallelPlan(1, 2, 2, 1, 4, 16)]),
            [ParallelPlan(1, 2, 2, 1, 4, 16), plan],
        )

    def test_constructor_preserves_invalid_candidates_for_validation(self):
        plan = ParallelPlan(0, True, -1, 1, 1, 0)

        self.assertEqual(plan.replicas, 0)
        self.assertIs(plan.attention_tp, True)


class PlanValidationTests(unittest.TestCase):
    def assert_reason(self, model, plan, reason_code, selected_total_cards=None):
        result = validate_plan(model, plan, selected_total_cards)

        self.assertIsInstance(result, PlanValidation)
        self.assertIs(result.plan, plan)
        self.assertFalse(result.feasible)
        self.assertEqual(result.reason_code, reason_code)
        self.assertIsInstance(result.reason, str)
        self.assertTrue(result.reason)
        self.assertEqual(
            result.reason,
            validate_plan(model, plan, selected_total_cards).reason,
        )

    def test_accepts_dense_exact_parallel_semantics(self):
        plan = ParallelPlan(2, 2, 1, 2, 1, 8)

        result = validate_plan(dense_model(), plan, selected_total_cards=4)

        self.assertEqual(result, PlanValidation(plan=plan, feasible=True))

    def test_accepts_moe_width_equation(self):
        plan = ParallelPlan(1, 2, 2, 1, 4, 8)

        result = validate_plan(moe_model(), plan, selected_total_cards=4)

        self.assertTrue(result.feasible)
        self.assertIsNone(result.reason_code)
        self.assertIsNone(result.reason)

    def test_nonpositive_parallelism_has_first_priority_and_rejects_bool(self):
        plan = ParallelPlan(0, True, 1, 1, 1, 1)
        self.assert_reason(
            dense_model(), plan, "NONPOSITIVE_PARALLELISM", -1
        )

    def test_selected_total_cards_must_be_positive_integer_and_match(self):
        model = dense_model()
        plan = ParallelPlan(1, 1, 1, 1, 1, 8)
        for selected_total_cards in (0, True, 2):
            with self.subTest(selected_total_cards=selected_total_cards):
                self.assert_reason(
                    model,
                    plan,
                    "TOTAL_CARDS_MISMATCH",
                    selected_total_cards,
                )

    def test_dense_parallelism_rule_precedes_divisibility(self):
        plan = ParallelPlan(1, 1, 2, 1, 2, 8)
        self.assert_reason(
            dense_model(intermediate_size=3073),
            plan,
            "DENSE_PARALLELISM_INVALID",
        )

    def test_moe_width_mismatch_precedes_head_divisibility(self):
        plan = ParallelPlan(1, 5, 1, 2, 2, 8)
        self.assert_reason(
            moe_model(), plan, "MOE_WIDTH_MISMATCH"
        )

    def test_attention_heads_not_divisible_precedes_kv_heads(self):
        plan = ParallelPlan(1, 5, 1, 5, 1, 8)
        self.assert_reason(
            moe_model(num_attention_heads=12, num_key_value_heads=3),
            plan,
            "ATTENTION_HEADS_NOT_DIVISIBLE",
        )

    def test_kv_heads_not_divisible_after_attention_heads_pass(self):
        plan = ParallelPlan(1, 4, 1, 4, 1, 8)
        self.assert_reason(
            moe_model(num_attention_heads=12, num_key_value_heads=3),
            plan,
            "KV_HEADS_NOT_DIVISIBLE",
        )

    def test_linear_attention_key_heads_must_divide_attention_tp(self):
        plan = ParallelPlan(1, 2, 1, 2, 1, 8)

        result = validate_plan(
            make_hybrid_model(
                num_attention_heads=4,
                num_key_value_heads=4,
                linear_num_key_heads=3,
                linear_num_value_heads=4,
            ),
            plan,
        )

        self.assertEqual(result.reason_code, "LINEAR_KEY_HEADS_NOT_DIVISIBLE")
        self.assertEqual(
            result.reason,
            "linear attention key heads must be divisible by attention_tp",
        )

    def test_linear_attention_value_heads_follow_key_heads_in_priority(self):
        plan = ParallelPlan(1, 2, 1, 2, 1, 8)

        result = validate_plan(
            make_hybrid_model(
                num_attention_heads=4,
                num_key_value_heads=4,
                linear_num_key_heads=4,
                linear_num_value_heads=3,
            ),
            plan,
        )

        self.assertEqual(
            result.reason_code, "LINEAR_VALUE_HEADS_NOT_DIVISIBLE"
        )
        self.assertEqual(
            result.reason,
            "linear attention value heads must be divisible by attention_tp",
        )

    def test_linear_head_dimensions_are_not_tensor_parallel_axes(self):
        plan = ParallelPlan(1, 2, 1, 2, 1, 8)
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_num_key_heads=4,
            linear_key_head_dim=3,
            linear_num_value_heads=4,
            linear_value_head_dim=5,
        )

        self.assertTrue(validate_plan(model, plan).feasible)

    def test_mla_skips_kv_head_divisibility(self):
        model = moe_model(
            num_key_value_heads=3,
            kv_lora_rank=64,
            qk_nope_head_dim=64,
            qk_rope_head_dim=32,
            v_head_dim=64,
        )
        plan = ParallelPlan(1, 4, 1, 4, 1, 8)

        self.assertTrue(validate_plan(model, plan).feasible)

    def test_dense_intermediate_uses_attention_tp(self):
        plan = ParallelPlan(1, 4, 1, 4, 1, 8)
        self.assert_reason(
            dense_model(intermediate_size=3073),
            plan,
            "INTERMEDIATE_NOT_DIVISIBLE",
        )

    def test_moe_intermediate_uses_moe_tp(self):
        plan = ParallelPlan(1, 2, 2, 4, 1, 8)
        self.assert_reason(
            moe_model(moe_intermediate_size=3074),
            plan,
            "INTERMEDIATE_NOT_DIVISIBLE",
        )

    def test_experts_not_divisible_is_last_rule(self):
        plan = ParallelPlan(1, 2, 3, 2, 3, 8)
        self.assert_reason(
            moe_model(num_routed_experts=8),
            plan,
            "EXPERTS_NOT_DIVISIBLE",
        )

    def test_shared_intermediate_divisibility_precedes_expert_count(self):
        plan = ParallelPlan(1, 2, 2, 2, 2, 8)
        self.assert_reason(
            moe_model(
                num_routed_experts=3,
                num_shared_experts=1,
                shared_expert_intermediate_size=5,
            ),
            plan,
            "SHARED_INTERMEDIATE_NOT_DIVISIBLE",
        )


class EnumerationTests(unittest.TestCase):
    def test_enumerates_sorted_deduplicated_full_axis_diagnostics(self):
        search_space = SearchSpace(
            total_cards=(2, 1, 2),
            replicas=(1, 1),
            attention_tp=(2, 1, 2),
            attention_dp=(1,),
            moe_tp=(1, 2, 1),
            expert_parallel=(1,),
            batch_sizes=(16, 8, 16),
        )

        first = list(enumerate_plans(dense_model(), search_space))
        second = list(enumerate_plans(dense_model(), search_space))

        expected = []
        for selected_total_cards in (1, 2):
            for replicas in (1,):
                for attention_tp in (1, 2):
                    for attention_dp in (1,):
                        for moe_tp in (1, 2):
                            for expert_parallel in (1,):
                                for batch_size in (8, 16):
                                    plan = ParallelPlan(
                                        replicas,
                                        attention_tp,
                                        attention_dp,
                                        moe_tp,
                                        expert_parallel,
                                        batch_size,
                                    )
                                    expected.append(
                                        validate_plan(
                                            dense_model(),
                                            plan,
                                            selected_total_cards,
                                        )
                                    )

        self.assertEqual(first, expected)
        self.assertEqual(second, first)
        self.assertEqual(len(first), 16)
        self.assertTrue(any(item.feasible for item in first))
        self.assertTrue(any(not item.feasible for item in first))

    def test_same_plan_keeps_matching_and_mismatching_total_card_candidates(self):
        search_space = SearchSpace(
            total_cards=(2, 1),
            replicas=(1,),
            attention_tp=(1,),
            attention_dp=(1,),
            moe_tp=(1,),
            expert_parallel=(1,),
            batch_sizes=(8,),
        )

        results = list(enumerate_plans(dense_model(), search_space))

        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].feasible)
        self.assertEqual(results[1].reason_code, "TOTAL_CARDS_MISMATCH")

    def test_invalid_manual_axis_candidate_returns_diagnostic(self):
        search_space = SearchSpace(
            total_cards=(1,),
            replicas=(0,),
            attention_tp=(1,),
            attention_dp=(1,),
            moe_tp=(1,),
            expert_parallel=(1,),
            batch_sizes=(8,),
        )

        results = list(enumerate_plans(dense_model(), search_space))

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].reason_code, "NONPOSITIVE_PARALLELISM")

    def test_hybrid_enumeration_keeps_valid_and_invalid_linear_tp_plans(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_num_key_heads=3,
            linear_num_value_heads=4,
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

        results = list(enumerate_plans(model, search_space))

        self.assertTrue(
            any(
                item.feasible and item.plan.attention_tp == 1
                for item in results
            )
        )
        self.assertTrue(
            any(
                item.reason_code == "LINEAR_KEY_HEADS_NOT_DIVISIBLE"
                and item.plan.attention_tp == 2
                and item.plan.moe_tp == 2
                for item in results
            )
        )


if __name__ == "__main__":
    unittest.main()
