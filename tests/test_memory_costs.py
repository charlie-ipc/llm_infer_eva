import unittest
from dataclasses import FrozenInstanceError, fields, replace

from infersim.cost import (
    MemoryBreakdown,
    kv_bytes_per_request,
    memory_breakdown,
    model_counts,
    recurrent_state_bytes_per_request,
)
from infersim.errors import InputValidationError
from tests.helpers import (
    make_dense_model,
    make_dense_plan,
    make_hardware,
    make_hybrid_model,
    make_mla_moe_model,
    make_moe_plan,
    make_w4a4_precision,
    make_w4a8_precision,
)


def breakdown(
    model,
    *,
    hardware=None,
    precision=None,
    plan=None,
    stage="prefill",
    batch_size=1,
    input_length=1,
    output_length=1,
):
    return memory_breakdown(
        model,
        hardware or make_hardware(),
        precision or make_w4a8_precision(),
        plan or make_dense_plan(batch_size=batch_size),
        stage=stage,
        batch_size=batch_size,
        input_length=input_length,
        output_length=output_length,
    )


class MemoryCostTests(unittest.TestCase):
    def test_w4_weights_are_exact_and_activation_uses_activation_bits(self):
        model = make_dense_model()
        w4a4 = breakdown(model, precision=make_w4a4_precision())
        w4a8 = breakdown(model, precision=make_w4a8_precision())

        expected = model_counts(model).total_weight_elements / 2
        self.assertEqual(w4a8.total_weight_bytes, expected)
        self.assertEqual(w4a4.total_weight_bytes, expected)
        self.assertEqual(w4a8.activation_bytes_per_card, 2 * w4a4.activation_bytes_per_card)

    def test_embedding_is_replicated_while_dense_attention_and_ffn_use_tp(self):
        model = make_dense_model(
            num_key_value_heads=2,
            tie_word_embeddings=False,
        )
        counts = model_counts(model)
        result = breakdown(
            model,
            plan=make_dense_plan(attention_tp=2, moe_tp=2),
        )

        self.assertEqual(result.embedding_weight_bytes, counts.embedding_weight_elements / 2)
        self.assertEqual(result.attention_weight_bytes, counts.attention_weight_elements / 2 / 2)
        self.assertEqual(result.dense_ffn_weight_bytes, counts.dense_ffn_weight_elements / 2 / 2)

    def test_linear_attention_weights_follow_attention_tp(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        )
        counts = model_counts(model)
        result = breakdown(
            model,
            plan=make_dense_plan(attention_tp=2, moe_tp=2),
        )

        self.assertEqual(result.linear_attention_weight_bytes, counts.linear_attention_weight_elements / 2 / 2)

    def test_moe_weight_components_use_their_distinct_placement_rules(self):
        model = make_mla_moe_model()
        counts = model_counts(model)
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=1,
            expert_parallel=4,
            batch_size=8,
        )

        result = breakdown(model, plan=plan, stage="decode", batch_size=8)

        self.assertEqual(result.embedding_weight_bytes, counts.embedding_weight_elements / 2)
        self.assertEqual(result.attention_weight_bytes, counts.attention_weight_elements / 2 / 2)
        self.assertEqual(result.routed_expert_weight_bytes, counts.routed_expert_weight_elements / 2 / 1 / 4)
        self.assertEqual(result.shared_expert_weight_bytes, counts.shared_expert_weight_elements / 2 / 1)

    def test_mha_kv_is_split_across_attention_tp_and_dp(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            num_routed_experts=4,
            num_experts_per_tok=2,
        )
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
            batch_size=8,
        )
        precision = make_w4a8_precision(kv_cache_bits=4)

        result = breakdown(
            model,
            precision=precision,
            plan=plan,
            stage="decode",
            batch_size=8,
            input_length=32,
            output_length=16,
        )

        full = kv_bytes_per_request(model, precision, 48)
        self.assertEqual(result.kv_bytes_per_card, full * 8 / 2 / 2)

    def test_mla_kv_is_only_split_across_attention_dp(self):
        model = make_mla_moe_model()
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=1,
            expert_parallel=4,
            batch_size=8,
        )
        precision = make_w4a8_precision(kv_cache_bits=4)

        result = breakdown(
            model,
            precision=precision,
            plan=plan,
            stage="decode",
            batch_size=8,
            input_length=32,
            output_length=16,
        )

        full = kv_bytes_per_request(model, precision, 48)
        self.assertEqual(result.kv_bytes_per_card, full * 8 / 2)

    def test_prefill_and_decode_use_their_respective_context_lengths(self):
        model = make_dense_model()
        precision = make_w4a8_precision(kv_cache_bits=4)
        prefill = breakdown(
            model,
            precision=precision,
            stage="prefill",
            batch_size=3,
            input_length=7,
            output_length=5,
        )
        decode = breakdown(
            model,
            precision=precision,
            stage="decode",
            batch_size=3,
            input_length=7,
            output_length=5,
        )

        self.assertEqual(prefill.kv_bytes_per_card, kv_bytes_per_request(model, precision, 7) * 3)
        self.assertEqual(decode.kv_bytes_per_card, kv_bytes_per_request(model, precision, 12) * 3)

    def test_hybrid_state_is_dp_split_and_tp_replicated(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            num_routed_experts=4,
            num_experts_per_tok=2,
        )
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
            batch_size=8,
        )

        result = breakdown(model, plan=plan, stage="decode", batch_size=8)

        full = recurrent_state_bytes_per_request(model)
        self.assertEqual(result.recurrent_state_bytes_per_card, full * 8 / 2)

    def test_worst_dp_rank_uses_the_same_local_request_count(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            num_routed_experts=4,
            num_experts_per_tok=2,
        )
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
            batch_size=5,
        )
        precision = make_w4a8_precision(kv_cache_bits=4)

        result = breakdown(
            model,
            precision=precision,
            plan=plan,
            batch_size=5,
            input_length=3,
        )

        local_requests = 3
        self.assertEqual(
            result.kv_bytes_per_card,
            kv_bytes_per_request(model, precision, 3)
            * local_requests
            / plan.attention_tp,
        )
        self.assertEqual(
            result.recurrent_state_bytes_per_card,
            recurrent_state_bytes_per_request(model) * local_requests,
        )
        self.assertEqual(
            result.activation_bytes_per_card,
            2
            * local_requests
            * 3
            * model.hidden_size
            * precision.activation_bits
            / 8,
        )

    def test_capacity_accounting_reports_feasible_and_infeasible_results(self):
        hardware = make_hardware(
            memory_capacity_gb=1,
            memory_reserve_fraction=0.25,
            runtime_workspace_gb=0.125,
        )
        result = breakdown(make_dense_model(), hardware=hardware)

        self.assertEqual(result.capacity_bytes, 1e9)
        self.assertEqual(result.reserved_bytes, 0.25e9)
        self.assertEqual(result.workspace_bytes, 0.125e9)
        self.assertEqual(result.usable_bytes, 0.625e9)
        self.assertEqual(
            result.resident_bytes,
            result.total_weight_bytes
            + result.kv_bytes_per_card
            + result.recurrent_state_bytes_per_card
            + result.activation_bytes_per_card,
        )
        self.assertEqual(
            result.total_required_bytes,
            result.resident_bytes + result.workspace_bytes + result.reserved_bytes,
        )
        self.assertEqual(result.capacity_margin_bytes, result.usable_bytes - result.resident_bytes)
        self.assertTrue(result.feasible)

        overcommitted = breakdown(
            make_dense_model(),
            hardware=make_hardware(
                memory_capacity_gb=1,
                memory_reserve_fraction=0.75,
                runtime_workspace_gb=1,
            ),
        )
        self.assertLess(overcommitted.usable_bytes, 0)
        self.assertFalse(overcommitted.feasible)

    def test_kv_helper_preserves_half_bytes_and_validates_context(self):
        model = make_mla_moe_model(num_hidden_layers=1)
        precision = make_w4a8_precision(kv_cache_bits=4)

        self.assertEqual(kv_bytes_per_request(model, precision, 1), 2.5)
        self.assertIsInstance(kv_bytes_per_request(model, precision, 1), float)
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                with self.assertRaises(InputValidationError) as caught:
                    kv_bytes_per_request(model, precision, value)
                self.assertEqual(caught.exception.path, "context_length")

    def test_memory_breakdown_does_not_require_compute_precision_modes(self):
        result = breakdown(
            make_dense_model(),
            hardware=make_hardware(
                compute_tflops={
                    "gemm": {"w4a4": 1},
                    "vector": {"fp4": 1},
                }
            ),
        )

        self.assertIsInstance(result, MemoryBreakdown)

    def test_validates_precision_bits_plan_stage_and_dimensions_with_full_paths(self):
        model = make_dense_model()
        invalid_plan = make_dense_plan(attention_tp=3, moe_tp=3)
        base = {
            "stage": "prefill",
            "batch_size": 1,
            "input_length": 1,
            "output_length": 1,
        }

        for field in ("weight_bits", "activation_bits", "kv_cache_bits"):
            with self.subTest(field=field):
                precision = replace(make_w4a8_precision(), **{field: 3})
                with self.assertRaises(InputValidationError) as caught:
                    breakdown(model, precision=precision)
                self.assertEqual(caught.exception.path, field)

        with self.assertRaises(InputValidationError) as caught:
            memory_breakdown(
                model,
                make_hardware(),
                make_w4a8_precision(),
                invalid_plan,
                **base,
            )
        self.assertEqual(caught.exception.path, "plan")
        self.assertIn("ATTENTION_HEADS_NOT_DIVISIBLE", caught.exception.message)

        for field, value in (
            ("stage", "train"),
            ("batch_size", True),
            ("batch_size", 0),
            ("input_length", -1),
            ("output_length", 1.5),
        ):
            with self.subTest(field=field, value=value):
                kwargs = dict(base)
                kwargs[field] = value
                with self.assertRaises(InputValidationError) as caught:
                    memory_breakdown(
                        model,
                        make_hardware(),
                        make_w4a8_precision(),
                        make_dense_plan(),
                        **kwargs,
                    )
                self.assertEqual(caught.exception.path, field)

    def test_huge_integer_overflow_is_a_validation_error(self):
        with self.assertRaises(InputValidationError) as caught:
            kv_bytes_per_request(
                make_dense_model(),
                make_w4a8_precision(),
                10**1000,
            )
        self.assertEqual(caught.exception.path, "kv_bytes_per_request")

        with self.assertRaises(InputValidationError) as caught:
            breakdown(
                make_dense_model(),
                stage="decode",
                batch_size=10**1000,
            )
        self.assertIn("bytes", caught.exception.path)

    def test_breakdown_is_frozen_and_all_byte_quantities_are_floats(self):
        result = breakdown(make_dense_model())

        self.assertIsInstance(result, MemoryBreakdown)
        with self.assertRaises(FrozenInstanceError):
            result.feasible = False
        for field in fields(result):
            if field.name.endswith("bytes"):
                self.assertIsInstance(getattr(result, field.name), float)


if __name__ == "__main__":
    unittest.main()
