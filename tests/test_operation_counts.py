from dataclasses import FrozenInstanceError
import unittest

from infersim.cost.operations import (
    GemmShape,
    StageOperations,
    VectorShape,
    kv_elements_per_token,
    model_counts,
    recurrent_state_bytes,
    recurrent_state_bytes_per_request,
    stage_operations,
)
from infersim.cost import __all__ as cost_exports
from infersim.errors import InputValidationError
from tests.helpers import (
    make_dense_model,
    make_dense_plan,
    make_hybrid_model,
    make_mla_moe_model,
    make_moe_plan,
)


def by_name(shapes):
    return {shape.name: shape for shape in shapes}


def gemm_useful_ops(shape):
    return (
        2
        * shape.m
        * shape.k
        * shape.n
        * shape.batch_repeats
        * shape.repeats
    )


class DescriptorTests(unittest.TestCase):
    def test_cost_package_preserves_kernel_exports(self):
        self.assertTrue(
            {
                "KernelCost",
                "gemm_cost",
                "kernel_cost",
                "vector_cost",
                "vector_mode_for_bits",
            }.issubset(cost_exports)
        )

    def test_descriptors_are_frozen_and_normalize_integral_dimensions(self):
        gemm = GemmShape(
            "q", 2.0, 8, 4, repeats=3.0, batch_repeats=4.0
        )
        vector = VectorShape("rope", 16.0, 6, repeats=2.0)

        self.assertEqual(
            (gemm.m, gemm.repeats, gemm.batch_repeats), (2, 3, 4)
        )
        self.assertEqual((vector.elements, vector.repeats), (16, 2))
        with self.assertRaises(FrozenInstanceError):
            gemm.m = 3

    def test_descriptors_reject_invalid_names_dimensions_and_collections(self):
        invalid_factories = (
            lambda: GemmShape("", 1, 1, 1),
            lambda: GemmShape("q", 0, 1, 1),
            lambda: GemmShape("q", True, 1, 1),
            lambda: GemmShape("q", 1.5, 1, 1),
            lambda: GemmShape("q", 1, 1, 1, batch_repeats=0),
            lambda: GemmShape("q", 1, 1, 1, batch_repeats=True),
            lambda: VectorShape("v", 1, 0),
            lambda: StageOperations(gemms=[], vectors=()),
            lambda: StageOperations(gemms=("not-a-shape",), vectors=()),
        )

        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(InputValidationError):
                    factory()


class ModelCountTests(unittest.TestCase):
    def test_tiny_dense_model_has_exact_unsharded_count(self):
        counts = model_counts(make_dense_model())

        self.assertEqual(counts.embedding_weight_elements, 256)
        self.assertEqual(counts.attention_weight_elements, 384)
        self.assertEqual(counts.linear_attention_weight_elements, 0)
        self.assertEqual(counts.dense_ffn_weight_elements, 768)
        self.assertEqual(counts.routed_expert_weight_elements, 0)
        self.assertEqual(counts.shared_expert_weight_elements, 0)
        self.assertEqual(counts.total_weight_elements, 1408)
        with self.assertRaises(FrozenInstanceError):
            counts.embedding_weight_elements = 0

    def test_untied_embeddings_add_one_vocab_by_hidden_matrix(self):
        tied = model_counts(make_dense_model())
        untied = model_counts(make_dense_model(tie_word_embeddings=False))

        self.assertEqual(
            untied.total_weight_elements - tied.total_weight_elements,
            32 * 8,
        )

    def test_attention_gate_doubles_only_q_projection_weights(self):
        plain = model_counts(make_dense_model())
        gated = model_counts(make_dense_model(attention_output_gate=True))

        self.assertEqual(
            gated.attention_weight_elements
            - plain.attention_weight_elements,
            2 * 8 * 2 * 4,
        )

    def test_mha_kv_elements_per_token_uses_kv_heads_and_full_layers(self):
        model = make_dense_model(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
        )

        self.assertEqual(kv_elements_per_token(model), 2 * 2 * 2 * 2)

    def test_mla_counts_and_kv_elements_use_lora_dimensions(self):
        model = make_mla_moe_model()
        counts = model_counts(model)

        per_layer_attention = (
            16 * 4
            + 4 * 4 * 4
            + 16 * (3 + 2)
            + 3 * 4 * (2 + 3)
            + 16 * 4 * 3
        )
        self.assertEqual(counts.attention_weight_elements, 2 * per_layer_attention)
        self.assertEqual(counts.routed_expert_weight_elements, 2 * 4 * 3 * 16 * 8)
        self.assertEqual(counts.shared_expert_weight_elements, 2 * 2 * 3 * 16 * 6)
        self.assertEqual(kv_elements_per_token(model), 2 * (3 + 2))

    def test_hybrid_linear_weights_use_all_projection_and_conv_terms(self):
        model = make_hybrid_model()
        counts = model_counts(model)
        key_dim = 2 * 2
        value_dim = 2 * 3
        per_layer = (
            2 * 12 * key_dim
            + 2 * 12 * value_dim
            + 2 * 12 * 2
            + (2 * key_dim + value_dim) * 3
            + value_dim * 12
        )

        self.assertEqual(counts.linear_attention_weight_elements, 2 * per_layer)


class RecurrentStateTests(unittest.TestCase):
    def test_dense_model_has_no_recurrent_state(self):
        model = make_dense_model()

        self.assertEqual(recurrent_state_bytes_per_request(model), 0)
        self.assertEqual(recurrent_state_bytes(model), 0)

    def test_hybrid_recurrent_state_matches_fixed_storage_formats(self):
        model = make_hybrid_model()
        conv_elements = (2 * 3 + 2 * 2 * 2) * (3 - 1)
        ssm_elements = 2 * 2 * 3
        expected_per_layer = conv_elements * 2 + ssm_elements * 4

        self.assertEqual(recurrent_state_bytes_per_request(model), 2 * expected_per_layer)
        self.assertEqual(recurrent_state_bytes(model), 2 * expected_per_layer)


class DenseStageOperationTests(unittest.TestCase):
    def test_prefill_and_decode_use_expected_projection_m(self):
        model = make_dense_model()
        plan = make_dense_plan()

        prefill = stage_operations(
            model,
            stage="prefill",
            batch_size=3,
            input_length=5,
            average_context=5,
            plan=plan,
        )
        decode = stage_operations(
            model,
            stage="decode",
            batch_size=3,
            input_length=5,
            average_context=7.0,
            plan=plan,
        )

        self.assertEqual(prefill.gemms[0].m, 15)
        self.assertEqual(decode.gemms[0].m, 3)
        self.assertTrue(
            all(
                type(value) is int
                for shape in decode.gemms
                for value in (
                    shape.m,
                    shape.k,
                    shape.n,
                    shape.repeats,
                    shape.batch_repeats,
                )
            )
        )

    def test_mha_projection_and_core_shapes_are_sharded_by_attention_tp(self):
        model = make_dense_model(
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=2,
            intermediate_size=16,
        )
        plan = make_dense_plan(attention_tp=2, moe_tp=2)
        ops = stage_operations(
            model,
            stage="decode",
            batch_size=3,
            input_length=8,
            average_context=11,
            plan=plan,
        )
        gemms = by_name(ops.gemms)

        self.assertEqual(gemms["attention.q_proj"], GemmShape("attention.q_proj", 3, 8, 4, 2))
        self.assertEqual(gemms["attention.k_proj"], GemmShape("attention.k_proj", 3, 8, 2, 2))
        self.assertEqual(gemms["attention.v_proj"], GemmShape("attention.v_proj", 3, 8, 2, 2))
        self.assertEqual(gemms["attention.o_proj"], GemmShape("attention.o_proj", 3, 4, 8, 2))
        self.assertEqual(
            gemms["attention.qk"],
            GemmShape(
                "attention.qk", 3, 2, 11, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            gemms["attention.pv"],
            GemmShape(
                "attention.pv", 3, 11, 2, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            gemm_useful_ops(gemms["attention.qk"]),
            2 * 3 * 4 * 11 * 2,
        )
        self.assertEqual(
            gemm_useful_ops(gemms["attention.pv"]),
            2 * 3 * 11 * 4 * 2,
        )

    def test_attention_gate_changes_q_shape_and_adds_gate_vector(self):
        model = make_dense_model(attention_output_gate=True)
        ops = stage_operations(
            model,
            stage="decode",
            batch_size=3,
            input_length=8,
            average_context=11,
            plan=make_dense_plan(),
        )
        gemms = by_name(ops.gemms)
        vectors = by_name(ops.vectors)

        self.assertEqual(gemms["attention.q_proj"].n, 2 * 2 * 4)
        self.assertEqual(gemms["attention.qk"].k, 4)
        self.assertEqual(gemms["attention.qk"].batch_repeats, 2)
        self.assertEqual(vectors["attention.output_gate"], VectorShape("attention.output_gate", 3 * 2 * 4, 6, 2))

    def test_dense_ffn_and_common_vectors_have_expected_constants(self):
        ops = stage_operations(
            make_dense_model(),
            stage="prefill",
            batch_size=2,
            input_length=3,
            average_context=3,
            plan=make_dense_plan(),
        )
        gemms = by_name(ops.gemms)
        vectors = by_name(ops.vectors)

        self.assertEqual(gemms["ffn.gate_proj"], GemmShape("ffn.gate_proj", 6, 8, 16, 2))
        self.assertEqual(gemms["ffn.up_proj"], GemmShape("ffn.up_proj", 6, 8, 16, 2))
        self.assertEqual(gemms["ffn.down_proj"], GemmShape("ffn.down_proj", 6, 16, 8, 2))
        self.assertEqual(vectors["ffn.silu_gate"], VectorShape("ffn.silu_gate", 6 * 16, 6, 2))
        self.assertEqual(vectors["norm.input"], VectorShape("norm.input", 6 * 8, 5, 2))
        self.assertEqual(vectors["norm.post_attention"], VectorShape("norm.post_attention", 6 * 8, 5, 2))
        self.assertEqual(vectors["residual.attention"], VectorShape("residual.attention", 6 * 8, 1, 2))
        self.assertEqual(vectors["residual.ffn"], VectorShape("residual.ffn", 6 * 8, 1, 2))
        self.assertEqual(vectors["attention.rope"], VectorShape("attention.rope", 6 * 3 * 4, 6, 2))
        self.assertEqual(vectors["attention.softmax"], VectorShape("attention.softmax", 6 * 2 * 3, 5, 2))


class MlaAndMoeStageOperationTests(unittest.TestCase):
    def test_attention_dp_splits_local_work_but_not_routed_assignments(self):
        model = make_mla_moe_model(
            num_routed_experts=8,
            num_experts_per_tok=3,
        )
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
        )
        ops = stage_operations(
            model,
            stage="prefill",
            batch_size=3,
            input_length=3,
            average_context=3,
            plan=plan,
        )
        gemms = by_name(ops.gemms)
        vectors = by_name(ops.vectors)

        # Two local requests make six local attention tokens. Routed
        # assignments remain ceil(9 * 3 / 2) = 14 across four local experts.
        self.assertEqual(gemms["attention.q_down_proj"].m, 6)
        self.assertEqual(vectors["attention.rope"].elements, 6 * 3 * 2)
        self.assertEqual(vectors["attention.softmax"].elements, 6 * 2 * 3)
        self.assertEqual(vectors["norm.input"].elements, 6 * 16)
        self.assertEqual(vectors["residual.ffn"].elements, 6 * 16)
        self.assertEqual(vectors["moe.routing"].elements, 6 * 8)
        self.assertEqual(gemms["moe.routed_gate_proj"].m, 4)
        self.assertEqual(gemms["moe.routed_gate_proj"].repeats, 2)
        self.assertEqual(gemms["moe.routed_gate_proj"].batch_repeats, 4)
        self.assertEqual(gemms["ffn.shared_gate_up"].m, 6)

    def test_mla_prefill_uses_no_absorb_shapes(self):
        ops = stage_operations(
            make_mla_moe_model(),
            stage="prefill",
            batch_size=2,
            input_length=3,
            average_context=3,
            plan=make_moe_plan(),
        )
        gemms = by_name(ops.gemms)

        self.assertEqual(gemms["attention.q_down_proj"], GemmShape("attention.q_down_proj", 6, 16, 4, 2))
        self.assertEqual(gemms["attention.q_up_proj"], GemmShape("attention.q_up_proj", 6, 4, 8, 2))
        self.assertEqual(gemms["attention.kv_down_proj"], GemmShape("attention.kv_down_proj", 6, 16, 5, 2))
        self.assertEqual(gemms["attention.kv_up_proj"], GemmShape("attention.kv_up_proj", 6, 3, 10, 2))
        self.assertEqual(
            gemms["attention.qk"],
            GemmShape(
                "attention.qk", 6, 4, 3, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            gemms["attention.pv"],
            GemmShape(
                "attention.pv", 6, 3, 3, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            sum(
                gemm_useful_ops(gemms[name])
                for name in ("attention.qk", "attention.pv")
            ),
            2 * 6 * 8 * 3 * 2 + 2 * 6 * 3 * 6 * 2,
        )
        self.assertNotIn("attention.q_wk", gemms)

    def test_mla_decode_uses_absorb_shapes_and_integral_context(self):
        ops = stage_operations(
            make_mla_moe_model(),
            stage="decode",
            batch_size=2,
            input_length=3,
            average_context=9.0,
            plan=make_moe_plan(),
        )
        gemms = by_name(ops.gemms)

        self.assertEqual(
            gemms["attention.q_wk"],
            GemmShape(
                "attention.q_wk", 2, 2, 3, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            gemms["attention.o_wv"],
            GemmShape(
                "attention.o_wv", 2, 3, 3, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            gemms["attention.qk"],
            GemmShape(
                "attention.qk", 2, 5, 9, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            gemms["attention.pv"],
            GemmShape(
                "attention.pv", 2, 9, 3, 2, batch_repeats=2
            ),
        )
        self.assertEqual(
            sum(
                gemm_useful_ops(gemms[name])
                for name in (
                    "attention.q_wk",
                    "attention.o_wv",
                    "attention.qk",
                    "attention.pv",
                )
            ),
            (
                2 * 2 * 4 * 3 * 2
                + 2 * 2 * 6 * 3 * 2
                + 2 * 2 * 10 * 9 * 2
                + 2 * 2 * 9 * 6 * 2
            ),
        )
        self.assertEqual(gemms["attention.o_proj"], GemmShape("attention.o_proj", 2, 6, 16, 2))
        self.assertNotIn("attention.kv_up_proj", gemms)

    def test_mla_without_q_lora_uses_direct_q_projection(self):
        model = make_mla_moe_model(q_lora_rank=None)
        ops = stage_operations(
            model,
            stage="decode",
            batch_size=2,
            input_length=3,
            average_context=9,
            plan=make_moe_plan(),
        )
        gemms = by_name(ops.gemms)

        self.assertEqual(gemms["attention.q_proj"], GemmShape("attention.q_proj", 2, 16, 8, 2))
        self.assertNotIn("attention.q_down_proj", gemms)
        self.assertNotIn("attention.q_up_proj", gemms)

    def test_routed_experts_use_active_expert_and_token_ceilings(self):
        model = make_mla_moe_model(num_routed_experts=8, num_experts_per_tok=3)
        plan = make_moe_plan(expert_parallel=4, attention_dp=2)
        ops = stage_operations(
            model,
            stage="decode",
            batch_size=5,
            input_length=3,
            average_context=9,
            plan=plan,
        )
        gemms = by_name(ops.gemms)
        vectors = by_name(ops.vectors)

        # ceil(5 * 3 / 4) = 4 assignments, two local experts, two tokens each.
        self.assertEqual(
            gemms["moe.routed_gate_proj"],
            GemmShape(
                "moe.routed_gate_proj",
                2,
                16,
                8,
                2,
                batch_repeats=2,
            ),
        )
        self.assertEqual(
            gemms["moe.routed_down_proj"],
            GemmShape(
                "moe.routed_down_proj",
                2,
                8,
                16,
                2,
                batch_repeats=2,
            ),
        )
        self.assertEqual(
            gemm_useful_ops(gemms["moe.routed_gate_proj"]),
            2 * 2 * 16 * 8 * 4,
        )
        self.assertEqual(vectors["moe.routing"], VectorShape("moe.routing", 3 * 8, 4, 2))
        self.assertEqual(
            vectors["moe.routed_silu_gate"],
            VectorShape("moe.routed_silu_gate", 2 * 8 * 2, 6, 2),
        )

    def test_shared_experts_are_dp_split_moe_tp_sharded_and_ep_replicated(self):
        model = make_mla_moe_model()
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
        )
        ops = stage_operations(
            model,
            stage="decode",
            batch_size=5,
            input_length=3,
            average_context=9,
            plan=plan,
        )
        gemms = by_name(ops.gemms)
        vectors = by_name(ops.vectors)

        self.assertEqual(
            gemms["ffn.shared_gate_up"],
            GemmShape("ffn.shared_gate_up", 3, 16, 12, 2),
        )
        self.assertEqual(
            gemms["ffn.shared_down"],
            GemmShape("ffn.shared_down", 3, 6, 16, 2),
        )
        self.assertEqual(
            [shape.name for shape in ops.gemms[-2:]],
            ["ffn.shared_gate_up", "ffn.shared_down"],
        )
        self.assertEqual(
            sum(
                gemm_useful_ops(gemms[name])
                for name in ("ffn.shared_gate_up", "ffn.shared_down")
            ),
            3 * (2 * 3 * 16 * 3 * 4),
        )
        self.assertEqual(
            vectors["ffn.shared_silu_gate"],
            VectorShape("ffn.shared_silu_gate", 3 * 2 * 3, 6, 2),
        )


class HybridStageOperationTests(unittest.TestCase):
    def test_linear_attention_shapes_and_vectors_have_linear_layer_repeats(self):
        model = make_hybrid_model()
        ops = stage_operations(
            model,
            stage="decode",
            batch_size=2,
            input_length=3,
            average_context=9,
            plan=make_dense_plan(),
        )
        gemms = by_name(ops.gemms)
        vectors = by_name(ops.vectors)

        self.assertEqual(gemms["linear_attention.qkvzba_proj"], GemmShape("linear_attention.qkvzba_proj", 2, 12, 24, 2))
        self.assertEqual(gemms["linear_attention.o_proj"], GemmShape("linear_attention.o_proj", 2, 6, 12, 2))
        self.assertEqual(vectors["linear_attention.core"], VectorShape("linear_attention.core", 2 * (2 * 4 + 2 * 6), 6, 2))
        self.assertEqual(vectors["norm.input"].repeats, 3)
        self.assertEqual(vectors["residual.ffn"].repeats, 3)

    def test_linear_attention_heads_are_sharded_across_attention_tp(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        )
        tp1 = stage_operations(
            model,
            stage="decode",
            batch_size=2,
            input_length=3,
            average_context=9,
            plan=make_dense_plan(),
        )
        tp2 = stage_operations(
            model,
            stage="decode",
            batch_size=2,
            input_length=3,
            average_context=9,
            plan=make_dense_plan(attention_tp=2, moe_tp=2),
        )
        tp1_gemms = by_name(tp1.gemms)
        tp2_gemms = by_name(tp2.gemms)
        tp1_vectors = by_name(tp1.vectors)
        tp2_vectors = by_name(tp2.vectors)

        self.assertEqual(tp1_gemms["linear_attention.qkvzba_proj"].n, 48)
        self.assertEqual(tp2_gemms["linear_attention.qkvzba_proj"].n, 24)
        self.assertEqual(tp1_gemms["linear_attention.o_proj"].k, 12)
        self.assertEqual(tp2_gemms["linear_attention.o_proj"].k, 6)
        self.assertEqual(tp1_vectors["linear_attention.core"].elements, 80)
        self.assertEqual(tp2_vectors["linear_attention.core"].elements, 40)


class StageValidationTests(unittest.TestCase):
    def test_rejects_invalid_stage_dimensions_context_and_plan(self):
        model = make_dense_model()
        invalid_cases = (
            ({"stage": "train", "batch_size": 1, "input_length": 1, "average_context": 1, "plan": make_dense_plan()}, "stage"),
            ({"stage": "decode", "batch_size": 0, "input_length": 1, "average_context": 1, "plan": make_dense_plan()}, "batch_size"),
            ({"stage": "decode", "batch_size": 1, "input_length": True, "average_context": 1, "plan": make_dense_plan()}, "input_length"),
            ({"stage": "decode", "batch_size": 1, "input_length": 1, "average_context": 1.5, "plan": make_dense_plan()}, "average_context"),
            ({"stage": "decode", "batch_size": 1, "input_length": 1, "average_context": 1, "plan": object()}, "plan"),
            ({"stage": "decode", "batch_size": 1, "input_length": 1, "average_context": 1, "plan": make_dense_plan(attention_tp=3, moe_tp=3)}, "plan"),
        )

        for kwargs, expected_path in invalid_cases:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(InputValidationError) as caught:
                    stage_operations(model, **kwargs)
                self.assertEqual(caught.exception.path, expected_path)

    def test_invalid_plan_error_includes_validation_reason_code(self):
        with self.assertRaises(InputValidationError) as caught:
            stage_operations(
                make_dense_model(),
                stage="decode",
                batch_size=1,
                input_length=1,
                average_context=1,
                plan=make_dense_plan(attention_tp=3, moe_tp=3),
            )

        self.assertIn("ATTENTION_HEADS_NOT_DIVISIBLE", caught.exception.message)


if __name__ == "__main__":
    unittest.main()
