import unittest
from dataclasses import FrozenInstanceError, fields
from math import ceil

from infersim.cost import (
    StageMetrics,
    activation_payload_bytes,
    all_reduce_cost,
    all_to_all_cost,
    evaluate_prefill,
    evaluate_prefill_scenarios,
    gemm_cost,
    stage_operations,
    vector_cost,
    vector_mode_for_bits,
)
from infersim.errors import InputValidationError
from tests.helpers import (
    make_dense_model,
    make_dense_plan,
    make_hardware,
    make_mla_moe_model,
    make_moe_plan,
    make_scenario,
    make_scenario_set,
    make_w4a4_precision,
    make_w4a8_precision,
)


def expected_kernel_totals(model, hardware, precision, plan, scenario):
    operations = stage_operations(
        model,
        stage="prefill",
        batch_size=plan.batch_size,
        input_length=scenario.input_length,
        average_context=scenario.input_length,
        plan=plan,
    )
    gemm_seconds = 0.0
    useful_gemm_ops = 0
    aligned_gemm_ops = 0
    for shape in operations.gemms:
        cost = gemm_cost(
            shape.m,
            shape.k,
            shape.n,
            hardware,
            precision,
            repeats=shape.batch_repeats,
        )
        gemm_seconds += cost.seconds * shape.repeats
        useful_gemm_ops += cost.useful_ops * shape.repeats
        aligned_gemm_ops += cost.aligned_ops * shape.repeats

    vector_seconds = 0.0
    useful_vector_ops = 0
    aligned_vector_ops = 0
    mode = vector_mode_for_bits(precision.vector_bits)
    for shape in operations.vectors:
        cost = vector_cost(
            shape.elements,
            shape.ops_per_element,
            mode,
            hardware,
            repeats=1,
        )
        vector_seconds += cost.seconds * shape.repeats
        useful_vector_ops += cost.useful_ops * shape.repeats
        aligned_vector_ops += cost.aligned_ops * shape.repeats
    return (
        gemm_seconds,
        vector_seconds,
        useful_gemm_ops,
        aligned_gemm_ops,
        useful_vector_ops,
        aligned_vector_ops,
    )


class PrefillEvaluatorTests(unittest.TestCase):
    def test_dense_prefill_metrics_and_capacities_are_stage_local(self):
        model = make_dense_model()
        hardware = make_hardware()
        precision = make_w4a8_precision()
        plan = make_dense_plan(replicas=3, batch_size=4)
        scenario = make_scenario(input_length=128)

        result = evaluate_prefill(model, hardware, precision, plan, scenario)

        self.assertEqual(result.stage, "prefill")
        self.assertEqual(result.scenario_name, scenario.name)
        self.assertEqual(result.plan, plan)
        self.assertEqual(set(result.component_seconds), {"gemm", "vector", "tp", "ep"})
        self.assertNotIn("pd", result.component_seconds)
        self.assertNotIn("decode", result.component_seconds)
        self.assertEqual(result.latency_seconds, sum(result.component_seconds.values()))
        self.assertEqual(result.request_capacity, plan.replicas * 4 / result.latency_seconds)
        self.assertEqual(result.prompt_token_capacity, plan.replicas * 4 * 128 / result.latency_seconds)
        self.assertEqual(result.average_context_length, 128.0)
        self.assertIsNone(result.tpot_seconds)
        self.assertIsNone(result.output_token_capacity)
        self.assertEqual(result.memory.stage, "prefill")

    def test_kernel_totals_preserve_batched_launches_and_layer_launches(self):
        model = make_dense_model(num_hidden_layers=1)
        hardware = make_hardware(
            gemm_tile={"m": 4, "n": 4, "k": 4},
            gemm_engines=1,
        )
        precision = make_w4a8_precision()
        plan = make_dense_plan(batch_size=1)
        scenario = make_scenario(input_length=3)

        result = evaluate_prefill(model, hardware, precision, plan, scenario)
        expected = expected_kernel_totals(
            model, hardware, precision, plan, scenario
        )

        self.assertEqual(result.gemm_seconds, expected[0])
        self.assertEqual(result.vector_seconds, expected[1])
        self.assertEqual(result.useful_gemm_ops, expected[2])
        self.assertEqual(result.aligned_gemm_ops, expected[3])
        self.assertEqual(result.useful_vector_ops, expected[4])
        self.assertEqual(result.aligned_vector_ops, expected[5])

        two_layers = evaluate_prefill(
            make_dense_model(num_hidden_layers=2),
            hardware,
            precision,
            plan,
            scenario,
        )
        self.assertGreater(
            two_layers.gemm_seconds - result.gemm_seconds,
            hardware.gemm_launch_latency_us * 1e-6,
        )

    def test_dense_tp_charges_attention_and_ffn_all_reduces_per_layer(self):
        model = make_dense_model(
            num_hidden_layers=3,
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        hardware = make_hardware()
        precision = make_w4a4_precision()
        plan = make_dense_plan(attention_tp=2, moe_tp=2, batch_size=3)
        scenario = make_scenario(input_length=5)
        local_tokens = ceil(3 * 5 / plan.attention_dp)
        payload = activation_payload_bytes(
            local_tokens * model.hidden_size, precision
        )
        one = all_reduce_cost(payload, 2, hardware).seconds

        result = evaluate_prefill(model, hardware, precision, plan, scenario)

        self.assertEqual(result.tp_seconds, 6 * one)
        self.assertEqual(result.ep_seconds, 0.0)

    def test_moe_uses_distinct_attention_tp_moe_tp_and_two_ep_exchanges(self):
        model = make_mla_moe_model(num_hidden_layers=2)
        hardware = make_hardware()
        precision = make_w4a8_precision()
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=1,
            expert_parallel=4,
            batch_size=5,
        )
        scenario = make_scenario(input_length=3)
        local_tokens = ceil(5 * 3 / 2)
        activation = activation_payload_bytes(
            local_tokens * model.hidden_size, precision
        )
        attention_tp = (
            model.num_full_attention_layers
            * all_reduce_cost(activation, 2, hardware).seconds
        )
        moe_tp = (
            model.num_hidden_layers
            * all_reduce_cost(activation, 1, hardware).seconds
        )
        expert_payload = activation_payload_bytes(
            local_tokens * model.experts_per_token * model.hidden_size,
            precision,
        )
        ep = (
            2
            * model.num_hidden_layers
            * all_to_all_cost(expert_payload, 4, hardware).seconds
        )

        result = evaluate_prefill(model, hardware, precision, plan, scenario)

        self.assertEqual(result.tp_seconds, attention_tp + moe_tp)
        self.assertEqual(result.ep_seconds, ep)

    def test_memory_infeasibility_is_reported_without_rejecting_metrics(self):
        result = evaluate_prefill(
            make_dense_model(),
            make_hardware(
                memory_capacity_gb=1e-9,
                memory_reserve_fraction=1,
            ),
            make_w4a8_precision(),
            make_dense_plan(),
            make_scenario(),
        )

        self.assertFalse(result.memory.feasible)
        self.assertGreater(result.latency_seconds, 0)

    def test_scenario_set_preserves_order_names_context_and_capacities(self):
        scenarios = (
            make_scenario(name="short", input_length=8),
            make_scenario(name="long", input_length=64),
        )
        plan = make_dense_plan(replicas=2, batch_size=3)

        results = evaluate_prefill_scenarios(
            make_dense_model(),
            make_hardware(),
            make_w4a8_precision(),
            plan,
            make_scenario_set(scenarios, policy="weighted"),
        )

        self.assertIsInstance(results, tuple)
        self.assertEqual([item.scenario_name for item in results], ["short", "long"])
        self.assertEqual([item.average_context_length for item in results], [8.0, 64.0])
        for item, scenario in zip(results, scenarios):
            self.assertEqual(
                item.prompt_token_capacity,
                plan.replicas * plan.batch_size * scenario.input_length / item.latency_seconds,
            )

    def test_larger_gemm_tiles_increase_alignment_and_latency(self):
        model = make_dense_model(num_hidden_layers=1)
        precision = make_w4a8_precision()
        plan = make_dense_plan(batch_size=1)
        scenario = make_scenario(input_length=1)
        compute = {
            "gemm": {"w4a4": 0.001, "w4a8": 0.001},
            "vector": {"fp4": 160, "int8": 120, "bf16": 60, "fp32": 30},
        }
        small = evaluate_prefill(
            model,
            make_hardware(
                compute_tflops=compute,
                gemm_tile={"m": 1, "n": 1, "k": 1},
                gemm_engines=1,
            ),
            precision,
            plan,
            scenario,
        )
        large = evaluate_prefill(
            model,
            make_hardware(
                compute_tflops=compute,
                gemm_tile={"m": 128, "n": 128, "k": 64},
                gemm_engines=4,
            ),
            precision,
            plan,
            scenario,
        )

        self.assertGreater(large.aligned_gemm_ops, small.aligned_gemm_ops)
        self.assertGreater(large.gemm_seconds, small.gemm_seconds)

    def test_vector_precision_modes_and_hardware_validation_are_respected(self):
        model = make_dense_model(num_hidden_layers=1)
        hardware = make_hardware()
        plan = make_dense_plan(batch_size=1)
        scenario = make_scenario(input_length=2)

        for precision in (make_w4a4_precision(), make_w4a8_precision()):
            with self.subTest(vector_bits=precision.vector_bits):
                result = evaluate_prefill(
                    model, hardware, precision, plan, scenario
                )
                expected = expected_kernel_totals(
                    model, hardware, precision, plan, scenario
                )
                self.assertEqual(result.vector_seconds, expected[1])

        no_fp4 = make_hardware(
            compute_tflops={
                "gemm": {"w4a4": 1200, "w4a8": 900},
                "vector": {"int8": 120},
            }
        )
        with self.assertRaises(InputValidationError) as caught:
            evaluate_prefill(
                model, no_fp4, make_w4a4_precision(), plan, scenario
            )
        self.assertEqual(caught.exception.path, "compute_tflops.vector.fp4")

    def test_invalid_types_plans_and_nonfinite_derived_latency_are_rejected(self):
        base = (
            make_dense_model(),
            make_hardware(),
            make_w4a8_precision(),
            make_dense_plan(),
            make_scenario(),
        )
        paths = ("model", "hardware", "precision", "plan", "scenario")
        for index, path in enumerate(paths):
            args = list(base)
            args[index] = object()
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    evaluate_prefill(*args)
                self.assertEqual(caught.exception.path, path)

        with self.assertRaises(InputValidationError) as caught:
            evaluate_prefill(
                make_dense_model(),
                make_hardware(),
                make_w4a8_precision(),
                make_dense_plan(attention_tp=3, moe_tp=3),
                make_scenario(),
            )
        self.assertEqual(caught.exception.path, "plan")
        self.assertIn("ATTENTION_HEADS_NOT_DIVISIBLE", caught.exception.message)

        overflow_hardware = make_hardware(
            kernel_launch_latency_us={
                "gemm": 1e308,
                "vector": 1e308,
                "collective": 0,
            },
        )
        with self.assertRaises(InputValidationError) as caught:
            evaluate_prefill(
                make_dense_model(num_hidden_layers=10_000_000),
                overflow_hardware,
                make_w4a8_precision(),
                make_dense_plan(),
                make_scenario(input_length=1),
            )
        self.assertEqual(caught.exception.path, "latency_seconds")

        with self.assertRaises(InputValidationError):
            evaluate_prefill(
                make_dense_model(),
                make_hardware(),
                make_w4a8_precision(),
                make_dense_plan(),
                make_scenario(input_length=10**1000),
            )

    def test_stage_metrics_are_frozen_with_a_deeply_immutable_mapping(self):
        result = evaluate_prefill(
            make_dense_model(),
            make_hardware(),
            make_w4a8_precision(),
            make_dense_plan(),
            make_scenario(),
        )

        expected_fields = (
            "stage", "scenario_name", "plan", "latency_seconds", "tpot_seconds",
            "prompt_token_capacity", "output_token_capacity", "request_capacity",
            "average_context_length", "gemm_seconds", "vector_seconds", "tp_seconds",
            "ep_seconds", "useful_gemm_ops", "aligned_gemm_ops", "useful_vector_ops",
            "aligned_vector_ops", "memory", "component_seconds",
        )
        self.assertEqual(tuple(field.name for field in fields(StageMetrics)), expected_fields)
        with self.assertRaises(FrozenInstanceError):
            result.latency_seconds = 0
        with self.assertRaises(TypeError):
            result.component_seconds["gemm"] = 0

        source = dict(result.component_seconds)
        copied = StageMetrics(
            **{
                **{field.name: getattr(result, field.name) for field in fields(result)},
                "component_seconds": source,
            }
        )
        source["gemm"] = -1
        self.assertEqual(copied.component_seconds["gemm"], result.gemm_seconds)

    def test_scenario_set_type_is_validated(self):
        with self.assertRaises(InputValidationError) as caught:
            evaluate_prefill_scenarios(
                make_dense_model(),
                make_hardware(),
                make_w4a8_precision(),
                make_dense_plan(),
                object(),
            )
        self.assertEqual(caught.exception.path, "scenario_set")


if __name__ == "__main__":
    unittest.main()
