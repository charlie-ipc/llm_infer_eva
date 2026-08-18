import unittest
from dataclasses import FrozenInstanceError
from math import ceil, floor

from infersim.cost import (
    activation_payload_bytes,
    all_reduce_cost,
    all_to_all_cost,
    evaluate_decode,
    evaluate_decode_scenarios,
    gemm_cost,
    kernel_cost,
    kv_bytes_per_request,
    recurrent_state_bytes_per_request,
    stage_operations,
    vector_cost,
    vector_mode_for_bits,
)
from infersim.errors import InputValidationError
from tests.helpers import (
    make_dense_model,
    make_dense_plan,
    make_hardware,
    make_hybrid_model,
    make_memory_bound_hardware,
    make_mla_moe_model,
    make_moe_plan,
    make_scenario,
    make_scenario_set,
    make_w4a4_precision,
    make_w4a8_precision,
)


class DecodeEvaluatorTests(unittest.TestCase):
    def test_decode_metrics_use_one_iteration_tpot_and_stage_capacities(self):
        model = make_dense_model()
        hardware = make_hardware()
        precision = make_w4a8_precision()
        plan = make_dense_plan(replicas=3, batch_size=8)
        scenario = make_scenario(input_length=128, output_length=32)

        result = evaluate_decode(model, hardware, precision, plan, scenario)

        self.assertEqual(result.stage, "decode")
        self.assertEqual(result.scenario_name, scenario.name)
        self.assertEqual(result.plan, plan)
        self.assertEqual(set(result.component_seconds), {"gemm", "vector", "tp", "ep"})
        self.assertEqual(result.latency_seconds, sum(result.component_seconds.values()))
        self.assertEqual(result.tpot_seconds, result.latency_seconds)
        self.assertIsNone(result.prompt_token_capacity)
        self.assertEqual(
            result.output_token_capacity,
            plan.replicas * 8 / result.tpot_seconds,
        )
        self.assertEqual(
            result.request_capacity,
            result.output_token_capacity / scenario.output_length,
        )
        self.assertEqual(result.average_context_length, 144.0)
        self.assertEqual(result.memory.stage, "decode")

    def test_mha_attention_replaces_cache_operand_with_exact_kv_bytes(self):
        model = make_dense_model(
            hidden_size=16,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            num_hidden_layers=1,
            num_routed_experts=4,
            num_experts_per_tok=2,
            moe_intermediate_size=16,
        )
        hardware = make_memory_bound_hardware(
            gemm_tile={"m": 1, "n": 1, "k": 1},
            gemm_engines=1,
        )
        precision = make_w4a8_precision(kv_cache_bits=8)
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
            batch_size=5,
        )
        scenario = make_scenario(input_length=100, output_length=20)
        context = ceil(100 + 20 / 2)
        local_requests = ceil(5 / 2)
        local_kv_heads = model.num_key_value_heads // plan.attention_tp
        cache_bytes = (
            local_requests
            * context
            * local_kv_heads
            * model.head_dim
            * precision.kv_cache_bits
            / 8
        )
        operations = stage_operations(
            model,
            stage="decode",
            batch_size=plan.batch_size,
            input_length=scenario.input_length,
            average_context=context,
            plan=plan,
        )
        expected_seconds = 0.0
        expected_cache_memory = 0.0
        for shape in operations.gemms:
            base = gemm_cost(
                shape.m,
                shape.k,
                shape.n,
                hardware,
                precision,
                repeats=shape.batch_repeats,
            )
            cost = base
            if shape.name in ("attention.qk", "attention.pv"):
                non_cache_bytes = (
                    shape.batch_repeats
                    * (shape.m * shape.k + shape.m * shape.n)
                    * precision.activation_bits
                    / 8
                )
                expected_cache_memory += cache_bytes * shape.repeats
                cost = kernel_cost(
                    useful_ops=base.useful_ops,
                    aligned_ops=base.aligned_ops,
                    compute_ops_per_second=(
                        hardware.gemm_tflops[precision.gemm_mode] * 1e12
                    ),
                    memory_bytes=non_cache_bytes + cache_bytes,
                    memory_bandwidth_bytes_s=(
                        hardware.memory_bandwidth_gbps * 1e9
                    ),
                    launch_seconds=hardware.gemm_launch_latency_us * 1e-6,
                )
            expected_seconds += cost.seconds * shape.repeats

        result = evaluate_decode(model, hardware, precision, plan, scenario)

        self.assertEqual(expected_cache_memory, 2 * cache_bytes)
        self.assertEqual(result.gemm_seconds, expected_seconds)

    def test_mla_attention_cache_is_replicated_across_tensor_parallelism(self):
        model = make_mla_moe_model(num_hidden_layers=1)
        hardware = make_memory_bound_hardware(
            gemm_tile={"m": 1, "n": 1, "k": 1},
            gemm_engines=1,
        )
        precision = make_w4a8_precision(kv_cache_bits=4)
        plan = make_moe_plan(attention_tp=2, batch_size=3)
        scenario = make_scenario(input_length=100, output_length=20)
        context = 110
        local_requests = 3
        operations = stage_operations(
            model,
            stage="decode",
            batch_size=plan.batch_size,
            input_length=scenario.input_length,
            average_context=context,
            plan=plan,
        )
        cache_bytes = {
            "attention.qk": (
                local_requests
                * context
                * (model.kv_lora_rank + model.qk_rope_head_dim)
                * precision.kv_cache_bits
                / 8
            ),
            "attention.pv": (
                local_requests
                * context
                * model.kv_lora_rank
                * precision.kv_cache_bits
                / 8
            ),
        }
        expected_seconds = 0.0
        for shape in operations.gemms:
            base = gemm_cost(
                shape.m,
                shape.k,
                shape.n,
                hardware,
                precision,
                repeats=shape.batch_repeats,
            )
            cost = base
            if shape.name in cache_bytes:
                non_cache_bytes = (
                    shape.batch_repeats
                    * (shape.m * shape.k + shape.m * shape.n)
                    * precision.activation_bits
                    / 8
                )
                cost = kernel_cost(
                    useful_ops=base.useful_ops,
                    aligned_ops=base.aligned_ops,
                    compute_ops_per_second=(
                        hardware.gemm_tflops[precision.gemm_mode] * 1e12
                    ),
                    memory_bytes=non_cache_bytes + cache_bytes[shape.name],
                    memory_bandwidth_bytes_s=(
                        hardware.memory_bandwidth_gbps * 1e9
                    ),
                    launch_seconds=hardware.gemm_launch_latency_us * 1e-6,
                )
            expected_seconds += cost.seconds * shape.repeats

        result = evaluate_decode(model, hardware, precision, plan, scenario)

        self.assertEqual(result.gemm_seconds, expected_seconds)

    def test_hybrid_recurrent_state_is_read_once_per_linear_layer(self):
        model = make_hybrid_model(
            num_attention_heads=4,
            num_key_value_heads=4,
            linear_num_key_heads=4,
            linear_num_value_heads=4,
        )
        hardware = make_memory_bound_hardware()
        precision = make_w4a8_precision()
        plan = make_dense_plan(attention_tp=2, moe_tp=2, batch_size=3)
        scenario = make_scenario(input_length=64, output_length=16)
        operations = stage_operations(
            model,
            stage="decode",
            batch_size=plan.batch_size,
            input_length=scenario.input_length,
            average_context=72,
            plan=plan,
        )
        state_per_layer = (
            recurrent_state_bytes_per_request(model)
            / model.num_linear_attention_layers
            * plan.batch_size
            / plan.attention_tp
        )
        mode = vector_mode_for_bits(precision.vector_bits)
        expected_seconds = 0.0
        for shape in operations.vectors:
            memory_bytes = None
            if shape.name == "linear_attention.core":
                memory_bytes = (
                    shape.elements * 2 * precision.vector_bits / 8
                    + state_per_layer
                )
            cost = vector_cost(
                shape.elements,
                shape.ops_per_element,
                mode,
                hardware,
                repeats=1,
                memory_bytes=memory_bytes,
            )
            expected_seconds += cost.seconds * shape.repeats

        result = evaluate_decode(model, hardware, precision, plan, scenario)

        self.assertEqual(result.vector_seconds, expected_seconds)
        self.assertEqual(
            result.memory.recurrent_state_bytes_per_card,
            recurrent_state_bytes_per_request(model) * plan.batch_size,
        )

    def test_max_batch_uses_full_context_variable_memory_and_reports_oom_zero(self):
        model = make_dense_model(
            num_attention_heads=2,
            num_key_value_heads=2,
        )
        hardware = make_hardware(memory_capacity_gb=0.001)
        precision = make_w4a8_precision()
        plan = make_dense_plan(
            replicas=3,
            attention_tp=2,
            moe_tp=2,
            batch_size=2,
        )
        scenario = make_scenario(input_length=100, output_length=20)

        result = evaluate_decode(model, hardware, precision, plan, scenario)
        variable_bytes = (
            kv_bytes_per_request(model, precision, 120) / plan.attention_tp
            + recurrent_state_bytes_per_request(model)
            + 2 * model.hidden_size * precision.activation_bits / 8
        )
        expected_local = floor(
            max(0, result.memory.usable_bytes - result.memory.total_weight_bytes)
            / variable_bytes
        )
        expected_batch = expected_local * plan.attention_dp

        self.assertEqual(result.max_supported_batch, expected_batch)
        self.assertEqual(
            result.max_supported_concurrency,
            plan.replicas * expected_batch,
        )

        oom = evaluate_decode(
            model,
            make_hardware(
                memory_capacity_gb=1e-9,
                memory_reserve_fraction=1,
            ),
            precision,
            plan,
            scenario,
        )
        self.assertEqual(oom.max_supported_batch, 0)
        self.assertEqual(oom.max_supported_concurrency, 0)

    def test_scenario_set_preserves_order_and_validates_its_type(self):
        scenarios = (
            make_scenario(name="short", input_length=8, output_length=4),
            make_scenario(name="long", input_length=64, output_length=16),
        )
        plan = make_dense_plan(replicas=2, batch_size=3)

        results = evaluate_decode_scenarios(
            make_dense_model(),
            make_hardware(),
            make_w4a8_precision(),
            plan,
            make_scenario_set(scenarios, policy="weighted"),
        )

        self.assertIsInstance(results, tuple)
        self.assertEqual([item.scenario_name for item in results], ["short", "long"])
        self.assertEqual(
            [item.average_context_length for item in results],
            [10.0, 72.0],
        )
        for item, scenario in zip(results, scenarios):
            self.assertEqual(
                item.request_capacity,
                plan.replicas
                * plan.batch_size
                / item.tpot_seconds
                / scenario.output_length,
            )

        with self.assertRaises(InputValidationError) as caught:
            evaluate_decode_scenarios(
                make_dense_model(),
                make_hardware(),
                make_w4a8_precision(),
                plan,
                object(),
            )
        self.assertEqual(caught.exception.path, "scenario_set")

    def test_odd_output_length_reports_half_context_but_rounds_kernel_up(self):
        model = make_dense_model(num_hidden_layers=1)
        hardware = make_memory_bound_hardware()
        precision = make_w4a8_precision()
        plan = make_dense_plan(batch_size=2)

        odd = evaluate_decode(
            model,
            hardware,
            precision,
            plan,
            make_scenario(input_length=100, output_length=21),
        )
        even = evaluate_decode(
            model,
            hardware,
            precision,
            plan,
            make_scenario(input_length=100, output_length=22),
        )

        self.assertEqual(odd.average_context_length, 110.5)
        self.assertEqual(even.average_context_length, 111.0)
        self.assertEqual(odd.gemm_seconds, even.gemm_seconds)
        self.assertNotEqual(
            odd.memory.kv_bytes_per_card,
            even.memory.kv_bytes_per_card,
        )

    def test_decode_moe_communication_uses_batch_and_local_requests(self):
        model = make_mla_moe_model(num_hidden_layers=2)
        hardware = make_hardware()
        precision = make_w4a8_precision()
        plan = make_moe_plan(
            attention_tp=2,
            attention_dp=2,
            moe_tp=2,
            expert_parallel=2,
            batch_size=5,
        )
        scenario = make_scenario(input_length=100, output_length=20)
        local_requests = ceil(plan.batch_size / plan.attention_dp)
        attention_payload = activation_payload_bytes(
            local_requests * model.hidden_size,
            precision,
        )
        routed_assignments = ceil(
            plan.batch_size
            * model.experts_per_token
            / plan.expert_parallel
        )
        routed_payload = activation_payload_bytes(
            routed_assignments * model.hidden_size,
            precision,
        )
        expected_tp = (
            model.num_full_attention_layers
            * all_reduce_cost(
                attention_payload, plan.attention_tp, hardware
            ).seconds
            + model.num_hidden_layers
            * all_reduce_cost(
                routed_payload, plan.moe_tp, hardware
            ).seconds
            + model.num_hidden_layers
            * all_reduce_cost(
                attention_payload, plan.moe_tp, hardware
            ).seconds
        )
        expert_payload = activation_payload_bytes(
            local_requests
            * model.experts_per_token
            * model.hidden_size,
            precision,
        )
        expected_ep = (
            2
            * model.num_hidden_layers
            * all_to_all_cost(
                expert_payload, plan.expert_parallel, hardware
            ).seconds
        )

        result = evaluate_decode(model, hardware, precision, plan, scenario)
        operations = stage_operations(
            model,
            stage="decode",
            batch_size=plan.batch_size,
            input_length=scenario.input_length,
            average_context=110,
            plan=plan,
        )

        self.assertEqual(operations.gemms[0].m, local_requests)
        self.assertEqual(result.tp_seconds, expected_tp)
        self.assertEqual(result.ep_seconds, expected_ep)

    def test_kv_and_activation_bit_widths_change_decode_memory_traffic(self):
        model = make_dense_model(num_hidden_layers=1)
        hardware = make_memory_bound_hardware(
            gemm_tile={"m": 1, "n": 1, "k": 1},
            gemm_engines=1,
        )
        plan = make_dense_plan(batch_size=2)
        scenario = make_scenario(input_length=100, output_length=20)

        kv4 = evaluate_decode(
            model,
            hardware,
            make_w4a8_precision(kv_cache_bits=4),
            plan,
            scenario,
        )
        kv8 = evaluate_decode(
            model,
            hardware,
            make_w4a8_precision(kv_cache_bits=8),
            plan,
            scenario,
        )
        a4 = evaluate_decode(
            model,
            hardware,
            make_w4a4_precision(kv_cache_bits=8),
            plan,
            scenario,
        )

        self.assertLess(kv4.gemm_seconds, kv8.gemm_seconds)
        self.assertEqual(kv4.useful_gemm_ops, kv8.useful_gemm_ops)
        self.assertLess(a4.gemm_seconds, kv8.gemm_seconds)
        self.assertLess(a4.vector_seconds, kv8.vector_seconds)

    def test_full_context_memory_invalid_inputs_and_immutability(self):
        model = make_dense_model()
        hardware = make_hardware()
        precision = make_w4a8_precision()
        plan = make_dense_plan()
        scenario = make_scenario(input_length=100, output_length=20)

        result = evaluate_decode(model, hardware, precision, plan, scenario)

        self.assertEqual(
            result.memory.kv_bytes_per_card,
            kv_bytes_per_request(model, precision, 120) * plan.batch_size,
        )
        self.assertNotIn("prefill", result.component_seconds)
        self.assertNotIn("pd", result.component_seconds)
        with self.assertRaises(FrozenInstanceError):
            result.max_supported_batch = 0
        with self.assertRaises(TypeError):
            result.component_seconds["gemm"] = 0

        base = (model, hardware, precision, plan, scenario)
        paths = ("model", "hardware", "precision", "plan", "scenario")
        for index, path in enumerate(paths):
            args = list(base)
            args[index] = object()
            with self.subTest(path=path):
                with self.assertRaises(InputValidationError) as caught:
                    evaluate_decode(*args)
                self.assertEqual(caught.exception.path, path)


if __name__ == "__main__":
    unittest.main()
