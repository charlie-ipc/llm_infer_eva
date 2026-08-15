import unittest
from dataclasses import FrozenInstanceError

from infersim.errors import InputValidationError
from infersim.schema.hardware import HardwareSpec
from infersim.schema.parallel import SearchSpace
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import PDLinkSpec, ScenarioSet, WorkloadScenario


def hardware_config(**overrides):
    config = {
        "name": "Test Accelerator",
        "memory_capacity_gb": 96,
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
        "kernel_launch_latency_us": {
            "gemm": 5,
            "vector": 3,
            "collective": 8,
        },
        "interconnect": {
            "intra_node_gbps": 900,
            "intra_node_latency_us": 1,
            "inter_node_gbps": 400,
            "inter_node_latency_us": 5,
        },
    }
    config.update(overrides)
    return config


def scenario_config(**overrides):
    config = {
        "name": "chat",
        "input_length": 2048,
        "output_length": 512,
        "request_rate": 4.5,
        "concurrency": 16,
        "ttft_limit_ms": 200,
        "tpot_limit_ms": 30,
    }
    config.update(overrides)
    return config


class PathAssertions:
    def assert_invalid_path(self, expected_path, callable_, *args):
        with self.assertRaises(InputValidationError) as context:
            callable_(*args)
        self.assertEqual(context.exception.path, expected_path)


class HardwareSpecTests(PathAssertions, unittest.TestCase):
    def test_normalizes_custom_hardware_and_defaults(self):
        spec = HardwareSpec.from_dict(hardware_config())

        self.assertEqual(spec.name, "Test Accelerator")
        self.assertEqual(spec.memory_capacity_gb, 96.0)
        self.assertEqual(spec.memory_bandwidth_gbps, 4000.0)
        self.assertEqual(spec.cards_per_node, 8)
        self.assertEqual(spec.gemm_tflops["w4a8"], 900.0)
        self.assertEqual(spec.vector_tflops["int8"], 120.0)
        self.assertEqual(spec.gemm_tile, (128, 128, 64))
        self.assertEqual(spec.gemm_engines, 4)
        self.assertEqual(spec.vector_width, 32)
        self.assertEqual(spec.vector_units, 16)
        self.assertEqual(spec.gemm_launch_latency_us, 5.0)
        self.assertEqual(spec.vector_launch_latency_us, 3.0)
        self.assertEqual(spec.collective_launch_latency_us, 8.0)
        self.assertEqual(spec.intra_node.bandwidth_gbps, 900.0)
        self.assertEqual(spec.intra_node.latency_us, 1.0)
        self.assertEqual(spec.inter_node.bandwidth_gbps, 400.0)
        self.assertEqual(spec.inter_node.latency_us, 5.0)
        self.assertEqual(spec.memory_reserve_fraction, 0.1)
        self.assertEqual(spec.runtime_workspace_gb, 0.0)
        self.assertIsNone(spec.cost_per_card_hour)

    def test_accepts_optional_memory_and_cost_fields_including_zero_cost(self):
        spec = HardwareSpec.from_dict(hardware_config(
            memory_reserve_fraction=0.25,
            runtime_workspace_gb=2,
            cost_per_card_hour=0,
        ))
        self.assertEqual(spec.memory_reserve_fraction, 0.25)
        self.assertEqual(spec.runtime_workspace_gb, 2.0)
        self.assertEqual(spec.cost_per_card_hour, 0.0)

    def test_is_deeply_immutable(self):
        spec = HardwareSpec.from_dict(hardware_config())
        with self.assertRaises(FrozenInstanceError):
            spec.name = "changed"
        with self.assertRaises(TypeError):
            spec.gemm_tflops["w4a4"] = 1
        with self.assertRaises(TypeError):
            spec.vector_tflops["int8"] = 1

    def test_rejects_nonpositive_memory_bandwidth_with_exact_path(self):
        self.assert_invalid_path(
            "memory_bandwidth_gbps",
            HardwareSpec.from_dict,
            hardware_config(memory_bandwidth_gbps=0),
        )

    def test_rejects_invalid_nested_hardware_values_with_exact_paths(self):
        cases = [
            ({"compute_tflops": {"gemm": {}, "vector": {"int8": 1}}},
             "compute_tflops.gemm"),
            ({"gemm_tile": {"m": 0, "n": 1, "k": 1}}, "gemm_tile.m"),
            ({"kernel_launch_latency_us": {"gemm": -1, "vector": 0,
                                            "collective": 0}},
             "kernel_launch_latency_us.gemm"),
            ({"interconnect": {"intra_node_gbps": 0,
                                "intra_node_latency_us": 0,
                                "inter_node_gbps": 1,
                                "inter_node_latency_us": 0}},
             "interconnect.intra_node_gbps"),
        ]
        for overrides, path in cases:
            with self.subTest(path=path):
                self.assert_invalid_path(
                    path, HardwareSpec.from_dict, hardware_config(**overrides)
                )

    def test_rejects_bool_numeric_and_invalid_optional_ranges(self):
        cases = [
            ({"memory_capacity_gb": True}, "memory_capacity_gb"),
            ({"cards_per_node": True}, "cards_per_node"),
            ({"memory_reserve_fraction": 0}, "memory_reserve_fraction"),
            ({"memory_reserve_fraction": 1.1}, "memory_reserve_fraction"),
            ({"runtime_workspace_gb": -1}, "runtime_workspace_gb"),
            ({"cost_per_card_hour": -1}, "cost_per_card_hour"),
        ]
        for overrides, path in cases:
            with self.subTest(path=path):
                self.assert_invalid_path(
                    path, HardwareSpec.from_dict, hardware_config(**overrides)
                )

    def test_reports_missing_and_wrong_nested_mapping_paths(self):
        config = hardware_config()
        del config["compute_tflops"]
        self.assert_invalid_path(
            "compute_tflops", HardwareSpec.from_dict, config
        )
        self.assert_invalid_path(
            "interconnect", HardwareSpec.from_dict,
            hardware_config(interconnect=[]),
        )


class PrecisionSpecTests(PathAssertions, unittest.TestCase):
    def test_supports_w4a4_and_w4a8(self):
        hardware = HardwareSpec.from_dict(hardware_config())
        for mode, activation_bits in (("w4a4", 4), ("w4a8", 8)):
            with self.subTest(mode=mode):
                spec = PrecisionSpec.from_dict({
                    "gemm_mode": mode,
                    "weight_bits": 4,
                    "activation_bits": activation_bits,
                    "vector_bits": activation_bits,
                    "accumulator_bits": 32,
                    "kv_cache_bits": 8,
                })
                self.assertEqual(spec.gemm_mode, mode)
                self.assertIsNone(spec.validate_hardware(hardware))

    def test_w4a8_requires_exact_gemm_mode(self):
        config = hardware_config()
        del config["compute_tflops"]["gemm"]["w4a8"]
        spec = PrecisionSpec.from_dict({
            "gemm_mode": "w4a8", "weight_bits": 4,
            "activation_bits": 8, "vector_bits": 8,
            "accumulator_bits": 32, "kv_cache_bits": 8,
        })
        self.assert_invalid_path(
            "compute_tflops.gemm.w4a8",
            spec.validate_hardware,
            HardwareSpec.from_dict(config),
        )

    def test_precision_requires_vector_mode_for_vector_bits(self):
        config = hardware_config()
        del config["compute_tflops"]["vector"]["int8"]
        spec = PrecisionSpec.from_dict({
            "gemm_mode": "w4a8", "weight_bits": 4,
            "activation_bits": 8, "vector_bits": 8,
            "accumulator_bits": 32, "kv_cache_bits": 8,
        })
        self.assert_invalid_path(
            "compute_tflops.vector.int8",
            spec.validate_hardware,
            HardwareSpec.from_dict(config),
        )

    def test_rejects_invalid_bit_widths_types_and_empty_mode(self):
        base = {
            "gemm_mode": "w4a4", "weight_bits": 4,
            "activation_bits": 4, "vector_bits": 4,
            "accumulator_bits": 32, "kv_cache_bits": 8,
        }
        cases = [
            ("weight_bits", 2),
            ("activation_bits", True),
            ("vector_bits", "8"),
            ("accumulator_bits", 64),
            ("kv_cache_bits", 0),
            ("gemm_mode", ""),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                config = dict(base)
                config[field] = value
                self.assert_invalid_path(
                    field, PrecisionSpec.from_dict, config
                )


class ScenarioTests(PathAssertions, unittest.TestCase):
    def test_normalizes_workload_and_default_weight(self):
        scenario = WorkloadScenario.from_dict(scenario_config())
        self.assertEqual(scenario.name, "chat")
        self.assertEqual(scenario.input_length, 2048)
        self.assertEqual(scenario.output_length, 512)
        self.assertEqual(scenario.request_rate, 4.5)
        self.assertEqual(scenario.concurrency, 16)
        self.assertEqual(scenario.ttft_limit_ms, 200.0)
        self.assertEqual(scenario.tpot_limit_ms, 30.0)
        self.assertEqual(scenario.weight, 1.0)
        with self.assertRaises(FrozenInstanceError):
            scenario.weight = 2

    def test_accepts_all_and_weighted_policies(self):
        all_set = ScenarioSet.from_dict({
            "policy": "all", "scenarios": [scenario_config(weight=7)]
        })
        weighted_set = ScenarioSet.from_dict({
            "policy": "weighted",
            "scenarios": [scenario_config(name="a", weight=1),
                          scenario_config(name="b", weight=2)],
        })
        self.assertEqual(all_set.scenarios[0].weight, 7.0)
        self.assertEqual(weighted_set.policy, "weighted")
        self.assertIsInstance(weighted_set.scenarios, tuple)

    def test_rejects_empty_or_unsupported_scenario_set(self):
        cases = [
            ({"policy": "all", "scenarios": []}, "scenarios"),
            ({"policy": "any", "scenarios": [scenario_config()]}, "policy"),
            ({"policy": "all", "scenarios": "chat"}, "scenarios"),
        ]
        for config, path in cases:
            with self.subTest(path=path):
                self.assert_invalid_path(path, ScenarioSet.from_dict, config)

    def test_reports_nested_scenario_path(self):
        self.assert_invalid_path(
            "scenarios[0].input_length",
            ScenarioSet.from_dict,
            {"policy": "all", "scenarios": [scenario_config(input_length=0)]},
        )

    def test_rejects_duplicate_scenario_names(self):
        self.assert_invalid_path(
            "scenarios[1].name",
            ScenarioSet.from_dict,
            {"policy": "weighted",
             "scenarios": [scenario_config(), scenario_config()]},
        )

    def test_rejects_invalid_workload_numeric_values_and_bool(self):
        cases = [
            ("output_length", 0),
            ("concurrency", True),
            ("request_rate", 0),
            ("ttft_limit_ms", -1),
            ("tpot_limit_ms", False),
            ("weight", 0),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                self.assert_invalid_path(
                    field, WorkloadScenario.from_dict,
                    scenario_config(**{field: value}),
                )


class PDLinkSpecTests(PathAssertions, unittest.TestCase):
    def test_accepts_efficiency_boundaries(self):
        for efficiency in (0.01, 1):
            with self.subTest(efficiency=efficiency):
                link = PDLinkSpec.from_dict({
                    "bandwidth_gbps": 400,
                    "latency_us": 5,
                    "efficiency": efficiency,
                    "max_concurrent_transfers": 4,
                })
                self.assertEqual(link.efficiency, float(efficiency))

    def test_rejects_invalid_link_values(self):
        base = {
            "bandwidth_gbps": 400,
            "latency_us": 5,
            "efficiency": 0.8,
            "max_concurrent_transfers": 4,
        }
        cases = [
            ("bandwidth_gbps", 0),
            ("bandwidth_gbps", True),
            ("latency_us", -1),
            ("efficiency", 0),
            ("efficiency", 1.01),
            ("max_concurrent_transfers", 0),
            ("max_concurrent_transfers", True),
        ]
        for field, value in cases:
            with self.subTest(field=field):
                config = dict(base)
                config[field] = value
                self.assert_invalid_path(field, PDLinkSpec.from_dict, config)


class SearchSpaceTests(PathAssertions, unittest.TestCase):
    AXES = (
        "total_cards", "replicas", "attention_tp", "attention_dp",
        "moe_tp", "expert_parallel", "batch_sizes",
    )

    def test_defaults_all_axes_to_powers_of_two_through_max_cards(self):
        search = SearchSpace.from_dict(max_cards=8)
        for axis in self.AXES:
            with self.subTest(axis=axis):
                self.assertEqual(getattr(search, axis), (1, 2, 4, 8))

    def test_explicit_arrays_override_individual_axes_deterministically(self):
        search = SearchSpace.from_dict({
            "total_cards": [8, 2, 4],
            "batch_sizes": [16, 1, 8],
        }, max_cards=16)
        self.assertEqual(search.total_cards, (2, 4, 8))
        self.assertEqual(search.batch_sizes, (1, 8, 16))
        self.assertEqual(search.replicas, (1, 2, 4, 8, 16))

    def test_explicit_batch_sizes_are_not_limited_by_max_cards(self):
        search = SearchSpace.from_dict({"batch_sizes": [128]}, max_cards=64)
        self.assertEqual(search.batch_sizes, (128,))

    def test_rejects_invalid_axis_values_with_exact_paths(self):
        cases = [
            ({"total_cards": []}, "total_cards"),
            ({"replicas": [1, 1]}, "replicas[1]"),
            ({"attention_tp": [0]}, "attention_tp[0]"),
            ({"attention_dp": [-1]}, "attention_dp[0]"),
            ({"moe_tp": [True]}, "moe_tp[0]"),
            ({"expert_parallel": "1,2"}, "expert_parallel"),
            ({"total_cards": [1, 9]}, "total_cards[1]"),
        ]
        for config, path in cases:
            with self.subTest(path=path):
                self.assert_invalid_path(
                    path, SearchSpace.from_dict, config, 8
                )

    def test_rejects_invalid_max_cards(self):
        for value in (0, -1, True, 1.5):
            with self.subTest(value=value):
                self.assert_invalid_path(
                    "max_cards", SearchSpace.from_dict, None, value
                )


if __name__ == "__main__":
    unittest.main()
