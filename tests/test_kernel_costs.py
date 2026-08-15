import math
import unittest
from dataclasses import FrozenInstanceError
from inspect import signature

from infersim.cost import (
    KernelCost,
    gemm_cost,
    kernel_cost,
    vector_cost,
    vector_mode_for_bits,
)
from infersim.errors import InputValidationError
from tests.helpers import (
    make_hardware,
    make_hardware_dict,
    make_w4a4_precision,
    make_w4a8_precision,
)


class PathAssertions:
    def assert_invalid_path(self, expected_path, callable_, *args, **kwargs):
        with self.assertRaises(InputValidationError) as context:
            callable_(*args, **kwargs)
        self.assertEqual(context.exception.path, expected_path)


class KernelCostTests(PathAssertions, unittest.TestCase):
    def test_models_memory_bound_roofline(self):
        cost = kernel_cost(
            useful_ops=1,
            aligned_ops=1,
            compute_ops_per_second=1e9,
            memory_bytes=1000,
            memory_bandwidth_bytes_s=100,
            launch_seconds=0,
        )

        self.assertEqual(cost.useful_ops, 1)
        self.assertEqual(cost.aligned_ops, 1)
        self.assertEqual(cost.compute_seconds, 1e-9)
        self.assertEqual(cost.memory_bytes, 1000.0)
        self.assertEqual(cost.memory_seconds, 10.0)
        self.assertEqual(cost.launch_seconds, 0.0)
        self.assertEqual(cost.seconds, 10.0)
        self.assertEqual(cost.bottleneck, "memory")

    def test_compute_wins_ties_and_launch_is_added(self):
        cost = kernel_cost(
            useful_ops=0,
            aligned_ops=10,
            compute_ops_per_second=10,
            memory_bytes=100,
            memory_bandwidth_bytes_s=100,
            launch_seconds=0.25,
        )

        self.assertEqual(cost.compute_seconds, cost.memory_seconds)
        self.assertEqual(cost.seconds, 1.25)
        self.assertEqual(cost.bottleneck, "compute")

    def test_record_is_frozen(self):
        cost = KernelCost(1, 1, 1.0, 1.0, 1.0, 1.0, 2.0, "compute")
        with self.assertRaises(FrozenInstanceError):
            cost.seconds = 3

    def test_names_common_throughput_in_ops_per_second(self):
        parameters = signature(kernel_cost).parameters

        self.assertIn("compute_ops_per_second", parameters)
        self.assertNotIn("compute_tops", parameters)

    def test_rejects_invalid_common_inputs(self):
        base = {
            "useful_ops": 1,
            "aligned_ops": 1,
            "compute_ops_per_second": 1,
            "memory_bytes": 1,
            "memory_bandwidth_bytes_s": 1,
            "launch_seconds": 0,
        }
        cases = [
            ("useful_ops", -1),
            ("useful_ops", True),
            ("aligned_ops", -1),
            ("aligned_ops", False),
            ("compute_ops_per_second", 0),
            ("compute_ops_per_second", math.nan),
            ("compute_ops_per_second", math.inf),
            ("memory_bytes", -1),
            ("memory_bytes", True),
            ("memory_bytes", math.nan),
            ("memory_bandwidth_bytes_s", 0),
            ("memory_bandwidth_bytes_s", math.inf),
            ("launch_seconds", -1),
            ("launch_seconds", math.nan),
        ]
        for field, value in cases:
            with self.subTest(field=field, value=value):
                inputs = dict(base)
                inputs[field] = value
                self.assert_invalid_path(field, kernel_cost, **inputs)

    def test_normalizes_derived_float_overflow_to_validation_errors(self):
        base = {
            "useful_ops": 1,
            "aligned_ops": 1,
            "compute_ops_per_second": 1,
            "memory_bytes": 0,
            "memory_bandwidth_bytes_s": 1,
            "launch_seconds": 0,
        }
        cases = [
            ("aligned_ops", {"aligned_ops": 10**1000}),
            ("memory_bytes", {"memory_bytes": 10**1000}),
            (
                "seconds",
                {"aligned_ops": 10**308, "launch_seconds": 1e308},
            ),
        ]
        for path, overrides in cases:
            with self.subTest(path=path):
                inputs = dict(base)
                inputs.update(overrides)
                self.assert_invalid_path(path, kernel_cost, **inputs)


class GemmCostTests(PathAssertions, unittest.TestCase):
    def test_aligns_tiles_across_engines(self):
        hardware = make_hardware(
            compute_tflops={
                "gemm": {"w4a8": 1},
                "vector": {"int8": 1},
            },
            gemm_tile={"m": 16, "n": 16, "k": 16},
            gemm_engines=2,
            memory_bandwidth_gbps=1e12,
            kernel_launch_latency_us={
                "gemm": 0,
                "vector": 0,
                "collective": 0,
            },
        )

        cost = gemm_cost(1, 1, 1, hardware, make_w4a8_precision())

        self.assertEqual(cost.useful_ops, 2)
        self.assertEqual(cost.aligned_ops, 2 * 2 * 16**3)
        self.assertEqual(cost.compute_seconds, cost.aligned_ops / 1e12)

    def test_uses_w4a8_memory_formula_and_ignores_accumulator_bits(self):
        cost = gemm_cost(
            2,
            3,
            5,
            make_hardware(),
            make_w4a8_precision(accumulator_bits=16),
        )

        expected = 2 * 3 * 8 / 8 + 3 * 5 * 4 / 8 + 2 * 5 * 8 / 8
        self.assertEqual(cost.memory_bytes, expected)

    def test_uses_w4a4_memory_formula(self):
        cost = gemm_cost(
            2, 3, 5, make_hardware(), make_w4a4_precision()
        )

        expected = 2 * 3 * 4 / 8 + 3 * 5 * 4 / 8 + 2 * 5 * 4 / 8
        self.assertEqual(cost.memory_bytes, expected)

    def test_repeats_share_engine_alignment_and_launch_once(self):
        hardware = make_hardware(
            compute_tflops={
                "gemm": {"w4a8": 1},
                "vector": {"int8": 1},
            },
            gemm_tile={"m": 1, "n": 1, "k": 1},
            gemm_engines=2,
            memory_bandwidth_gbps=1e12,
            kernel_launch_latency_us={
                "gemm": 7,
                "vector": 0,
                "collective": 0,
            },
        )

        cost = gemm_cost(
            1, 1, 1, hardware, make_w4a8_precision(), repeats=3
        )

        self.assertEqual(cost.useful_ops, 6)
        self.assertEqual(cost.aligned_ops, 8)
        self.assertEqual(cost.memory_bytes, 7.5)
        self.assertEqual(cost.launch_seconds, 7e-6)
        self.assertEqual(
            cost.seconds,
            max(cost.compute_seconds, cost.memory_seconds) + 7e-6,
        )

    def test_only_requires_requested_gemm_mode(self):
        hardware = make_hardware(
            compute_tflops={
                "gemm": {"w4a8": 1},
                "vector": {"fp4": 1},
            }
        )

        self.assertIsInstance(
            gemm_cost(1, 1, 1, hardware, make_w4a8_precision()),
            KernelCost,
        )

    def test_reports_missing_gemm_mode_with_full_path(self):
        self.assert_invalid_path(
            "compute_tflops.gemm.missing",
            gemm_cost,
            1,
            1,
            1,
            make_hardware(),
            make_w4a8_precision(gemm_mode="missing"),
        )

    def test_rejects_invalid_dimensions_and_repeats(self):
        hardware = make_hardware()
        precision = make_w4a8_precision()
        for field, args in (
            ("m", (0, 1, 1, hardware, precision)),
            ("k", (1, True, 1, hardware, precision)),
            ("n", (1, 1, -1, hardware, precision)),
        ):
            with self.subTest(field=field):
                self.assert_invalid_path(field, gemm_cost, *args)
        self.assert_invalid_path(
            "repeats",
            gemm_cost,
            1,
            1,
            1,
            hardware,
            precision,
            repeats=False,
        )

    def test_normalizes_huge_dimensions_and_repeats_to_memory_path(self):
        hardware = make_hardware()
        precision = make_w4a8_precision()
        cases = [
            ((10**1000, 1, 1), 1),
            ((1, 1, 1), 10**1000),
        ]
        for dimensions, repeats in cases:
            with self.subTest(dimensions=dimensions, repeats=repeats):
                self.assert_invalid_path(
                    "memory_bytes",
                    gemm_cost,
                    *dimensions,
                    hardware,
                    precision,
                    repeats=repeats,
                )

    def test_rejects_nonfinite_scaled_gemm_throughput(self):
        hardware = make_hardware(
            compute_tflops={
                "gemm": {"w4a8": 1e300},
                "vector": {"int8": 1},
            }
        )

        self.assert_invalid_path(
            "compute_ops_per_second",
            gemm_cost,
            1,
            1,
            1,
            hardware,
            make_w4a8_precision(),
        )


class VectorCostTests(PathAssertions, unittest.TestCase):
    def test_aligns_each_repeat_to_full_vector_wave(self):
        hardware = make_hardware(
            compute_tflops={
                "gemm": {"w4a8": 1},
                "vector": {"int8": 1},
            },
            vector_width=16,
            vector_units=2,
            memory_bandwidth_gbps=1e12,
            kernel_launch_latency_us={
                "gemm": 0,
                "vector": 0,
                "collective": 0,
            },
        )

        cost = vector_cost(17, 1, "int8", hardware, repeats=3)

        self.assertEqual(cost.useful_ops, 51)
        self.assertEqual(cost.aligned_ops, 96)
        self.assertEqual(cost.compute_seconds, 96 / 1e12)
        self.assertEqual(cost.memory_bytes, 102.0)

    def test_explicit_memory_bytes_override_and_launch_once(self):
        hardware = make_hardware(
            kernel_launch_latency_us={
                "gemm": 0,
                "vector": 11,
                "collective": 0,
            }
        )

        cost = vector_cost(
            17, 2, "int8", hardware, repeats=3, memory_bytes=12.5
        )

        self.assertEqual(cost.memory_bytes, 12.5)
        self.assertEqual(cost.launch_seconds, 11e-6)
        self.assertEqual(
            cost.seconds,
            max(cost.compute_seconds, cost.memory_seconds) + 11e-6,
        )

    def test_maps_canonical_vector_modes(self):
        self.assertEqual(
            {bits: vector_mode_for_bits(bits) for bits in (4, 8, 16, 32)},
            {4: "fp4", 8: "int8", 16: "bf16", 32: "fp32"},
        )

    def test_reports_unknown_bits_and_missing_mode_with_full_paths(self):
        self.assert_invalid_path("bits", vector_mode_for_bits, 2)
        self.assert_invalid_path("bits", vector_mode_for_bits, True)
        self.assert_invalid_path(
            "compute_tflops.vector.missing",
            vector_cost,
            1,
            1,
            "missing",
            make_hardware(),
        )

    def test_rejects_invalid_structure_and_memory_inputs(self):
        hardware = make_hardware()
        cases = [
            ("elements", (0, 1, "int8"), {}),
            ("elements", (True, 1, "int8"), {}),
            ("ops_per_element", (1, 0, "int8"), {}),
            ("ops_per_element", (1, False, "int8"), {}),
            ("vector_mode", (1, 1, ""), {}),
            ("vector_mode", (1, 1, 8), {}),
            ("repeats", (1, 1, "int8"), {"repeats": 0}),
            ("memory_bytes", (1, 1, "int8"), {"memory_bytes": -1}),
            ("memory_bytes", (1, 1, "int8"), {"memory_bytes": True}),
            (
                "memory_bytes",
                (1, 1, "int8"),
                {"memory_bytes": math.inf},
            ),
        ]
        for field, args, kwargs in cases:
            with self.subTest(field=field, args=args, kwargs=kwargs):
                self.assert_invalid_path(
                    field, vector_cost, *args, hardware, **kwargs
                )

    def test_normalizes_huge_structure_values_to_derived_paths(self):
        hardware = make_hardware()
        cases = [
            ("memory_bytes", (10**1000, 1), {"repeats": 1}),
            (
                "aligned_ops",
                (1, 10**1000),
                {"repeats": 1, "memory_bytes": 0},
            ),
            ("memory_bytes", (1, 1), {"repeats": 10**1000}),
        ]
        for path, args, kwargs in cases:
            with self.subTest(path=path, args=args):
                self.assert_invalid_path(
                    path, vector_cost, *args, "int8", hardware, **kwargs
                )

    def test_rejects_nonfinite_scaled_vector_bandwidth(self):
        hardware = make_hardware(memory_bandwidth_gbps=1e300)

        self.assert_invalid_path(
            "memory_bandwidth_bytes_s",
            vector_cost,
            1,
            1,
            "int8",
            hardware,
        )


class HelperTests(unittest.TestCase):
    def test_hardware_dict_overrides_do_not_mutate_other_calls(self):
        first = make_hardware_dict()
        first["compute_tflops"]["gemm"]["w4a8"] = 1

        self.assertEqual(
            make_hardware_dict()["compute_tflops"]["gemm"]["w4a8"], 900
        )


if __name__ == "__main__":
    unittest.main()
