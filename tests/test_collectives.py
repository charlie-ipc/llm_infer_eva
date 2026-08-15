import math
import unittest
from dataclasses import FrozenInstanceError, fields, replace

from infersim.cost.collective import (
    CollectiveCost,
    activation_payload_bytes,
    all_reduce_cost,
    all_to_all_cost,
)
from infersim.errors import InputValidationError
from tests.helpers import (
    make_hardware,
    make_w4a4_precision,
    make_w4a8_precision,
)


class CollectiveCostTests(unittest.TestCase):
    def assert_invalid_path(self, expected_path, callable_, *args):
        with self.assertRaises(InputValidationError) as context:
            callable_(*args)
        self.assertEqual(context.exception.path, expected_path)

    def test_all_reduce_uses_ring_bytes_and_additive_intra_node_time(self):
        hardware = make_hardware(
            cards_per_node=8,
            kernel_launch_latency_us={
                "gemm": 0,
                "vector": 0,
                "collective": 0,
            },
            interconnect={
                "intra_node_gbps": 100,
                "intra_node_latency_us": 2,
                "inter_node_gbps": 20,
                "inter_node_latency_us": 10,
            },
        )

        cost = all_reduce_cost(1000, 4, hardware)

        self.assertEqual(cost.kind, "all_reduce")
        self.assertEqual(cost.payload_bytes, 1000.0)
        self.assertEqual(cost.group_size, 4)
        self.assertEqual(cost.path, "intra_node")
        self.assertEqual(cost.transfer_bytes, 1500.0)
        self.assertEqual(cost.bandwidth_seconds, 1500 / 100e9)
        self.assertEqual(cost.latency_seconds, 6 * 2e-6)
        self.assertEqual(cost.launch_seconds, 0.0)
        self.assertEqual(
            cost.seconds,
            cost.bandwidth_seconds + cost.latency_seconds,
        )

    def test_all_to_all_uses_exact_ring_fraction_and_inter_node_link(self):
        hardware = make_hardware(
            cards_per_node=4,
            kernel_launch_latency_us={
                "gemm": 0,
                "vector": 0,
                "collective": 0,
            },
            interconnect={
                "intra_node_gbps": 100,
                "intra_node_latency_us": 2,
                "inter_node_gbps": 25,
                "inter_node_latency_us": 7,
            },
        )

        cost = all_to_all_cost(800, 8, hardware)

        self.assertEqual(cost.kind, "all_to_all")
        self.assertEqual(cost.path, "inter_node")
        self.assertEqual(cost.transfer_bytes, 700.0)
        self.assertEqual(cost.bandwidth_seconds, 700 / 25e9)
        self.assertEqual(cost.latency_seconds, 7 * 7e-6)
        self.assertEqual(
            cost.seconds,
            cost.bandwidth_seconds + cost.latency_seconds,
        )

    def test_group_boundary_selects_intra_then_inter_node_path(self):
        hardware = make_hardware(cards_per_node=4)

        self.assertEqual(all_reduce_cost(1, 4, hardware).path, "intra_node")
        self.assertEqual(all_reduce_cost(1, 5, hardware).path, "inter_node")

    def test_single_member_collectives_are_strictly_zero(self):
        hardware = make_hardware()

        for function, kind in (
            (all_reduce_cost, "all_reduce"),
            (all_to_all_cost, "all_to_all"),
        ):
            with self.subTest(kind=kind):
                cost = function(123, 1, hardware)
                self.assertEqual(cost.kind, kind)
                self.assertEqual(cost.payload_bytes, 123.0)
                self.assertEqual(cost.group_size, 1)
                self.assertEqual(cost.path, "none")
                for field in (
                    "transfer_bytes",
                    "bandwidth_seconds",
                    "latency_seconds",
                    "launch_seconds",
                    "seconds",
                ):
                    self.assertEqual(getattr(cost, field), 0.0)

    def test_collective_launch_is_added_exactly_once(self):
        hardware = make_hardware(
            kernel_launch_latency_us={
                "gemm": 0,
                "vector": 0,
                "collective": 9,
            }
        )

        for function in (all_reduce_cost, all_to_all_cost):
            with self.subTest(function=function.__name__):
                cost = function(100, 4, hardware)
                self.assertEqual(cost.launch_seconds, 9e-6)
                self.assertEqual(
                    cost.seconds,
                    cost.bandwidth_seconds
                    + cost.latency_seconds
                    + 9e-6,
                )

    def test_activation_payload_uses_activation_precision_and_half_bytes(self):
        w4a4 = activation_payload_bytes(3, make_w4a4_precision())
        w4a8 = activation_payload_bytes(3, make_w4a8_precision())

        self.assertEqual(w4a4, 1.5)
        self.assertEqual(w4a8, 3.0)
        self.assertEqual(w4a8, 2 * w4a4)
        self.assertIsInstance(w4a4, float)
        self.assertEqual(activation_payload_bytes(0, make_w4a8_precision()), 0.0)

    def test_activation_payload_validates_elements_and_local_precision_bits(self):
        precision = make_w4a8_precision()
        for value in (-1, True, 1.5):
            with self.subTest(elements=value):
                self.assert_invalid_path(
                    "elements", activation_payload_bytes, value, precision
                )

        for value in (3, True):
            with self.subTest(activation_bits=value):
                self.assert_invalid_path(
                    "activation_bits",
                    activation_payload_bytes,
                    1,
                    replace(precision, activation_bits=value),
                )

    def test_collectives_validate_payload_and_group(self):
        hardware = make_hardware()
        for function in (all_reduce_cost, all_to_all_cost):
            for value in (-1, True, math.nan, math.inf, -math.inf):
                with self.subTest(function=function.__name__, payload=value):
                    self.assert_invalid_path(
                        "payload_bytes", function, value, 2, hardware
                    )
            for value in (0, -1, True, 1.5):
                with self.subTest(function=function.__name__, group=value):
                    self.assert_invalid_path(
                        "group_size", function, 1, value, hardware
                    )

    def test_huge_inputs_and_nonfinite_results_are_validation_errors(self):
        hardware = make_hardware()
        self.assert_invalid_path(
            "activation_payload_bytes",
            activation_payload_bytes,
            10**1000,
            make_w4a8_precision(),
        )
        for function in (all_reduce_cost, all_to_all_cost):
            with self.subTest(function=function.__name__, case="payload"):
                self.assert_invalid_path(
                    "payload_bytes", function, 10**1000, 2, hardware
                )
        self.assert_invalid_path(
            "transfer_bytes", all_reduce_cost, 1.7e308, 4, hardware
        )

        nonfinite_latency = replace(
            hardware,
            intra_node=replace(hardware.intra_node, latency_us=1e308),
        )
        self.assert_invalid_path(
            "latency_seconds",
            all_reduce_cost,
            1,
            4,
            nonfinite_latency,
        )
        self.assert_invalid_path(
            "latency_seconds",
            all_to_all_cost,
            1,
            10**308,
            hardware,
        )
        nonfinite_launch = replace(
            hardware, collective_launch_latency_us=math.inf
        )
        self.assert_invalid_path(
            "launch_seconds", all_to_all_cost, 1, 2, nonfinite_launch
        )

    def test_result_is_frozen_and_all_byte_and_time_fields_are_floats(self):
        cost = all_reduce_cost(100, 2, make_hardware())

        self.assertIsInstance(cost, CollectiveCost)
        with self.assertRaises(FrozenInstanceError):
            cost.seconds = 0
        for field in fields(cost):
            if field.name.endswith("bytes") or field.name.endswith("seconds"):
                self.assertIsInstance(getattr(cost, field.name), float)

    def test_public_symbols_are_exported_from_cost_package(self):
        from infersim.cost import (
            CollectiveCost as ExportedCollectiveCost,
            activation_payload_bytes as exported_activation_payload_bytes,
            all_reduce_cost as exported_all_reduce_cost,
            all_to_all_cost as exported_all_to_all_cost,
        )

        self.assertIs(ExportedCollectiveCost, CollectiveCost)
        self.assertIs(exported_activation_payload_bytes, activation_payload_bytes)
        self.assertIs(exported_all_reduce_cost, all_reduce_cost)
        self.assertIs(exported_all_to_all_cost, all_to_all_cost)


if __name__ == "__main__":
    unittest.main()
