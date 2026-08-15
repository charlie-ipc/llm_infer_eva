from copy import deepcopy

from infersim.schema.hardware import HardwareSpec
from infersim.schema.precision import PrecisionSpec


def make_hardware_dict(**overrides):
    config = {
        "name": "Test Accelerator",
        "memory_capacity_gb": 96,
        "memory_bandwidth_gbps": 4000,
        "cards_per_node": 8,
        "compute_tflops": {
            "gemm": {"w4a4": 1200, "w4a8": 900},
            "vector": {
                "fp4": 160,
                "int8": 120,
                "bf16": 60,
                "fp32": 30,
            },
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
    config.update(deepcopy(overrides))
    return config


def make_hardware(**overrides):
    return HardwareSpec.from_dict(make_hardware_dict(**overrides))


def _make_precision(default_gemm_mode, default_activation_bits, **overrides):
    config = {
        "gemm_mode": default_gemm_mode,
        "weight_bits": 4,
        "activation_bits": default_activation_bits,
        "vector_bits": default_activation_bits,
        "accumulator_bits": 32,
        "kv_cache_bits": 8,
    }
    config.update(overrides)
    return PrecisionSpec.from_dict(config)


def make_w4a8_precision(**overrides):
    return _make_precision("w4a8", 8, **overrides)


def make_w4a4_precision(**overrides):
    return _make_precision("w4a4", 4, **overrides)
