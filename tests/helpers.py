from copy import deepcopy

from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import ScenarioSet, WorkloadScenario


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


def make_dense_model(**overrides):
    config = {
        "model_type": "tiny-dense",
        "hidden_size": 8,
        "num_hidden_layers": 2,
        "vocab_size": 32,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "intermediate_size": 16,
        "tie_word_embeddings": True,
    }
    config.update(deepcopy(overrides))
    return ModelSpec.from_dict(config)


def make_mla_moe_model(**overrides):
    config = {
        "model_type": "tiny-mla-moe",
        "hidden_size": 16,
        "num_hidden_layers": 2,
        "vocab_size": 64,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 4,
        "q_lora_rank": 4,
        "kv_lora_rank": 3,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "v_head_dim": 3,
        "moe_intermediate_size": 8,
        "num_routed_experts": 4,
        "num_experts_per_tok": 2,
        "num_shared_experts": 2,
        "shared_expert_intermediate_size": 12,
        "tie_word_embeddings": True,
    }
    config.update(deepcopy(overrides))
    if config["q_lora_rank"] is None:
        del config["q_lora_rank"]
    return ModelSpec.from_dict(config)


def make_hybrid_model(**overrides):
    config = {
        "model_type": "tiny-hybrid",
        "hidden_size": 12,
        "num_hidden_layers": 3,
        "vocab_size": 48,
        "num_attention_heads": 3,
        "num_key_value_heads": 3,
        "head_dim": 4,
        "intermediate_size": 24,
        "tie_word_embeddings": True,
        "num_full_attention_layers": 1,
        "num_linear_attention_layers": 2,
        "linear_conv_kernel_dim": 3,
        "linear_key_head_dim": 2,
        "linear_num_key_heads": 2,
        "linear_value_head_dim": 3,
        "linear_num_value_heads": 2,
    }
    config.update(deepcopy(overrides))
    return ModelSpec.from_dict(config)


def make_dense_plan(**overrides):
    values = {
        "replicas": 1,
        "attention_tp": 1,
        "attention_dp": 1,
        "moe_tp": 1,
        "expert_parallel": 1,
        "batch_size": 2,
    }
    values.update(deepcopy(overrides))
    return ParallelPlan(**values)


def make_moe_plan(**overrides):
    values = {
        "replicas": 1,
        "attention_tp": 2,
        "attention_dp": 1,
        "moe_tp": 1,
        "expert_parallel": 2,
        "batch_size": 2,
    }
    values.update(deepcopy(overrides))
    return ParallelPlan(**values)


def make_scenario(**overrides):
    values = {
        "name": "interactive",
        "input_length": 128,
        "output_length": 32,
        "request_rate": 1,
        "concurrency": 4,
        "ttft_limit_ms": 100,
        "tpot_limit_ms": 20,
        "weight": 1,
    }
    values.update(deepcopy(overrides))
    return WorkloadScenario(**values)


def make_scenario_set(scenarios=None, **overrides):
    values = {
        "policy": "all",
        "scenarios": tuple(scenarios) if scenarios is not None else (make_scenario(),),
    }
    values.update(deepcopy(overrides))
    return ScenarioSet(**values)
