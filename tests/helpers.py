from copy import deepcopy

from infersim.cost import MemoryBreakdown, StageMetrics
from infersim.schema.hardware import HardwareSpec
from infersim.schema.model import ModelSpec
from infersim.schema.parallel import ParallelPlan, SearchSpace
from infersim.schema.precision import PrecisionSpec
from infersim.schema.scenario import PDLinkSpec, ScenarioSet, WorkloadScenario
from infersim.search import (
    SearchResult,
    StageCandidate,
    pareto_frontier,
    recommend,
)


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


def make_memory_bound_hardware(**overrides):
    values = {
        "memory_bandwidth_gbps": 1,
        "compute_tflops": {
            "gemm": {"w4a4": 1e9, "w4a8": 1e9},
            "vector": {
                "fp4": 1e9,
                "int8": 1e9,
                "bf16": 1e9,
                "fp32": 1e9,
            },
        },
        "kernel_launch_latency_us": {
            "gemm": 0,
            "vector": 0,
            "collective": 0,
        },
    }
    values.update(deepcopy(overrides))
    return make_hardware(**values)


def _make_precision(default_gemm_mode, default_activation_bits, **overrides):
    config = {
        "gemm_mode": default_gemm_mode,
        "weight_bits": 4,
        "activation_bits": default_activation_bits,
        "vector_bits": default_activation_bits,
        "accumulator_bits": 32,
        "kv_cache_bits": 8,
        "tp_reduce_bits": default_activation_bits,
        "ep_dispatch_bits": default_activation_bits,
        "ep_combine_bits": default_activation_bits,
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


def make_search_space(**overrides):
    values = {
        "total_cards": (1,),
        "replicas": (1,),
        "attention_tp": (1,),
        "attention_dp": (1,),
        "moe_tp": (1,),
        "expert_parallel": (1,),
        "batch_sizes": (2,),
    }
    values.update(deepcopy(overrides))
    return SearchSpace(**values)


def make_metrics(
    *,
    name="interactive",
    stage="prefill",
    ttft_ms=50.0,
    tpot_ms=10.0,
    request_capacity=10.0,
    memory_feasible=True,
    max_supported_concurrency=4,
    plan=None,
):
    plan = plan or make_dense_plan()
    latency_ms = ttft_ms if stage == "prefill" else tpot_ms
    memory = MemoryBreakdown(
        stage=stage,
        embedding_weight_bytes=1.0,
        attention_weight_bytes=1.0,
        linear_attention_weight_bytes=0.0,
        dense_ffn_weight_bytes=1.0,
        routed_expert_weight_bytes=0.0,
        shared_expert_weight_bytes=0.0,
        total_weight_bytes=3.0,
        kv_bytes_per_card=1.0,
        recurrent_state_bytes_per_card=0.0,
        activation_bytes_per_card=1.0,
        workspace_bytes=1.0,
        reserved_bytes=1.0,
        resident_bytes=5.0,
        total_required_bytes=7.0,
        capacity_bytes=10.0,
        usable_bytes=8.0,
        capacity_margin_bytes=3.0 if memory_feasible else -1.0,
        feasible=memory_feasible,
    )
    latency_seconds = latency_ms / 1000
    return StageMetrics(
        stage=stage,
        scenario_name=name,
        plan=plan,
        latency_seconds=latency_seconds,
        tpot_seconds=latency_seconds if stage == "decode" else None,
        prompt_token_capacity=(request_capacity * 128 if stage == "prefill" else None),
        output_token_capacity=(request_capacity * 32 if stage == "decode" else None),
        request_capacity=request_capacity,
        average_context_length=128.0,
        gemm_seconds=latency_seconds,
        vector_seconds=0.0,
        tp_seconds=0.0,
        ep_seconds=0.0,
        useful_gemm_ops=1,
        aligned_gemm_ops=1,
        useful_vector_ops=1,
        aligned_vector_ops=1,
        memory=memory,
        component_seconds={"gemm": latency_seconds},
        max_supported_batch=plan.batch_size,
        max_supported_concurrency=max_supported_concurrency,
    )


def make_stage_candidate(
    *,
    candidate_id="candidate",
    plan=None,
    metrics=None,
    feasible=True,
    reason_codes=(),
    warnings=(),
    total_cards=None,
    hourly_cost=None,
    request_capacity=0.0,
    request_capacity_per_card=0.0,
    ttft_ms=None,
    tpot_ms=None,
    scenarios=None,
):
    plan = plan or make_dense_plan()
    metrics = tuple(metrics) if metrics is not None else (
        make_metrics(plan=plan),
    )
    if scenarios is None:
        scenarios = tuple(
            make_scenario(
                name=metric.scenario_name,
                ttft_limit_ms=100.0,
                tpot_limit_ms=20.0,
                request_rate=1.0,
                concurrency=1,
            )
            for metric in metrics
        )
    return StageCandidate(
        candidate_id=candidate_id,
        plan=plan,
        metrics=metrics,
        feasible=feasible,
        reason_codes=reason_codes,
        warnings=warnings,
        total_cards=plan.total_cards if total_cards is None else total_cards,
        hourly_cost=hourly_cost,
        request_capacity=request_capacity,
        request_capacity_per_card=request_capacity_per_card,
        ttft_ms=ttft_ms,
        tpot_ms=tpot_ms,
        scenarios=scenarios,
    )


make_candidate = make_stage_candidate


def make_prefill_candidate(
    *,
    candidate_id="prefill",
    latency_ms=20.0,
    request_capacity=100.0,
    total_cards=1,
    hourly_cost=None,
    scenarios=None,
):
    plan = make_dense_plan(replicas=total_cards)
    scenario_values = tuple(scenarios) if scenarios is not None else (
        make_scenario(),
    )
    metrics = tuple(
        make_metrics(
            name=scenario.name,
            stage="prefill",
            ttft_ms=latency_ms,
            request_capacity=request_capacity,
            max_supported_concurrency=scenario.concurrency,
            plan=plan,
        )
        for scenario in scenario_values
    )
    return make_stage_candidate(
        candidate_id=candidate_id,
        plan=plan,
        metrics=metrics,
        total_cards=total_cards,
        hourly_cost=hourly_cost,
        request_capacity=request_capacity,
        request_capacity_per_card=request_capacity / total_cards,
        ttft_ms=latency_ms,
        scenarios=scenario_values,
    )


def make_decode_candidate(
    *,
    candidate_id="decode",
    tpot_ms=5.0,
    request_capacity=80.0,
    total_cards=1,
    hourly_cost=None,
    scenarios=None,
):
    plan = make_dense_plan(replicas=total_cards)
    scenario_values = tuple(scenarios) if scenarios is not None else (
        make_scenario(),
    )
    metrics = tuple(
        make_metrics(
            name=scenario.name,
            stage="decode",
            tpot_ms=tpot_ms,
            request_capacity=request_capacity,
            max_supported_concurrency=scenario.concurrency,
            plan=plan,
        )
        for scenario in scenario_values
    )
    return make_stage_candidate(
        candidate_id=candidate_id,
        plan=plan,
        metrics=metrics,
        total_cards=total_cards,
        hourly_cost=hourly_cost,
        request_capacity=request_capacity,
        request_capacity_per_card=request_capacity / total_cards,
        tpot_ms=tpot_ms,
        scenarios=scenario_values,
    )


def make_pd_link(**overrides):
    values = {
        "bandwidth_gbps": 100.0,
        "latency_us": 10.0,
        "efficiency": 1.0,
        "max_concurrent_transfers": 16,
    }
    values.update(deepcopy(overrides))
    return PDLinkSpec.from_dict(values)


def make_search_result(candidates, *, stage=None):
    ordered = tuple(sorted(candidates, key=lambda item: item.candidate_id))
    if stage is None:
        stage = ordered[0].metrics[0].stage
    feasible = tuple(candidate for candidate in ordered if candidate.feasible)
    frontier = tuple(
        sorted(pareto_frontier(ordered), key=lambda item: item.candidate_id)
    )
    return SearchResult(
        stage=stage,
        candidates=ordered,
        feasible_candidates=feasible,
        pareto_frontier=frontier,
        recommendation=recommend(ordered),
        dominant_rejection=None,
    )
