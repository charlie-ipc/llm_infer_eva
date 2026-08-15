# PD-Aware Parallel Configuration Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add custom-accelerator, W4A4/W4A8, stage-specific TP/DP/EP search and PD KV-transfer-aware pairing to InferSim while preserving its existing command line.

**Architecture:** Build a new dependency-free `infersim` package beside the existing modules. Pure dataclasses normalize model, hardware, precision, workload, and parallel plans; pure analytical cost functions evaluate prefill and decode; search and reporting modules enumerate, constrain, rank, pair, and serialize results. The legacy `main.py` remains unchanged until a final compatibility pass.

**Tech Stack:** Python 3.10+ standard library, `dataclasses`, `json`, `csv`, `argparse`, `unittest`, existing InferSim formulas and fixtures.

---

## File Map

New production files:

- `infersim/__init__.py`: public package exports and version.
- `infersim/__main__.py`: `python -m infersim` entry point.
- `infersim/errors.py`: structured input and unsupported-model errors.
- `infersim/schema/model.py`: Hugging Face model normalization.
- `infersim/schema/hardware.py`: accelerator and interconnect schema.
- `infersim/schema/precision.py`: arithmetic and storage precision schema.
- `infersim/schema/scenario.py`: scenario grid and policy schema.
- `infersim/schema/parallel.py`: parallel plan and search-space schema.
- `infersim/cost/types.py`: immutable timing, memory, and stage result records.
- `infersim/cost/kernels.py`: GEMM, vector, and DRAM roofline functions.
- `infersim/cost/operations.py`: model operation shapes and byte counts.
- `infersim/cost/memory.py`: sharded weights, KV/state, and capacity accounting.
- `infersim/cost/collective.py`: TP/EP communication model.
- `infersim/cost/stage.py`: independent prefill and decode evaluators.
- `infersim/cost/pd.py`: KV/state transfer and paired-stage metrics.
- `infersim/search/enumerate.py`: legal candidate generation.
- `infersim/search/constraints.py`: feasibility and reason codes.
- `infersim/search/pareto.py`: deterministic dominance and recommendation order.
- `infersim/search/runner.py`: stage search orchestration.
- `infersim/search/pair.py`: pruned PD candidate pairing.
- `infersim/report.py`: stable CSV, JSON, and text outputs.
- `infersim/cli.py`: `search` and `pair-pd` argument parsing.

New tests and fixtures:

- `tests/helpers.py`: compact model/hardware/scenario factories.
- `tests/test_model_schema.py`
- `tests/test_input_schemas.py`
- `tests/test_parallel_search_space.py`
- `tests/test_kernel_costs.py`
- `tests/test_operation_counts.py`
- `tests/test_memory_costs.py`
- `tests/test_collectives.py`
- `tests/test_prefill_evaluator.py`
- `tests/test_decode_evaluator.py`
- `tests/test_constraints_and_pareto.py`
- `tests/test_reporting.py`
- `tests/test_pd_pairing.py`
- `tests/test_cli.py`
- `tests/test_legacy_cli.py`
- `examples/search/custom_npu.json`
- `examples/search/w4a8.json`
- `examples/search/scenarios.json`
- `examples/search/search_space.json`
- `examples/search/pd_link.json`

Modified files:

- `README.md`: document analytical search, inputs, commands, and limits.
- `.gitignore`: ignore generated search output directories.

## Task 1: Package Foundation and Model Normalization

**Files:**
- Create: `infersim/__init__.py`
- Create: `infersim/errors.py`
- Create: `infersim/schema/__init__.py`
- Create: `infersim/schema/model.py`
- Create: `tests/__init__.py`
- Create: `tests/test_model_schema.py`

- [ ] **Step 1: Write failing normalization tests**

Create tests for dense MHA, MoE MLA, wrapped text config, hybrid linear
attention, field aliases, encoder-decoder rejection, multimodal rejection, and
missing-field paths. The central tests must include:

```python
import unittest

from infersim.errors import InputValidationError, UnsupportedModelError
from infersim.schema.model import ModelSpec


class ModelSpecTests(unittest.TestCase):
    def test_normalizes_dense_gqa(self):
        spec = ModelSpec.from_dict({
            "model_type": "example",
            "hidden_size": 4096,
            "num_hidden_layers": 32,
            "vocab_size": 32000,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "intermediate_size": 11008,
            "tie_word_embeddings": True,
        })
        self.assertEqual(spec.attention_kind, "gqa")
        self.assertEqual(spec.head_dim, 128)
        self.assertFalse(spec.is_moe)

    def test_normalizes_moe_aliases(self):
        spec = ModelSpec.from_dict({
            "model_type": "example_moe",
            "hidden_size": 1024,
            "num_hidden_layers": 8,
            "vocab_size": 4096,
            "num_attention_heads": 16,
            "num_key_value_heads": 4,
            "num_experts": 64,
            "num_experts_per_token": 4,
            "moe_intermediate_size": 256,
        })
        self.assertEqual(spec.num_routed_experts, 64)
        self.assertEqual(spec.experts_per_token, 4)

    def test_rejects_encoder_decoder(self):
        with self.assertRaisesRegex(UnsupportedModelError, "encoder-decoder"):
            ModelSpec.from_dict({"model_type": "t5", "is_encoder_decoder": True})

    def test_reports_missing_field_path(self):
        with self.assertRaisesRegex(InputValidationError, "hidden_size"):
            ModelSpec.from_dict({"model_type": "broken"})
```

- [ ] **Step 2: Run tests and confirm the import failure**

Run:

```powershell
python -m unittest tests.test_model_schema -v
```

Expected: `ModuleNotFoundError: No module named 'infersim'`.

- [ ] **Step 3: Implement structured errors and immutable `ModelSpec`**

Use an error carrying a JSON path:

```python
class InputValidationError(ValueError):
    def __init__(self, path: str, message: str):
        self.path = path
        self.message = message
        super().__init__(f"{path}: {message}")


class UnsupportedModelError(InputValidationError):
    pass
```

Implement `ModelSpec.from_dict()` as a strict normalizer. It must:

- Reject `is_encoder_decoder=True`.
- Reject roots containing `vision_config` or `vision_config_dict`.
- Unwrap `text_config` when present.
- Infer `head_dim` only when `hidden_size % num_attention_heads == 0`.
- Infer MHA/MQA/GQA from head counts and MLA from `kv_lora_rank`.
- Accept `num_experts_per_tok` and `num_experts_per_token`.
- Accept `num_routed_experts` and `num_experts`.
- Validate positive dimensions, `experts_per_token <= num_routed_experts`, and
  shared-expert intermediate-size divisibility.
- Capture all approved full/linear-attention fields as optional integers.

The dataclass signature is:

```python
@dataclass(frozen=True)
class ModelSpec:
    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    attention_kind: str
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    num_routed_experts: int
    experts_per_token: int
    num_shared_experts: int
    shared_expert_intermediate_size: int
    tie_word_embeddings: bool
    attention_output_gate: bool
    num_full_attention_layers: int
    num_linear_attention_layers: int
    q_lora_rank: int | None = None
    kv_lora_rank: int | None = None
    qk_nope_head_dim: int | None = None
    qk_rope_head_dim: int | None = None
    v_head_dim: int | None = None
    linear_conv_kernel_dim: int | None = None
    linear_key_head_dim: int | None = None
    linear_num_key_heads: int | None = None
    linear_value_head_dim: int | None = None
    linear_num_value_heads: int | None = None

    @property
    def is_moe(self) -> bool:
        return self.num_routed_experts > 1
```

- [ ] **Step 4: Run model tests**

Run: `python -m unittest tests.test_model_schema -v`

Expected: all model-schema tests pass.

- [ ] **Step 5: Commit the model schema**

```powershell
git add infersim tests/test_model_schema.py tests/__init__.py
git commit -m "feat: normalize decoder-only model configs"
```

## Task 2: Hardware, Precision, Scenario, and Search-Space Schemas

**Files:**
- Create: `infersim/schema/hardware.py`
- Create: `infersim/schema/precision.py`
- Create: `infersim/schema/scenario.py`
- Create: `infersim/schema/parallel.py`
- Create: `tests/test_input_schemas.py`

- [ ] **Step 1: Write failing schema tests**

Cover valid custom hardware, W4A4/W4A8, missing throughput keys, invalid units,
all-scenarios/weighted policies, PD link validation, and explicit search grids:

```python
class InputSchemaTests(unittest.TestCase):
    def test_w4a8_requires_gemm_and_vector_modes(self):
        precision = PrecisionSpec.from_dict({
            "gemm_mode": "w4a8",
            "weight_bits": 4,
            "activation_bits": 8,
            "vector_bits": 8,
            "accumulator_bits": 16,
            "kv_cache_bits": 8,
        })
        hardware = make_hardware()
        precision.validate_hardware(hardware)

    def test_missing_w4a8_throughput_names_exact_path(self):
        hardware = make_hardware(gemm_modes={"bf16": 100.0})
        with self.assertRaisesRegex(InputValidationError,
                                    "compute_tflops.gemm.w4a8"):
            make_w4a8_precision().validate_hardware(hardware)

    def test_rejects_nonpositive_bandwidth(self):
        data = make_hardware_dict()
        data["memory_bandwidth_gbps"] = 0
        with self.assertRaisesRegex(InputValidationError,
                                    "memory_bandwidth_gbps"):
            HardwareSpec.from_dict(data)
```

- [ ] **Step 2: Run tests and verify missing classes**

Run: `python -m unittest tests.test_input_schemas -v`

Expected: import failures for `HardwareSpec`, `PrecisionSpec`, and scenario types.

- [ ] **Step 3: Implement strict immutable schemas**

Implement these public types:

```python
@dataclass(frozen=True)
class InterconnectSpec:
    bandwidth_gbps: float
    latency_us: float


@dataclass(frozen=True)
class HardwareSpec:
    name: str
    memory_capacity_gb: float
    memory_bandwidth_gbps: float
    cards_per_node: int
    gemm_tflops: Mapping[str, float]
    vector_tflops: Mapping[str, float]
    gemm_tile: tuple[int, int, int]
    gemm_engines: int
    vector_width: int
    vector_units: int
    gemm_launch_latency_us: float
    vector_launch_latency_us: float
    collective_launch_latency_us: float
    intra_node: InterconnectSpec
    inter_node: InterconnectSpec
    memory_reserve_fraction: float
    runtime_workspace_gb: float
    cost_per_card_hour: float | None


@dataclass(frozen=True)
class PrecisionSpec:
    gemm_mode: str
    weight_bits: int
    activation_bits: int
    vector_bits: int
    accumulator_bits: int
    kv_cache_bits: int


@dataclass(frozen=True)
class WorkloadScenario:
    name: str
    input_length: int
    output_length: int
    request_rate: float
    concurrency: int
    ttft_limit_ms: float
    tpot_limit_ms: float
    weight: float = 1.0


@dataclass(frozen=True)
class ScenarioSet:
    policy: str
    scenarios: tuple[WorkloadScenario, ...]


@dataclass(frozen=True)
class PDLinkSpec:
    bandwidth_gbps: float
    latency_us: float
    efficiency: float
    max_concurrent_transfers: int
```

`SearchSpace` contains tuples for `total_cards`, `replicas`, `attention_tp`,
`attention_dp`, `moe_tp`, `expert_parallel`, and `batch_sizes`. Defaults are
powers of two bounded by `max_cards=64`; explicit JSON arrays override each
axis. Reject duplicates, zero, negatives, unsupported policies, empty scenario
lists, bit widths outside `{4, 8, 16, 32}`, and reserve/efficiency values
outside `(0, 1]` as applicable.

- [ ] **Step 4: Run schema tests**

Run: `python -m unittest tests.test_input_schemas -v`

Expected: all input-schema tests pass.

- [ ] **Step 5: Commit input schemas**

```powershell
git add infersim/schema tests/test_input_schemas.py
git commit -m "feat: add custom hardware and workload schemas"
```

## Task 3: Parallel Plan Validation and Enumeration

**Files:**
- Modify: `infersim/schema/parallel.py`
- Create: `infersim/search/__init__.py`
- Create: `infersim/search/enumerate.py`
- Create: `tests/test_parallel_search_space.py`

- [ ] **Step 1: Write failing parallel-plan tests**

```python
class ParallelSearchSpaceTests(unittest.TestCase):
    def test_moe_width_equation(self):
        plan = ParallelPlan(
            replicas=2,
            attention_tp=2,
            attention_dp=2,
            moe_tp=1,
            expert_parallel=4,
            batch_size=16,
        )
        self.assertEqual(plan.cards_per_replica, 4)
        self.assertEqual(plan.total_cards, 8)

    def test_enumerator_records_invalid_head_divisibility(self):
        model = make_dense_model(num_attention_heads=12,
                                 num_key_value_heads=3)
        results = list(enumerate_plans(model, make_search_space(attention_tp=(4,))))
        self.assertEqual(results[0].reason_code, "KV_HEADS_NOT_DIVISIBLE")

    def test_dense_plan_disallows_attention_dp_and_ep(self):
        result = validate_plan(
            make_dense_model(),
            ParallelPlan(1, 1, 2, 1, 2, 8),
        )
        self.assertEqual(result.reason_code, "DENSE_PARALLELISM_INVALID")
```

- [ ] **Step 2: Run tests and confirm failures**

Run: `python -m unittest tests.test_parallel_search_space -v`

Expected: imports fail for `ParallelPlan` and `enumerate_plans`.

- [ ] **Step 3: Implement plans, validation, and deterministic enumeration**

Use:

```python
@dataclass(frozen=True, order=True)
class ParallelPlan:
    replicas: int
    attention_tp: int
    attention_dp: int
    moe_tp: int
    expert_parallel: int
    batch_size: int

    @property
    def cards_per_replica(self) -> int:
        return self.attention_tp * self.attention_dp

    @property
    def total_cards(self) -> int:
        return self.replicas * self.cards_per_replica


@dataclass(frozen=True)
class PlanValidation:
    plan: ParallelPlan
    feasible: bool
    reason_code: str | None = None
    reason: str | None = None
```

Validation order and stable codes:

1. `NONPOSITIVE_PARALLELISM`.
2. `TOTAL_CARDS_MISMATCH` against the selected total-card axis.
3. `DENSE_PARALLELISM_INVALID`.
4. `MOE_WIDTH_MISMATCH` for
   `attention_tp * attention_dp != moe_tp * expert_parallel`.
5. `ATTENTION_HEADS_NOT_DIVISIBLE`.
6. `KV_HEADS_NOT_DIVISIBLE` for MHA/MQA/GQA only.
7. `INTERMEDIATE_NOT_DIVISIBLE` for dense/`moe_tp` FFN sharding.
8. `EXPERTS_NOT_DIVISIBLE`.

Enumeration sorts every input axis, de-duplicates complete plans, and yields
both valid and invalid `PlanValidation` records so diagnostics remain complete.

- [ ] **Step 4: Run parallel tests**

Run: `python -m unittest tests.test_parallel_search_space -v`

Expected: all plan and enumeration tests pass.

- [ ] **Step 5: Commit parallel enumeration**

```powershell
git add infersim/schema/parallel.py infersim/search tests/test_parallel_search_space.py
git commit -m "feat: enumerate legal TP DP EP plans"
```

## Task 4: GEMM, Vector, and DRAM Kernel Cost Primitives

**Files:**
- Create: `infersim/cost/__init__.py`
- Create: `infersim/cost/types.py`
- Create: `infersim/cost/kernels.py`
- Create: `tests/helpers.py`
- Create: `tests/test_kernel_costs.py`

- [ ] **Step 1: Write exact failing cost tests**

```python
class KernelCostTests(unittest.TestCase):
    def test_gemm_rounds_to_tile_and_engine_wave(self):
        hardware = make_hardware(
            gemm_tflops={"w4a8": 1.0},
            gemm_tile=(16, 16, 16),
            gemm_engines=2,
            memory_bandwidth_gbps=1e9,
            launch_us=0,
        )
        cost = gemm_cost(1, 1, 1, hardware, make_w4a8_precision())
        self.assertEqual(cost.useful_ops, 2)
        self.assertEqual(cost.aligned_ops, 2 * 2 * 16 * 16 * 16)
        self.assertAlmostEqual(cost.compute_seconds,
                               cost.aligned_ops / 1e12)

    def test_vector_rounds_to_full_unit_wave(self):
        hardware = make_hardware(vector_width=16, vector_units=2,
                                 vector_tflops={"int8": 1.0}, launch_us=0)
        cost = vector_cost(17, 1, "int8", hardware)
        self.assertEqual(cost.aligned_ops, 32)

    def test_roofline_uses_slower_component(self):
        cost = kernel_cost(useful_ops=1, aligned_ops=1,
                           compute_tops=1e9, memory_bytes=1000,
                           memory_bandwidth_bytes_s=100, launch_seconds=0)
        self.assertEqual(cost.seconds, 10)
        self.assertEqual(cost.bottleneck, "memory")
```

- [ ] **Step 2: Run tests and confirm imports fail**

Run: `python -m unittest tests.test_kernel_costs -v`

Expected: missing `infersim.cost` imports.

- [ ] **Step 3: Implement immutable kernel cost records and formulas**

```python
@dataclass(frozen=True)
class KernelCost:
    useful_ops: int
    aligned_ops: int
    compute_seconds: float
    memory_bytes: float
    memory_seconds: float
    launch_seconds: float
    seconds: float
    bottleneck: str
```

`gemm_cost(m, k, n, hardware, precision, repeats=1)` computes tile count,
rounds it to a complete `gemm_engines` wave, uses
`hardware.gemm_tflops[precision.gemm_mode] * 1e12`, and accounts for input,
weight, and output bytes using activation/weight bits. `vector_cost()` rounds
chunks to a complete `vector_units` wave and uses the vector mode mapped from
`vector_bits`: `4 -> fp4`, `8 -> int8`, `16 -> bf16`, `32 -> fp32`.

The common roofline function returns:

```python
seconds = max(compute_seconds, memory_seconds) + launch_seconds
bottleneck = "compute" if compute_seconds >= memory_seconds else "memory"
```

Reject missing throughput keys with their full hardware JSON path.

- [ ] **Step 4: Run kernel tests**

Run: `python -m unittest tests.test_kernel_costs -v`

Expected: all exact alignment and roofline tests pass.

- [ ] **Step 5: Commit kernel primitives**

```powershell
git add infersim/cost tests/helpers.py tests/test_kernel_costs.py
git commit -m "feat: model tiled GEMM and vector kernels"
```

## Task 5: Operation Shapes and Model Byte Counts

**Files:**
- Create: `infersim/cost/operations.py`
- Create: `tests/test_operation_counts.py`

- [ ] **Step 1: Write failing operation-shape tests**

Use a tiny dense model with hidden size 8, two query heads, one KV head,
head-dim 4, FFN size 16, two layers, and vocabulary 32. Assert:

```python
class OperationCountTests(unittest.TestCase):
    def test_dense_weight_elements(self):
        model = make_dense_model(hidden_size=8, num_hidden_layers=2,
                                 num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=4,
                                 intermediate_size=16, vocab_size=32)
        counts = model_counts(model)
        # Per layer: Q/K/V/O = 64+32+32+64, FFN = 3*8*16.
        # Tied embedding = 32*8.
        self.assertEqual(counts.total_weight_elements,
                         2 * (192 + 384) + 256)

    def test_mha_kv_elements_per_token(self):
        model = make_dense_model(hidden_size=8, num_hidden_layers=2,
                                 num_attention_heads=2,
                                 num_key_value_heads=1, head_dim=4)
        self.assertEqual(kv_elements_per_token(model), 2 * 2 * 1 * 4)

    def test_prefill_shapes_use_batch_times_input_length(self):
        ops = stage_operations(make_dense_model(), stage="prefill",
                               batch_size=3, input_length=5,
                               average_context=5, plan=make_dense_plan())
        self.assertEqual(ops.gemms[0].m, 15)
```

Add MLA KV, shared/routed MoE, attention output gate, untied LM head, and hybrid
state-count cases.

- [ ] **Step 2: Run tests and verify missing operation builder**

Run: `python -m unittest tests.test_operation_counts -v`

Expected: missing `infersim.cost.operations`.

- [ ] **Step 3: Implement explicit operation descriptors and formulas**

Create:

```python
@dataclass(frozen=True)
class GemmShape:
    name: str
    m: int
    k: int
    n: int
    repeats: int = 1


@dataclass(frozen=True)
class VectorShape:
    name: str
    elements: int
    ops_per_element: int
    repeats: int = 1


@dataclass(frozen=True)
class StageOperations:
    gemms: tuple[GemmShape, ...]
    vectors: tuple[VectorShape, ...]
```

Use the existing `flops/flops.py` and `params/params.py` equations, expressed
as integer shapes:

- MHA/GQA Q, K, V, O projections shard head outputs by `attention_tp`.
- MHA/GQA QK and PV use local heads and the stage context length.
- MLA uses q-down/up, kv-down, q-WK, O-WV, O projection, and absorb/no-absorb
  attention shapes matching existing InferSim behavior.
- Dense FFN emits gate/up/down GEMMs with intermediate size divided by
  `attention_tp`.
- Routed MoE assigns `batch_tokens * experts_per_token` tokens across local
  experts, uses ceiling tokens per local active expert, emits three GEMMs per
  active expert, and shards intermediate size by `moe_tp`.
- Shared experts are replicated across EP and sharded by `moe_tp`.
- Vector coefficients are explicit constants: two RMSNorm passes at five ops
  per element, two residuals at one op per element, RoPE at six ops per rotated
  element, softmax at five ops per attention-score element, SiLU/gating at six
  ops per FFN intermediate element, and MoE routing at four ops per
  expert-score element.

Implement full-model weight elements, per-token KV elements, and hybrid
recurrent-state bytes. Do not mix precision bits into element counts.

- [ ] **Step 4: Run operation tests**

Run: `python -m unittest tests.test_operation_counts -v`

Expected: all dense, MLA, MoE, and hybrid count tests pass.

- [ ] **Step 5: Commit operation modeling**

```powershell
git add infersim/cost/operations.py tests/test_operation_counts.py
git commit -m "feat: describe decoder operation shapes"
```

## Task 6: Parallel-Aware Memory Accounting

**Files:**
- Create: `infersim/cost/memory.py`
- Create: `tests/test_memory_costs.py`

- [ ] **Step 1: Write failing memory tests**

Cover W4 weight bytes, W4A4 versus W4A8 activation bytes, dense TP sharding,
MoE routed/shared expert placement, MHA KV TP/DP placement, MLA KV replication
across TP, hybrid states, reserve, workspace, and capacity margin:

```python
class MemoryCostTests(unittest.TestCase):
    def test_w4_weights_use_half_byte_per_element(self):
        result = memory_breakdown(make_dense_model(), make_hardware(),
                                  make_w4a8_precision(), make_dense_plan(),
                                  stage="prefill", batch_size=1,
                                  input_length=1, output_length=1)
        self.assertEqual(result.total_weight_bytes,
                         model_counts(make_dense_model()).total_weight_elements / 2)

    def test_mla_kv_is_replicated_across_tp_but_split_by_attention_dp(self):
        plan = ParallelPlan(1, 2, 2, 1, 4, 8)
        result = memory_breakdown(make_mla_moe_model(), make_hardware(),
                                  make_w4a8_precision(), plan,
                                  stage="decode", batch_size=8,
                                  input_length=32, output_length=16)
        full = kv_bytes_per_request(make_mla_moe_model(),
                                    make_w4a8_precision(), 48)
        self.assertEqual(result.kv_bytes_per_card, full * 8 / plan.attention_dp)
```

- [ ] **Step 2: Run tests and verify missing memory module**

Run: `python -m unittest tests.test_memory_costs -v`

Expected: missing `infersim.cost.memory`.

- [ ] **Step 3: Implement memory placement and feasibility**

Create `MemoryBreakdown` with embedding, attention, dense/shared/routed FFN,
KV, recurrent state, activation, workspace, reserved, total, usable, margin,
and feasible fields.

Placement rules:

- Embeddings are replicated per model replica.
- Attention weights are divided by `attention_tp` and replicated across
  `attention_dp`.
- Dense FFN weights are divided by `attention_tp`.
- Routed expert weights are divided by `moe_tp * expert_parallel`.
- Shared expert weights are divided by `moe_tp` and replicated across EP.
- MHA/GQA KV for a local batch is divided by both `attention_tp` and
  `attention_dp`.
- MLA KV ranks are not TP-sharded and are divided only by `attention_dp`.
- Hybrid recurrent state is divided by `attention_dp`; it is replicated across
  TP unless its corresponding state dimension is explicitly sharded.
- Decode cache uses `input_length + output_length`; prefill resident KV uses
  `input_length`.
- `usable_bytes = capacity * (1 - reserve_fraction) - workspace`.

- [ ] **Step 4: Run memory tests**

Run: `python -m unittest tests.test_memory_costs -v`

Expected: all precision, placement, and margin tests pass.

- [ ] **Step 5: Commit memory accounting**

```powershell
git add infersim/cost/memory.py tests/test_memory_costs.py
git commit -m "feat: account for sharded weights and KV memory"
```

## Task 7: TP and EP Collective Costs

**Files:**
- Create: `infersim/cost/collective.py`
- Create: `tests/test_collectives.py`

- [ ] **Step 1: Write failing collective tests**

```python
class CollectiveCostTests(unittest.TestCase):
    def test_tp_allreduce_uses_ring_formula(self):
        hw = make_hardware(cards_per_node=8, intra_gbps=100,
                           intra_latency_us=2)
        result = all_reduce_cost(payload_bytes=1000, group_size=4,
                                 hardware=hw)
        expected_transfer = 2 * (4 - 1) / 4 * 1000
        expected = expected_transfer / 100e9 + 2 * (4 - 1) * 2e-6
        self.assertAlmostEqual(result.seconds, expected)

    def test_cross_node_group_uses_inter_node_path(self):
        hw = make_hardware(cards_per_node=4, intra_gbps=1000,
                           inter_gbps=10)
        result = all_to_all_cost(1000, 8, hw)
        self.assertEqual(result.path, "inter_node")
```

- [ ] **Step 2: Run tests and verify missing collective functions**

Run: `python -m unittest tests.test_collectives -v`

Expected: missing `infersim.cost.collective`.

- [ ] **Step 3: Implement analytical collectives**

`CollectiveCost` reports kind, payload, group, path, transfer bytes, bandwidth
seconds, latency seconds, and total seconds. Use:

```text
ring all-reduce bytes = 2 * (N - 1) / N * payload
ring all-reduce latency steps = 2 * (N - 1)
all-to-all bytes = (N - 1) / N * payload
all-to-all latency steps = N - 1
```

Return zero for group size one. Select intra-node only when the complete group
fits within `cards_per_node`; otherwise select inter-node. Add configured
collective launch latency once per collective. Payload precision uses
`activation_bits`.

- [ ] **Step 4: Run collective tests**

Run: `python -m unittest tests.test_collectives -v`

Expected: all topology and formula tests pass.

- [ ] **Step 5: Commit communication costs**

```powershell
git add infersim/cost/collective.py tests/test_collectives.py
git commit -m "feat: model TP and EP collectives"
```

## Task 8: Independent Prefill Evaluator

**Files:**
- Modify: `infersim/cost/types.py`
- Create: `infersim/cost/stage.py`
- Modify: `tests/helpers.py`
- Create: `tests/test_prefill_evaluator.py`

- [ ] **Step 1: Write failing prefill tests**

```python
class PrefillEvaluatorTests(unittest.TestCase):
    def test_prefill_reports_component_sum_and_capacity(self):
        result = evaluate_prefill(
            make_dense_model(), make_hardware(), make_w4a8_precision(),
            make_dense_plan(batch_size=4), make_scenario(input_length=128),
        )
        self.assertAlmostEqual(
            result.latency_seconds,
            result.gemm_seconds + result.vector_seconds
            + result.tp_seconds + result.ep_seconds,
        )
        self.assertAlmostEqual(result.request_capacity,
                               result.plan.replicas * 4 / result.latency_seconds)

    def test_prefill_does_not_include_decode_or_pd_time(self):
        result = evaluate_prefill(
            make_dense_model(), make_hardware(), make_w4a8_precision(),
            make_dense_plan(batch_size=4), make_scenario(input_length=128),
        )
        self.assertEqual(result.stage, "prefill")
        self.assertNotIn("pd", result.component_seconds)
```

Add tests for memory rejection, MoE EP time, all-scenario metrics, and larger
GEMM tiles increasing small-shape latency.

- [ ] **Step 2: Run tests and verify evaluator is absent**

Run: `python -m unittest tests.test_prefill_evaluator -v`

Expected: missing `evaluate_prefill`.

- [ ] **Step 3: Implement prefill evaluation**

Define one immutable result per plan and scenario:

```python
@dataclass(frozen=True)
class StageMetrics:
    stage: str
    scenario_name: str
    plan: ParallelPlan
    latency_seconds: float
    tpot_seconds: float | None
    prompt_token_capacity: float | None
    output_token_capacity: float | None
    request_capacity: float
    average_context_length: float
    gemm_seconds: float
    vector_seconds: float
    tp_seconds: float
    ep_seconds: float
    useful_gemm_ops: int
    aligned_gemm_ops: int
    useful_vector_ops: int
    aligned_vector_ops: int
    memory: MemoryBreakdown
    component_seconds: Mapping[str, float]
```

For prefill, `tpot_seconds` and `output_token_capacity` are `None` and
`latency_seconds` is the prefill TTFT contribution. `evaluate_prefill()` must:

1. Build shapes with `m = batch_size * input_length`.
2. Evaluate every GEMM/vector kernel and multiply repeated layers exactly once.
3. Add two TP all-reduces per full attention/dense FFN layer when TP > 1.
4. Add dispatch and combine All-to-All per MoE layer when EP > 1.
5. Evaluate memory independently and attach the breakdown.
6. Set prompt-token throughput to
   `replicas * batch_size * input_length / latency_seconds`.
7. Set request capacity to `replicas * batch_size / latency_seconds`.
8. Return raw component totals and useful/aligned operation counts.

Do not add scheduler, queue, decode, or PD transfer constants.

- [ ] **Step 4: Run prefill tests and current schema/cost tests**

Run:

```powershell
python -m unittest tests.test_prefill_evaluator -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit prefill evaluator**

```powershell
git add infersim/cost tests/test_prefill_evaluator.py
git commit -m "feat: evaluate prefill plans independently"
```

## Task 9: Independent Decode Evaluator

**Files:**
- Modify: `infersim/cost/stage.py`
- Modify: `tests/helpers.py`
- Create: `tests/test_decode_evaluator.py`

- [ ] **Step 1: Write failing decode tests**

```python
class DecodeEvaluatorTests(unittest.TestCase):
    def test_decode_capacity_uses_tpot_and_output_length(self):
        scenario = make_scenario(input_length=128, output_length=32)
        result = evaluate_decode(
            make_dense_model(), make_hardware(), make_w4a8_precision(),
            make_dense_plan(batch_size=8), scenario,
        )
        expected_tokens = result.plan.replicas * 8 / result.tpot_seconds
        self.assertAlmostEqual(result.output_token_capacity, expected_tokens)
        self.assertAlmostEqual(result.request_capacity,
                               expected_tokens / scenario.output_length)

    def test_decode_uses_average_growing_context(self):
        result = evaluate_decode(
            make_dense_model(), make_hardware(), make_w4a8_precision(),
            make_dense_plan(batch_size=8),
            make_scenario(input_length=100, output_length=20),
        )
        self.assertEqual(result.average_context_length, 110)
```

Add tests for KV capacity, W4A4/W4A8 activation traffic difference, MLA KV
placement, and EP communication.

- [ ] **Step 2: Run tests and verify decode evaluator is absent**

Run: `python -m unittest tests.test_decode_evaluator -v`

Expected: import or attribute failure for `evaluate_decode`.

- [ ] **Step 3: Implement decode evaluation**

`evaluate_decode()` must:

1. Use `m = batch_size` for projection/FFN GEMMs.
2. Use `input_length + output_length / 2` as average context.
3. Include KV read bytes in each attention kernel roofline cost.
4. Include recurrent-state reads for hybrid layers.
5. Apply the same TP/EP collective locations as prefill.
6. Set `tpot_seconds` to one complete decoder iteration.
7. Set output-token capacity to
   `replicas * batch_size / tpot_seconds`.
8. Set request capacity to output-token capacity divided by OSL.
9. Attach decode KV capacity and maximum supported batch/concurrency.

No prefill or PD time is included.

- [ ] **Step 4: Run decode tests and full suite**

Run:

```powershell
python -m unittest tests.test_decode_evaluator -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit decode evaluator**

```powershell
git add infersim/cost/stage.py tests/test_decode_evaluator.py
git commit -m "feat: evaluate decode plans independently"
```

## Task 10: Constraints, Pareto Frontier, and Recommendation Policy

**Files:**
- Create: `infersim/search/constraints.py`
- Create: `infersim/search/pareto.py`
- Modify: `tests/helpers.py`
- Create: `tests/test_constraints_and_pareto.py`

- [ ] **Step 1: Write failing constraint and ranking tests**

```python
class ConstraintAndParetoTests(unittest.TestCase):
    def test_all_policy_names_failing_scenario(self):
        outcome = evaluate_stage_constraints(
            make_stage_candidate(metrics=(
                make_metrics(name="short", feasible=True),
                make_metrics(name="long", ttft_ms=600, ttft_limit_ms=500),
            )),
            policy="all",
        )
        self.assertFalse(outcome.feasible)
        self.assertIn("long:TTFT_SLO", outcome.reason_codes)

    def test_recommendation_minimizes_cards_before_throughput(self):
        low_cards = make_candidate(total_cards=4, throughput=10)
        high_cards = make_candidate(total_cards=8, throughput=100)
        self.assertIs(recommend([high_cards, low_cards]), low_cards)

    def test_pareto_removes_dominated_candidate(self):
        best = make_candidate(total_cards=4, throughput=100, ttft_ms=10)
        dominated = make_candidate(total_cards=8, throughput=90, ttft_ms=20)
        self.assertEqual(pareto_frontier([dominated, best]), [best])
```

- [ ] **Step 2: Run tests and verify search functions are absent**

Run: `python -m unittest tests.test_constraints_and_pareto -v`

Expected: missing constraint and Pareto imports.

- [ ] **Step 3: Implement stable constraints and ordering**

Aggregate per-scenario evaluations in:

```python
@dataclass(frozen=True)
class StageCandidate:
    candidate_id: str
    plan: ParallelPlan
    metrics: tuple[StageMetrics, ...]
    feasible: bool
    reason_codes: tuple[str, ...]
    warnings: tuple[str, ...]
    total_cards: int
    hourly_cost: float | None
    request_capacity: float
    request_capacity_per_card: float
    ttft_ms: float | None
    tpot_ms: float | None
```

Stable reason codes include plan codes plus `MEMORY_CAPACITY`, `TTFT_SLO`,
`TPOT_SLO`, `REQUEST_RATE`, and `CONCURRENCY`. Under `all`, every scenario must
pass. Under `weighted`, hard plan/memory constraints must pass for every
scenario; latency and rate metrics are combined by normalized scenario weight
for ranking and retain per-scenario violations as warnings.

Pareto objectives are:

```text
minimize total_cards
minimize hourly_cost when present
maximize request_capacity
minimize ttft_ms
minimize tpot_ms
```

Recommendation sort key is:

```python
(
    total_cards,
    hourly_cost if hourly_cost is not None else math.inf,
    -request_capacity_per_card,
    plan.replicas,
    plan.attention_tp,
    plan.attention_dp,
    plan.moe_tp,
    plan.expert_parallel,
    plan.batch_size,
)
```

If every candidate has unknown cost, omit the cost position rather than
letting `math.inf` hide the throughput tie-breaker.

- [ ] **Step 4: Run constraint and full tests**

Run:

```powershell
python -m unittest tests.test_constraints_and_pareto -v
python -m unittest discover -s tests -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit constraints and Pareto logic**

```powershell
git add infersim/search tests/test_constraints_and_pareto.py
git commit -m "feat: filter and rank parallel candidates"
```

## Task 11: Stage Search Orchestration and Stable Reports

**Files:**
- Create: `infersim/search/runner.py`
- Create: `infersim/report.py`
- Modify: `tests/helpers.py`
- Create: `tests/test_reporting.py`

- [ ] **Step 1: Write failing runner and reporting tests**

Use a temporary output directory and assert required filenames, deterministic
headers, sorted candidate IDs, JSON key order, fixed float rounding, infeasible
diagnostics, and non-mutation of input objects:

```python
class ReportingTests(unittest.TestCase):
    def test_stage_search_writes_required_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_stage_search(
                stage="prefill",
                model=make_dense_model(),
                hardware=make_hardware(),
                precision=make_w4a8_precision(),
                scenario_set=make_scenario_set(),
                search_space=make_search_space(),
            )
            write_stage_reports(Path(tmp), result)
            self.assertEqual(
                {p.name for p in Path(tmp).iterdir()},
                {"all_candidates.csv", "feasible_candidates.csv",
                 "pareto_frontier.csv", "recommendation.json", "summary.txt"},
            )

    def test_no_feasible_plan_still_writes_diagnostics(self):
        result = run_stage_search(
            stage="decode",
            model=make_dense_model(),
            hardware=make_hardware(memory_capacity_gb=0.001),
            precision=make_w4a8_precision(),
            scenario_set=make_scenario_set(),
            search_space=make_search_space(),
        )
        self.assertIsNone(result.recommendation)
        self.assertEqual(result.dominant_rejection, "MEMORY_CAPACITY")
```

- [ ] **Step 2: Run tests and verify runner/reporter are absent**

Run: `python -m unittest tests.test_reporting -v`

Expected: missing runner/report imports.

- [ ] **Step 3: Implement stage runner and dependency-free reports**

`run_stage_search()` consumes normalized inputs, enumerates every validation,
evaluates valid plans across scenarios, records invalid plans without calling
the evaluator, applies constraints, computes frontier/recommendation, and
returns:

```python
@dataclass(frozen=True)
class SearchResult:
    stage: str
    candidates: tuple[StageCandidate, ...]
    feasible_candidates: tuple[StageCandidate, ...]
    pareto_frontier: tuple[StageCandidate, ...]
    recommendation: StageCandidate | None
    dominant_rejection: str | None
```

`report.py` uses `csv.DictWriter` and `json.dump(sort_keys=True, indent=2)`.
CSV rows are one candidate per row with aggregate/worst metrics and
semicolon-separated reason codes; `recommendation.json` contains complete
per-scenario metrics. Round serialized seconds to 12 decimals, milliseconds to
6, rates to 6, bytes to integers, and costs to 6 decimals. `summary.txt` names
the selected stage, plan, card count, SLO status, component bottleneck, and top
three rejection reasons.

- [ ] **Step 4: Run reporting and full tests**

Run:

```powershell
python -m unittest tests.test_reporting -v
python -m unittest discover -s tests -v
```

Expected: all tests pass and temporary outputs are deterministic.

- [ ] **Step 5: Commit runner and reports**

```powershell
git add infersim/search/runner.py infersim/report.py tests/test_reporting.py
git commit -m "feat: run stage searches and write reports"
```

## Task 12: PD KV/State Transfer and Candidate Pairing

**Files:**
- Create: `infersim/cost/pd.py`
- Create: `infersim/search/pair.py`
- Modify: `tests/helpers.py`
- Create: `tests/test_pd_pairing.py`

- [ ] **Step 1: Write failing PD tests**

```python
class PDPairingTests(unittest.TestCase):
    def test_pd_metrics_include_transfer_and_first_decode_step(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(latency_ms=20, request_capacity=100),
            make_decode_candidate(tpot_ms=5, request_capacity=80),
            make_pd_link(bandwidth_gbps=10, latency_us=100),
            kv_state_bytes=1_000_000,
            scenario=make_scenario(request_rate=50),
        )
        transfer_ms = (100e-6 + 1_000_000 / 10e9) * 1000
        self.assertAlmostEqual(metrics.ttft_ms, 20 + transfer_ms + 5)
        self.assertEqual(metrics.bottleneck, "decode")

    def test_link_can_be_system_bottleneck(self):
        metrics = evaluate_pd_pair(
            make_prefill_candidate(latency_ms=5, request_capacity=1000),
            make_decode_candidate(tpot_ms=5, request_capacity=1000),
            make_pd_link(bandwidth_gbps=0.01, latency_us=10),
            kv_state_bytes=1_000_000,
            scenario=make_scenario(request_rate=5),
        )
        self.assertEqual(metrics.bottleneck, "pd_link")

    def test_pair_search_minimizes_total_cards(self):
        prefill = make_search_result(candidates=(
            make_prefill_candidate(total_cards=2, request_capacity=100),
            make_prefill_candidate(total_cards=4, request_capacity=200),
        ))
        decode = make_search_result(candidates=(
            make_decode_candidate(total_cards=4, request_capacity=100),
            make_decode_candidate(total_cards=8, request_capacity=200),
        ))
        result = pair_stage_results(
            prefill, decode, make_pd_link(), make_scenario_set(),
            kv_state_bytes_by_scenario={"default": 1_000_000},
        )
        self.assertEqual(result.recommendation.total_cards, 6)
```

Add a hybrid test proving recurrent-state bytes join the PD transfer payload
and a concurrency test proving link concurrency changes overlap feasibility
without changing payload bytes.

- [ ] **Step 2: Run tests and verify PD modules are absent**

Run: `python -m unittest tests.test_pd_pairing -v`

Expected: missing `evaluate_pd_pair` and `pair_stage_results`.

- [ ] **Step 3: Implement transfer metrics and pruned pairing**

Use:

```text
effective_bandwidth = bandwidth_gbps * 1e9 * efficiency
transfer_seconds = latency_us * 1e-6 + payload_bytes / effective_bandwidth
link_request_capacity = effective_bandwidth / payload_bytes
system_request_capacity = min(prefill, decode, link)
pd_ttft = prefill_latency + transfer_seconds + decode_tpot
```

Payload is full-model prompt KV plus the terminal recurrent state required by
hybrid linear-attention layers. It is independent of source/destination
sharding because all shards collectively cross the PD boundary.

`max_concurrent_transfers` must satisfy
`ceil(request_rate * transfer_seconds) <= max_concurrent_transfers`; failure
uses `PD_TRANSFER_CONCURRENCY`. Link rate failure uses `PD_LINK_RATE`.

Before pairing, retain stage candidates on the local Pareto frontier plus the
local recommendation. Pair in deterministic candidate-ID order. Pair cost is
the sum of prefill and decode card-hour costs; pair card count is the sum of
phase card counts. Reuse the Task 10 ranking policy with combined PD metrics.

- [ ] **Step 4: Run PD and full tests**

Run:

```powershell
python -m unittest tests.test_pd_pairing -v
python -m unittest discover -s tests -v
```

Expected: all PD formula, pruning, and recommendation tests pass.

- [ ] **Step 5: Commit PD pairing**

```powershell
git add infersim/cost/pd.py infersim/search/pair.py tests/test_pd_pairing.py
git commit -m "feat: pair prefill and decode deployments"
```

## Task 13: CLI, Examples, Documentation, and Legacy Regression

**Files:**
- Create: `infersim/cli.py`
- Create: `infersim/__main__.py`
- Create: `tests/test_cli.py`
- Create: `tests/test_legacy_cli.py`
- Create: `examples/search/custom_npu.json`
- Create: `examples/search/w4a8.json`
- Create: `examples/search/scenarios.json`
- Create: `examples/search/search_space.json`
- Create: `examples/search/pd_link.json`
- Modify: `README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing CLI and compatibility tests**

Use `subprocess.run()` with the complete command arrays shown in Step 5, replacing
only output paths with temporary directories. Assert:

- `search --stage prefill` exits zero and writes five files.
- `search --stage decode` exits zero and writes five files.
- `pair-pd` writes `prefill/`, `decode/`, and `pd/` report directories.
- Invalid JSON names the exact field path and exits 2.
- No feasible plan writes reports and exits 1.
- `python main.py --help` remains zero and includes `--config-path`.
- One bundled `qwen3-8B` legacy command exits zero and prints `TPOT (ms)`.

- [ ] **Step 2: Run CLI tests and confirm module entry point is absent**

Run: `python -m unittest tests.test_cli tests.test_legacy_cli -v`

Expected: `No module named infersim.__main__` for new commands; legacy help
already passes.

- [ ] **Step 3: Implement CLI and checked-in examples**

The parser has:

```text
infersim search
  --model PATH
  --hardware PATH
  --precision PATH
  --scenarios PATH
  --search-space PATH (optional)
  --stage {prefill,decode}
  --output DIRECTORY

infersim pair-pd
  --model PATH
  --prefill-hardware PATH
  --decode-hardware PATH
  --pd-link PATH
  --precision PATH
  --scenarios PATH
  --prefill-search-space PATH (optional)
  --decode-search-space PATH (optional)
  --output DIRECTORY
```

Map schema/unsupported-model errors to exit 2, no feasible result to exit 1,
and success to exit 0. Always write diagnostic reports before returning 1.
Do not catch unexpected programming errors.

Populate example JSON with a small search grid that completes in under two
seconds. Add generated `results/` paths to `.gitignore`.

- [ ] **Step 4: Document commands, formulas, and limits**

Update `README.md` with:

- Original upstream attribution retained unchanged.
- A new PD-aware search overview.
- Hardware, precision, scenario, search-space, and PD-link examples.
- W4A4/W4A8 and GEMM/VECTOR distinction.
- Parallel width equations and the two DP meanings.
- Prefill/decode independence and PD pairing formula.
- Exact example commands.
- Output file descriptions.
- Explicit analytical-model and unsupported-architecture limits.

- [ ] **Step 5: Run complete verification**

Run:

```powershell
python -m unittest discover -s tests -v
python -m infersim search --model hf_configs/qwen3-8B_config.json --hardware examples/search/custom_npu.json --precision examples/search/w4a8.json --scenarios examples/search/scenarios.json --search-space examples/search/search_space.json --stage prefill --output tmp/verify-prefill
python -m infersim search --model hf_configs/qwen3-8B_config.json --hardware examples/search/custom_npu.json --precision examples/search/w4a8.json --scenarios examples/search/scenarios.json --search-space examples/search/search_space.json --stage decode --output tmp/verify-decode
python -m infersim pair-pd --model hf_configs/qwen3-8B_config.json --prefill-hardware examples/search/custom_npu.json --decode-hardware examples/search/custom_npu.json --pd-link examples/search/pd_link.json --precision examples/search/w4a8.json --scenarios examples/search/scenarios.json --prefill-search-space examples/search/search_space.json --decode-search-space examples/search/search_space.json --output tmp/verify-pd
python main.py --config-path hf_configs/qwen3-8B_config.json --device-type H20 --world-size 1 --tp-size 1 --decode-only
git diff --check
```

Expected: tests pass; all three new commands exit zero and name a
recommendation; the legacy command exits zero and prints `TPOT (ms)`; Git
reports no whitespace errors.

- [ ] **Step 6: Commit the complete user-facing feature**

```powershell
git add infersim examples/search tests/test_cli.py tests/test_legacy_cli.py README.md .gitignore
git commit -m "feat: expose PD-aware deployment search CLI"
```

- [ ] **Step 7: Review branch scope and upstream attribution**

Run:

```powershell
git status --short
git log --oneline upstream/main..HEAD
git diff --stat upstream/main...HEAD
```

Expected: clean worktree; one design commit, one plan commit, and focused
feature commits; Apache-2.0 `LICENSE` and original README acknowledgement remain
present.
