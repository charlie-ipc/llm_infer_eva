# PD-Aware Parallel Configuration Search Design

Date: 2026-08-15

Status: Approved for implementation planning

Upstream: `alibaba/InferSim`

## 1. Objective

Extend InferSim from a single-point analytical estimator into a lightweight
design-space exploration tool for prefill/decode-disaggregated LLM serving.
Given a Hugging Face model configuration, one or two custom accelerator
descriptions, precision settings, workload scenarios, and SLOs, the tool will:

1. Evaluate prefill and decode independently.
2. Enumerate accelerator counts and legal TP/DP/EP/replica configurations.
3. Reject configurations that violate model divisibility, memory, throughput,
   or latency constraints.
4. Pair feasible prefill and decode configurations while accounting for KV
   transfer latency and bandwidth.
5. Recommend the minimum-card feasible deployment and emit Pareto frontiers.

The implementation must preserve InferSim's lightweight, analytical, pure
Python character. It will borrow llm-optimizer's ideas for parameter grids,
constraint filtering, stable result records, and Pareto reporting, but it will
not import llm-optimizer's runtime orchestration.

## 2. Scope

### 2.1 Supported models

The first release supports decoder-only Hugging Face models whose
`config.json` can be normalized into the following layer types:

- MHA, MQA, and GQA attention.
- MLA attention.
- Dense FFN.
- Routed MoE with Top-K routing.
- MoE with shared experts.
- Models that mix full attention and linear attention layers.

The parser must recognize common Hugging Face field aliases and normalize them
into a stable internal schema. Missing structural fields and unsupported layer
types must produce explicit errors. The tool must not silently substitute a
standard Transformer formula for an unknown architecture.

### 2.2 Explicit non-goals

The first release does not cover:

- Multimodal encoders or projectors.
- Encoder-decoder models.
- Arbitrary custom operators.
- Runtime management for vLLM, SGLang, or TensorRT-LLM.
- Request-arrival or continuous-batching discrete-event simulation.
- A web UI.
- Automatic checkpoint or model-weight downloads.

## 3. Architecture

The system is split into narrow modules with explicit inputs and outputs:

```text
HF config.json ----> ModelSpec / LayerSpec
                            |
Hardware JSON -----> HardwareSpec
                            |
Precision JSON ----> PrecisionSpec
                            |
Scenarios JSON ----> WorkloadScenario[]
                            |
                            v
                  CandidateEnumerator
       cards, replicas, attention TP/DP, MoE TP/EP, batch
                            |
              +-------------+-------------+
              |                           |
              v                           v
       PrefillEvaluator             DecodeEvaluator
       TTFT/input rate              TPOT/KV/output rate
              +-------------+-------------+
                            |
                            v
                    PDPairingEvaluator
          KV transfer, link capacity, phase rate matching
                            |
                            v
                 Constraints + Pareto ranking
                            |
                    CSV / JSON / text
```

Responsibilities:

- `ModelSpec` represents model structure only. It contains no hardware timing
  assumptions.
- `HardwareSpec` describes compute engines, memory, topology, launch costs, and
  card cost.
- `PrecisionSpec` defines storage and arithmetic modes independently.
- `ParallelPlan` represents placement without stage-specific performance data.
- `PrefillEvaluator` and `DecodeEvaluator` are independent analytical models.
- `PDPairingEvaluator` combines immutable phase results and adds KV transfer.
- The search layer owns enumeration, pruning, constraints, ranking, and output.

Existing single-point command-line behavior remains available. New behavior is
provided through subcommands so current examples do not break.

## 4. Core Data Models

### 4.1 Normalized model specification

`ModelSpec` contains at least:

```text
model_type
hidden_size
num_hidden_layers
vocab_size
attention_kind
num_attention_heads
num_key_value_heads
head_dim
intermediate_size
num_routed_experts
num_shared_experts
experts_per_token
moe_intermediate_size
full_attention_layer_count
linear_attention_layer_count
MLA dimensions, when applicable
```

`LayerSpec` or an equivalent compact block description identifies which
operation families occur in each layer. Repeated layers may share a template;
the design does not require materializing a large operator graph.

### 4.2 Hardware specification

Hardware uses decimal GB, GB/s, TFLOPS, and microseconds at the user boundary.
Internal calculations use bytes and seconds.

```json
{
  "name": "custom_npu",
  "memory_capacity_gb": 96,
  "memory_bandwidth_gbps": 3200,
  "cards_per_node": 8,
  "compute_tflops": {
    "gemm": {
      "bf16": 400,
      "fp8": 800,
      "w8a8": 800,
      "w4a8": 1200,
      "w4a4": 1600
    },
    "vector": {
      "bf16": 20,
      "fp8": 40,
      "int8": 40,
      "fp4": 80
    }
  },
  "gemm_tile": {"m": 16, "n": 16, "k": 32},
  "gemm_engines": 64,
  "vector_width": 1024,
  "vector_units": 16,
  "kernel_launch_latency_us": {
    "gemm": 1.0,
    "vector": 0.5,
    "collective": 2.0
  },
  "interconnect": {
    "intra_node_gbps": 900,
    "intra_node_latency_us": 2,
    "inter_node_gbps": 100,
    "inter_node_latency_us": 8
  },
  "memory_reserve_fraction": 0.1,
  "cost_per_card_hour": 0
}
```

There are no stage-wide `prefill_compute` or `decode_compute` efficiency
settings. Prefill/decode differences are derived from operation shapes,
tile/vector alignment, memory traffic, and communication.

### 4.3 Precision specification

Storage, GEMM, vector, accumulation, and KV precision are independent:

```json
{
  "gemm_mode": "w4a8",
  "weight_bits": 4,
  "activation_bits": 8,
  "vector_bits": 8,
  "accumulator_bits": 16,
  "kv_cache_bits": 8
}
```

At minimum, GEMM hardware may expose `bf16`, `fp8`, `w8a8`, `w4a8`, and
`w4a4`. W4A4 and W4A8 affect:

- The selected GEMM throughput.
- Weight capacity and DRAM traffic.
- Activation capacity and DRAM traffic.
- Collective activation payloads.
- Vector precision selection.

KV storage remains controlled by `kv_cache_bits`.

### 4.4 Parallel plan

Inference DP has two distinct meanings and must not be overloaded:

```text
replicas: complete serving replicas
attention_tp: tensor parallel degree for attention/dense paths
attention_dp: attention data-parallel degree within an MoE worker
moe_tp: tensor parallel degree inside each expert computation
expert_parallel: expert sharding degree
```

For a MoE worker:

```text
attention_tp * attention_dp
    == moe_tp * expert_parallel
    == cards_per_replica

stage_total_cards = replicas * cards_per_replica
```

For a dense model:

```text
attention_dp = 1
moe_tp = attention_tp
expert_parallel = 1
cards_per_replica = attention_tp
```

Candidate parallel degrees default to powers of two but users may provide an
explicit candidate set. Plans must satisfy applicable head, KV-head, expert,
and tensor-dimension divisibility constraints. An invalid plan becomes a
rejected result with reason codes; it does not abort the search.

## 5. Analytical Performance Model

### 5.1 Operation families

GEMM work includes:

- Q, K, V, and output projections.
- QK and probability-value products.
- Dense FFN matrices.
- Routed and shared expert matrices.
- Grouped GEMM after expert routing.

Vector work includes:

- RMSNorm.
- RoPE.
- Softmax.
- Activation and gating functions.
- Residual operations.
- MoE routing, scatter, and gather bookkeeping.

Memory work includes model weights, activations, KV cache, expert weights, and
intermediate reads/writes. Communication work includes TP collectives, EP
All-to-All, and PD KV transfer.

### 5.2 Shape-derived utilization

GEMM and vector work is aligned to explicit hardware geometry and issued in
whole engine waves rather than scaled by an arbitrary stage efficiency:

```text
gemm_tiles =
    ceil(M / tile_m)
  * ceil(N / tile_n)
  * ceil(K / tile_k)

aligned_gemm_work =
    ceil(gemm_tiles / gemm_engines)
  * gemm_engines
  * work_per_tile

aligned_vector_work =
    ceil(
      ceil(elements / vector_width) / vector_units
    )
  * vector_units
  * vector_width
```

This causes small decode matrices to pay for unused tile lanes while large
prefill matrices approach peak throughput naturally. The configured GEMM and
vector throughput values are aggregate device throughputs; engine and unit
counts only determine wave/alignment waste and must not divide the throughput
a second time.

For each kernel:

```text
compute_time = aligned_work / selected_compute_throughput
memory_time = bytes_transferred / memory_bandwidth
kernel_time = max(compute_time, memory_time) + launch_latency
```

The evaluator sums sequential GEMM, vector, and communication kernels. It must
report compute and memory components even when one is hidden by the roofline
maximum. This avoids double counting within a kernel while preserving a clear
bottleneck breakdown.

### 5.3 Memory feasibility

Per-card memory includes:

- Sharded or replicated weights according to `ParallelPlan`.
- KV cache according to stage, batch/concurrency, context, and KV precision.
- Activation/intermediate storage.
- Runtime workspace reservation.
- The configured safety reserve.

The result reports each component and the capacity margin. Decode KV capacity
is evaluated independently from prefill transient memory.

### 5.4 Communication

The collective model uses message size, group size, latency, and effective
bandwidth. A group contained within `cards_per_node` uses intra-node values; a
larger group uses the inter-node path. The first release may use analytical
ring formulas, but the interface must permit a future measured collective
profile without changing evaluators.

## 6. Prefill and Decode Evaluation

Prefill and decode use independent hardware pools and independent plans.

Prefill reports:

- TTFT contribution before KV transfer.
- Prompt/input token throughput.
- Request capacity for every workload scenario.
- GEMM, vector, memory, TP, and EP time breakdowns.
- Transient and resident memory.

Decode reports:

- First decode-step latency.
- TPOT.
- Output token throughput.
- Request capacity after accounting for OSL.
- KV capacity and maximum supported concurrency.
- GEMM, vector, memory, TP, and EP time breakdowns.

Neither evaluator embeds the other phase or the PD link.

## 7. PD KV Transfer and Pairing

The PD link is configured independently:

```json
{
  "bandwidth_gbps": 100,
  "latency_us": 10,
  "efficiency": 0.8,
  "max_concurrent_transfers": 16
}
```

Link efficiency is permitted because it describes protocol/link payload
efficiency, not accelerator compute utilization.

For each scenario, the pairer calculates total prompt KV bytes from the model,
ISL, layer count, and KV precision. It then computes:

```text
kv_transfer_time = fixed_latency + kv_bytes / effective_link_bandwidth
pd_link_capacity = effective_link_bandwidth / kv_bytes_per_request

system_request_capacity = min(
    prefill_request_capacity,
    decode_request_capacity,
    pd_link_capacity
)

pd_ttft =
    prefill_latency
  + kv_transfer_latency
  + first_decode_step_latency
```

Transfer concurrency is used to determine whether latency can overlap across
different requests; it does not reduce the bytes transferred. The pairer must
report whether prefill, decode, or the PD link is the bottleneck.

The pairer first prunes phase candidates that fail local constraints, then
pairs the remaining Pareto-relevant candidates. This avoids an unbounded full
Cartesian product while preserving candidates that can be globally optimal.

## 8. Workload Scenarios and Constraints

The first release uses a scenario grid rather than an arrival-process
simulator. Each scenario contains at least:

```text
name
input_length
output_length
request_rate
concurrency
ttft_limit_ms
tpot_limit_ms
optional weight
```

A search may require one plan to satisfy every scenario or may rank aggregate
results using scenario weights. The selected policy is explicit in the input.
No P99 queueing claim may be made from this analytical grid.

Constraint evaluation order:

1. Input/schema and model support validation.
2. Parallel divisibility and topology feasibility.
3. Per-card memory feasibility.
4. Stage-local latency and throughput SLOs.
5. PD link capacity and combined TTFT.

Rejected candidates retain stable reason codes and human-readable details.

## 9. Ranking and Pareto Selection

The default recommendation policy is lexicographic:

1. Keep only configurations satisfying capacity, latency, and throughput
   constraints.
2. Minimize total accelerator count.
3. At equal card count, minimize hourly cost when cost is configured.
4. If cost is absent or tied, maximize per-card throughput.
5. Apply deterministic tie-breakers based on parallel-plan fields.

The tool also emits a Pareto frontier over at least:

- Total card count.
- Hourly cost, when available.
- System request throughput.
- TTFT.
- TPOT.

The recommendation is one policy choice on top of the frontier, not a claim
that the other Pareto configurations are universally inferior.

## 10. Command-Line Interface

Existing use remains valid:

```powershell
python main.py --config-path model_config.json ...
```

New stage search:

```powershell
python -m infersim search `
  --model model_config.json `
  --hardware hardware.json `
  --precision precision.json `
  --scenarios scenarios.json `
  --stage prefill `
  --output results/prefill

python -m infersim search ... --stage decode
```

New PD pairing search:

```powershell
python -m infersim pair-pd `
  --model model_config.json `
  --prefill-hardware prefill_hardware.json `
  --decode-hardware decode_hardware.json `
  --pd-link pd_link.json `
  --precision precision.json `
  --scenarios scenarios.json `
  --output results/pd
```

The package name and exact parser layout may be adjusted during implementation
only if needed to preserve backwards compatibility; the user-visible command
semantics above are required.

## 11. Outputs

Each search directory contains:

- `all_candidates.csv`: every evaluated plan, feasibility, and rejection
  reasons.
- `feasible_candidates.csv`: plans satisfying all selected constraints.
- `pareto_frontier.csv`: nondominated plans.
- `recommendation.json`: selected plan, metrics, bottleneck, assumptions, and
  normalized input summary.
- `summary.txt`: concise human-readable report.

Column and JSON keys are stable, documented, and deterministic. Floating-point
serialization uses a fixed precision suitable for reproducible diffs.

## 12. Error Handling

- Unsupported architectures identify the model field or operator that cannot
  be normalized.
- A missing hardware precision reports the exact key, such as
  `compute_tflops.gemm.w4a8` or `compute_tflops.vector.int8`.
- Invalid individual plans are recorded and do not stop the search.
- If no feasible plan exists, the tool still writes diagnostic outputs,
  summarizes the dominant rejection reasons, and exits nonzero.
- Schema errors include the JSON field path, received value, expected type or
  unit, and valid range.
- Negative capacities, rates, dimensions, and bandwidths are rejected before
  evaluation.

## 13. Compatibility and Migration

The current `main.py` path and bundled examples remain operational. Existing
hardware definitions may be adapted internally into `HardwareSpec`; users are
not required to migrate existing scripts for the first release.

The new implementation should progressively move calculations out of printing
methods into pure result-returning functions. Formatting becomes a consumer of
result objects. This refactor is limited to code needed by search and testing.

## 14. Testing Strategy

Development follows test-driven implementation.

### 14.1 Unit tests

- Hugging Face field normalization and unsupported-model errors.
- W4A4 and W4A8 parameter, activation, and communication byte counts.
- GEMM tile and vector-width alignment.
- GEMM, vector, memory, and launch-latency calculations.
- Dense and MoE parallel-plan constraints.
- Weight and KV sharding rules.
- Intra-node versus inter-node collective selection.
- PD KV byte, latency, and link-capacity calculations.
- Constraint evaluation and deterministic ranking.
- Pareto dominance and stable tie-breaking.

### 14.2 Enumeration invariants

Generated candidates must always satisfy, when applicable:

```text
cards_per_replica == attention_tp * attention_dp
cards_per_replica == moe_tp * expert_parallel
stage_total_cards == replicas * cards_per_replica
```

Explicit invalid candidates must have a reason code rather than an exception.

### 14.3 Integration tests

- Prefill search writes all required outputs.
- Decode search writes all required outputs.
- PD pairing selects a known minimum-card feasible pair.
- An infeasible search writes diagnostics and exits nonzero.
- Multiple scenarios obey all-scenarios and weighted policies.

### 14.4 Regression tests

Existing InferSim example commands continue to run. Key reported values remain
within documented tolerances unless a test identifies and records an upstream
formula correction.

## 15. Suggested Package Layout

```text
infersim/
  cli.py
  schema/
    model.py
    hardware.py
    precision.py
    scenario.py
    parallel.py
  model/
    normalize.py
    operations.py
  cost/
    gemm.py
    vector.py
    memory.py
    collective.py
    prefill.py
    decode.py
    pd_transfer.py
  search/
    enumerate.py
    constraints.py
    pareto.py
    rank.py
    pair_pd.py
  report/
    csv_report.py
    json_report.py
    text_report.py
tests/
```

The implementation may consolidate very small modules. It must not create
abstractions that do not remove real complexity.

## 16. Acceptance Criteria

The first release is complete when:

1. A user can define a custom accelerator without editing Python source.
2. W4A4 and W4A8 use distinct GEMM throughput and correct storage/traffic
   widths.
3. Prefill and decode can be searched independently over card count,
   replicas, attention TP/DP, and MoE TP/EP.
4. Dense and MoE candidates obey the documented parallel constraints.
5. PD pairing accounts for KV transfer latency and link throughput.
6. The default recommendation returns the minimum-card feasible pair using the
   documented tie-breakers.
7. CSV and JSON outputs expose component timing, capacity, bottleneck, and
   rejection reasons.
8. Unsupported model structures fail explicitly.
9. Existing InferSim single-point examples remain usable.
10. The complete automated test suite passes from a clean checkout.
