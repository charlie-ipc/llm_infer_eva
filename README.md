# InferSim: A Lightweight LLM Inference Performance Simulator

InferSim is a lightweight simulator for LLM inference, written in pure Python without any 3rd-party dependencies. It calculates the TTFT, TPOT and throughput TGS (tokens/GPU/second) based on computation complexity FLOPs (Floating-Point Operations), GPU computing power FLOPS (Floating-Point Operations per Second), GPU memory bandwidth and MFU (Model FLOPs Utilization) obtained by benchmarking the state-of-the-art LLM kernels. For multi-GPU, multi-node deployment, InferSim also estimates the communication latency according to data volume and bandwidth.

The main use cases of InferSim include:
- **Model-Sys co-design**: predicting inference performance given the hyperparameters of a model.
- **Inference performance analysis**: quantifying performance bottlenecks, such as compute-bound or IO-bound, and supporting optimization efforts.

For more details, please check [InferSim Technical Report](https://github.com/user-attachments/files/23016438/infersim_tech_report.pdf).

## Simulation Result

| Model | GPU | Prefill TGS(Actual) | Prefill TGS(Sim) | Decode TGS(Actual) | Decode TGS(Sim) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| DeepSeek-V3 | H800 | 7839 | 9034 | 2324 | 2675 | Actual data from [deepseek/profile-data](https://github.com/deepseek-ai/profile-data/). Simulated with same setup: [example/deepseek-v3/](./example/deepseek-v3/). |
| Qwen3-30B-A3B-BF16 | H20 | 16594 | 17350 | 2749 | 2632 | Actual data tested with SGLang, simulation example: [example/qwen3-30B-A3B/](./example/qwen3-30B-A3B/). |
| Qwen3-8B-FP8 | H20 | 15061 | 16328 | 2682 | 2581 | Actual data tested with SGLang, simulation example: [example/qwen3-8B/](./example/qwen3-8B/). |
| Qwen3.5-35B-A3B-FP8 | PRO5000 | 33139 | 36449 | 2150 | 2175 | Actual data tested with SGLang, simulation example: [example/qwen3-35B/](./example/qwen3.5-35B-A3B). |

The accuracy of simulation results relies heavily on the kernel benchmark results. Please help us improve the [kernel_benchmark](./kernel_benchmark) and append the data to [bench_data](./bench_data).

## Supported Features

- **Attention**: MHA/GQA, MLA. Benchmarked on FlashInfer, FlashAttention-3, FlashMLA.
- **MoE**: GroupedGEMM. Benchmarked on DeepGEMM.
- **Linear**: GEMM. Benchmarked on DeepGEMM.
- **Parallelization**: DP Attn, EP MoE.
- **Large EP**: DeepEP dispatch and combine, with normal and low_latency mode.

## Help

```
$ python3 main.py --help
usage: main.py [-h] --config-path CONFIG_PATH
               [--device-type {H20,H800,H200,GB200,PRO5000}]
               [--world-size WORLD_SIZE] [--tp-size TP_SIZE]
               [--num-nodes NUM_NODES]
               [--max-prefill-tokens MAX_PREFILL_TOKENS]
               [--prefill-attn-mfu PREFILL_ATTN_MFU]
               [--prefill-moe-mfu PREFILL_MOE_MFU] [--decode-bs DECODE_BS]
               [--target-tgs TARGET_TGS] [--target-tpot TARGET_TPOT]
               [--target-isl TARGET_ISL] [--target-osl TARGET_OSL]
               [--use-fp8-gemm] [--use-fp8-kv]
               [--decode-scheduler-overhead-ms DECODE_SCHEDULER_OVERHEAD_MS]
               [--enable-shared-expert-overlap] [--enable-deepep]
               [--enable-tbo] [--sm-ratio SM_RATIO] [--prefill-only]
               [--decode-only]

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        The path of the hf model config.json
  --device-type {H20,H800,H200,GB200,PRO5000}
                        Device type
  --world-size WORLD_SIZE
                        Num of GPUs
  --tp-size TP_SIZE     Tensor parallel size. If >1, both attention and MoE
                        use TP; if =1, attention uses DP and MoE uses EP.
  --num-nodes NUM_NODES
                        Num of nodes
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        Max prefill tokens per GPU
  --prefill-attn-mfu PREFILL_ATTN_MFU
                        Override prefill attention MFU. If omitted, read MFU
                        from bench data.
  --prefill-moe-mfu PREFILL_MOE_MFU
                        Override prefill routed MoE/FFN MFU. If omitted, read
                        MFU from bench data.
  --decode-bs DECODE_BS
                        Decoding batchsize per GPU. If not specified, bs = tgs * tpot.
  --target-tgs TARGET_TGS
                        Target tokens/s per GPU
  --target-tpot TARGET_TPOT
                        TPOT in ms
  --target-isl TARGET_ISL
                        Input sequence length, in tokens
  --target-osl TARGET_OSL
                        Output sequence length, in tokens
  --use-fp8-gemm        Use fp8 gemm
  --use-fp8-kv          Use fp8 kvcache
  --decode-scheduler-overhead-ms DECODE_SCHEDULER_OVERHEAD_MS
                        Override decode scheduler overhead in milliseconds. If
                        omitted, use the existing model default.
  --enable-shared-expert-overlap
                        Model Qwen3.5 MoE Decode shared and routed experts as
                        parallel CUDA-stream paths. Only supported for the
                        Qwen3.5 MoE HybridModel.
  --enable-deepep       Enable DeepEP
  --enable-tbo          Enable two batch overlap
  --sm-ratio SM_RATIO   In TBO DeepEP normal mode, the SM ratio used for computation
  --prefill-only        Only simulate prefill
  --decode-only         Only simulate decoding
```

## Example

```
$ bash example/qwen3-30B-A3B/decode.sh

================ Simulator Result ================
Device type:                             H20
World size:                              4
Attn type:                               MHA/GQA
Use FP8 GEMM:                            0
Use FP8 KV:                              0
------------------Model Weights-------------------
One attn params size (MB):               36.00
One expert params size (MB):             9.00
Per GPU params size (GB):                15.19
---------------------KV Cache---------------------
KV cache space (GB):                     60.81
Input seq len:                           4096
Output seq len:                          2048
Target decode batchsize:                 100
Target per-token KV cache size (KB):     103.79
Current per-token KV cache size (KB):    96.00
----------------------FLOPs-----------------------
Num hidden layers:                       48
Per-token per-layer attn core (GFLOPs):  0.08
Per-token per-layer MoE/FFN (GFLOPs):    0.08
Per-token per-layer others (GFLOPs):     0.04
Per-token attn core (GFLOPs):            4.03
Per-token MoE (GFLOPs):                  3.62
Per-token others (GFLOPs):               1.81
Per-token total (GFLOPs):                9.46
---------------------Decoding---------------------
Attn core MFU:                           0.15
Attn core latency (us):                  361.77
KV loading latency (us):                 298.02
QKV_proj latency (us):                   31.03
O_proj latency (us):                     16.95
Routed experts/FFN MFU:                  0.18
Routed experts/FFN latency (us):         269.28
Experts loading latency (us):            85.83
Comm before MoE/FFN (us):                4.24
Comm after MoE/FFN (us):                 4.24
TPOT (ms):                               38.00
Throughput (TGS):                        2632
```

## PD-Aware Deployment Search

The `infersim` module adds an analytical deployment search for decoder-only
LLMs. It reads a Hugging Face model config plus explicit hardware, precision,
workload, and parallel-search inputs. Prefill and decode are searched
independently, so they may use different accelerators and different TP/DP/EP
plans; `pair-pd` then rate-matches the two phases and models the KV/state
transfer between them.

The checked-in inputs are intentionally small and complete:

- [`custom_npu.json`](examples/search/custom_npu.json) defines DRAM capacity and
  bandwidth, GEMM and VECTOR throughput, tile/engine geometry, launch latency,
  intra/inter-node topology, and card cost. `compute_tflops.gemm` contains
  W4A4 and W4A8 tensor modes, while `compute_tflops.vector` independently
  contains FP4, INT8, BF16, and FP32 vector modes.
- [`w4a8.json`](examples/search/w4a8.json) keeps weight, activation, vector,
  accumulator, and KV-cache bit widths separate. To evaluate W4A4, set
  `gemm_mode` to `w4a4`, `activation_bits` and `vector_bits` to `4`; the
  hardware must provide the corresponding `w4a4` GEMM and `fp4` VECTOR modes.
- [`scenarios.json`](examples/search/scenarios.json) gives ISL, OSL, request
  rate, concurrency, TTFT/TPOT limits, weights, and the scenario policy.
- [`search_space.json`](examples/search/search_space.json) gives the candidate
  card counts, replicas, attention TP/DP, MoE TP/EP, and batch sizes.
- [`pd_link.json`](examples/search/pd_link.json) gives PD bandwidth, latency,
  efficiency, and the maximum concurrent transfers.

Run the complete examples from the repository root:

```powershell
python -m infersim search --model hf_configs/qwen3-8B_config.json --hardware examples/search/custom_npu.json --precision examples/search/w4a8.json --scenarios examples/search/scenarios.json --search-space examples/search/search_space.json --stage prefill --output results/prefill
python -m infersim search --model hf_configs/qwen3-8B_config.json --hardware examples/search/custom_npu.json --precision examples/search/w4a8.json --scenarios examples/search/scenarios.json --search-space examples/search/search_space.json --stage decode --output results/decode
python -m infersim pair-pd --model hf_configs/qwen3-8B_config.json --prefill-hardware examples/search/custom_npu.json --decode-hardware examples/search/custom_npu.json --pd-link examples/search/pd_link.json --precision examples/search/w4a8.json --scenarios examples/search/scenarios.json --prefill-search-space examples/search/search_space.json --decode-search-space examples/search/search_space.json --output results/pd-system
```

Omit a search-space option to use the bounded CLI default: total cards
`[1, 2, 4, 8]`, replicas `[1, 2]`, attention TP `[1, 2, 4]`, attention DP
`[1, 2]`, MoE TP `[1, 2, 4]`, expert parallel `[1, 2]`, and batch sizes
`[1, 4, 16]`. This is 864 raw combinations with at most eight cards. Use an
explicit search-space JSON for larger card counts or wider axes. Every valid
replica obeys:

```text
attention_tp * attention_dp == moe_tp * expert_parallel == cards_per_replica
total_cards == replicas * cards_per_replica
```

`attention_dp` shards attention work within one whole-model replica. `replicas`
duplicates the entire model and scales request throughput. It is therefore not
the same DP dimension. Dense models additionally require `attention_dp == 1`,
`expert_parallel == 1`, and `moe_tp == attention_tp`. MoE plans may shard
routed experts with EP while their attention width remains legal.

The cost model derives GEMM and VECTOR time from operation shapes, tiling,
engine/vector alignment, memory traffic, and communication; there is no manual
prefill/decode efficiency knob. W4A4 and W4A8 select separate GEMM peaks, and
vector kernels use the mode implied by `vector_bits`.

For each scenario, PD transfer uses:

The existing `bandwidth_gbps` schema name denotes decimal GB/s, consistent
with the hardware bandwidth fields.

```text
payload_bytes = prompt_KV_bytes + terminal_recurrent_state_bytes
effective_bandwidth_bytes_per_second = bandwidth_gbps * 1e9 * efficiency
transfer_seconds = latency_us / 1e6 + payload_bytes / effective_bandwidth
PD_link_request_capacity = effective_bandwidth / payload_bytes
first_decode_TTFT = prefill_latency + transfer_seconds + first_decode_step_latency
```

The pairer checks stage request capacity, link rate, transfer concurrency, TTFT,
and TPOT. The default recommendation first requires feasibility, then minimizes
total cards, minimizes known hourly cost, maximizes per-card request capacity,
and finally applies stable candidate IDs. Pareto reports retain nondominated
card/cost/capacity/latency alternatives.

Each stage directory contains:

- `all_candidates.csv`: every raw parallel-plan combination, including invalid
  plans, aggregate metrics, warnings, and rejection diagnostics.
- `feasible_candidates.csv`: only plans satisfying memory, rate, concurrency,
  and the selected scenario-policy constraints.
- `pareto_frontier.csv`: feasible nondominated card, cost, capacity, and stage
  latency alternatives.
- `recommendation.json`: the selected plan, complete per-scenario metrics,
  bottleneck, assumptions, and normalized model/hardware/search inputs.
- `summary.txt`: the selected plan and cards, SLO state, bottleneck, and leading
  rejection reasons in a concise text form.

The `pd/` directory contains:

- `all_pairs.csv`: every retained prefill/decode pair with both plan summaries,
  cards, cost, capacity, latency, bottleneck, reasons, and transfer metrics.
- `feasible_pairs.csv`: only system pairs satisfying phase, link, concurrency,
  TTFT, and TPOT constraints.
- `pareto_frontier.csv`: nondominated feasible system-level pair alternatives.
- `recommendation.json`: the selected pair with complete PD metrics, phase
  candidate summaries, transfer payload/time/capacity, assumptions, and all
  normalized phase/scenario/link inputs.
- `summary.txt`: the selected phase IDs, total cards, capacity, TTFT/TPOT,
  bottleneck, and SLO state, or the dominant rejection when none is feasible.

Reports are written even when no feasible recommendation exists; the command
then exits 1. Input and schema errors, invalid UTF-8, duplicate JSON keys, and
output I/O errors are concise stderr diagnostics and exit 2. A `pair-pd` result
root is published as one generation, so `prefill/`, `decode/`, and `pd/` never
mix results from different runs.

This is an analytical planning model, not a measured serving benchmark or a
P99 latency guarantee. The first version supports decoder-only dense/MoE models
with MHA/MQA/GQA, MLA, shared/routed experts, and supported full/linear-attention
hybrids. Encoder-decoder models, multimodal roots, and arbitrary custom
operators are rejected. Production selection should be calibrated against
kernel and end-to-end measurements on the target stack.

## Acknowledgement

This work is developed and maintained by Alimama AI Infra Team & Future Living Lab, Alibaba Group.
