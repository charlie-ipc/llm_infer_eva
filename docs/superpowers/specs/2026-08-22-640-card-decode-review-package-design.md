# 640-Card Decode Performance Review Package Design

## 1. Purpose

Create a self-contained review package for the accepted 640-card, decode-only
deployment estimate. The package must let a systems expert:

1. identify every modeling assumption;
2. trace batch, TP, attention-DP, MoE-TP, and EP placement to per-rank work;
3. reproduce the intra-node bandwidth sensitivity scan without generating a
   new CSV file;
4. distinguish the historical CSV estimate from the current communication
   precision model; and
5. identify known limitations before approving the deployment conclusion.

This work documents and reproduces the current model. It does not change the
search, collective, operation, memory, or ranking implementation.

## 2. Deliverables

### 2.1 Technical analysis

Add `docs/640-card-decode-performance-analysis.md` as the authoritative
calculation note. It will contain:

- model, accelerator, workload, and precision inputs;
- the selected 640-card parallel plan and topology;
- batch-to-rank and expert-token derivations;
- TP and EP payload formulas with substituted numbers;
- ring all-reduce and all-to-all timing formulas;
- GEMM, VECTOR, TP, EP, and total decode-step composition;
- the 100--800 decimal GB/s intra-node bandwidth scan;
- target-margin and diminishing-return interpretation;
- an explanation of all-reduce dissemination versus a separate broadcast;
- reconciliation with `results/decode_640_cards.csv`; and
- explicit limitations and expert review questions.

Formula symbols and units must be defined locally. Byte conversions use
decimal GB/s because the implementation multiplies bandwidth by `1e9`.

### 2.2 Reproduction script

Add `examples/analysis/scan_640_card_intra_node.py`. The script will:

- depend only on the repository and Python standard library;
- call `payload_bytes`, `all_reduce_cost`, and `all_to_all_cost` from the
  production package;
- fix the accepted 640-card plan and current communication formats;
- scan `intra_node_gbps` from 100 through 800 in 100 GB/s increments;
- keep `inter_node_gbps` fixed at 800 GB/s;
- print per-rank workload, payload sizes, selected paths, component latency,
  user token rate, and system token rate; and
- write no files.

The script is a transparent reproduction of the sensitivity calculation, not
a second deployment-search implementation.

### 2.3 Expert handoff

Add root-level `HANDOFF.md` as the review entry point. It will summarize:

- the review objective and repository revision;
- the accepted system configuration and headline result;
- exact reproduction and test commands;
- links to the technical note, script, historical CSV, and source formulas;
- assumptions that materially affect the answer;
- known gaps and review questions; and
- the expected expert sign-off output.

### 2.4 README entry

Add a short section to `README.md` linking the technical note, reproduction
script, handoff, and historical result. The README will remain an index rather
than duplicating the full analysis.

## 3. Fixed Calculation Contract

The review package uses the following fixed inputs:

```text
model: synthetic 10T MoE, 50B active parameters, 80 layers
hidden size: 8192
routed experts: 1024
experts per token: 4
shared experts: 0
context: 4096 tokens (4095 prior tokens plus one decode token)
precision: W4A4, KV4, VECTOR FP4
accelerator: 1024 TOPS GEMM, 32 TOPS VECTOR, 2 TB/s DRAM, 200 GB DRAM
system: 640 cards, 8 cards/node, 80 nodes
plan: replicas=16, batch/replica=64, cards/replica=40
parallelism: attention_tp=4, attention_dp=10, moe_tp=5, expert_parallel=8
communication: TP reduce=FP32, EP dispatch=FP4, EP combine=BF16
latency: intra-node=1 us, inter-node=5 us, collective launch=8 us
bandwidth scan: intra-node=100..800 GB/s, inter-node fixed at 800 GB/s
```

The calculation retains the accepted fixed compute components:

```text
GEMM = 20.801879 ms
VECTOR = 1.949727 ms
```

These are held fixed so the scan isolates intra-node communication bandwidth.

## 4. Required Derivations

The note and script must agree on these intermediate values:

```text
total batch = 16 replicas * 64 requests/replica = 1024
local attention requests = ceil(64 / 10) = 7
local routed assignments = ceil(64 * 4 / 8) = 32

attention TP payload = 7 * 8192 * 32 / 8 = 229376 bytes
routed MoE TP payload = 32 * 8192 * 32 / 8 = 1048576 bytes
EP dispatch payload = 7 * 4 * 8192 * 4 / 8 = 114688 bytes
EP combine payload = 7 * 4 * 8192 * 16 / 8 = 458752 bytes
```

All communication group sizes are at most eight, so compact placement selects
the intra-node path for attention TP, routed MoE TP, and EP. The target model
has no shared expert, so no shared-expert TP collective is added. Inter-node
bandwidth is therefore not on the active path for this plan.

The ring collective formulas are:

```text
all-reduce transfer bytes = payload * 2 * (group_size - 1) / group_size
all-reduce latency steps = 2 * (group_size - 1)
all-to-all transfer bytes = payload * (group_size - 1) / group_size
all-to-all latency steps = group_size - 1

collective time = transfer_bytes / (bandwidth_GBps * 1e9)
                + latency_steps * link_latency_us / 1e6
                + collective_launch_latency_us / 1e6
```

Per-layer collectives are multiplied by 80 layers. Total decode-step latency
is `GEMM + VECTOR + TP + EP`; per-user rate is `1000 / total_ms`; system rate
is `16 * 64 * per-user rate`.

## 5. Historical Result Reconciliation

`results/decode_640_cards.csv` remains unchanged as the accepted historical
search result. Its communication estimate predates independent communication
formats. The new review note must explain that the current model increases TP
traffic to FP32 reduce and EP combine traffic to BF16, so the current 800 GB/s
scan result is lower than the historical CSV value. Neither number may be
silently overwritten or presented without its precision assumptions.

## 6. Verification

Acceptance requires fresh evidence for:

1. the reproduction script exits zero and prints all eight bandwidth rows;
2. the 800 GB/s row is approximately 35.966099 user token/s and 36829.285
   system token/s;
3. all collective paths printed by the script are `intra_node`;
4. the script creates no output file;
5. documentation contains no unfinished placeholder content;
6. Markdown links resolve to tracked repository files;
7. `python -m unittest discover -s tests -v` passes; and
8. `python -m compileall -q infersim tests examples` and
   `git diff --check` succeed.

## 7. Out of Scope

- Re-running the full plan search or selecting a different 640-card topology.
- Regenerating `results/decode_640_cards.csv`.
- Modeling topology-aware contention between simultaneous independent
  collectives.
- Modeling switch oversubscription, duplex directionality, protocol headers,
  retries, congestion, or collective overlap with compute.
- Replacing analytical peak-throughput inputs with measured kernel curves.
- Claiming production capacity without expert review and hardware validation.

## 8. Expert Review Focus

The handoff asks reviewers to confirm or amend:

- whether compact placement of every TP/EP group within one 8-card node is
  operationally achievable;
- whether ring all-reduce/all-to-all are the right collective approximations;
- whether FP32 TP reduction, FP4 dispatch, and BF16 combine match kernels;
- whether collective latency is paid once per modeled launch per layer;
- whether the target model definition correctly excludes shared experts;
- whether compute and communication overlap should be introduced; and
- whether effective bandwidth efficiency or topology contention should reduce
  the nominal intra-node bandwidth before capacity sign-off.
