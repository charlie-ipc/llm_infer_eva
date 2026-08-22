# 640-Card Decode Performance Review Handoff

## Review Objective

Review the analytical estimate for a 640-card, decode-only, 10T-class MoE
deployment at batch 1024 and a target of 30 token/s per user. The current model
predicts 35.966099 token/s per user at 800 GB/s intra-node bandwidth and
33.817553 token/s at 100 GB/s.

Review the commit containing this file. Record the exact revision with:

```powershell
git rev-parse HEAD
```

## Start Here

- [Full calculation and assumptions](docs/640-card-decode-performance-analysis.md)
- [Deterministic Python reproduction](examples/analysis/scan_640_card_intra_node.py)
- [Historical 640-card result](results/decode_640_cards.csv)
- [Review-package design](docs/superpowers/specs/2026-08-22-640-card-decode-review-package-design.md)
- [Implementation plan](docs/superpowers/plans/2026-08-22-640-card-decode-review-package.md)

Relevant production formulas:

- [Collective path and ring timing](infersim/cost/collective.py)
- [Stage TP/EP communication assembly](infersim/cost/stage.py)
- [Independent communication precision](infersim/schema/precision.py)

## Fixed Review Configuration

```text
model: synthetic 10T MoE, approximately 50B active, 80 layers, hidden 8192
experts: 1024 routed, top-k 4, no shared expert
context: 4096 tokens
precision: W4A4, KV4, TP reduce FP32, EP dispatch FP4, EP combine BF16
accelerator: 1024 TOPS GEMM, 32 TOPS VECTOR, 2 TB/s DRAM, 200 GB DRAM
system: 640 cards, 8 cards/node, 80 nodes, 16 model replicas
replica plan: batch 64, 40 cards, attention TP/DP 4/10, MoE TP/EP 5/8
scan: intra-node 100..800 decimal GB/s; inter-node fixed at 800 GB/s
```

## Reproduce

From the repository root:

```powershell
python examples/analysis/scan_640_card_intra_node.py
python -m unittest tests.test_640_card_analysis -v
python -m unittest discover -s tests -v
python -m compileall -q infersim tests examples
```

The script writes no file. Its 800 GB/s row should be:

```text
800,2.602179,2.450176,27.803961,35.966099,36829.285,intra_node,intra_node,intra_node,intra_node
```

## Important Model Semantics

- System batch 1024 is explicitly modeled as 16 replicas times batch 64.
- The worst attention rank uses `ceil(64/10)=7` requests.
- Routed MoE TP uses `ceil(64*4/8)=32` token-expert assignments.
- Ring all-reduce includes reduce-scatter and all-gather, so no additional
  broadcast is charged after TP reduction.
- BF16 EP combine counts all top-k expert hidden vectors returning to the
  requesting rank; the weighted sum itself is local VECTOR work.
- TP=4/5 and EP=8 fit within `cards_per_node=8`, so the compact-placement model
  selects intra-node links for all four collectives.
- Compute and communication are added serially; overlap is not modeled.

## Correction Recorded During Review Preparation

The historical CSV used 4-bit communication for all three phases and reports
36.235890 token/s. Independent communication formats reduce the current
800 GB/s estimate to 35.966099 token/s.

An earlier transient scan also added a shared-expert TP all-reduce and reported
about 34.339880 token/s. That term was removed because the fixed target model
has no shared expert. The focused regression test locks this correction.

## Requested Expert Output

Please return:

1. `APPROVED`, `APPROVED WITH CALIBRATION`, or `NOT APPROVED`;
2. accepted and rejected assumptions, with corrected values;
3. any corrected payload, collective, or topology formula;
4. measured launch latency, effective link bandwidth, and overlap data if
   available; and
5. whether the 30 token/s requirement is average, percentile, or hard minimum.

The highest-risk assumptions are compact rank placement, full per-group link
bandwidth without contention, communication formats, ring collective choice,
and the absence of compute/communication overlap.
