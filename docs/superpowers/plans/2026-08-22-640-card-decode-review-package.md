# 640-Card Decode Performance Review Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an expert-reviewable explanation, deterministic Python reproduction, handoff, and README entry for the corrected 640-card decode estimate.

**Architecture:** A small example module owns the fixed 640-card inputs and delegates collective timing to the production `infersim.cost.collective` API. A focused unit test locks the per-rank batch mapping, payloads, active intra-node paths, and 800 GB/s result; Markdown documentation explains those same values and links back to the implementation and historical CSV.

**Tech Stack:** Python 3.10+, standard-library `dataclasses`/`unittest`, existing InferSim schema and collective APIs, Markdown, Git.

---

## File Map

- Create `examples/analysis/scan_640_card_intra_node.py`: fixed-input, no-file-output sensitivity calculation.
- Create `tests/test_640_card_analysis.py`: independent invariants and CLI-output regression tests.
- Create `docs/640-card-decode-performance-analysis.md`: authoritative derivation and result interpretation.
- Create `HANDOFF.md`: concise expert review entry point and checklist.
- Modify `README.md`: link the review package without duplicating it.
- Modify `docs/superpowers/specs/2026-08-22-640-card-decode-review-package-design.md`: correct the previously assumed shared-expert TP term and 800 GB/s acceptance value.

### Task 1: Lock the corrected calculation contract

**Files:**
- Create: `tests/test_640_card_analysis.py`
- Modify: `docs/superpowers/specs/2026-08-22-640-card-decode-review-package-design.md`

- [x] **Step 1: Write the failing calculation tests**

Create tests that import `examples.analysis.scan_640_card_intra_node` and assert:

```python
class Scan640CardAnalysisTests(unittest.TestCase):
    def test_batch_mapping_and_payloads(self):
        self.assertEqual(analysis.LOCAL_ATTENTION_REQUESTS, 7)
        self.assertEqual(analysis.LOCAL_ROUTED_ASSIGNMENTS, 32)
        self.assertEqual(analysis.ATTENTION_TP_PAYLOAD_BYTES, 229376.0)
        self.assertEqual(analysis.ROUTED_TP_PAYLOAD_BYTES, 1048576.0)
        self.assertEqual(analysis.EP_DISPATCH_PAYLOAD_BYTES, 114688.0)
        self.assertEqual(analysis.EP_COMBINE_PAYLOAD_BYTES, 458752.0)

    def test_800_gbps_result_uses_only_target_model_collectives(self):
        row = analysis.calculate_row(800)
        self.assertAlmostEqual(row.tp_ms, 2.60217856)
        self.assertAlmostEqual(row.ep_ms, 2.450176)
        self.assertAlmostEqual(row.total_ms, 27.80396056)
        self.assertAlmostEqual(row.user_tokens_per_s, 35.9660989758)
        self.assertAlmostEqual(row.system_tokens_per_s, 36829.2853512)
        self.assertEqual(row.paths, ("intra_node", "intra_node", "intra_node"))

    def test_scan_has_eight_rows_and_prints_no_csv(self):
        rows = analysis.scan_rows()
        self.assertEqual([row.intra_node_gbps for row in rows], list(range(100, 801, 100)))
        output = io.StringIO()
        with redirect_stdout(output):
            analysis.main()
        self.assertIn("intra_GBps,tp_ms,ep_ms,total_ms", output.getvalue())
        self.assertNotIn(".csv", output.getvalue())
```

- [x] **Step 2: Run the focused test and verify RED**

Run: `python -m unittest tests.test_640_card_analysis -v`

Expected: import failure because `examples.analysis.scan_640_card_intra_node` does not exist.

- [x] **Step 3: Correct the approved design specification**

Remove the erroneous shared-expert TP implication, change the 800 GB/s
acceptance result from `34.339880` to `35.966099` user token/s and from
`35164.037` to `36829.285` system token/s, and state that the target synthetic
model has no shared expert.

### Task 2: Implement the deterministic reproduction script

**Files:**
- Create: `examples/analysis/scan_640_card_intra_node.py`
- Test: `tests/test_640_card_analysis.py`

- [x] **Step 1: Define fixed inputs and result type**

Use a frozen `ScanRow` dataclass with fields `intra_node_gbps`, `tp_ms`,
`ep_ms`, `total_ms`, `user_tokens_per_s`, `system_tokens_per_s`, and `paths`.
Define the accepted plan constants, current communication widths, fixed GEMM
and VECTOR milliseconds, and payload constants using production
`payload_bytes`.

- [x] **Step 2: Build the hardware schema through the public API**

Implement `_hardware(intra_node_gbps)` with `HardwareSpec.from_dict`. Supply
the accepted 200 GB capacity, 2 TB/s DRAM, 1024 TOPS W4A4/W4A8 GEMM, 32 TOPS
vector modes, eight cards per node, 1 us intra-node latency, 800 GB/s and 5 us
inter-node link, and 8 us collective launch latency.

- [x] **Step 3: Calculate only collectives present in the target model**

Implement `calculate_row(intra_node_gbps)` as:

```python
attention = all_reduce_cost(ATTENTION_TP_PAYLOAD_BYTES, ATTENTION_TP, hardware)
routed = all_reduce_cost(ROUTED_TP_PAYLOAD_BYTES, MOE_TP, hardware)
dispatch = all_to_all_cost(EP_DISPATCH_PAYLOAD_BYTES, EXPERT_PARALLEL, hardware)
combine = all_to_all_cost(EP_COMBINE_PAYLOAD_BYTES, EXPERT_PARALLEL, hardware)
tp_ms = LAYERS * (attention.seconds + routed.seconds) * 1000
ep_ms = LAYERS * (dispatch.seconds + combine.seconds) * 1000
```

Do not add a shared-expert all-reduce because the fixed synthetic model has no
shared expert.

- [x] **Step 4: Add the printable scan entry point**

`scan_rows()` returns exactly 100 through 800 GB/s. `main()` prints per-rank
work, all four payloads, a CSV-shaped stdout header, and eight formatted rows;
it does not open any output file.

- [x] **Step 5: Run focused tests and direct script**

Run:

```powershell
python -m unittest tests.test_640_card_analysis -v
python examples/analysis/scan_640_card_intra_node.py
```

Expected: tests pass; script prints eight rows and the 800 GB/s row reports
`2.602179,2.450176,27.803961,35.966099,36829.285`.

### Task 3: Write the analysis and expert handoff

**Files:**
- Create: `docs/640-card-decode-performance-analysis.md`
- Create: `HANDOFF.md`
- Modify: `README.md`

- [ ] **Step 1: Write the authoritative technical analysis**

Include sections for scope, fixed inputs, 640-card topology, batch mapping,
communication precision, payload derivations, ring collective formulas,
per-layer and 80-layer aggregation, corrected scan output, target margin,
all-reduce dissemination semantics, historical CSV reconciliation, modeling
limitations, and expert review questions.

- [ ] **Step 2: Write the handoff**

State the revision under review, reproduce with:

```powershell
python examples/analysis/scan_640_card_intra_node.py
python -m unittest tests.test_640_card_analysis -v
python -m unittest discover -s tests -v
```

Link the analysis, script, historical CSV, collective implementation, stage
communication assembly, precision schema, and design specification. Require
reviewers to return accepted assumptions, requested formula corrections,
calibration data, and a sign-off verdict.

- [ ] **Step 3: Add a concise README index**

Insert a `640-Card Decode Review Package` section before Acknowledgement with
links to `HANDOFF.md`, the technical analysis, reproduction script, and
historical CSV. Mention that the current result uses independent TP/EP
communication widths and that the CSV is preserved as a historical baseline.

- [ ] **Step 4: Check documentation consistency**

Run searches for unfinished placeholders and both old/new headline values.
The old `36.235890` value must occur only in historical reconciliation; the
current `35.966099` value must be identified as the 800 GB/s current result.

### Task 4: Verify, commit, and publish

**Files:**
- All files listed above.

- [ ] **Step 1: Run fresh focused and full verification**

```powershell
python -m unittest tests.test_640_card_analysis -v
python examples/analysis/scan_640_card_intra_node.py
python -m unittest discover -s tests -v
python -m compileall -q infersim tests examples
git diff --check
```

Expected: all commands exit zero, the full suite has zero failures, and the
script prints eight intra-node rows.

- [ ] **Step 2: Review scope and tracked files**

Run `git status --short`, `git diff --stat`, and `git diff`. Confirm the
historical CSV is unchanged and no generated output was added.

- [ ] **Step 3: Commit the review package**

```powershell
git add HANDOFF.md README.md docs/640-card-decode-performance-analysis.md docs/superpowers/specs/2026-08-22-640-card-decode-review-package-design.md docs/superpowers/plans/2026-08-22-640-card-decode-review-package.md examples/analysis/scan_640_card_intra_node.py tests/test_640_card_analysis.py
git commit -m "docs: explain 640-card decode performance model"
```

- [ ] **Step 4: Push and verify the remote revision**

Run `git push origin main`, then compare `git rev-parse HEAD` with
`git ls-remote origin refs/heads/main`. They must be identical.
