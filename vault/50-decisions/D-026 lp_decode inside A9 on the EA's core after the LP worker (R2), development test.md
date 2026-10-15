# D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test (2026-10-14, AgentRA)

**Status:** RESULT 2026-10-15 13:2x: **ADOPT** (hint 0.758; every no-harm leg and mechanism holds; § Result) — `-3way-ld0.yaml` is A9's development default. decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run; open to a
superseding decision on evidence. **Not run; no code merged** (branch `r1-bsync`, worktree `../i-trea-ii-r1`).
**Clarified and amended 2026-10-14 17:4x, before any run** (the section at the end: names as built, the labels
A9@r2ld / A9@r2ctl with new ids, the gate's final states, the no-harm band on item_placement and sing44 now gates, the
C1 re-read's harm leg, Berk's answer (a)). **Runs only
after [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]]
has its verdict** (and so after D-023's), never in D-025's window before that verdict — two changes would stack.
Development only (D-010). Candidate R2 of [[2026-10-14 C1 diagnosis and improvement candidates]] §5 (the note's
candidate, not D-021's rule R2). A C1 item: D-019's addendum lists "lp_decode in A9 (core layout)"; this resolves #30's
lp_decode question without re-splitting F; #27.

## Why
[[D-017 LP decode — re-optimise the continuous variables of selected candidates]] fixes the integers of a few candidates
per generation and solves the LP over the continuous variables; it polishes feasible candidates and completes infeasible
ones, and it is A3's default (`configs/itrea2-dev-rc-ld.yaml`, f030e23909ad). `pair.py` refuses it in A9 (and A3x) for
two reasons: the decoder would share the EA's single X925 core (with the LP worker of `seeding.lp_source: highs_async`);
and one more pending batch takes the candidate set's concatenate past CuPy 14.2's 8-part limit, above which CuPy switches
to a pointer-array kernel that nothing warms — it would compile on the clock and void the run. After R1 (D-025), core 6
has room for the decoder.

**Evidence (the note's §5 R2).**
- D-017 (A3, 6 instances × 2 seeds): mean d +0.219 [+0.041, +0.429]. item_placement 0.311 → 0.118, sing44 0.377 →
  0.238; ca +0.005, glass4 about 0.
- **The mechanism is solid:** item had 43 / 47 decoded incumbents per run, 11 / 17 of them from completion; sing44 had 5
  per seed, and its +0.369 lies beyond its band of 0.11.
- **The P anchor is weaker:** item's D-017 gain lies inside its own D-016 band of 0.68; A3 controls on item range
  0.134–0.469 across seeds; D-017 also records that lp_decode changes the random stream (fhnw moved +0.348 with no decoder
  row).
- Decoder CPU in A3@ld1: item about 2.4–2.9 s, sing44 about 2.3 s, ca about 14–15 s, glass4 about 0.3 s per 120-s run.
- In A9 the EA carries both target instances (EA credit 0.382 on item, 0.483 on sing44; the note's §3).
- Non-singleton continuous columns: item 33, sing44 15,995, ca 100,704, glass4 19, kumeu 658.

## Decision — what is tested
**The change:**
- **Concatenate:** deterministic warm-up entries for a 9+-part concatenate in each dtype used (gene dtype, float64 key /
  violation, bool feasible). The pending batches are not restructured.
- **Placement:** the decoder runs on the EA's core.
- **Gate:** the decoder's priming starts only after the LP worker has reported done or failed, so it never takes CPU from
  the LP worker in any branch. This resolves the conflict with R1 on ca: in A3@ld1 the decoder used about 10 s of solve
  CPU on ca and was ready at 4.3 s, which would overlap ca's 17-s LP.
- **Priority (after the gate), fixed now from D-025's measurement:** the input is item_placement_9903's EA self-CPU per
  wall-second in the adopted R1 variant's own set (`r1-bsync`, or the fallback set that was adopted in its place), the mean
  of its two treatment runs: **< 0.5 → the decoder at nice 10 relative to the EA host; ≥ 0.5 → nice 0.** In r13x it is
  about 1.0 (120.55 CPU-s over 120 s), so whether that was spin or real host work decides the branch.
- **The refusal** in `pair.py` is lifted by this record **only for an EA configuration that also carries the gate**; any
  other `lp_decode` on one X925 core stays refused.
- **Unchanged:** F, SCIP's core and cuOpt's cores, so D-021 R5's M2 soak does not repeat. **Memory fits:** A3@ld1 ran the
  EA, the LP worker and the decoder under the same 6-GB MemoryMax, with EA maxrss 0.65–0.86 GB.

**Configuration.** A new variant file = the development default as D-023 and D-025 leave it, verbatim, + `repair.continuous:
lp_decode` + the gate (its field registered in `config._LATE_DEFAULTS` like D-025's, so every existing hash is unchanged):
`configs/itrea2-dev-rc-3way-<default variant>-ld.yaml` — `-bsync-ld.yaml` if D-025 adopts bsync and D-023 does not adopt
boY, `-boY-bsync-ld.yaml` if both adopt, the adopted D-025 fallback's stem + `-ld` if one was adopted in bsync's place,
and `-ld.yaml` / `-boY-ld.yaml` in the branch where R1 is not adopted (below). The note's arms {A9@bsync+ld, A9@bsync} are
the first case.

## Test — fixed before any run
- **Set `r2-ld`**, job file `jobs/r2-ld.jsonl`, arms {treatment `A9@r2ld` = the `-ld` file, control `A9@r2ctl` = the
  development default it is built on, re-run in the same window}, 120 s, harness **ext**, one block per instance × seed:
  - hint instances {item_placement_9903, sing44} × seeds **0–3**: 16 runs (item's within-day seed spread is large, 0.355 /
    0.261 in bo1s; more seeds address that noise while the modelled effect stays reachable);
  - no-harm instances {ca__Scenario400_reserves_1, glass4, neos-3656078-kumeu} × seeds 0–1: 12 runs;
  - **28 runs** (≈ 1.6 h at 3.44 min per run, excluding the window's open and close, the checks below and D-021's 0.55).
- **Validity (must hold):** 28 of 28 completed or a recorded void rerun; 0 compile on the clock, the 9+-part concatenate
  included; equal P and gene dtype per block; harness ext on every run; each arm's EA config hash as in the job file; the
  decoder ran in every treatment run and in no control run — else INVALID. Failed runs are outcomes of their arm
  (D-002 §6, `--failed keep`), reported both ways.
- **Mechanism checks (gating):**
  - **ready ≤ 10 s after the gate opens** (D-017 M0, read on the worker's own ready stamp `ready_t_worker`), on every
    treatment run where the gate opened. **Re-anchored here:** D-017 read 10 s from t0; under the gate the decoder cannot
    prime before the LP worker ends, and ca's LP takes about 17 s of CPU, so a limit from t0 would fail on ca by
    construction;
  - **≥ 1 decoded incumbent per seed** on item_placement and sing44 (all 8 treatment runs);
  - **the gate costs the LP nothing:** ca's LP-worker solve time (the worker's own stamps, D-025's instrumentation) in
    the treatment ≤ 1.10 × the same-day control's, per seed (the note's "within 10 % of R1's"; the control is R1's
    configuration, and longer is the harm, so one-sided).
- **Reported, not gating:** `lpd_wait_s_max` and `lpd_host_s` against A3@ld1 (D-017's M1 failed by the letter on item:
  1.39–1.63 % of wall); decoder CPU; decoded incumbents per run and from completion; the LP worker's finish times.
- **No-harm (gating, one-sided, on seed means, spreads in raw P):**
  - **ca:** P̄(A9@r2ld) − P̄(A9@r2ctl) ≤ max(**0.02**, the control's seed-0 vs seed-1 spread). ca's spread is 0.005, so
    0.05 would let a lost R1 gain pass;
  - **glass4 and kumeu:** P̄(treatment) − P̄(control) ≤ max(0.05, the control's seed-0 vs seed-1 spread).
- **Hint (futility-only, fixed before the runs):** the ratio of means of P(120), A9@r2ld / A9@r2ctl, over {item_placement,
  sing44} × seeds 0–3 **≤ 0.90**, reported per instance as well as pooled.
  - **Null replicate** at 2 seeds: **0.918**. The null SD is about 0.06 at 2 seeds and about 0.04 at 4, so 0.90 is about
    2.4 null SDs; the modelled range 0.64–0.88 stays reachable.
- **Verdict:** **ADOPT** iff validity, the three mechanism checks, the no-harm rules and the hint all hold; **NOT
  ADOPTED** otherwise, and the report names the rule; **INVALID** when a validity leg fails. No fallback beyond the
  branches below is pre-stated.
- **Analysis:** a report script (`tools/d026_report.py`), written and tested before the window, reading only the run table
  and the set's run directories.

## Branches fixed now
- **R1 adopted** (bsync, or a D-025 fallback in its place): as above, on that default.
- **R1 not adopted — the pre-written fallback:** the decoder still gated after the LP, at **nice 0** beside the spinning
  host, tested against the development default as it stands (`-ld.yaml` against `itrea2-dev-rc-3way.yaml`, or
  `-boY-ld.yaml` against `-boY.yaml`), same set design, same rules; the LP check reads against that same-day control.
- **D-023 adopted boY:** lp_decode then runs beside active back-off for the first time (`_bo_pace` / `_lpd_poll` /
  `lpd_sync` interact), so back-off × lp_decode CPU tests are owed before the window.

## What adoption would mean
As D-025 defines it: the line widens; the `-ld` file becomes **A9's and A3x's development default** (new hash) and
`configs/frozen-v2.yaml` carries it; nothing claims that R2 improves A9. C1's reading follows Berk's open question 1
(default (a): one re-read on the frozen-v2 hash with A6F re-run in the same window, 68 runs ≈ 3.9 h), run once after this
record's verdict.

## Before the first timed run
**GPU checks (inside a lease, untimed):**
1. **A cold-cache GPU test that provably reaches the decoder path in the A9 layout** (EA on one X925 core, with
   `exchange.from_cuopt`): a decoded candidate goes through the 9+-part concatenate in every dtype used, from an empty
   CUDA cache, with `compiled_on_clock` false.
2. **The gate:** the decoder's priming starts only after the LP worker's done or failed report (a slow LP worker or a
   stub; asserted on the stamps), in the A9 layout.
3. **The priority:** the decoder's nice as the branch says, read back from `/proc`.
4. The repository's GPU tests (`pytest -m gpu`) on the merged code.

**CPU tests before the merge:** every existing configuration keeps its hash (D-025's list); every existing job file
regenerates byte for byte; the refusal is lifted only with the gate; `make-jobs r2-ld` writes the 28 runs on the
default it is given, with distinct run ids; back-off × lp_decode tests if boY is the default; the report's legs, each with
a test.

## Prediction (not a criterion)
Modelled; interim; the transfer from A3 to A9 is unmeasured. If ¼ to ¾ of D-017's effect transfers: item −0.06 to
−0.18, sing44 −0.04 to −0.10 (a ratio over the two of about 0.64–0.88). ca, glass4 and kumeu near 0. No action on the
three losses.

## Risks
- The transfer from A3 (3 A725 cores, no cuOpt) to A9 (core 6, GPU contention) is unmeasured.
- Host overhead per offer: D-017's M1 failed by its letter on item (1.39–1.63 % of wall).
- A new pending-batch shape is a compile-on-clock risk until the cold-cache test reaches it.

## Cost and code owed
Code moderate (about 1 day plus review; the engine's pending-batch path has not been read, so the scope is uncertain),
on branch `r1-bsync` after R1's code, merged after the running pool ends: the warm-up entries, the gate, the priority,
the scoped lift of the refusal, the `-ld` variant file once D-025 has decided (written then from this record's rule),
`make-jobs r2-ld`, `tools/d026_report.py`, and tests. Box: 28 runs ≈ 1.6 h. Needs Berk: no; a development decision record
and a normal window.

## Clarification and amendment 2026-10-14 17:4x — before any run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
From the same adversarial review as [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]]'s
section of this time (branch `r1-bsync`; build commit 61c552e, fixes in 91d38cd). No `r2-ld` run exists.

**Berk's answer to open question 1 (2026-10-14 16:0x):** **"Re-run C1 on frozen hash"** — C1 is re-read once on the
exact frozen-v2 hash with A6F re-run in the same window (68 runs ≈ 3.9 h), after this record's verdict.

**Waits for the R1 line's final outcome** (D-025's fallbacks included), not only for bsync's first verdict.

**Names as built:**

| item | record | build (binding from here) |
|---|---|---|
| fields | the gate in `_LATE_DEFAULTS` | `repair.lp_decode_gate: none \| lp_worker` and `repair.lp_decode_nice` (the decoder's nice increment over the host; default 10 = the old constant), both in `_LATE_DEFAULTS`: every existing hash stands |
| set, job file | `r2-ld`, `jobs/r2-ld.jsonl` | the same; **not committed before D-025's verdict**; written then with `itrea make-jobs r2-ld --base bsync\|boY-bsync --nice 10\|0` (or `--ea-cpu X`, D-025's measured mean EA self-CPU per wall-second on item_placement_9903 → the rule below; no default) or `--base r13x\|boY` (R1 not adopted); sha256 into that day's journal |
| labels | `A9@r2ld` / `A9@r2ctl` | the same; the control runs its base's configuration file under `A9@r2ctl`: the same run hash as R1's treatment (bsync: 330a10a77556) or A9@r13x / A9@boY, a new run id. The build had used `A9@bsync` / `A9@bsync-ld`, re-using R1's ids; withdrawn |
| files, R1 adopted | `-bsync-ld.yaml`, `-boY-bsync-ld.yaml` | `-3way-bsync-ld.yaml` (decoder nice 10), `-3way-bsync-ld0.yaml` (nice 0), `-3way-boY-bsync-ld.yaml`, `-3way-boY-bsync-ld0.yaml`: the suffix encodes the decoder's nice (`ld` = +10, `ld0` = +0) |
| files, R1 not adopted | `-ld.yaml` / `-boY-ld.yaml` (nice 0) | **`-3way-ld0.yaml` / `-3way-boY-ld0.yaml`**: named by the same nice code, and kept apart from A3's `itrea2-dev-rc-ld.yaml` (this record's own name-clash risk) |
| a fallback base | "the adopted D-025 fallback's stem + `-ld`" | **not written**: after D-025's verdict, with a statement of the decoder's nice against a reniced host (under bnice the decoder, started after the renice, inherits the host's raised nice) |
| order, rerun | — | positions alternate (the control first at even seeds); a rerun gets its own set (`--set`) |
| report | `tools/d026_report.py` | written and tested: `itrea d026-report --base … [--nice 10\|0]`; on today's table INVALID, 0 of 28 present |

Hashes (EA configuration / A9 run hash under ext): `-bsync-ld` 119c6e17f830 / dc45bdbb771f; `-bsync-ld0` b7af15f4b62a /
93a8ed06b59b; `-boY-bsync-ld` c296898cddc4 / 3ef4bf1254fe; `-boY-bsync-ld0` 2b6baa898edf / 37a47a123c67; `-ld0`
a05af2c732dc / 341267810eed; `-boY-ld0` aabac30b5056 / 58f72d467a23.

**The gate, as built:** it opens on any **final state** of the EA's LP worker — done, no_solution, failed or stopped
(the 60-s LP limit: chromaticindex1024-7, rail01; under "done or failed" the decoder would never start there) — **when
the EA reads that state at a generation boundary** (`gate_t`), not when the worker finishes (`lp.worker_done_t`); the
whole decode worker's start waits (its read of the instance included). The lag `gate_t − worker_done_t` is printed per
run (reported, not gating). The refusal in `pair.py` is lifted only for a configuration with the gate **and**
`seeding.lp_source: highs_async` (without the EA's LP worker the gate opens at t0 and gates nothing).

**Validity, made exact:** "each arm's EA config hash as in the job file" = each run's id is its job's (configuration,
harness, footprint), and the base's flag state holds in both arms (bsync base: blocking sync verified; otherwise CUDA's
default read back); "the decoder ran" = its gate opened and its worker started (summary `lp_decode.gate` = "open"); in
no control run is there an `lp_decode` block. Failed runs: outcomes of their arm (`keep`, D-022) for the verdict, the
letter beside; D-027 keeps the letter, deliberately.

**GPU check 1 as met:** two tests together — the concatenate probe (9- and 12-part concatenates, one-row and mixed, in
the gene dtype, float64 and bool: nothing new compiles after the warm-up) and the A9-layout cold-cache run (decoded
copies entered, nothing compiled after t0). A toy run need not reach 9 parts; every run now records `concat_parts_max`.
**Check 2** is asserted on the stamps: `gate_t` ≥ `lp.worker_done_t` and `ready_t` ≥ `gate_t`.

**Amendment (rule change before any run):**
1. **No-harm on the hint instances:** item_placement_9903 and sing44 each P̄(A9@r2ld) − P̄(A9@r2ctl) ≤ max(0.05, the
   control's range over its four seeds), one-sided, gating — so a large gain on one cannot carry a loss on the other
   through the pooled hint (D-024 rule 3). ADOPT now needs validity, the three mechanism checks, every no-harm leg (ca,
   glass4, kumeu, item_placement, sing44) and the hint.
2. **The C1 re-read's harm leg** is D-025's (one re-read covers both changes): any harm goes to Berk beside the go /
   no-go, and the frozen-v2 candidate is not presumed.

**Correction:** the null replicate at 2 seeds is 1.484 / 1.618 = **0.917** (the record says 0.918).

## Result (2026-10-15 13:2x, AgentRA): ADOPT — `-3way-ld0.yaml` becomes A9's and A3x's development default
D-025's line ended without adoption (bsync and ysync NOT ADOPTED, 08:1x / 11:2x; D-029's follow-up pending), so R2 ran on the
pre-written branch "R1 not adopted": `itrea make-jobs r2-ld --base r13x` → `jobs/r2-ld.jsonl` (sha256 1ad5191f…8042), A9@r2ld
(`configs/itrea2-dev-rc-3way-ld0.yaml`: the decoder gated after the LP worker, nice 0 beside the spinning host; run hash
341267810eed) vs A9@r2ctl (`-3way.yaml`, 28e5a5e9c725). Pool `20261015-111740-3344629` (11:17–13:16): **28 of 28 completed**, 0
failed / void. `itrea build-runs`, `itrea d026-report --base r13x` (VALID):
- **Mechanisms hold:** (1) ready after the gate 0.23–0.26 s on item / glass4 / sing44 / kumeu, 5.83 / 5.91 s on ca (≤ 10 s); (2)
  decoded incumbents per seed item 54 / 22 / 45 / 38, sing44 6 / 2 / 5 / 3 (≥ 1); (3) ca's LP worker solve wall 37.87 vs 36.46 s
  (1.039) and 37.76 vs 40.19 s (0.940) (≤ 1.1).
- **No harm, every leg:** ca −0.006 (band 0.021), glass4 −0.030 (0.050), kumeu −0.054 (0.138), item_placement −0.133 (0.104),
  sing44 −0.062 (0.050) — all in the treatment's favour.
- **Hint PRESENT: 0.758** (ΣP 2.4362 / 3.2145 over item_placement and sing44 × seeds 0–3; threshold 0.90): item_placement 0.598
  (P 0.330 → 0.198; better on all four seeds: 0.162 / 0.286 / 0.187 / 0.155 vs 0.261 / 0.365 / 0.362 / 0.334), sing44 0.869
  (0.473 → 0.411; better on all four seeds).
- **Verdict: ADOPT.** As fixed above: `configs/itrea2-dev-rc-3way-ld0.yaml` (EA hash a05af2c732dc) becomes A9's and A3x's
  development default and the frozen-v2 candidate; nothing claims that R2 improves A9 in general (development, n = 2–4 seeds on
  5 instances). C1 is re-read once on the frozen-v2 hash with A6F re-run in the same window (Berk 2026-10-14: 68 runs ≈ 3.9 h),
  with D-025 amendment 3's harm leg against A9@r13x per instance. If D-029 (running, set r1f-ysync, control pinned to `-3way.yaml`)
  also adopts ysync, frozen-v2 would combine ysync + ld0 — never run together; that combination is Berk's call before the C1 re-run.
- Code follow-up (not before the running pool ends): `ITX_A9_CONFIG` in `make_jobs.py` (and wherever the runner names the A9 /
  A3x default) points at `-3way-ld0.yaml` — a src change, merged only between pools.
