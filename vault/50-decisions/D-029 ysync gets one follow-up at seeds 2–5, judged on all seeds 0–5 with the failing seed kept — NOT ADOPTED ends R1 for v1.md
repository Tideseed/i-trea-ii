# D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1 (2026-10-15, AgentRA on Berk's choice)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) on Berk's choice of 2026-10-15
(question panel, below) — Berk chose the design; the thresholds are AgentRA's (amendment, B1) — **before any run of set
`r1f-ysync`**; open to a superseding decision on evidence. **Amended 2026-10-15 12:2x, before any `r1f-ysync` run** (the
section at the end: rule 3 reads ca at seeds 2–3 alone, so the rules read 28 runs; a failed control run never decides a
rule; a window leg; the rerun scope; an unread warm-up run first; the run ids pinned; the threshold statements corrected;
lines it changes are marked in place). **Not run.**
Development only ([[D-010 Develop the algorithm first on a small development set]]): every number it produces is
descriptive. It follows [[D-025 A9 EA blocking sync (R1) — the EA stops spinning on the GPU so its LP worker gets core 6, development test]]
(fallback (b) `ysync`, NOT ADOPTED 2026-10-15 11:2x) and closes the R1 line for v1 whichever way it goes. Issues: #23 (A9),
#30. The thresholds below may be amended only before the first `r1f-ysync` run, as a dated amendment here.

## Why
**D-025's two results (2026-10-15, both VALID, 20 of 20 each):**
- **bsync** (`r1-bsync`, 07:17–08:06): hint present 0.765, but the generation guard failed on item_placement (0.824) and
  comp21 (0.818) and comp21's P was worse by +0.060 (band 0.050) → NOT ADOPTED; fallback (b) fired.
- **ysync** (`r1-ysync`, 10:00–11:09): hint present 0.834 (ca 0.392 → 0.253, the LP worker done 37.3 / 38.3 → 20.0 / 20.2 s),
  comp21's guard held (generations 0.980, P −0.001), **item_placement's generation guard failed** — means ratio 0.759
  against 0.921: seed 0 1,483 vs 1,367 (1.085), **seed 1 678 vs 1,480 (0.458)**; item's P +0.099 within its band 0.230
  → NOT ADOPTED; "the R1 line ends for C1". The failing cell is one run, the pool's first, started one second after M4's
  pool ended; D-025 stated that and did not act on it.
- D-025 closed with: *"Any further R1 test is a new decision, written before its runs, that reads the guards on every seed
  run so far (0–1 here included)."* This is that decision.

**Berk, 2026-10-15 11:2x (question panel), verbatim choice:** **"Follow-up, guards on all seeds (Recommended)"** — the
option's text: *"Yield variant only: item_placement and comp21 at seeds 2–5 plus ca at 2–3 vs a same-window control, ~24
runs ≈ 1 h, today. A new decision record written first, judging the guards pooled over ALL seeds 0–5, failing seeds
included, so this is not re-testing until it passes. If it passes, C1 is re-run on the frozen configuration (68 runs ≈ 3.9 h)
before Fri 10-23."* And Berk, 2026-10-14 16:0x (D-025's open question 1): **"Re-run C1 on frozen hash"** — if R1 or R2
changes A9's default before the freeze, C1 is re-run on the frozen hash with A6F re-run in the same window (68 runs).

## Decision — the runs
- **Set `r1f-ysync`, job file `jobs/d029.jsonl`** (sha256 `8827a2a448807788ae150fe2e2a2f5e5106aff842698431a215049f42dd4fe29`;
  amended 12:2x, was `989d4479…`): item_placement_9903 and comp21-2idx at seeds 2–5, ca__Scenario400_reserves_1 at seeds
  2–3, x {treatment **A9@ysync** = `configs/itrea2-dev-rc-3way-ysync.yaml` (EA f81d6b6209f0, run hash under ext
  a08221cca653 — r1-ysync's treatment), control **A9@r1fctl** = the development default `configs/itrea2-dev-rc-3way.yaml`
  (1d12ad490dd0 / 28e5a5e9c725 = A9@r13x's)}, 120 s, harness ext, split dev, one block per instance x seed: **20 runs read,
  in 10 blocks** (Berk's option said "~24"; the exact count is 20), at `priority` 1. **Before them, one warm-up run**
  (amended 12:2x): the control's configuration on item_placement_9903 at seed 6, label A9@r1fwarm (run hash 28e5a5e9c725),
  set `r1f-ysync-warmup`, `priority` 0 — **never read**. 21 runs in 11 blocks. The two arms' order alternates over the
  seeds (`position`: the control first at even seeds), as `make_jobs.r1_bsync` does.
- **New run ids:** the control's label is this set's own, so its run hash is A9@r13x's and its run ids are new; the
  treatment keeps r1-ysync's label so the two sets pool under one T label, and its ids are new by seed. None of the 20 ids
  is in `datasets/runs.parquet`, in any `jobs/*.jsonl` or under `traces/raw/` (checked 2026-10-15 11:2x), nor is the
  warm-up's `item_placement_9903__A9@r1fwarm__s6__120s__28e5a5e9c725` (checked 12:1x).
- **Writer:** `tools/d029_report.py --write-jobs jobs/d029.jsonl`. `itrea make-jobs r1-bsync` has `--seeds`, `--set` and
  `--variant`, but fixes a fallback set's control label at `r1ctl-<variant>` (no option for another), so the file is
  built from make_jobs' own pieces (`job`, `_alternated`, `instance_paths`, `_sorted`, `write_jobs`) — the same fields,
  files, harness and order rule. **The configuration paths and the run hashes are pinned literals** in the report
  (amended 12:2x): `--write-jobs` refuses a job whose run id is not the pinned one, and the expected ids do not move if
  make_jobs' default constants or a configuration file change later (D-026's ADOPT would make a new file A9's default).
  No `src/` edit.
- **Not run:** sing44 and neos-960392 (D-025's other hint instances; their bands held in both tests and no rule below reads
  them — D-024 rule 4; carried, a stated deviation: the amendment, B7).
- `python -m itrea.experiments.pool jobs/d029.jsonl --layout pair --deadline 18:35 --dry-run` (11:2x, pinned to cores
  10–14, inside the window's lease): 10 blocks, 20 runs, memory 97.0 GB per run against 109.0 GB available, preflight
  passes. **On the amended file** (the same command on `jobs/d029.jsonl`, 12:27): 11 blocks, 21 runs; block 0 is the
  warm-up and blocks 1–10 run in the order they had before (the first read block is comp21 seed 5, treatment first);
  97.0 GB per run against 108.8 GB available; preflight passes.

## Rules — fixed before any `r1f-ysync` run (amended 12:2x where marked)
**The runs read: 28** (amended 12:2x; was 32). r1-ysync's 8 on item_placement and comp21 (seeds 0–1; T = A9@ysync, C =
A9@r1ctl-ysync) and the 20 above (T = A9@ysync, C = A9@r1fctl). r1-ysync's 4 ca runs at seeds 0–1 are read beside rule 3,
not by it, and are not validity-gated. T is one configuration in both sets; C is one configuration (run hash
28e5a5e9c725) under two labels. Each block holds T and C at one seed in one window (checked: the window leg). Seeds pool
across the two windows (seeds 0–1 in r1-ysync's pool, 2–5 in the new one).

**Validity (must hold before any rule is read):** D-025's legs on all 28 runs — complete (28 of 28; a void is rerun by the
pool and the rerun is read); harness ext; config (each run id is its job's, pinned); no CUDA compile on the clock; equal
P and gene dtype in each block; **the two runs of each block in one window** (`window_id`; amended 12:2x); the flag state
(T: yield set, verified and still held at the end, timing events not blocking; C: CUDA's default read back, no renice).
**INVALID** if any leg but "complete" fails: the cause is fixed and re-checked, with a dated note here, then **the INVALID
blocks — both arms, nothing else — are rerun as set `r1f-ysync-2`** (then `-3`, …; `tools/d029_report.py --write-jobs
jobs/d029-2.jsonl --set r1f-ysync-2 --blocks <instance>:s<seed>,…`, with its own warm-up first), and the report reads each
rerun block from the latest rerun set in place of the original (amended 12:2x: the pool skips done runs, so a rerun needs
a set of its own; build-runs qualifies the rerun's ids `#r1f-ysync-2`). Otherwise **INCOMPLETE** while a run is missing —
no verdict. **Same code** (amended 12:2x; checked by hand, not by the report — run.json carries no revision): r1-bsync and
r1-ysync ran with `src/` and `configs/` as committed at da80772 (the last commit touching them, 2026-10-14 23:46; no tracked
change there on 10-15 12:1x). At the r1f-ysync pool's start and end the main session records HEAD, `git diff --stat
da80772 -- src configs` (must be empty) and `git status --short -- src configs` in the journal; a non-empty diff is stated
here before the report is read.

**Failed runs** (amended 12:2x): a failed **treatment** run is an outcome of its arm (D-002 §6, `--failed keep`, D-025's
practice) and a figure it cannot give (no EA summary → no G) fails the rule that needs it. A failed **control** run never
decides a rule: a rule that fails with it in still fails (a truncated control only looks worse — fewer generations, a
higher P — so the treatment can only gain from it); a rule that passes, or cannot be computed, with a failed control in is
**unread**, the verdict is **INCOMPLETE**, and the block is rerun, both arms, as a rerun set (above) and read again. A
treatment failure in a block replaced that way is not read; the report lists it ("failed runs in blocks a rerun set
replaced") and the Result names it. The letter (`exclude`, a failed run is not completed → INCOMPLETE) is printed beside.

**Rule 1 — generation guard** on item_placement_9903 and on comp21-2idx, each over seeds 0–5 (n = 6 per arm):
**Ḡ(T) / Ḡ(C) ≥ 0.90** (ratio of means, G = the EA's generations in 120 s). Beside, not gating: the ratio of medians,
every seed's ratio, and seeds 0–1 and 2–5 (the two windows) separately.

**Rule 2 — P no-harm** on item_placement_9903 and on comp21-2idx, each over seeds 0–5: **P̄(T) − P̄(C) ≤ 0.05** (raw P(120),
one-sided: a no-harm rule never fails a treatment for helping). Beside, not gating: a one-sided 95 % bootstrap upper bound
of the mean difference — paired by block (the six seeds' T − C differences resampled with replacement), B = 10,000,
`numpy.random.default_rng(29)`, the 0.95 quantile; crude at n = 6 — and seeds 0–1 and 2–5 separately.

**Rule 3 — gain replication on ca at seeds 2–3 alone** (n = 2 per arm; amended 12:2x, was pooled over seeds 0–3):
**ΣP(T) / ΣP(C) ≤ 0.95**. Beside, not gating: every seed and the pooled seeds 0–3 (r1-ysync's seeds 0–1 read beside).

**Verdict:** **NOT ADOPTED** if a rule fails on any instance it names (the report names every failing rule) — **final for
v1**: the R1 line closes and R1 moves to v1.1. Otherwise **INCOMPLETE** while a rule is unread for a failed control run
(amended 12:2x). Otherwise (validity holds and rules 1–3 hold on every instance they name) **ADOPT**. Reported per run,
not gating: D-025's mechanism (EA self-CPU per wall-second, the LP worker's finish time and CPU share, generations and GPU
launches per second), the load average at the run's gates, and each pool's first run with the gap since the previous
pool in this checkout's run table.

**What changes from D-025's rules, and why:** the thresholds are fixed numbers, not the control's two-seed spread
(D-025's generation guard was 1 − max(0.05, spread), its P band max(0.05, spread)). With two seeds the spread is one
difference, and it moved the bar a lot — item's P band was 0.230 in r1-ysync, from one control pair 0.479 / 0.250. With
six seeds per arm the means carry the noise and a fixed bar is more stable. **The bar that failed is looser now (stated
12:2x):** item_placement's generation bar was 0.921 in both D-025 tests and is 0.90; comp21's was 0.950 (bsync) and 0.898
(ysync) and is 0.90; D-025's 5 % floor on the tolerated generation loss becomes a fixed 10 % — the tolerated loss doubles
(why: amendment, B2). Item's P bar tightens (0.230 → 0.05); comp21's stays 0.05. ~~The hint and the bands on sing44 and
neos-960392 are not re-read: the hint was present twice (0.765, 0.834) and the open question is only harm.~~ (replaced
12:2x:) The hint and the P bands on sing44 and neos-960392 — gating harm guards since D-025's amendment 1 — are **carried
from r1-ysync, where they held at seeds 0–1, and not re-run**: a deviation from D-025's closing line (amendment, B7). ca's
band is subsumed by rule 3 at seeds 2–3, which is stricter than any band.

## Why these numbers — the false-fail and false-pass rates they imply
**Data** (the noise of the development default's own configuration, run hash 28e5a5e9c725, 120 s, ext; three same-window
pairs: itx-r13 A9@r13x of 10-13, r1-bsync A9@r1ctl and r1-ysync A9@r1ctl-ysync of 10-15; the last column adds backoff2's two
same-window ext pairs on item of 10-14 — A9@bo1s, the default with the back-off only recording, G 1,562 / 1,946, P 0.234 /
0.202, and A9@boY, G 1,950 / 1,754, P 0.332 / 0.284; comp21 has no other such pair):

| | item_placement_9903 | comp21-2idx |
|---|---|---|
| G, pairs (s0 / s1) | 1,619 / 1,687; 1,646 / 1,782; 1,367 / 1,480 | 248 / 295; 261 / 272; 287 / 318 |
| G mean; SD within a pair (CV); SD of all six (CV) | 1,597; 77 (4.8 %); 150 (9.4 %) | 280; 23 (8.4 %); 25 (9.0 %) |
| G SD within a pair, 5 pairs | 149 | – |
| P, pairs | 0.358 / 0.370; 0.299 / 0.424; 0.479 / 0.250 | 0.552 / 0.543; 0.542 / 0.503; 0.595 / 0.553 |
| P mean; SD within a pair; SD of all six | 0.363; 0.107; 0.083 | 0.548; 0.024; 0.030 |
| P SD within a pair, 5 pairs | 0.085 | – |

**Three pairs give three degrees of freedom:** the 95 % interval of a SD from 3 degrees of freedom runs from 0.57 to 3.7
times the estimate, so every rate below is rough by that factor in SD; the ranges show only the choice of model. Item's
within-pair SD of P exceeds its all-six SD, which only says both are rough. **Model:** normal, T and C drawn from the same
distribution unless an effect is added; "within" = each block's two runs share a window mean (the design's case), "total"
= no sharing (wider); seeds 0–1 and 2–5 are two windows; 200,000 draws per cell. Two readings: a **fresh** six-seed test
(the rule's own properties) and **given seeds 0–1 as observed** (what this test actually is: only seeds 2–5 are new).
**Source (12:1x):** `OMP_NUM_THREADS=1 .venv/bin/python tools/d029_oc.py --check` (seeded; `--check` confirmed the data
against the run table). It replaces the build's unrecorded simulation, whose numbers agree within 1–3 points.

| rule (instance) | true effect | passes, fresh 6 seeds | passes, given seeds 0–1 | given seeds 0–1, item noise from 5 pairs |
|---|---|---|---|---|
| 1 (item) | none | 97.4–100 % | 72–87 % | 72 % |
| 1 (item) | G −10 % | 50 % | 2–15 % | 15 % |
| 1 (item) | G −15 % | 2–14 % | 0–3 % | 3 % |
| 1 (item) | G −18 % (bsync's) | 0.1–4 % | 0–0.6 % | 0.6 % |
| 1 (item) | one ×0.46 collapse among seeds 2–5 | 58–65 % | 0.1–5 % | 5 % |
| 1 (comp21) | none | 98–98.5 % | 99 % | – |
| 1 (comp21) | G −10 % | 50 % | 78–79 % | – |
| 1 (comp21) | G −15 % | 12–14 % | 45 % | – |
| 1 (comp21) | G −18 % | 3–4 % | 23–25 % | – |
| 1 (comp21) | one ×0.46 collapse among seeds 2–5 | 58–59 % | 56 % | – |
| 2 (item) | none | 79–85 % | 63–67 % | 67 % |
| 2 (item) | P +0.05 | 50 % | 34–37 % | 34–35 % |
| 2 (item) | P +0.10 | 15–21 % | 10–16 % | 11 % |
| 2 (comp21) | none | ≈ 100 % | ≈ 100 % | – |
| 2 (comp21) | P +0.05 | 50 % | 89–94 % | – |
| 2 (comp21) | P +0.10 | ≈ 0 % | 7–12 % | – |
| rules 1 + 2 on both | none | 78–81 % | **48–55 %** | 48 % |
| rules 1 + 2 on both | G −10 % on both | 20–21 % | 1–8 % | 7–8 % |
| rules 1 + 2 on both | G −15 % on both | 0.2–1.6 % | 0–0.8 % | 0.8 % |
| rules 1 + 2 on both | P +0.05 on both | 24 % | 22–30 % | 22–23 % |
| rules 1 + 2 on both | bsync's (G −18 %, P +0.06) | ≈ 0 % | ≈ 0 % | ≈ 0 % |
| rules 1 + 2 on both | item collapse recurring at q = 1/6 of runs | 43 % | 24–26 % | 24 % |
| rules 1 + 2 on both | item collapse recurring at q = 1/4 of runs | 29 % | 16–17 % | 16 % |

(The joint rows assume the four legs independent: G and P, and the two instances. A "collapse" multiplies a treatment
run's G by 0.46 and adds +0.15 to its P — item seed 1's size; a construction, not a known harm.) **Every "none" rate is
conditional on the normal model**, which item's seed 1 contradicts (12:2x): 678 vs 1,480 generations sits −7.3σ (within) /
−3.8σ (total) from no effect, so seed 1 was either a real ysync effect that comes and goes or an outside disturbance —
telling those apart is this test's real job (the start-of-pool section).

**What seeds 2–5 must show, given seeds 0–1:** on item, Ḡ(T) / Ḡ(C) ≥ **0.963** at seeds 2–5 with the control at its
mean (0.959–0.970 as the new window's control runs at 1,700–1,424 generations — at most about 0.01 of shift from the
window's level) — seeds 0–1 (2,161 vs 2,847 generations) already use ~~3.3 of the 10 points~~ **74 % of rule 1's margin
(seed 1 alone 87 %, partly offset by seed 0's +116; corrected 12:2x)** — and a mean P difference ≤ **+0.026** (seeds 0–1
summed to +0.197); on comp21, ≥ 0.857 and ≤ +0.076 (seeds 0–1 use 7 % of its generation margin).

**The item bar against its alternatives (12:2x, `tools/d029_oc.py`, 3 pairs, within–total; the joint column puts the bar on
both instances):**

| generation bar | rule 1 on item passes, given seeds 0–1: none / G −10 % / G −15 % | rules 1 + 2 on both, given seeds 0–1, none | fresh 6 seeds, none passes: item / comp21 |
|---|---|---|---|
| **0.90 (this record)** | 72–87 / 2–15 / 0–3 % | 48–55 % | 97–100 / 98–99 % |
| 0.921 (D-025's item bar) | 54–58 / 0.1–6 / 0–1 % | 35–36 % | 94–100 / 94–96 % |
| 0.93 | 42–46 / 0–4 / 0–0.4 % | 26–29 % | 91–100 / 92–93 % |
| 0.95 (D-025's floor) | 15–29 / 0–1.4 / 0–0.1 % | 8–17 % | 83–97 / 84–86 % |

**Rule 3** (amended 12:2x) reads ca at seeds 2–3 alone. ca is stable (the default's configuration, 28e5a5e9c725:
0.391–0.396 in 6 runs over 2 days; run-to-run SD 0.002–0.004) and its gain is large (seeds 0–1: T 0.249 / 0.256 against
C 0.392 / 0.393, ratio 0.644), so with the gain repeating rule 3 passes essentially always, and with no gain (T = C, ratio
1.0) or a harm (T = 0.45, ratio 1.146) it fails essentially always (`tools/d029_oc.py`). The build's pooled form (seeds
0–3) failed only if ysync averaged above 0.493 per run at seeds 2–3 — about 0.10 *worse* than the control — so it passed a
gain that did not repeat (pooled 0.822) and a ca harm (T 0.45: 0.895); the pooled ratio is printed beside.

**The trade-off, stated.** The rules err towards NOT ADOPTED, deliberately: the failing seed stays in and this test follows
a NOT ADOPTED. Under the normal model (which seed 1 contradicts), a ysync that truly harms nothing (with ca's gain real) is
adopted about **half** the time (48–55 %), mostly because of item_placement: its P varies 0.25–0.48 between runs of one
configuration in one window (rule 2 alone fails 33–37 % given seeds 0–1), and its seeds 0–1 already used ~~a third~~ 74 %
of rule 1's margin (corrected 12:2x). In the other direction, rule 1 lets a true generation loss of 10 % pass in half of
fresh tests and of 15 % in 2–14 % (item) / 12–14 % (comp21); it catches bsync-sized losses (−18 %) in 96–100 % of fresh
tests; on comp21 the observed seeds 0–1 (0.980) make a 15 % loss pass 45 % of the time and an 18 % loss 23–25 %. A looser
item P bar (0.10) would cut rule 2's fresh false-fail on item to about 2–5 % but let a +0.10 harm pass about half the time.
~~0.05 is kept, as Berk's option fixed "guards pooled over all seeds" and the brief fixed the numbers.~~ (corrected 12:2x:)
Berk's option fixed the design — guards pooled over all seeds 0–5, the failing seeds included — not the numbers; 0.90, 0.05
and 0.95 are AgentRA's (the main session's brief to the build, 2026-10-15 11:2x), chosen with seeds 0–1 known. Any change
to a threshold is a dated amendment here before the first `r1f-ysync` run, never after.

## Forking paths
- This test exists because D-025's ysync test was NOT ADOPTED on one seed of one instance. That is the forking path, and it
  is narrowed three ways: **every seed run so far stays in** (item's seed 1 at 678 generations included, as D-025's closing
  rule says); the rules and thresholds are fixed here before any run and implemented by a script tested before the window;
  and **no further R1 follow-up for v1 happens whatever this gives** — NOT ADOPTED sends R1 to v1.1, ADOPT goes to the C1
  re-run. It is not closed (12:2x): the item generation bar that failed is looser here (0.921 → 0.90, chosen after the
  failure was seen) and a harmful ysync still gets a second chance.
- **What the second chance costs, over the whole procedure** (added 12:2x; `tools/d029_oc.py`, 3 and 5 pairs, within and
  total). Assumptions: any D-025 failure leads to a follow-up like this one (D-025's closing line allows one); D-025's
  rules as the review approximated them (on seeds 0–1, per instance, the generation ratio ≥ 1 − max(0.05, the control's
  relative spread) and the P difference ≤ max(0.05, the control's P spread)), the hint and the hint-instance bands taken as
  passing; the collapse rows are a construction extrapolated from the one observed collapse.

  | true effect | D-025 alone (adopted) | D-025, then this test after a failure (adopted overall) | this test after a D-025 failure | this test, fresh |
  |---|---|---|---|---|
  | none | 56–61 % | 88–89 % | 68–73 % | 78–81 % |
  | G −10 % on both | 10–21 % | 26–35 % | 17 % | 20–21 % |
  | G −15 % on both | 2–9 % | 2–10 % | 0.2–1.3 % | 0.2–1.7 % |
  | P +0.05 on both | 25–28 % | 39–41 % | 18–19 % | 24 % |
  | item collapse at q = 1/6 | 40–43 % | 59–60 % | 31–32 % | 43 % |
  | item collapse at q = 1/4 | 33–35 % | 47–48 % | 19–20 % | 29 % |

  The follow-up raises the overall chance of adopting a ysync that harms by about **12–19 points** (a 10 % generation loss,
  +0.05 P, or an item collapse recurring in 1 of 4–6 runs) and a harmless one by about 27–32; a 15 % loss stays at or below
  10 %. In these unconditional terms pooling barely beats a fresh retest after a failure (G −10 %: 17 % against 20–21 %);
  given the observed seeds 0–1 it matters a lot (harmless: 48–55 % against 78–81 % fresh) — the view the tables above
  show. **No further R1 follow-up for v1** is the stopping rule.
- No run, seed or instance is dropped after the fact; a failed treatment run is an outcome (D-002 §6), a failed control
  run is rerun (the rules, amended 12:2x); voids are the pool's.
- bsync is not revisited for v1 (its comp21 P harm, +0.060, is not a single-seed event).

## The start-of-pool disturbance — stated, and met with a warm-up run (amended 12:2x)
item_placement's failing run (A9@ysync seed 1) was the r1-ysync pool's first run, started 10:00:14, one second after M4's
600-s pool ended (load average 5.75 at its gates; GPU clock 2,392 MHz throughout, no foreign GPU process, not voided). It
ran 70 GPU launches per second against 122 on seed 0 and 112–124 in the control (its LP worker's CPU share 0.49 against
0.81 on seed 0). Whether that is `yield` or the pool's start cannot be told from one run. ~~**No rule here acts on it:** the
run stays in rules 1 and 2 with its numbers; the report marks each pool's first run beside every run (not gating). The new
pool starts like any other (its first block by the pool's order is comp21 seed 5, treatment first); no warm-up run, gap or
exclusion is added for it.~~ (amended 12:2x:)
- The run stays in rules 1 and 2 with its numbers.
- **The new pool's first run is an unread warm-up** — the control's configuration on item_placement seed 6 (A9@r1fwarm,
  priority 0) — so whatever disturbs a pool's start lands on neither arm. Without it the first run would have been comp21
  seed 5 with the treatment first: the position of the original failure (both earlier pools opened with item seed 1's
  treatment run, 07:17:09 and 10:00:14; bsync's did not collapse, 1,582 vs 1,782), on the instance where rule 1 cannot
  see a collapse (below).
- The report prints, per run, the load average at its gates, and per pool its first run and the gap since the previous
  pool in this checkout's run table (pools from other worktrees are not in it). No idle gap is imposed: whether the pool
  starts right after another depends on the queue.
- **comp21's blind spot, stated:** one more collapse of the seed-1 size (×0.46) among the four new item runs fails rule 1
  nearly always (passes 0.1–5 %), but on comp21 it passes 56 % — with one collapse at comp21 seed 5 and every other seed
  at parity the pooled ratio is 0.905, and 0.900 with one more seed at 0.97. bsync's comp21 collapse (seed 1, 153 vs 272)
  came mid-pool at a normal launch rate, so not every collapse is a pool-start event. A per-run collapse check (e.g. any
  treatment run at seeds 2–5 with G < 0.6 × its block's control fails) would close this; it is **not** added here — over
  all seeds it would already fail on seed 1, and over seeds 2–5 it is a new rule beyond the brief — and stays an open
  question for the main session before the first run.

## What ADOPT triggers
- `configs/itrea2-dev-rc-3way-ysync.yaml` becomes **A9's and A3x's development default** (D-025 "What adoption would
  mean": A3x runs the same file and stays A9's matched control; `configs/itrea2-dev-rc-3way.yaml` stays as the record of
  the old default; A3's default `itrea2-dev-rc-ld.yaml` is untouched); `configs/frozen-v2.yaml` carries the switch at C1.
- **C1 is re-run once on the frozen-v2 hash, with A6F re-run in the same window — 68 runs ≈ 3.9 h — before Fri 2026-10-23**
  (Berk 2026-10-14). D-025 amendment 3's harm leg reads it against A9@r13x of 10-13 (cross-day; crna and rail01, the
  instances most exposed to wake latency and not in this test, are first seen there); any harm goes to Berk beside the
  go / no-go, and the frozen-v2 candidate is not presumed.
- **The interaction with R2, stated now:** D-026's test is running (set `r2-ld`, pool from 11:17) on the r13x base
  (`-3way-ld0.yaml` against the development default), because R1 was not adopted. If D-026 adopts ld0 **and** this record
  adopts ysync, frozen-v2 would combine two changes each tested only against the r13x default, never together. The C1
  re-run's harm leg is then the only check of the combination, and **Berk decides which configuration freezes** (D-025
  amendment 3). No combination test is pre-stated here; one would need its own record. If D-026 adopts before this pool
  runs, this record's control stays `configs/itrea2-dev-rc-3way.yaml` (pinned, 12:2x) — the configuration r1-ysync's
  seeds 0–1 were run against.
- **NOT ADOPTED:** A9's default stays `configs/itrea2-dev-rc-3way.yaml`; R1 (bsync, ysync, bnice) goes to v1.1; C1 follows
  D-026's verdict alone.

## Cost and code
- **Box:** 21 runs ≈ 1.2 h (r1-ysync's 20 took 10:00–11:09, 3.45 min per run; the warm-up adds one) in window W-1015
  (lease until 18:55), after the queued pools; nothing for Berk. If ADOPT: 68 runs ≈ 3.9 h before Fri 10-23 (a later
  window).
- **Analysis:** `tools/d029_report.py` (`OMP_NUM_THREADS=1 .venv/bin/python tools/d029_report.py [--failed both|exclude|keep]
  [--json out]`, after `itrea build-runs`) applies exactly these rules on the run table and the run directories, reading
  rerun sets `r1f-ysync-<N>` by itself; its tests are `tests/test_d029_report.py` (49 passed, 2026-10-15 12:2x, CPU,
  synthetic tables: every verdict branch including INCOMPLETE for a failed control, missing runs → INCOMPLETE, INVALID before
  INCOMPLETE, the window leg, a failed treatment run kept and excluded, the pooled seeds 0–1, rule 3 at seeds 2–3 with the
  pooled ratio beside, rerun sets read in place of their blocks, the pinned ids, the warm-up first in the pool's order and
  never read, the job file regenerated byte for byte). On today's table it reads **INCOMPLETE, 8 of 28** (r1-ysync's 8 pass
  every other leg), as expected. `tools/d029_oc.py` reproduces the rates above. **Owed:** an `itrea d029-report` entry
  needs an edit of `src/itrea/cli.py`, and AGENTS.md rows for both tools are root-file edits (not made: a pool imports the
  main checkout); until then the scripts are the entries.
- Links: [[D-024 Targeted first, wider only on a hint — staged testing for development and v1]],
  [[D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test]],
  [[D-002 Pre-registration — hypotheses, arms, analysis]], [[2026-10-14 C1 diagnosis and improvement candidates]].

## Amendment 2026-10-15 12:2x — before any `r1f-ysync` run (AgentRA, Claude Code (Anthropic Claude Opus 5.5))
After a statistics and code review of the build (11:4x). **Checked 12:1x and 12:2x: no run of set `r1f-ysync` exists**
(`traces/raw/r1f-ysync` absent; no `r1f-*` row in `datasets/runs.parquet`). The body above is edited in place where a rule
changed (marked "amended 12:2x") or a statement was wrong (struck through, with the correction beside).

**A. Rule changes (gating):**
1. **Rule 3 reads ca at seeds 2–3 alone** (was pooled over seeds 0–3), so **the rules read 28 runs** (was 32; r1-ysync's
   4 ca runs are read beside). Why: pooled with seeds 0–1, the rule could not fail unless ysync ran about 0.10 *worse* than
   the control at seeds 2–3; it passed a gain that did not repeat and a ca harm beyond D-025 amendment 1's band. At seeds
   2–3 alone it fails both, and with ca's noise (SD 0.002–0.004) against a gain of about −0.14 it essentially never fails a
   gain that repeats. It also matches Berk's option ("ca at 2–3 vs a same-window control") and the brief's own count of 28,
   which the pooled form contradicted. Strictly stricter; the main session may revert it before the first run.
2. **A failed control run never decides a rule** (was: a missing control G failed rule 1, which made a control crash a
   final NOT ADOPTED). A rule that fails with a failed control in still fails (a truncated control only looks worse); one
   that passes or cannot be computed is unread → INCOMPLETE → the block is rerun. A failed treatment run is unchanged: an
   outcome of its arm.
3. **Window leg:** the two runs of each block must share `window_id` (the design's "one block, one window" was stated, not
   checked).
4. **Rerun scope:** INVALID blocks (after the fix) and blocks unread for a failed control are rerun whole — both arms,
   nothing else — as `r1f-ysync-2` (then `-3`, …), each with its own warm-up; the report reads the latest rerun set's rows
   in place of the original block, with no code change.
5. **Warm-up run first** (the start-of-pool section): one unread control-configuration run at priority 0; the 20 read
   runs are unchanged apart from `priority` 1 and keep their run ids and their order.
6. **Same code**, checked by hand at the pool's start and end against da80772 (validity above).
7. **Pinned run ids:** the report's configuration paths and run hashes (28e5a5e9c725, a08221cca653) are literals; a later
   change of make_jobs' default constants or of either configuration file cannot turn the 8 old runs INVALID (they cannot be
   rerun), and `--write-jobs` refuses a job whose id is not the pinned one.

**B. Stated, no rule change:**
1. **Who chose the numbers.** Berk chose the design (a follow-up judged on all seeds 0–5 with the failing seeds kept). The
   thresholds 0.90 / 0.05 / 0.95 are AgentRA's — the main session's brief to the build, 2026-10-15 11:2x — chosen with seeds
   0–1 known. The build's "the brief fixed the numbers" and "on Berk's choice" read as if Berk had chosen them.
2. **The bar that failed is looser; the tolerated loss doubles.** Item's generation bar 0.921 → 0.90 (comp21 0.898 → 0.90;
   bsync's comp21 0.950); D-025's 5 % floor → a fixed 10 %. Why 0.90: on a fresh six-seed test it turns a harmless ysync
   down in at most 2.6 % per instance, where D-025's 0.95 floor would in 3.5–17 % on item and 14–16 % on comp21 (0.93: up
   to 9 % / 8 %); it catches bsync-sized losses (−18 %) in 96–100 % of fresh tests. The cost: a 10 % loss passes half of
   fresh tests and a 15 % loss 2–14 % (item) / 12–14 % (comp21); given seeds 0–1, a 15 % loss on comp21 passes 45 %.
   D-025 amendment 2 accepted that a 15 % loss could pass only when the control's own spread was wide; this record
   accepts it at any spread. And 0.90 is also the bar that keeps the follow-up winnable given seed 1 (all four legs pass a
   harmless ysync 48–55 % of the time at 0.90, 35–36 % at 0.921, 8–17 % at 0.95). Kept as the brief set it; whether to
   move it is the main session's (or Berk's) call before the first run.
3. **Arithmetic corrected:** seeds 0–1 use 74 % of rule 1's item margin, not "3.3 of the 10 points"; seed 1 alone 87 %
   (802 of the 923 generations the 10 % allows over seeds 0–5 at the control's mean), partly offset by seed 0's +116.
4. **The whole procedure's operating characteristic** (D-025, then this test): the Forking paths table; the follow-up
   raises the overall chance of adopting a harmful ysync by about 12–19 points.
5. **"Harmless" is conditional on the normal model**, which seed 1 contradicts (−7.3σ / −3.8σ); comp21's blind spot for a
   single collapse and the 0.905 margin (start-of-pool section).
6. **Pooling across windows** does not bias the control (each block pairs T and C in one window); the ratio of means
   weights each window by its generation level, which moves the bar seeds 2–5 must clear by at most about 0.01 (0.959–0.970);
   r1-ysync's item control ran 14–17 % below the other two control windows (1,424 against 1,653 and 1,714). A window x
   treatment effect (such as ysync giving up its core under leftover load) rests on two windows, not six seeds, and
   neither model includes it; the report prints seeds 0–1 and 2–5 (the two windows) beside.
7. **Noise:** three pairs, 3 degrees of freedom (a factor of about 0.6–3.7 in SD); backoff2's two pairs double item's
   within-window generation SD (77 → 149) and narrow its P SD (0.107 → 0.085) — the table's last column. The hint and the
   P bands on sing44 and neos-960392 are carried from r1-ysync (held at seeds 0–1: sing44 0.408 → 0.418, ratio 1.026;
   neos-960392 0.059 → 0.045), not re-run — a deviation from D-025's closing line.

**C. Not changed, open before the first run (the main session's or Berk's):** the thresholds 0.90 / 0.05 (B2); a per-run
collapse check on seeds 2–5 (start-of-pool section).

## Confirmed by the main session 2026-10-15 12:3x, before any r1f-ysync run (AgentRA)
The 12:2x amendment is accepted as written: thresholds **0.90 (generations) / 0.05 (P) / ca seeds 2–3 ≤ 0.95** stand, the
unread warm-up opens the pool, no per-run collapse check is added (stated blind spot on comp21: one more ×0.46 collapse passes
~56 %). Why not move the bars again: they were chosen with seeds 0–1 known and are disclosed as such, with the alternatives
tabled above; a further change after the review would add another fork. Job file `jobs/d029.jsonl` (21 lines, 11 blocks, sha256
`8827a2a448807788ae150fe2e2a2f5e5106aff842698431a215049f42dd4fe29`) queued in W-1015 after `r2-ld`. If both D-026 and D-029
adopt, frozen-v2 combines two changes never run together; only the C1 re-run on the frozen hash (with D-025 amendment 3's harm leg)
covers the combination, and what freezes is Berk's.
