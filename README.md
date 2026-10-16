# I-TREA II

**A GPU evolutionary algorithm (EA) that runs beside exact MIP solvers and trades solutions with them while they run.**
The aim: good feasible solutions sooner than SCIP alone or NVIDIA cuOpt alone, on each solver's hardest instances.

## What this means for SCIP and cuOpt users

**Short answer: yes, there is a real chance of a large gain on hard instances, but not on every one.** On instances where a
solver alone struggles to find good solutions in two minutes, running it inside I-TREA II found good solutions much sooner
in our interim tests: about **half** the primal integral of SCIP alone and **a fifth less** than cuOpt alone on identical
hardware. Where cuOpt alone already finds a good solution within seconds, I-TREA II can be slower.

| If you use … | on its hard instances¹ | change in P(120)² with I-TREA II | largest gains | losses |
|---|---|---|---|---|
| **SCIP** (one CPU core, its default) | 15 of our 17 | **−50 %** on average; better on 12, worse on 0 | ca 1.00 → 0.26 (SCIP alone found no solution), neos-960392 1.00 → 0.04 (SCIP alone stayed at a trivial solution) | none |
| **cuOpt** (same 5 cores + GPU as I-TREA II) | 11 of our 17 | **−20 %** on average; better on 7, worse on 3 (p = 0.03) | neos-960392 0.58 → 0.04, item_placement 0.49 → 0.22, ca 0.56 → 0.26 | doon 0.86 → 1.00, rail01 0.98 → 1.00, toguru 0.55 → 0.61 |

¹ Hard = the solver alone has P(120) ≥ 0.5. ² P(120) = the average gap to the best known solution over a 2-minute run:
1 = no solution, 0 = optimal from the first second; lower is better. Four seeds per instance.

- **Why it helps:** I-TREA II runs SCIP, cuOpt and a GPU evolutionary algorithm (EA) side by side and passes each one's
  best solutions to the others. SCIP's hard cases gain from cuOpt's GPU heuristics and the EA's LP-based repair;
  cuOpt's hard cases gain from SCIP's incumbents and the EA's repair.
- **What the SCIP comparison measures:** adopting I-TREA II on a GPU machine — SCIP alone uses one core, I-TREA II five
  cores and the GPU. Much of that gain comes from having a GPU solver at all; the cuOpt row, on identical hardware, shows
  what I-TREA II adds beyond cuOpt.
- **When not to expect it:** if cuOpt alone already finds a good solution within seconds, the EA's GPU work slows cuOpt
  down (chromaticindex: 0.08 → 0.54, outside cuOpt's hard set). Where cuOpt alone finds a solution only late and on some
  seeds (doon, rail01), I-TREA II missed it. Where no solver finds one (cryptanalysis), neither does I-TREA II.
- **For the cuOpt and SCIP developers:** partner solutions and LP-based repair lift each solver's weakest cases. cuOpt
  would gain more if it could take outside solutions mid-solve (its `SetSolutionCallback`, fixed for presolve in 26.10 on
  our report) and if two processes could share the GPU without slowing cuOpt (CUDA MPS, planned).
- **How sure:** interim development results — 17 instances that we tuned on, 2-minute runs, one DGX Spark (GB10), cuOpt
  26.08; SCIP alone ran on earlier days; cuOpt alone crashed on comp21 on 3 of 4 seeds (counted as its result). The test
  on unseen instances, with cuOpt 26.10, follows in November ([Roadmap]).

## Summary

- **Stage:** phase 1 (development, on cuOpt 26.08) closed on Fri 2026-10-16 with the author's (Berk Orbay's) go at
  checkpoint C1 ([D-033]). No confirmatory run yet.
- **C1 result (interim, four seeds):** A9 (SCIP + GPU EA + cuOpt) against cuOpt alone on the same hardware share: hint
  **0.798** (needs ≤ 0.95), guard **+0.045** (needs ≥ −0.05); 136 runs, all valid ([D-031]).
- **Reported beside, not criteria:** mean P(120) (lower is better) 0.690 → 0.550 (−20 %) against cuOpt alone on its 11
  weak instances (7 better / 3 worse). Against SCIP alone (earlier days, not hardware-matched): −38 % on the same 11
  (8 / 0); −50 % on SCIP's 15 weak (12 / 0), a set selected on the compared runs.
- **Frozen:** `configs/frozen-v2.yaml` — A9 with LP decode and `yield` (below).
- **Caveats:** 17 development and class-probe instances, chosen and tuned on; descriptive only. Two harm flags (glass4,
  comp21-2idx) and cuOpt alone's crashes on comp21-2idx are carried into phase 2 (footnote 1).
- **Paper:** [the interim report on phase 1](paper/main.pdf) (PDF, in progress).
- **Next:** v1 by Mon 2026-11-30 (aim Fri 11-27): unseen instances, cuOpt 26.10, ITX-1 ([Roadmap]).

## Terms

| term | meaning |
|---|---|
| P(T) | primal integral up to T s, in [0, 1] (1 = no solution); lower is better ([metrics]) |
| d | bounded per-instance difference of seed means, in [−1, 1]; positive = the coupled arm is better |
| F | A9's footprint: 5 CPU cores and the GPU of one GB10 |
| A0 · A6F · A9 | SCIP alone · cuOpt alone on F · SCIP ↔ EA ← cuOpt ([D-015]) |
| the 17 | 9 development + 8 class-probe instances ([D-010], [D-014]) |
| weak set | instances where a solver alone has P ≥ 0.5, on seeds not used for the evaluation ([D-013]); "fixed 11" = cuOpt's, among the 17, chosen on earlier, separate cuOpt-alone runs |
| C1 | phase 1's closing checkpoint ([D-019]) |
| hint | ratio of mean P(120), A9 / A6F, over the fixed 11; present iff ≤ 0.95. Futility-only: it gates widening and declares nothing ([D-022], [D-024]) |
| guard | mean d over the 17 ≥ −0.05 ([D-022]) |
| v1 | A9 frozen, with two results: a pre-registered confirmatory test on unseen hard instances and an external run under the MIPfeas protocol ([D-019]) |
| ITX-1 | our external benchmark: every GPU arm on F, 600 s, MIPfeas instances ([D-021]) |

## Configuration and plan (2026-10-16)

| | |
|---|---|
| **Frozen** | `configs/frozen-v2.yaml` = a byte copy of `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA `835c7c5a249c`, A9 run `f345de0ab936`), written after the C1 go ([D-033]) |
| **Adds** | to the old default `configs/itrea2-dev-rc-3way.yaml`: LP decode after the EA's LP worker ([D-026]); CUDA context at `yield` ([D-029]) |
| **Before prereg-v2** | one checker rule for every arm (#36), the test-set size (#32), the prereg-v2 tag ([D-033]) |
| **Phase 2** | #26 (recombine a partner incumbent with the elite), #35 (two-way A9 through cuOpt 26.10's SET callback), CUDA MPS ([Roadmap]) |

## Results so far (interim, development data)

**C1 readings:** A9 against cuOpt alone (A6F), both arms of each instance and seed run in the same session; 120 s,
the 17 instances.

| reading | result | source |
|---|---|---|
| C1, old default (10-13), seeds 0–1 | hint 0.827 · guard +0.063 | [D-022] |
| C1, frozen configuration, seeds 0–1 | hint 0.809 · guard +0.050 | [D-030] |
| **C1, frozen configuration, seeds 0–3: the phase-1 interim result** | hint **0.798** (needs ≤ 0.95) · guard **+0.045** (needs ≥ −0.05); seeds 2–3 alone 0.787 · +0.034 ¹ | [D-031] |
| vs cuOpt alone, fixed 11, seeds 0–3 | 7 lower / 3 higher; P 0.690 → 0.550 (−20 %); d +0.249 [+0.078, +0.441]; Wilcoxon p 0.027 | [D-031] |
| vs SCIP alone, fixed 11 ² | 8 lower / 0 higher; −38 % | [D-031] |
| vs SCIP alone, its 15 weak ² | 12 lower / 0 higher; P 0.889 → 0.448 (−50 %); d +0.499 [+0.323, +0.676]; p 0.0005 | [D-031] |

1. Harm flags beside, not criteria: glass4 (d_i −0.114; read in one window as day-to-day drift, [D-032]) and
   comp21-2idx (d_i −0.051 against its band 0.050). cuOpt alone crashed on comp21-2idx on 3 of 4 seeds; those runs are
   kept as outcomes, which favours A9 there; without them the reading is INCOMPLETE.
2. SCIP alone ran on earlier days on SCIP's own harness, not in the same window, so this comparison is not
   hardware-matched (A9 also runs cuOpt and the GPU), and on seeds 0–1 only. Contrary to D-013's rule (see Terms),
   SCIP's weak set was selected on the runs it is compared with; the fixed 11 were not. No guard against SCIP alone is
   computed.

**Adopted changes** (not C1 readings: each tested alone against the old default, on its own instances and seeds):

| change | result | source |
|---|---|---|
| LP decode in A9 | hint 0.758 (needs ≤ 0.90); P item_placement 0.330 → 0.198, sing44 0.473 → 0.411 (seeds 0–3) | [D-026] |
| `yield` | ca__Scenario400_reserves_1 at seeds 2–3 alone: ΣP ratio 0.675 (needs ≤ 0.95); no generation or P harm on item_placement, comp21 (seeds 0–5) ³ | [D-029] |

3. Bars set with seeds 0–1 known; D-025's original bars give the same verdict.

Titles of D-029 and D-030 state the pre-run framing: D-029 was adopted; D-030 ran ysync-ld0 (D-030 § Which candidate
runs). Known loss: chromaticindex1024-7 to cuOpt alone ([D-027]). Every number and caveat: [Phase 1 note].

### The C1 decision
**Go** (Berk Orbay, 2026-10-16): freeze yield + LP decode as `configs/frozen-v2.yaml` and close phase 1 on these
numbers. The two harm flags, cuOpt alone's crashes on comp21-2idx and the loss on chromaticindex1024-7 are carried
into phase 2 and v1 as stated caveats ([D-033]).

## How it works

- **In A9, three processes share one GB10**, each pinned to its own cores: SCIP (version 10), the GPU EA (CuPy),
  cuOpt 26.08.
- **Three ranks:** Rank 0 = infeasible solutions better than the best feasible one (at most x %); Rank 1 = the best
  y % of feasible ones, protected; Rank 2 = the rest ([algorithm]).
- **Exchange** through a mailbox (`/dev/shm` files or Redis). SCIP sends its incumbent, pool, bound, root LP and good
  infeasible node-LP points ([D-011]); the EA returns what beats SCIP's incumbent. cuOpt only sends (on 26.08,
  receiving disables its presolve).
- **`lp_decode`:** HiGHS re-optimises a candidate's continuous variables with its integers fixed ([D-017]).
- **`yield`:** while waiting on the GPU, the EA's host thread yields to other work on its core, so the LP worker there
  gets CPU time ([D-029]).
- **Checked:** every incumbent is re-checked in exact arithmetic before it counts ([process]).

## Quick start

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,plots,gpu]"                                   # CPU + GPU backends, HiGHS, SCIP, tests
uv pip install -e ".[cuopt]" --extra-index-url https://pypi.nvidia.com # cuOpt 26.08 (arms A2, A6, A9, H2)
pytest -q                                                              # toy-scale; GPU tests skip without CuPy

# Toy run: SCIP alone, 10 s, no GPU, no lease
python -c "from itrea.benchmarks.toy import knapsack, write_mps; write_mps(knapsack(), '/tmp/toy.mps')"
itrea pair --instance /tmp/toy.mps --arm A0 --seed 0 --budget 10 --set smoke

# Timed runs: only inside a box window (skills itrea-window, itrea-new-experiment)
itrea make-jobs c1r-rerun --a9 ysync-ld0 --set c1r-ysync-ld0-2      # a rerun gets its own set
itrea pool jobs/c1r-ysync-ld0-2.jsonl --layout pair --deadline HH:MM
itrea build-runs                                                     # traces -> datasets/*.parquet
itrea c1r-report --a9 ysync-ld0 --set c1r-ysync-ld0-2                # C1's hint, guard, D-013 terms
```

`itrea --help` lists every command. Shared-box rules: [AGENTS.md], [D-003]. Install detail: [Background] § 9.

## Reproducibility

- **Pre-registered:** [D-002] v3.1, tag `prereg-v1`; `prereg-v2` precedes any confirmatory run.
- **Traces** are not committed: unpack the release archive (planned, with checksums) or re-run the job files on the
  public instances, then `itrea build-runs`, `itrea prereg`, `itrea reproduce`.
- **Transfer:** wall clock on one DGX Spark; GPU arithmetic is not bit-reproducible. Expect the conclusions, not the
  numbers, to transfer ([Background] § 12).

## Repository map

| path | what |
|---|---|
| `src/itrea/` | the package: `ea/`, `solvers/`, `exchange/`, `experiments/` (runner, pool, job builder), `analysis/` (checker, statistics) |
| `configs/` | EA configurations, hashes pinned |
| `jobs/` · `traces/` · `datasets/` | job files (one line per run) · run outputs · derived tables |
| `tools/` | decision reports, `serving-window.sh`, probes |
| `tests/` | toy-scale tests |
| `data/` | instances (not in git) |
| `paper/` | the interim report on phase 1 (LaTeX, in progress; built as `paper/main.pdf`) |
| `vault/` | research record: journal, decisions, results, literature |
| `.claude/skills/` | skills for AI agents |

## Documentation

- [AGENTS.md] — rules for AI agents and the code map. **Agents read it first.**
- [Home] — the vault's start page.
- [Paper] — the interim report on phase 1 (PDF, in progress).
- [Phase 1 note] — phase 1's results, every number and caveat.
- [Decisions] — D-001 … D-033.
- [Roadmap] — what is next.
- [Background] — lineage (I-TREA part I, optopt), claim, pre-registration, v1 targets, status detail, layout, safety.

## Credits and AI statement

Author: Berk Orbay (berk.orbay@tideseed.com). Developed under his direction with AgentRA (Agent Research Assistant),
which designs with him and executes the experiments; AgentRA is mainly Claude Code (Anthropic Claude Opus 5.5). Code
adapted from the predecessor carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## Licence

[MIT](LICENSE). Third-party files keep their own terms.

[AGENTS.md]: AGENTS.md
[Home]: vault/00-index/Home.md
[Roadmap]: vault/00-index/Roadmap.md
[Background]: <vault/00-index/Project background.md>
[Phase 1 note]: <vault/30-results/2026-10-17 Phase 1 — development until C1.md>
[Paper]: paper/main.pdf
[Decisions]: vault/50-decisions/
[algorithm]: <vault/10-design/I-TREA II algorithm.md>
[process]: <vault/10-design/Process interface.md>
[metrics]: <vault/10-design/Trace format and metrics.md>
[D-002]: <vault/50-decisions/D-002 Pre-registration — hypotheses, arms, analysis.md>
[D-003]: <vault/50-decisions/D-003 Resources, memory and thermal rules.md>
[D-010]: <vault/50-decisions/D-010 Develop the algorithm first on a small development set.md>
[D-011]: <vault/50-decisions/D-011 Solver streams good infeasible solutions to the EA.md>
[D-013]: <vault/50-decisions/D-013 Aim — lift each solver's weakest cases.md>
[D-014]: <vault/50-decisions/D-014 Class probe — predictions fixed before the runs.md>
[D-015]: <vault/50-decisions/D-015 Use cuOpt's feasible solutions — three-way arm A9.md>
[D-017]: <vault/50-decisions/D-017 LP decode — re-optimise the continuous variables of selected candidates.md>
[D-019]: <vault/50-decisions/D-019 v1 — deadline and performance target.md>
[D-021]: <vault/50-decisions/D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope.md>
[D-022]: <vault/50-decisions/D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test.md>
[D-024]: <vault/50-decisions/D-024 Targeted first, wider only on a hint — staged testing for development and v1.md>
[D-026]: <vault/50-decisions/D-026 lp_decode inside A9 on the EA's core after the LP worker (R2), development test.md>
[D-027]: <vault/50-decisions/D-027 Same-day diagnostic of cuOpt's thread count and SCIP's presence (R3) — reading rules fixed before the runs.md>
[D-029]: <vault/50-decisions/D-029 ysync gets one follow-up at seeds 2–5, judged on all seeds 0–5 with the failing seed kept — NOT ADOPTED ends R1 for v1.md>
[D-030]: <vault/50-decisions/D-030 C1 is re-read once on the frozen-v2 candidate (ld0) with A6F in the same window — 68 runs under D-022's hint and guard and D-025's harm leg.md>
[D-031]: <vault/50-decisions/D-031 The phase-1 interim result rests on four seeds — C1 runs again at seeds 2–3 on the frozen-v2 candidate and is read with D-030's seeds 0–1.md>
[D-032]: <vault/50-decisions/D-032 glass4's C1 harm flag is read in one window — old default vs yield + lp_decode vs lp_decode, 4 seeds.md>
[D-033]: <vault/50-decisions/D-033 C1 go — phase 1 closes with configs-frozen-v2.yaml = yield + lp_decode.md>
