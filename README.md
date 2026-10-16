# I-TREA II

**A GPU evolutionary algorithm coupled with exact MIP solvers.** I-TREA II (Interactive Three Ranks Evolutionary
Algorithm, part II) runs a GPU-resident evolutionary algorithm (EA) beside a mixed-integer programming solver and lets
them exchange solutions while they run. **SCIP** is the primary partner, in both directions: its incumbent, very good
pool solutions, proven bound, root LP and a stream of good infeasible node-LP points flow into the population, and the
population's better incumbents flow into SCIP as new starting points. **HiGHS** is a replication partner. **NVIDIA
cuOpt** is a control (cuOpt alone, SCIP ‖ cuOpt) and, since 2026-10-07, a one-way source: in the three-way arm **A9
(SCIP ↔ EA ← cuOpt)** cuOpt streams its incumbents into the EA, which hands SCIP whatever beats SCIP's incumbent —
cuOpt's vectors unchanged as well as the EA's own improvements. cuOpt never receives solutions mid-solve: on cuOpt
26.08 its `SetSolutionCallback` disables its presolve. cuOpt 26.10 keeps presolve on (NVIDIA/cuopt#2073, fixed by PR
#2098; checked on a 26.10 nightly on 2026-10-13), so a two-way A9 becomes possible there (issue #35, not started). The
population is organised in three ranks: Rank 0 holds infeasible solutions whose objective beats the best feasible one
(at most x % of the population), Rank 1 protects the best y % of the feasible solutions, and Rank 2 holds the rest.
Beside the ranks the EA keeps a frontier of diverse near-optimal feasible solutions. The project succeeds two pieces
of the author's work: his master's research I-TREA (Orbay 2011, M.S. thesis, Middle East Technical University, open
access at hdl:11511/20509; thesis) — an interactive three-rank
evolutionary algorithm for a bi-objective facility relocation problem — whose idea of keeping good infeasible
solutions it carries over (part I reserved a share of the population for them as a floor; part II caps it), and the
optopt study ([`berkorbay/optopt`](https://github.com/berkorbay/optopt)), whose CPU evolutionary partner beside SCIP
found no gain at 60 s and whose verdict named the changes tested here: a GPU population, longer budgets and instances
where the solver's own heuristics struggle.

**The claim under test** (the author, 2026-10-06, vault `D-007`): *a GPU-based evolutionary algorithm metaheuristic
coupled with an exact solver can beat either single solution.* Pre-registered as vault `D-002` v3.1 (binding from the
git tag `prereg-v1`): on hard MIP instances, does the coupled system find good feasible solutions sooner — a lower
primal integral over 300 s — than **the solver alone and the EA alone**, and does the exchange add anything beyond
running the two side by side? **Refined by the author on 2026-10-07** (vault `D-013`): the coupling is meant to lift
each solver's weakest cases, not to be better in general — the primary measure is the improvement on each solver's
**weak set** (the instances where that solver alone has P ≥ 0.5, chosen on seeds not used for the evaluation), with the
mean over all instances as a non-inferiority guard (mean d ≥ −0.05). The refinement and the v1 targets below enter a
revised pre-registration (`prereg-v2`) before any confirmatory run.

**v1** (vault `D-019`, decided by the author 2026-10-10): arm A9 frozen — SCIP 10 ↔ GPU EA ← cuOpt on one GB10.
Checkpoint **C1 on Sat 2026-10-17** (moved from Fri 10-23 on 2026-10-15, `D-019` addendum: phase 1 ends there;
development closes on cuOpt 26.08; MIPfeas harness calibrated; an interim A9 result on the 17 development and
class-probe instances; go / no-go for the final phase). **v1 at the end of November 2026**
(aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10; if 26.10 is not released by 11-06, the v1 runs proceed on 26.08 and
26.10 follows as an addendum (to be confirmed with the author then). Daytime windows only. Targets:
- **Confirmatory, unseen hard instances** (about 32 × 2 seeds × {SCIP alone, cuOpt alone, A9}, T = 300 s): on SCIP's
  weak set at minimum a significant gain (mean d > 0 with its 95 % interval above 0), target mean P −40 % with ≥ 70 %
  of the weak set improved (stretch −55 %); on cuOpt's weak set at minimum a significant gain, target mean P −25 % with
  ≥ 60 % improved (stretch −40 %); on all instances mean d ≥ −0.05 against each solver alone.
- **Mittelmann's MIPfeas protocol** (233 feasible MIPLIB 2017 benchmark instances, 600 s, primal integral 0–2, shifted
  geometric mean with shift 0.001): at minimum A9's geometric mean ≤ our own cuOpt-26.10-alone run on the same GB10;
  target mean integral −25 % on the instances hard for cuOpt (our cuOpt-alone integral ≥ 0.1 or no solution) and
  geometric mean −10 %; stretch ≤ 0.0274 (SCIP concurrent 11, published). The reference is our own same-version run
  because the published rows are cuOpt 26.08 and aggregates only. **Since 2026-10-11 the external run follows our own
  standard ITX-1** (vault `D-021`, accepted by the author): every GPU arm gets A9's footprint F (cuOpt alone on the same
  5 cores and the GPU; no whole-box run), 600 s, decisions on the 215 instances outside the development and
  class-probe sets that fit F (each decision also reported without the 31 instances seen in calibration or the memory
  probe), an ε-floored guard plus solutions found
  as the minimum, the shift-0.001 geometric mean reported beside; D-019's literal readings are reported beside too.
  **Staged (vault `D-024`, 2026-10-13):** a first stage on a 20-instance panel plus the 5 instances where cuOpt finds
  no solution, with a futility-only hint fixed in advance (ratio of means ≤ 0.95); the rest of the decisive set only
  after a hint; the guard stage runs either way.

None of the parts is new, and no "first" is claimed (`vault/40-literature/Prior art.md`): heuristics and
branch-and-bound exchanged incumbents both ways in Puchinger, Raidl & Gruber (2005) and Nwana et al. (2005); ReXi runs
local-search workers beside SCIP over a shared pool; CHAP pools GPU and CPU heuristics' feasible and infeasible
candidates; cuOpt itself runs a GPU population beside its own branch-and-bound; IDEA reserved a share of the population
for infeasible solutions. What is studied here is the combination: a GPU population coupled with an external, general
MIP solver, with a capped, incumbent-relative share of infeasible solutions.

## Status (2026-10-16)
**Development (phase 1, until checkpoint C1). No confirmatory experiment has run;** every number below comes from
development runs (small samples: two seeds or fewer per instance unless stated, at most six) and is descriptive, not a
performance claim. Phase 1 wraps up on Sat 2026-10-17 (Berk, 2026-10-15: "Assume we are wrapping up phase 1 on
Saturday"; `D-019` addendum).
- **Pre-registration and screening.** `D-002` v3.1 is tagged `prereg-v1` (2026-10-06). Hard-pool screening: stage 1
  complete (324 candidates, SCIP alone, 300 s, seed 900); stage 2 (seed 901) 35 of 234 runs done, deferred until the
  pool freeze (`D-010` §5, issue #1).
- **Development** since 2026-10-06 on a 9-instance set (`D-010`, `datasets/dev_set.csv`, permanently excluded from the
  test set) and 8 class-probe instances (`D-014`, `datasets/classprobe_set.csv`), judged on the coupled arm at 120 s
  (`D-011`). Every change, result and failure: `vault/30-results/Development log.md`. Since 2026-10-13 testing is
  staged (`D-024`): performance runs first, targeted at the instances where a change can act, wider only after a hint
  whose threshold is fixed before the runs.
- **Development defaults.** A3 (SCIP ↔ EA): `configs/itrea2-dev-rc-ld.yaml` (hash `f030e23909ad`) — self-adaptive repair
  strategy, the EA's own HiGHS LP, SCIP's LP stream, deep repair under the repair clock, LP decode. A9 and its matched
  control A3x (A3 with A9's EA footprint): since 2026-10-15 `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (EA hash
  `835c7c5a249c`, A9 run hash under the ext harness `f345de0ab936`) = the old default `configs/itrea2-dev-rc-3way.yaml`
  (`1d12ad490dd0`) + LP decode on the EA's core, started after the EA's own LP worker (`D-026`, #30) + the
  EA's CUDA context at `yield` (`D-029`). By `D-032`'s rule it is the configuration that freezes for phase 1;
  `configs/frozen-v2.yaml` is written on Saturday. `make_jobs.ITX_A9_CONFIG` still names `-3way.yaml` until a `src/`
  change after the running pool, so a writer that uses it writes the old default.
- **Development numbers** (P = primal integral, lower is better; d = bounded per-instance difference, positive = the
  coupled arm lower; seed means):
  - A3@lps1 (the 2026-10-06 baseline) against SCIP alone: lower P on 7 of 8 instances, mean d +0.43 [+0.23, +0.65],
    Wilcoxon p = 0.016; against cuOpt alone 6 lower / 3 higher, +0.11 [−0.33, +0.50], inconclusive
    (`vault/30-results/Quantification 2026-10-06 (dev set).md`).
  - Out of sample (class probe, 8 new instances, A3@lp1, no cuOpt; `D-014` inconclusive): against SCIP alone
    3 lower / 1 higher of 4 informative instances, mean d +0.14 [−0.00, +0.28], lower on only 2 of SCIP's 8 weak
    instances; against cuOpt alone 1 lower / 5 higher, mean d −0.51 [−0.84, −0.12]. The development set overstates
    the effect.
  - A9 pilot (7 instances, seed 0): against SCIP alone lower on 7 of 7, mean d +0.68 [+0.42, +0.90]; against cuOpt
    alone 3 lower / 4 higher, −0.07 [−0.42, +0.25]; cuOpt runs slower beside the EA (chromaticindex1024-7: cuOpt's
    P inside A9 0.545 vs 0.071 alone).
  - LP decode (`D-017`, 6 instances × 2 seeds): mean d +0.219 [+0.041, +0.429] against the previous default
    (item_placement 0.311 → 0.118, sing44 0.377 → 0.238) — adopted for A3.
  - Rank 1 by cuOpt's clearing rule (`D-016`, 9 instances, seeds 0–3, rail01 seeds 0–1 only; 102 runs): mean d
    −0.033 [−0.122, +0.050] — kept as an option (`ranks.rank1_mode`, off by default), not adopted.
  - A9 GPU back-off (`D-018`): stage 0 (18 runs) gave PROCEED; **stage 1 (84 runs, 2026-10-11) NOT ADOPTED** — every
    outcome rule held (chromaticindex1024-7: A9's P 0.545 → 0.091, cuOpt alone 0.083; back-off vs its shadow control
    9 better / 5 worse, mean d +0.083 [−0.054, +0.232]) but the mechanism failed (M3: the duty ladder stepped the EA down
    while it still led; M1: phase Y ended 0.56 / 0.80 s after cuOpt's first incumbent on toguru, limit 0.5 s; M5:
    more local-search cap hits per call on ca and item_placement). Its phase Y alone (yield until cuOpt's first
    incumbent, then full duty; `D-023`, 66 runs, 2026-10-14) was **NOT ADOPTED** either (failed M2, M5, T, G:
    chromaticindex1024-7 d +0.030 where +0.30 was needed; cuOpt's first incumbent inside A9 at 63.47 / 63.41 s vs
    10.87 / 10.52 s alone on F); the back-off line is closed for v1.
- **External harness** (`D-020`, `D-021`). GAMS's published MIPfeas archive is on disk and its primal integral is
  ported exactly (equal to GAMS's own function on all 3,495 of its traces). ITX-1's footprint F passed its thermal soak
  (steady p95 66–73 °C, maximum 76 °C over four 600-s runs); three instances do not fit F's memory (eilA101-2, nw04,
  supportcase6) and are excluded in advance. The calibration of cuOpt alone on F against GAMS's published GB10 row (M4)
  is complete, 30 of 30 (2026-10-15), and reads **OUTSIDE** the band [0.870, 1.150]: ratio of shifted geometric means
  1.313 [1.091, 1.618] on the call clock (1.963 [1.606, 2.408] on the launch clock), feasible 27 vs GAMS 27, late-clock
  check passed (median −0.119 s over strata 0–1). With the clock checked, the difference is the footprint (cuOpt on F's
  5 threads and the GPU against GAMS's 24-thread whole box) and the machine, so v1's external targets stay anchored on
  our own A6F run (`vault/30-results/2026-10-11 D-021 measurements M1–M3 and the interim calibration.md`, addendum
  2026-10-15; its interim 19-of-30 reading is superseded). A0 (SCIP alone, 600 s) on the 81 instances v1's decisions
  read: 80 completed + 1 failed (s100: SCIP did not stop at its limit and was killed at 701 s; kept as an A0 failure).
- **Phase 1's last week (2026-10-13 → 10-16).** Each verdict by its record's rules, fixed before any of its runs was
  read (`D-022`'s second amendment came with 7 of the 58 runs of its first pool done, none read; `D-029`'s, below):
  - Adopted: `D-022` (74 runs) — A9 under the ext harness (cuOpt's MIP gaps 0, thread variables unset; R13 + R14):
    valid, no harm on the 3 instances where cuOpt's own 1e-4 stop occurs; its mechanism was met only vacuously (no A9
    run in either arm ended on cuOpt's "optimal": SCIP proves first), so the adoption rests on D-021's fairness and
    GAMS fidelity. `D-026` (28 runs) — LP decode inside A9 after the LP worker: hint 0.758 (threshold 0.90;
    item_placement P 0.330 → 0.198, sing44 0.473 → 0.411, lower on all four seeds of both), every mechanism and
    no-harm leg held. `D-029` (21 runs) — the EA's CUDA context at `yield`, its guards read over seeds 0–5 with the
    failing seed of D-025 kept: generations ratio item_placement 1.020, comp21 1.152 (≥ 0.90); P difference +0.016 /
    +0.015 (≤ +0.05); ca's ratio at seeds 2–3 0.675 (≤ 0.95). These thresholds were set with seeds 0–1 (8 of its 28
    read runs) already known, and item_placement's generation bar was loosened from D-025's 0.921 to 0.90 (disclosed in
    D-029's amendment B1–B2). At D-025's bars the verdict is the same (item_placement 1.020 > 0.921; comp21 1.152 >
    D-025's 0.898 for yield, 0.950 for blocking sync), so this qualifies the claim, not the result.
  - Not adopted: `D-023` (above); `D-025` (R1, 20 runs each) — blocking sync (hint 0.765; the generation guard failed
    on item_placement and comp21, comp21's P +0.060 beyond its band 0.050) and its fallback `yield` at seeds 0–1
    (hint 0.834; item_placement's generation guard failed on one seed, 678 vs 1,480 generations), then reopened by
    `D-029`.
  - Diagnostics: `D-027` (22 runs) — cuOpt's first incumbent on chromaticindex1024-7 comes at 9.80 / 7.33 s alone on
    F, 9.24 / 8.21 s on a 3-thread lane and 6.53 / 9.65 s beside SCIP, against 62–66 s inside A9: the EA's presence
    on the GPU, not cuOpt's thread count or SCIP (rule 4, EA PRESENCE); the layout question is closed for v1. `D-028`
    (4 runs) — not D-022's harness: A9 with phase Y under `r13-off` still 63.25 / 63.24 s against L = 19.82 s.
    chromaticindex1024-7 stays a known v1 loss to cuOpt alone; CUDA MPS is not for v1 (Berk, 2026-10-15).
  - **C1 readings** (A9 vs cuOpt alone on F, A6F, in the same window; seeds 0–1; hint = ratio of mean P(120) A9 /
    A6F over the fixed 11 weak development instances ≤ 0.95, futility-only; guard = mean bounded d over all 17 ≥
    −0.05): on the old default (2026-10-13, `D-022` Result) hint present 0.827, guard +0.063; **on the frozen-v2
    candidate (`D-030`, 68 runs, 2026-10-15/16) hint present 0.809, guard +0.050** — read with cuOpt alone's crashes
    on comp21-2idx (SIGSEGV on both seeds) kept as A6F outcomes, from which this reading gains (A6F P 1 on both
    seeds); with those two runs left out, the hint and guard are INCOMPLETE. Reported beside, not criteria: on the
    fixed 11 against cuOpt alone 7 lower / 3 higher, P 0.677 → 0.548 (−19 %), d +0.235 [+0.052, +0.433], Wilcoxon p
    0.049; against SCIP alone (A0 from earlier days) on SCIP's weak set among the 17 (15 instances) 12 lower / 0
    higher, P 0.889 → 0.447 (−50 %), d +0.502 [+0.329, +0.675], p 0.0005. D-030's harm leg (cross-day, against the
    old default's 10-13 runs) flagged glass4 (d_i −0.088 against −0.057) and, at floor level, neos-3004026-krka;
    `D-032` re-read glass4 in one window (12 runs, seeds 0–3): mean P old default 0.2176, candidate 0.2233, LP decode
    alone 0.2177, d −0.025 ≥ −0.057 → DRIFT, so the candidate freezes for phase 1 by D-032's rule (fixed before its
    runs); D-031's four-seed harm reading may still put which configuration freezes to Berk. The four-seed reading
    (`D-031`) is running: section below. The go / no-go on C1 is Berk's.
- **cuOpt 26.10.** NVIDIA/cuopt#2073 (filed 2026-10-07) is fixed in 26.10 by PR #2098 and checked on a 26.10 nightly in a
  separate environment: presolve stays on with a `SetSolutionCallback`, injected solutions are accepted exactly,
  invalid ones rejected; cuOpt reads injected solutions only when its GPU population records a new best. Open: our
  exact checker accepted incumbents below MIPLIB's proven optimum (glass4, from EA runs — cuOpt rejects that point
  when offered; mas74, a cuOpt incumbent) and rejected cuOpt incumbents on snowy (1) and neos-933966 (3) — a tolerance
  mismatch under investigation (#36).
- **Tests:** non-GPU suite 1,256 passed, 1 xfailed on main after the R1–R3 merges (2026-10-14); 1,339 passed on branch
  `c1-rerun` before its merge (2026-10-15; 256 affected tests passed on main after it). GPU checks before the 10-15
  runs: the GPU suites 45 passed, 1 skipped; `test_cuda_sched.py` 5 and `test_lp_decode_gate.py` 4 passed; the
  candidate's A9-layout cold-cache test passed at 15:07.
- **Next:** Saturday 2026-10-17 (C1, end of phase 1): `configs/frozen-v2.yaml`, the phase-1 results note, a
  checkpoint, docs and issues, the go / no-go to Berk with the numbers (`D-019` addendum). Carried to phase 2: #26
  (recombine each partner incumbent with the elite, not built), #35 (A9 two-way through cuOpt 26.10's SET callback,
  v1.1), CUDA MPS (v1.1), chromaticindex1024-7 as a known loss. Due before `prereg-v2`, not Saturday: #36 (checker
  tolerance) and #32's test-set size. Also open: #37 (code owed by D-024), #38 (the pool should enforce R26's opening
  gate; `tools/r26-gate.sh` until then), #33 (control runs moved aside by run-id reuse). GitHub milestones *C1
  checkpoint (2026-10-23)* (still named for the old date) and *v1 (end of November 2026)* — #1 (pool freeze), #31
  (cuOpt 26.10), #32 (`prereg-v2`, confirmatory and MIPfeas runs), #12 (paper). Plan: `vault/00-index/Roadmap.md`.

### Four-seed result (D-031) — filled in after the run
**Placeholder: no number of D-031 exists yet.** Set `c1r-ysync-ld0-s23` (`jobs/d031.jsonl`): D-030's 68 C1 runs at seeds
2–3 on the same candidate, started 2026-10-16 08:57, read together with D-030's seeds 0–1 by `tools/d031_report.py
--d030-json logs/d030-reading.json`. Its `PHASE-1 INTERIM RESULT` line goes here once the 136 runs are read. It carries
the four-seed hint and guard, with seeds 2–3 alone beside as the selection check, only if all 136 read VALID; otherwise
it carries D-030's two-seed reading and names why (INCOMPLETE or INVALID, n of 136). Until then the phase-1 interim
reading is D-030's two-seed one above.

## How to start
1. Read [`AGENTS.md`](AGENTS.md) (every rule for agents and the code map), then the vault's
   [`vault/00-index/Home.md`](vault/00-index/Home.md), the newest checkpoint in `vault/90-checkpoints/` and the
   Roadmap.
2. Install and run the tests (below).
3. A toy run needs no lease and no GPU — SCIP alone on a toy knapsack for 10 s, the runner's own example
   (`src/itrea/experiments/pair.py`); it takes the runner lock, so it refuses to start while a pool runs, and `--no-pin`
   skips the GB10 core pinning on other machines:
   ```bash
   python -c "from itrea.benchmarks.toy import knapsack, write_mps; write_mps(knapsack(), '/tmp/toy.mps')"
   itrea pair --instance /tmp/toy.mps --arm A0 --seed 0 --budget 10 --set smoke
   ```
4. Timed runs on real instances happen only inside a window on the shared DGX Spark — repository skills
   `itrea-window` (box-window lease, the local LLM servers stopped and restored with `tools/serving-window.sh`, memory,
   thermal and GPU-health gates) and `itrea-new-experiment` (job file, frozen configuration, pool). Job files run with
   `itrea pool JOBS --layout pair`; then `itrea build-runs` and the experiment's report. Two practices learned from
   incidents (`vault/60-incidents/Mistakes and incidents.md`): before a window that runs new GPU code, the GPU-marked
   cold-cache tests (`pytest -m gpu -k cold_cache`) must pass and must reach the new code after `t0`; and no commit to
   `src/` or `configs/` while a pool can still launch runs (each run imports the current source) — develop in a git
   worktree.

## Install
```bash
git clone https://github.com/Tideseed/i-trea-ii && cd i-trea-ii     # private while the study runs
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,plots]"          # CPU backend (NumPy/SciPy/Numba), HiGHS, SCIP, Redis client, tests, plots
uv pip install -e ".[gpu]"                # + CuPy for CUDA 13 (the GPU backend)
uv pip install -e ".[cuopt]" --extra-index-url https://pypi.nvidia.com   # + NVIDIA cuOpt 26.08 (arms A2, A6, A9, H2)
pytest -q                                 # toy-scale tests (GPU tests skip without CuPy; -m "not integration" is faster)
```
Versions are pinned in `pyproject.toml` to the ones used for the study: Python 3.12, NumPy 2.4.6, SciPy 1.18.1,
Numba 0.64.0, HiGHS 1.15.1, PySCIPOpt 6.2.1 / SCIP 10, pandas 3.0.3, PyArrow 23.0.1, redis-py 8.1.0, PyYAML 6.0.3,
CuPy 14.2.0 (`cupy-cuda13x`), cuOpt 26.08, pytest 9.1.1, Hypothesis 6.168.1. cuOpt 26.10 will get its own environment
when it is released (issue #31); a 26.10 nightly sits in a separate venv for the cuopt#2073 check only. One entry
point, `itrea <command>`, reaches every tool (`itrea --help` lists the commands, `itrea <command> --help` their
arguments; `src/itrea/cli.py`).

**On the author's machine the environment was built offline** (no network): every wheel came from uv's local cache;
cuOpt's dependency closure (28 distributions) was hard-linked from the optopt environment following each
distribution's `RECORD` file, because uv's cache had lost the NVIDIA index metadata; redis-py 8.1.0 came from the cache
without `hiredis` (the pure-Python parser is used). On a networked machine the commands above install the same
versions.

The exchange platform is a Redis 7 server of the project's own on 127.0.0.1:6390 (container `itrea-redis` from the
`redis:7-alpine` image, no persistence, 2 GB cap): `itrea exchange up|status|flush RUN_ID|down`. Without Redis, the
file mailbox on `/dev/shm` offers the same API; the runner and the pool use it by default (`--mailbox file|redis`).

## Layout
| path | what |
|---|---|
| `src/itrea/` | the package: `cli` (the `itrea` command), `config`, `problem`, `backend`, `trace`, `paths`; `ea/` (evaluation, population, three-rank survival, elite clearing, operators, feasibility-jump local search on GPU and CPU, continuous decoding and the LP decode worker, repair clock, A9 GPU back-off, equality aggregation (not wired in), frontier, engine, EA process); `exchange/` (file mailbox, Redis); `solvers/` (SCIP, HiGHS, cuOpt processes); `experiments/` (gates, GPU health, Redis container control, environment record, runner, job builder, pool); `analysis/` (exact checker, metrics, run tables, pre-registered statistics, lineage, re-derivation, the back-off report); `benchmarks/` (toy models, hard-pool screening and freeze, reference values) |
| `configs/` | `itrea2-default.yaml` (every parameter at its default, generated; hash `9ac4a93bf93b`); development files (never confirmatory): `itrea2-adaptive.yaml` (`584ddee83747`), `itrea2-adaptive-lps.yaml` (`a2ad94e9188d`), `itrea2-adaptive-lps-rc.yaml` (the repair clock in shadow mode), `itrea2-dev-rc-3way.yaml` (repair clock active; `1d12ad490dd0`; A9 / A3x until 2026-10-15, the old default), `itrea2-dev-rc-ld.yaml` (+ LP decode; `f030e23909ad`; the A3 default), `itrea2-dev-rc-3way-bo*.yaml` (A9 back-off: `bo0` shadow, `bof50` / `bof25` / `bof12` forced duty, `bo1` active, `bo1s` its shadow control, `boY` phase Y only; not adopted, D-018 / D-023); the C1 diagnosis's variants of `-3way.yaml` and of `-3way-boY.yaml`, each its base + one change (R1, D-025 / D-029: `-bsync`, `-bnice`, `-ysync`; R2, D-026: `-bsync-ld`, `-bsync-ld0`, `-ld0`); **`itrea2-dev-rc-3way-ysync-ld0.yaml`** (`-ld0` + `yield`; EA `835c7c5a249c`, A9 run hash `f345de0ab936`; A9's and A3x's development default since D-026 + D-029 and the configuration that freezes for phase 1, D-032; `frozen-v2.yaml` is written from it on Saturday); `itrea1-faithful.yaml` (closest to part I; a record, not a planned comparison); `ablations/` (x0 — the H6 arm, y0, no_bound_demote, one-way exchange both ways, post_frontier); `tests/test_configs.py` checks them |
| `tests/` | correctness tests on toy models and synthetic problems (`-m gpu` needs CuPy, `-m solver` runs a real solver for seconds, `-m integration` launches real processes through the runner) |
| `tools/` | `serving-window.sh` (stop / restore the host's local LLM servers for an experiment window), `bench_ls.py` (local-search throughput, `itrea bench-ls`), `probe_lp_decode.py` (the LP decode worker on stored vectors), `cuopt_verify_mipstart.py` (cuOpt MIP start check, GPU), `probe_cuopt_setsol.py` (cuopt#2073's check, GPU), `d016_report.py` / `d017_report.py` / `d022_report.py` (decision reports), `d027_report.py` (D-027, the C1 diagnosis's R3, `itrea d027-report`), `d025_report.py` / `d026_report.py` (D-025 / D-026, the C1 diagnosis's R1 / R2, `itrea d025-report` / `d026-report`), `bsync_check.py` (R1's spin check before its window, GPU), `c1r_report.py` (D-030, the C1 re-read on the frozen-v2 candidate, `itrea c1r-report`), `d029_report.py` (D-029's report and job writer, `--write-jobs`), `d029_oc.py` (the pass rates D-029's rules imply, a seeded simulation), `d031_report.py` (D-031, C1 on four seeds — D-030's and D-031's sets as one reading — and its job writer; no `itrea` entry for D-029 / D-031: the scripts are the entry), `r26-gate.sh` (D-021 R26's opening gate for a chain: 3 min idle, then a 60-s median of the hottest zone ≤ 50 °C and the fan governor alive; stopgap until the pool enforces it, #38), `derive_mipfeas_strata.py` / `m0_replay.py` (`D-021`'s sets and replay), development probes |
| `jobs/`, `traces/`, `datasets/` | job files; run outputs (raw traces are not committed, resource records are); derived tables (`dev_set.csv`, `classprobe_set.csv`, the run tables) |
| `data/` | instances (not in git; symbolic links to the public sets, see `vault/10-design/Datasets.md`) |
| `paper/` | LaTeX source of the paper (skeleton; no results) |
| `vault/` | research vault (Obsidian): journal, design notes, decisions D-001–D-032, results, literature, incidents, upstream drafts, reviews, checkpoints — start at `vault/00-index/Home.md` |
| `tutorials/` | tutorials (planned) |
| `.claude/skills/` | repository skills for AI agents (window, journal, result note, new experiment, literature, reproduce) |

## How a run is assembled
A run is one (instance, arm, seed, budget). The runner (`itrea pair`; job files run through `itrea pool`, which applies
every gate) creates a run directory `traces/raw/<set>/<run_id>/` and a fresh exchange space — a directory
`/dev/shm/itrea/<run_id>/` or the Redis key space `itrea:<run_id>:*` (`redis://127.0.0.1:6390/0?run=<run_id>`) —
writes the canonical variable order, and launches the arm's processes, each one pinned, memory-capped and
single-threaded:
```bash
python -m itrea.solvers.scip_run --instance P --seed S --budget T --run-dir D --mailbox M --recv --send
python -m itrea.ea.run --instance P --config configs/itrea2-default.yaml --seed S --budget T --run-dir D --mailbox M
```
In A9 a third process, `itrea.solvers.cuopt_run --send`, writes cuOpt's incumbents to its own channels; SCIP runs on
one Cortex-X925 core, the EA on a second, cuOpt on three Cortex-A725 cores, and the memory gate counts two CUDA
processes. Each process reads the instance and reports ready; the runner releases a start barrier with a shared `t0` on
the system-wide monotonic clock (kernel warm-up, including every crossover and mutation operator on 1-, 2- and
3-child batches, happens before it), waits for the budget, writes `stop`, checks every exit code, and writes
`run.json` (cores, caps, temperatures, GPU health, co-running services, git revision) with the merged trajectories.
Every process stores every incumbent vector, so the independent exact checker can verify each one before it enters a
metric. Contract: `vault/10-design/Process interface.md`; file formats and metrics: `vault/10-design/Trace format and
metrics.md`.

`itrea.exchange.redis_mailbox.open_mailbox(spec, role)` returns either transport behind one API (`to_solver`,
`to_ea`, `info`, `lp`, `lps` channels, A9's `to_ea_cuopt` / `info_cuopt`, barrier, stop flag, verdicts); Redis adds a
shared solution pool and a history of every exchange. Which transport the confirmatory runs use is decided by a
pre-registered overhead measurement (vault `D-006`; not run yet — every run so far used the file mailbox).

## Reproducibility plan
- **Pre-registration.** Hypotheses, arms, instance-pool rules, metrics and analysis were written before any run
  (vault `D-002`), committed and tagged `prereg-v1`; the pool will be frozen as `pool-v1`, the configuration as
  `frozen-e2`. The pool refuses test-set jobs unless the configuration hash equals `frozen-e2`'s and `prereg-v1` is an
  ancestor of HEAD, and other non-smoke jobs without the tag. Later changes are dated decisions; the D-013 aim and the
  D-019 targets go into `prereg-v2` before any confirmatory run. Results under the old plan are still reported.
- **Exact definitions.** Job files in `jobs/` define every run (instance, arm, seed, budget, configuration hash);
  seeds are fixed (screening 900/901, development 0–3, tuning 0–99, confirmatory 1000/1001).
- **Checked numbers.** Every reported objective comes from an incumbent re-checked with another library in exact
  arithmetic; every crash and void is reported.
- **Re-derivation.** `itrea build-runs` turns the run directories into `datasets/runs.parquet` and
  `datasets/incumbents_checked.parquet`; `itrea prereg` computes every pre-registered comparison from them; `itrea
  reproduce` re-derives every table in `datasets/` and `paper/generated/` from the traces and lists any difference. The
  raw traces will be published as a versioned archive with checksums, as optopt did.
- **What a re-run can reproduce.** Results are wall-clock measurements on one NVIDIA DGX Spark; GPU arithmetic is not
  bit-reproducible (atomic scatter order). Expect the conclusions, not the exact numbers, to transfer to other hardware.

## Safety rules on a shared machine
The study runs on a shared DGX Spark (unified memory, two local LLM servers, other sessions). The runner enforces the
rules in vault `D-003`: a box-window lease before any timed run; a run-set memory gate that counts GPU memory
explicitly (a systemd `MemoryMax` does not charge GPU allocations on this machine), with a 20 GB reserve and a live
watchdog (A9 needs about 73 GB, 97 GB under the external harness's 24-GB cuOpt limits, so both LLM servers are
stopped); one GPU run at a time, with the same GPU memory limit for the EA's pool and cuOpt (12 GB, set by cuOpt's
measured need); pinned single-threaded processes; a thermal gate (`D-008`, `D-021`: GPU arms start below 80 °C,
CPU-only arms below 86 °C; pause at 92 °C; external runs also need the host's fan governor alive with its clock cap
off); the fan governor's rest breaks
and no night windows (`D-009`, `D-019`); GPU-health voiding (stuck-low-clock fault, foreign GPU activity); the LLM
servers stopped and restored only through `tools/serving-window.sh`; no downloads; toy models only in tests. See
`AGENTS.md` → "Rules for new experiments".

## Credits
Author: Berk Orbay (berk.orbay@tideseed.com). The design and the code are developed with **AgentRA** (Agent Research
Assistant) — AgentRA is mainly Claude Opus 5.5, working through Claude Code — under the author's direction. Code
adapted from the predecessor carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## License
[MIT](LICENSE). Third-party files keep their own terms.
