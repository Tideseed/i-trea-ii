# AGENTS.md — companion for AI agents (and humans)

- **What it is.** **I-TREA II** (Interactive Three Ranks Evolutionary Algorithm, part II) runs a GPU evolutionary
  algorithm (EA) beside exact MIP solvers — SCIP, and in arm A9 also NVIDIA cuOpt — on one NVIDIA DGX Spark. It asks
  whether the coupling lifts each solver's weakest cases.
- **Who and where.** Research project by Berk Orbay (berk.orbay@tideseed.com). Repository `Tideseed/i-trea-ii`
  (private while the study runs).
- **Code and paths.** The code is the Python package `itrea` under `src/`; paths are relative to the repository root.

**This file holds every rule for agents working in the repository** (`CLAUDE.md` points here). Read "At a glance" and
"Read first" before anything else. Detail and history moved out of this file are in
`vault/00-index/Agent companion — detail and history.md` (called "the detail note" below).

**Contents:** At a glance · Read first · Terms · State, readings and decisions (pointers) · Four-seed result (D-031)
and its fill-in checklist · Commands · Setting · The question, the aim, the v1 targets · The three ranks · Map (code,
tools, configurations, job writers) · Rules for new experiments · Writing rules · Where things are · Changelog
(pointer).

## At a glance (2026-10-16)

| | |
|---|---|
| **Phase** | development (phase 1). It ends at checkpoint **C1 on Sat 2026-10-17**. No confirmatory run yet. |
| **Method to freeze** | arm **A9** (SCIP ↔ EA ← cuOpt) on `configs/itrea2-dev-rc-3way-ysync-ld0.yaml` (lp_decode + yield; EA hash `835c7c5a249c`). By D-032's rule it is copied to `configs/frozen-v2.yaml` on Saturday; D-031's harm reading may still put the choice to Berk. |
| **Interim result** — development, descriptive, 2 seeds (D-030) | A9 against cuOpt alone on the same footprint: mean P(120) 0.677 → 0.548 (−19 %) on the fixed 11; **HINT 0.809** (needs ≤ 0.95), **GUARD +0.050** (needs ≥ −0.05). |
| **Running now** | D-031: C1's 68 runs again at seeds 2–3 on the candidate, started 2026-10-16 08:57. Its result is a placeholder below; never guess it. |
| **Next** | Saturday: `configs/frozen-v2.yaml`, the phase-1 results note, the go / no-go (Berk's). v1: end of November 2026 on cuOpt 26.10. |

**Never write**, in any file, that I-TREA II "beats", "improves on" or "matches" anything until the corresponding
experiment note in `vault/30-results/` says so with its numbers. Every hypothesis and target in this file is
pre-registered or planned, not a finding.

## Read first — rules every agent follows

The runner enforces most run rules (`itrea.experiments.gates`, `vault/50-decisions/D-003`); documentation alone is not a
control. The full text of each rule is in "Rules for new experiments" and "Writing rules" below; the terms are
defined in "Terms" right after this block.

**Box and windows**
- **Lease first.** No timed run without this project's `box-window` lease. Someone else's lease means wait. A lease is ≤ 12 h.
- **Daytime windows only.** No window in 00:00–07:00 (Berk 2026-10-10: "No nights"; D-019).
- **Serving window.** Stop and restore the two local LLM servers only with `tools/serving-window.sh`. Schedule the
  restore before stopping; verify the restore afterwards.
- **Pin anything you run beside timed runs.** Never run analysis with default thread counts beside timed runs, and keep
  CPU work off the run cores. The concrete form is `OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python …`:
  cores 10–14 are A725 (efficiency) cores outside the timed-run lanes, except the whole-box A6m calibration lane, and
  other sessions' CPU-heavy jobs go there too (agreed with the [other project] session 2026-10-15 19:0x).
- **Pausing a pool is not ending a window.** `logs/pool.stop` stops a pool; a chain script that tests it then restores
  the servers. To pause or cut a pool inside a window, kill the chain by its PID first.
- **Never `pkill -f` / `pgrep -f` from an agent shell**: the pattern matches the shell itself.

**Code and commits**
- **No commits to `src/` or `configs/` while a pool can launch runs** (a chained pool included). Develop in a git
  worktree; merge after the pool has ended.
- **Commit identity:** author and committer `Berk Orbay <5690139+berkorbay@users.noreply.github.com>` (his GitHub
  noreply), no Claude trailer (`vault/90-checkpoints/2026-10-06-0300.md`). Berk's standing permission (2026-10-06
  18:5x): commits, pushes and issue management in this repository without asking (`vault/90-checkpoints/2026-10-06-1700.md`).
- **New GPU code:** the GPU cold-cache tests run inside the window before the first timed run, and must reach the new code.
- **Trap: stale job-writer default.** `make_jobs.ITX_A9_CONFIG` still names `configs/itrea2-dev-rc-3way.yaml` (the old
  default) until a `src/` change after the running pool. A writer that uses it writes the old default: A9 in `itx-m1` …
  `itx-m5`, `itx-r13`, `itx-r13f`; the `r13x` control of `r1-bsync` and `r2-ld`. `itx-v1` takes
  `configs/frozen-v2.yaml`; `c1r-rerun` takes its pinned files (`src/itrea/experiments/make_jobs.py`).
- **Toy models only in tests.**
- **Redis:** only the project's own container `itrea-redis` on 127.0.0.1:6390.

**Data and the outside world**
- **No downloads** (instances, wheels, models) without Berk's explicit ask. **No customer data** (customer) anywhere.
- **No personal data to outside services** (no e-mail address to any API).
- **Upstream reports and submissions** (issues, PRs, MIPLIB solutions, arXiv): draft in `vault/70-upstream/`; file only
  with Berk's yes for that one post.

**Experiments and decisions**
- **Performance runs first; targeted before wide** (Berk 2026-10-13, D-024).
- **A decision record before the runs.** One variable per change. Its mechanism check and outcome rules, with their
  numbers, are written before any run. The control re-runs in the same window. A narrowed test gets its own rule. A rule
  that fails by the letter is a stated deviation, never silently re-read.
- **Pre-registration changes** are new, dated decisions. They go into prereg-v2 (#32), which is written and tagged before
  any confirmatory run.
- **The test set runs only with the frozen configuration.** Tuning happens on the tuning set only, within the logged E4 budget.
- **Development results are descriptive.** Label them interim.
- **Record everything:** journal (append-only), decision records D-NNN, a result note per experiment with its failures,
  incidents. Skills `itrea-journal`, `itrea-result-note`.

**Writing and publication**
- Plain, precise English, no hype; never claim "first" or "novel".
- Author line: Berk Orbay. **No [organisation] in publications** (see "Writing rules").
- Name harnesses and models plainly, e.g. "Claude Code (Anthropic Claude Opus 5.5)"; no internal agent pseudonyms.
- Cite only works with a legally readable full text (skill `itrea-literature`).
- Updates to Berk open with a Performance block (Good news / Bad news); quantified, never adjectives.

## Terms (defined once)

| term | meaning | source |
|---|---|---|
| **P(T)** | primal integral ∈ [0, 1] (Berthold 2013), built only from incumbents that pass the independent exact checker; lower is better. T = 120 s in development, 300 s in the confirmatory test | `vault/10-design/Trace format and metrics.md` |
| **d** (bounded d) | per instance, seeds averaged: d_i = (P_ctrl − P_arm)/max(P_ctrl, P_arm, 10⁻³); positive = the arm is better | D-002 |
| **A0** | SCIP alone | D-002 |
| **A3** | SCIP ⇄ GPU EA, two-way exchange | D-002 |
| **A9** | SCIP ↔ EA ← cuOpt: three processes; cuOpt streams its incumbents and bound to the EA one way, the EA passes them on to SCIP | D-015 |
| **A3x** | A9's matched control: A3 with A9's EA footprint, no cuOpt | D-015 |
| **A6 / A6F / A6m** | cuOpt alone / cuOpt alone on footprint F (5 threads) / cuOpt as GAMS ran it for MIPfeas (24 threads, the whole box) | D-020, D-021 |
| **A2F** | SCIP on X925 5 + cuOpt on F's other four cores, no exchange | D-021 |
| `<arm>@<label>` | an arm under a run set's label, e.g. `A9@r13x` (the old default under ext, 2026-10-13), `A9@c1r` (the candidate in D-030 / D-031) | the decision records |
| decision titles | a record's title states its pre-run framing, not its outcome: D-029 ("… NOT ADOPTED ends R1 for v1") was **ADOPTED**; D-030 ("… frozen-v2 candidate (ld0)") ran **ysync-ld0** | D-029 § Result; D-030 § Which candidate runs |
| `frozen-e2` / `configs/frozen-v2.yaml` | `frozen-e2` = D-002's pre-registered freeze tag for test-set runs (E2), written after the E4 tuning as `configs/frozen-e2.yaml` (`record_env.FROZEN_TAG`). `configs/frozen-v2.yaml` = the C1 freeze, registered with prereg-v2, required by external A9 jobs. How the two relate is for prereg-v2 to fix: it re-specifies E2, and the freeze "comes with prereg-v2 (#32)" | D-002; `vault/30-results/Experiment index.md` rows E4, E2; D-021 |
| **F** (footprint) | A9's layout: X925 5–6, A725 1–3, the GPU. Every compared GPU arm gets it | D-021 |
| **ITX-1** | our own external benchmark standard, accepted by Berk 2026-10-11 | D-021 |
| **ext / itx1** | harness presets: `ext` = cuOpt MIP gaps 0, no thread variables, its own RMM 24 GB and host `MemoryMax` 24 GB, the project's CUDA JIT cache; `itx1` = ext + `ref_stop` + `cuopt_kill_on_end` + `scoped_hash` | D-021, "Rules for new experiments" |
| **C1** | the checkpoint that ends phase 1: Sat 2026-10-17 | D-019 addendum 2026-10-15 |
| **hint** | C1: ratio of mean P(120), A9 / A6F, over the fixed 11 ≤ 0.95. Futility-only: a hint gates widening and never declares success | D-022, D-024 |
| **guard** | C1: mean bounded d over the 17 ≥ −0.05 | D-022 |
| **harm leg** | per instance, the new A9 against 10-13's `A9@r13x`; band_i = max(0.05, the seed spread of A9@r13x or of A6F@r13x), so each instance's threshold is fixed by 10-13's runs | D-025 amendment 3, D-030 |
| **the 17** / **fixed 11** | the 17 development instances (development set 9 + class probe 8) / the 11 cuOpt-weak ones among them that D-022's hint reads | D-010, D-014, D-022 |
| **W_B** (weak set) | the instances where solver B alone has P ≥ 0.5 on screening / reference seeds never used for the evaluation | D-013 |
| **H_cu / W_SCIP** | D-021's external sets: GAMS's cuOpt-GB10 26.08 `pt_gams` ≥ 0.1 or no solution / our screening's P(300) ≥ 0.5 | D-021 R3 |
| **lp_decode** | `repair.continuous: lp_decode`: a HiGHS worker fixes the integers of a few candidates and solves the LP over the continuous variables | D-017, D-026 |
| **yield** | `host.cuda_sched: yield`: the EA's CUDA context yields the core instead of spinning | D-029 |

## State, readings and decisions — where they live

These tables moved to the detail note on 2026-10-16 so that each fact has one home; "At a glance" above is the short
form, and the living sources are kept current.

| what | detail note | living source |
|---|---|---|
| current state (phase, defaults and hashes, running sets, external harness, due dates) and the open GitHub issues | § 7 | `vault/00-index/Home.md` § Status, `vault/00-index/Roadmap.md` |
| interim readings (A9 pilot, both C1 readings, glass4, M4) and how to read them | § 8 | the phase-1 results note § 4–§ 5 |
| development decisions D-011 … D-032: verdicts and key numbers | § 9 | the phase-1 results note § 3, `vault/30-results/Development log.md` |
| changelog (dated milestones) | § 10 | `vault/30-results/Development log.md` § Changes, the journal |

## Four-seed result (D-031) — filled in after the run

**Four-seed result (D-031) — filled in after the run.** *Placeholder — no D-031 number exists yet.* Set
`c1r-ysync-ld0-s23` (`jobs/d031.jsonl`, 68 runs at seeds 2–3 on the candidate, started 2026-10-16 08:57) is read with
D-030's seeds 0–1 by `tools/d031_report.py --d030-json logs/d030-reading.json`; its PHASE-1 INTERIM RESULT line goes
here once the 136 runs are read. Until then the phase-1 interim reading is D-030's two-seed one.

### D-031 fill-in checklist

When `tools/d031_report.py` has read the 136 runs, fill in every place below in one piece of work. Until then each
stays a placeholder; never guess a number.
- `AGENTS.md`: the paragraph above; "At a glance" (Interim result, Running now).
- `README.md`: Summary (C1, Running); "Configuration and plan" (Running); § "Four-seed result (D-031) — filled in after
  the run".
- `vault/00-index/Home.md`: "Where it stands" (the C1 and Running bullets); Status (the "C1, 2 seeds" row); Decisions (the D-031 row).
- `vault/30-results/2026-10-17 Phase 1 — development until C1.md`: the title ("DRAFT until D-031 is read"); Summary
  (Result, Pending); "Status of this note" and "About this note"; § 1's table; § 3's D-031 row; § 7.
- Files with their own placeholder or "running" line: `vault/00-index/Roadmap.md` (the D-019 row);
  `vault/30-results/Experiment index.md` (the opening "Running now", DEV-31, C1-e); `vault/30-results/Development log.md`
  § "Four-seed result (D-031) — filled in after the run"; `.claude/skills/itrea-window/SKILL.md` (the D-031 row of the
  run-set table); the record D-031 itself (§ Result); the journal (skill `itrea-journal`); `paper/` (its own workflow).
- Not updated: the three snapshot notes in `vault/00-index/` — `Agent companion — detail and history.md`,
  `Home — long form.md` and `Project background.md` (§ 6–§ 7). They record 2026-10-16 as it stood.

## Commands

All entry points go through `itrea <command>` (`src/itrea/cli.py`; `itrea --help`) unless a row says otherwise. During a
window, run analysis pinned: `OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python …`.

**Set up and test**

| task | command | note |
|---|---|---|
| install | `uv venv && source .venv/bin/activate && uv pip install -e ".[dev,plots,gpu]"` | see README for cuOpt |
| all tests | `pytest -q` | toy-scale correctness tests; GPU tests skip without CuPy; `-m "not integration"` is faster |
| CPU tests in a worktree beside a pool | `CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 ../i-trea-ii/.venv/bin/python -m pytest -q -m "not gpu"` | on cores outside the run lane (10–14) |
| GPU tests | `pytest -q -m gpu`, inside the lease | includes the cold-cache tests (see "Validity of a run") |

**Run a window** (skill `itrea-window`; a full example chain: `logs/chain-1016.sh`)

| step | command |
|---|---|
| 1. claim the lease | `box-window claim --who "i-trea-ii (Claude Code)" --why "…" --until HH:MM [--stops ornith-vllm,llama-qwen38] [--mem GB]` |
| 2. schedule the restore | `tools/serving-window.sh schedule 'YYYY-MM-DD HH:MM:00'` (before `stop`) |
| 3. stop the servers | `ITREA_HOLD_UNTIL=<ISO time ≥ the lease end> tools/serving-window.sh stop` — always set it: the script's default is the next 07:45 (a night-window default), so a daytime window would hold the servers too long |
| 4. opening gate | `tools/r26-gate.sh --deadline HH:MM` (restore if it does not open) |
| 5. run a job file | `.venv/bin/python -m itrea.experiments.pool jobs/<file>.jsonl --layout pair --deadline HH:MM` (= `itrea pool`) |
| 6. restore | `tools/serving-window.sh restore` (starts both units, waits for health, resumes the canaries, releases the lease if ours), then `systemctl --user stop itrea-restore-serving.timer`; verify `is-active` **and** `is-enabled` |
| stop a pool before its next block | `touch logs/pool.stop` (kill the chain by its PID first if you only pause) |
| Redis container | `itrea exchange up\|down\|status\|flush RUN_ID` |

**Write job files and read results**

| task | command |
|---|---|
| write a job file | `itrea make-jobs <writer> …` (writers: "Job writers" below) |
| run table | `itrea build-runs` → `datasets/runs.parquet`, `datasets/incumbents_checked.parquet`; pinned during a window: `OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14 .venv/bin/python -m itrea.analysis.build_runs` |
| exact check of stored incumbents | `itrea check` |
| pre-registered analysis | `itrea prereg` |
| re-derive and compare every table | `itrea reproduce` (skill `itrea-reproduce`) |
| environment record | `itrea env` |
| local-search benchmark | `itrea bench-ls` |
| hard pool | `itrea screen-candidates`, `itrea screen-stage2`, `itrea pool-freeze` |

**Development reports** (after `itrea build-runs`, with `OMP_NUM_THREADS=1`)

| record | command |
|---|---|
| D-016, D-017 | `tools/d016_report.py`, `tools/d017_report.py` |
| D-018, D-023 | `itrea backoff-report --stage 0\|1\|1y` |
| D-020, D-021 M4 | `itrea mipfeas-report --arm A6F` (D-021's M4 calibration); `--arm A6m` = D-020's. GAMS's archive in `data/mipfeas/gams-2026-09-16/` |
| D-022 | `itrea d022-report [--followup]` |
| D-025 | `itrea d025-report [--set r1-bsync\|r1-bnice\|r1-ysync] [--control r13x\|boY]` |
| D-026 | `itrea d026-report --base … [--nice 10\|0]` |
| D-027 | `itrea d027-report [--failed both\|exclude\|keep] [--json F]` |
| D-029 | `.venv/bin/python tools/d029_report.py [--failed both\|exclude\|keep] [--json F]` (no `itrea` entry) |
| D-030 | `itrea c1r-report --a9 ld0\|ysync-ld0 [--set S] [--failed both\|exclude\|keep] [--json F]` (`--set` defaults to `c1r-<a9>`) |
| D-031 | `.venv/bin/python tools/d031_report.py [--failed both\|exclude\|keep] [--json G] [--d030-json logs/d030-reading.json]` (no `itrea` entry) |

**D-021 phase 2** (pinned, `OMP_NUM_THREADS=1`)

| task | command |
|---|---|
| the external sets (from the screening's run table) | `python tools/derive_mipfeas_strata.py --runs datasets/runs.parquet [--check]` |
| M0 replay | `python tools/m0_replay.py --runs … --inc … --root . [--plan]` (under a `MemoryMax` scope) |
| the v1 analysis | `itrea itx-report [--cuopt 26.10] [--frozen configs/frozen-v2.yaml] [--bca]` (`--m6` for M6) |
| GAMS traces of our runs | `itrea gams-trc --out DIR` |

Re-running experiments needs the public instances in `data/` and the job files in `jobs/`. Wall-clock numbers depend on
the machine (`vault/10-design/Compute resources.md`).

## Setting

### Machine
- NVIDIA DGX Spark — GB10: 10 × Cortex-X925 @ 3.9 GHz = cores 5–9, 15–19; 10 × Cortex-A725 @ 2.8 GHz = cores 0–4,
  10–14; Blackwell GPU; 121 GiB unified memory shared by CPU and GPU.
- Ubuntu 24.04.4, kernel 6.17.0-1031-nvidia, driver 580.173.02, CUDA 13.0.
- The machine is shared with other sessions and with two local LLM servers.

### Software (all pinned in `pyproject.toml`)

| component | version | role |
|---|---|---|
| Python | 3.12.3 | — |
| NumPy, SciPy | 2.4.6, 1.18.1 | — |
| CuPy | 14.2.0 | GPU; CuPy (GPU) and NumPy (CPU) as one code path (`itrea.backend`) |
| SCIP | 10 (PySCIPOpt 6.2.1) | primary partner |
| HiGHS | 1.15.1 | replication partner, after its user-solution round-trip test; also the EA's own LP and the lp_decode worker |
| NVIDIA cuOpt | 26.08 (`cuopt-cu13==26.8.*`) | no longer only a control: alone it is arm A6, and since D-015 (2026-10-07) it is the third process of arm **A9** |
| Numba | 0.64.0 | the CPU feasibility-jump reference |
| redis-py | 8.1.0 | the exchange platform |
| pandas, PyArrow | 3.0.3, 23.0.1 | the derived tables |

### cuOpt: what it does in A9, and what 26.10 changes
- **In A9** cuOpt streams its incumbents and bound to the EA on its own channels, and the EA passes them on to SCIP.
- **cuOpt never receives a solution mid-run.** On the installed 26.08 its `SetSolutionCallback` disables its presolve
  (feature request NVIDIA/cuopt#2073, filed 2026-10-07).
- **Fixed in 26.10** by PR #2098 (merged 2026-10-12; Berk confirmed 2026-10-13). Verified on the nightly `26.10.0a294`
  in the separate venv `~/venvs/cuopt-2610n` (`tools/probe_cuopt_setsol.py`): presolve stays on, injected solutions in
  original order are accepted within 1–6 ms and echoed exactly, infeasible offers are rejected.
- **Limitation:** cuOpt polls SET callbacks only when its GPU population records a new best (crna: 11 calls, all at
  0.70–0.71 s; sing44: first call at 28.3 s).
- **A9 stays one-way** until a two-way arm has its own record and test (#35, not started).
- **MIP start:** a MIP start at launch is used with presolve on (verified on glass4 2026-10-07, ~10 s overhead per
  restart), but no arm uses it yet.
- **v1 runs on cuOpt 26.10** once it is released (separate venv, API check, A6 re-baselined on the 17 instances; #31).
  If 26.10 is not out by Fri 2026-11-06, the v1 runs proceed on 26.08 and 26.10 follows as an addendum (D-019; to be
  confirmed with Berk then).

### Instances
- **On disk, public:** MIPLIB 2017 benchmark, PGLib-UC unit commitment, ML4CO item placement (`vault/10-design/Datasets.md`).
- **Confirmatory study:** a frozen pool of **hard** instances chosen on SCIP alone (`vault/10-design/Hard instance pool.md`;
  not frozen yet, #1).
- **Development:** the 17 instances (see "Terms"; detail note § 7).
- **v1's external run:** the 233 feasible MIPLIB 2017 benchmark instances of Mittelmann's MIPfeas
  (`data/miplib/benchmark-v2.test` minus its 7 infeasible ones; no download needed).

### Metric
- **P(T)** as defined in "Terms"; definitions in `vault/10-design/Trace format and metrics.md`.
- Results are read in D-013's terms: weak sets first, the mean as a guard.
- **v1's external metric:** Mittelmann's MIPfeas primal integral (0–2, 600 s, shifted geometric mean, shift 0.001; D-019).

### Predecessor
- `berkorbay/optopt` (public, MIT, doi:10.5281/zenodo.22947442). Its CPU evolutionary partner beside SCIP (experiment
  X2) found no gain at 60 s and lost to SCIP ‖ SCIP; its strongest result was the SCIP/HiGHS ‖ cuOpt race.
- Code reused from it carries the line "adapted from berkorbay/optopt@4dd48395, MIT".

## The question, the aim, the v1 targets and the pre-registered hypotheses

### Main claim (D-007)
Berk's brief (2026-10-05) and main claim (2026-10-06, `vault/50-decisions/D-007`): *"a GPU-based evolutionary
algorithm metaheuristic coupled with an exact solver can beat either single solution."*
- **Starting point:** I-TREA — Berk's master's research (Orbay 2011, METU M.S. thesis, open access, `orbay2011itrea`;
  thesis, unpublished).
- **The principle carried over** is **keeping good infeasible solutions** (Rank 0). Part I reserved a share of the
  population for them as a floor; part II caps it, as Berk's brief says.

### Aim (D-013, 2026-10-07; it amends D-007)
Berk: *"I don't aim our coupling to be generally better but to help both SCIP, cuOpt (and Highs if possible) in some of
the hardest problems where they are the weakest. Our performance metric is improvement upon the worst performances, of
course with a regard to mean to some degree."* (`vault/50-decisions/D-013`)
- **Weak set** W_B: the instances where B alone has P ≥ 0.5 on screening / reference seeds never used for the evaluation.
- **Primary:** the coupling against B alone on W_B — share improved, mean P reduction, mean bounded d with its bootstrap
  interval and the Wilcoxon test, worst case after coupling.
- **Guard:** the mean over all instances, d ≥ −0.05 against each solver alone.

### v1 and its targets (`vault/50-decisions/D-019`, decided by Berk 2026-10-10)
v1 = arm A9 frozen (SCIP 10 ↔ GPU EA ← cuOpt on one GB10) with the development defaults that survive D-018 stage 1 and
#26, plus a pre-registered confirmatory test on unseen hard instances and an external run under Mittelmann's MIPfeas
protocol. The targets enter prereg-v2 verbatim:

| | minimum (v1 passes) | target | stretch |
|---|---|---|---|
| unseen hard instances: A9 vs SCIP alone on W_SCIP | mean d > 0, 95 % CI > 0 | mean P −40 %, ≥ 70 % of W_SCIP improved | −55 % |
| unseen hard instances: A9 vs cuOpt alone on W_cuOpt | mean d > 0, 95 % CI > 0 | mean P −25 %, ≥ 60 % of W_cuOpt improved | −40 % |
| guard, all instances | mean d ≥ −0.05 against each solver alone | | |
| MIPfeas protocol (233 instances, 600 s) | A9's geo-mean ≤ our own cuOpt-26.10-alone geo-mean on the same GB10 | mean PI −25 % on the hard part (our cuOpt-alone PI ≥ 0.1 or no solution) and geo-mean −10 % | geo-mean ≤ 0.0274 (SCIPConc-11, published) |

The external targets are anchored to our own same-version cuOpt run because Mittelmann publishes only cuOpt 26.08 and
only aggregates (cuOpt-26.08 on GB10: 0.0393). Dates: C1 on Sat 2026-10-17 (was Fri 10-23; D-019 addendum 2026-10-15;
development closes on cuOpt 26.08); v1 at the end of November 2026 (aim Fri 11-27, hard end Mon 11-30) on cuOpt 26.10,
daytime windows only.

### External benchmark: our own standard ITX-1 (D-021, accepted by Berk 2026-10-11)
D-020 (GAMS's MIPfeas archive, downloaded on Berk's yes; GAMS's primal integral ported exactly — `gams_integral` equals
GAMS's own function on all 3,495 of its traces) is narrowed by D-021, whose [Berk] items Berk accepted on 2026-10-11.
- **Equal footprint F** = A9's layout for every GPU arm (A6F = cuOpt alone on F); no whole-box run.
- **The call clock decides**, with claims on both clocks; 600 s.
- **Independent hard / weak sets** (GAMS's cuOpt-GB10 row, our screening); decisions on the unseen instances (215 after
  M3's "does not fit F" list: eilA101-2, nw04, supportcase6).
- **The minimum** is an ε-floored guard + solutions found; the shift-0.001 geo-mean is reported, not decisive. D-019's
  literal readings are reported beside.
- **Measurements:** F accepted on M2's soak (steady p95 66–73 °C, maximum 76 °C over 4 runs of 600 s); M3 fixed the
  "does not fit F" list (decision set 215); M4 OUTSIDE (see "At a glance"; detail note § 8).
- **Phase 2 built 2026-10-12** (branch `dev-itx2`, merged as 6f0339d; D-021's addendum): R12's stop at the proven
  optimum; the sets fixed in `datasets/mipfeas_strata.csv` (sha256 `c0221d429f33…` after the same day's review fixes —
  R33's `exposed` column, GAMS's values read round trip; every set and D-021 count unchanged); the v1 job writers,
  `itrea itx-report`, `itrea gams-trc` and M0 (replay: `pt_gams` unchanged; A9's memory plan: no instance refused).
- **Review fixes** (12 findings, D-021's addendum): the minimum read on R31's stratified estimates while T3 is
  incomplete; "incomplete (n of N)" readings; every decision also without R33's exposed instances; external A9 jobs
  only on `configs/frozen-v2.yaml` registered with prereg-v2; A0's run ids on its SCIP-side harness only; R12's trigger
  = the earliest record across processes.
- **Run time:** the MIPfeas runs need ≈ 78 h of run time — about 8–10 long daytime windows with the LLM servers off.

### Staged testing (D-024, Berk 2026-10-13)
- Performance runs first; targeted runs where a change can act; wider only after a hint whose threshold is fixed before
  the runs (futility-only: a hint gates widening, never declares success).
- **v1's staging** (accepted by Berk): stage 1 = the 20-instance panel plus the 5 H_cu-unseen instances where cuOpt finds
  no solution, × {A9, A6F}, hint = ratio of means ≤ 0.95; the guard-only stage runs whatever the hint says; A0 only on
  the 81 instances a decision reads (only the 25 stage-1 instances if there is no hint, D-024 (f)).

### Milestones (GitHub)
- **C1 checkpoint (2026-10-23)** — the milestone keeps this old name; the date moved to Sat 2026-10-17 (D-019 addendum
  of 2026-10-15):
  - #23: D-018 stage 1 decided; D-023 NOT ADOPTED.
  - #26: recombine each new partner incumbent with the elite — carried to phase 2, not built.
  - #27: lp_decode.
  - #30: lp_decode inside A9 — no new core, the decoder gated after the EA's LP worker (D-026 ADOPT).
  - #29: the MIPfeas harness, calibrated with cuOpt 26.08 alone on a ~30-instance sample against the published GB10
    row (M4 OUTSIDE, the footprint).
  - An interim A9 result on the 17 instances × 2 seeds against SCIP alone and cuOpt alone (D-030; four seeds with D-031).
  - A go / no-go for the final phase.
  - Also in the GitHub milestone (checked 2026-10-16): #33 (run-id reuse), #36 (checker tolerance), #37 (code owed by
    D-024); the detail note § 7 has each in the open-issues table.
- **v1 (end of November 2026):** #31 (cuOpt 26.10), #32 (prereg-v2, confirmatory and MIPfeas runs), #1 (pool freeze),
  #12 (paper).
- **The confirmatory test-set size is open:** D-002 v3.1 says 32 × 2 seeds, D-010 §4 proposed 16 (#11), D-019's time
  estimate assumed ~32; prereg-v2 fixes it.

### Pre-registration v3.1 — history; binding since the tag `prereg-v1` (2026-10-06); none of it tested
`vault/50-decisions/D-002 Pre-registration — hypotheses, arms, analysis.md` was written for arm A3 (SCIP ⇄ GPU EA)
before D-013 and D-019 changed the aim, the primary arm (A9) and the targets. Any change is a new, dated decision; those
changes go into **prereg-v2** (#32), which is pending and must be written and tagged before any confirmatory run. The
v3.1 summary, unchanged:

| | hypothesis (lower P(300) is better) | control |
|---|---|---|
| **H1a** (primary) | A3 SCIP ⇄ GPU EA, two-way exchange < A0 SCIP alone (harness-matched) | — |
| **H1b** (primary) | A3 < A5 the EA alone (GPU) | — |
| H2 (secondary, Holm) | A3 < the uncoupled portfolio: pointwise best of A0 and A5 of the same block (live check: A3r, no exchange) | "the exchange adds value beyond running both" |
| H3 | A3 < A2 SCIP ‖ cuOpt race on the same GPU and cores | is our EA a better partner than cuOpt? |
| H4 | A3 < A1 SCIP ‖ SCIP | equal CPU |
| H5 | A3 < A0* tuned SCIP alone | tuning budget equal to the EA's |
| H6 (mechanism) | A5 (x = 10 %) < A5x0 (x = 0), on the EA's own P(300) | do good infeasible solutions help? |

- **Decision:** the main claim holds only if H1a **and** H1b are supported (intersection–union, each at α = 0.05).
- **Per comparison:** d_i = (P_ctrl − P_arm)/max(P_ctrl, P_arm, 10⁻³) per instance (seeds averaged), Wilcoxon
  signed-rank on d_i, mean d_i with a 95 % bootstrap interval; "negative" only if the whole interval lies below the
  +0.05 margin, otherwise "inconclusive" unless supported.
- **Size:** 32 test instances × 2 seeds (1000, 1001); minimum detectable effect about 0.20–0.25 in d.
- **Secondary metric:** a MIPfeas-style integral is reported beside P(T) (a published GB10 cuOpt baseline exists under it).
- **Exploratory:** A5y0 (protected elite), A4 (CPU EA: the GPU's share), A6 (cuOpt alone), exchange-direction and other
  ablations, HiGHS as partner, 1,800 s, strata.

## The three ranks (part I's names; `src/itrea/ea/ranks.py`)

| rank (I-TREA name) | who | size | rule |
|---|---|---|---|
| **Rank 0 — infeasible good** | infeasible individuals whose key beats the incumbent (the population's best feasible or the solver's incumbent, whichever is better) | at most ⌊x · P⌋, x = 10 % — a **total cap** | over the quota: Pareto fronts in (key, violation), the overflowing front thinned by nearest distance; candidates beyond the solver's **proven** bound are eliminated (the solver answers for part I's decision maker); over-quota candidates are eliminated, never kept as Rank 2 |
| **Rank 1 — feasible good** | the best feasible individuals | ⌈y · min(#feasible, P)⌉, y = 10 % | protected every generation; equal-key twins leave Rank 1. Option, **not adopted** (D-016, `ea/elite.py`): `ranks.rank1_mode: clearing` — up to ⌈y · P⌉ members chosen best-first by cuOpt's clearing rule (any two differ in ≥ k integer variables, k = 1–10 by the number of integers); seeds 0–3 against the default: mean d −0.033 [−0.122, +0.050]; the size-only variant `ranks.elite_basis: population` was not adopted either (−0.000 [−0.044, +0.049]). The default stays `top` |
| **Rank 2 — rejects** | everything else | the rest of P | generational by default: only this generation's offspring, injected and immigrant individuals compete (the old Rank 2 dies); non-dominated sorting in (key, violation) with crowding distance |

- **Tournament order:** Rank 1 < Rank 0 < Rank 2. Before any incumbent exists the population is in a feasibility phase
  (ranked by violation).
- **The solver is the "interactive" counterpart:** it supplies its incumbent, very good pool solutions, its proven bound
  and its root LP, and receives the EA's better incumbents as new starting points. Since D-011 SCIP also streams good
  infeasible node-LP points to the EA (the LP stream), and in A9 cuOpt streams its incumbents to the EA one way (D-015).
- **The frontier:** beside the ranks the EA keeps a **frontier**: an archive of diverse near-optimal feasible solutions,
  non-dominated in (objective, distance from the incumbent) (`vault/50-decisions/D-005`; exploratory metrics only).

## Map

**Status key.** **core** = written before the design critique of 2026-10-05 and revised after it; **built** = written by
build agents from 2026-10-06 on and covered by the test suite. Latest suite counts: non-GPU 1,256 passed, 1 xfailed on
main after the R1–R3 merges of 2026-10-14; 1,339 passed on branch `c1-rerun` before its merge on 2026-10-15; GPU suites
45 passed, 1 skipped on 2026-10-15. Earlier counts: detail note § 3.

**Modules without their own test file:** `ea/population.py`, `ea/evaluate.py`, `ea/frontier.py`,
`exchange/redis_mailbox.py`, `trace.py` and `problem.py` are exercised through the engine, EA-process and runner tests
(default settings); `ea/ranks.py` has `tests/test_ranks.py` and `tests/test_elite.py`, the file mailbox
`tests/test_review_mailbox.py`. Check a non-default branch before relying on it
(`vault/80-reviews/2026-10-05 design critique — response.md`).

Rows marked "detail § 4" have their full note in the detail note.

### Code (`src/itrea/`)

| path | what | status |
|---|---|---|
| `src/itrea/cli.py` | the `itrea` entry point: one table maps every command to its module | built |
| `src/itrea/config.py` | `EAConfig`: every parameter, validated, hashed; YAML in `configs/` | core |
| `src/itrea/problem.py` | MIP read with HiGHS into arrays; per-row float64 feasibility check under the partner's tolerance rule (`tol_mode`) | core |
| `src/itrea/backend.py` | CuPy / NumPy backend; the CuPy pool limit is the only GPU memory cap on GB10 | core |
| `src/itrea/trace.py` | per-process event stream, summary, every incumbent vector (`save_incumbents`); monotonic shared clock | core |
| `src/itrea/paths.py` | repository paths; instance paths stored relative to the root | core |
| `src/itrea/benchmarks/toy.py` | toy models (knapsack with DP optimum, set cover, mixed) and `write_mps` — the only models tests may solve | core |
| `src/itrea/benchmarks/references.py` | reference values: MIPLIB `.solu`, the frozen `datasets/reference.csv`, provisional references before the freeze | built |
| `src/itrea/benchmarks/hardset.py` | the hard pool: screening candidates, the stage-2 rule, strata, family split, freeze (`itrea screen-candidates`, `screen-stage2`, `pool-freeze`) | built |
| `src/itrea/ea/evaluate.py` | device copy of the problem; batched, chunked evaluation (key, violation V, feasibility) | core |
| `src/itrea/ea/population.py` | population arrays, ids, provenance bits, compact gene dtypes, exact duplicate detection | core |
| `src/itrea/ea/ranks.py` | three-rank survival | core |
| `src/itrea/ea/elite.py` | Rank 1 as a distinct elite (D-016, #25): cuOpt's clearing rule on the device (`clearing_pick`, blocked distances; `min_diff_auto` = cuOpt's k by the number of integers) and the Rank 1 spread metrics (`elite_spread`, `r1_*`, logged in both modes); used by `ranks.py` under `ranks.rank1_mode: clearing` — 2026-10-08, run in D-016's test (seeds 0–3), not adopted | built |
| `src/itrea/ea/frontier.py` | the frontier archive (k-objective non-dominated sort, 2-D hypervolume) | core |
| `src/itrea/ea/selection.py` | parent selection (tournament; comparators `itrea`, `deb`, `stochastic`) | built |
| `src/itrea/ea/crossover.py` | `agreement`, `uniform`, `block`, `k_point` | built |
| `src/itrea/ea/mutation.py` | `flip`, `step`, `reset`, `toward_lp`, `violation_guided`, `objective_guided` | built |
| `src/itrea/ea/seeding.py` | initial population, immigrants, LP point after `t0` | built |
| `src/itrea/ea/repair.py` | `LocalSearch` dispatcher over the three implementations below | built |
| `src/itrea/ea/ls_fj_cpu.py` | feasibility jump on one CPU core (reference and arm A4's repair) | built |
| `src/itrea/ea/ls_fj_gpu.py` | feasibility jump as a CuPy `RawKernel`, one thread block per individual | built |
| `src/itrea/ea/ls_dense.py` | vectorised full rescore (baseline only) | built |
| `src/itrea/ea/continuous.py` | closed-form decoding of singleton continuous columns (`repair.continuous: auto`) | built |
| `src/itrea/ea/lp_decode.py` | the LP decode (`repair.continuous: lp_decode` = auto + a HiGHS worker process that fixes the integers of a few candidates per generation and solves the LP over the continuous variables; D-017, #27). Refused for A9 / A3x unless gated after the EA's own HiGHS LP worker (`repair.lp_decode_gate: lp_worker` with `seeding.lp_source: highs_async`). Adopted as the A3 default (D-017) and inside A9 (D-026, #30) — detail § 4 | built |
| `src/itrea/ea/repair_clock.py` | the repair clock's controller: one lane-step horizon per generation from each horizon's measured gain per second (`repair_clock`, off by default; shadow mode logs only) — built 2026-10-06; Stage 1 (shadow) and Stage 2 (active) run 2026-10-07; **active in the development default** since D-012 addendum 2 | built |
| `src/itrea/ea/backoff.py` | A9's GPU back-off (D-018): a contribution-gated duty cycle (`backoff`, off by default, inert unless `exchange.from_cuopt`; modes active / shadow / forced). Stage 1 NOT ADOPTED 2026-10-11; phase Y alone NOT ADOPTED 2026-10-14 (D-023) — detail § 4 | built |
| `src/itrea/ea/aggregate.py` | exact aggregation of two-variable equality rows (a presolve-lite for the genome) — 2026-10-06, **not wired into the EA** (the payoff probes on radiationm were negative) | built |
| `src/itrea/ea/engine.py` | the generation loop, exchange in/out, genealogy, frontier updates; the EA host's settings (`host`), the LP decode's gate, and per-run counters (`lp` event, `cpu_clock`, `ls_launches`, `concat_parts_max`) — detail § 4 | built |
| `src/itrea/ea/cuda_sched.py` | the EA's primary CUDA context's scheduling flag (`host.cuda_sched`: blocking_sync / yield) through the driver API; under "auto" the flags are only read back. `yield` adopted by D-029, `blocking_sync` not (D-025) — detail § 4 | built |
| `src/itrea/ea/run.py` | EA process: `python -m itrea.ea.run …` | built |
| `src/itrea/exchange/mailbox.py` | file mailbox on `/dev/shm` (fallback transport) | core |
| `src/itrea/exchange/redis_mailbox.py` | Redis transport and shared pool; `open_mailbox(spec, role)` | core |
| `src/itrea/solvers/common.py` | what the solver processes share: arguments, barrier, recorder, exchange accounting, exit codes | built |
| `src/itrea/solvers/scip_run.py` | SCIP process; injection through a heuristic plugin (`trySol`) | built |
| `src/itrea/solvers/highs_run.py` | HiGHS process; user-solution callback | built |
| `src/itrea/solvers/cuopt_run.py` | cuOpt process; never receives solutions; RMM limit; in A9 (`--send`) it posts its incumbents and bound on its own channels (`to_ea_cuopt` / `info_cuopt`); `--mip-gap-zero` (both MIP gaps 0, GAMS's MIPfeas settings, arm A6m, D-020); the solve's parameters recorded as the summary's `settings` | built |
| `src/itrea/experiments/gates.py` | lease, memory gate, thermal gate, run lock, watchdog; readers of the host's fan governor (state file: alive, cap; log: clock-cap intervals) and of thermal-watch's 98 °C stop lines (D-021) | core |
| `src/itrea/experiments/gpu_health.py` | per-second SM clock, temperature and foreign-GPU sampling; block voiding (stuck clock < 2,000 MHz halts; D-021's cap floor 2,300 MHz in `CAP_FLOOR_MHZ`) | built |
| `src/itrea/experiments/exchange_ctl.py` | the `itrea-redis` container: up, status, flush one run's keys, down | built |
| `src/itrea/experiments/record_env.py` | versions, hardware, git state for every run record | built |
| `src/itrea/experiments/pair.py` | one run: launch the arm's processes, start barrier, stop, every exit code checked, `run.json`; every arm's layout (A9 / A3x, A6m, A6F / A2F), the job-level `harness` (presets `ext`, `itx1`), named scopes, the void causes, R12's stop at the proven optimum — detail § 4 | built |
| `src/itrea/experiments/pool.py` | a job file under every D-002/D-003 control: Williams arm order per block, blocks never split, gates per run; layouts `pair`, `screen`, `full`; D-021's gates and breaker; R33's gate for external jobs — detail § 4 | built |
| `src/itrea/experiments/make_jobs.py` | job files (`itrea make-jobs …`): see "Job writers" below; full record: detail § 6 | built |
| `src/itrea/analysis/metrics.py` | primal gap, P(T), primal-dual integral, times to target (adapted from optopt); GAMS's MIPfeas integral exactly (`gams_integral`, D-020: equals GAMS's own function on all 3,495 traces of their archive) | built |
| `src/itrea/analysis/check_solutions.py` | independent exact check of every stored incumbent (adapted from optopt) | built |
| `src/itrea/analysis/build_runs.py` | run directories → `datasets/runs.parquet`, `datasets/incumbents_checked.parquet`; the MIPfeas columns (`pt_gams`, `pt_gams_launch`, `startup_offset_s`) and D-021's run-record columns — detail § 4 | built |
| `src/itrea/analysis/itx_strata.py` | D-021's instance sets: the rules, R33's `exposed` list (calibration sample + M3's probe), `datasets/mipfeas_strata.csv`'s reader (`read_strata`) and counts | built |
| `src/itrea/analysis/itx_report.py` | D-021's external analysis (`itrea itx-report`): completeness, target, minimum, the Holm secondaries, geo-means, D-019's literal readings, both clocks; only runs with the registered settings — detail § 4 | built |
| `src/itrea/analysis/gams_trc.py` | GAMS `.trc` export per run and clock (`itrea gams-trc`, R16): GAMS's own script reproduces `pt_gams` to 1e-12 (its default `read_csv` returns our repr floats within ~1e-12 relative, not bit for bit) | built |
| `src/itrea/analysis/stats.py` | d_i, Wilcoxon, bootstrap, decision rule, censored endpoints | built |
| `src/itrea/analysis/lineage.py` | depth-1 matched Rank 0 odds ratio, Mantel–Haenszel (descriptive) | built |
| `src/itrea/analysis/prereg.py` | the pre-registered analysis of D-002 v3 from `runs.parquet` (`itrea prereg`) | built |
| `src/itrea/analysis/reproduce.py` | re-derive `datasets/` and `paper/generated/` and compare (`itrea reproduce`) | built |
| `src/itrea/analysis/mipfeas.py` | D-020's MIPfeas calibration (`itrea mipfeas-report`): A6m against GAMS's published cuOpt 26.08 GB10 row, verdict PASS / FAIL / INCOMPLETE; `--arm A6F` = D-021's M4 with the late-clock check, reading INSIDE / OUTSIDE / CLOCK-FAIL — detail § 4 | built |
| `src/itrea/analysis/backoff.py` | D-018's stage-0 calibration rules and stage-1 criteria, and D-023's stage 1y (`itrea backoff-report --stage 0\|1\|1y`) — detail § 4 | built |
| `src/itrea/analysis/c1_devtests.py` | what the D-025 / D-026 reports share: the selection by set (reused `#<set>` ids, attempts), each run's `run.json` / `ea.summary.json`, the common validity legs (complete, harness, config = the job's run id, compile, block), the one-sided seed-mean band and the ratio of means | built |

### Tools (`tools/`)

| path | what | status |
|---|---|---|
| `tools/serving-window.sh` | stop / restore the two local LLM servers for a window (`schedule`, `stop`, `restore`) | core |
| `tools/r26-gate.sh` | D-021 R26's opening gate for a chain (`tools/r26-gate.sh [--deadline HH:MM] [--idle-s 180] [--limit-c 50]`): exit 0 open, 1 at the deadline; a stopgap until the pool enforces it (#38) — detail § 4 | built |
| `tools/screening-night.sh` | the screening chain: stage 1 → `build_runs` → stage 2 under one deadline (D-002 v3.1 §1.2) | built |
| `tools/derive_mipfeas_strata.py` | D-021's sets → `datasets/mipfeas_strata.csv` (counts beside D-021's, sha256; `--check` re-derives) | built |
| `tools/m0_replay.py` | D-021 M0: R12 replayed offline on recorded runs and GAMS's 3,495 traces, hours saved, A9's shape-only memory plan over the 233 | built |
| `tools/bench_ls.py` | local-search throughput, GPU vs CPU reference (`itrea bench-ls`) | built |
| `tools/bsync_check.py` | R1's mechanism check before its window (GPU: only inside a lease): host CPU of a kernel-launch loop per CUDA scheduling mode, alone and beside a busy process; run by `tests/test_cuda_sched.py`; figures in `traces/bench/bsync/` — detail § 4 | built |
| `tools/d016_report.py` | D-016's analysis: A3@cl1 vs A3@rc1d and the size-only A3@tp1 (mechanism and outcome rules of the record) from `runs.parquet` and the run directories | built |
| `tools/d017_report.py` | D-017's analysis: A3@ld1 vs A3@rc1e (M0–M4 and the outcome rules) | built |
| `tools/d022_report.py` | D-022's rules (`itrea d022-report [--followup]`) and C1's interim hint and guard vs A6F — detail § 4 | built |
| `tools/d025_report.py` | D-025's rules (`itrea d025-report`); read 2026-10-15 on `r1-bsync` and `r1-ysync` (both NOT ADOPTED) — detail § 4 | built |
| `tools/d026_report.py` | D-026's rules (`itrea d026-report --base …`); read 2026-10-15 13:2x on `r2-ld` with `--base r13x` (ADOPT) — detail § 4 | built |
| `tools/d027_report.py` | D-027's rules 1–4 on set `r3-diag` (`itrea d027-report`); read 2026-10-15 09:2x: rule 4 EA PRESENCE, rule 1 closed, rule 3 SCIP ruled out, rule 2 not read — detail § 4 | built |
| `tools/d029_report.py` | D-029's rules over `r1-ysync` (seeds 0–1) + `r1f-ysync` (seeds 2–5); no `itrea` entry; `--write-jobs` writes `jobs/d029.jsonl`; read 2026-10-15 15:0x: ADOPT — detail § 4 | built |
| `tools/d029_oc.py` | D-029's operating characteristic: the pass rates its rules imply under a normal model fitted to the development default's same-window control pairs (a seeded simulation, no solver, no GPU; `--check` compares its data with the run table) — the source of the numbers in D-029's "Why these numbers" and its amendment | built |
| `tools/c1r_report.py` | D-030's C1 re-read on set `c1r-<a9>` (`itrea c1r-report --a9 ld0\|ysync-ld0`): validity gate, C1's hint and guard, D-025 amendment 3's harm leg; read 2026-10-16 08:2x on `c1r-ysync-ld0` (68 of 68, VALID; `--json logs/d030-reading.json`, D-030's Result and § Correction 2026-10-16 10:0x) — detail § 4 | built |
| `tools/d031_report.py` | D-031: C1 on the candidate read on four seeds (sets `c1r-ysync-ld0` + `c1r-ysync-ld0-s23`) as one reading, one `PHASE-1 INTERIM RESULT` line; no `itrea` entry; `--write-jobs jobs/d031.jsonl`. Its set runs since 2026-10-16 08:57, not read yet — detail § 4 | built |
| `tools/probe_lp_decode.py` | probes the LP decode worker on stored vectors (CPU, read-only; D-017's gates before a window) | built |
| `tools/cuopt_verify_mipstart.py` | cuOpt 26.08 MIP-start and stopping check on glass4 (GPU: only inside a window; run 2026-10-07) | built |
| `tools/probe_cuopt_setsol.py`, `tools/probe_cuopt_setsol_report.py` | cuopt#2073's check: `SetSolutionCallback` with presolve (modes none / noop / inject / bad / raw) under any interpreter with cuOpt (the 26.08 venv or `~/venvs/cuopt-2610n`), and its report (GPU: only inside a lease; run 2026-10-13: 31 solves, most on the 26.10 nightly, the injection also on 26.08) | built |
| `tools/agg_probe.py`, `tools/agg_improve_probe.py` | CPU probes of `ea/aggregate.py` (feasibility and improvement payoff; 2026-10-06, negative on radiationm) | built |
| `tools/lps_smoke.py` | hand-launched smoke run of the LP stream (SCIP + the NumPy EA through a file mailbox) | built |

### Configurations (`configs/`)

Development defaults are never a confirmatory configuration (D-010). Every file's hash is pinned in
`tests/test_cuda_sched.py` (`ALL_CONFIG_HASHES`); that test also checks, once `configs/frozen-v2.yaml` exists, that its
hashes equal a configuration C1 was re-read on. Names below drop the prefix `itrea2-dev-rc` where they start with `-3way`.
Full record: detail § 5.

| file | what it is | EA hash | A9 run hash | record, status |
|---|---|---|---|---|
| `itrea2-default.yaml` | all defaults, generated from `EAConfig()`; `tests/test_configs.py` keeps it equal to the code | `9ac4a93bf93b` | — | core |
| `itrea2-dev-rc-ld.yaml` | **A3 development default**: `itrea2-dev-rc-3way` + `repair.continuous: lp_decode` | `f030e23909ad` | — | D-017, adopted |
| `-3way-ysync-ld0.yaml` | **A9 / A3x development default since 2026-10-15 15:0x and the freeze candidate**: `-3way-ld0` + `host.cuda_sched: yield`; its A9-layout cold-cache test passed 2026-10-15 15:07 | `835c7c5a249c` | `f345de0ab936` (ext) | D-026 + D-029; A9@c1r in D-030 (seeds 0–1) and D-031 (seeds 2–3, running); A9@g4yl in D-032 |
| `-3way-ld0.yaml` | `-3way` + lp_decode gated after the LP worker on the EA's core, nice 0 | `a05af2c732dc` | `341267810eed` (ext) | D-026 ADOPT (13:2x); A9@g4ld in D-032 |
| `-3way.yaml` | the **old** A9 default: repair clock active, adaptive repair strategy, the EA's own HiGHS LP, LP stream, deep repair. The runner adds `exchange.from_cuopt` for A9 and refuses an ungated lp_decode there | `1d12ad490dd0` | `28e5a5e9c725` (= A9@r13x's) | kept as the record of the old default; A9@g4old in D-032; still `ITX_A9_CONFIG` |
| `-3way-bo1.yaml` | D-018 stage 1: back-off active with the stage-0 values (D_Y ½, floor ⅛, yield_until 30 s, patiences 40 / 40) | `c5b5e01856eb` | run hash `f67354be775d` | NOT ADOPTED |
| `-3way-bo1s.yaml` | the same values in shadow mode = the control | `198e0d2ca540` | run hash `62bf82b236f6` | control of D-018 stage 1 and D-023 |
| `-3way-boY.yaml` | D-023: phase Y only — bo1's values, active, patiences 100000 so the ladder never steps down | `5c8b535fb245` | — | NOT ADOPTED (2026-10-14) |
| `-3way-bsync.yaml` | R1 on the A9 default: `host.cuda_sched: blocking_sync` | `305c8902d763` | `330a10a77556` (ext) | D-025 NOT ADOPTED |
| `-3way-bnice.yaml` | D-025's fallback (a): `host.nice_after_lp_spawn: 10`, +10 relative to the LP worker, no context flag | `96a299330981` | `d266e4aaecbe` (ext) | D-025 |
| `-3way-ysync.yaml` | D-025's fallback (b): `yield` | `f81d6b6209f0` | `a08221cca653` (ext) | NOT ADOPTED at seeds 0–1 (D-025); ADOPTED on seeds 0–5 (D-029) |
| `-3way-boY-bsync.yaml`, `-boY-bnice`, `-boY-ysync` | the same three on boY | `29e61aa41619`, `5b1c4779c74b`, `666865eac174` | `b760e770cb34`, `b4f4af8376dd`, `22137f009005` (ext) | D-025 |
| `-3way-bsync-ld.yaml` | R2: bsync + lp_decode gated after the LP worker, nice 10 | `119c6e17f830` | `dc45bdbb771f` (ext) | D-026 |
| `-3way-bsync-ld0.yaml` | the same, nice 0 | `b7af15f4b62a` | `93a8ed06b59b` (ext) | D-026 |
| `-3way-boY-bsync-ld.yaml`, `-boY-bsync-ld0`, `-boY-ld0` | R2 on boY | `c296898cddc4`, `2b6baa898edf`, `aabac30b5056` | `3ef4bf1254fe`, `37a47a123c67`, `58f72d467a23` (ext) | D-026 |
| `-3way-bo0.yaml`, `-bof50`, `-bof25`, `-bof12` | history: D-018 stage-0 arms — shadow, forced duty ½ / ¼ / ⅛ | `b7ab026b8a5d`, `7427bf6ca4d7`, `9a8d7e711bf9`, `edca6f71f266` | — | stage 0: PROCEED (2026-10-09) |
| `itrea2-adaptive.yaml`, `-adaptive-lps.yaml`, `-adaptive-lps-rc.yaml` | history: earlier development steps (D-010–D-012) | `584ddee83747`, `a2ad94e9188d`, `26481ba0c675` | — | history |
| `itrea1-faithful.yaml` | a record of part I's choices, not a planned comparison | `b517bfc675d4` | — | history |
| `ablations/*.yaml` | D-002's E3 | — | — | planned |

EA hashes not quoted in the decision records (bo1, bo1s, bo0, bof*, the adaptive files, itrea1-faithful) are read from
`ALL_CONFIG_HASHES`. "(ext)" = the A9 run hash under the ext harness.

### Job writers (`itrea make-jobs <writer>`)

Full record of every writer and option: detail § 6.

| writer | what it writes | record |
|---|---|---|
| `screen1`, `screen2` | the SCIP screening, stages 1 and 2 | D-002 v3.1 §1.2 |
| `e1`, `e4`, `confirm`, `smoke` | pre-registered experiments and smoke runs | D-002 |
| `backoff0`, `backoff1` | D-018 stages 0 / 1 | D-018 |
| `mipfeas-cal` | A6m on the 30-instance sample (its file left `jobs/` when D-021's M4 replaced it) | D-020 |
| `itx-m1` … `itx-m5`, `itx-r13` | D-021 phase 1 → `jobs/itx-*.jsonl` | D-021 |
| `itx-r13f` | D-022's follow-up at seeds 2–3 | D-022 |
| `backoff2 --harness ext\|r13-off [--instances amended\|stage1\|a,b]` | default the 11 of its 2026-10-13 amendment = 66 runs, `jobs/backoff2.jsonl`; `stage1` = D-018's 14 | D-023 |
| `itx-v1 --tier T1..T5 [--cuopt 26.10] --out F` | the v1 tiers from `datasets/mipfeas_strata.csv`; A9 on `configs/frozen-v2.yaml`, refused while it does not exist; not written before 26.10 | D-021 phase 2 |
| `itx-a0 [--select all\|d024\|stage1]` | `jobs/itx-a0.jsonl` = all 230; `itx-a0-81.jsonl` = D-024 (f)'s 81 (the 25 v1 stage-1 instances at priority 0, the other 56 at 1); `stage1` = the 25 | D-024 (f) |
| `itx-m6`, `itx-m6a` | `jobs/itx-m6.jsonl`, `jobs/itx-m6a.jsonl` | D-021 |
| `r1-bsync --control r13x\|boY [--variant bsync\|bnice\|ysync] [--set S]` | 20 runs, set `r1-<variant>`, `jobs/r1-<variant>.jsonl` | D-025 |
| `r2-ld --base bsync\|boY-bsync --nice 10\|0 \| --ea-cpu X` or `--base r13x\|boY` | 28 runs, set `r2-ld`, `jobs/r2-ld.jsonl`, labels `A9@r2ld` / `A9@r2ctl` | D-026 |
| `r3-diag` | 22 runs in 8 blocks, `jobs/r3-diag.jsonl` | D-027 |
| `c1r-rerun --a9 ld0\|ysync-ld0 [--set S]` | 68 runs in 34 blocks, set `c1r-<a9>` (a rerun gets its own `--set`), `jobs/c1r-ld0.jsonl` / `jobs/c1r-ysync-ld0.jsonl`; run hashes pinned | D-030 |

- **Job files written outside `make-jobs`** (no `src/` edit while pools ran): `jobs/d028.jsonl` (D-028),
  `jobs/d029.jsonl` (`tools/d029_report.py --write-jobs`), `jobs/d031.jsonl` (`tools/d031_report.py --write-jobs`),
  `jobs/d032.jsonl` (D-032: from D-030's glass4 A9 job, 12 runs).
- **The stale default:** `make_jobs.ITX_A9_CONFIG` still names the old A9 default (see "Read first", the trap).

### Data, jobs, paper, vault, skills

| path | what | status |
|---|---|---|
| `datasets/` | derived tables; `dev_set.csv` and `classprobe_set.csv` are the two lists excluded from the test set; `mipfeas_strata.csv` = D-021's fixed external sets (hashed in prereg-v2); `runs.parquet` / `incumbents_checked.parquet` are rebuilt by `itrea build-runs` and not committed yet; `hardness_from_optopt.csv` is background only | — |
| `jobs/`, `traces/` | job files; raw traces (not committed) and resource records (committed) | — |
| `paper/` | LaTeX skeleton (`main.tex`, `sections/`), `refs.bib` (legally readable works only) | skeleton |
| `vault/` | the research vault (Obsidian): journal, design, decisions D-NNN, results, literature, incidents, reviews, checkpoints | — |
| `.claude/skills/` | repository skills: `itrea-window`, `itrea-journal`, `itrea-result-note`, `itrea-new-experiment`, `itrea-literature`, `itrea-reproduce` | — |

## Rules for new experiments

The runner enforces most of these (`itrea.experiments.gates`, `vault/50-decisions/D-003`); documentation alone is not a
control. Skills: `itrea-window`, `itrea-new-experiment`.

### Box: lease, windows, servers, memory, deadline
- **Lease first.** No timed run without the box-window lease held by this project:
  `box-window claim --who "i-trea-ii (Claude Code)" --why "…" --until HH:MM [--stops ornith-vllm,llama-qwen38] [--mem GB]`;
  `box-window release` when done. Other sessions share the box; a held lease by someone else means wait.
  - `box-window` refuses a lease longer than 12 h ("not a window; split it"). On 2026-10-14 a scheduled chain's
    `--until 20:15` from 07:05 was refused and 7 h 36 min of a window were lost.
  - So a scheduled chain claims ≤ 12 h, runs under `OnFailure=agent-failure-alert@`, exits 1 on any early stop, and is
    checked within 15 min of its start (skill `itrea-window`; the 10-15 and 10-16 chains, 11 h 50 min each, opened on
    schedule).
- **Daytime windows only.** No window in the night rest (00:00–07:00): Berk, 2026-10-10, "No nights" (D-019; before
  that D-009 asked for his yes for each night).
  - Long run sets (v1's MIPfeas runs, ≈ 78 h) are split into daytime windows.
  - The runner honours the host fan governor's rest flag (D-009): no new run starts while it exists.
- **Serving window.** Confirmatory runs need both local LLM servers stopped (vLLM `ornith-vllm` :8094, llama.cpp
  `llama-qwen38` :8092; Berk's yes, 2026-10-05). Development windows stop both too whenever the memory gate needs it —
  A9 (two CUDA processes) needs ≈ 73 GB.
  - Stop and restore only with `tools/serving-window.sh stop|restore`, and schedule the restore on a timer before starting.
  - Verify after restore: `is-active` **and** `is-enabled`, health endpoints answer, canaries resumed, lease released.
  - Jobs with a harness need both model servers stopped (`--allow-serving` refused).
- **Memory gate.** Σ host `MemoryMax` + Σ CuPy pool limits + 1 GB non-pool overhead per CUDA process + cuOpt's RMM
  limit + the Redis cap (2 GB) + 20 GB reserve ≤ MemAvailable at launch; the watchdog stops the run set below 18 GB.
  `MemoryMax` does **not** charge GPU memory on GB10. Gate figures per arm are under "Layout" below.
- **Deadline and lease end at the start of every run.** After its gate waits, a run whose estimate would end after
  `--deadline` or the lease end does not start: a block not yet started counts as never opened and the pool stops; a
  block partly run keeps running (a stop never splits a block).
- **Pausing a pool is not ending a window.** The stop file `logs/pool.stop` stops a pool before its next block, and a
  chain script that tests it after the pool goes on to restore the servers. To pause or cut a pool inside a window, kill
  the chain first by its PID, then set the stop file, then start a chain that removes it; to end the window, the stop
  file alone is right (`vault/60-incidents/Mistakes and incidents.md`, 2026-10-13).

### Layout: one GPU arm at a time, footprint F, harness
- **One GPU arm at a time.** Timed solvers pinned one per X925 core; the partner footprint (GPU + the same 3 A725
  cores + the same GPU memory limit, **12 GB**: the EA's CuPy pool = cuOpt's RMM limit, `EAConfig.gpu_mem_gb`, set by
  cuOpt's measured need of ~10.5 GB, D-003) is identical for cuOpt and the EA.
  - **A9:** SCIP runs on X925 5, the EA on X925 6 and cuOpt on A725 1–3 (two CUDA processes, 12 GB each); its matched
    control A3x has the same EA footprint without cuOpt (D-015).
  - **A6m is the exception** (D-020, the MIPfeas calibration: cuOpt as GAMS ran it, 24 threads, MIP gaps 0): it has the
    whole box to itself — the pool's `--layout full`, one lane with all 20 cores, the serving check implied,
    `--cuopt-mem-gb 24 --gpu-mem-gb 36` (81 GB), the file mailbox (Redis and `--cuopt-gpu-peak-gb` refused) — and runs
    nowhere else; no other arm runs on that lane (D-021 R9 keeps it out of v1).
  - **A0 and M6** run on single-core X925 lanes: `--layout screen --cpu-lanes 4` (M6's alone runs: 1; 4 × 9 GB + the
    20-GB reserve = 56 GB).
- **ITX-1's footprint F (D-021).** Every compared GPU arm gets the same share **F** = A9's layout (the GPU lane: X925
  5–6, A725 1–3, the GPU): **A6F** = cuOpt alone on all of F with |F| = 5 threads, **A2F** = SCIP on X925 5 + cuOpt on
  the other four cores (4 threads), no exchange.
- **A job's `harness`** (`"ext"`, or a dict overriding its keys):
  - gives every cuOpt process MIP gaps 0, no thread variables, its own RMM limit (24 GB, separate from the EA's 12-GB
    pool) and host `MemoryMax` 24 GB, and every process the project's CUDA JIT cache `~/.cache/itrea/cuda`;
  - enters the run hash only when present (every earlier run id is unchanged);
  - is refused on A6m and on test / screening jobs (a prereg-v2 item).
  - Gates under ext: A9 97 GB, A2F 78, A6F 69 (+ 2 with Redis); M3's A6F at RMM 40 / host 40: 101.
- **Phase-2 switches (2026-10-12).** Three optional harness switches, absent from the resolved harness — and the hash —
  unless on:
  - `ref_stop` (R12): the runner ends the run when a process's stored incumbent reaches MIPLIB's `=opt=` z* — GAMS's
    gamma <= 0 and Berthold's gap <= 1e-9, obj recomputed from the vector, the earliest such record over every process;
    z* never reaches a process; never supportcase22, PGLib, ML4CO; stop kind `ref_reached`, the trigger checked exactly
    after the run, unconfirmed = void `ref_rejected`, rerun once with the stop off.
  - `cuopt_kill_on_end`: cuOpt SIGKILLed at once after a partner's proof or ref_reached.
  - `scoped_hash`: an arm without cuOpt hashes only the harness keys that act on it, so A0's run ids survive a change
    to cuOpt's settings — R33's "A0 stands if the SCIP harness hash is unchanged".
  - Preset **`itx1`** = ext + all three: the v1 tiers and A0's external jobs.
- **External jobs** (split `ext`) of an arm with the EA need the tag `prereg-v2` and the configuration registered with
  it — `configs/frozen-v2.yaml` (the C1 freeze) in the tagged commit, unchanged since, every A9 job hashing to it (R33's
  sealing, `pool.check_frozen`; `make-jobs itx-v1` refuses an A9 tier without that file). Every external job needs a
  clean tree under `src/` and `configs/`. A job-level `priority` orders block groups (A0's H_cu ∪ W_SCIP unseen first).
- **Thread variables.** `OMP_NUM_THREADS=OPENBLAS_NUM_THREADS=MKL_NUM_THREADS=1` in every process but A6m's cuOpt and
  cuOpt under a harness with `cuopt_env_unset`, whose environment has the three removed (GAMS ran it so; libcuopt links
  libomp). Never run analysis with default thread counts beside timed runs.

### Heat and GPU health
- **Thermal.** GPU arms start only below 80 °C (ACPI maximum), CPU-only arms below 86 °C (D-008's 10-06 addendum,
  D-021 R24; 88 until 2026-10-11); pause the run set at ≥ 92 °C, resume below 80 °C.
  - A block voided by the pause is rerun first, at most `--max-thermal-requeues` times (default 1): a further void skips
    it (pool status `skipped`) — in screening stage 2 (2026-10-06) the same 8 blocks were voided 8 times each before
    the cap existed.
  - Each run records the hottest ACPI zone seen while it ran (`run.json` `acpi_during`) and the 1-min load average
    before and after.
- **Fan governor (D-021).** A GPU run also starts only while the host fan governor is alive
  (`~/.local/state/thermal/fan-state` < 30 s old) with its clock cap off (`--no-governor` only on a host without it; a
  governor gone 120 s halts the pool, also during its rest flag).
  - `--pause-c` lowers the pause / void line (M2's soak: 90, never above 92).
  - A run whose scope thermal-watch stopped at ≥ 98 °C is void (`thermal_watch`, never an arm crash).
  - Run scopes are named `itrea-run-<sha1(run)>-<role>-<epoch_ms>` and stop within 5 s (cuOpt's by SIGKILL).
  - Every run records the fan state at its start and end and its start temperature.
  - The governor's rests stay as they are (Berk, 2026-10-11). Journal 2026-10-15 14:4x: four 30-min rests by 14:57,
    about every 1.5–2 h under load, about 25 % of window time (an estimate; 30 min every 1.5–2 h is 25–33 %).
- **R26's breaker.** After 2 thermal or cap voids in a run set (`--max-heat-voids`; one thermal pause counts once) no
  new block opens and the pool ends `halted`. It fired on 2026-10-15 17:30: two `gpu_cap` voids after a governor rest,
  the hottest zone at 85–87 °C within ~4 min; the next window resumed the unfinished blocks, the voided one whole.
- **R26's opening gate** (a 60-s median of the hottest zone ≤ 50 °C after 3 min idle with both servers off) is not
  enforced by the pool (#38): a chain runs `tools/r26-gate.sh --deadline HH:MM` after `serving-window.sh stop` and
  restores if it does not open. `tools/r26-gate.sh` was written 2026-10-13 after W-1013b opened warm; it opened the
  10-15 and 10-16 windows at 07:08 (median 46.0 °C).
- **Other sessions' CPU jobs.** Other sessions' CPU-heavy jobs belong on cores 10–14 at nice 19 during our windows
  (agreed with the [other project] session 2026-10-15 19:0x); unpinned ones can land on the run cores.
- **GPU health.** A GPU run with a median SM clock below 2,000 MHz or foreign GPU activity above 2 % of its budget
  voids its whole block (all arms of that instance × seed) and halts the run set.
- **Cap void (D-021 R25).** A GPU run whose busy SM median is below 2,300 MHz (`--cap-floor-mhz`) or that overlaps a
  fan-governor clock cap (2,200 MHz, which passes the 2,000 floor) is void (`gpu_cap`): its block is rerun once, after
  the fan reads AUTO, then skipped and listed. A low clock with the governor's cap off before and after and no cap
  interval in its log (an unknown lock) halts the pool at once.
- **cuOpt memory and foreign kills.** Under a harness cuOpt's CONFIRMED cap hit — an RMM error with the peak ≥ 0.8 × the
  limit, or a kill with the scope's Result oom-kill or maxrss ≥ 0.95 × MemoryMax — is void too (`cuopt_memory`, rerun
  once; M3's harness records it instead). A failure of SCIP or the EA, their caps included, stays an arm outcome. A
  SIGKILL nobody in the harness sent, at no cap, voids any arm's run (`external_kill`, rerun once).

### Validity of a run
- **Frozen configuration.** Test-set instances run only with the configuration whose hash equals `frozen-e2`'s, with
  `prereg-v1` an ancestor of HEAD. Tuning happens on the tuning set only, within the logged E4 budget.
- **Every incumbent checked.** Every incumbent vector of every process is stored and checked by the independent exact
  checker; only checked incumbents enter a metric. Exit codes of every process are checked; a crash is kept and
  truncated, never silently re-run.
- **Clock rule.** Before the barrier only parse, imports, CUDA context and kernel warm-up on a synthetic problem; any
  LP, presolve or evaluation of the instance's population after `t0`, in every arm.
  - A run that compiles a kernel after `t0` (`compiled_on_clock`) voids its block (D-002 v3.1).
  - Under a harness the runner checks it itself (D-021 R15): an entry of the project's CUDA JIT cache written after `t0`
    (the `index` file aside) or CuPy cache growth after `t0` voids the run (`compiled_on_clock`) and the pool reruns
    the block once.
- **Deterministic warm-up.** Before `t0` the warm-up runs:
  - every crossover and mutation operator on 1-, 2- and 3-child batches with the run's gene dtype (87bc759, 2026-10-09);
  - the elite kernels on three copies of one warm-up vector, so they do not depend on how many distinct members the
    warm-up population holds (6ae6a6f, 2026-10-08) — operators drawn at random cover only by chance;
  - (branch `r1-bsync`, 2026-10-14) a concatenate of 9 parts, every device array of a candidate set, one-row and mixed
    (CuPy 14.2 changes kernels above 8 parts).
  - A new kernel, operator or batch shape gets its own warm-up entry.
- **Cold-cache GPU test before the timed run.** Before the first timed run of new or changed GPU code, run the
  gpu-marked tests inside the window (lease held), including the cold-cache tests (`test_cupy_cold_cache_*` in
  `tests/test_ea_run.py`, `test_elite.py`, `test_lp_decode.py`, and for R1 / R2 `test_cuda_sched.py`,
  `test_lp_decode_gate.py`: an empty `CUPY_CACHE_DIR`, nothing may compile after `t0`).
  - The test must prove the run reached the new code, not only that nothing compiled. Incident 2026-10-08: a vacuous
    pass let a block compile on the clock, voided. On 2026-10-09 the test caught the block crossover's one-child chunk
    compiling after `t0` before any timed run.

### Code, process and conduct
- **No commits to `src/` or `configs/` while a pool can launch runs** (including a chained pool): every run's processes
  import the current source and the pool reads the config file when it plans a run (incidents 2026-10-06 15:15–16:35:
  one pool's runs spanned several code states; 16:54: an edited config halted a pool).
  - Develop in a git worktree on its own branch (`git worktree add -b <branch> ../i-trea-ii-<name>`; `dev-next` and
    `dev-backoff` were built that way).
  - Test there on the CPU only, on cores outside the run lane (`CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src
    OMP_NUM_THREADS=1 ../i-trea-ii/.venv/bin/python -m pytest -q -m "not gpu"`), and merge after the pool has ended.
- **Performance runs first (Berk 2026-10-13).** Box time goes to runs that measure the method (A9 against the solvers,
  development mechanisms where they can act), targeted first and wider only on a pre-stated hint (D-024). Calibration,
  harness and methodology runs only when something structural is broken, or to fill otherwise idle daytime hours.
- **Development changes: one variable, the test fixed first.** Each change gets a decision record whose mechanism
  check and outcome rules (per-instance bands from the control's seed-to-seed spread) are written before any run; the
  control is re-run in the same window; a rule that fails by the letter is recorded as a stated deviation, never
  silently re-read (D-012, D-016–D-018). A rule or hint is fixed, with its number, before the runs it reads, and a
  narrowed test gets its own rule (`vault/00-index/Home.md`).
- **Redis.** Only the project's own container `itrea-redis` on 127.0.0.1:6390, keys under `itrea:<run_id>:`, deleted
  at the end of each run. Never the host's other Redis instances.
- **No downloads** (instances, wheels, models) without Berk's explicit ask. No customer data (customer) anywhere.
- **No personal data to outside services.** Never send Berk's (or anyone's) e-mail address or other identifying data
  to an outside API; where an API asks for a contact address (Unpaywall, Crossref, OpenAlex), send none or a project
  placeholder (incident 2026-10-05, `vault/60-incidents/`).
- **Never `pkill -f` / `pgrep -f` from an agent shell**: the pattern is in the shell's own command line, so it matches
  (and kills) the shell itself (exit 144; incident 2026-10-06 01:4x). Stop a helper by the PID it wrote or by the file
  it holds or polls; wait on an output file, not on a process name.
- **Toy models only in tests**: `itrea.benchmarks.toy` or tiny generated MPS; never a solver or the EA on a real
  MIPLIB / PGLib / ML4CO instance inside a test (parsing one file for a shape check is fine).
- **Record everything.** Journal entries (`vault/20-journal/`, append-only, host local time — CEST, CET from
  2026-10-25), decision records D-NNN, a result note per experiment with failures included, incidents in
  `vault/60-incidents/`. Skills: `itrea-journal`, `itrea-result-note`.
- **Upstream reports and submissions** (issues, PRs, MIPLIB solutions, arXiv): drafts in `vault/70-upstream/`, filed
  only with Berk's yes for that one post.

## Writing rules

- **Plain, precise English; no hype.**
- **Never claim "first" or "novel".** `vault/40-literature/Prior art.md` (sweeps A–E, 91 works) records close prior art:
  - cuOpt's own GPU population with recombination runs beside cuOpt's internal CPU branch-and-bound with solutions
    flowing both ways, keeps infeasible members and protects the best feasible one;
  - ReXi runs local-search and LNS workers beside SCIP workers over a shared pool;
  - Puchinger, Raidl and Gruber (2005) and Nwana et al. (2005) exchange incumbents both ways between a heuristic and a
    B&B solver;
  - CHAP pools GPU and CPU heuristics' feasible and infeasible candidates;
  - IDEA reserves a share of the population for infeasible solutions.
  - None found has the whole combination; the differences are stated as differences.
- **No result claims ahead of the record.** Never write that I-TREA II "beats", "improves on" or "matches" anything
  until the corresponding experiment note in `vault/30-results/` says so with its numbers. Development results are
  descriptive; label them interim.
- **Names.** Name harnesses and models plainly (e.g. "Claude Code (Anthropic Claude Opus 5.5)"); no internal agent
  pseudonyms. AgentRA (Agent Research Assistant) — the AI agent that designs with the author and executes the
  experiments — is mainly Claude Opus 5.5, working through Claude Code.
- **Author line: Berk Orbay.**
- **No [organisation] in publications** (Berk, 2026-09-23; a standing rule of his, recorded outside this repository): [organisation]
  is not named in papers, the author line is his name only, and a public code link must not rely on the [organisation] org.
- **Citations.** Cite only works with a legally readable full text; paywalled works live only in the vault's paywalled
  list and the paper's paywalled appendix (skill `itrea-literature`).

### Updates to Berk
His rules, 2026-10-06 / 10-08 / 10-11:
- **Frequent, short, quantified:** effects with intervals, counts, p — not adjectives.
- **Performance first.** Each update opens with a short **Performance** block — Good news / Bad news about how the
  approach itself does (effects vs SCIP alone, cuOpt alone and the controls, weak-set numbers, targets met or missed;
  "nothing new measured" if so).
- **Then implementation:** a separate Good news / Bad news block for implementation, harness, box and bugs.
- **What counts as bad news:** a gap that comes with a concrete remedy is not bad news; bad news is a measured loss,
  failure or regression.

## Where things are

| what | where |
|---|---|
| every rule, the state at a glance, the code map | this file |
| current state, open issues, interim readings, development decisions, changelog (moved 2026-10-16) | the detail note §§ 7–10 |
| detail and history moved out of this file | `vault/00-index/Agent companion — detail and history.md` |
| the vault's front page, the plan, the question | `vault/00-index/Home.md`, `vault/00-index/Roadmap.md`, `vault/00-index/Research question and spec.md` |
| the newest session checkpoint (read it right after "At a glance" in a new session) | `vault/90-checkpoints/` — newest `2026-10-14-2350.md`; it predates the results of D-025–D-027 and everything from D-028 on |
| decisions | `vault/50-decisions/D-001` … `D-032` (each with § Result where it ran) |
| journal (append-only) | `vault/20-journal/YYYY-MM-DD.md` |
| every development run and reading | `vault/30-results/Development log.md` |
| every experiment, failures included | `vault/30-results/Experiment index.md` |
| the phase-1 results note (draft) | `vault/30-results/2026-10-17 Phase 1 — development until C1.md` |
| the external-harness measurements | `vault/30-results/2026-10-11 D-021 measurements M1–M3 and the interim calibration.md` |
| the C1 diagnosis (R1–R3) | `vault/30-results/2026-10-14 C1 diagnosis and improvement candidates.md` |
| design: algorithm, processes, traces and metrics, pool, datasets, compute (window log) | `vault/10-design/` |
| literature: prior art, paywalled list | `vault/40-literature/Prior art.md`, `vault/40-literature/Paywalled research.md` |
| incidents and lessons | `vault/60-incidents/Mistakes and incidents.md` |
| upstream drafts and records | `vault/70-upstream/` (`drafts/` not posted) |
| design reviews and consults | `vault/80-reviews/` |
| job files | `jobs/<set>.jsonl` |
| raw traces (not committed) | `traces/raw/<set>/`; voided runs in `traces/raw/<set>/void/` with `void.jsonl` |
| resource records (committed) | `traces/resources/` |
| benchmark figures | `traces/bench/` (e.g. `traces/bench/bsync/`) |
| run table | `datasets/runs.parquet`, `datasets/incumbents_checked.parquet` (rebuilt by `itrea build-runs`; not committed yet) |
| instance lists and sets | `datasets/dev_set.csv`, `datasets/classprobe_set.csv`, `datasets/mipfeas_strata.csv` |
| instances | `data/` (MIPLIB 2017 benchmark, PGLib-UC, ML4CO); `data/miplib/benchmark-v2.test`; GAMS's archive `data/mipfeas/gams-2026-09-16/` |
| pool logs and status | `logs/<job file>-<MMDD>.log` (from chains, e.g. `logs/d031-1016.log`; the pool's own default `logs/pool-<date>.log`), `logs/pool-status.json`, stop file `logs/pool.stop` |
| window chains | `logs/chain-*.sh` and their `.out` |
| saved readings | `logs/d030-reading.json` |
| the paper | `paper/` (`main.tex`, `sections/`, `refs.bib`, `generated/`) |
| skills | `.claude/skills/` |
| the cuOpt 26.10 nightly venv | `~/venvs/cuopt-2610n` |
| the project's CUDA JIT cache | `~/.cache/itrea/cuda` |
| the host's fan governor and thermal logs | `~/.local/state/thermal/` (`fan-state`, `rest`, `fan-governor.log`, `<YYYY-MM-DD>.log`) |
| issues | GitHub `Tideseed/i-trea-ii` (labels `phase:E0` … `phase:E6`, `dev`, `infra`, `paper`, `after-tuning`, `needs-berk`) |

## Changelog

Moved on 2026-10-16 to the detail note § 10 (dated milestones; day-by-day detail: `vault/30-results/Development log.md`
§ Changes and the journal).
