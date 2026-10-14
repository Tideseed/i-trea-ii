---
name: itrea-new-experiment
description: >
  Set up and launch an I-TREA II experiment: check it against the pre-registration, build the job file with make_jobs,
  freeze the configuration, run it under the pinned, capped pool runner inside a window, and watch it without
  disturbing it. Use for any timed run on candidate or pool instances (screening, calibration, E1–E5) and for any new
  experiment idea before code or runs are written.
---

# New I-TREA II experiment

## 1. Is it allowed, and is it pre-registered?
- Read `vault/50-decisions/D-002 Pre-registration — hypotheses, arms, analysis.md` (current version) and the newest
  checkpoint. Is this experiment in the plan (which phase, which hypothesis or gate)? If not, it is **exploratory**:
  say so in its id, journal entry and result note — or write a decision record first if it changes the plan.
- Order of gates (D-002 §1): toy tests pass → tag `prereg-v1` → screening → `pool-v1` → references → E0b → E1 gate →
  E4 → `frozen-e2` → E2/E3. Nothing timed on a candidate instance before `prereg-v1`; nothing on the test set before
  `frozen-e2`. `prereg-v1` is tagged (2026-10-06); screening stage 1 ran, stage 2 and the pool freeze are deferred (#1).
- Since 2026-10-06 the algorithm is developed on the development set and the class-probe instances (D-010, judged on
  the coupled arm, D-011). A development test is a decision record D-NNN written before any run: one variable, the
  mechanism and outcome rules with per-instance noise bands, and a control re-run in the same window under a fresh
  variant name (`rc1d`, `rc1e`) so earlier runs are not reused; amendments are dated and made before the first run
  (D-024 rule 2; incident 2026-10-13 15:16 — see "Never read a run before the rule is fixed" below for the recovery
  when a rule is found missing while runs execute).
- **Staged testing (D-024, Berk 2026-10-13: "Shall we do targeted and surgical tests? and only if we see hints of
  improvement we can go wider"; "always aim for performance runs unless something structural is buggy in our
  experimentation methodology"; "If the box is idle make the most of it").** Before writing the job file:
  - **Performance runs first.** Box time goes to runs that measure the method (A9 against the solvers, a development
    mechanism where it can act). Calibration, harness and methodology runs only for a structural bug, or to fill
    otherwise idle daytime hours (D-021's M4 was stopped after 19 of 30 runs on that rule; its remaining strata-0–1 runs
    come before v1's stage 1).
  - **Targeted.** Name, in the decision record and before any run, the instances where the change can act, with the
    trace evidence (D-022's amendment: in the 141 recorded cuOpt runs at 120 s on the 17 instances, cuOpt's own 1e-4
    "optimal" stop occurred only on fhnw-binpack4-48 9/9, neos-3004026-krka 10/10, neos-960392 3/7 — so R13 is tested
    on those three), or where the claim lives (D-013's weak cases). A test of a mechanism on every instance where it can
    act is a complete test, not a stage 1. State any assumption the selection rests on (D-022: cuOpt uses the MIP gaps
    only as a termination test — not verified) and let the wider guard carry its check.
  - **The hint is written before the runs start:** one one-sided threshold on the targeted runs' own data (e.g. v1
    stage 1's ratio of call-clock `pt_gams` A9 / A6F ≤ 0.95 on the panel + the 5 no-solution instances, D-024's
    addendum; C1's hint is the counterexample — its threshold, ratio of means ≤ 0.95 on the fixed 11, was fixed at
    15:28:42 after the itx-r13 pool had started at 15:16:11, with 7 of 58 runs done and none read: a near miss in
    `vault/60-incidents/Mistakes and incidents.md`), the fixed instance set it reads (never
    re-selected on the runs it judges), how failures and skipped blocks count, and **what it gates** (which line widens,
    which stage runs anyway — v1's guard-only stage runs whatever the hint says). A hint is **futility-only**: it gates
    widening and never declares success; the decisive reading comes later on all instances, the targeted ones
    included. Do not use an "improved > worse" count as a hint (under no effect it passes about half the time), nor a
    mean over a handful of floor-level instances (it reads noise): use a floor limb on ΔP and per-instance bands; for a
    no-harm test the floor limb is one-sided (D-022's clarification, fd75476), not D-018's two-sided |ΔP| ≤ 0.005.
  - **Never read a run before the rule is fixed.** The rule and the hint are written, with their numbers, before the
    job file that serves them is launched (D-024 rule 2). If a rule is found missing or wrong while runs execute — a
    recovery, not a permission — fix it before any run of the set is read, date the amendment, state how many runs are
    done and that none were read, and log the near miss in the incident log (D-022's second amendment: 7 of 58 done,
    none read). After any reading, a change is a stated deviation or a new record, never a silent re-reading.
  - **Breadth no decision reads is not run** (D-024 rule 4): A0 runs only on the instances a comparison reads (v1: the
    81 of H_cu ∪ W_SCIP unseen, the stage-1 instances first; with no hint only the 25 stage-1 instances, D-024 (f)).
- Timeline (D-019, GitHub milestones `C1 checkpoint (2026-10-23)` and `v1 (end of November 2026)`): development on cuOpt
  26.08 closes at C1, Fri 2026-10-23; v1 (A9 frozen) runs its confirmatory test and the MIPfeas run under prereg-v2
  (#32) on cuOpt 26.10 (#31; on 26.08 if 26.10 is not out by 2026-11-06, to be confirmed with Berk then).
- Daytime windows only: no I-TREA II window in 00:00–07:00 (Berk, 2026-10-10, "No nights", D-019; until then D-009
  asked for his yes for each night). Long run sets (MIPfeas ≈ 78 h) are split into daytime windows.
- Tuning happens on the tuning side of the pool only, within the logged E4 budget (≤ 12 configurations × 16
  instances × 120 s × 1 seed); D-010 §2 puts the development set in that place. Seeds: screening 900/901, tuning and
  development 0–99, confirmatory 1000/1001 — never mixed.

## 2. Configuration
- Main arm on the test set: `configs/frozen-e2.yaml` once it exists. `configs/itrea2-default.yaml` (9ac4a93bf93b) is the
  pre-registered starting configuration. Development defaults: **A3** → `configs/itrea2-dev-rc-ld.yaml` (f030e23909ad,
  lp_decode adopted, D-017); **A9** and its matched control **A3x** → `configs/itrea2-dev-rc-3way.yaml` (1d12ad490dd0;
  the runner adds `exchange.from_cuopt` for A9 and refuses `repair.continuous: lp_decode` in A9 / A3x until the
  two-core layout, #30); D-018 stage 1 → `itrea2-dev-rc-3way-bo1.yaml` (active) vs `-bo1s.yaml` (shadow control) — not
  adopted; D-023 → `itrea2-dev-rc-3way-boY.yaml` (phase Y only, `5c8b535fb245`) vs `-bo1s.yaml`, under the harness
  D-022 leaves (`make-jobs backoff2 --harness ext|r13-off`; A6F always ext); `ranks.rank1_mode` stays off (D-016 not
  adopted). The C1 diagnosis's R1 / R2 (D-025 / D-026; branch `r1-bsync`): variant files `-3way-[boY-]bsync.yaml`
  (fallbacks `-bnice`, `-ysync`) and `-3way-[boY-]bsync-ld[0].yaml` / `-[boY-]ld0.yaml`, each its base + one change;
  `make-jobs r1-bsync --control r13x|boY` is written only after D-023's verdict, `make-jobs r2-ld --base …` only after
  D-025's (with `--nice` / `--ea-cpu` from D-025's measurement); the control runs under the set's own label (`A9@r1ctl`,
  `A9@r2ctl`: the same run hash, a new id); read with `itrea d025-report` / `itrea d026-report`. Ablations:
  `configs/ablations/*.yaml` or `--set key=value` overrides on the base (the comment at the top of each ablation file
  gives the dotted keys; a job file carries them as `overrides`). Record the **config hash** (`EAConfig.hash()`); the
  runner refuses test-set instances unless it equals `frozen-e2`'s.
- Never edit a configuration file that a running or finished experiment used; make a new one.
- **No commits to `src/` or `configs/` while a pool, or a chained pool, can still launch runs**: every run imports the
  current source and reads its config file at launch (incidents 2026-10-06 15:15–16:35 and 16:54). Develop in a git
  worktree (`git worktree add ../i-trea-ii-<name> -b <branch>`); run its CPU tests there with
  `CUDA_VISIBLE_DEVICES="" PYTHONPATH=$PWD/src OMP_NUM_THREADS=1 taskset -c <cores> ../i-trea-ii/.venv/bin/python -m
  pytest -q -m "not gpu"` on cores no lane uses (lane g0 = X925 5–6 + A725 1–3, Redis 0, runner 4 and 10–14; with the
  default `--cpu-lanes 0`, e.g. 15–17); merge into main only after the pool ends.
- `backend` must be explicit per arm (`cupy` for A3/A5, `numpy` for A4); `auto` is not allowed in timed runs.

## 3. Job file
- Build it with the job builder, never by hand: `itrea make-jobs
  screen1|screen2|e1|e4|confirm|smoke|backoff0|backoff1|mipfeas-cal|itx-…` (`itrea.experiments.make_jobs`; `mipfeas-cal` =
  D-020's A6m calibration, run under `--layout full`, skill `itrea-window`); a development job file is written with
  `make_jobs.job(...)` and `make_jobs.write_jobs(...)`, one `variant` label per arm. Instance paths relative to the repository root, arm, seed,
  budget, config path and overrides; the pool applies the Williams arm order per block from hash(instance, seed,
  20261005) and runs blocks in random order with strata interleaved.
- D-021 (ITX-1) jobs: phase 1 `itrea make-jobs itx-m1 … itx-m5 | itx-r13` (D-022: `jobs/itx-r13.jsonl` narrowed
  before any run to 58 — the 3 R13 instances × {A9@r13, A9@r13x, A6F@r13x} + C1's fixed 11 × {A9@r13x, A6F@r13x},
  seeds 0–1 — plus `jobs/itx-r13b.jsonl`, the guard's 16 on the 4 remaining instances, same set `itx-r13`; follow-up
  `itx-r13f --instances …|all`); phase 2 `itx-v1 --tier T1..T5
  [--cuopt 26.10] --out F` (the v1 tiers; split `ext`, preset `itx1`, labels `<arm>@v1c<tag>`, sets `itx-t<k>` — a
  26.08 run gets `-c2608` in its set and variant, so it never shares a run id with 26.10), `itx-a0`, `itx-m6`,
  `itx-m6a`. A9's configuration is `configs/frozen-v2.yaml` (the C1 freeze, committed in the commit tagged
  `prereg-v2`); the writer refuses a tier with A9 while that file does not exist (`--a9-config` names another file for
  a dry run only — the pool refuses it). Their instance sets come only from the committed `datasets/mipfeas_strata.csv`
  (`tools/derive_mipfeas_strata.py --check` re-derives it; never edit it by hand: it is hashed in prereg-v2; its
  `exposed` column is R33's list — add any further pre-tag exposure there, by its constants, before the tag). R33:
  the pool refuses an external A9 job unless `prereg-v2` is an ancestor of HEAD, `configs/frozen-v2.yaml` equals the
  tagged file and the job's config hashes to it; every external job needs a clean tree under `src/` and `configs/`.
  `itx1` hashes A0's runs on their SCIP-side harness only (`scoped_hash`), so a change to cuOpt's settings before the
  tag leaves A0's runs standing. A job-level `priority` (integers, on every job of the file) orders block groups;
  `position` orders the arms inside a block.
- Commit the job file before the run; record its SHA-256 in the journal.

## 4. Run
- Open a window (skill `itrea-window`): lease, LLM servers stopped for confirmatory work, gates.
- **Before the first timed run on changed GPU code**, inside the window: `pytest -m gpu` on the touched modules,
  including the cold-cache tests (`test_cupy_cold_cache_*` in `tests/test_ea_run.py`, `test_elite.py`,
  `test_lp_decode.py`), then a short smoke run whose summary says `compiled_on_clock = false` (as on 2026-10-09: one
  A3 job, sing44, seed 9, 30 s, `configs/itrea2-dev-rc-3way.yaml` + override `repair.continuous: lp_decode`, variant
  `ld1smoke`, written with `make_jobs.job(...)` — not `make-jobs smoke`, which is for toy instances only). A
  cold-cache test must prove the timed run reached the new code, not only that nothing compiled (incident
  2026-10-08: D-016's cold-cache test passed vacuously and the first block was void). New kernels
  and shapes are warmed in `Engine.prepare`; the warm-up runs every crossover and mutation operator on 1/2/3-child
  batches (87bc759).
- Launch the pool runner **detached** from the Claude Code shell through a chain script (skill `itrea-window`;
  `python -m itrea.experiments.pool jobs/<file>.jsonl --layout pair --deadline HH:MM > logs/<id>.log`, `--dry-run`
  first): every process pinned (one timed solver per X925 core 5–9, 15–19; the partner footprint on the same 3 A725
  cores in A2 and A3; in A9 SCIP on X925 5, the EA on X925 6, cuOpt on A725 1–3; A3x = A9 without cuOpt),
  `MemoryMax` + `MemorySwapMax=0`, CuPy pool limit, `OMP_NUM_THREADS=1`, one GPU arm at a time.
- Never edit a script while it runs (bash reads incrementally): write a new file and move it into place.
- A block is never split across windows; if the window will end, stop between blocks.

## 5. Watch without disturbing
- Wait on artefacts, not process names: the pool's last log line,
  `until grep -qE ' pool [0-9-]+ (done|stopped|halted)| refused: ' logs/<id>.log; do sleep 60; done` (a refused pool
  logs `refused: …` and exits 2), or the chain's `.out` line, which also catches a crash. `logs/pool-status.json`
  still holds the previous pool's state until the new pool writes `running` or `refused` (a held runner lock writes
  no status).
  `pgrep -f`/`pkill -f` in a Claude Code shell match the shell's own command line.
- Check periodically: completed runs vs expected, exit codes in `run.json`, voided blocks, `compiled_on_clock` in every
  summary, MemAvailable, ACPI maximum, `nvidia-smi` for foreign processes. Stop at the first anomaly and write it down.
- No analysis with default thread counts while timed runs are going; analysis afterwards, single-threaded.

## 6. Afterwards
Close the window (skill `itrea-window`), run the analysis, write the result note (skill `itrea-result-note`) and the
index row, journal entry (skill `itrea-journal`). For a development test the analysis is the test's report, committed
before the pool ends (as for D-017 and D-018) — `itrea build-runs`, then `tools/d016_report.py`, `tools/d017_report.py`,
`itrea backoff-report --stage 0|1` (D-018) or `--stage 1y [--harness ext|r13-off]` (D-023), `itrea d022-report
[--followup]` (D-022 and C1's interim hint and guard), all with `OMP_NUM_THREADS=1` — and the result goes into the
Development log and the decision record's Result section (D-010 §3). A staged test reports its hint as written (present
/ no hint / incomplete) and what it gated; an interim reading is labelled interim and decides nothing (skill
`itrea-result-note`).
