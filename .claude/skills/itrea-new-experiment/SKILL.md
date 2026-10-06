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
  `frozen-e2`.
- Tuning happens on the tuning side of the pool only, within the logged E4 budget (≤ 12 configurations × 16
  instances × 120 s × 1 seed). Seeds: screening 900/901, tuning 0–99, confirmatory 1000/1001 — never mixed.

## 2. Configuration
- Main arm: `configs/frozen-e2.yaml` once it exists; before that `configs/itrea2-default.yaml`. Ablations:
  `configs/ablations/*.yaml` or `--set key=value` overrides on the base (the comment at the top of each ablation file
  gives the dotted keys). Record the **config hash** (`EAConfig.hash()`); the runner refuses test-set instances unless
  it equals `frozen-e2`'s.
- Never edit a configuration file that a running or finished experiment used; make a new one.
- `backend` must be explicit per arm (`cupy` for A3/A5, `numpy` for A4); `auto` is not allowed in timed runs.

## 3. Job file
- Build it with the job builder (`make_jobs`, being built in `itrea.experiments`) from the pool file, never by hand:
  instance paths relative to the repository root, arm, seed, budget, config path and hash, Williams arm order per
  block from hash(instance, seed, 20261005), blocks in random order with strata interleaved.
- Commit the job file before the run; record its SHA-256 in the journal.

## 4. Run
- Open a window (skill `itrea-window`): lease, LLM servers stopped for confirmatory work, gates.
- Launch the pool runner **detached** from the Claude Code shell (`setsid nohup … > logs/<id>.log 2>&1 &`): every
  process pinned (one timed solver per X925 core 5–9, 15–19; the partner footprint on the same 3 A725 cores in A2 and
  A3), `MemoryMax` + `MemorySwapMax=0`, CuPy pool limit, `OMP_NUM_THREADS=1`, one GPU arm at a time.
- Never edit a script while it runs (bash reads incrementally): write a new file and move it into place.
- A block is never split across windows; if the window will end, stop between blocks.

## 5. Watch without disturbing
- Wait on artefacts, not process names: `until [ -f traces/raw/<set>/DONE ]; do sleep 60; done`, or tail the run log.
  `pgrep -f`/`pkill -f` in a Claude Code shell match the shell's own command line.
- Check periodically: completed runs vs expected, exit codes in `run.json`, voided blocks, MemAvailable, ACPI maximum,
  `nvidia-smi` for foreign processes. Stop at the first anomaly and write it down.
- No analysis with default thread counts while timed runs are going; analysis afterwards, single-threaded.

## 6. Afterwards
Close the window (skill `itrea-window`), run the analysis, write the result note (skill `itrea-result-note`) and the
index row, journal entry (skill `itrea-journal`).
