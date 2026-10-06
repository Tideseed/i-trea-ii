# D-008 Thermal start gate by arm type (2026-10-06 02:5x, during the screening pass)

**Status:** decided by AgentRA (Claude Code) on evidence; changes a D-003 resource rule, not the pre-registered
analysis (D-002). Open to a superseding decision.

**Observed.** The screening pass started at 02:52 with 9 SCIP lanes (X925 cores 5–9, 15–18). Within a minute the
hottest ACPI zone read 82 °C and then held at **88–89 °C** with 8 lanes running (host thermal-watch log, every 30 s).
Under D-003's per-run start rule ("start only below 80 °C") a lane that finished could not start its next run while
the others kept the box at 88 °C; lanes would idle in waves, and a lane waiting longer than an hour halts the whole run
set. The X925 cores ran at their full **3.9 GHz** at 88–89 °C (`scaling_cur_freq` = `cpuinfo_max_freq` on all ten) —
no CPU throttling.

**Decision.** The per-run start threshold depends on whether the arm uses the GPU:
- **GPU arms** (any EA-on-CuPy or cuOpt process: A2, A3, A3r, A5, A5x0, A5y0, A6, H2, H3): start only below **80 °C**,
  as before — the rule protects the GPU's clock state between compared arms;
- **CPU-only arms** (SCREEN, A0, A0*, A1, H0, and A4, whose EA runs on NumPy): start below
  **88 °C**.
The safety rules are unchanged: pause the run set at ≥ 92 °C (in-flight blocks voided and requeued), resume below
80 °C; the host's thermal-watch stops experiment units at 98 °C. Every run record carries the temperatures at start
and end (`gates.snapshot`) and both thresholds.

**Why this does not bias the screening.** The screening measures SCIP alone on X925 cores; at 88–89 °C those cores were
not throttled, so a run started at 87 °C runs at the same clock as one started at 79 °C. Runs completed before the
change (started below 80 °C) and after it are comparable on that evidence.

**Mechanics.** `gates.THERMAL_WAIT_C_CPU = 88.0`; `pool._arm_uses_gpu(job)` picks the threshold. The stage-1 pool was
stopped (SIGTERM; its in-flight blocks were voided and requeued) and restarted on the committed change.

## Addendum 2026-10-06 05:00 — two hours of evidence
- 02:59–05:00, 8 lanes, CPU start gate 88 °C: 213 runs completed, **0 failed**; the box sat at **89–91 °C**
  (thermal-watch, 30 s) and touched **92.0 °C twice** (pool watchdog, 5 s) — pauses ended 03:32 and 04:52, each
  voiding the 8 in-flight blocks (16 runs, requeued and rerun in the same pass).
- X925 clocks at 05:00 (≈ 90 °C): 3.9 GHz on all 8 lanes — the measurement is unaffected; the cost is only the voided
  lane time (~1 pause per hour ≈ 8 runs).
- Not changed mid-pass (a restart voids the 8 in-flight runs, the same as one more pause). **For the next CPU-only
  window:** 7 lanes, or a CPU start gate of 86 °C, to keep the box below the 92 °C pause; decide on the stage-1 tail data.

## Addendum 2026-10-06 07:5x — stage 2 hit a pause storm
- Stage 2 (234 jobs: 142 MIPLIB by the rule + all 92 PGLib/ML4CO; larger, harder instances) on the same 8 lanes and
  88 °C CPU start gate: **8 thermal pauses between 06:08 and 06:32**, **64 runs voided**, 34 completed in 53 min before
  the 07:10 deadline. Stage 1 had 2 pauses in 3 h.
- **Next window (decided):** stage 2's remainder on **6 lanes**, CPU start gate 86 °C; if pauses still exceed one per
  hour, make the lane count adaptive (open a lane only while the box is below 86 °C, close one above 90 °C) before E1.
