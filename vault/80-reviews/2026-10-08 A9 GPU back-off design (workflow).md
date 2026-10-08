# A9 GPU back-off design (workflow, 2026-10-08) — EA yields the GPU to cuOpt

Produced by a design workflow (a code/evidence map, three independent designs — contribution-gated duty cycle, fixed duty cycle, resource-level yield — two adversarial judges, synthesis; Claude Code subagents, read-only, 2026-10-08 10:3x–11:4x). Both judges chose the contribution-gated duty cycle as the base, with corrections (patience calibrated in A9, not on A3/A3x; ca, not sing44, as the discovery-phase test; a pre-stated fallback when chromaticindex's target fails). Prerequisite of #26 and of A9's next measurement. Status: **design only — nothing built or run.** The synthesis follows verbatim (headings demoted); its draft number D-017 is provisional (decision numbers are assigned when a record is written).

## D-017 (draft): the EA backs off the GPU beside cuOpt in arm A9 by a contribution-gated duty cycle

**Status.** Design only. Nothing is built or run. Development only (D-010); a confirmatory use needs prereg-v2 first. Design by Claude Code (Anthropic Claude Opus 5.5) from three candidate designs and two judge reviews. Every number below was re-read from `traces/raw/{a9pilot,dev}` and `datasets/runs.parquet` unless it is marked otherwise.

**Code state.** Line numbers are at HEAD `6ae6a6f` (D-016 plus its warm-up fix). The D-016 development test (`jobs/dev-cl1.jsonl`, pool started 11:26) is still running. This test's base configuration is fixed only after dev-cl1 reports. If D-016 is adopted, every arm carries it, controls included; otherwise the base is `configs/itrea2-dev-rc-3way.yaml`. Either way there is one variable.

## Summary

The EA keeps running every generation and every exchange tick. Only its **GPU duty** changes: the share of wall time in which it may keep work on the GPU.

1. **Phase Y.** While no process has an incumbent, the EA runs at a reduced duty, D_Y. Phase Y ends at once when SCIP or cuOpt posts an incumbent, which the EA learns from a new non-consuming peek during its repair calls. It also ends when the EA finds one itself, or at `yield_until_s`.
2. **Phase P.** The EA starts at full duty. While it makes no material lead, it steps down a ladder of 1 → ½ → ¼ (→ ⅛). How long it waits before stepping down depends on its state:
   - a long patience while it has no feasible point of its own (the ca case);
   - a short patience once it has one but has not yet beaten the system's best (the kumeu and radiationm case);
   - a patience taken from its own lead rate after its first lead.
   Any material lead restores full duty.
3. **Enforcement.** A host-side token bucket sleeps at two points where the EA's GPU queue is empty: between FJ launches (ls_fj_gpu.py:895 syncs there) and between generations, after `bk.synchronize()`.
4. **Calibration first.** A fixed-duty calibration (stage 0) sets D_Y, the floor and both patiences from measurements taken in A9, under rules stated here. It stops the plan if no duty restores cuOpt's first incumbent on chromaticindex1024-7.

---

## 1. The rule

### 1.1 Duty and clocks
- Level ℓ ∈ {0..L}, with duty D_ℓ = 2^−ℓ.
- Busy clock: b = (now − t0) − (seconds slept by the governor), counting sleeps between generations and inside calls.
- Every patience is counted in busy seconds, so the same EA work is needed at every duty.
- The repair clock also runs on b (section 4), so pausing does not change how it chooses its level.

### 1.2 Contribution events
- **LEAD (load-bearing).** In `_new_incumbent` (engine.py:1573), take an incumbent with `src == "self"`, i.e. not an unchanged SCIP or cuOpt vector (engine.py:1577, `_is_solver_find` at 513).
  - Before `best_key` is overwritten (engine.py:1582), set ref = min(best_key, inc_external).
  - dg = 1 if ref = +∞. Otherwise dg = min((ref − key) / max(|ref|, |key|, |dual_key| if finite, 1e-9), 1). This is the repair clock's gap unit (repair_clock.py:187-190).
  - The lead counts as material if dg ≥ `materiality` = 1e-3.
  - A polished copy of an injected cuOpt row has the same key as ref, so it gives dg = 0. That is why kumeu and radiationm show own feasible points but no leads.
- **ASSIST.** An accepted SCIP verdict on an EA post, not on an unchanged cuOpt row, whose note has `improved` true (scip_run.py:351-352).
  - The gain is (primal_before → primal_after) in key units, and must be ≥ 1e-3.
  - The note is already parsed for `row` in `_verdicts` (engine.py:2045-2051).
  - In the pilot this changes no decision. It is kept because the EA's value reaches SCIP through posts on item_placement-type instances.

### 1.3 Phase Y (no incumbent anywhere)
- **Entry.** At t0, when inc_external = +∞ and the EA has no incumbent. Duty is D_Y, applied from the first pace point, so it covers the initial repair at engine.py:1014. That repair runs before the first exchange tick at 1024.
- **Exit** on whichever comes first, after which ℓ = 0 and phase P starts:
  - **Peek:** a finite `incumbent` in SCIP's `info` or cuOpt's `info_cuopt`, read without consuming it at most every `exchange.poll_s` = 0.25 s, inside calls and between generations.
  - **Tick:** an exchange tick that makes inc_external finite.
  - **Own find:** the EA's own first incumbent.
- **Expiry.** At `yield_until_s` with still no incumbent: ℓ = 0 and phase Y is never re-entered.

### 1.4 Phase P (an incumbent exists)
- The EA starts at ℓ0. Escalation ℓ0 → ℓ1 happens when b − b_last ≥ τ. Each further level follows after `step_s` = 4 more busy seconds without a contribution, down to ℓ = L.
- b_last is reset at: phase-P start; every level change; every material contribution; and the EA's transition to "own feasible".
- τ depends on the EA's state:

| EA state | test | τ |
|---|---|---|
| **search** | no feasible member of its own: `_rc_own_feasible()` (engine.py:1284) false; usable without the repair clock | `patience_search_s` |
| **own** | an own feasible member, no material lead yet | `patience_own_s` |
| **led** | at least one material lead | clamp(κ/λ̂, `patience_min_s` = 2, `patience_max_s` = 40), κ = 3; λ̂ = material events per busy second, decayed with half-life 20 busy-s |

- **Resume:** any material lead or assist sets ℓ = 0 at once.
- The level is updated at every pace point, so escalation also happens inside long calls.

### 1.5 Enforcement (token bucket, credit c in seconds)
- Busy time b lowers credit: c −= b·(1 − D). Sleep time s raises it: c += s·D.
- When c < −`burst_s` (0.25), sleep min(−c/D, `max_sleep_s` = 4) in slices of at most 0.25 s (STOP_POLL_S, engine.py:220).
- Long-run duty = D. At ¼ the cycle is about 0.33 s of running and 1 s asleep. At ⅛ it is about 0.29 s and 2 s.
- There is no separate probe: at the floor the EA still runs, and a lead resets ℓ to 0.
- **Pace point 1, between generations.** In `step()` before `tg` (engine.py:1037), so `max_gen_s` excludes the pause.
  - First `bk.synchronize()` (backend.py:52-55), so queued survival and evaluate kernels do not run during the "pause". The sync wait counts as busy.
  - Each slice calls `_should_stop()` and runs `_exchange_in()` when `next_poll` is due. `_lps_use()` (engine.py:1855) is skipped while asleep, so LP-stream work does not pile up per wall second.
  - If a stop arrived, `step()` returns without a generation. The main loop handles this (engine.py:887-888).
- **Pace point 2, inside `FJGpu.run` between launches.** After the horizon and deadline checks (ls_fj_gpu.py:906-913) and before the stop poll (914-916), so a stop that arrives during a sleep is seen at once. Also before each chunk (before 848).
  - Inside a call the governor only peeks (in phase Y) and checks for a stop. There is no exchange tick, so the cutoff stays constant within a generation, as the repair clock assumes.

### 1.6 Wake
- **Trigger:** during a between-generation sleep, a tick shows a partner improvement — inc_external fell by ≥ materiality, or a new solver or cuOpt batch is pending.
- **Action:** the sleep ends now and the next generation runs, so cuOpt rows are injected and relayed to SCIP (the relay at engine.py:1596).
- Credit is forgiven (c = 0) at most once per `wake_min_s` = 5 s. The level is not changed.

### 1.7 Modes
- `active`: the rule.
- `shadow`: bookkeeping only. No sync, no sleep, no in-call `pace` argument. It logs the would-be phase, level and escalations at generation boundaries. The would-be phase-Y exit is computed afterwards from the partners' event files. This is the control arm.
- `forced`: a fixed duty `forced_duty` from t0 to the end, with no phase logic. Stage 0 only, never a default.

### 1.8 Parameters (section `backoff`, off by default, out of the hash while off)

| field | default (prior) | set by |
|---|---|---|
| enabled | false | |
| mode | active | `shadow` for the control, `forced` for stage 0 |
| forced_duty | 1.0 | stage 0 only |
| yield_level | 2 (D_Y = ¼) | stage 0 rule (a) |
| levels (L) | 2 (floor ¼) | stage 0 rule (b): 1–3 |
| yield_until_s | 30 | stage 0 rule (c) |
| patience_search_s | 75 | stage 0 rule (d) |
| patience_own_s | 30 | stage 0 rule (e) |
| materiality | 1e-3 | fixed (the repair clock's unit) |
| patience_kappa / min_s / max_s | 3 / 2 / 40 | fixed |
| half_life_s | 20 (busy) | fixed |
| step_s | 4 | fixed |
| burst_s / max_sleep_s / slice_s | 0.25 / 4 / 0.25 | fixed |
| wake_min_s | 5 | fixed |
| assist | true | fixed |

The governor is inert unless `exchange.from_cuopt` is true (engine.py:927), i.e. arm A9 (pair.py:169). A3x, which shares the engine, stays bit-identical.

---

## 2. Why (evidence)

### 2.1 The cost is cuOpt time-sliced against a busy EA
- **sing44 (cleanest evidence).** cuOpt reaches 363.59M at 49.03 s alone (A6), 50.90 s beside SCIP (A2) and 96.95 s in A9: a 1.98× stretch. The trajectories are shared only up to about 362.31M (judge 1).
- **The EA's footprint is GPU only.** The EA has its own X925 core and cuOpt keeps its A725 cores (pair.py:523-527).
- **GPU utilisation (gpu_health `mean_util`):** 91.2–96.0 % in A9 against 71.3–88.8 % for cuOpt alone.
- **Generation-0 repair is slower beside cuOpt** (A3x → A9):

| instance | A3x | A9 |
|---|---|---|
| chromaticindex1024-7 | 25.69 s | 48.31 s |
| kumeu | 0.34 s | 0.53 s |
| sing44 | 0.33 s | 0.50 s |
| radiationm | 0.27 s | 0.43 s |

  That is 1.5–1.9× per busy second.
- **optopt A4** (`optopt/vault/30-results/A4 GPU contention.md`, re-read):

| beside cuOpt | P(30) |
|---|---|
| nothing (alone) | 0.226 |
| idle resident model | 0.240 |
| one call per 5 s | 0.235 |
| continuous inference | 0.452, worse in 10/10 |

  "Rule: sequential or low duty cycle only." Duty is the lever.

### 2.2 What is at stake, per instance (A9 pilot, seed 0)
- **chromaticindex1024-7.** cuOpt posts exactly one incumbent, objective 4.0, the optimum: at 8.56 / 9.44 s alone, 6.50 s in A2 and 65.35 s in A9.
  - So P on this instance is the first-incumbent time ÷ 120 (0.545 vs 0.071). Only phase Y can move it; the ladder plays no role on the main target.
  - The EA never had a feasible point of its own there (2 generations).
- **kumeu.** The loss is cuOpt's burst right after its first incumbent.
  - Alone: −12036.4 by 6.45 s, next improvement at 30.40 s.
  - In A9: the burst stopped at −11366.6 at 4.87 s, next improvement at 56.62 s.
  - The difference is 0.148 vs 0.097–0.107, almost all of it in the first 7 s. Phase Y ends at cuOpt's first incumbent (4.67 s in A9), so this rule does not protect that burst (section 7).
- **ca.** cuOpt's first incumbents (24.06 s, ≈5.96M) have gap ≈ 0.99 against a final ≈34k.
  - Speeding cuOpt's first ca incumbent therefore buys almost nothing in P. Phase Y on ca is pure cost to the EA.
  - The EA's first ca point is its gap-1 % point: 32.8–36.9 s in A3@rc1, seeds 0–3, run without cuOpt.
- **Floor instances** (P < 0.012): krka, fhnw, radiationm. fhnw's first cuOpt incumbent is the optimum: 0.56 s alone vs 1.03 s in A9.
- **Instances where cuOpt carries and keeps improving late** (A6, seed 0):

| instance | cuOpt alone (A6) P | best A3 variant P | when cuOpt improves |
|---|---|---|---|
| comp21-2idx | 0.498 | 0.651 | 18 of 38 improvements at 15–60 s |
| neos-4763324-toguru | 0.583 | 0.771 | 26 of 28 at 60–120 s |

  These are where the ladder matters. Neither has run in A9.

### 2.3 Where the EA contributes: own-feasible and first-lead times
Leads are computed from incumbent events against min(EA best, partner incumbents posted before t). "Own feasible" is the first rclock event with phase `post`, written at the end of that generation.

| run | partner's first incumbent (s) | EA own feasible (s) | first material lead (s) | material leads |
|---|---|---|---|---|
| item_placement A9 s0 | 0.02 | 0.49 | 0.68 | 227 (last 86.3 s) |
| sing44 A9 s0 | 0.29 | 19.52 | 34.57 (dg 0.91) | 7 (last 88.9 s) |
| kumeu A9 s0 | 4.67 | 7.82 | none | 0 |
| radiationm A9 s0 | 0.29 | 3.75 | none | 0 |
| chromaticindex / fhnw / krka A9 s0 | 65.35 / 1.03 / 0.33 | none | none | 0 |
| ca A3@rc1 s0–s3 (SCIP partner only) | 33.9–46.0 | at the lead | 32.8–36.9 | 1–3 |
| loue A3@rc1 s0, s1 | 3.3–3.8 | ≈4 | 2.1–2.2 | 26–31 |
| neos-960392 A3@rc1 s0, s1 | 0.07–0.08 | 1.6 | 1.6 | 6–10 (last 10–22 s) |
| glass4 A3@rc1 s0, s1 | 0.02 | 10.0 / 11.3 | 47.7 / 15.6 | 1 / 2 |

What this means for the patience:
- **One patience cannot serve all.** ca needs a long search patience: its lead comes 33–37 s into the run without cuOpt, and in A9 it will be slower — 1.5–1.9× from contention, plus a cutoff row armed by cuOpt's incumbent before the EA has a feasible point of its own.
- **The "own" patience must cover the own-feasible → lead lag:** about 15 s on sing44 in A9, 4.4 s / 37.7 s on glass4 (A3), ≤ 2 s elsewhere.
- **It must also escalate kumeu and radiationm before ~40 s.** With the split, kumeu escalates at about 7.8 + 30 ≈ 38 busy-s and radiationm at about 34. A single patience of 75 would wait until ~80 s.
- **Generation counts cannot serve as the unit.** sing44's first lead took 36–51 generations in A3 against 10 in A9.
- **Lane-steps per busy second** vary by orders of magnitude between instances and with the cutoff row. So both patiences are measured in A9 (stage 0), not carried over from A3.

### 2.4 sing44 is low-stakes for the system
- SCIP alone scores 0.518. The pointwise best of A0 and A6 is 0.497, against A9's 0.477 (judge 1).
- The EA's whole marginal there is about 0.02–0.06. Design 1's "+0.24 if throttled" was refuted.
- sing44 is judged on the system, with SCIP as its floor. It is not an EA-protection case.

### 2.5 Why duty and not other levers
- The two CUDA contexts are time-sliced (no MPS daemon; checked by design 3, not re-checked here).
- Stream priorities act only inside one context, and green contexts only inside one process.
- MPS and the time-slice length are system changes (root, parity across arms, fault isolation) and need Berk (section 7).
- Between launches is the only point inside a long call where the EA's queue is empty.
- Lane state lives in device arrays and the RNG is seeded per (seed, individual, call) (ls_fj_gpu.py:55-57). So without a horizon or cap, pacing changes when rows are repaired, not what they become.

---

## 3. Taken from the other designs and from the judges

1. **Stage-0 fixed-duty calibration** (from design 3, with design 2's dose ladder).
   - 50-ms launches only, so launch length is not a second variable.
   - It has a stop rule.
   - It sets D_Y, the floor, `yield_until_s` and both patiences by rules stated here.
2. **`bk.synchronize()` before the between-generation pause, and busy time for the repair clock** (design 2): `_rc_decide` at engine.py:1282 and `_rc_end` at 1331. Otherwise its decay (repair_clock.py:359-364) and o = gen_seconds − call_s (291-292) would turn its level choice into a second variable.
3. **Mechanism checks that do not trust the governor's own bookkeeping** (designs 2 and 3): EA lane-steps and generations per wall second, gpu_health `mean_util`, and the sing44 stretch over the shared prefix. The HiGHS LP-point arrival time is reported as a confound, because a sleeping EA frees core 6.
4. **Same-day A6 in every block** (design 2): the cuOpt-speed and noise reference.
5. **Shadow-mode control** (design 3; judge 2): it shows whether the governor would have escalated before each first lead in the control's own trajectory.
6. **Guards re-weighted to where the EA carries the system** (both judges): item_placement, ca, loue, neos-960392, crna and glass4 are binding; sing44 is judged on the system; floor instances are judged in seconds.
7. **ca diagnosis from events** (judge 2): a ca failure is classed by what the events show — escalation before the first lead, or phase-Y length. Not by "phase Y ≥ 15 s", which is always true on ca.
8. **sing44's cutoff-row cost goes to its own issue** (design 3's diagnosis; judge 1): 3 generations by 20 s in A9 vs about 35 in A3x. It is not mixed into this test.

---

## 4. Implementation plan

No CUDA kernel edits. With `enabled: false`, nothing below runs.

**1. `src/itrea/config.py`**
- New `@dataclass BackoffCfg` after `RepairClockCfg` (260-287), holding the fields of 1.8, and `EAConfig.backoff` after `repair_clock` (308).
- `hash()`: pop `backoff` while disabled, beside 325-331. Every recorded hash then stands (9ac4a93bf93b, 1d12ad490dd0, the pilot's 0f011a2094a2, D-016's).
- `validate()` (beside 450-464):
  - mode ∈ {active, shadow, forced}; 1 ≤ levels ≤ 3; 0 ≤ yield_level ≤ levels; 0 < forced_duty ≤ 1; every *_s > 0; patience_min_s ≤ patience_max_s;
  - `enabled` requires `exchange.enabled` and `repair.method == "fj"`.
- Regenerate `configs/itrea2-default.yaml` (`tests/test_configs.py` compares it with the code; the hash is unchanged).
- New config files over the base of the day:
  - `…-bo0.yaml` (shadow);
  - `…-bof50/-bof25/-bof12.yaml` (forced, stage 0);
  - `…-bo1.yaml` (active, with the stage-0 values; written and committed after stage 0, before stage 1).

**2. New `src/itrea/ea/backoff.py`.** `Backoff(cfg, t0)`, host-only like `repair_clock.py`: no backend, no random number, an injectable clock.
- Methods:
  - `phase_y_exit(now, why)`, `own_feasible(now)`, `contribution(now, dg, kind, ref)`;
  - `update(now)` → a level-change record or None;
  - `sleep_due(now)` (0 in shadow mode), `slept(s, site)`, `wake(now)`;
  - `busy(now)`, `gen_fields()`, `record()`.
- It keeps the level, phase, state, credit, λ̂ sums, and paused seconds split between generations and inside calls.

**3. `src/itrea/ea/engine.py`**
- **`begin()`**, after the repair clock (1000-1006): `self.bo = Backoff(...)` if the section is enabled and `self.from_cuopt` (927), else None. Write the start-up description beside the repair clock's at 682.
- **`step()`** (1033-1040): `if self.bo is not None and mode != "shadow": self._bo_between()`, then `if self._should_stop(): return` before `tg` (1037).
- **`_bo_between()`:** sync, `update`, `sleep_due`, then a slice loop under `with self.phase("paused")`. Each slice runs `_should_stop`, `_exchange_in` when due, and the wake test. It sets `self._bo_sleeping` for the `_lps_use` skip at 1855.
- **`_bo_pace()`:** the in-call callback. Sliced sleep, `_should_stop`, `_bo_peek()` while in phase Y, `update`; returns the seconds slept.
- **`_bo_peek()`:** `mb.info.peek()` and `mb.info_cuopt.peek()`.
- **`_ls_run`:** pass `pace=self._bo_pace` to `ls.run` at 1254 only in active and forced modes.
- **Busy clock `_bo_busy_now()`** (= wall when there is no governor):
  - `_rc_decide` (1282) and `_rc_end` (1331) use it;
  - store `_rc_b_gen` beside `_rc_t_gen` (1004, 1047);
  - pass gen_seconds = busy now − `_rc_b_gen`.
- **`_step`**, beside 1052: one `_rc_own_feasible()` check per generation feeds `bo.own_feasible` on its first True. The method needs no repair clock.
- **`_new_incumbent`:** compute ref and dg before 1582; call `bo.contribution`; write a `contrib` event {t, kind, dg, id, ref, key, busy_s}.
- **`_verdicts`** (2038): parse `improved`, `primal_before` and `primal_after` from the note (already parsed at 2045-2051) and credit assists.
- **`_gen_record`** (2088): bo_level, bo_phase, bo_state, bo_duty, bo_busy_s, bo_paused_gen_s, bo_paused_call_s, bo_wakes, bo_contrib, ls_steps (Σ lane-steps since the last record).
- **`finish()`**, beside `repair_clock` at 2182: a `backoff` block with seconds per level (wall and busy), phase-Y length and exit reason, contributions by kind, wakes, pauses, and the would-be timeline in shadow mode.
- **Events:** `backoff` {t, gen, mode, phase, state, level, duty, reason, busy_s, paused_s, lam, patience_s} on every change.
- **Module docstring:** a "Back-off" paragraph beside the repair clock's (117) and A9's (136).

**4. `src/itrea/ea/ls_fj_gpu.py`, `FJGpu.run` (762)**
- New argument `pace=None` and a local `paused`, called at the two points in 1.5.
- Subtract `paused` wherever the call measures its own time:
  - `late` (854) and the deadline test (909) become busy-time comparisons;
  - profile `t` (855, 902) and `end` (943);
  - `out["seconds"]` and `last_profile["seconds"]` (959-961) become busy time; add `paused_s` and `wall_s`.
- `dt` (899) already excludes the pause (`tl` is set at 880).
- With `pace=None`, every value is bit-identical.

**5. `src/itrea/ea/repair.py`, `LocalSearch.run` (135-144).** Forward `pace` for `fj` only; ValueError otherwise, like horizon and cap_s at 136-138. `ls_fj_cpu.FJCpu.run` gets the same argument in its slice loop, for CPU wiring tests only.

**6. Mailboxes.** `peek()` on `InfoChannel` (mailbox.py:114-143): read the JSON without touching `_seen` or `last_seq`. The same on `redis_mailbox._Info` (113-128): `r.get` without `_seen`.

**7. Analysis**
- `build_runs.py` columns: bo_busy_share, bo_paused_s, phase-Y length and exit, seconds per level, n_lead, n_assist, first-lead busy time, and the would-be first escalation (shadow).
- **New `src/itrea/analysis/backoff.py` and `itrea backoff-report`**, committed before stage 0. It applies the stage-0 rules and the stage-1 criteria below, computes the sing44 shared-prefix stretch (median t_arm(v)/t_A6(v) over shared cuOpt objectives with t_A6 ≥ 1 s and at least 5 shared values, else "n/a"), and prints the verdict.

**8. Tests (toy models only)**
- **`tests/test_backoff.py`** (CPU, fake clock):
  - long-run duty within ±2 % at each level;
  - escalation times for each state;
  - the own-feasible reset;
  - κ/λ̂ clamp;
  - phase-Y exit and expiry;
  - one forgiven wake per 5 s;
  - the max_sleep cap;
  - forced and shadow modes.
- **Engine on NumPy:**
  - off → bit-identical on the four reference scenarios, including `tests/fixtures/engine_reference_traffic.npz`;
  - shadow → identical genealogy to off;
  - enabled with from_cuopt false → inert;
  - active with scripted cuOpt traffic: sleeps happen, a stop file is honoured within one slice, the deadline is never overslept, the repair clock's gen_s and o_s exclude pauses, leads and assists are credited.
- **GPU-marked (`tests/test_fj_gpu.py`), run in a GPU window before any timed run:**
  - a pace that sleeps 0 gives the same arrays as no pace;
  - a sleeping pace gives the same X, status and steps without horizon or cap;
  - with cap_s, the cut happens at busy ≈ cap, not wall ≈ cap;
  - a stop during a sleep ends the call as `stopped`.

---

## 5. Development test

### 5.1 Stage 0: calibration (mechanism only, not an adoption test)
- **Window and conditions.** Lease, both LLM servers stopped (A9's memory gate), thermal and GPU-health gates, skills `itrea-window` and `itrea-new-experiment`. Seed 2 throughout, which is not a test seed. 120 s per run.
- **Duty response:** chromaticindex1024-7, sing44 and comp21-2idx × {A6, bo0 (shadow = duty 1), bof50, bof25, bof12} = 15 runs, interleaved per instance.
- **First-lead timing in A9 at full duty:** ca and glass4 × bo0 = 2 runs.
- **Total:** 17 runs, about 40 min.

**Readouts**
- S_first: cuOpt's first chromaticindex incumbent ÷ same-day A6's.
- sing44 S: the shared-prefix stretch.
- comp21: cuOpt-inside P against A6.
- Measured EA duty, and EA lane-steps per wall second ÷ bo0's.
- gpu_health `mean_util`.
- Model check: S against 1/(1 − d/2), i.e. 1.33 / 1.14 / 1.07 at ½ / ¼ / ⅛.

**Rules (applied by `itrea backoff-report`; nothing is chosen by eye)**
- **Sanity first.** For the forced runs to count, each must have measured duty within ±0.03 of its setting, and lane-steps per wall second within [0.75, 1.5] × d of bo0's.
- **(a) D_Y** = the largest d ∈ {½, ¼, ⅛} with S_first ≤ 2.0. **If none qualifies, stop:** no stage 1. The pacing mechanism cannot protect cuOpt's first incumbent; report it. MPS is the next candidate, and it needs Berk.
- **(b) Floor D_min** = the largest d ∈ {½, ¼, ⅛} with sing44 S ≤ 1.15 (or "n/a") and comp21 cuOpt-inside P ≤ A6 + max(0.02, A6's two-seed spread). If none qualifies, D_min = ⅛, noted as a shortfall. Then L = log2(1/D_min).
- **(c) `yield_until_s`** = max(30, ⌈2 × chromaticindex's first cuOpt incumbent at D_Y⌉).
- **(d) `patience_search_s`** = clamp(⌈1.25 × (t_first_lead − t_first_partner)⌉, 40, 90), from the ca bo0 run (busy = wall at full duty). It is 40 if ca makes no material lead in 120 s.
- **(e) `patience_own_s`** = clamp(⌈1.25 × the larger own-feasible → first-lead lag⌉, 20, 40), over the sing44 and glass4 bo0 runs. Instances without a lead are left out; 30 if neither leads.

### 5.2 Stage 1: one variable, the governor active vs shadow

**Arms**
- **A9@bo0 (control):** base of the day + `backoff` in shadow mode. Its hash differs from the plain development default, but its genealogy is proven identical (section 4, tests).
- **A9@bo1:** the same file with mode active and the stage-0 values.
- **A6:** cuOpt alone, same day, in every block. M2 and the sing44 stretch need a same-seed A6, and its two seeds give the band.
- Blocks = instance × seed, the three arms in Williams order, blocks never split. Seeds 0 and 1. 120 s.

**Instances (14)**

| role | instances | why |
|---|---|---|
| target (phase Y) | chromaticindex1024-7 | one cuOpt incumbent = optimum; 65.35 s in A9 vs 8.56 / 9.44 s alone |
| ladder targets | neos-3656078-kumeu, comp21-2idx, neos-4763324-toguru | cuOpt carries and improves after 15 s; the EA is expected to make no leads |
| guards (EA carries; cuOpt alone weak or worse than A3) | item_placement_9903, ca__Scenario400_reserves_1, neos-3024952-loue, neos-960392, neos-2657525-crna, glass4, sing44 | A6 0.31–0.78 against A3 variants 0.02–0.58 |
| floor | neos-3004026-krka, fhnw-binpack4-48, radiationm18-12-05 | P < 0.012, judged in seconds |

- **Size.** 84 runs. krka and fhnw end in about 10–15 s on cuOpt's proof; the rest take about 2.2 min each. About 2.8 h plus D-009 rest breaks, over **two windows**: window 1 = seed-0 blocks, window 2 = seed-1 blocks. The comparison is within-block, so the window affects only the seed-to-seed band.
- **Not included:**
  - rail01 (no A3 variant ever found a feasible point; cuOpt's one incumbent comes at 107 s);
  - cryptanalysiskb128n5obj16 and neos-2746589-doon (nobody finds anything).

**Mechanism criteria (must hold, or the outcome is not read as the governor's)**
- **M1 (phase Y and peek).** On every instance where a partner posts, phase Y ends within 0.5 s of that partner's first posted incumbent (`cuopt.events` / `solver.events` t).
- **M2 (target).** On chromaticindex, cuOpt-inside-A9's first incumbent ≤ 2 × same-day same-seed A6's, on both seeds.
- **M3 (protection)**, on item_placement, loue, neos-960392 and crna:
  - material leads ≥ 0.8 × the control's;
  - the first material lead no later than the control's + max(2 s, 10 %);
  - on item_placement, ℓ0 for ≥ 90 % of the wall time up to the control's last material lead.
  - M3 carries the item_placement guard, whose seed band is about 0.68 (A3@rc1 0.134–0.417).
- **M4 (back-off happens, checked independently)**, on kumeu, comp21, toguru and radiationm, wherever the control makes no material lead after 30 s:
  - bo1's busy share over 60–120 s ≤ D_min + 0.1;
  - EA lane-steps per wall second over 60–120 s ≤ 0.5 × the control's;
  - gpu_health `mean_util` below the control's on ≥ 3 of the 4.
- **M5 (accounting):**
  - per generation, busy gen_s + paused = wall within 2 %; repair-clock o_s ≥ 0;
  - cap hits per call ≤ the control's;
  - no process overruns its budget by more than one launch + 0.25 s, and a stop is honoured within that;
  - every incumbent checked; GPU health ok.
- **M6 (attribution, reported):** from the control's shadow log — would-be escalation times against first leads on every guard instance.

**Outcome criteria** (bounded d of bo1 vs bo0, seed means; positive = bo1 better)
- **T (target):** chromaticindex d ≥ +0.30 and mean P ≤ 0.30.
- **G (binding guards):**
  - item_placement, ca, loue, crna, glass4, sing44: each d ≥ −band_i;
  - neos-960392 (P ≈ 0.03, where d is unstable): ΔP ≥ −max(0.01, the control's P spread);
  - the guard-set mean d ≥ −0.05;
  - every guard where bo0 beats same-day A6 by d ≥ +0.15 is still beaten by bo1 by ≥ +0.15.
- **Ladder targets:** kumeu, comp21 and toguru each d ≥ −band_i. Their mean d is reported.
- **F (floor):** |ΔP| ≤ 0.005 on krka, fhnw and radiationm. Also reported: seconds to cuOpt's first or optimal incumbent (fhnw, krka) and to within 0.1 % of A6's final best (radiationm).
- **Mean guard:** mean d over the 11 non-floor instances ≥ 0.
- **D-013 report:** for each of SCIP and cuOpt, the weak-set results of both A9 arms against A0 and same-day A6, with the mean as the guard. Also reported: the relay latency (cuOpt post → SCIP incumbent) and the HiGHS LP-point arrival time.
- **Bands:** band_i = max(0.05, the larger same-day seed-0 vs seed-1 bounded difference of bo0 or of A6 on instance i), per the D-012 addendum-2 rule.

**Decision**
- **Adopt** as A9's development default if T, G, F, the mean guard and M1–M5 hold.
- **Noise.** A guard that fails by the letter while the events show the governor never left ℓ0 and phase Y lasted < 1 s there is noise: reported, not counted (as in D-012).
- **If ca's guard fails**, classify from the `backoff` and `contrib` events:
  - **(P)** bo1 escalated before the EA's first ca lead → the next test sets `patience_search_s` = 1.25 × this test's control lag, nothing else changed.
  - **(Y)** no escalation before the first lead and phase Y ≥ 10 s → the phase-Y trade (2.2). The next variable is D_Y one step up, accepted only if stage 0 showed chromaticindex S_first ≤ 3 at that duty. Otherwise this is recorded as "no single phase-Y duty serves both"; the deferred progress exit (section 7) is the next candidate.
- **If glass4's or sing44's guard fails** with an escalation before their first lead: the next test sets `patience_own_s` + 10 (≤ 50).
- **If T fails with M2 failing:** duty does not restore cuOpt at run level, although stage 0 said it would. Stop and report; MPS needs Berk.
- **Inside the band:** seeds 3 and 4 on those instances, same rules.

**Predictions** (stated before any run; not criteria)
- chromaticindex: cuOpt's first incumbent at about 12–19 s, P ≈ 0.10–0.17, d ≈ +0.7.
- kumeu: P ≈ 0.13–0.145. The burst is not protected; relief comes only from first-incumbent timing and the late improvement.
- comp21 and toguru: d ≥ 0, small.
- ca: d between −0.25 and 0. Phase Y costs the EA about 0.75 × 24 s ≈ 18 busy seconds at D_Y = ¼ (≈ 12 at ½). At the judges' ≈0.0046 P per second of delay (more if cuOpt's ≈0.99-gap early incumbents dominate the shifted window), that is about +0.08 P (+0.055 at ½). This is the most likely guard failure.
- item_placement, loue, neos-960392, crna: ≈ 0.
- glass4: 0 to −0.05.
- sing44: ≈ 0; SCIP holds the system.
- fhnw: cuOpt's first incumbent moves from 1.03 s toward 0.56–0.8 s. krka and radiationm: unchanged.
- Fair-share prediction for stage 0: sing44 S ≈ 1.33 / 1.14 / 1.07 at ½ / ¼ / ⅛, so the floor is predicted at ¼. chromaticindex is non-linear (7.6× in the pilot), so D_Y may need ⅛.

---

## 6. Risks
1. **Phase Y costs ca by construction** (section 5 predictions). The online signals cannot separate ca from chromaticindex before a partner posts: both have no incumbent and a falling least violation. The guard is binding, and the fallback is diagnosed from events.
2. **kumeu's main loss stays.** It is cuOpt's burst at 4.7–6.5 s, after phase Y ends.
3. **A lead that comes after the floor is reached is delayed by 1/D_min.** A glass4-s0-type lag (37.7 s in A3, likely 55–70 busy seconds in A9) beyond `patience_own_s` (≤ 40) can be badly delayed. glass4 is a binding guard for this reason. Keeping the floor at the largest qualifying duty limits the damage.
4. **Patiences and duties come from one seed per instance** (stage 0, seed 2), with a 1.25× margin. A9's EA speed also depends on when cuOpt arms the cutoff row.
5. **The dose–response is inferred from one stage-0 seed.** Time-slicing of 50-ms launches may hurt cuOpt beyond the EA's busy share; M2 and M4 detect this but do not explain it.
6. **Accounting.** A missed busy-time subtraction would make the repair clock (cap, decay, probes) a second variable, or trigger cap hits. Tests and M5 cover this. Gen-event phase seconds are CUDA-event intervals that include in-call sleeps; analyses subtract `bo_paused_call_s`.
7. **False leads.** inc_external is read only at ticks, so a "lead" can beat a cuOpt post the EA has not read yet. The cost is one spurious reset to ℓ0, bounded and logged in `contrib` with ref.
8. **Relay latency.** Wakes forgive credit at most once per 5 s, so on long-generation instances a cuOpt row can wait before reaching SCIP. Reported.
9. **Confounds.** Sleeping frees X925 core 6 for the HiGHS LP worker (lp events reported) and lowers heat. Both are part of the treatment and are logged.
10. **The control is in shadow mode.** Genealogy identity is proven on fixtures, not on instances. The peek is a cheap read and shadow mode has no in-call hook.
11. **Code state.** dev-cl1 is still running. Building before its verdict, or using a stale control, would mix two variables.

## 7. Rejected and deferred
- **Fixed duty as the rule (design 2).** It throttles the EA where it carries the system and cuOpt is not harmed: item_placement, where cuOpt scored 0.620 inside A9 vs 0.666 alone. Kept as the stage-0 instrument (`forced` mode).
- **Credit-graded duty with 5-ms launches (design 3).**
  - It starves ca's discovery (u ≤ 0.5 until the EA earns credit, ≈ +0.2–0.3 P by the judges' estimate).
  - It leaves in-call idle in gen_seconds (repair_clock.py:291-292), so the repair clock's levels move with u.
  - It bundles shorter launches with duty.
  - It adds an NVML sampler to every GPU run.
  - Kept from it: stage 0, the peek, and shadow mode.
- **Holding the EA during cuOpt's post-first-incumbent burst** (to recover kumeu). It costs cuOpt's weak set:
  - loue: 241 cuOpt improvements in 0–5 s, EA lead ≈ 2 s;
  - crna: 21 improvements in 0–5 s, EA lead ≈ 0.3 s.
  D-013 puts those first.
- **Feasibility-progress exit from phase Y (deferred).** Tested here on the traces: "least violation halves within 10 busy seconds" fires on ca at 7–9 s (most A3 variants), never on chromaticindex, rail01 or cryptanalysis, but also on doon (harmless: nobody finds anything there). At D_Y = ¼ that is 28–36 s of wall time, after cuOpt's first ca incumbent at 24 s, so it would not help ca. It is the candidate fallback only if a higher D_Y is chosen. It was fitted on one instance.
- **A single patience, or patience in generations or lane-steps** (section 2.3).
- **NVML per-process SM sampling.** A separate instrumentation commit for all GPU arms alike; not in this test.
- **MPS, stream priority, green contexts, the GPU time-slice length.** Respectively: needs Berk, parity across arms and fault-isolation work; acts within one context only; acts within one process only; needs root and affects the LLM servers.
- **Shorter launches (`launch_ms`).** A separate variable; only if stage 0 and stage 1 leave a latency gap.

## 8. Records
- **Vault:** a D-017 decision record (this note), a Development-log entry per stage, and a journal entry (`itrea-journal`).
- **GitHub (Tideseed/i-trea-ii):** one issue for the back-off, and a separate issue for sing44's cutoff-row cost in A9 (3 generations by 20 s vs about 35 in A3x).
- **Roadmap:** a line that D-017 follows D-016's verdict.
- **Commit order:** the stage-0 rules and `itrea backoff-report` are committed before any run.
## Judges (verbatim)

### Judge 1

Best: Contribution-gated duty cycle ("backoff"). Of the three, it is the only rule that protects the EA's discovery phase, which is where its value lies on cuOpt's weak set: ca, loue and neos-960392. Those count most under D-013. It also yields where the EA has measurably made no material lead: krka, fhnw, kumeu (158 post-phase generations with zero repair-clock gain events), radiationm and chromaticindex. It is one flag, off by default and out of the hash while off, with no effect when from_cuopt is false (engine.py:891). Its mechanism criteria M1-M5 are stated before the run.

Two corrections before adoption.

1. Re-base patience0 on ca, not sing44. The sing44 "+0.24" figure is refuted: SCIP alone reaches 18.66M by 33 s, so a 32-s delay of the EA costs the system only about 0.01-0.06. ca's 32.8-36.9 s first leads were measured without cuOpt, so ca in A9 is the real test of the 40-busy-s window.
2. Add a fallback, stated before the run, for when the chromaticindex target M2 fails.

Design 2 is the cleanest calibration but by construction gives up item_placement and ca. Design 3 recovers cuOpt most broadly but starves ca's discovery: about +0.2-0.3 P, on an instance its predictions leave out.

Graft: 1. From design 3: run its stage 0 before the design 1 test. That is a forced-duty calibration on chromaticindex1024-7, sing44 (its cuOpt trajectory is shared up to 362.31M, which makes it a stretch instrument) and kumeu, at forced duty 1, 0.5, 0.25 and 0.1 with 50-ms launches only, so launch length does not become a second variable. Use its stop rule: if no duty brings S_first ≤ 2 on chromaticindex, do not run the governor test. Use the measured curve to set yield_level and the floor instead of guessing 1/4 and 1/16. This also gives design 1 its missing chromaticindex fallback: raise yield_level, or lengthen yield_until_s past cuOpt's measured first incumbent at that duty.
2. From design 2: call bk.synchronize() before the between-generation pause. Put the repair clock on a work clock in _rc_decide and _rc_end, so its decay and probe timing (repair_clock.py:150-156, 359-364) see busy time and not sleeps.
3. From designs 2 and 3: independent mechanism checks that do not rely on the governor's own bookkeeping. These are EA generations and lane-steps per wall-second, gpu_health mean utilisation, and, optionally, NVML per-process SM share. Add the sing44 shared-prefix stretch, the median t_A9/t_A6 over shared cuOpt objectives. Report the HiGHS LP-point arrival time as a confound, since a sleeping EA frees core 6.
4. From design 2: add a same-day A6 run per block as the cuOpt-speed and noise reference.
5. Reframe the guards. item_placement (A0 0.631 → A9 0.362) and ca are where the EA's value is real. On sing44 the EA's marginal is only about 0.02-0.06, because SCIP alone gets 0.518 and the pointwise min(A0, A6) is 0.497. Judge sing44 on the system with SCIP as the floor, not as an EA-protection case. Make ca's guard binding and log ca's first EA lead in busy-seconds, to test patience0 directly.
6. From design 3's diagnosis: the EA's sing44 slowness in A9 comes from the cutoff row armed by cuOpt's incumbent (about 3 generations by 20 s against about 35 in A3x), not from GPU contention. File it as a separate lever in its own issue; do not mix it into this test.

- **Contribution-gated duty cycle ("backoff"): phase Y + duty ladder 1 to 1/16 driven by material EA leads, patience 40 busy-s** — effect 5, measurability 7, risk (10 = low) 6. Fatal flaws: None fatal. The code claims check out at 667a360: begin() 874-994, from_cuopt 891, rclock 964-969, initial repair 976-978 before the first tick at 988, _step 1007-1016, _new_incumbent 1529-1557, _verdicts 1994-2024, _lps_use 1810-1811, config 249-274/296/318-319, FJGpu.run sync 895, deadline 853/854/909, profile 855/902/943, seconds 959. The verdict note fields are real (scip_run.py:312, 351-352), and peek() is trivial on both transports (mailbox.py:127-143, redis_mailbox.py:123-128). D-016 is now committed (HEAD 346a164), so the line shifts apply.

Weaknesses:
(1) The headline rationale for patience0 = 40 s is wrong. The design says a 20-s patience would cost sing44 about +0.24 in P. I recomputed it from the traces: shifting the EA's own sing44 leads by 32 s and taking the pointwise minimum with SCIP alone (A0 s0: 18.66M at 33.1 s, 8.21M at 84.8 s) and cuOpt gives P = 0.489-0.496. Without the shift it is 0.433, and actual A9 is 0.477. So the system cost is about 0.01-0.06, not 0.24, because SCIP's own trajectory caps it. More generally, A9's sing44 win over cuOpt alone (0.477 vs 0.726) is mostly SCIP's: A0 0.518, min(A0, A6) 0.497. The EA's marginal there is about 0.02-0.06.
(2) The real support for patience0 is ca: first EA leads at 32.8-36.9 s in A3@rc1 (seeds 0-3). Those runs had no cuOpt. In A9 the EA is slower under both cuOpt's GPU share and a cutoff row armed by cuOpt's incumbent. On sing44 the EA ran about 3 generations by 20 s in A9 against about 35 in A3x. So ca's first lead may come after the 40 busy-s window, which is counted from the phase-Y exit (a level change resets b_last). Then the governor escalates and delays the one lead that matters.
(3) If duty 1/4 for 30 s does not bring chromaticindex's first cuOpt incumbent in, phase Y expires, the EA returns to full duty, and the pilot situation repeats. The 7.6x delay is non-linear: no fair-share model predicts it. The test pre-states fallbacks only for ca and sing44, none for the main target (M2).
(4) The between-generation sleep at the top of _step has no bk.synchronize(). Designs 2 and 3 include one, and I did not verify that the queue is empty there (the frontier and survival device work happens just before).
(5) The repair clock's decide and _decay stay on wall time (engine.py:1246; repair_clock.py:359-364). Sleeps thin its evidence, so its level choice becomes a second-order part of the treatment.
(6) There are 14 parameters, tuned on one A9 seed plus A3 timings. Relief on kumeu and radiationm starts only at about 45 s, and there is no cuOpt relief on sing44, by design.
- **gpu_share: fixed GPU duty cycle f for the A9 EA, dose ladder {1, 0.5, 0.25}** — effect 3, measurability 9, risk (10 = low) 8. Fatal flaws: This is not a back-off rule. It throttles the EA everywhere, including where it is the contributor and cuOpt is not harmed (item_placement: cuOpt 0.620 inside A9 vs 0.666 alone). So by construction it cannot meet "keep the EA's gains".

Expected harm on cuOpt's weak set:
- ca: the EA's post-first gap is about 0.006 against cuOpt's about 0.56, so each second of delay to the EA's first ca lead (about 35 s uncontended) costs about 0.0046 P. f = 0.5 is about +0.15; at f = 0.25 the lead falls past 120 s, and ca reverts to cuOpt's 0.65.
- item_placement P tracks generations.
- The design itself expects the guard to fail. That makes it a calibration, not a rule.

The guard is weak where it matters most. item_placement's band is about 0.68 (rc1 seed spread). The guard-set mean of at least -0.05 over six instances can absorb an item_placement loss of about 0.2-0.3, and "still beats A6 by 0.15" passes at P up to about 0.56. So it could adopt an f that gives away item_placement's EA value.

Minor inaccuracies:
- The "69 identical incumbent objectives" claim is off. The counts are 69 in A6, A2 and A9, but the trajectories diverge after 362.31M (finals: 14.25M in A6, 13.94M in A2, 10.55M in A9). The shared-prefix stretch metric still works (363.59M at 49.03 / 50.90 / 96.95 s, verified).
- The resume-on-final-status is nearly inert (cuopt_run.py:236-238; pair.py DEFINITIVE at line 74).

What it gets right:
- the bk.synchronize() before the boundary pause;
- the work clock for _rc_decide and _rc_end;
- an independent mechanism check (generations/s, GPU utilisation: verified 91-96 % in A9 vs 72-90 % for A6);
- reporting the LP-arrival confound.
- **gpu_yield: duty-paced yield with credit-weighted u, PRE u=0.1, 5-ms launches while yielding, staged (forced-u calibration, then controller)** — effect 4, measurability 6, risk (10 = low) 5. Fatal flaws: (1) It starves the EA's discovery on ca-like instances, which are cuOpt's weak set and D-013's primary target, and its predictions leave ca out. On ca:
- SCIP never has an incumbent (A0: 1.0), and cuOpt's first comes at about 24 s, so PRE holds u = 0.1 that long (about 2.4 busy-s).
- After that, GRADED gives u ≤ 0.5 until the EA earns credit: the quiet prior is 0.005/0.01, cuOpt's credits push u toward 0.1, and the first finite K* is credited to nobody.
- The EA needs about 33-37 busy-s for its first ca lead (A3@rc1). That puts the lead at about 85-110 s wall.
- At about 0.0046 P per second of delay (the EA's post-first gap is about 0.006, cuOpt's about 0.56), that costs roughly +0.2-0.3 P on ca.
- The guard would catch it, most likely by rejecting the rule.

(2) SCIP's credit only moves K*. EA work that reaches the system through SCIP (EA posts that SCIP polishes, as on item_placement with 387 SCIP incumbents) earns the EA nothing, which lowers u where the EA does matter. There is no assist credit.

(3) Stage 1 is not one variable. launch_ms_yield = 5 ms comes bundled with the duty. Stage-0 rule (b) can set launch_ms_full = 5, which changes the EA even at u = 1 on every A9 instance. Up to four parameters are reset by stage 0. Under hardware time-slicing (or optopt's bandwidth reading: A4, 'SM clock unchanged, bandwidth contention'), launch length may not matter at all.

(4) The repair clock's decide, decay and probe stay on wall time (engine.py:1246; repair_clock.py:150-156, 359-364). Only gen_seconds is corrected.

(5) It has the largest new surface: passive second readers, launch_ms mutated mid-call, an NVML per-process sampler in gpu_health (optional, but runner-side code shared by all GPU arms), and three modes.

What it gets right:
- the code claims (LAUNCH_MS_MIN at ls_fj_gpu.py:97, one block per lane at 886, the sync at 895, cuOpt reporting only via GetSolutionCallback at cuopt_run.py:166-208, the final status at 236-239);
- its diagnosis that sing44's EA slowness in A9 is mainly the cutoff armed by cuOpt's incumbent (A9 about 3 generations by 20 s vs about 35 in A3x, inc_external finite from generation 0), not contention;
- its point that sing44's downside is bounded by SCIP;
- its stage-0 forced-u calibration with a stop rule, which is the best mechanism test of the three.

### Judge 2

Best: Base: the contribution-gated duty cycle (design 1), with its patience and fallback fixed before any run.

**Why it is the best base**
- It is the only design that, by construction, leaves the EA at full duty where it carries the system. item_placement: A9 0.362 vs the no-EA portfolio of A0 and A6 at 0.629.
- It still yields during the pre-incumbent window that decides chromaticindex. cuOpt alone has P 0.071 with its first incumbent at 8.56 s, so P there is essentially the first-incumbent time divided by 120.
- Off by default, out of the hash, pace=None unchanged, inert without from_cuopt.

**What is weaker about the other two**
- gpu_share is the cleanest one-variable dose-response test, but a poor rule: it taxes item_placement, neos-960392 and ca.
- gpu_yield recovers more of cuOpt's speed but has a repair-clock accounting bug (in-call idle left in gen_seconds; repair_clock.py:291-292). It also endangers the EA's ramp on ca and adds NVML sampling to every GPU run.

**What design 1 gets wrong**
- Its patience is in busy seconds, calibrated on A3/A3x arms that run without cuOpt.
- Inside A9 the EA is 1.5-1.9x slower per busy second. Identical gen-0 repair: chromaticindex 25.69 s in A3x vs 48.31 s in A9; kumeu 0.34 vs 0.53; sing44 0.33 vs 0.50.
- Its sing44 rationale overstates the stake. The portfolio without the EA is 0.497 vs A9's 0.477, so the EA's total marginal there is about 0.02 in P, not 0.24.
- Its ca fallback cannot tell a phase-Y loss from a patience loss.

**Disclosure.** I accidentally created /tmp/judge2_engine_667.py (a git-show copy of engine.py) and deleted it within the next call. No other file was written, nothing was run on the GPU, and all reads used git show or stdin heredocs.

Graft: 1. **Count patience in EA work, not busy seconds.** Use lane-steps (FJGpu.run already returns per-lane steps; the rclock call records keep them), so first-lead budgets measured in A3 transfer to A9. Alternatively, add a feasibility-phase progress signal (best_viol falling across generations) as a contribution, so a ca-type ramp is not throttled before the EA's first, near-optimal incumbent. At the least, scale the 40-s patience by the measured A9/A3x slowdown of 1.5-1.9x.

2. **Fix the pre-stated ca branch.** Branch on what the `backoff` events show: the level at the EA's first own feasible point, and whether escalation came before it. Not on 'phase Y >= 15 s', which is always true on ca.

3. **From gpu_yield: run a stage-0 forced-duty calibration first.** Use sing44, whose cuOpt trajectory is deterministic (69 identical incumbent objectives across A6/A2/A9), as a stretch instrument, plus chromaticindex. That is 14 runs. It measures the dose-response that design 1 admits is unmeasured (optopt A4 has only about 0.6 % and 100 % duty), and sets the ladder's duties and phase-Y level from data. Also take gpu_yield's shadow mode: log the would-be level inside the control arm, for attribution.

4. **From gpu_share:**
   - bk.synchronize() before the boundary sleep (backend.py:52);
   - a work clock for rclock.decide and its half-life (repair_clock.py:269 is wall time);
   - the M1 check, independent of the governor's bookkeeping: EA generations and lane-steps per wall second, and GPU-health utilisation;
   - the analysis tool committed before the runs;
   - a same-day A6 reference arm;
   - floor instances (krka, fhnw, radiationm) judged in seconds to the first or optimal incumbent, not in d.

5. **Re-weight the guards to where the EA actually carries the system:** item_placement, ca, loue, neos-960392 (SCIP alone 0.63-1.0, A3 0.03-0.35). sing44 is low-stakes either way.

6. **Optional, from gpu_yield and gated on stage 0:** short launches (5 ms, the existing LAUNCH_MS_MIN) while yielding, only if stage 0 shows they cut chromaticindex's first-incumbent delay by at least 25 %.

- **Contribution-gated duty cycle for the A9 EA ("backoff": phase Y yield, then duty ladder 1 to 1/16 driven by material leads, 40-busy-s patience)** — effect 6, measurability 6, risk (10 = low) 6. Fatal flaws: Nothing that makes it wrong or untestable. Line references check out at 667a360: engine.py 513-516, 976-978, 988, 1009-1016, 1218, 1290-1297, 1529-1557 (best_key updated at 1538), 1810-1811, 1994-2024 and 2072-2076; ls_fj_gpu.py 847-850, 853-855, 895, 899, 902, 909-916, 943 and 959; mailbox.py 114-143; redis_mailbox.py 113-128; repair_clock.py 189-190; scip_run.py 312/352. Exchange ticks during the sleep only append to pending (_receive, _lp_arrived at 1976, _lps_use at 1942); every repair runs inside the generation, so ticking while asleep is safe. Five serious problems:

(1) The patience is calibrated on arms without cuOpt on the GPU. The 40 busy seconds come from ca in A3@rc1 (32.8-36.9 s) and from fhnw/krka in A3x. Inside A9 the EA runs 1.5-1.9x slower per busy second. The identical gen-0 repair took 25.69 s in A3x vs 48.31 s in A9 on chromaticindex; kumeu 0.34 vs 0.53, sing44 0.33 vs 0.50, radiationm 0.27 vs 0.43. So ca's ramp needs roughly 50-70 busy seconds in A9, but the ladder escalates at about 6 (phase Y) + 40 = 46. On ca SCIP alone scores 1.0, A6 0.647 and A3 0.296, and the EA's first incumbent is also its gap-1 % point (t_first = t_gap1). It is the most valuable guard instance, and the EA would be throttled just before its lead.

(2) The pre-stated ca fallback ('phase Y ran >= 15 s, so retest with yield_level 1') always fires on ca, because cuOpt's first incumbent comes at 24.06 s. It therefore misattributes a loss caused by the patience. This is a measurability flaw.

(3) The sing44 stake is misquoted. The pointwise-best portfolio of A0 and A6 (no EA) gives P = 0.497, against A9's 0.477. A delayed sing44 lead can therefore cost at most about 0.02-0.03 in P, not +0.24. The 40-s patience protects little and gives up most of the kumeu relief.

(4) The between-generation sleep has no bk.synchronize(), which design 2 has. CuPy is asynchronous and _Phases syncs only in collect() (engine.py about 470-473), so queued kernels may run during the 'yield'. M4 reads the governor's own busy share and cannot detect this.

(5) The repair clock's half-life stays in wall seconds (repair_clock.py:269, decide() at engine.py:1246), so sleeps thin its evidence.
- **gpu_share: fixed GPU duty cycle f for the EA beside cuOpt (A9), dose ladder {1, 0.5, 0.25}** — effect 3, measurability 9, risk (10 = low) 7. Fatal flaws: Nothing wrong or untestable. Code claims check out: the pause after 914-916 and before 917-920 keeps dt clean, because tl is reset at 880. The pair.py:523-535 core layout, the DEFINITIVE stop at pair.py:74 and ~906, and cuOpt's final info at cuopt_run.py:236-238 are all as described, so the resume condition is inert, as the design says. Its gen_seconds = wall - (paused_s - _rc_p0) correctly removes in-call pauses from the repair clock's o (repair_clock.py:291-292).

As a rule it taxes the EA everywhere by construction:
- item_placement: the EA made 1,411 self incumbents and cuOpt was unharmed (0.620 inside A9 vs 0.666 alone), so the throttle is pure cost there.
- neos-960392: A3 P is about 0.03, so bounded d swings on tiny absolute changes.
- ca: the EA's first incumbent is also its near-optimal one (32.7-38.0 s uncontended, about 1.5-1.9x longer inside A9); at f = 0.5 it lands near the end of the budget or misses it.

Guard G will very likely fail at both f, so the probable verdict is 'reject; the curves calibrate a gated rule'. That makes it the best calibration experiment and the weakest back-off rule.

Minor:
- It edits FJCpu.run (A4's repair path) only to test the wiring; pace=None keeps that path unchanged.
- 88 runs over two windows is the most expensive test of the three.
- **Duty-paced GPU yield (gpu_yield): contribution-graded duty u from decayed credit, PRE state u=0.1, 5-ms launches while yielding** — effect 5, measurability 6, risk (10 = low) 4. Fatal flaws: (1) An accounting bug that turns the test into two variables. _rc_end passes 'gen_seconds minus this generation's boundary idle' only. The FJ profiles become busy-time, but in-call idle stays in gen_seconds. repair_clock.end_generation computes o = gen_seconds - call_s and T = o + tsum (repair_clock.py:291-292), so every level's T grows by roughly (1/u - 1) x the call time. Repair is about 90 % of EA time, so most idle is in-call. The repair clock's level choice then moves with u, which is a second variable. Fixable; design 1's and design 2's formulas get it right.

(2) The EA's ramp-up is not protected. PRE holds u = 0.1 until the first finite incumbent; after that, quiet periods give u = 0.5 and cuOpt bursts push it toward 0.1. On ca (SCIP alone 1.0, cuOpt's first at 24.06 s, the EA's first = its gap-1 % point at 32.7-38.0 s uncontended, about 1.5-1.9x longer in A9) the EA's decisive first incumbent would move to roughly 80 s or later. The predictions omit ca, loue and neos-960392, exactly the instances where the EA is the contributor.

(3) It is not inert for arms without cuOpt: it adds a 1-Hz NVML per-process sampler to gpu_health for every GPU run and changes run.json.

(4) Stage 0 sets four parameters from one seed per arm. Stage 1 bundles the controller with 5-ms launches, so it is not one mechanism.

Code claims otherwise check out:
- The RNG is seeded per (seed, individual, call) (ls_fj_gpu.py:55-57, _k_init at 875), so pacing without a horizon or cap does not change results.
- LAUNCH_MS_MIN = 5 is at line 97.
- launch_ms is read once at 842.
- InfoChannel is latest-wins (mailbox.py:127-143), so a second passive reader on the same file is sound.

