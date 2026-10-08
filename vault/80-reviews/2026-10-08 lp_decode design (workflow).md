# lp_decode design (workflow, 2026-10-08) — issue #27

Produced by a design workflow (a code map, two independent designs — pre-survival / post-survival — and an adversarial judge with synthesis; Claude Code subagents, read-only, 2026-10-08 11:0x–11:5x) for issue #27, after the cuOpt review ([[2026-10-08 cuOpt population and diversity — what transfers]], item 6). Status: **design only — nothing built or run.** The decision record (D-017) is written from this before any run; its control depends on D-016's outcome (risk 8 below). The synthesis follows verbatim (headings demoted).

## D-017 lp_decode: recommended design after the adversarial review (issue #27)

**Anchors.** Line numbers are at HEAD `6ae6a6f` (2026-10-08 11:26). HEAD moved from `346a164` during this review, because another session committed the D-016 warm-up fix. Between `346a164` and `6ae6a6f`, engine.py lines after ~663 shift by +3 to +8. config.py, seeding.py, ranks.py, population.py and lineage.py did not change.

**Evidence labels.** Every number below is one of three kinds:
- **[probe]**: my own toy-LP probes. highspy 1.15.1, core 14, nice 19, 1 thread, random LPs of 1,500×800 and 600×300, nothing written to the repository.
- **[trace]**: read from the A3@rc1 seed 0–1 `ea.summary.json` / `ea.events.jsonl` in `traces/raw/dev/`.
- **[design]**: taken from a design's "code map", not checked here. These are: ca cold solve 1.47 s with presolve / 8.54 s without; warm 0.36–0.41 s; readModel 1.85 s; `feasibilityRelaxation` 3.9–4.2 s; stored-vector gains ca −0.75 % and sing44 −17.8 %.

## 1. Verdict

Neither design is adopted as written. The recommendation combines parts of both:
- **From Design 2:**
  - the post-survival pick site over the whole candidate set;
  - the genealogy semantics;
  - the activation predicate;
  - restoring the basis after a non-optimal solve;
  - the stub-decoder warm-up.
- **From Design 1:**
  - screening, caching and budget split done inside the worker;
  - two new config fields kept out of the hash;
  - polling the worker at the pick site;
  - the runner refusing single-core arms;
  - a separate time limit for the first ("priming") solve.
- **Corrections:**
  - leave presolve at its default;
  - copy rows to the host with no gather kernel;
  - force the warm-up stub to run;
  - make the GPU test non-vacuous;
  - check P equality at run time;
  - make the ca mechanism criterion tolerant of scarcity.

## 2. Claims ledger

**Design 1 (pre-survival, children only, sibling genealogy)**

| claim | verdict | evidence |
|---|---|---|
| engine.py anchors 968-970, 1060-1078, 1403-1418, 1421-1422, 1506-1569, 1609-1616, 1786-1787 | stale | They come from the tree before the D-016 commit. At HEAD they are 996-998, 1088-1107, 1433-1449, 1453-1454, 1538-1602, 1641-1648, 1818-1819. |
| config.py anchors (95-98, 155-158, 166, 237-245, 289-290, 319-336, 414) | confirmed | Read at HEAD. |
| The pick must come before survival, because over-quota Rank 0 children are eliminated | refuted as a reason | `cand` is still alive between `_survive` and `_after_survival` (engine.py:1105-1110). A post-survival pick over `cand` sees every loser. |
| rank0_over_quota is 64-88 on radiationm and 122-271 on glass4 | consistent | [trace] medians 81/78 and 221/187. |
| Polling only at exchange ticks leaves the worker idle on fast instances | confirmed | A tick runs only at the start of `_step` once `next_poll` has passed (engine.py:1048), with poll_s 0.25 (config.py:166). item_placement ran 2087/1813 generations in 120 s [trace]. |
| Sibling genealogy: the copy takes its source's pa/pb and counts as a bred child | refuted | `_with_pending` stamps birth = the entry generation (engine.py:1384-1385), and lineage stratifies by birth (lineage.py:142-150). The copy would be matched against a later generation's children, with exposure taken from an earlier mating, and it duplicates its source's exposure. |
| The warm-up can force "auto", because the decode path uses only kernels that are already warmed | refuted as a control | The same argument failed for D-016: a path the warm-up never ran compiled on the clock and the block was voided (commit 6ae6a6f; elite.py:53-56; test_elite.py:305-307). |
| `_Phases` misses host time on the GPU | partly true | CUDA events on an idle stream approximate wall time (engine.py:452-463). perf_counter is still the right instrument for M1. |
| Passing names at spawn pushes 305,712 strings through the pipe | true, but already done | LPWorker does exactly this today (seeding.py:343). A digest is cheaper, so it is adopted. |
| HiGHS's time limit is cumulative, and a cold solve that hits it leaves no basis | confirmed | [probe] time_limit = 0.9×getRunTime → "Time limit reached", 0 iterations, ~2 ms; rebasing to getRunTime()+5 → Optimal. A fresh object with time_limit 1e-4 and presolve on → basis valid False, primal_solution_status 0. |
| RawArray shared memory is needed | not needed in v1 | Requests go only to an idle worker; payloads are ≤ ~7 MB on ca (§3). Deferred. |
| doon is inert | irrelevant | doon is not a development instance (D-016 lists the 9). |
| "Feasible first" starves completion where polishing cannot help | accepted | The budget split in §3 follows from it. |

**Design 2 (post-survival tiers, warm-started worker)**

| claim | verdict | evidence |
|---|---|---|
| Line numbers at 346a164 | correct then | Shifted +3 to +8 after line 663 at HEAD. |
| Activate on continuous columns with lb<ub and degree ≥2; `prob.conts.size` is the wrong test | confirmed | The decoder takes only degree-1 columns with lb<ub (continuous.py:51). |
| radiationm has "zero EA incumbents" | true only for source "self" | n_incumbents is 14 and 5, all with source "solver" [trace]. |
| sing44 has n0 = 0 and no Rank 0 candidates | confirmed | rank0_candidates median 0 [trace]. |
| Turning presolve off after priming speeds warm solves | refuted | [probe] with a valid basis, warm solves cost 1–5 iterations / 1.0–1.4 ms with presolve on or off; HiGHS does not presolve when a basis is valid. Turning it off only makes any later cold solve slower (ca 8.54 s vs 1.47 s [design]). |
| Restore the last optimal basis after a non-optimal item | confirmed, about 10× | [probe] after an Infeasible fixed-integer LP the basis still reports valid, but the next solve took 580–687 iterations (~0.10 s) without a restore and 2–4 iterations (~0.009 s) after `setBasis(last optimal)`. |
| Spawn re-imports run → engine → cupy | refuted | run.py imports itrea lazily (run.py:101-239). Its module level is stdlib plus environment variables only (run.py:28-49). |
| origin = ORIGIN_LP_DECODE, pa = pb = −1, ancestry via child_ancestry | adopted | population.py:150-172; engine.py:1116-1126. |
| The decoded reply is read only at exchange ticks | defect | Same idle gap as above. Fixed by polling at the pick site. |
| Range screen on the EA thread | correct, moved | Moved into the worker, so EA-thread work stays a row copy. |
| Constants only, no new config fields | acceptable | Two fields are adopted instead; the hash is unaffected (§4). |
| Warm-up with an in-process stub | adopted, with a fix | `synthetic_problem` wires each column into 3–6 random rows (engine.py:385-396), so the synthetic problem may hold no degree-≥2 continuous column. The stub must be forced to run. |
| "A test asserts P unchanged" | not possible in a unit test | Tests run on toy models only (AGENTS.md). Moved to the run-time check M0. |
| Separate T1 / T3 tiers | collapsed | In "top" mode both reduce to "feasible by key", since F is ordered by key (ranks.py:221-222). |

**Both designs**
- The existing `LPWorker` is one-shot: it solves once, sends once and exits (seeding.py:291-307). The issue's "on the EA's CPU LP worker" cannot be taken literally. A second, persistent worker is the only route that leaves the root-LP path untouched, and both designs do that.

## 3. Rule

**Mode and activation**
- `repair.continuous: "lp_decode"` means "auto" plus the LP decode. The singleton decoder is built for both modes (engine.py:653 becomes `in ("auto", "lp_decode")`), so the treatment differs from its control by the LP decode alone.
- **Active columns** = continuous, lb < ub, column degree ≥ 2: the "other" column of the #27 sizing table. One helper computes this and is used by both `plan_memory` and `prepare`.
  - No active column → the feature is inert: no worker, no counters, no events, only `startup.lp_decode = {state: "inert", reason}`.
  - On the development set it is active on ca, sing44, radiationm, fhnw, item_placement and glass4, and inert on loue, neos-960392 and rail01.
- `cont_obj` = some continuous column has c ≠ 0. It is false on radiationm and fhnw (#27 sizing).

**Worker**
- A separate spawn process, started in `begin()` at t0 beside `LPWorker` (engine.py:996-998), so the model read and every solve are on the clock (engine.py:10-16).
- It holds one persistent `highspy.Highs`: output off, 1 thread, `solve_relaxation` True (seeding.py:254-256), presolve at its default, `os.nice(10)`.
- It checks a digest of its column names and integrality (integer = kInteger or kImplicitInteger, problem.py:175, 183) against the EA's. A mismatch means state "failed".
- Until it replies "ready" it counts as busy.

**Pick site: every generation, after survival, over the whole candidate set**
- In `_step`, between `self._survive(cand, Np, meta)` (engine.py:1108-1109) and `self._after_survival()` (1110).
- `_survive` stores its host arrays after the demotion rounds (`hk`, `hv`, `hf`), plus `dup` and `surv` (engine.py:1397-1457), only when the feature is on.
- Steps:
  1. Poll the worker without blocking (also polled in `_exchange_in` beside 1818-1819, with no `lp_x` gate).
  2. If it is not idle, send nothing and count `lpd_busy`.
  3. Otherwise select and send one batch.
- At most one request is outstanding, so at most one decoded batch per generation.

**Which rows** (host only; no random draws)
- Eligible rows are:
  - not a duplicate;
  - finite key and V;
  - not an unchanged solver find (`_is_solver_find`, engine.py:513-516);
  - not carrying ORIGIN_LP_DECODE;
  - not already offered (a set of offered ids).
- **F (polish)**, only if `cont_obj`: feasible rows after the demotions, by (key, id) ascending, at most N = `lp_decode_per_gen` (4).
- **I (completion)**, only if `lp_decode_infeasible`:
  - infeasible rows with key < inc − imp_tol·max(1, |inc|), where inc = `surv.inc` (ranks.py:224-225; imp_tol at config.py:97);
  - in the feasibility phase (inc = +inf), every infeasible row;
  - if not `cont_obj`, also key ≥ dual_key − imp_tol·max(1, |dual_key|): the key then depends on the integers only, so no completion can beat the proven bound;
  - ordered by (V, key, id), D-011's order (config.py:237-245);
  - at most 3N rows (OFFER_FACTOR = 3).
- Each offered row is copied with `bk.asnumpy(cand.X[i])`, the `_check_candidate` pattern (engine.py:1493): a view plus a copy, no gather kernel.
- Its integer part is rounded and clipped as `_receive` does (engine.py:1904-1907) and sent as int32 when the integer bounds fit, otherwise float64.
- Payload on ca: 146,400 integers, 0.59 MB per row as int32, ≤ 16 rows ≈ 9.4 MB per request. The reply is ≤ 4 × 159,312 float64 ≈ 5 MB.

**Budget per batch (decided in the worker)**
1. Cache: rows whose integer part was already solved (blake2b of the part) get CACHED and no LP.
2. Range screen, for I rows only: for every row, [A_I·x_I + cmin_i, A_I·x_I + cmax_i] must meet [rl_i − tol, ru_i + tol], with tol = feas_tol·max(1, |side|) (config.py:98). cmin/cmax are the activity range of the row's continuous columns over their bounds. This is a necessary condition only; a failing row gets RANGE_FAIL.
3. Split, with n_F' = F rows not cached and n_I' = I rows passing the cache and the screen (tested in priority order until enough have passed):
   - s_I = min(n_I', max(N − n_F', ⌊N/2⌋)), s_F = min(n_F', N − s_I);
   - with N = 4 that is 2 + 2 when both kinds exist; a kind that is short passes its share to the other.
4. Solve F first, then I. Rows offered beyond the budget get NOT_SOLVED.

**Each solve**
1. `changeColsBounds(int columns, x, x)`.
2. If the previous solve was not Optimal and an anchor basis exists, `setBasis(anchor)` [probe: about 10× fewer iterations].
3. Set `time_limit = getRunTime() + L`, because the limit is cumulative [probe]:
   - L = `lp_decode_cold_limit` (10 s) while no anchor basis exists (priming);
   - otherwise `lp_time_limit` (2 s) for F rows and `lp_time_limit`/2 for I rows (cuOpt's 2 s / 1 s, vault/80-reviews/2026-10-08 …:64-66).
4. `run()`.
5. Accept only if primal_solution_status is FEASIBLE (seeding.py:268).
6. On Optimal, the anchor becomes `getBasis()`.
- No solve starts after t0 + budget.

**Result**
- Copy = the source's integer part + the LP's continuous values clipped to [lb, ub]; key = `prob.key(Y)` (problem.py:88-89).
- F-source copies are kept only if key < src_key − imp_tol·max(1, |src_key|); otherwise `lpd_no_gain`. This drops the tolerance-edge cases.
- I-source copies are always kept; the device evaluation decides.
- Kept copies enter as one `_Batch(repair=False, src="lp_decode")`:
  - built like `_receive`'s rows: `bk.xp.asarray(Y).astype(gene_dtype)` (engine.py:1909; the gene dtype is float64 whenever continuous columns exist, 613-618);
  - they become candidates through `_with_pending` (1354-1391) and `_survive` (1393-1470);
  - the float64 re-check of Rank 1 entrants (1433-1449) and the incumbent / post path (1538-1602) are unchanged; `_admit` is not used.
- Provenance of a copy:
  - origin = ORIGIN_LP_DECODE (population.py:47), which falls under `_GENE_BITS_EA` (engine.py:222), so its incumbents have source "self";
  - strat = its source's strategy (no `_random_strategies` draw, 1153-1160);
  - ancestry = `child_ancestry(depth, r_src, r_src, o_src, o_src, …, pa_src=src_bits(o_src), pb_src=src_bits(o_src))`, with r_src = the source's rank in this survival, or NO_RANK if it was eliminated;
  - pa = pb = −1 and rank_pa = rank_pb = −1, as for every pending row (1384-1387);
  - a new genealogy column `lpd_src` holds the source id.
- No entry, each with its counter: infeasible LP, time limit, other status, RANGE_FAIL, CACHED, NOT_SOLVED, after the deadline.
- If the worker dies or the digest mismatches: state "failed" for the rest of the run, recorded; the EA never waits on it.

## 4. Implementation plan

**New `src/itrea/ea/lp_decode.py`**
- Imports numpy, scipy, os, time, hashlib and multiprocessing only. highspy is imported inside the worker function, and the module never imports engine or cupy: the spawned child unpickles its target from this module.
- `active_columns(prob)`, `model_digest(names, isint)`, `select_offers(...)` and `split_budget(...)`: pure functions.
- `_decoder_main(conn, path, digest, int_idx_dtype, cold_limit, time_limit, feas_tol, deadline)`:
  - pipe EOF, OSError or BrokenPipe → exit;
  - any exception → `("failed", repr, traceback)`;
  - every reply carries stats: solves, priming_s, solve_s_total/max, iterations, time_limit_hits, cache hits, screen passes, ru_maxrss, cpu_s.
- `class LPDecoder`:
  - spawn context, duplex Pipe, `daemon=True`;
  - `OMP/OPENBLAS/MKL=1` and `CUDA_VISIBLE_DEVICES=""` set around `start()` (the seeding.py:345-355 pattern; pair.py:545 precedent);
  - states: starting / idle / busy / failed / stopped;
  - `offer()` refuses unless idle;
  - `poll()` calls `conn.poll()` and reads only when a message is waiting;
  - `stop()` = terminate → join(2) → kill → join(2) (seeding.py:396-408), idempotent.
- `LPWorker` (seeding.py:310-426) is untouched.
- Constants (OFFER_FACTOR = 3, I-limit factor 0.5, nice 10, cache cap 65,536) are listed in the docstring and echoed in the summary. Any of them must become a `_LATE_DEFAULTS` field before it is changed.

**config.py**
- RepairCfg gets `lp_decode_cold_limit: float = 10.0` and `lp_decode_infeasible: bool = True`.
- Both enter `_LATE_DEFAULTS` (289-290).
- `validate()` (366+, beside 414) adds: per_gen is an int ≥ 1, `lp_time_limit` > 0, `cold_limit` ≥ `lp_time_limit`, the switch is a bool.
- Correct the comments at 155-158 and add a docstring paragraph like 29-31.
- Regenerate `configs/itrea2-default.yaml`; its header hash stays 9ac4a93bf93b.

**engine.py**
1. `plan_memory` (262-317): K at 282 += N only when continuous == "lp_decode" and `active_columns(prob)` is non-empty.
2. `prepare`:
   - replace 645-646;
   - require the instance file only when active columns exist (the pattern at 647-650);
   - 653 becomes `in ("auto", "lp_decode")`;
   - set `self.lpd_cols` and `self.lpd_cont_obj`;
   - add `rec["lp_decode"]` only in lp_decode mode.
3. A new Engine keyword `decoder_factory=None` (tests and warm-up).
4. `_warmup` (718-757), when active:
   - the synthetic Engine gets `decoder_factory=_WarmupDecoder` (an in-process stub, like `_WarmupMailbox` at 2275) and a force flag;
   - the stub answers on the next poll and returns OK for every offered row;
   - if selection yields nothing, the forced path offers population rows as I rows;
   - the no-gain filter is bypassed in warm-up only;
   - record `lp_decode_warmed` per pass: ≥ 2 rows in "many", 1 row in "one" (precedent `elite_warmed`, 752);
   - the `"seeding.lp_source": "solver"` override (732) is unchanged.
5. `begin` (910-1030): `self.lpd = None`; start `LPDecoder` beside 996-998 when active; add `lpd_*` counters only then (precedents 966-969, 974-978); keep host perf_counter accumulators as floats.
6. `_survive`: when `self.lpd is not None`, keep references to (hk, hv, hf, dup, surv) after the loop (1433-1457).
7. `_step`: between 1109 and 1110, `if self.lpd is not None: self._lpd_tick(cand)`.
   - `_lpd_tick` = poll, then select / copy / offer, timed by `time.perf_counter()` into `lpd_host_s` and `lpd_host_s_max`.
8. `_exchange_in`: after 1818-1819, `if self.lpd is not None: self._lpd_poll()`.
   - `_lpd_poll` handles ready / result / failed, builds the copies, applies the no-gain filter, appends the `_Batch` and writes one `lp_decode` trace event per reply.
9. `_Batch` (503-510): optional `anc_rank`, `anc_ops`, `src_ids` = None.
   - `_with_pending` (1384-1387) passes the ancestry to `_make_pop` when it is set, and records id → source id.
   - With None it behaves exactly as today.
10. `_new_incumbent` (1573-1602): count `lpd_incumbents` and `lpd_posts` by membership in the copy-id set. The origin bit is OR-merged over duplicates (docstring 170-176), so it is not exact attribution.
11. Stop the decoder in `run`'s finally (889-891) and in `finish` (2144-2145). An outstanding request at stop counts as `lpd_lost`.
12. `_gen_record` (2088-2135) extras, lp_decode mode only: state, offers, solved, entered, incumbents, host_s.
13. A `lp_decode` summary block, modelled on `lp_stream` (2174-2180).
14. `genealogy` (2196-2215): add `lpd_src` only in this mode (precedent `cuopt`, 2212-2214).
15. Module docstring section "LP decode".

**Elsewhere**
- `experiments/pair.py` 523-525 (the `spec.x925` branch): raise RunError when the config has `repair.continuous: lp_decode`, until a decision allows it. AGENTS.md: documentation alone is not a control.
- `build_runs`: nullable `lpd_*` columns.
- Documentation: D-017 before any run; algorithm.md:73-75 (step 6 currently promises a post-survival "Rank 1 / posted" pick); config.py:157; Trace format; an AGENTS.md map row; the journal; a comment on #27.

## 5. Configuration and hashes

- Control `configs/itrea2-dev-rc-3way.yaml` hashes to 1d12ad490dd0.
- Control + `repair.continuous: lp_decode` hashes to **f030e23909ad**, computed at HEAD. It stays the same after the change, because the two new fields are dropped while at their defaults (config.py:332-335). Pin it in test_configs before the window.
- The default 9ac4a93bf93b and every recorded hash are unchanged (test_configs.py:119-131).

## 6. Tests

CPU tests use the NumPy backend and `toy.mixed` (y0 and y1 have degree 2 and non-zero costs; toy.py:44-60) written with `write_mps` (63). Tests marked "solver" spawn the worker.

1. **Hashes:**
   - recorded hashes unchanged; treatment hash pinned;
   - each new field away from its default changes the hash;
   - validation rejects cold < limit and per_gen 0;
   - the default YAML equals the code.
2. **Default path unchanged:** the existing reference captures (`tests/_engine_reference.py`, `fixtures/engine_reference_disabled.npz`, `engine_reference_traffic.npz`) still match.
   - lp_decode on knapsack and set_cover (no active column, no file needed) also matches the reference.
   - An all-singleton toy under lp_decode gives `eng.lpd is None` and no `lpd_*` keys.
3. **Units:**
   - `select_offers`: exclusions, both orders, the Rank 0 threshold, the feasibility phase, `cont_obj` false, the bound exclusion;
   - `split_budget`: 2 + 2, a short kind passing its share;
   - the range screen on hand-built rows.
4. **Worker (solver):**
   - ready arrives;
   - the fixed-integer optimum equals a fresh HiGHS solve (relative 1e-9);
   - INFEASIBLE, RANGE_FAIL and CACHED codes;
   - 100 solves with L = 0.05 s and cumulative time beyond it all return Optimal (rebasing);
   - a time-limited cold solve is followed by a solve under the cold limit;
   - basis restore: the next solve's iterations after an infeasible one are ≤ those after an optimal one + small slack;
   - digest mismatch → failed;
   - `poll()` returns in < 5 ms during a slow batch (test-only sleep hook);
   - EOF → exit;
   - no solve after the deadline;
   - `stop()` is idempotent, and the process is gone afterwards (`os.kill(pid, 0)` raises; never pgrep).
5. **Engine on mixed.mps (solver):**
   - copies enter with ORIGIN_LP_DECODE, the source's strategy, pa = pb = −1 and `lpd_src`;
   - their integer parts equal the source's (via the event);
   - continuous values are LP-optimal (linprog);
   - F copies beat their sources by imp_tol;
   - every decoded incumbent passes `Problem.check`;
   - counters add up;
   - `lineage.analyse` runs;
   - the worker is gone after `finish` and after an exception in `step`.
6. `_with_pending` with no batch fields still gives −1 parents and zero ancestry.
7. `plan_memory`: K grows by exactly N only in lp_decode mode with active columns.
8. Warm-up: `lp_decode_warmed` ≥ 1 in both passes, and no child process after `prepare` (`multiprocessing.active_children()`).
9. The runner refuses lp_decode for A9 and A3x.
10. **GPU** (the test_elite.py:281-318 pattern, empty CuPy cache, mixed.mps):
    - `compiled_on_clock` False and `new_after_t0` empty;
    - `lpd_entered` ≥ 1 after t0, so the test cannot pass vacuously (lesson at test_elite.py:305-307);
    - decoded rows built on CuPy equal NumPy's.

## 7. Development test D-017 "dev-ld1" (written before any run; development only, D-010)

**Before the window**
- CPU suite and GPU suite pass.
- `tools/probe_lp_decode.py`: the real worker on the stored A3@rc1 seed-0 vectors of ca, sing44, radiationm, glass4 and item_placement; 1 thread, nice, a core outside the GPU lane, ≤ 2 min. Gates:
  - ca priming Optimal within 10 s;
  - warm median ≤ 1 s;
  - no spurious time-limit hit after > 2 s of cumulative run time;
  - ca worker RSS ≤ 1.5 GB.
- Inside the window: one 30 s sing44 seed-9 smoke run, not analysed.

**Arms**
- Control = the development default at that time, re-run in the same window: A3@rc1, or A3@cl1 if D-016 is adopted.
- Treatment = control + `repair.continuous: lp_decode`, variant "ld1". Exactly one field differs.
- If the control is cl1, each band is max(D-016's band, cl1's own two-seed spread, 0.02).

**Instances and seeds**
- ca, sing44, radiationm, fhnw, item_placement, glass4.
- Seeds 0–1, 120 s, A3 cores (pair.py:208, 527), both LLM servers stopped, interleaved per block.
- 24 runs, about 1.2 h, under the itrea-window lease, job file `jobs/dev-ld1.jsonl`.
- loue, neos-960392 and rail01 are covered by CPU test 2 instead of runs.

**Mechanism — must hold, otherwise the outcome is not read as lp_decode's**
- **M0 validity:** per block, equal P and gene dtype between the arms (startup records); `compiled_on_clock` False; worker never "failed"; ready ≤ 10 s after t0.
- **M1 the GPU never waits:** `lpd_host_s` ≤ 1 % of EA wall per run; `lpd_host_s_max` ≤ 50 ms. The generation-count ratio is reported but not gated: glass4 ran 19 vs 28 generations between seeds [trace].
- **M2 the worker is sane:** after priming, time-limit hits ≤ 10 % of solved LPs per instance (seeds pooled); ca's priming solve is Optimal in every ca run.
- **M3 polishing acts:**
  - sing44: median relative gain of admitted F copies ≥ 1e-2, and ≥ 1 decoded EA incumbent in each seed;
  - ca: median gain ≥ 1e-3, and ≥ 1 decoded incumbent in at least one seed. The control had 3 and 1 EA incumbents, the first at 34.9 s [trace].
- **M4 completion:** per-instance counts are reported (offered, passed the screen, LP feasible, entered, incumbents). If no I-source LP is feasible across all 12 treatment runs, the record says completion is dead weight on this set; `lp_decode_infeasible: false` then becomes a separate variant.

**Outcome** (bounded d on coupled A3 P(120) from checker-passed incumbents, seed means, positive = ld1 better; D-016's rules)
- **Adopt** if the mean d over the 6 is ≥ +0.05 and no instance is worse than −band_i.
- **Reject** if the mean d ≤ −0.05, or any instance is worse than −band_i on both seeds.
- **Otherwise inconclusive:** seeds 2–3 on the in-band instances where the mechanism acted, same rules.
- Bands: ca 0.10, fhnw 0.02, glass4 0.37, item_placement 0.68, radiationm 0.46, sing44 0.11 (D-016).
- Also reported: the 9-instance mean with d = 0 for the three inert instances, and the results in D-013 terms.

**Predictions — stated, not criteria**
- sing44 positive; ca small positive.
- glass4 and item_placement about zero: their copies sit at the tolerance edge and are filtered.
- radiationm rests entirely on completion; all its incumbents came from the solver [trace].
- fhnw inert unless completion fires.

**Diagnostics**
- Feasible children per generation, treatment vs control (the cutoff-row interaction, engine.py:1641-1648, config.py:144-146).
- Decoded share of Rank 1; LP time per kind; worker RSS and CPU seconds.
- SCIP-accepted posts made from decoded copies.
- Time to the first EA incumbent (control: ca 34.9 s, sing44 15.3 / 16.3 s, fhnw never).

**Record:** a vault/30-results note, with every failure, before any number is quoted elsewhere.

## 8. Risks

1. **Cutoff row.** Decoded keys lower the population best, so FJ's cutoff row then asks FJ-level continuous values for LP-level keys. Feasible children may collapse, on sing44 above all. This is pre-stated as a diagnostic.
2. **Distant warm starts.** A warm start helps only when consecutive integer parts are close. [probe] moving 100 of 1,500 fixed values cost 1,706 iterations, nearly a cold solve. The designs' ca warm times come from close vectors; real children may be slower. M2 and the per-LP times measure it.
3. **Cores.** In A3 the EA host, the root-LP worker (ca: LP arrived at 28.9 s [trace]) and the decoder take exactly the 3 A725 cores, inside D-002's footprint. A9/A3x would share one core, so the runner refuses them.
4. **Concatenate parts.** `_with_pending` concatenates cand + pending batches. A3's worst case is cand + 2 LP-stream batches (per_tick = 2, config.py:254) + solver + LP rounding + decode = 6 parts. A9 would reach 8, which is CuPy's limit before it switches to an unwarmed kernel (elite.py:53-56). This is another reason for the refusal.
5. **Memory.** The decoder holds a second model copy inside the EA's 6 GB scope (pair.py:222). The EA's own maxrss was 858.9 MB on ca [trace]. The probe gates the worker's RSS.
6. **IPC.** A pipe `recv` waits while the worker writes ≤ 5 MB on ca. It is measured within M1; shared memory is the fallback.
7. **Tolerance-edge instances.** Up to half of each batch is spent on F rows that are then filtered (glass4, item_placement). No adaptive switch in v1.
8. **Rank 1 crowding.** Under "top", a source and its copy can both sit in Rank 1. Under "clearing" (D-016) they share a niche and the better one is kept. So the order of D-016 and D-017 matters.
9. **Genealogy.** Decoded cases fall into `cases_without_parents` (lineage.py:15, 134). If lp_decode ever enters a confirmatory configuration, lineage needs a rule (credit through `lpd_src`), decided then.
10. **Strategy credit.** Copies carry their source's strategy, so the strategy statistics mix FJ and LP effects (engine.py:1590-1592).
11. **Non-determinism.** Which generation receives a batch depends on wall time. No random number is drawn by this path.
12. **Orphan worker.** On a SIGKILL of the EA the daemon flag does not fire. The worker exits at pipe EOF, at most one LP later; the systemd scope stop is the backstop.
13. **Model order.** The canonical order is HiGHS's column order of the same file (problem.py:7-9). The digest turns a mismatch into "failed" instead of wrong vectors.
14. **Motivating numbers.** The stored-vector gains are [design] numbers from one seed. They motivate the test and are not results.

## 9. Rejected or deferred

- Pre-survival pick over children only (Design 1): the post-survival pick sees the same losers plus population and pending rows, the current inc and the dup flags.
- Sibling genealogy (Design 1): wrong lineage stratum and duplicated exposure.
- Warm-up forced to "auto" (Design 1): the D-016 incident is the counterexample.
- Presolve off after priming (Design 2): no warm gain [probe], and slower cold fallbacks.
- EA-thread range screen and polling only at exchange ticks (Design 2): moved to the worker; poll added at the pick site.
- Extending the one-shot `LPWorker`: the root LP takes ~29 s on ca, and the change would alter the default path.
- `feasibilityRelaxation` for infeasible sources: about 4 s on ca for 12–21 % less V, and it ignores the objective [design].
- cuOpt's "both versions enter" for feasible sources: the no-gain filter is used instead.
- Decoding every child: CPU-bound; ≤ N LPs per batch.
- The issue's "feasible first": the split reserves half for completion.
- **Deferred:** shared-memory IPC; an adaptive per-kind budget; reading the model in `prepare()` (the clock rule allows the parse before t0, engine.py:10-16; v1 starts at t0 as `LPWorker` does); lp_decode in A9/A3x.