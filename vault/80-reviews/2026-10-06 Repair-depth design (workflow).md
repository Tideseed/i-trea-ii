# 2026-10-06 Repair-depth design (multi-agent workflow)

**What:** after the one-variable test (deep repair `lps1` beats the default `lps0` on 6 of 9 instances; the default
wins only item_placement_9903 — [[Development log]]), a design workflow (Claude Code subagents, 2026-10-06 17:36–18:40,
read-only on the repository, no GPU): a code map, a run-data analysis of the lps0/lps1 traces and an open-access
literature survey in parallel; three designs from different angles (per-generation depth bandit; a repair clock — a
step horizon enforced between GPU launches; a deep "ride-along" lane); two adversarial judges (expected effect;
implementation risk and measurement honesty); one synthesis. Scores: repair clock 6.5 / 7.5, bandit 5.5 / 6.5,
ride-along 3.5 / 3.5. **Outcome: build the repair clock, staged — instrumentation and SHADOW mode first (no behaviour
change), the active controller only if the shadow signal passes pre-stated criteria** (plan below; it becomes D-012
when built).

## Plan (synthesis, verbatim)

I-TREA II adaptive repair depth: build Design 2 (repair clock) first, in stages, adding Design 1's safeguards

DECISION
Each repair call gets a horizon counted in lane steps, enforced between GPU launches. The controller picks the horizon from what the previous call showed. Why:
- The stall rule ends 90-100 % of lanes, and every call waits for its slowest lane. So the slow tail is what to control.
- Until the EA has its own first feasible solution, every call runs exactly as lps1. That keeps deep repair's earlier first feasible solutions:
  - sing44: 12.7 / 16.6 s, against 52.9 / 62.7 s at the default depth;
  - ca seed 0: 35.5 s against 58 s;
  - loue: 2.1 s against 7.3 s;
  - fhnw seed 0: the optimum, found only with deep repair.
- item_placement is feasible at 0.05 s, so control starts there almost at once.
- It is the only design that can be falsified without changing behaviour.

Not Design 1: on item_placement seed 1 the two depths tie at 2-8 s, below its switching margin. Not Design 3: it edits the kernel used by every arm, and the residency it relies on is unmeasured.

MECHANISM
- Levels:
  - The levels are h = 2^7 to 2^16 lane steps, plus "none".
  - The repair settings stay those of lps1 (max_steps 20000, stall 1000).
  - One level is chosen per generation and applies to all of that generation's repair calls.
- The cut:
  - Each launch runs iters = min(_iters, h - cum) steps.
  - Lanes still active at h get the host-side status ST_HORIZON = 4 and keep their projected x.
  - After every launch all active lanes have the same S_STEPS, so each call can be replayed exactly from the logged H_eff.
- Phase: until h_feas.any() is true, there is no horizon and no cap, which is lps1. A finite inc_external does not end this phase; glass4 already has one at gen 0.
- Value of level h in generation g:
  - Delta_g(h) = max dg = (cutoff - key) / max(|cutoff|, |dual_key| if finite, 1e-9). The maximum is over lanes that ended FEASIBLE with S_STEPS <= h and are still feasible after the decoder.
  - A dg below 1e-3 counts as zero.
  - This is exact for the children and the pending batches, because the cutoff is fixed within a generation.
- Cost of level h:
  - T_g(h) = the non-repair time plus the sum of t_call(h), read from the (cum, t) pair recorded after each launch.
  - A horizon-h call is a prefix of the unbounded call, so its cost is measured, not modelled.
- Choice:
  - Each level's rate is G/T, with gains and times decayed at a half-life of 20 s.
  - Switch only when all three hold: at least 5 observed generations, at least 2 material gains, and a rate above 1.25 x the current level's.
  - With no evidence, play "none".
- Probes play "none", for at most 10 % of the time. One unbounded call re-measures every level.
- The one-shot LP batch always runs at "none", uncapped. LP-stream batches follow the current level.
- Cap (active mode, after the first feasible only):
  - The cap is min(10 s, 3 x the level's median call time); for "none" it is 10 s alone.
  - It is split over chunks. A cut is logged as H_eff, and chunks reached after the deadline are still projected.
- Shadow mode plays "none" with no cap. It logs its choices, every call's (cum, t) profile, and each lane's S_STEPS and status.

CHANGES (only after the running pool ends)
- config.py: a new section repair_clock.
  - Fields: enabled False, mode shadow|active, level_min_log2 7, level_max_log2 16, min_gens 5, min_events 2, materiality 1e-3, half_life_s 20, hysteresis 0.25, probe_share 0.10, cap_s 10, cap_factor 3, pin_lp True.
  - hash() drops the section while it is disabled, so the four recorded hashes do not change.
  - validate() requires repair.method fj and objective_mode cutoff_row.
  - Regenerate the default yaml (its header hash does not change) and add itrea2-adaptive-lps-rc.yaml.
- FJGpu.run: new arguments horizon, cap_s and profile. The change is host-only, with no kernel edit. With the arguments unset, the loop is today's. It also keeps last_steps (from the existing st copy) and last_profile.
- CPU path: max_lane_steps in _fj_batch; horizon and cap_s in FJCpu.run.
- repair.py passes on only the arguments that are given.
- New ea/repair_clock.py: RepairClock with decide, add_call, end_generation and record. It draws no random numbers.
- engine.py:
  - decide the level after the exchange tick;
  - _ls_run passes the new arguments only when the section is enabled;
  - one helper pairs each lane with its evaluated key, for the children, each pending batch, and the rep subset in _admit;
  - new rc_* fields in gen events, an rclock event, and a summary block.
- Docs: the FJ spec, Trace format, decision record D-012, and the Development log.
- CPU tests:
  - horizon semantics;
  - the counterfactual identity with the cutoff armed;
  - unset arguments give today's results;
  - the controller: a finite inc_external alone does not activate it, a single gain never triggers a switch, and the LP batch stays pinned;
  - a shadow run's genealogy equals the disabled run's;
  - the pairing helper;
  - the four hashes, the dd31d67 reference and the one-call spy.
- GPU tests, run later in a GPU window:
  - every active lane has the same S_STEPS after every launch;
  - ref_fj with the horizon rule;
  - multi-chunk projection.

RISKS
- The signal only sees the current generation. item_placement gains from running more generations, and glass4's large jumps may need a population built by deep repair. Stage 1 tests this before any behaviour changes.
- A shallow level is not lps0 (the stall limit stays 1000 and lanes are cut at h). Its cost is measured; its outcome is known only after Stage 2.
- Many tiny improvements could inflate a level's rate (glass4: 200 steps worth 0.001 in total). The 1e-3 materiality threshold and the two-gain rule stop that.
- A wrong switch is re-checked by the probes. The LP batch is pinned, so its timing cannot cost ca seed 0 its early feasible solution.
- Level choices depend on measured time, as generation counts already do. Every played H and H_eff is logged, so each call can be replayed. The cap is the only thing that cuts a lane by time inside a call.
- The settings are frozen before Stage 2. rc0 and rc1 count against D-010's E4 cap.
- Still owed: a run with the strategy off at the default depth, to resolve the glass4/loue confound.

MEASUREMENT (box-window lease and gates; the control is lps1 run in the same window)
Stage 1: shadow arm A3@rc0, about 17 min. item_placement seeds 0 and 1; glass4, sing44, neos-960392, loue and ca at seed 0.
Pass requires all three:
- (a) item_placement, both seeds: within 10 s of the first feasible, it would leave "none" for a level whose rate is above 1.25 x that of "none" and whose children calls take at most 0.1 s.
- (b) glass4, loue, neos-960392: "none" or one level below in at least 80 % of the generations after the first incumbent.
- (c) generation counts within 10 % of lps1.
Also report the S_STEPS of the lanes behind glass4's large jumps. If (a) or (b) fails, fall back to Design 1 with three changes: the phase taken from h_feas, the LP batch pinned deep, and a shorter alternation.

Stage 2: active arm A3@rc1, interleaved with lps1. 9 instances x seeds 0-1 (about 80 min), then seeds 2-3 on item_placement and ca.
Reject unless all of these hold:
- item_placement beats the control on at least 3 of 4 seeds, with mean d of at least +0.15;
- sing44, glass4, loue and ca each have d of at least -0.05;
- radiationm and rail01 are within +/-0.01;
- the mechanism agrees: at least 60 % of item_placement's time below "none"; at least 60 % of glass4's and loue's time at "none" or one level below; cap hits only on ca;
- every incumbent checked, GPU health ok, compiled_on_clock False.
fhnw and neos-960392 are reported but not judged.

LITERATURE (open full text)
- Spoorendonk 2026: https://arxiv.org/abs/2609.22938
- Pauphilet & Wu 2026: https://arxiv.org/abs/2610.03090
- Luteberget & Sartor 2023: https://luteberget.github.io/preprints/feasibilityjump-2023-01-09.pdf
- Hendel 2022: https://link.springer.com/content/pdf/10.1007/s12532-021-00209-7.pdf
- Chmiela et al. 2023: https://arxiv.org/abs/2304.03755
- Sudholt 2012: https://arxiv.org/abs/1109.6441
- Cotta 2025: https://link.springer.com/content/pdf/10.1007/s11750-024-00694-8.pdf
- Fialho et al. 2010: https://inria.hal.science/inria-00519579 (read through an archive capture; check that the live copy is still open before citing)

Not citable: Ong & Keane 2004 (open access unverified); Ishibuchi 2003; Sudholt TCS 2009; Molina EC 2010; Chen 2011; Neri & Cotta 2012; the Hart and Land theses; Davies 2024.

(A copy is at /tmp/itrea-design/plan.txt. Nothing in the repository was touched and no GPU was used.)

## Judges (verbatim)

### Design 2: repair clock (lane-step horizon ladder + 10 s cap + counterfactual Δ(h)/T(h) controller) — 6.5/10
**Strengths.** 1. Before the first incumbent it runs exactly as lps1, so it keeps the deep-repair first-feasible wins: sing44 12.7/16.6 s, ca s0 35.5 s, loue 2.1 s, and the fhnw s0 find. No cap applies in that phase.
2. Its central invariant holds in the kernel. fj_steps runs exactly `iters` steps per active block per launch (loop ~383-536, stop break ~386), so after every launch all active lanes have equal S_STEPS. A horizon cut between launches is therefore an equal-steps cut that can be replayed, with no kernel edit and no recompile.
3. The counterfactual is exact for the children and pending calls. The cutoff is constant within a generation: _refresh_cutoff runs only at engine 1275, after the immigrant repair, and at 1607/1623 in the exchange at the top of _step. So one deep call reveals every shallower horizon's outcome, and exploration is only needed upward.
4. t(h) comes from host data already present (the per-launch timing and st = cp.asnumpy(STI[:kc]) at ~855).
5. The 10 s cap bounds ca's 56.8/58.7 s generations.
6. On item_placement it decides fast: 5 deep generations ≈ 1.6 s.
7. Shadow mode is the cheapest experiment that yields the per-lane evidence every design needs.
8. Wall-clock note: my window decomposition puts only 0.011 of item_placement's P gap in 0-10 s, so fast detection is useful but not decisive.

**Flaws.** 1. FACTUAL ERROR in its phase rationale. It says the phase test is "the EA's own cutoff state, not SCIP's incumbent". But _refresh_cutoff (engine.py 1420) sets cutoff = min(pop_best, inc_external). In glass4 lps1 both seeds have a finite inc_external at the gen-0 record (t = 5.5 / 5.9 s). So on glass4 the controller and the cap are live from gen 0, and its "pre-incumbent = lps1" guarantee does not hold there.
2. NO MATERIALITY THRESHOLD. glass4 lps1 has 20 / 16 tiny EA steps (sum ≤ 0.001) beside only 2 big ones (0.076 at 32.4 and 50.7 s on s0; 0.054 at 15.6 and 37.2 s on s1). Polish lanes credit every horizon level equally, so the per-second rate favours the shallowest level. min_gens = 5 at about 4.9 s per generation puts the first decision at about 30 s on s0, on polish-only evidence, just before the 32.4 s jump. lps0 found no step ≥ 0.01 on glass4. A switch there drifts toward lps0's 0.316 / 0.331.
3. THE GAIN UNIT IS INFLATED. D_g = cutoff − dual is relative gap closure, not the P(T) gap unit. At glass4's gap of about 0.3 that roughly triples the gains, which makes polish look bigger still.
4. EXPLORATION IS ONLY +2 LEVELS. From a shallow level the controller never re-observes "none". Uniform decay leaves the ratio G/T of an unobserved level frozen at its stale value, so a wrong early switch is sticky.
5. THE SIGNAL IS MYOPIC, AND THE THIN SPOT IS EXACTLY item_placement's mechanism. The shallow edge there is big steps:
   - lps0 s0: 18 steps ≥ 0.01 = 0.479 of 0.616 EA gain;
   - lps0 s1: 10 steps = 0.248;
   - deep's small-step sum is HIGHER (0.296 / 0.252 vs 0.137 / 0.245).
   A per-generation max-gain counterfactual computed on a deep population sees only small steps. The realized per-generation ratio is 0.31 for a 4.7x time ratio, which gives a rate ratio of about 1.45, barely above the 1.25 hysteresis. It may stay at "none".
6. THE SHALLOW HORIZON ARM IS NOT lps0. With stall 1000 and H < 1000, no lane stops on stall before H, so every unfinished lane runs to H. That arm is not a measured configuration.
7. Dev-tuned knobs, 2 seeds, and the item_placement seed spread (lps0 s0 0.300 vs s1 0.417) are shared weaknesses of all three designs.

### Design 1: phase-split, time-normalised depth bandit (one of two measured depths per generation, deep prior) — 5.5/10
**Strengths.** 1. Both arms are the two measured endpoints (lps1, lps0), so neither arm is an unknown configuration.
2. Host-only and O(1): no kernel change, no extra local-search call (the GPU _calls counter and the RNG streams are unchanged), no random draw, and K = 1 reproduces a fixed-depth run.
3. Phase F (EA-own feasibility AND inc_external = inf) defaults to deep, which matches the data. My recomputation (/tmp/itrea-design/depth_analysis/steps.pkl):
   - sing44: gap stays 0.985 to 40 s in lps0 vs 0.495 / 0.576 at 20 s in lps1;
   - ca s0: lps0 still at 1.000 at 40 s vs 0.004 deep.
4. The reward is in P(T) gap units: dg / max(|P_before|, |dual|).
5. The θ = 1e-3 threshold plus min_evidence = 2 protects glass4. lps0 glass4 had at most one small step per run (s1: 0.003), so shallow cannot displace deep there.
6. It credits only EA-own gains over min(EA best, inc_external).
7. Wall-clock note: 61 % of item_placement's P difference accrues in 40-120 s (s0 window 0.328 vs 0.171) and only 0.011 in 0-10 s. A switch at 20-40 s still recovers much of it.

**Flaws.** 1. THE ONE-SHOT LP BATCH IS REPAIRED AT WHATEVER ARM HOLDS THAT GENERATION. _lp_arrived (engine 1740-1762) appends one 32-row pending batch with repair=True, repaired in _with_pending (1172-1174). The phase-F floor gives shallow about 10 % of the time. If the HiGHS point lands in a shallow probe generation, ca s0's win is lost: lps0's LP immigrant ended at V 0.042, feasibility came only at 58 s, and P was 0.487 vs 0.298. The same exposure applies to sing44's LP at 10.3 s.
2. item_placement's shallow edge is lumpy big steps (18 / 10 steps ≥ 0.01; 0.479 / 0.248 of the gain), while deep's small-step sum is higher (0.296 / 0.252 vs 0.137 / 0.245). So in any 2 s epoch without a big step, mu_shallow < mu_deep and the 25 % margin fails, systematically rather than by noise.
   - Big steps fall in about 10 % of 2 s windows, so the 4 s of shallow exploration expects only 0.35-0.6 of them.
   - Under a deep prior with a 10 % floor, selection is a coin flip early and a slow drift later. Seed 1 is the hard case: lps0 s1's gap at 10 s was 0.705 vs deep's 0.747, i.e. no separation.
   - The claimed 0.38-0.45 is optimistic; I expect about 0.45-0.52, with bimodal seeds.
3. THE ALTERNATION TAX HITS neos-960392 HARDEST. Its whole P accrues in 0-15 s (lps1 gap at 10 s 0.021 / 0.004 vs lps0 0.059 / 0.168). The 4 explore epochs (≈ 8-10 s) put half of that window at shallow depth, which costs up to −0.1 to −0.18 bounded d.
4. Slow deep generations make an epoch one generation: glass4 ≈ 4.9 s, and ca's first phase-I deep epoch is one 57 s generation. The probe guard is specified only for the floor step (2), not for the alternation step (1).
5. Carry-over bias: a shallow epoch inherits a deep-built population and gets credit for it. This is unmodelled.

### Design 3: ride-along deep lane (persistent deep FJ slots co-launched in the children grid, work quota, adaptive slot count) — 3.5/10
**Strengths.** 1. It is the only design that gives fast generations and deep repair at the same time rather than alternating them.
2. item_placement should stay close to lps0's pace by construction: the children are always shallow and D drifts to slots_min = 4.
3. LP rows always get deep repair.
4. Its kernel facts check out: STI is cp.zeros with N_SI = 8 and slots 0-6 used (ls_fj_gpu 78-79), so S_WORK = 7 is free; and `if (st[S_STATUS] != 0) return` exists, so paused blocks cost nothing.
5. It removes the serial LP-stream calls (glass4 lps0: 430 stream batches beside 220 generations).
6. Gate 0 is the right micro-benchmark for the co-residency question.

**Flaws.** 1. DEEP THROUGHPUT, NOT RESIDENCY, IS THE STRUCTURAL PROBLEM.
   - sing44's first feasible was an ordinary l2/lp child at gen 38/48. That is about 10k deep lanes at a 0.17 % feasible yield.
   - D starts at 16 and grows ×1.5 per 10 generations, so it needs about 60 generations (≈ 18 s at 0.3 s per generation) to reach 128, already later than lps1's 12.7 s first feasible.
   - Even at D_max, its own estimate is 40-190 completions/s vs lps1's 170/s, and the upper end only if co-residency is free (unmeasured).
   - So sing44 and ca slide toward lps0: sing44 lps0 0.551, ca s0 lps0 0.487.
2. ca is memory-capped at about 22 slots (11.6 MB per slot), fewer than its own 32 LP rows. Its quota per call is one shallow tail, which it estimates gives about 10 s after the LP vs 6 s in lps1.
3. The residency figure of 384 blocks (64 regs → 8 blocks/SM × 48 SMs) is derived from a cached cubin, not measured. Adding kernel arguments, a work counter and a rebase mode can raise the register count, and that lowers D_max.
4. Editing fj_steps and fj_init changes the compiled kernel for EVERY run, including runs with the lane disabled. Bitwise-equal outputs do not mean equal speed, so the lps0/lps1 baselines cannot be reused without re-measuring. The gpu-marked tests cannot run during the current experiments.
5. The CPU probe behind the item_placement prediction is unrepresentative: 253/256 shallow children ended feasible against a gen-0 cutoff, vs 1.7 % feasible ends in the real lps0 run.
6. route_lp is a second variable in the same arm.
7. Winner-take-all tier credit, with harvest latencies of several generations, under-credits deep on glass4. Its slots need about 7.8 calls each (idle tail share 0.10).
8. It has the largest implementation surface: a kernel edit, a Numba twin with its own RNG, a plan_memory term that can lower P, and an occupancy query.

**Recommendation.** Build a graft.

STEP 1 — Design 2's instrumentation and SHADOW mode, before any controller acts. It is host-only and changes no behaviour:
- FJGpu.run keeps last_steps from the st[:, S_STEPS] copy it already makes (ls_fj_gpu ~855), plus the per-launch (cum, t) pairs.
- The engine logs, per generation and per horizon level, Δ(h) and T(h), together with each feasible lane's S_STEPS and key.

Run it on lps1: item_placement, glass4, sing44, neos-960392, loue and ca, seed 0, about 15 min. This answers the open questions directly:
- Do glass4's 0.054-0.076 jumps come from late-finishing lanes?
- Does item_placement's predicted rate at a shallow level beat lps1's realized rate by more than the hysteresis? The realized per-generation ratio is 0.31 for a 4.7x time ratio, i.e. a rate ratio of about 1.45 against the 1.25 margin, so make the pass criterion quantitative.

STEP 2 — the active controller: Design 2's horizon ladder and the 10 s cap, with Design 1's safeguards grafted in.
(a) Phase test = EA-own feasibility (h_feas.any()), not `self.cutoff is None`. The cutoff includes inc_external (engine 1420), and glass4 has a finite inc_external at gen 0. Before the EA's own feasible: always unbounded depth, no cap.
(b) Materiality θ = 1e-3 in P(T) gap units, using Design 1's denominator max(|P_before|, |dual|), not cutoff − dual. Otherwise glass4's 16-20 polish steps per run drive it shallow.
(c) The one-shot LP batch (src 'lp', _lp_arrived 1740-1762) is always repaired at full depth, whatever level is in force.
(d) A periodic probe of \"none\" itself, not only +2 levels, so a stale deep rate can be re-measured.
(e) Off by default and popped from the hash.

STEP 3 — the verdict:
- Run against an interleaved same-window lps1 control.
- item_placement needs ≥ 4 seeds or a rule relative to that control: lps0's own seeds were 0.300 vs 0.417, with gap at 10 s 0.348 vs 0.705.
- Freeze the knobs before tuning-set use.

FALLBACKS:
- If shadow mode shows the counterfactual stays at \"none\" on item_placement, fall back to Design 1 with these fixes: LP batches pinned deep, longer phase-I epochs or a larger shallow floor, and a reward that keeps big-step credit (extreme or windowed-max).
- Park Design 3 until a GPU co-launch micro-benchmark (its gate 0) and a measured occupancy exist. Its kernel edit would also force re-measuring every baseline.

### 1. Phase-split, time-normalised depth bandit (one FJ depth per generation, deep prior) — 6.5/10
**Strengths.** Smallest change of the three. No CUDA edit and no Numba signature change: FJGpu.run passes np.int64(self.cfg.max_steps) and np.int64(self.cfg.stall_steps) as kernel arguments at every launch (ls_fj_gpu.py ~838), and FJCpu passes int(c.max_steps) and int(c.stall_steps) (ls_fj_cpu.py ~623). A per-call override is therefore a local substitution with no recompile, so compiled_on_clock stays False. No extra local-search call, so the _calls RNG counter and plan_memory are untouched. It draws no random number, and `_ls_run` passes the kwargs only when they are non-empty, which keeps the fixed-signature spy at test_strategy.py:362 valid with the feature off (verified). The hash pattern is correct: hash() pops a disabled section, and _build fills missing sections from defaults. The code claims check out: best_viol is a running minimum at engine.py 1336-1337 (init inf at 846); _new_incumbent fires at most once per generation; the arm covers children, pending batches and _admit inside one _step. Everything is CPU-testable, including a strong K=1 equivalence test. On measurement, it has a verdict rule fixed in advance, leaves the fhnw lottery and tiny-P neos-960392 out of the verdict, and adds a mechanism-agreement check (phase-F and phase-I time shares) so a lucky P cannot be credited to the mechanism.

**Flaws.** (1) The central claim, item_placement, is the weakest. Recomputed from steps.pkl, EA material gap-drop rates over 2-8 s on seed 1 are lps0 0.0000/0.0028/0.0134 vs lps1 0.0048/0.0050/0.0061 per s: a tie, below the 1.25 margin. So after the 4 alternation epochs the greedy rule keeps the deep prior, and shallow then gets only the 10 % floor: about one 2 s epoch per ~18 s, discounted with H=30 s. The P gap between depths opens at 5-10 s (portfolio gap 0.53 vs 0.75 at 10 s), so a switch at 20-40 s gives away most of the 0.21 gain, and the `<=0.45` target is a coin flip on seed 1. (2) The phase rule contradicts the reward. Phase I starts when inc_external is finite even if h_feas.any() is False. The phase-I reward then needs an EA-own feasible member and is structurally zero, while the signal that does move (best_viol) is no longer booked. The alternation runs blind and spends shallow epochs in what is still the EA's feasibility phase. (3) It cannot bound a deep generation it has already started: ca lps1 max_generation_s is 56.8/58.7. The design admits this; the probe guard only stops it from repeating. (4) The reward cannot see SCIP-seeded gains (glass4 SCIP-own credit 0.72 vs 0.68), and glass4 gives 0-2 EA jumps per run, so phase-I selection there rests on the prior. (5) The verdict uses lps0/lps1 recorded in other windows, with a same-window control on only item_placement and glass4. The loue threshold, at most max(lps1)+0.05 = 0.457, is already met by lps0 (0.440), so that half of criterion 2 cannot tell deep from shallow there. (6) All nine controller defaults were set from the same 36 dev traces they are judged on. Arm sequences depend on wall time, so a seed does not reproduce a run (they are logged).

### 2. Repair clock: per-call lane-step horizon enforced between launches, wall-clock cap, counterfactual controller — 7.5/10
**Strengths.** Its key technical claim is correct. In fj_steps an active block runs exactly `iters` steps per launch and leaves the loop only on its own stop status, so after every launch all still-active lanes have the same S_STEPS. A host-side truncation iters = min(_iters, H-cum) is therefore an exact equal-lane horizon with no kernel edit and no NVRTC recompile. The counterfactual identity holds: blocks are independent, the RNG is seeded per (seed, idx0+b, call), and no calls are added. The cutoff is constant across a generation's calls: _refresh_cutoff runs only in _exchange_in (1607, 1623) and in _after_survival AFTER _stagnation (1273-1275), so even stagnation immigrants share it. Per-lane S_STEPS and status already reach the host in `st = cp.asnumpy(STI[:kc])`, so no device read is added. Before the first incumbent, horizon=None and cap=None take the old code path, so the run is lps1 by construction; that protects sing44/ca/fhnw/loue phase F. After it, the controller starts at 'none' and moves only on evidence, which keeps the downside near lps1. It also fixes a real gap: today a run-level stop leaves later chunks unprojected (ls_fj_gpu 806-872). Its measurement plan is the most honest: a CPU identity test; Stage 1 shadow mode (6 runs, no behaviour change) whose pass criteria (a)/(b) can falsify the signal cheaply; an interleaved lps1 control on all 9 instances; and seeds 2-3 on item_placement and ca before any gain is written.

**Flaws.** (1) The design calls the myopia its own largest weakness. Δ_g(h) is computed from lanes of a deep-evolved population in one generation and cannot see the generation-count and recombination effect that is item_placement's advantage (EA steps of at least 0.01: 1.17 vs 0.17 per 10 s). So the controller may stay deep there. (2) A shallow horizon with stall 1000 is a new regime with no measured evidence. It is not lps0: lanes are cut mid-search instead of at a 100-step plateau, and with 98.7 % of item_placement lps0 lanes ending STALL (584,736 vs 7,909 FEASIBLE), nearly every lane runs to H. Whether that extra work is free depends on residency, which is unmeasured (Stage 0b). (3) The real implementation risk is engine plumbing. add_call must pair each call's per-lane status and steps with keys evaluated later, at three sites. _admit repairs only the `rep` subset (`_repair(Xr, strat[rep])`) while _evaluate(Xn) covers all rows, so an index map is needed. _repair applies the decoder after FJ, so FEASIBLE is pre-decoder, which is why the 'evaluate-feasible' condition must be kept. (4) The 10 s cap binds on ca only (max_generation_s 56.8/58.7; sing44's maximum is 8.44/9.17), so its only target rests on one seed. Once it fires, repair outcomes depend on timing; this is replayable only through the logged H_eff, and the CPU reference has different deadline semantics. (5) The Stage-2 success rule requires no instance worse than -0.05 d vs lps1 and puts fhnw (a one-event lottery: 0.302 vs 1.000, d of about ±0.7) and neos-960392 (0.007 absolute, about 0.18 in d) under that hard bound. The verdict can then pass or fail on luck even though the code path before the first incumbent is the same. 'Mean d vs lps1 > 0' over 9 instances × 2 seeds is a weak bar. (6) The ladder has 11 levels and n_j ≥ 5 each, and probes are capped at 10 %, so returning from a shallow level to 'none' takes several probe rounds of +2 levels.

### 3. Ride-along deep lane: persistent deep FJ slots co-launched with the children — 3.5/10
**Strengths.** It aims at the right GPU fact: a call waits for its slowest block, and there may be idle residency. Several claims check out. fj_steps uses 64 registers and a 1,040 B frame; I parsed .nv.info and found REGCOUNT 64, and 40 for fj_init. At 48 SMs that gives 384 resident blocks. STI slot 7 is free and zero-initialised (N_SI=8, S_RNG=6, cp.zeros). The uint16 origin bits are full (ORIGIN_X_AGREE=1<<15). CuPy exposes driver.occupancyMaxActiveBlocksPerMultiprocessor, so D_max can be computed at runtime. Gate 0, a co-launch micro-benchmark with pass ratios of 1.10 and 1.25, is a good, cheap falsification of the core assumption before any coupled run. Decisions use work counters, not the clock.

**Flaws.** (1) The disabled path is output-inert but not time-inert. fj_steps sits exactly at the 8-blocks/SM boundary: 64 regs × 128 threads = 8,192, and 65,536/8,192 = 8. Registers are allocated in steps of 8 per thread. The edit adds pointer selection, a work counter, quota and caps, so 72 regs gives 7 blocks/SM (336 resident, D_max 80), 80 gives 288 (D_max 32), and 88 or more gives 240 < 256: the children call itself then runs in two waves in EVERY arm, including lps1 controls. No __launch_bounds__(BD, 8) floor is proposed. (2) The item_placement prediction is contradicted by the run data. The CPU probe's '253/256 FEASIBLE at the shallow budget' compares with item_placement lps0 s0 at 7,909 FEASIBLE vs 584,736 STALL (1.3 %). Stalled candidates are abundant every generation, so the claim that D drifts to slots_min by itself has no support; D depends entirely on the winner-take-all credit. (3) The work formula does not charge what the text says. eval_var walks each candidate variable's full CSC column twice (the c0..c1 loops), so the cost of a step is proportional to the sum of column lengths over the chosen row, not Wn/BD. The 'off the critical path by construction' quota only holds if slots and children pick rows of similar column length, which is exactly what differs near an armed cutoff row. (4) Rebase keeps weights across calls and across cutoff changes. That is a new FJ variant: the spec resets weights to 1 every call. So lps1's evidence for deep repair does not transfer, and the CPU side needs a second Numba implementation (_fj_deep_batch) with its own splitmix/xorshift RNG to stay semantically equal. (5) route_lp=True is bundled into the primary arm. On ca and sing44 the LP rows are where deep repair paid (ca seed 0: an LP immigrant feasible 6 s after the LP), so Gate 1 cannot attribute the effect; Gate 1b runs only if Gate 1 passes. (6) The new plan_memory term (about 11.6 MB per slot on ca, m_aug 379,621) can lower P, which breaks the one-variable comparison. (7) The largest change surface: both kernels, persistent device state, a new pending source, the controller. Its timing claims cannot be tested on the CPU (no co-residency). Acceptance 'mean d vs lps1 ≥ -0.05 across the six' puts the fhnw lottery (d of about ±0.7, so ±0.12 on the mean) into the mean.

**Recommendation.** Build Design 2 first, in its staged form, with grafts from Design 1. The deciding trade-off is that Design 2 is the only proposal whose signal can be falsified without changing behaviour, and its worst case before the first incumbent is lps1 by construction. Design 1 changes less code, but its central claim cannot be checked before a timed run: on item_placement seed 1 the two depths tie at 2-8 s, below its 1.25 margin.

Order:
(a) Stage 0, CPU. The horizon rule in _fj_batch; the counterfactual-identity test on STEPS with the cutoff armed; the controller unit tests; the hash-invariance tests.
(b) Stage 0b, inside a leased GPU window. The occupancy query and the gpu-marked FJ tests.
(c) Stage 1, shadow mode: 6 runs at seed 0. Stop if criterion (a) or (b) fails. In that case fall back to Design 1.

Changes required in Design 2:
1. Take Design 1's verdict exclusions. Report fhnw and neos-960392 but keep them out of the 'no instance worse than -0.05' rule.
2. Take Design 1's mechanism-agreement criterion. Before crediting any P result, require: level time share mostly shallow after the incumbent on item_placement; 'none' or one level below on glass4 and loue.
3. Write the add_call pairing as one helper and test it on CPU at all three call sites (children, each pending batch, _admit's `rep` subset), so lane results map onto the evaluated keys.
4. Log the per-call (cum, t) profile and H_eff even in shadow mode. That is what answers the open glass4 question: do its large jumps come from lanes that finish late?
5. Add seeds 2-3 on item_placement and ca before writing any gain, as the design already says.

Keep Design 1 as the fallback if the myopic counterfactual fails in shadow mode. If it is used, fix two things. First, define phase I by the EA's own feasibility (h_feas.any()), not by a finite inc_external. Second, lower the margin or shorten the alternation so item_placement can switch within about 10 s.

Do not start Design 3 until a Gate-0-style micro-benchmark has run with the modified kernel. That benchmark must confirm fj_steps stays at 64 registers (or add __launch_bounds__(BD, 8)) so that runs with the lane disabled keep one wave. Separate route_lp from the lane.

## Run-data analysis (verbatim)

- **item_placement_9903 (the one that wants fast generations)** — P(120): deep (lps1) 0.569 (seed 0 0.558, seed 1 0.580); default (lps0) 0.358 (0.300 / 0.416); A3 default config 0.400 (0.522 / 0.277). The seed spread at the default depth is large, but all four default-depth runs (0.277 to 0.522) beat both deep runs. Evidence is robust.

Where the difference comes from: the EA. EA credit is 0.43 at the default depth vs 0.23 deep; SCIP's own credit is about the same (0.21 vs 0.20).
- Feasibility is trivial. The EA is feasible at 0.05 to 0.07 s in every run, and the gen-0 population is 83 to 84 % feasible at the default depth and 100 % deep.
- Both depths improve the incumbent in almost every generation (incumbents per generation 0.975 default vs 0.999 deep). Deep repair has 10.0 % of repaired lanes ending feasible-and-improving vs 1.7 %, and about 30 improving children per generation vs 5.7. The extra ones are redundant: only one per generation becomes the incumbent.
- The default depth runs 4.8x more generations (1,805 vs 380; 0.067 vs 0.316 s per generation), so it makes more and larger sequential improvements. EA portfolio steps of at least 0.01 gap: 1.17 vs 0.17 per 10 s. Total EA gap reduction 0.62 vs 0.42.
- The difference shows by 5 to 10 s (portfolio gap at 10 s: 0.53 default vs 0.75 deep).

Repair counters (default vs deep):
- Repair plus immigrant phases: 71 % of wall time (repair 30 % + immigrants 40 %) vs 93 %.
- Moves per repaired lane: 150 vs 742.
- Stall rule ends 98 % vs 90 % of repairs; feasible ends 1.7 % vs 10 %; zero real budget stops.
- Posts: 1,757 posted, 462 accepted, 422 became SCIP incumbents vs 378 / 359 / 353. At the default depth SCIP accepts only about 26 % of what the EA posts.
- SCIP nodes/s show no consistent depth effect (126 / 155 default vs 110 / 253 deep).
- **sing44** — P(120): deep 0.362 (0.320 / 0.404); default 0.551 (0.562 / 0.540); A3 default config 0.651. Robust on both seeds.

Where the difference comes from: the time to the EA's first feasible solution.
- Deep repair makes the EA feasible at 12.7 / 16.6 s. The first feasible solution is an ordinary child of the l2/lp strategy (gen 38 / 48), not an LP immigrant, and it cuts the portfolio gap from 0.985 to 0.50 / 0.58. The HiGHS LP point had arrived at 10.3 / 10.4 s.
- At the default depth the EA never becomes feasible before SCIP's own 50 s solution. Its first feasible solution comes at 62.7 / 52.9 s.
- Credit: EA 0.55 deep vs 0.14 default; SCIP's own 0.08 vs 0.31 (at the default depth SCIP's 50 s solution carries the portfolio).

Feasibility phase at the default depth:
- best_viol stays flat at 19.6 +/- 0.1 from about 5 s to about 50 s (300+ generations with no real decrease).
- The rhs strategies die out (population [121, 135, 0, 0]).
- Deep: best_viol goes 16.6 -> 13.7 -> 6.1 -> feasible.

Repair counters (deep vs default):
- Generations 80 vs 450 (1.5 vs 0.27 s per generation). Repair 93 % vs 64 % of wall time.
- Moves per lane 1,044 vs 120. Feasible ends 0.17 % vs 0.04 %; improving lanes per generation 0.44 vs 0.11.
- Posts 16.5 vs 22 (accepted 6.5 vs 9).
- The HiGHS LP arrives 2.5 s later at the default depth (12.9 vs 10.3 s): it runs in the EA process on the same 3 cores.

Phase dependence: once the cutoff row is armed, even the default repair slows from 0.13 s to as much as 2.4 s per generation (the cutoff row is long). After feasibility the default depth progressed at least as fast per second (about 0.29 gap in about 40 s, vs about 0.20 in 65 s deep). That comparison is suggestive only, since the starting points differ.
- **glass4** — P(120): deep 0.228 (0.229 / 0.228); default 0.323 (0.316 / 0.330). But the A3 default config at the default depth gets 0.247 (0.257 / 0.238), so this result is confounded (see caveats).

Where the difference comes from: SCIP dominates the portfolio, and the EA's contribution is a few large structural jumps.
- SCIP's own credit: 0.72 deep vs 0.68 default. EA credit: 0.048 vs 0.001 (A3 default 0.051).
- Deep EA portfolio steps: seed 0 at 32.4 s (-0.039) and 50.7 s (-0.037); seed 1 at 15.6 s (-0.044) and 37.2 s (-0.011).
- SCIP then makes large steps of its own (seed 0 at 42.6 s -0.055 and 44.4 s -0.034; seed 1 at 50.3 s -0.078). These are probably seeded by the EA's solutions, but that cannot be proven from the traces.
- At the default depth the EA's 198 to 200 portfolio steps are worth 0.001 in total. They are continuous-variable polish of about 1e-7 relative, and there are no EA steps of at least 0.01 (vs 0.074 per generation deep).
- The EA's first own feasible solution is later deep (5.5 / 10.9 s, from an LP immigrant) than at the default depth (1.5 / 2.1 s). This does not matter because SCIP is already ahead.
- The gap between the arms shows from about 20 s.

Counters (deep vs default):
- Generations 27 vs 222 (4.4 vs 0.54 s per generation). Repair plus immigrant phases 99.5 % vs 96 % of wall time; the immigrant phase alone is 23 % vs 30 to 35 %.
- Moves per lane 318 vs 85, but about 750 local minima per lane deep: most deep steps are weight bumps.
- Feasible ends 0.64 % vs 0.55 %; improving lanes per generation 2.4 vs 1.8; incumbents per generation 0.96 in both.
- Posts 24.5 deep (all accepted, 19.5 became incumbents) vs 208 / 208 / 196.5.

Count-based success signals cannot tell the two depths apart here; only the size of the improvements can.
- **ca__Scenario400_reserves_1** — P(120): deep 0.297 (0.298 / 0.296); default 0.395 (0.486 / 0.304). The A3 default config gets 1.000, because without HiGHS it has no LP point (SCIP's root LP never finishes). Evidence: one seed only; seed 1 is nearly equal, with first feasible at 35.3 s in both arms.

Where the difference comes from: the EA's first feasible solution, all EA credit (0.70 vs 0.61; SCIP never finds a solution itself).
- Seed 0 deep: an LP-rounding immigrant repaired 6 s after the HiGHS LP (29.3 s) becomes feasible at 35.5 s, already at gap 0.002.
- Seed 0 default: the LP immigrant at 33.7 s ends at V 0.042, not feasible. Feasibility comes only at 58.0 s, from an ordinary child.
- The LP itself arrives 3 to 4 s later at the default depth (32.4 to 33.1 s vs 29.3 to 29.5 s): the faster host loop competes with HiGHS for the same cores.

Feasibility phase:
- Deep: best_viol 10,356 -> 356 in 10 generations (14.6 s), with the rhs/lp strategy prospering.
- Default: 10,900 -> 9,450 in 30 generations (28 s), and the rhs strategies die out.

After feasibility, deep is very slow: one generation took 57 s (35.5 to 92.3 s), giving 2 EA improvements in 85 s, vs 6 to 10 at the default depth.

Counters (deep vs default): generations 18 vs 73; repair 89 % vs 60 % of wall time; moves per lane 950 vs 142; feasible ends 0.6 % vs 0.05 %; improving lanes per generation 1.7 vs 0.13; posts 1.5 vs 8 (1 accepted in each).
- **neos-3024952-loue** — P(120): deep 0.369 (0.332 / 0.407); default 0.440 (0.454 / 0.426). But the A3 default config at the default depth gets 0.344 (0.375 / 0.312), which beats deep. Strongly confounded.

Where the difference comes from: the EA (credit 0.55 vs 0.46). SCIP alone never finds a first solution here; its later own credit is 0.08 deep vs 0.11 default.
- Earlier first feasible: 2.1 / 2.2 s deep (a gen-7 child) vs 7.3 / 7.1 s default (A3 default config 5.3 / 4.7 s). P10 is 0.67 vs 0.87.
- Then larger EA jumps deep (0.03 to 0.12 gap at 5 to 56 s). EA steps of at least 0.01: 0.075 per generation and 0.29 per 10 s deep, vs 0.0035 and 0.125 default.
- The A3 default config gets there by many small 0.002 to 0.007 steps at about 2.5 generations per second (total EA gain 0.73 vs 0.61 deep vs 0.50 default).

Counters (deep vs default):
- Generations 46.5 vs 432 (2.6 vs 0.28 s per generation). Repair plus immigrant phases 99 % vs 94 %; immigrants alone take 28 % of wall time at the default depth.
- Moves per lane 800 vs 135.
- Feasible ends 1.05 % vs 0.044 %: 24x the yield for 9x the time per lane.
- Improving lanes per generation 2.9 vs 0.12; incumbents per generation 0.78 vs 0.12.
- Posts 35.5 / 35.5 / 32 vs 45.5 / 43 / 41.5.

At the same default depth, the adaptive run's yield is far below the A3 default config's (improving lanes per generation 0.12 vs 1.0). Something in the adaptive strategy, HiGHS LP or LP stream hurts this instance at the default depth, and deep repair hides it.
- **neos-960392** — P(120): deep 0.032 (0.033 / 0.031); default 0.039 (0.035 / 0.042); A3 default config 0.042. Both seeds agree, but the gap is small: 0.007 absolute, about 0.18 in bounded d only because P is tiny.

All credit is EA (0.97). SCIP proved optimality and the deep runs ended at 12.7 / 18.7 s, so whole-run generation counts (10.5 vs 232) cannot be compared.

Where the difference comes from: how much each generation improves.
- Deep generations (1.0 to 1.5 s) each cut the gap by 0.1 to 0.43 (gen 1: -0.38 / -0.43). Deep-repaired LP immigrants add -0.20 / -0.31 at 6.1 to 6.9 s. Gap 0.05 / 0.013 by about 6.5 s.
- Default generations (0.33 s) cut 0.01 to 0.2 each; gap 0.27 / 0.23 at 6 s.
- The lead shows only between 8 and 15 s (gap at 10 s: 0.01 vs 0.11).
- The EA's first own feasible is earlier at the default depth (0.5 vs 0.9 s).

Counters (deep vs default):
- Moves per lane 1,756 vs 396. Feasible ends 5.5 % vs 0.46 %. Improving lanes per generation 24 vs 1.5: as saturated as item_placement, yet deep still wins because each improvement is larger.
- This is the only instance with real step-budget stops: 7.7 % of default repairs end on max_steps 2000 (16 % in the A3 default config), i.e. lanes cut while still moving.
- Posts 8 / 3 accepted vs 33 / 5.5.
- **fhnw-binpack4-48** — P(120): deep 0.651 (seed 0 0.302, seed 1 1.000); default 1.000 / 1.000; A3 default config 1.000. Evidence: ONE seed.

The whole difference is one event: on deep seed 0 the EA found the optimum (objective 0) at 36.2 s, from a gen-6 child (toward_lp mutation, l2/lp strategy); SCIP accepted it at 36.4 s, and EA credit is 100 %. No other run in any configuration found a feasible solution.

Counters (deep vs default):
- Generations 7 / 37 vs 183 / 308 (4.2 vs 0.52 s per generation). Repair 99 % vs 94 to 96 % of wall time.
- Moves per lane 1,194 to 1,259 vs 113 to 132. Feasible ends 0.05 % / 0 vs 0. No real budget stops; the 12 % budget share on deep seed 0 is lanes interrupted at the end of the run.
- Feasibility phase: best_viol 0.91 / 1.13 at 10 s deep. Default 1.30 / 1.67 at 10 s, falling slowly to 0.47 / 0.93 at 120 s. The A3 default config reaches 0.40 to 0.50 early, then stalls.

Here neither best_viol stagnation nor its level cleanly favours deep.
- **radiationm18-12-05** — P(120) is the same everywhere: deep 0.234, default 0.235, A3 default config 0.237.
- SCIP does all the work (its own credit 0.76 to 0.77 in every arm; EA 0.003 or less). The EA never becomes feasible on its own except one late find (49 s default, 110 s deep, on seed 1).
- Deep costs only 1.7x in generations (295 vs 510). Repair is 41 % vs 22 % of wall time; immigrants take about 45 % in both.
- Depth is irrelevant and gives no signal. This instance needs a non-local operator (development log).
- **rail01** — P(120) is 1.000 in every run; no process finds a solution (only cuOpt alone does, outside these arms).
- Generations 681 deep vs 854 default: only 1.25x. Repair is just 30 % vs 15 % of wall time.
- At the default depth best_viol is lower (10 to 15) than deep (28 to 36), the opposite of sing44 and ca. Depth is irrelevant here.

**Across instances.** I ran no experiments; this is analysis of the existing traces only. My recomputed P(120) matches `datasets/runs.parquet` exactly.

**1. The depth lever is the stall patience, not the step budget.**
- The stall rule ends 90 to 100 % of repairs at both depths.
- Real budget stops (`ls_finished_budget` minus interrupted lanes) are about zero, except neos-960392 at the default depth (7.7 %; 16 % in the A3 default config).
- So the c01 change works through `stall_steps` 100 -> 1000.
- Moves per repaired lane: 85 to 475 at the default depth, 318 to 1,756 deep.
- On repair-bound instances deep costs 4 to 11x in generations; on rail01 and radiationm only 1.25 to 1.7x.

**2. What deep repair buys on the six.**
- (a) The first feasible solution: turning LP roundings or LP-guided children feasible. sing44 is 36 to 50 s earlier, loue 5 s, ca seed 0 23 s, fhnw seed 0 only deep, plus the LP-immigrant jumps on neos-960392.
- (b) Larger jumps per generation from more ambitious children (glass4, loue, neos-960392).
- On sing44, ca and loue, success at the default depth is rare: 0.04 to 0.05 % of lanes end feasible, and deep raises that 4 to 27x.

**3. What makes item_placement different.**
- Feasibility is trivial and the incumbent improves in about every generation at both depths, so the run is saturated.
- Deep repair only adds redundant improving children (about 30 vs 5.7 per generation).
- The scarce resource is generations: sequential improvement plus recombination. With 4.8x more generations the default depth gets more and larger steps (EA steps of at least 0.01 gap: 1.17 vs 0.17 per 10 s).

**4. Counting successes misleads; the size of each improvement must be weighed.**
- glass4 at the default depth shows as many "successes" as deep (improving lanes per generation 1.8 vs 2.4; incumbents per generation 0.96 in both), but they are worthless polish.
- On item_placement, deep has more 1e-3 steps per 10 s (11.1 vs 6.7) but fewer 1e-2 steps and less total gain.

**5. The right depth depends on the phase.**
- ca deep: one 57 s generation and 2 improvements after feasibility, vs 6 to 10 at the default depth.
- sing44: after feasibility the default depth progressed at least as fast per second.
- Once the cutoff row is armed, the step cost rises at any depth (sing44 default: 0.13 -> up to 2.4 s per generation).
- This argues against one fixed depth per instance: deep until the EA's own first feasible, then re-decide.

**6. Side effect.** The HiGHS LP in the EA process arrives 2.5 to 4 s later at the default depth (sing44, ca). The busier host loop shares the 3 cores with it.

**Online signals (cheapest and earliest first):**
- **S1, gen-0 repair feasibility.** The share of the repaired initial population that satisfies the original rows. item_placement: 0.67 to 1.00. Every other run, in all three configurations: 0.074 or less (54 runs). It is free and available at about 0.1 s. It is a hypothesis fit to one fast instance.
- **S2, redundancy per generation.** k = lanes of the children call that end feasible-and-improving, with a rule like the 1/5 success rule: shorten the repair when k is well above 1 for several generations, deepen when k = 0 and nothing else moves.
  - The summary-level k mixes in immigrant, LP and stream batches: deep 29 to 30 on item_placement, 18 to 30 on neos-960392, 3.5 or less elsewhere; default 4.5 to 7 on item_placement and 3 or less elsewhere.
  - On its own it would wrongly shorten neos-960392 (small cost) and would not deepen glass4 at the default depth.
- **S3, progress rate weighted by size.** The sum of incumbent improvements in primal–dual-gap units per second (the EA already receives the dual bound as `dual_key`), compared across depths by alternating epochs (bandit or racing).
  - From the trajectories, this separates item_placement by 5 to 10 s, neos-960392 and loue at 5 to 15 s, and glass4 at about 20 s.
  - sing44, ca and fhnw show their difference only through the first-feasible event (15 to 45 s), so they need S4.
- **S4, feasibility-phase stagnation.** If best_viol has not decreased for G generations, deepen. sing44 at the default depth was flat for 300+ generations; ca's best_viol fell 10,900 -> 9,450 in 30 generations at the default depth vs 10,356 -> 356 in 10 deep. Counter-case: on rail01 the default depth has the lower best_viol, which is harmless there.
- **S5, real step-budget stop share.** The only trace-visible sign of lanes "cut while still moving". Only neos-960392 shows it; deepen `max_steps` there.
- **S6, kernel counters to add.** This is the only way to answer "does the stall rule end repairs that were still improving". Per lane, record: the longest plateau broken before finishing, the step at finish, and the step at which the plateau counter first reached 100, 300 and 1000. The kernel is deterministic per seed, lane and call, so one deep call then gives the exact yield and step counts any shallower stall would have produced on the same children. That yields an improvement-per-second estimate per depth from one probe generation. It only evaluates shallower settings, so deep must be probed periodically.

**Mechanism options consistent with the data (none validated):**
- A controller using S1 + S3 + S4, with depth decided by phase.
- A resumable deep lane: the kernel already keeps per-lane state in global memory and relaunches in slices. A small batch (LP roundings, stream points, the best stalled children or Rank 0) could continue repairing across generations while the children call stays shallow. Per-individual depth inside one call would not shorten a generation, because the call waits for its slowest lane.

**Caveats.** **1. Confound with the adaptive configuration.** On glass4 (0.247) and loue (0.344) the A3 default config at the default depth matches or beats deep. On loue seed 1 it gets 0.312 vs 0.407 deep. lps0 (adaptive strategy + HiGHS LP + LP stream at the default depth) is the worst of the three on both. lps0 vs lps1 isolates depth within the adaptive configuration. But part of "these instances need deep repair" may really be "deep repair hides something in the adaptive configuration that hurts at the default depth". On loue, the adaptive run's improving lanes per generation at the default depth are 0.12, vs 1.0 for the A3 default config. One more one-variable run would separate the two: the strategy switched off at the default depth.

**2. Evidence grades.**
- Robust (both seeds, margin above 0.07): sing44, glass4, loue, item_placement. On item_placement all four default-depth runs beat both deep runs, but the default-depth seed spread is 0.277 to 0.522.
- One seed carries it: ca (seed 1 is 0.296 vs 0.304) and fhnw (seed 0 only: one find of the optimum).
- Tiny: neos-960392 (0.007 absolute; its runs end at proven optimality after 12.7 to 32.6 s).
- 2 seeds, 120 s. The two arms ran at different times (lps1 16:03 to 16:35, lps0 16:55 to 17:30). GPU health was ok on all runs.

**3. Counters that are not in the traces.**
- `ls_finished_*` and every "improving lanes per generation" number include the initial, immigrant, LP and stream repair batches, not only the children call.
- gen events are written every 10 generations.
- The genealogy keeps survivors only.
- Children-only, per-call counts would make S2 clean.

**4. The stall question cannot be answered from these traces.** "How often the stall rule ends a repair while still improving" is not observable; S6 is the counter that would answer it.

**5. Attribution.** EA vs SCIP credit is computed by first finder, as P = 1 - sum of gap drop x (T - t)/T. SCIP's own steps that were seeded by EA solutions count as SCIP's (likely on glass4).

**6. S1 is a hypothesis.** The gen-0 separator is fitted to a single fast instance; it needs more ML4CO-type instances and at least one other easy-feasibility instance.

**7. Use the portfolio-step table, not the own-incumbent one.** My scratch table `ea_steps.pkl` measures EA steps against the EA's own previous incumbent. It overstates EA progress where SCIP leads (it made deep look better on item_placement). Every EA number reported above comes from the portfolio steps in `steps.pkl`.

**8. Proposals only.** The controller, the deep lane and the kernel counters would all be code changes to the repository; nothing was changed. I made no edits to the repository, used no GPU, and ran every Python command pinned to cores 10–14, single-threaded.

Files are in `/tmp/itrea-design/depth_analysis/`:
- `collect.py` builds `runs.pkl` (one row per run, 54 runs) and `steps.pkl` (portfolio steps labelled ea / scip_self).
- `summary.py` writes the per-instance seed means to `summary_seedmean.csv`.
- `table1.py`, `table2.py`, `table3.py`, `traj.py`, `genview.py`, `feasphase.py`, `early.py`, `gen0.py`, `bigsteps.py` produce the tables quoted above.

## Literature (as reported; only "open full text" entries may be cited — verify before citing)

| key | citation | access | URL | relevance |
|---|---|---|---|---|
| ong2004metalamarckian | Yew Soon Ong, Andy J. Keane. Meta-Lamarckian learning in memetic algorithms. IEEE Transactions on Evolutionary Computation 8(2):99-110, 2004. doi:10.1109/TEVC.2003.819944. Copy read: University of Southampton ePrints 22794 (ong_04.pdf, scanned publisher-layout PDF), through an Internet Archive capture of 2024-05-10. | open full text | https://web.archive.org/web/20240510113131/https://eprints.soton.ac.uk/22794/1/ong_04.pdf | Origin of 'meta-Lamarckian learning'. The MA keeps a pool of nine local searchers and picks one for each chromosome. MA-S1 (sub-problem decomposition): after a training stage of g generations, use the LS with the best reward among the k nearest archived chromosomes. MA-S2 (biased roulette wheel): each LS gets one trial during training, then selection probability is proportional to its accumulated reward. Reward eta = beta*|pf - cf|/mu, i.e. fitness gain per LS function evaluation (wall-clock allowed instead), scaled relative to the best solution so far. LS length is FIXED at 100 evaluations and LS probability is 1, so the paper adapts WHICH local search runs, not how much. Evidence (Sphere, Griewank and Bump benchmarks, 20 runs each, plus an aerodynamic wing design): the adaptive strategies come close to the best single-LS MA without knowing it in advance. MA-S2 is judged most competitive because it has no control parameters; MA-S1 and a simple inheritance mechanism are sensitive to g, k and the innovation rate. Statistically, MA-S2 beats them on Sphere and Griewank and ties on Bump. For I-TREA II, the reward form (gain per unit of LS effort) is the natural credit for choosing a repair configuration. ACCESS CAVEAT: OpenAlex marks the Southampton location as not open, and the live server now serves a bot challenge (not bypassed). Captures from 2011 to 2024 show the PDF served publicly. Recheck live access before adding to refs.bib. |
| ong2006classification | Yew-Soon Ong, Meng-Hiot Lim, Ning Zhu, Kok-Wai Wong. Classification of adaptive memetic algorithms: a comparative study. IEEE Transactions on Systems, Man, and Cybernetics, Part B 36(1):141-152, 2006. doi:10.1109/TSMCB.2005.856143. Copy read: Murdoch University research portal, a repository copy in the journal's layout marked 'Research: Open' (also deposited in DR-NTU). | open full text | https://researchportal.murdoch.edu.au/esploro/outputs/journalArticle/Classification-of-adaptive-memetic-algorithms-A/991005540114507891 | Classifies meme (LS-choice) adaptation along two axes. By TYPE: static; adaptive-qualitative (only whether a meme improved, e.g. Randomdescent or tabu choice); adaptive-quantitative (reward values, e.g. Straightchoice, Roulettechoice, sub-problem decomposition); self-adaptive (meme carried in the genome, i.e. multimemes). By LEVEL of historical knowledge: external, local (nearby individuals) or global (whole population history). On continuous benchmarks, every adaptive scheme is competitive with or better than the best fixed-meme MA after a learning stage. External-level adaptation (no on-line learning) is worst, global-level ranks best on every benchmark, and the overhead differences are negligible. Citing Ong and Keane, it notes that a larger meme pool lengthens the learning stage. The authors attribute the global-level win partly to ease of implementation and to the uniform landscapes of their test functions, and call for more work on local-level and self-adaptive schemes. Like Ong and Keane, it adapts the choice of LS, not its intensity. For I-TREA II, the level axis separates one repair configuration per run (global) from one per individual or lineage (local or self-adaptive, as the current strategy inheritance does). |
| krasnogor2005tutorial | Natalio Krasnogor, Jim Smith. A tutorial for competent memetic algorithms: model, taxonomy, and design issues. IEEE Transactions on Evolutionary Computation 9(5):474-488, 2005. doi:10.1109/TEVC.2005.850260. Copy read: UWE Bristol research repository (01514472.pdf, in the journal's layout), through an Internet Archive capture of 2021-12-15; the live repository served a bot challenge to this host. | open full text | https://web.archive.org/web/20211215090726/https://uwe-repository.worktribe.com/preview/1046942/01514472.pdf | Models an MA as local searchers applied by 'schedulers'. A fine-grain scheduler sees one individual and decides when and where. A coarse-grain scheduler sees the population and can decide which LS, how often, on whom and with what intensity from population statistics. A meta-scheduler also uses history. Memes are defined as static, adaptive or self-adaptive, and adaptation can live in one population-level pdf or in a pdf per individual. The design checklist names 'which individuals are improved', 'where and when LS is applied' and 'how much CPU budget goes to LS'. It notes that most MAs apply LS to every individual every generation and that this is not required. It cites Land's 'sniffs' (a short partial LS probes a basin; only promising basins later get an extended budget). From the authors' TSP results, the best LS operator is instance- and time-dependent, so they recommend keeping several with adaptive probabilities. This is a framework paper with little head-to-head evidence on intensity rules. For I-TREA II, the strategy redraw/inheritance is a self-adaptive meme in this taxonomy, and a coarse-grain scheduler is where a per-individual step budget would sit. |
| smith2007coevolving | J. E. Smith. Coevolving memetic algorithms: a review and progress report. IEEE Transactions on Systems, Man, and Cybernetics, Part B 37(1):6-17, 2007. doi:10.1109/TSMCB.2006.883273. Copy read: author preprint (18169.pdf) in the UWE Bristol research repository, through an Internet Archive capture of 2023-11-15. | open full text | https://web.archive.org/web/20231115114834/https://uwe-repository.worktribe.com/preview/1029809/18169.pdf | COMA evolves a population of local-search rules ('memes') beside the solutions. A meme encodes the pivot rule (greedy or steepest), the DEPTH (iterations, from one improving step to local optimality), a pairing rule and a condition:action move pattern of evolvable length. Pairing is linked (the meme is inherited with the solution, which is self-adaptation), random, or fitness-based tournament. A meme's credit is the fitness gain at its last application. Evidence comes from binary benchmarks (concatenated deceptive traps of 40-200 bits, and H-IFF), with population 500 and 50 runs. A fixed bit-flip MA did very poorly. Random pairing with steepest ascent fell from 50/50 successes at 40 bits to 5 at 100 bits and 0 above. Linked and tournament-paired variants solved up to 200 bits, and evaluations grew near-linearly for the best three. The gain comes from discovering and reusing problem regularities. For I-TREA II, this is the direct precedent for genetically encoding repair parameters (depth included) and inheriting them from a parent; the 'linked' variant matches the current first-parent inheritance. There is no MIP evidence. |
| sudholt2012parametrization | Dirk Sudholt. Parametrization and balancing local and global search. In F. Neri, C. Cotta, P. Moscato (eds.), Handbook of Memetic Algorithms, Studies in Computational Intelligence 379, Springer, 2012, pp. 55-72. doi:10.1007/978-3-642-23247-3_5. Copy read: arXiv:1109.6441 preprint (2011), titled 'Memetic Algorithms: Parametrization and Balancing Local and Global Search'. Theorems 3-4 come from Sudholt, Theoretical Computer Science 410(26):2511-2528, 2009, doi:10.1016/j.tcs.2009.03.003 (closed). | open full text | https://arxiv.org/abs/1109.6441 | Survey plus theory of LS FREQUENCY (every tau generations, or with probability p_LS) and LS DEPTH (a cap on improving steps). It is my source for Hart (1994), Land (1998), Ishibuchi et al. (2003) and Sindhya et al. (a cyclic p_LS schedule); see mechanisms. It tabulates factors that lower the optimal amount of LS: stronger GA exploration, larger neighbourhoods, steepest descent, expensive non-incremental evaluation, and multi-objective problems. Theory: a (mu+lambda) MA optimises a constructed function in polynomial time with depth delta = D, but needs superpolynomial time if |delta - D| >= log^3 n. For the (1+1) MA, halving the frequency from 1/tau to 1/(2tau) turns polynomial into exponential time on one function, and does the reverse on another. The conclusion is that depth and frequency interact ('the strongest dependency') and that no a-priori setting is safe, which motivates adaptive or per-instance tuned control. For I-TREA II, this is the theoretical case against a single fixed repair step budget across instances. |
| nguyen2009probabilistic | Quang Huy Nguyen, Yew-Soon Ong, Meng Hiot Lim. A probabilistic memetic framework. IEEE Transactions on Evolutionary Computation 13(3):604-623, 2009. doi:10.1109/TEVC.2008.2009460. Copy read: DR-NTU (Nanyang Technological University) repository, a copy in the journal's layout. | open full text | https://dr.ntu.edu.sg/bitstream/10356/90539/1/A%20Probabilistic%20Memetic%20Framework.pdf | Treats each step as a choice between evolution and individual learning, and derives an upper bound on the learning intensity per individual (LS evaluations): t_il <= (t_gs/n) * ln(1-p2)/ln(1-p1). Here p1 and p2 are the probabilities that evolution and local search reach the global optimum's basin; the bound does not depend on LS frequency. The approximate version (APrMF) estimates p1 and p2 per individual from the archived LS traces of its q nearest neighbours. It runs LS only if the expected required intensity is within the bound, and otherwise skips LS for that individual. On a step function, a derivative-based LS gets a bound <= 1 and is therefore not used. Tested on continuous benchmarks against several EAs, MAs and hybrids. For I-TREA II, this is a principled per-individual 'spend or skip' rule built from past repair traces of similar individuals. The nearest-neighbour bookkeeping may not pay off at GPU batch scale. |
| molina2010maswchains | Daniel Molina, Manuel Lozano, Francisco Herrera. MA-SW-Chains: memetic algorithm based on local search chains for large scale continuous global optimization. IEEE Congress on Evolutionary Computation (WCCI 2010), 2010. doi:10.1109/CEC.2010.5586034. Copy read: the authors' group site (SCI2S, Universidad de Granada). It is the open proxy for the closed journal paper Molina, Lozano, Garcia-Martinez, Herrera, 'Memetic algorithms for continuous optimisation based on local search chains', Evolutionary Computation 18(1):27-63, 2010, doi:10.1162/evco.2010.18.1.18102 (MA-CMA-Chains). | open full text | https://sci2s.ugr.es/sites/default/files/ficherosPublicaciones/1290_maswchains_large.pdf | 'Local-search chains'. Each LS call gets a fixed intensity stretch I_str (500 evaluations). The LS state (Solis-Wets step size and success/failure counters) is stored with the individual and resumed the next time it is chosen, so an individual chosen n times has received n*I_str evaluations; intensity grows only for individuals that keep being selected. A fixed ratio r_L/G = 0.5 of all evaluations goes to LS, and the GA runs n_frec = I_str(1 - r_L/G)/r_L/G evaluations between LS calls. LS goes to the best individual that has never had LS or whose last LS gain exceeded delta_LS. If none qualifies, the population restarts with the best kept. The authors say the fixed ratio prevents drift into 'super exploitation'. Evidence: CEC 2010 large-scale benchmarks, and CEC 2005 in the journal paper; Cotta (2025) also reports LS chains effective in continuous settings. For I-TREA II, resumable repair (keep the FJ weights/state with the individual) plus a fixed global LS share is a cheap alternative to per-individual budgets. |
| cotta2025harnessing | Carlos Cotta. Harnessing memetic algorithms: a practical guide. TOP 33(2):327-356, 2025. doi:10.1007/s11750-024-00694-8. CC BY 4.0 (publisher open access; also Universidad de Malaga repository hdl:10630/37778). | open full text | https://link.springer.com/content/pdf/10.1007/s11750-024-00694-8.pdf | A recent open practical guide. It names LS breadth (neighbours per step) and depth (trajectory length) as high-impact parameters. Because theory shows inherent barriers, it points to automatic configuration (irace, SMAC, ParamILS). It describes partial Lamarckianism (Houck et al. 1997: LS on only some offspring, or with a probability) as effective in several domains and useful when LS is expensive. It covers LS chains, and meta-Lamarckian, multimeme and coevolving MAs as adaptive options. Its case study charges LS work against a shared evaluation budget. It is the citable open survey to use in place of the closed Neri & Cotta (2012) and Chen et al. (2011) surveys. |
| fialho2010analyzing | Alvaro Fialho, Luis Da Costa, Marc Schoenauer, Michele Sebag. Analyzing bandit-based adaptive operator selection mechanisms. Annals of Mathematics and Artificial Intelligence 60:25-64, 2010. doi:10.1007/s10472-010-9213-y. Copy read: HAL inria-00519579 author manuscript, through an Internet Archive capture; HAL served a bot challenge to this host. | open full text | https://web.archive.org/web/2020/https://inria.hal.science/inria-00519579v1/file/banditAOS-AMAI10.pdf | Adaptive operator selection has two parts: credit assignment (instantaneous, windowed-average or windowed-extreme fitness gain) and a selection rule (probability matching PM, adaptive pursuit AP, UCB multi-armed bandit, dynamic MAB with a Page-Hinkley restart, or the new sliding-window MAB). Hyper-parameters were racing-tuned first, then tested on artificial reward scenarios and Royal Road. Extreme credit is best when good operators give rare large gains. PM is consistently poor, and AP is dominated by the bandits; the authors blame the fixed minimum selection probability p_min and slow change detection. Plain UCB reacts slowly after long steady periods. DMAB is often best but its restart threshold must be tuned; SlMAB is nearly as good with fewer parameters. For I-TREA II, a bandit over repair configurations (step-budget levels x row scale x LP guidance) is a direct alternative to self-adaptive inheritance. The project's 5 % per-strategy floor is a p_min-style floor of the kind this paper found slows adaptation, so its size matters. |
| hendel2022adaptive | Gregor Hendel. Adaptive large neighborhood search for mixed integer programming. Mathematical Programming Computation 14(2):185-221, 2022 (online 2021). doi:10.1007/s12532-021-00209-7. CC BY 4.0. | open full text | https://link.springer.com/content/pdf/10.1007/s12532-021-00209-7.pdf | ALNS in SCIP (on by default since SCIP 5.0). A bandit (epsilon-greedy, EXP3 or alpha-UCB) chooses among eight LNS neighbourhoods. The reward in [0,1] combines new-incumbent, closed-gap and failure terms, scaled by effort (sub-MIP nodes relative to the limit). INTENSITY CONTROL: each neighbourhood has a target fixing rate. It is lowered when the sub-MIP is solved or infeasible within budget, raised when no solution is found, and kept otherwise; the step is 0.2, shrinking x0.75 per update. A global stall-node limit is affine in the main tree's node count, and the heuristic runs at SCIP frequency 20. In the calibration simulation ALNS ran only about 18-20 times per instance, so learning has few rounds; alpha-UCB > epsilon-greedy > EXP3. On MIPLIB 2017 brackets: primal integral -18 % and -20 %, time -2 % and -5 %; solved-count differences were traced to performance variability. This is the closest MIP analogue of adapting a local-search configuration and its intensity online, measured on the same primal integral I-TREA II uses. |
| chmiela2023online | Antonia Chmiela, Ambros Gleixner, Pawel Lichocki, Sebastian Pokutta. Online learning for scheduling MIP heuristics. CPAIOR 2023, Lecture Notes in Computer Science, Springer, pp. 114-123. doi:10.1007/978-3-031-33271-5_8. Open preprint arXiv:2304.03755. | open full text | https://arxiv.org/abs/2304.03755 | One bandit agent schedules both LNS and diving heuristics in SCIP and adapts each heuristic's working limit. The LNS target fixing rate is multiplied by (1-gamma) after a success or an infeasible sub-MIP and by (1+gamma) otherwise (gamma = 0.1, bounds 0.3-0.9). The diving LP-resolve threshold is updated the same way (eta = 0.1, bounds 0.05-0.3). A failure therefore makes the next call more expensive but more likely to succeed. It uses a modified epsilon-greedy because, in their tests, exploitation beat exploration for primal heuristics: a heuristic that does badly early rarely recovers. The reward adds gap, failure and conflict terms to the rare success signal. Evidence: consistent node reductions on the MIPLIB 2017 benchmark and a 4 % speedup on instances needing >= 1000 s. For I-TREA II, the multiplicative success/failure update is a cheap, validated template for a repair step budget. |
| luteberget2023feasibility | Bjornar Luteberget, Giorgio Sartor. Feasibility Jump: an LP-free Lagrangian MIP heuristic. Mathematical Programming Computation 15(2):365-388, 2023. doi:10.1007/s12532-023-00234-8. CC BY 4.0. Copy read: the authors' preprint dated 2023-01-09 (the published version is also open at Springer). | open full text | https://luteberget.github.io/preprints/feasibilityjump-2023-01-09.pdf | FJ's budget is a deterministic effort count (the sum of the bounds of every loop executed), not wall-clock time. It stops after a set effort since the last improvement of the weighted violation, or after a total effort; both thresholds were tuned on MIPLIB 2017. The adaptive element is the constraint weights, raised at each local minimum, not the budget. FJ ran as two threads (original and presolved problem) beside Xpress. The authors saw no significant improvement in FJ when increasing the computational effort, and argue that a quick heuristic is preferable inside a solver. For I-TREA II, whose repair is FJ, this is evidence that FJ's returns flatten fast, and that effort units, not time, are the right currency for a deterministic per-individual budget. |
| lin2024localmip | Peng Lin, Mengchuan Zou, Shaowei Cai. An efficient local search solver for mixed integer programming. CP 2024, LIPIcs 307, 19:1-19:19. doi:10.4230/LIPIcs.CP.2024.19. Open (Dagstuhl DROPS). The journal version, Lin, Cai, Zou, Lin, Artificial Intelligence 348:104405, 2025, doi:10.1016/j.artint.2025.104405, is closed. | open full text | https://drops.dagstuhl.de/storage/00lipics/lipics-vol307-cp2024/LIPIcs.CP.2024.19/LIPIcs.CP.2024.19.pdf | Local-MIP uses mixed tight, lift and breakthrough moves, a two-level score, a PAWS-style probabilistic weighting of constraints and objective (smoothing probability 0.0003) and tabu tenure 3 + rand(10). It runs until a time cutoff (evaluated at 10, 60 and 300 s) with no adaptive budget or restart rule; its adaptivity is in the weights and move scoring, not the intensity. Included as negative evidence for the task. For I-TREA II, it is a stronger objective-chasing repair than FJ but gives no guidance on how long to run it per individual; Spoorendonk's patience and PaNGEA's per-node caps supply that for Local-MIP-style search. |
| spoorendonk2026highsheur | Simon Spoorendonk. Presolve heuristics in HiGHS: implementation and computational study. arXiv:2609.22938 [math.OC], v1, 19 Sep 2026. | open full text | https://arxiv.org/abs/2609.22938 | Runs FJ, FPR, LocalMIP and Scylla in HiGHS under one budgeting layer, on independently seeded parallel workers. Each heuristic has an effort budget and a patience, in multiples of nnz<<10. The patience stops a heuristic after effort k without improving the best objective, capped at a quarter of the budget (the ratio HiGHS's own FJ uses). A presolve-only probe ran each heuristic alone on 233 mipfeas instances with unbounded effort and a 30 s wall-clock cap, logging PRODUCTIVE effort (before the last improvement) and STALE effort (after it). Medians: FJ 1.6e7 vs 7.1e10, FPR 1.4e6 vs 6.8e10, LocalMIP 1.2e8 vs 4.6e10, Scylla 2.8e6 vs 8.8e9. Stale effort exceeds productive effort by about 370x (LocalMIP) to about 49,000x (FPR); the stale tail is bounded by the 30 s clock, not intrinsic. Defaults are set from the probe: effort e_h = median effort at the last improvement, patience k_h = min(q95 of inter-improvement effort, e_h/4). A racing search over mix, budgets and patiences then follows. Result: primal integral -23.7 % on all instances and -12.9 % on held-out instances. Scylla and the LP-guided dive ship at effort 0; FJ ships with a small effort (0.33) and no patience. For I-TREA II, this is the closest evidence on how much FJ/LocalMIP budget is productive on MIP, and a measurable recipe (productive vs stale effort) for setting the repair budget and a stall rule. |
| pauphilet2026pangea | Jean Pauphilet, Yupeng Wu. PaNGEA: Parallel Node Generation and Exploration Algorithm on GPU. arXiv:2610.03090, v1, 2 Oct 2026. | open full text | https://arxiv.org/abs/2610.03090 | The only work found that documents a GPU batch of local searches bounded by its slowest member. B = W x R nodes (subproblems) run batched PDLP relaxations and batched Local-MIP-style searches that share kernels. In the authors' words, 'each main loop iteration terminates once all nodes have completed their work', so a larger batch 'raises the risk of one node slowing down the others'. Each node's local search stops at 5 s, 2,000,000 iterations, or 2 s without improvement. Relaxations can end early (solved or infeasible), but local search has no such certificate, so every node runs to its limit. The gap integral is U-shaped in W on MIPcc26: 124.7 at W=1, 118.1 at W=4, and worse than W=1 at W=16. The fix is an adaptive rule that sets W from instance size and free GPU memory; no single W dominates. A background CPU Local-MIP thread adds intensification because batching gives each GPU search fewer iterations. For I-TREA II: per the repair.py docstring, the GPU repair runs one thread block per individual and counts finished_feasible, finished_stall and finished_budget. The per-individual step budget and stall rule therefore set how long a batched repair waits for its slowest individual, and the batch size trades breadth against that wait. I did not read ls_fj_gpu.py, so how a launch actually ends there is unverified. |

**Mechanisms.** SCOPE. Mechanisms for adapting how much local search (LS) a memetic or hybrid algorithm spends, where, and on whom. All works were read in full at legal copies unless marked 'via' (secondary) or listed under CLOSED. Evidence base: the memetic-algorithm results (Ong, Nguyen, Molina, Smith, Fialho, Sudholt's theorems) are on continuous or artificial benchmark functions. The results closest to I-TREA II are the MIP line (Hendel, Chmiela, FJ, Spoorendonk, PaNGEA) on MIPLIB 2017, mipfeas and MIPcc26, mostly measured with the primal integral.

1. FIXED BUT TUNED INTENSITY (the baseline every adaptive scheme is measured against).
- Knobs: frequency (LS every tau generations, or with probability p_LS: 'partial Lamarckianism', Houck et al. 1997, via Cotta 2025), depth (a cap on improving steps or evaluations), breadth (neighbours per step), and the fraction of the population improved.
- Hart 1994 PhD thesis (via Sudholt 2012): with large populations, infrequent LS works best; more LS is needed when the GA cannot locate promising regions; elitism reduces the need.
- Ishibuchi, Yoshida, Murata 2003 (closed; via Sudholt 2012): neighbours examined per step (k) and p_LS are strongly negatively correlated at the optimum, and performance is best when k*p_LS lies in [1, 10]. In effect, total LS effort is what must be set.
- Sudholt (2009 theory, read through the 2012 chapter): halving the frequency, or setting depth log^3 n away from the ideal, flips polynomial and superpolynomial or exponential time, in either direction. Depth and frequency interact, and no fixed setting is safe a priori.
- MIP evidence: FJ's effort thresholds were tuned once on MIPLIB 2017, and more effort gave no significant gain (Luteberget & Sartor 2023). Spoorendonk 2026 race-tuned budgets and patiences on mipfeas (primal integral -23.7 % all, -12.9 % held-out); two of the heuristics were best at zero effort.
- Conclusion: in MIP, a small, offline-tuned budget plus a stall rule is the measured working baseline. The theory says the best value is instance-dependent.

2. STALL / PATIENCE RULES (call length adapts, nothing is learned).
- Stop after X effort without improvement: FJ (effort since the last improvement, plus a total cap), Spoorendonk (patience = min(q95 of inter-improvement effort, e/4)) and PaNGEA (2 s without improvement, plus 5 s and 2M iteration caps).
- Evidence: Spoorendonk's probe found median stale effort 370x to 49,000x the productive effort for FJ, FPR, LocalMIP and Scylla (capped by a 30 s clock).
- Why it works: improvements arrive early, so cutting the tail is nearly free. It is the cheapest mechanism, and I-TREA II's finished_stall counter (repair.py docstring) is of this kind.

3. WHO GETS LS (per-individual frequency).
- Similarity: Hart (via Sudholt) lowers LS frequency for duplicates and, by a genotype distance, for similar individuals.
- Fitness bias has a cost: per Land 1998 (via Sudholt), biasing LS toward the fittest loses diversity, and individuals near local optima gain least.
- Land's 'sniffs' (via Sudholt; Krasnogor & Smith 2005): a short partial LS estimates gain per effort ('local search potential'), and only promising basins get an extended budget. Drawbacks: the sniff itself costs effort, and early progress may not predict later progress.
- Molina et al. LS chains: LS goes to the best individual that is new or whose last LS gain exceeded delta; the global LS share is fixed.
- Nguyen et al. PMF: each individual's spend-or-skip decision comes from a bound on useful intensity, estimated from its nearest neighbours' LS traces.
- Sindhya et al. (via Sudholt): a cyclic schedule raises p_LS linearly from 0 to 1/popsize and drops it back.

4. ADAPTIVE DEPTH / BUDGET PER CALL OR INDIVIDUAL.
- LS chains (Molina 2010): fixed stretches (I_str = 500 evaluations), LS state stored with the individual and resumed, and a fixed LS share r_L/G against 'super exploitation'. Evidence on continuous CEC 2005/2010 benchmarks.
- PMF (Nguyen 2009): a theoretical per-individual upper bound on intensity; continuous benchmarks.
- MIP working-limit control: Hendel's target fixing rate (additive step 0.2, shrinking x0.75) and Chmiela's multiplicative rule (gamma = eta = 0.1). After a success the next call is made cheaper; after a failure it is made larger or more likely to succeed. Both were validated inside SCIP on MIPLIB 2017. This success/failure rule is the simplest transferable template for a repair step budget.

5. ON-LINE CHOICE OF AN LS CONFIGURATION (meme selection, AOS, bandits).
- Meta-Lamarckian learning (Ong & Keane 2004): reward = gain per LS evaluation, and a biased roulette wheel after one training trial per LS. It is robust when the best LS is unknown and its cost is a learning stage that grows with pool size (Ong et al. 2006). Ong et al. 2006: global-level adaptation ranks best on continuous benchmarks, which the authors attribute partly to uniform landscapes. NOTE: both papers keep LS length fixed (100 evaluations) and LS probability at 1; they adapt which LS runs, not how much.
- AOS (Fialho et al. 2010): extreme-value credit when good operators give rare large gains; dynamic/sliding-window bandits beat adaptive pursuit, which beats probability matching; a fixed p_min floor slows adaptation; restart thresholds need tuning.
- MIP bandits: in Hendel's simulation alpha-UCB > epsilon-greedy > EXP3. Chmiela found an exploitation-leaning epsilon-greedy better for primal heuristics. Hendel's ALNS made only about 18-20 decisions per instance in calibration, so priors and initial rounds matter. An EA repairing every offspring makes far more decisions per run, which helps learning.
- Intensity levels can themselves be bandit arms, or each arm can carry its own adapted intensity (Hendel).

6. SELF-ADAPTIVE (genetically encoded) LS PARAMETERS.
- Taxonomy: Krasnogor & Smith 2005 (static, adaptive and self-adaptive memes; schedulers).
- Smith 2007 COMA: memes encode the pivot rule, depth, pairing and move rule. Linked inheritance (self-adaptation) and tournament pairing found and reused problem regularities on traps and H-IFF, with near-linear scaling; random pairing failed beyond 100 bits. Credit is the last fitness gain, so selection on memes is indirect and noisy. Ong et al. 2006 list self-adaptive schemes as under-explored.
- No MIP evidence found. I-TREA II's strategy inheritance (from the first parent, redrawn with p = 0.1, every strategy kept at >= 5 % of offspring; itrea2-adaptive.yaml header) belongs to this class. Its 5 % floor plays the role of the p_min that Fialho et al. found slows adaptation.

7. GPU BATCHES BOUNDED BY THEIR SLOWEST MEMBER.
- PaNGEA (Pauphilet & Wu 2026) is the only explicit source. Its loop waits for all B nodes, each local search is capped (5 s, 2M iterations, or 2 s without improvement), the gap integral is U-shaped in batch width (best at W = 4, worse than W = 1 at W = 16), W is set from instance size and GPU memory, and a CPU Local-MIP thread supplies depth that the batch cannot.
- Also read: Corduk & Sielski, cuOpt GPU primal heuristics, arXiv:2510.20499. Its GPU-Local-MIP balances load WITHIN one search (a prefix-sum mapping of warps to 32 nonzeros instead of a block per row), not across a batch of searches.
- NEGATIVE SEARCH (2026-10-06). Queries: 'GPU memetic algorithm batched local search variable length load imbalance thread divergence slowest individual budget' and 'Luong Melab Talbi GPU computing for parallel local search metaheuristic algorithms'. They returned only neighbourhood-level parallelism inside one LS (Luong, Melab, Talbi, IEEE Trans. Computers 62(1):173-185, 2013, doi:10.1109/TC.2011.206; not read) and a GPU memetic paper hosted only on ResearchGate (not citable). No work was found that adapts per-individual LS budgets inside a GPU-batched EA to cut the wait for the slowest individual.

8. CONSISTENT FINDINGS ACROSS BOTH LINES.
(a) Improvements come early and stale effort dominates, so stall or patience rules pay.
(b) The right intensity is instance-dependent and interacts with frequency, so adapt online or tune per instance class.
(c) Bandits beat probability matching, and exploitation-leaning bandits suit primal heuristics.
(d) Multiplicative success/failure updates of a working limit are simple and validated in SCIP.
(e) On a GPU, batch width and per-member caps trade breadth against waiting for the slowest member.
Open gap: no source measures self-adaptive or bandit control of LS intensity for MIP inside an EA, or on a GPU.

CLOSED / NOT READ (for the paywalled list):
- Chen, Ong, Lim, Tan, 'A multi-facet survey on memetic computation', IEEE TEVC 15(5):591-607, 2011, doi:10.1109/TEVC.2011.2132725. Closed per OpenAlex; abstract only.
- Neri & Cotta, 'Memetic algorithms and memetic computing optimization: a literature review', Swarm and Evolutionary Computation 2:1-14, 2012, doi:10.1016/j.swevo.2011.11.003. Closed.
- Ishibuchi, Yoshida, Murata, IEEE TEVC 7(2):204-223, 2003, doi:10.1109/TEVC.2003.810752. Closed; known only via Sudholt.
- Molina et al., Evolutionary Computation 18(1):27-63, 2010, doi:10.1162/evco.2010.18.1.18102. Closed; open proxy is MA-SW-Chains.
- Davies, Didier, Perron, ViolationLS, CPAIOR 2024, LNCS 14742:243-258, doi:10.1007/978-3-031-60597-0_16. Paywalled per the vault; not read.
- Sudholt, Theoretical Computer Science 410(26):2511-2528, 2009, doi:10.1016/j.tcs.2009.03.003. Closed; results read through the open chapter.
- Hart 1994 and Land 1998 PhD theses (UC San Diego). No open copy found in one search; known via Sudholt and Krasnogor & Smith.
- Houck et al. 1997 (Evolutionary Computation 5(1)), via Cotta.
- Not checked: Jakob 2010 cost-benefit multimeme adaptation (Memetic Computing).

ACCESS NOTES. The live hosts served bot challenges to this machine (not bypassed), so four works were read through Internet Archive captures of institutional repositories: Ong & Keane 2004 (Southampton; OpenAlex marks it not open, so recheck live access before citing), Krasnogor & Smith 2005 and Smith 2007 (UWE), and Fialho et al. 2010 (HAL). OpenAlex and Crossref were queried without any contact e-mail, and no personal data was sent. Scratch copies are in /tmp/itrea-design/lit/; nothing in the repository was touched.

## Code map (verbatim)

**budget_semantics.** Paths are relative to the repository root. Numbers labelled MEASURED come from the 36 run dirs traces/raw/dev/*__A3@lps{0,1}__*: their ea.summary.json and the `seconds` field of their `gen` events. The scripts that read them are in /tmp/itrea-design/ (gen_stats.py, calls.py, phases.py, series.py, chunks.py).

DEFAULTS (src/itrea/config.py RepairCfg 115-138):
- method "fj" (117)
- max_steps 2000 (120). The comment there says "fj: moves per child per generation at most".
- stall_steps 100 (121)
- weight_bump 1.0 (128), max_candidates 16 (129), p_random_walk 0.0 (130), row_scan_cap 0 (131)
- Validation: max_steps >= 0, stall_steps >= 1 (351-353).
- lps1 is the default file configs/itrea2-adaptive-lps.yaml plus the job overrides repair.max_steps=20000 and repair.stall_steps=1000 (run.json config.overrides).

WHAT max_steps COUNTS: applied moves, not steps.
- GPU: the kernel parameter max_moves comes from np.int64(self.cfg.max_steps) (src/itrea/ea/ls_fj_gpu.py:838, param 345). Thread 0 checks the stop rules at the top of every step, in this order: nviol==0 → ST_FEASIBLE; else s_moves >= max_moves → ST_BUDGET; else s_since >= stall_steps → ST_STALL (377-387). The same rules are re-checked at the end of every launch (537-543).
- s_moves grows only when a move is applied (526), whether improving or a random-walk move. A local-minimum step that only bumps weights does not count.
- CPU is the same: `if moves >= max_moves` (src/itrea/ea/ls_fj_cpu.py:379); moves++ at 460; the scalars int(c.max_steps) and int(c.stall_steps) are passed at 623.

WHAT stall_steps COUNTS: consecutive steps (moves plus local minima) without a strict new best of F.
- GPU: s_since resets to 0 when s_F < s_Fbest − 1e-12·max(1,|s_Fbest|), else it grows by 1 (529-533). CPU: 462-467.
- F is the total weighted violation Σ w_i·v_i. It starts unweighted, because fj_init sets w=1 (295-312; w=1 at 296).
- At a local minimum F grows by bump·Σv over the violated rows (gpu 480-483; cpu 434). An applied move adds its score (gpu 526; cpu 461).
- Weights only grow inside a call and reset to 1 at every call, so nothing carries over between generations. A "new best" needs the weighted F below the lowest weighted F seen in the call.
- Upper bound derived from the code: steps per row ≤ about (max_steps+1)·stall_steps, because every new best needs a move. That is about 2e5 steps at the default and about 2e7 for lps1.

WHAT "FEASIBLE" MEANS: nviol==0 over the augmented rows, including the cutoff row m: cmin·x ≤ inc − δ (set_cutoff 714-742; the engine refreshes it in _refresh_cutoff, engine.py 1417-1424). Once an incumbent exists, a child finishes "feasible" only if it is feasible AND strictly better than the incumbent. Otherwise it runs until stall or budget.

GPU AND CPU DIFFER IN ONE PLACE: with p_random_walk > 0 (inert at the default 0), the GPU adds the pre-bump score s_bdv to F (484, 526), while the CPU recomputes it with the post-bump weights (436-438).

The spec vault/10-design/Feasibility jump local search.md:51-54 agrees: "after max_steps moves, or stall_steps consecutive steps without a decrease of the best total weighted violation seen in this call".

MEASURED: the stall rule ends almost every row.
- Real move-cap stops are ls_finished_budget − ls_interrupted. They are 0 in all 18 lps1 runs.
- In lps0 they are: neos-960392 3026 and 7587 rows; radiationm 203 and 201; sing44 198 and 107; fhnw 22 and 25; glass4 4 and 1; 0 everywhere else.
- Rows that finish feasible are a small share: glass4 lps0 s0 665 of about 72k rows; item_placement lps0 s0 7909 of about 593k.
- So in practice the deep setting works through stall_steps 100→1000. max_steps 2000 rarely binds.

**call_structure.** ONE GENERATION (Engine._step, src/itrea/ea/engine.py:931-994):
1. Exchange tick, only if poll_s has elapsed (934-936).
2. Select, crossover, mutate. With the strategy on: one mutate call with an lp_rows mask (956-960).
3. Children repair `_repair(C, cs)` (962-963). This is ONE LocalSearch call over all O children. O=256 and P=256 in all 36 runs. With the strategy on, each child carries its own row-scale index (1110-1114).
4. `_continue` check (964-968). If the deadline or a stop arrived during this repair, the generation is dropped and `aborted_generations` is incremented.
5. Evaluate (971-972).
6. `_with_pending` (989; 1164-1194) makes one `_repair` call PER pending batch with repair=True (1170-1176):
   - the LP-point batch: `_lp_arrived` 1740-1762, k = round(lp_rounding·immigrant_frac·P) = 32 rows, once;
   - each LP-stream point is its OWN batch of `roundings`=4 rows, at most per_tick=2 points per tick (`_lps_use` 1708-1729).
   Injected solver rows are not repaired (repair=False, 1648).
7. Survival.
8. `_after_survival` (1264-1278):
   - shortfall immigrants (P − pop.size rows): `_immigrants` + `_admit`, which repairs the rows with SEED_RANDOM|SEED_LP origin in one call (1461-1468);
   - stagnation immigrants (1427-1447): immigrant_frac of Rank 2 after 50 generations without a new best, also repaired in `_admit`.

OTHER CALLS:
- Initial population: one call over P rows in begin (902-904), round-robin strategies.
- Empty-population path: 937-940.

ONLY THE CHILDREN CALL IS ABORT-GUARDED. If a stop fires during a pending-batch or immigrant repair, the cut rows are evaluated and enter survival in that same step.

CALL CHAIN:
- `_repair` (1101-1122): singleton decoder → `_ls_run` → decoder.
- `_ls_run` (1124-1143): up to 3 OOM retries, each halving the budget (1089-1099); passes should_stop=self._should_stop and scale_idx; adds the counters to self.c; returns the moved flags from last_moves.
- `LocalSearch.run` (src/itrea/ea/repair.py:133-139) forwards to FJGpu.run or FJCpu.run.

MEASURED CALL COUNTS (ls_calls vs generations; "extra" = ls_calls − gens − 1):

| run | extra calls | generations | of which LP-stream batches |
|---|---|---|---|
| glass4 lps0 s0 | 649 | 220 | 430 |
| glass4 lps1 s0 | 81 | 27 | 56 |
| item_placement lps0 s0 | 2098 | 1770 | — |
| item_placement lps1 s0 | 569 | 388 | — |
| fhnw lps0 s0 | 370 | 183 | — |
| loue lps0 s0 | 438 | 429 | — |
| rail01 (no stream) | about 11 | — | — |

Every extra call pays its own ramp and its own slowest-row tail, one after another.

GPU HOST WRAPPER (FJGpu.run, src/itrea/ea/ls_fj_gpu.py:749-886):
- Chunk size: chunk = min(k, mem_budget // per_row) (745-747), with per_row = 24·m_aug + 80, plus 8n if X is not float64 C-contiguous.
- The local_search budget is min(1 GiB, max(32 MiB, gpu_mem_gb/8)) = 1 GiB at 12 GB (engine.py 200-204).
- MEASURED: every dev instance fits 256 rows in one chunk except ca__Scenario400_reserves_1 (float64 genes, m_aug 379,621 → 117 rows per chunk). There one call is 3 chunks in sequence, so 3 tails.
- Chunks run in sequence (806-872). Per chunk:
  - prep elementwise (816), then fj_init (825-829);
  - loop: fj_steps with `iters` steps per block (833-841);
  - count active rows with a device→host read (842); stop if 0;
  - poll should_stop (848);
  - adapt iters toward launch_ms=50 ms, growing at most 4x per launch, capped at 2^20 (851-854).
- `_iters` is reset to ≤ ITERS_START=8 at EVERY call (804).
- A finished block returns at once in later launches (350).
- should_stop is polled before every chunk (807) and after every launch (848).

THE ENGINE'S STOP CALLABLE (`_should_stop`, 1145-1162) checks:
- stop_reason;
- the deadline t0+budget;
- the mailbox stop flag, at most every STOP_POLL_S=0.25 s (172).
It SETS stop_reason, which ends the run.

PER-INDIVIDUAL ARGUMENTS TODAY:
- GPU: only the scale. scale_idx becomes d_sidx int32 (768-774), sliced per chunk (817-820); the kernels read row sidx[b] of the scale table (fj_init 288, fj_steps 351). max_moves and stall_steps are scalars shared by every block (838).
- CPU (FJCpu.run 581-658): rows run one after another in one Numba call. With should_stop they run in slices of 1 row, doubling while a slice takes less than slice_ms/2 = 25 ms (slice_ms 50, 515; loop 635-645). With several scales the call is split into runs of equal scale index (627-633).

**timing_facts.** WALL TIME OF ONE GPU CALL:
- It is the sum, over chunks run in sequence, of the launches of each chunk.
- A launch lasts about as long as its slowest still-active block running `iters` steps.
- Launches repeat until every block in the chunk has stopped (FEASIBLE, BUDGET or STALL). So the call ends when its slowest row stops.
- The code does not say whether all blocks are resident at once, and I did not measure it.
- Per-step cost depends on the chosen row: every variable of the row is scored, and each thread walks its column once over all its candidates (eval_var 200-273).
- The cutoff row spans the whole objective support, so "a full scan costs ~nnz per step" (ls_fj_gpu.py docstring 21-25). row_scan_cap 0 (the default) means a full scan.
- Overheads per call:
  - one device→host sync per launch (842);
  - the ramp, which restarts at 8 steps per launch and grows 4x per launch, every call (804, 851-854);
  - fj_init per chunk.

THE KERNEL HAS NO CLOCK: fj_init and fj_steps never read a clock (no clock64 or globaltimer in _KERNEL_SRC 92-549).
- The only time-based stop is on the host: should_stop between launches (about 50 ms apart after the ramp) and before chunks.
- The engine's callable checks only the run deadline and the stop file. There is no per-call time budget today.
- What a stop does:
  - The chunk in progress is written back with partial x; its unfinished rows keep ST_ACTIVE and count as both interrupted and finished_budget (873-878).
  - Later chunks are left untouched — not even projected, because prep runs per chunk (816).
  - CPU: a Numba call cannot be interrupted, so each row is all-or-nothing (635-645).
- launch_ms=50 is a constructor argument (605), ITERS_START=8 (86). Kernels are sliced because this GB10 reports kernelExecTimeoutEnabled=1 (docstring 39-43).

GENERATION WALL TIME is the sum of:
- exchange (at most once per generation, 934-936);
- select, crossover, mutation;
- the children repair;
- evaluate;
- the pending repairs and their evaluations;
- survival;
- after-survival work: shortfall immigrant repair and evaluation, the incumbent check, at most one post per generation (1345-1360, 1382-1386), stagnation, the cutoff refresh.

PHASE ACCOUNTING (`_Phases`, CUDA events, 396-435; written in each gen event's `seconds`):
- `_repair` opens no phase itself.
- The children call is wrapped in phase("repair") (962), and so are the pending batches (1173).
- Shortfall and stagnation immigrant repairs fall inside phase("immigrants") (1269-1271, 1443-1445).

MEASURED, ms per generation, seed 0, lps1 → lps0:

| instance | repair | immigrants | other |
|---|---|---|---|
| ca | 5404 → 773 | 591 → 96 | evaluate about 390 |
| fhnw | 5217 → 628 | — | — |
| glass4 | 3391 → 357 | 1031 → 165 | — |
| item_placement | 219 → 20.5 | 69 → 27 | all phases about 310 → 68 per generation |
| loue | 2581 → 185 | — | — |
| neos-960392 | 945 → 193 | 362 → 85 | — |
| radiationm | 156 → 52 | 169 → 110 | — |
| rail01 | 53 → 21.5 | — | evaluate 77 dominates |
| sing44 | 1553 → 160 | — | — |

MEASURED, CHANGES OVER TIME:
- sing44 lps1 s0: repair 0.2–0.3 s per generation in the feasibility phase, then 1.0–4.6 s once the first incumbent arms the cutoff row. sing44 lps0: 0.035 → 0.6–2.3 s.
- item_placement lps1: 0.28 → 0.15 s per generation over the run.
- glass4 lps0: steady at about 0.3–0.6 s.

MEASURED, EXTREMES:
- ca lps1: max_generation_s 56.8 and 58.7 s (18 generations in 120 s).
- Every ca run has aborted_generations = 1.

MEASURED, NO PER-CALL TIMING: the traces do not time individual calls or rows; only the per-generation phase sums exist. GPU `launches` and `max_launch_s` are returned by run() but the engine does not keep them.

**hook_points.** ENGINE CALL-LEVEL HOOK: every repair goes through Engine._repair(X, strat) (engine.py 1101-1122) → _ls_run(X, scale_idx) (1124-1143) → LocalSearch.run(X, should_stop, scale_idx) (repair.py 133-139).

Call sites of _repair:

| call site | lines | what it repairs |
|---|---|---|
| begin | 903-904 | initial population |
| _step | 962-963 | children |
| _with_pending | 1172-1174 | pending batches; `_Batch.src` = 'lp', 'lps' or 'solver' (455-462) |
| _admit | 1461-1468 | immigrants, reached from: shortfall in _after_survival (1267-1271), stagnation (1443-1445), the empty-population path (937-940), warm-up _warm_paths (736-739) |

The natural per-generation place is _step (931-994). gen events (1824-1863) already carry per-phase `seconds`.

DEPTH PER CALL TODAY, without rebuild or recompile:
- FJGpu reads self.cfg.max_steps and self.cfg.stall_steps at every run (ls_fj_gpu.py:838); FJCpu does the same (ls_fj_cpu.py:623).
- That cfg is the engine's own cfg.repair object, passed by reference: engine 627/631/640 → repair.py 102-116 → FJGpu.__init__ self.cfg (613) and FJCpu (516).
- Both values are scalar kernel arguments.
- The run's config dict and hash are recorded at start (src/itrea/ea/run.py:231), so mutating cfg.repair mid-run would not match the record. The explicit route is per-call run() arguments, with signatures at FJGpu.run 749, FJCpu.run 581, _Project.run 70, LocalSearch.run 133, _ls_run 1124, _repair 1101.

TIME BUDGET PER CALL: should_stop can be any callable, polled per launch and chunk (GPU) or per slice (CPU).
- A per-call budget needs its OWN callable. The engine's _should_stop sets stop_reason (1150-1161), which ends the run, and _step then drops the generation (964-968).

DEPTH PER INDIVIDUAL INSIDE ONE CALL:
- The kernel's max_moves and stall_steps are scalar parameters (345-347, used at 381-382 and 541-542).
- Per-block arrays would follow the scale-index pattern: sidx/use_sidx in fj_init (279, 288) and fj_steps (341, 351); host copy 768-774, per-chunk slice 817-820.
- CPU passes scalars into _fj_batch (325, 379-384) and already splits a call into runs of equal scale index (631-633).
- Per-row depth inside one GPU call does not shorten the call: it still waits for the slowest row (vault/30-results/Development log.md:220-221).

PER-ROW RESULTS ALREADY AVAILABLE:
- FJGpu.last_status (k,) int8 and last_moves (k,) (630-631, 883).
- Per-row S_STEPS and S_LMIN exist in STI but are only summed (860-862).
- FJCpu also has last_steps (529, 655).
- The engine uses last_moves for the ORIGIN_REPAIR bit (1140-1142) and last_status for repaired_feasible (1116-1119).
- There are no per-row or per-strategy seconds anywhere.

WHAT THE STRATEGY MACHINERY ALREADY PROVIDES (config.py StrategyCfg 189-209):
- A strategy is an index into strategies() = row_scales × lp_guidance, scale-major (207-209); defaults mutation_rate 0.1, min_share 0.05.
- Per-individual gene Population.strat, uint8 (src/itrea/ea/population.py 11-12, 77-85), carried by take (98) and concat (110).
- Initial population: round robin (engine 902).
- Children (_child_strategies 1010-1032):
  - inherit parent A's strategy, redrawn uniformly with probability mutation_rate;
  - then every strategy is topped up to min(ceil(min_share·O), O//S) children, taken at random from the strategies above that floor.
- Immigrants and injected rows draw independent uniform strategies; LP roundings go only to LP-guided strategies (1034-1070, 1723, 1755).
- Lookup tables strat_scale and strat_lp (636-646).
- One repair call with a per-row scale index (1110-1114); one mutate call with an lp_rows mask (958-960).

CREDIT COUNTERS:
- sc, cumulative: children, best_events, incumbents, best_events_solver, incumbents_solver, repaired, repaired_feasible, immigrants (877-880). Incremented at 982-983, 1069, 1115-1119, 1377-1378, 1396-1400, 1729, 1762; written to summary.strategy (1886-1890).
- sc_since: children and best_events (881), reset at every gen record (1842-1847).
- c_strat: strategy_redrawn and strategy_reassigned (882, 1020, 1031).
- Solver finds are not credited to a strategy (_is_solver_find 465-468).
- All of these are RECORDS ONLY. No code reads them to allocate children or choose a strategy: there is no bandit or credit-based allocation. Allocation is inheritance + redraw + the min_share floor.
- There is no per-strategy time or cost.

IF A DEPTH AXIS IS ADDED TO strategies(), these consumers of its (scale, lp) tuple shape must change:
- config validate n_strat (379-382);
- engine 636-646, the prepare record 608-612, finish 1886-1890;
- tests/test_configs.py:95-101 (asserts the 4-tuple list);
- tests/test_strategy.py.

WHAT THE VAULT ALREADY RECORDS (vault/30-results/Development log.md:194-224):
- candidates: a time budget per repair call; alternating deep and shallow generations; depth as a strategy gene with the generation time charged to the strategy;
- the development baseline is lps1.

**constraints.** CLOCK RULE AND COMPILATION:
- Before t0 only parse, CUDA context, device copy, building the operators, and a warm-up on a synthetic problem (engine.py docstring 10-16; prepare 515-621; _warmup 648-686).
- The warm-up sub-engines use self.cfg.with_overrides(...) (669), so the configured repair depth and any new config field are warmed. There are two passes, "many" and "one"; each runs begin, 3 steps, _warm_paths, then 1 step.
- NVRTC compiles the FJ RawModule once per block size, with -DBD and -DMAXC (ls_fj_gpu.py 555-563).
  - New runtime values passed as kernel arguments do not recompile.
  - Editing the kernel source recompiles once — during warm-up, provided the path runs there.
- New ElementwiseKernels, dtypes or array ranks must run in warm-up (comment 651-653; _warm_scans 688-707). Otherwise they compile on the clock.
- run.py records compiled_on_clock by comparing the CuPy kernel-cache file count at t0 and at the end (src/itrea/ea/run.py 271-292). It is False in all 36 lps runs.

MEMORY:
- The local-search transient budget is 1 GiB at gpu_mem_gb 12 (engine 200-204).
- FJ chunking: ls_fj_gpu 745-747. ca already needs 3 chunks.
- Per-row budget arrays (k × 4–8 bytes) are negligible.
- plan_memory's fixed terms (224-230) do not depend on max_steps or stall_steps.
- strategy_ls_bytes adds (R−1)·8·(m+1) bytes for extra row scales (272-280). Any new per-strategy device table must be counted the same way, and that can change P.
- On OOM the budget is halved (1089-1099).

DETERMINISM AND REGRESSION TESTS:
- The GPU per-block RNG is seeded from (seed, global row index idx0+b, call counter _calls) (327, 775, 828). Adding or splitting calls changes the random streams of all later calls.
- A wall-clock cut would make repair outcomes depend on timing. Today only the run deadline and the stop file do that.
- tests/_engine_reference.py with tests/test_strategy.py:83-95: with the strategy disabled, the engine must reproduce commit dd31d67's run column by column — the genealogy plus SUMMARY_KEYS, which include ls_moves and ls_calls.
- The engine docstring requires that no extra random number is drawn when the feature is off (87-88). So a new mechanism must be off by default and inert.
- tests/test_fj_gpu.py:108-150 `ref_fj` mirrors the stop rules (moves ≥ max_steps, since ≥ stall_steps). Kernel stop changes must be mirrored there and in ls_fj_cpu; the spec requires that implementations agree in semantics (vault/10-design/Feasibility jump local search.md:3-7).
- tests/test_fj_cpu.py:287-311 cover max_steps=0, the stall stop and max_steps=1.
- tests/test_review_ea.py:107-200 pin the should_stop semantics: slicing gives identical rows, the interrupted counts, the written-back partial chunk, and launches==0 on an immediate stop.
- tests/test_strategy.py:353-386 assert one local-search call per repair (a spy on run at 362).
- GPU tests (marker gpu) cannot run during the current experiments.

HASH RULE:
- EAConfig.hash (config.py 266-281) leaves out the strategy and lp_stream sections while they are disabled, and any field in _LATE_DEFAULTS while it is at its default (237-238; today only ranks.feas_alt_scale).
- A new RepairCfg field changes every recorded hash unless it is added to _LATE_DEFAULTS with an inert default: default 9ac4a93bf93b, adaptive 584ddee83747, lps0 a2ad94e9188d, lps1 4a96b872ffdc.
- A new StrategyCfg field changes the hashes of the strategy-enabled configs (584ddee83747, a2ad94e9188d, 4a96b872ffdc) unless "strategy.<field>" goes into _LATE_DEFAULTS. That works because the lookup splits at one dot (277-280).
- tests/test_configs.py:
  - 55-63: configs/itrea2-default.yaml must list every field, so regenerate it;
  - 65-70: its header records the hash;
  - 104-111: a disabled section stays outside the hash.
- The run id ends in the effective hash, which includes the job overrides (src/itrea/experiments/pair.py 306-319). lps1's 4a96b872ffdc is a2ad94e9188d plus the repair overrides.

MECHANICS COUNTED IN GENERATIONS — their wall-clock meaning changes with generation length:
- stagnation.gens 50: since_best counts generations (1341, 1361, 1427-1431);
- frontier.update_every 5 (1260);
- log_every 10 (993);
- one exchange tick per generation at most (934-936);
- at most one new incumbent posted per generation (1345-1360);
- LP-stream per_tick applies per tick, so long generations miss stream ids (Development log.md:153-160).

ABORT GUARD: only the children repair call is guarded (964-968).

PROJECT RULES (AGENTS.md): box-window lease, one GPU arm at a time, memory and thermal gates, GPU-health voiding; tuning only on the tuning set within the logged E4 budget; the test set only with the frozen-e2 hash.

STALE DOC: vault/10-design/I-TREA II algorithm.md step 5 (66-72) still describes a dense O(nnz·P) repair with ls_steps, ls_obj_weight and repair: greedy. That no longer matches RepairCfg (config.py 115-138) or ls_fj_gpu.

## Designs (verbatim)

### Phase-split, time-normalised depth bandit: one FJ depth per generation, deep arm as the prior

**mechanism.** WHAT IS CHOSEN. One repair depth per generation, from K = 2 arms given as (max_steps, stall_steps):
- arm 0 = (20000, 1000), the lps1 budget. It is the prior arm.
- arm 1 = (2000, 100), the lps0 default.

Every local-search call made inside that generation uses the generation's arm: the children call (engine.py 962), every pending batch (the LP point and each LP-stream point, 1172-1174), and the shortfall and stagnation immigrants in `_admit` (1461-1468). `begin()` repairs the initial population with the prior arm. There is no per-row depth, so a call still waits for its slowest lane, at one depth. That wait is exactly the arm's cost.

WHEN. The decision is made at the top of `Engine._step`, after the exchange tick (934-936, so the phase already reflects any `inc_external` just read) and before select. Generations are grouped into epochs. An epoch closes at the first generation boundary where either:
- (a) its wall time is at least `epoch_s` = 2.0 s. An epoch always has at least one generation, so a 4.4 s glass4 deep generation is a one-generation epoch;
- (b) the phase changed.

The arm can change only at an epoch boundary. That gives at most about 60 decisions in 120 s (item_placement); glass4 at deep gets about 25 and ca about 15.

PHASES (non-stationarity 1). F means no feasible candidate and `inc_external` = inf (the same test as ranks.py:211). I means everything else. Each phase keeps its own statistics; entering I starts I's statistics from zero. Both the reward unit and the step cost change at the switch: sing44 lps0 went from 0.13 s to as much as 2.4 s per generation, and ca lps1 from 1.3 s to one 56.8 s generation once the cutoff row was armed. Going back from I to F (only possible if every feasible member is demoted) reuses F's statistics.

CREDIT. Every reward is charged against wall time.
- Time: dt is the wall time of the epoch's generations, read with `time.monotonic` at `_step` boundaries (the host mirrors force a sync there). This covers the WHOLE generation: exchange tick, breeding, children repair, pending batches, evaluation, survival, immigrants and cutoff refresh. The immigrants phase alone is 23-45 % of wall time on glass4, item_placement and radiationm, and it is repaired at the generation's depth, so it is part of the arm's cost.
- Phase F: g = ln(V_start / V_end), where V is `Engine.best_viol`, the running minimum already kept in `_update_incumbent` (1331-1340). It starts at inf (846), so V_start is read after begin's survival. An epoch counts as a material success if g >= `materiality_v` = 0.01.
- Phase I: g is the sum, over the epoch's EA-own new incumbents, of dg = max(0, P_before - key_new) / max(|P_before|, |dual_key| if finite, 1e-9), where P_before = min(best_key before the update, inc_external). This is gain in primal-dual-gap units over the incumbent the EA knows. It is booked in `_new_incumbent` only when src == "self"; `_is_solver_find` rows are excluded, the same rule as the strategy credit. A step counts only if dg >= theta = `materiality` = 1e-3. LP-immigrant and LP-stream roundings ARE credited: they were repaired at this arm's depth, and that is exactly where ca's and sing44's deep advantage came from.
- Statistics per (phase, arm): discounted gain G, discounted time T and discounted material count N. At each epoch close, every arm's G, T and N are multiplied by 2^(-dt/H), with H = `half_life_s` = 30 s; then the played arm gets += (g, dt, n_material). The rate is mu = G/T, a ratio of sums (the renewal-reward rate), so a long epoch weighs more than a short one.

SELECTION. Deterministic, with no random draw, evaluated in this order at each epoch boundary:
1. Phase I only: the first `explore_epochs` = 4 epochs alternate the arms, starting with the arm in use at the switch, so each arm gets 2 epochs. That is enough for item_placement's shallow evidence: 67-95 material steps per 120 s, so about 2.5 in 4 s of shallow time.
2. Floor: an arm whose discounted time share T_a/sum(T) is below `floor` = 0.10 gets the next epoch as a probe. The exception: it is not probed if its last generation time in this phase exceeds `probe_max_frac` = 0.25 x the remaining budget. In phase F this gives the shallow arm one probe right after the first deep epoch, then about one every 20 s.
3. Greedy with a prior: use the prior arm 0 unless a challenger c has N_c >= `min_evidence` = 2 and mu_c > (1 + `margin`) x mu_prior, with margin = 0.25; then use the best such challenger. Ties and "no evidence yet" go to the prior.

This is exploitation-leaning, which Chmiela et al. 2023 found better for primal heuristics. The floor is Fialho et al.'s p_min expressed in time units, kept small because they found that p_min slows adaptation.

CREDIT-ASSIGNMENT DELAY.
(i) Direct effect: none. A child's repair, evaluation, survival and the incumbent update all happen in the same `_step` (962-994, then `_after_survival` and `_update_incumbent`), so an improvement lands in the epoch of the arm that repaired it.
(ii) First-feasible events come late: sing44 at 12.7-16.6 s deep, ca at 35 s, fhnw at 36 s. Phase F's log-V reward gives graded credit long before them. The early phase-F windows are level-biased: sing44 shallow shows 0.37-0.41 /s over 0-2 s only because its gen-0 V is 41.5 vs deep's 16.6 (the initial population had been repaired deep). That bias is why the prior arm repairs the initial population and runs first.
(iii) Carry-over of population state is not modelled: a deep epoch builds low-V and Rank 0 members that a later shallow epoch exploits. Short epochs and alternation share this between the arms.
(iv) SCIP steps seeded by EA posts are invisible to the reward.

NON-STATIONARITY. Four parts: the F/I split; wall-time discounting with H = 30 s; the time-share floor, which keeps the losing arm's estimate fresh; and alternation on entering phase I, so both arms see the early, easy part of that phase.

KEEPING THE COUPLED EXCHANGE FAIR.
- Only EA-own incumbents earn credit, and only by how much they beat min(EA best, inc_external). A gain SCIP already had counts 0. SCIP's finds are never credited; they only raise the bar, and they raise it for whichever arm is running. The reward is therefore the EA's marginal contribution to the portfolio integral.
- The EA's view of SCIP's incumbent lags. SCIP posts its incumbent only from its heuristic callback (scip_run.py `_post_info`, called from `on_call`/heurexec). On sing44 lps1, `inc_external` stayed inf until about 46 s, although SCIP had a gap-0.985 solution at 4.8 s. The phase switch and the phase-I reward both use what the EA knows. That lag is the same for both arms, but it is not the true portfolio.
- The reward does not charge the generation-length side effects: one exchange tick and at most one post per generation, LP-stream `per_tick`, and the post load on SCIP. Only the P measurement sees those.
- The run footprint is unchanged: O(K) host arithmetic on the EA's own 3 cores, no extra local-search call, no GPU memory, no random draw.

**why_it_fits.** THE LEVER AND THE ARMS (code map, run data):
- The stall rule ends 90-100 % of repairs at both depths, and real max_steps stops are about zero except neos-960392 at the default depth. So the lever is stall_steps 100 -> 1000.
- Both measured configurations keep max_steps/stall = 20. The two arms are exactly those two measured endpoints.
- A middle arm has no evidence and would add exploration cost, so it is a follow-up, not part of the first measurement.

ONE DEPTH PER CALL respects how the GPU repair ends:
- `FJGpu.run` loops launches until `count_nonzero(STI==ST_ACTIVE)==0` (ls_fj_gpu.py 842-843).
- Per-row depth inside a call would not shorten a generation (Development log 220-221).
- The arm's measured cost (wall time per generation, 3-9x apart) is the quantity the bandit trades against gain.

THE REWARD SIGNALS SEPARATE THE ARMS IN THE RIGHT DIRECTION. I recomputed these from today's traces (`/tmp/itrea-design/depth_analysis/runs.pkl`, `steps.pkl`).

Phase F, ln-V decrease per s (lps1 vs lps0, seeds 0/1):

| instance | window | deep (lps1) | default (lps0) | reading |
|---|---|---|---|---|
| ca | 2-20 s | 0.04-0.25 | 0.004-0.007 | 10-50x |
| sing44 | 5-10 s | 0.021 / 0.007 | 0.001 / 0.000 | — |
| sing44 | 2-5 s | 0.031 / 0.003 | 0.001 / 0.003 | ties on seed 1 |
| fhnw | 10-40 s | 0.055-0.106 | 0.016-0.093 | weak |
| rail01 | — | — | — | no separation, harmless |

Phase I, EA-own material gain (dg >= 1e-3) after the EA's first incumbent; the first-feasible step is excluded:

| instance | deep (lps1) | default (lps0) | reading |
|---|---|---|---|
| item_placement, total over 120 s | 0.432 / 0.252 (127-139 steps) | 0.616 / 0.492 (67-95 steps) | shallow 1.4-2x; per-5 s windows are noisy, e.g. 0-5 s seed 1: 2.0 shallow vs 3.7 deep |
| loue, total | 0.179 / 0.230 | 0.036 / 0.048 | deep about 10x in rate over 20-40 s |
| glass4 | 0.076 / 0.054, from 2 steps per seed | 0.000 / 0.003, from 0-1 step | the lumpy case |
| sing44 | 0.21-0.24 | 0.29 from 50 s on | shallow faster per second, but from a higher gap; unclear |
| ca | — | — | negligible (at most 0.011 total) |

Glass4 is why `min_evidence` = 2 and theta = 1e-3 exist: a single 0.003 shallow step must not displace deep, while item_placement's shallow evidence arrives within seconds. At theta = 1e-3, glass4's shallow polish (198-200 steps worth 0.001 in total) drops out, and item_placement's rates barely change (for example 18.2 -> 17.5).

A PHASE SPLIT, BECAUSE THE DATA SHOW A REGIME CHANGE:
- Phase F: deep wins or is harmless on every informative instance (sing44, ca, fhnw, loue).
- Phase I: the winner is instance-dependent — deep on glass4 and loue, shallow on item_placement, neutral or unclear on sing44, ca and neos-960392.
- Step cost jumps when the cutoff row is armed (sing44, ca).
- The analysis's cross-instance finding 5 says "deep until the EA's own first feasible, then re-decide". This design is that rule, learned instead of fixed.

A DEEP PRIOR WITH A FLOOR:
- Deep wins on 6 of 9 instances.
- Choosing wrong costs -0.16 to -0.35 bounded d on the deep-favoured instances, and -0.37 on item_placement.
- A deep prior with a 10 % floor and an alternation start caps the cost on item_placement-type instances at roughly the first 10 s plus 10 % of the rest.
- Deep-first in phase F also keeps lps1's HiGHS LP arrival: the busier shallow loop delayed it from 10.3 to 12.9 s on sing44 and from 29.3 to 32.4-33.1 s on ca.

LITERATURE:
- Hendel 2022 and Chmiela et al. 2023: bandit or epsilon-greedy control of heuristic choice and working limits inside SCIP, judged on the primal integral; exploitation-leaning works best for primal heuristics.
- Ong & Keane 2004: reward = gain per unit of local-search effort.
- Fialho et al. 2010: windowed or discounted credit for drift, and a p_min that must stay small.
- Sudholt 2012 and Cotta 2025: the right depth is instance-dependent and no fixed setting is safe.
- Krasnogor & Smith 2005: this is a coarse-grain scheduler deciding LS intensity from population statistics.
- Open gap: no source measures bandit control of local-search intensity inside a GPU-batched EA for MIP.

FITS THE CODE:
- FJGpu and FJCpu read `cfg.max_steps` and `cfg.stall_steps` at every run (ls_fj_gpu.py:838, ls_fj_cpu.py:623). A per-call override is a local substitution.
- The GPU kernel takes these as np.int64 kernel arguments and the CPU path passes int(...) into the same Numba signature, so nothing recompiles.
- The per-individual strategy gene (row scale x LP guidance) is orthogonal and untouched.

**implementation.** Paths are relative to the repository root. Nothing here has been written; the repository is untouched.

1. src/itrea/config.py
- New dataclass `RepairDepthCfg` with these fields and defaults:

| field | default |
|---|---|
| enabled | False |
| arms | [[20000, 1000], [2000, 100]] (index 0 is the prior) |
| prior | 0 |
| epoch_s | 2.0 |
| explore_epochs | 4 |
| half_life_s | 30.0 |
| floor | 0.10 |
| materiality | 1e-3 |
| materiality_v | 0.01 |
| min_evidence | 2 |
| margin | 0.25 |
| probe_max_frac | 0.25 |

- New field `EAConfig.repair_depth`. It is not called `depth`: `Engine.depth` already means lineage_depth (engine.py:832, 975).
- `hash()`: `if not self.repair_depth.enabled: d.pop("repair_depth")`. This is the same section rule as strategy and lp_stream. `_LATE_DEFAULTS` is not used; it is for single fields.
- `validate()`, when enabled:
  - require `repair.method == "fj"`. Method "none" calls `_Project.run(X, should_stop)` with no budget, and dense ignores max/stall;
  - 1 <= K; each arm has max_steps >= 0 and stall >= 1; 0 <= prior < K;
  - epoch_s > 0, half_life_s > 0, 0 <= floor < 1/K, explore_epochs >= 0, materiality >= 0, min_evidence >= 0, margin >= 0, 0 < probe_max_frac <= 1.
  - K = 1 is allowed; it is used as a test.
- `_build` fills a missing section from its defaults, so old run.json configs without the key still load.

2. src/itrea/ea/repair_depth.py (new)
- `DepthBandit(cfg, clock=time.monotonic)`: pure Python, no RNG.
- Methods:
  - `begin_arm()` returns (max_steps, stall_steps) of the prior;
  - `decide(phase, best_viol, remaining_s)` returns a kwargs dict and closes or opens epochs;
  - `book_incumbent(p_before, key_new, dual_key)`;
  - `events()` returns the epoch records;
  - `summary()`.
- Statistics are kept per phase as arrays G, T, N and last_gen_s, plus the epoch accumulators.

3. src/itrea/ea/ls_fj_gpu.py
- `FJGpu.run(..., max_steps=None, stall_steps=None)`: ms and ss default to self.cfg. Pass `np.int64(ms)` and `np.int64(ss)` at line 838.
- Same kernel and argument types, so no recompile, and `compiled_on_clock` stays False.
- Update the docstring.

4. src/itrea/ea/ls_fj_cpu.py
- Same change in `FJCpu.run`, at 623 (`int(...)`, so the Numba signature is unchanged).

5. src/itrea/ea/repair.py
- `LocalSearch.run(X, should_stop=None, scale_idx=None, max_steps=None, stall_steps=None)` forwards the two only when not None, and raises for method "none" or "dense".

6. src/itrea/ea/engine.py
- (a) In `__init__`/`begin`: `self.dctl = DepthBandit(cfg.repair_depth) if enabled else None` and `self._ls_kw = {}`. `begin()` sets `self._ls_kw = dctl.begin_arm()` before the initial repair (902-904).
- (b) In `_step`, right after the exchange tick (936):
  - phase = "I" if `self.h_feas.any()` or `math.isfinite(self.inc_external)`, else "F";
  - `self._ls_kw = self.dctl.decide(phase, self.best_viol, self.deadline - now)`.
- (c) In `_ls_run` (1124-1143): call `ls.run(X, should_stop=..., [scale_idx=...], **self._ls_kw)` only if `_ls_kw` is non-empty. With the feature off the call is identical to today's. This matters because the spy in tests/test_strategy.py:362 has the signature (X, should_stop=None, scale_idx=None) and would raise a TypeError on any extra kwarg.
- (d) In `_new_incumbent` (1365): before `best_key` is overwritten, compute p_before = min(self.best_key, self.inc_external). If dctl is on and src == "self", call `dctl.book_incumbent(p_before, key, self.dual_key)`.
- (e) In `_gen_record`: add `depth_arm` when on.
- (f) New trace event `depth` at each epoch close, with fields:
  - t, gen, phase, arm, dt, gens, g, n;
  - the vectors mu, T and N;
  - next, and why (explore | floor | greedy | prior | phase | guard).
- (g) In `finish`: `summary["repair_depth"]` with arms, and per phase: time_s[], gens[], epochs[], gain[], material[], plus switches, blocked_probes and t_phase_I.
- (h) Warm-up: the sub-engines get the section through `with_overrides` (669), so the path runs before t0.

7. Configs and jobs
- Regenerate configs/itrea2-default.yaml: test 55-63 requires every field, and the header hash stays 9ac4a93bf93b because the disabled section is popped.
- New configs/itrea2-adaptive-lps-depth.yaml = itrea2-adaptive-lps + `repair_depth.enabled: true` (new effective hash), and a job for arm A3@db1.

8. Documentation
- vault/10-design/Trace format and metrics.md: event kind `depth`, `gen.depth_arm`, `summary.repair_depth`.
- vault/10-design/Feasibility jump local search.md: per-call budget arguments; the stop semantics are unchanged.
- `itrea.analysis.build_runs`: new columns deep_share_F, deep_share_I and depth_switches.

TESTS (CPU and NumPy only, run as `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 10-14 .venv/bin/python -m pytest -m "not gpu"`):
- tests/test_repair_depth.py, the controller with a scripted clock:
  - the prior goes first;
  - the phase-F floor probe comes after the first epoch;
  - phase I alternates for 4 epochs;
  - the discount equals 2^(-dt/H);
  - an epoch closes only at >= epoch_s or on a phase change;
  - one material challenger step does NOT displace the prior (the glass4 case);
  - a challenger that is 2x better with >= 2 steps takes over right after the alternation (the item_placement case);
  - the phase switch resets phase I;
  - the probe guard blocks a slow arm near the deadline;
  - gains below theta and solver-find incumbents are not booked;
  - identical inputs give identical arms.
- Engine on toy models (`itrea.benchmarks.toy`, backend numpy, FJCpu, scripted clock):
  - every `ls.run` call within one generation gets the same (max_steps, stall_steps), equal to the logged arm;
  - the initial population gets the prior;
  - still one local-search call per repair (extend test_strategy 353-386 with the feature on);
  - `rng_host` is consumed exactly as without the feature;
  - `depth` events and the summary are present.
- K = 1 equivalence: arms [[m, s]] reproduces, genealogy column by column, the run with `repair.max_steps=m, repair.stall_steps=s`.
- tests/test_fj_cpu.py: `run(X, max_steps=m, stall_steps=s)` equals a fresh FJCpu whose cfg has (m, s), row by row (X, last_status, last_moves, last_steps), including m = 0 and m = 1; None equals cfg.
- tests/test_fj_gpu.py: the same check on FJGpu, marked gpu (run later, outside the experiments).
- tests/test_configs.py:
  - extend 104-111 to `repair_depth`;
  - assert that 9ac4a93bf93b, 584ddee83747, a2ad94e9188d and 4a96b872ffdc are unchanged;
  - `validate` refuses method != fj.
- tests/test_strategy.py:83-95: the dd31d67 reference reproduction must still pass unchanged.

**cost_and_risk.** OVERHEAD
- Per generation: one `time.monotonic` read and about 20 float operations.
- At most about 60 `depth` events per run.
- No added local-search call, so the GPU RNG call counter `_calls` advances exactly as before.
- No device table, so `plan_memory` and P are unchanged. No random draw.
- With K = 1 the run is the fixed-depth run.

SLOWEST INDIVIDUAL
- One depth per call means the arm's cost includes its tail, and for ca also the 3-chunk sequence.
- A controller that decides per generation cannot shorten a generation it has already started. In ca lps1, gens 10 -> 18 took 14.6 -> 120.8 s, one of them 56.8 s, right after feasibility.
- The probe guard only stops that from being repeated near the deadline.
- On ca this is P-irrelevant: seed 0's first feasible was already at gap 0.002, and phase-I EA gains are at most 0.011.
- On an unseen instance, a deep generation in phase I could cost tens of seconds. The complementary guard is a per-call time cap, which is a separate mechanism and not part of this design.

ADAPTIVE STRATEGY
- The two axes are orthogonal.
- At the default depth the rhs strategies die out (sing44 final population [121, 135, 0, 0]). After a long shallow stretch, a return to deep finds no rhs members; only the 5 % children floor brings them back.

LP STREAM
- Deep generations poll less often (lps1 missed 95-358 ids).
- Each stream point's roundings are repaired at the generation's depth.

HiGHS LP
- It shares the 3 cores. The phase-F shallow probe slows it slightly; deep-first keeps lps1's arrival times.

EXCHANGE
- `inc_external` lags SCIP (about 40 s on sing44), and the phase switch and the I-reward follow what the EA knows.
- Shallow epochs post more: item_placement lps0 posted 1,757 times, 26 % accepted.
- About half of glass4's deep advantage is SCIP-own credit (0.72 vs 0.68), probably seeded by EA posts, and the reward cannot see it.

MECHANISMS COUNTED IN GENERATIONS
- `stagnation.gens` = 50, `frontier.update_every`, `log_every`, and one post per generation now mean different wall times in different epochs.

RISKS OF FOOLING OURSELVES
1. lps0 and lps1 ran in different windows (16:03-16:35 vs 16:55-17:30). A same-window control is needed.
2. Only 2 seeds. item_placement's default-depth runs spread 0.28-0.52, so a pass on P alone could be luck; the time-share diagnostics must agree before the mechanism gets credit.
3. glass4's EA-own signal is 0-2 jumps per run, so phase-I selection there leans on the prior. If glass4 fails while item_placement passes, the cause is that SCIP-seeded credit gap, not tuning.
4. fhnw is a one-event lottery (one find on one seed).
5. neos-960392's P is tiny, so 0.003 absolute becomes up to -0.18 bounded d. The bandit may pick shallow there, since early rates are equal.
6. Confound: lps0 is the worst of three configurations on glass4 and loue. The bandit compensates for a weakness of the adaptive configuration at the default depth; it does not diagnose it. The one-variable run with the strategy off at the default depth is still owed.
7. Arm sequences depend on wall time, so a seed does not reproduce a run. Every decision is logged for that reason.
8. The default values (epoch_s, H, floor, theta, margin, min_evidence, explore_epochs) were set from today's 36 dev traces. That is dev-set tuning: freeze them before any tuning-set use and count them in the E4 budget (AGENTS.md).
9. Carry-over bias: a shallow epoch that follows a deep one inherits a better population and gets credit for it.

**measurement.** BEFORE ANY GPU TIME
- Run the CPU suite pinned: `OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 taskset -c 10-14 .venv/bin/python -m pytest -q -m "not gpu"`. It covers the new tests, the dd31d67 reference reproduction and the hash invariance.
- Then follow AGENTS.md: box-window lease, memory and thermal gates, one GPU arm at a time. Run the gpu-marked FJGpu equivalence test only inside that window.

RUNS
- Arm A3@db1 = itrea2-adaptive-lps + `repair_depth.enabled: true`. Everything else is identical, so it is one variable against both lps0 and lps1.
- 9 dev instances x seeds 0-1 x 120 s = 18 runs, about 35 min (lps0 took 16:55-17:30).
- Same-window control: A3@lps1 on item_placement_9903 and glass4, x 2 seeds = 4 runs, about 8 min. The two endpoints are the instances most sensitive to drift.
- Total about 45 min.

VERDICT RULE, fixed before the run:
1. item_placement: seed mean <= 0.45, and both seeds below 0.558 (lps1's better seed).
2. sing44, glass4, loue and ca: each seed mean at most max(lps1 seeds) + 0.05, i.e. at most 0.454, 0.279, 0.457 and 0.348. Each must also beat lps0's seed mean (0.551, 0.323, 0.440, 0.395) on at least 3 of the 4.
3. radiationm and rail01 must stay at 0.234 / 1.000 +/- 0.01.
4. fhnw and neos-960392 are reported but excluded from the verdict.
5. Mechanism agreement, from the `depth` events:
   - phase-F deep time share >= 0.8 on sing44 and ca;
   - phase-I shallow share >= 0.6 on item_placement;
   - phase-I deep share >= 0.6 on glass4 and loue.
   If P passes but these shares disagree, the result is not credited to the mechanism.
6. Hygiene: 22/22 completed, every incumbent checked, GPU health ok, compiled_on_clock False.

REPORT
- Mean bounded d vs lps1 and vs lps0.
- The control's drift: if the same-window lps1 differs from the recorded lps1 by more than 0.05 on either instance, compare against the control.
- Per instance: the time of the phase switch, the arm sequence, and the number of epochs.

**expected_effect.** This is an estimate; nothing has been run.

PER INSTANCE

| instance | expected P(120) | lps1 | lps0 | why |
|---|---|---|---|---|
| item_placement_9903 | 0.38-0.45 | 0.569 | 0.358 | 50/50 for the first ~10 s (alternation), then about 90 % shallow |
| sing44 | 0.36-0.40 | 0.362 | 0.551 | — |
| ca | about 0.30 | 0.297 | 0.395 | — |
| glass4 | 0.23-0.26 | 0.228 | 0.323 | — |
| loue | 0.37-0.40 | 0.369 | 0.440 | — |
| neos-960392 | 0.032-0.039 | 0.032 | 0.039 | may settle on shallow |
| fhnw | 0.30 or 1.00 | 0.651 | 1.000 | one-event lottery |
| radiationm18-12-05 | unchanged | 0.234 | 0.235 | depth irrelevant |
| rail01 | unchanged | 1.000 | 1.000 | depth irrelevant |

On sing44, ca, glass4 and loue: deep runs about 90 % of phase F everywhere, and of phase I on glass4 and loue. The small costs come from the 10 % floor and the phase-I alternation.

MEAN EFFECT
- Mean bounded d vs lps1: about -0.02 to +0.03. One large gain on item_placement (about +0.25) is offset by small exploration costs on five instances. The neos-960392 term depends on whether the bandit settles there on shallow or deep.
- Mean bounded d vs lps0: about +0.10.

The claim is robustness, not a mean gain over lps1: no -0.37 on an item_placement-type instance, and no -0.16 to -0.35 on the deep-favoured ones. That is the goal of getting the deep results on six instances and the fast generations on item_placement at once.

EVIDENCE IS THIN
- 2 seeds; one fast instance; glass4's lumpy signal; fhnw's single event.
- The adaptive-configuration confound.
- If glass4 misses, the next variant is to credit SCIP incumbents that follow an accepted EA post within a short window — not to retune the bandit.

### Repair clock: a per-call lane-step horizon enforced between launches, a wall-clock cap, and a controller that reads each horizon's marginal value from the counterfactual profile of the call it just ran

**mechanism.** WHAT IS ENFORCED, AND WHERE
1. Lane-step horizon H (primary, deterministic).
   - Fact from the kernel (ls_fj_gpu.py fj_steps, the loop at about 383-536): an active block runs exactly `iters` steps per launch, or stops early on FEASIBLE / BUDGET / STALL. S_STEPS grows by 1 per step and starts at 0 in fj_init.
   - So after every launch, every still-active lane has S_STEPS = Σ iters, the same for all lanes. A cut between launches is therefore already an equal lane-step horizon for every lane.
   - Enforcement needs no kernel change. In FJGpu.run the host truncates the next launch to iters = min(self._iters, H − cum). When cum reaches H it stops launching, and lanes still ACTIVE get the new status ST_HORIZON = 4. The status is set on the host, in status_all; it is not written to STI.
   - The CPU reference adds one scalar to _fj_batch: `if steps >= max_lane_steps: stt = 4; break`. It is checked after the feasible / budget / stall rules, the same order the GPU end-of-slice re-check gives.
2. Wall-clock cap (backstop, host side, between launches).
   - Per call: a cap in seconds, split over chunks. Chunk c of C gets the deadline t0_call + cap·(c+1)/C. ca has 3 chunks of 117 rows.
   - When the deadline passes, the host stops launching. Active lanes are cut at H_eff = cum, which is again equal for every lane.
   - So even a clock cut is a horizon cut, and every call can be replayed from one logged integer (H_eff per chunk) on either backend.
   - Unlike today's run-level stop (ls_fj_gpu 806-872), a chunk whose deadline has already passed still runs prep + fj_init with horizon 0: it is projected, never left raw.
   - Launch granularity: launch_ms_eff = clamp(cap/8, 5 ms, 50 ms), so the overshoot is at most about 1/8 of the cap.
   - The cap check is the call's own test. It never touches stop_reason, so the abort guard of _step (964-968) still reacts only to the run deadline and the stop file.
3. Rejected for v1: an in-kernel %globaltimer check.
   - Each block would read the clock at a different step. That breaks the equal-steps invariant.
   - A call would then no longer be replayable from a logged value, and the sequential CPU reference (rows one after another, ls_fj_cpu 635-645) could not mirror it.
   - Its possible benefit is unmeasured: cheap-step lanes would get more steps while expensive-step lanes, which pick the cutoff row, catch up. It is kept as a variant to measure if the cap fires often.
4. Repair settings stay those of lps1 (max_steps 20000, stall_steps 1000). The horizon level "none" is therefore lps1, row for row: results do not depend on slicing (tests/test_review_ea.py 107-200).

THE MEASUREMENT THAT DRIVES THE CHOICE (the marginal value of repair time, from the call that was actually run)
- Lanes are independent blocks. The cutoff is constant across every repair call of one generation: _refresh_cutoff runs only in _exchange_in (1607, 1623) at the start of the step and as the last act of _after_survival (1275), after the immigrant repairs.
- So for any h ≤ the played horizon, a lane that finished FEASIBLE at S_STEPS = s ≤ h would have finished identically under horizon h: same x, same status.
  - Checked now on the CPU, with the existing move budget standing in for the horizon (/tmp/itrea-design/rclock/cf_identity.py).
  - On 3 toys, 64 rows, M ∈ {1, 3, 10, 30, 100}, every row that was feasible by M in the long call was identical (x and status) in the short call. The short call's feasible count equalled that number every time.
- The cost side is equally exact and comes from host data only. The (cum_steps, elapsed) pair after every launch gives t_call(h) by linear interpolation inside a launch, with t_call(h) = t_end once every lane has stopped.
  - No device read is added: the per-lane S_STEPS and status already come to the host in `st = cp.asnumpy(STI[:kc])`. FJGpu only needs to keep `last_steps` as FJCpu does.
  - For the sequential CPU, t(h) ≈ call_seconds · Σ min(steps_row, h) / Σ steps_row.
- Per generation g and ladder level h_j:
  - Δ_g(h_j) = max, over every lane repaired in the generation (children, pending batches, immigrants) with status FEASIBLE, evaluate-feasible and S_STEPS ≤ h_j, of (cutoff − key) / D_g.
  - D_g = max(cutoff − dual_key, 1e-9·max(1, |cutoff|)) when a finite dual_key exists, else max(|cutoff|, 1). It is shared by every level within g.
  - T_g(h_j) = o_g + Σ_calls t_call(h_j), where o_g = measured generation time − measured repair-call time.
  - Δ is the MAX per generation, not a count: one incumbent per generation, the redundant improving children are worth about nothing (item_placement deep: about 30 per generation), and size matters (glass4 default: 200 polish steps worth 0.001 in total).
- The immigrant calls' inputs depend on survival, so their counterfactual is approximate; children and pending batches are exact.

CONTROLLER (host-side, no RNG at all)
- Ladder of horizons:
  - h_j = 2^j lane steps, j = 7..16, plus "none". A level is in force when the generation starts and applies to every repair call of that generation: children, pending batches, shortfall and stagnation immigrants.
  - A level h_j is "observed" in generation g if the played horizon was ≥ h_j, or if every lane stopped before h_j.
- Accumulators per level, decayed by 2^(−age/half_life), half_life 20 s of wall time: G_j = Σ w_g Δ_g(h_j), T_j = Σ w_g T_g(h_j), and n_j = observed generations.
- Phase rule.
  - Before the EA has any incumbent (`self.cutoff is None`): play "none" with NO cap. This is lps1 exactly, including the initial-population call in begin.
  - After it: the controller decides, with a per-call cap = min(cap_s = 10 s, 3 × the level's recent median call time). The "none" level gets cap_s only.
- Choice, after each generation:
  - j* = argmax G_j/T_j over the levels with n_j ≥ 5.
  - Move from the current level only if rate(j*) > 1.25 × rate(current) (hysteresis).
  - If every G_j = 0 (no evidence), go to or stay at "none", the lps1 baseline. The design moves away from the baseline only on evidence.
- Probes. Deeper levels cannot be observed from shallower play. When the wall time spent in probe generations is below 10 % of the post-incumbent elapsed time, and the level two steps deeper has not been observed within one half-life, the next generation plays min(current + 2 levels, "none"). The probe's call also re-measures every shallower level for free.
- Start after the first incumbent: "none" until 5 generations are observed. One deep call observes every level at once.
- Shadow mode: always plays "none"; logs what it would choose.

INDIVIDUALS CUT SHORT
- A horizon-cut or cap-cut lane returns its current, projected, partially repaired x, with ST_HORIZON. This is exactly how a STALL lane returns today: FJ keeps the current x, not the best-F point.
- It is evaluated and enters survival like any child: Rank 0 if infeasible with a key below the incumbent (≤ 10 % total cap), else Rank 2. A lane that is feasible on the original rows but not below the cutoff counts as feasible-not-improving.
- ORIGIN_REPAIR is set from last_moves as now. `repaired_feasible` (status == 1) is unchanged.
- There is no resumption across generations: weights reset at every call, and keeping W per individual would cost 8·m_aug·P bytes (ca: about 0.78 GB).

DETERMINISM
- Per call: result = f(rows, seed, global row index, call counter, H_eff). No call is added or split, so the RNG streams (ls_fj_gpu 327, 775, 828) are unchanged.
- The controller's choices depend on measured times, as the generation count already does today. Each played H and H_eff is logged, so any call is reproducible.
- With the section disabled: the old code path, no new kwargs, no new counters and no RNG draw. The engine docstring's "no extra random number when off" (87-88) holds, and tests/_engine_reference.py is untouched.

**why_it_fits.** 1. The lever is effort per lane, and the horizon caps the tail the call waits for.
- In 18/18 lps1 runs real move-cap stops are 0, and the stall rule ends 90–100 % of lanes at both depths.
- Deep repair works by letting lanes run longer (moves per lane 85–475 at the default depth, 318–1,756 deep). The call waits for its slowest lane (ls_fj_gpu 833-872).
- A lane-step horizon caps exactly that tail, uniformly for all lanes, in the effort unit FJ itself uses (Luteberget & Sartor: effort counts, not time). The clock only bounds what the horizon cannot foresee: the step cost changes when the cutoff row arms.
  - sing44 lps1: about 0.2–0.24 s of repair per generation before the first incumbent, then 1–4 s.
  - ca lps1: 56.8 / 58.7 s generations after its 35.5 s incumbent.

2. Pre-incumbent identical to lps1 keeps every deep-repair win that came through first feasibility:
- sing44: EA feasible at 12.65 / 16.6 s deep vs 50.5 s default;
- ca: 35.5 s;
- loue: 2.1 vs 7.3 s;
- fhnw seed 0: the one find of the optimum at 36.2 s. It came after a gen-0 record of 11.9 s of repair and generations of about 3.5 s, so any cap there could have removed it.
- The phase test is the EA's own cutoff state, not SCIP's incumbent. sing44 lps0: SCIP's 7.7 s solution reached the EA only at 50.2 s, and the EA's feasibility_phase stayed true until then.
- By construction this also keeps the HiGHS LP arriving at the lps1 time (2.5–4 s earlier than at the default depth on sing44 and ca), because the host loop is not busier in that phase.

3. Post-incumbent, the run data say the right depth is instance-dependent and is decided by the size of the improvements per second. The counterfactual measures exactly that:
- item_placement: the incumbent improves in about every generation at both depths (0.975 vs 0.999), so evidence is plentiful at once. Feasible at 0.05 s, so the run is post-incumbent almost from the start. Shallow should win if the best-of-a-few-lanes improvement per generation is above about 1/4.7 of the deep one. Recorded actual runs: EA gain per generation 3.4e-4 at the default depth vs 1.1e-3 deep, i.e. 3.2× per generation for 4.7× the time.
- glass4: the value is a few large jumps (EA portfolio steps ≥ 0.01: 0.074 per deep generation vs 0 at the default depth). Max-improvement credit, not counts, keeps it deep, provided those jumps come from late-finishing lanes. The traces cannot tell; shadow mode answers it.
- neos-960392: large per-generation cuts deep (−0.1 to −0.43); evidence is plentiful.
- ca: no evidence after its incumbent, so it stays at "none". The 10 s cap turns a 57 s generation into several, and the controller can then find a shallower level if gains appear. The evidence that deep is worse there post-incumbent is one seed: seed 1 lps0 0.304 vs lps1 0.296.
- radiationm, rail01: depth is irrelevant (P equal). rail01 stays pre-incumbent, i.e. lps1.

4. The whole generation is bounded, not only the children call. Extra calls are large: glass4 lps0 870 calls in 220 generations, item_placement lps0 3,869 in 1,770. The immigrant phase is 23–40 % of wall time, and glass4 lps1 gen 0 had pop 28 and shortfall 230. So the level applies to every repair call of the generation, each with its own cap.

5. Literature.
- Spoorendonk's productive-vs-stale effort is what the per-call profile measures: the steps at the last feasible finish vs the horizon.
- PaNGEA caps each member of a GPU batch (5 s, 2M iterations, 2 s without improvement) because the batch waits for its slowest member.
- Chmiela et al. found that exploitation-leaning schedules suit primal heuristics, hence a small probe share and stay-on-baseline when there is no evidence.
- Fialho et al.: extreme or size credit when gains are rare and large.
- Sudholt: no fixed depth is safe a priori.
- Structural difference from those bandits: playing a level reveals every shallower level exactly, so exploration is needed only upward.

**implementation.** Nothing has been edited. All paths are relative to the repository root.

CONFIG — src/itrea/config.py
- New dataclass RepairClockCfg, top-level field `repair_clock` on EAConfig beside strategy and lp_stream. Fields:
  - enabled: false
  - mode: "active" | "shadow"
  - level_min_log2: 7, level_max_log2: 16 (the ladder 2^7..2^16 plus "none")
  - min_gens: 5
  - half_life_s: 20.0
  - hysteresis: 0.25
  - probe_share: 0.10
  - probe_up: 2
  - cap_s: 10.0
  - cap_factor: 3.0
  - launch_ms_min: 5.0
- Hash: in EAConfig.hash (266-281) add `if not self.repair_clock.enabled: d.pop("repair_clock", None)`, the same pattern as strategy and lp_stream. Recorded hashes 9ac4a93bf93b, 584ddee83747, a2ad94e9188d and 4a96b872ffdc stay unchanged.
- validate():
  - enabled needs repair.method == "fj" and objective_mode == "cutoff_row" (the gain signal relies on FEASIBLE ⇒ strictly below the cutoff);
  - level_min_log2 ≤ level_max_log2 ≤ 20 (the iters cap);
  - 0 < probe_share < 1; cap_s > 0.
- Regenerate configs/itrea2-default.yaml: the section is listed, the header hash stays 9ac4a93bf93b.
- New dev config configs/itrea2-adaptive-lps-rc.yaml = itrea2-adaptive-lps + repair.max_steps 20000 + stall_steps 1000 + repair_clock.enabled. Equivalently, the lps1 job plus the override; its effective hash differs from 4a96b872ffdc.

GPU — src/itrea/ea/ls_fj_gpu.py, FJGpu.run (749-886). Host only, no kernel or source change, so no NVRTC recompile and no new ElementwiseKernel.
- New kwargs horizon=None, cap_s=None. With both None, the existing loop runs unchanged.
- Otherwise:
  - iters = min(self._iters, horizon − cum);
  - record (cum, perf_counter() − t0_chunk) after every launch;
  - stop when cum ≥ horizon, or the chunk deadline has passed (then H_eff = cum), or active == 0;
  - lanes still ACTIVE at a horizon or cap stop → status 4. A run-level should_stop still leaves them ACTIVE (= interrupted, as today);
  - target launch time = clamp(cap/8, launch_ms_min, launch_ms);
  - adapt _iters from the per-step time (dt/iters), not from a truncated iters;
  - a chunk whose deadline has passed still runs prep + fj_init and is marked status 4 with 0 steps.
- New attributes: last_steps, from st[:, S_STEPS], already on the host; last_profile = per chunk {H_played, H_eff, cap_hit, [(cum, t)]}.
- New out keys finished_horizon and cap_hits, only when horizon or cap_s is given.
- Constant ST_HORIZON = 4.

CPU — src/itrea/ea/ls_fj_cpu.py
- _fj_batch (323-...) gets the scalar max_lane_steps and the stop rule after the stall rule (status 4). Off path: 2^62. Numba recompiles once (cache=True).
- FJCpu.run gets horizon=None, cap_s=None. With cap_s, the rows not started by the deadline run with max_lane_steps = 0 (projection only, status 4, 0 steps). This is the sequential meaning of a call deadline, stated in the docstring.
- last_profile = {"kind": "sequential", seconds, steps}.

DISPATCH
- src/itrea/ea/repair.py LocalSearch.run (133-139): forward horizon and cap_s only when given.
- _Project.run (70) and ls_dense DenseLS.run (161): accept them and raise if they are set; the validation already forbids that.

CONTROLLER — new src/itrea/ea/repair_clock.py
- Class RepairClock(cfg, levels), pure NumPy, no backend, no RNG. Methods:
  - decide(now, has_incumbent) → Decision(level, horizon or None, cap_s or None, probe);
  - add_call(profile, steps, status, keys, feasible) — gains are computed when the keys are known;
  - end_generation(now, gen_seconds, cutoff, dual_key);
  - record() → dict for the trace.
- t(h) is computed from GPU profiles by interpolation, or from CPU profiles by the step share.

ENGINE — src/itrea/ea/engine.py
- prepare: build self.rclock when enabled. Record it in the startup / prepare record (608-612) only when enabled.
- _step (931-994):
  - before breeding: dec = self.rclock.decide(...);
  - _ls_run (1124-1143) passes horizon / cap_s only when the clock is enabled. The spy in tests/test_strategy.py:353-386 has a fixed signature, so the off path must not pass them;
  - after _evaluate(C): self.rclock.add_call(ls.impl.last_profile, ls.impl.last_steps, ls.impl.last_status, host keys, feasibility) — one extra host copy of k keys;
  - the same after the evaluations in _with_pending (1164-1194) and _admit (1461-1468);
  - at the end of _step: end_generation.
- Counters ls_finished_horizon and ls_cap_hits are added to self.c only when enabled, so the summary is unchanged when off. SUMMARY_KEYS is not touched.
- gen events (1824-1863): fields rc_level, rc_H_eff (min / max), rc_probe, rc_cap_hits, only when enabled.
- New event kind `rclock` at each gen record: per level G, T, n, rate, and the current choice.
- summary.repair_clock: time share per level, probe time share, switches, cap hits, the H_eff distribution.
- Warm-up (_warmup 648-686) uses cfg.with_overrides, so the controller path is exercised before t0. No new device kernels, so compiled_on_clock stays False.

DOCUMENTS
- vault/10-design/Feasibility jump local search.md: the horizon stop rule and status 4 for both implementations.
- vault/10-design/Trace format and metrics.md: the rclock event and the rc_* gen fields.
- A D-NNN decision record. A Development log entry for each stage.

TESTS
- tests/test_fj_cpu.py:
  - horizon 0 projects only (status 4, steps 0);
  - horizon h: every non-finished row has last_steps == h;
  - counterfactual identity: rows with status FEASIBLE and last_steps ≤ h in an unbounded call are identical (x, status) in a horizon-h call. This is cf_identity.py turned into a test, on steps instead of moves, with the cutoff armed;
  - stall still wins ties as specified;
  - CPU cap: rows after the deadline have status 4 and 0 steps (with a fake clock).
- tests/test_fj_gpu.py (marker gpu; runs only in a GPU window):
  - the same identity;
  - truncated iters give the same rows as an unsliced run;
  - ref_fj (108-150) gets the horizon rule;
  - GPU and CPU agree in semantics on status 4;
  - all active lanes have equal S_STEPS after every launch;
  - multi-chunk: no chunk left unprojected.
- New tests/test_repair_clock.py, synthetic profiles:
  - picks the shallow level when its rate is higher by more than the hysteresis;
  - stays at "none" with no gains;
  - never caps and never leaves "none" before the first incumbent;
  - respects probe_share;
  - decay lets the level follow a phase change;
  - a level is observable when all lanes stopped early.
- tests/test_configs.py:
  - the hash is unchanged while disabled (104-111 pattern);
  - the default yaml lists the section (55-63);
  - the header hash (65-70) is unchanged.
- tests/test_strategy.py reference test (83-95) unchanged.
- One NumPy-backend engine test on a toy with repair_clock enabled (active and shadow): it runs, records rclock events, and the H played is logged per generation.

**cost_and_risk.** COST
- GPU: none added inside kernels.
- Host per launch: one perf_counter and one tuple append. The existing count_nonzero sync is kept. Ladder alignment is not needed (t(h) is interpolated), so no extra launches.
- Per call: one extra device-to-host copy of k keys (256 × 8 B) for the gains. The per-lane steps and status are already copied.
- Controller: O(levels × calls) host arithmetic per generation, about 12 levels.
- CPU: one Numba recompile.
- Memory: none added, so plan_memory and P are unchanged.

INTERACTION WITH THE SLOWEST-INDIVIDUAL EFFECT
- At a finite horizon, every lane runs to H unless it finishes, so the long-plateau tail (lanes resetting the stall counter for thousands of steps) is removed. What remains sets the pace: the lane with the most expensive steps. Those lanes keep choosing the cutoff row, which costs about nnz per step (ls_fj_gpu docstring 21-25).
- A shallow horizon with stall 1000 is NOT lps0. Lanes that would stop at a 100-step plateau keep working to H. That could be better or worse than lps0 at equal wall time.
- Whether that extra work is free depends on whether all 256 blocks are resident in one wave. 48 SMs and the register count of fj_steps are UNVERIFIED; one occupancy query is needed in the GPU window.
- Step-cost heterogeneity under the cutoff row is unmeasured. The sing44 shift from about 0.2 to 1–4 s per generation is its symptom, and the cap and the decayed t(h) handle it.

INTERACTION WITH THE ADAPTIVE STRATEGY
- One call per repair with per-row scale indices stays one call. The horizon is a call-level scalar; strategy allocation and credits are untouched.
- The rhs strategies died out at the default depth during the feasibility phase (sing44, ca). That phase stays deep here.

INTERACTION WITH THE LP STREAM
- Each stream point is its own 4-row call, now bounded by the same level.
- During the deep pre-incumbent phase the stream still misses ids as in lps1 (95–358 on glass4, fhnw, loue). Post-incumbent at a shallow level it should miss about as few as lps0 (0–17).

INTERACTION WITH THE EXCHANGE
- Shallow levels mean more generations, so more posts (item_placement lps0: 1,757 posted, 26 % accepted). No consistent SCIP nodes/s cost has been measured.
- Generation-counted mechanisms (stagnation 50, frontier every 5, at most one tick and one post per generation) speed up in wall time, as in lps0.

RISKS OF FOOLING ITSELF
1. The counterfactual is myopic. It sees which lanes of THIS generation would have finished, and how fast, but not the population and recombination effect of running more generations. item_placement's default-depth win is partly that effect (EA steps ≥ 0.01: 1.17 vs 0.17 per 10 s), so the controller may stay deeper than optimal there. It may also go too shallow on glass4 if its big jumps need a deep-evolved population rather than a deep last call. This is the largest thin spot.
2. Shadow-mode "would choose" logs are not outcome evidence. Only P(120) against an interleaved control is.
3. The controller's knobs (ladder, half-life 20 s, hysteresis, 10 % probes, cap 10 s) are set from 9 dev instances. Any tuning must be logged as development work, never done on the test set (AGENTS.md: tuning only on the tuning set, within the logged E4 budget).
4. The 10 s cap's only measured target is ca, and that rests on one seed.
5. fhnw's deep benefit is one event in one seed.
6. The loue confound (the A3 default config at 0.344 beats both lps arms) is not addressed by this design.
7. Probes cost up to 10 % of post-incumbent wall time, which matters most on item_placement.
8. A seed spread of 0.277–0.522 at the default depth on item_placement means 2 seeds cannot confirm a gain there.
9. The CPU reference shares the horizon semantics but not the call-deadline semantics (rows after the deadline get horizon 0). This is documented and has no effect on the GPU arms.

**measurement.** STAGE 0 — CPU only, now-safe once implemented (no GPU)
- The NumPy tests above, under the project's pinned single-thread rule. The key test is the counterfactual identity on steps.
- Its moves analogue already holds today (/tmp/itrea-design/rclock/cf_identity.py): 3 toys, 64 rows, M ∈ {1, 3, 10, 30, 100}, identical x and status for every row feasible by M. It was run with NUMBA_CACHE_DIR and PYTHONDONTWRITEBYTECODE pointed away from the repository only because of today's read-only rule; the real suite does not need that.

STAGE 0b — GPU window, about 1 minute
- Occupancy of fj_steps at 128 threads (registers per thread, resident blocks per SM × SM count vs 256): is a horizon's extra work on early-stopping lanes free?
- The gpu-marked FJ tests.

STAGE 1 — shadow mode, the cheapest test that the signal works (no behaviour change)
- Arm A3@rc0 = lps1 + repair_clock.enabled, mode shadow (plays "none" = lps1, logs its choices and the per-level G and T).
- item_placement_9903, glass4, sing44, neos-960392, neos-3024952-loue, ca__Scenario400_reserves_1 × seed 0 × 120 s = 6 runs, about 15 minutes, under the project's lease, gates and one-GPU-arm rule.
- Pass:
  - (a) on item_placement the shadow choice moves to a level whose t̂ is ≤ 0.1 s per children call within about 2 s of its first incumbent;
  - (b) on glass4 and neos-960392 it stays at "none" or one level below for most post-incumbent generations;
  - (c) cap hits occur only on ca;
  - (d) P and generation counts match the recorded lps1 runs within seed noise (no hidden overhead).
- The logs also answer the open question the traces cannot: on glass4, do the large jumps come from lanes finishing late (high S_STEPS)?
- If (a) or (b) fails, stop. The myopic signal is then wrong, and the design needs a population-level signal.

STAGE 2 — active, one variable against lps1
- A3@rc1 = lps1 + repair_clock.enabled (active), INTERLEAVED with an A3@lps1 control in the same pool run.
- 9 dev instances × seeds 0–1 × 2 arms × 120 s = 36 runs, about 75–80 minutes.
- Do not compare against the 16:03–17:30 batches: different time, drift and contention.
- Report:
  - P(120) per instance and bounded d vs the interleaved lps1, plus vs the recorded lps0 for reference only;
  - per run: time share per level, probe share, cap hits, the H_eff distribution, generations;
  - hygiene: compiled_on_clock False, GPU health ok, every incumbent checked.
- Success:
  - item_placement P ≤ about 0.42 (lps1 0.569, lps0 0.358);
  - none of fhnw, sing44, glass4, ca, neos-960392, loue worse than −0.05 in d vs the interleaved lps1;
  - radiationm and rail01 equal;
  - mean d vs lps1 > 0.
- Then add seeds 2–3 on item_placement (spread 0.277–0.522) and ca (one-seed evidence) before writing any gain in the Development log.

**expected_effect.** Hypotheses, none measured. The CPU identity check is the only evidence so far.

- Pre-incumbent:
  - identical to lps1 by construction;
  - the first-feasible wins should therefore carry over: sing44 12.7/16.6 s, ca 35.5 s, loue 2.1 s, fhnw seed 0's find, with P near lps1's 0.362 / 0.297 / 0.369 / 0.651;
  - LP arrival times as in lps1.
- item_placement (post-incumbent from 0.05 s):
  - after about 5 deep generations (about 1.6 s) the controller should settle at a horizon giving generations of about 0.07–0.12 s, with up to a 10 % probe tax;
  - expected P(120) between lps0's 0.358 and about 0.45;
  - the myopic signal may undershoot the population effect, so it may land nearer 0.45.
- glass4, neos-960392, loue:
  - stay deep if the shadow logs confirm that their large gains come from late-finishing lanes;
  - expected about lps1: 0.228 / 0.032 / 0.369.
  - If not, glass4 drifts toward lps0's 0.323, and stage 1 should catch that before stage 2.
- ca:
  - the 10 s cap removes the 57–59 s post-incumbent generations;
  - neutral to a small gain (one-seed evidence).
- radiationm18-12-05 and rail01: unchanged (P 0.234 / 1.000).
- Net target: mean bounded d vs lps1 of about +0.03 to +0.05. Almost all of it would come from item_placement (up to +0.37 if it reaches lps0's level), with no instance worse than −0.05.
- A secondary gain beyond P: every generation is bounded. No children or immigrant call outlasts its cap after the first incumbent, and each one is logged with its effective horizon, so it can be replayed.

### Ride-along deep lane: persistent deep feasibility-jump slots launched in the same grid as the children call, with a fixed per-call work quota and an adaptive slot count

**mechanism.** WHAT RUNS
- Every generation's children call keeps the DEFAULT repair budget (max_steps 2000, stall_steps 100) for all O=256 children. That is lps0's shallow tier.
- Beside it, the engine keeps D_active "deep slots" (at most D_max) whose FJ state persists across generations: x, act, w, violated list and positions, the STI counters (including the RNG) and STD (F, Fbest).
- The deep slots use lps1's caps: deep_lane.max_steps 20000 and deep_lane.stall_steps 1000.
- Deep slots are NOT a separate call. They are extra blocks in the SAME fj_steps grid as the children:
  - grid = kc + D_max blocks; block b < k_sh (= kc) is a child, exactly as today;
  - block b >= k_sh is slot b - k_sh, with its own array base pointers, scale index and deep caps;
  - empty, paused and finished slots return at once (the existing `if (st[S_STATUS] != 0) return`).
- A slot's search spans several generations, as in "local-search chains" (Molina et al. 2010).
- On a call that has several chunks (ca: 117 rows per chunk), the slots ride in every chunk's launches until they pause.

STOP RULE: PAUSE AT A DETERMINISTIC WORK QUOTA, NEVER A CLOCK CUT
- Thread 0 adds a work count to the free STI slot 7 (N_SI = 8; only slots 0-6 are used; STI comes from cp.zeros, so the count is inert for children). Per step it adds:
  - 1 + ceil(Wn/BD), where Wn is the scored length of the chosen row. This charges cutoff-row steps, which the docstring says cost about nnz each;
  - + ceil(colLen(j)/BD) when a move is applied;
  - + ceil(nv/BD) at a local minimum.
- A deep block whose work reaches its per-call target qend[slot] writes the new status ST_PAUSE = 4 and returns. The host's `active` count then no longer waits for it.
- At the next call the host resets status 4 to 0 by a host-to-device write of the D-entry status column. This needs no new kernel.
- Per-slot quota for call g+1:
  - Q = min(quota_frac × Σ over chunks of the max child work in call g, work_share × total child work in call g / occupied slots);
  - defaults quota_frac 1.0, work_share ρ 0.5.
- The first cap keeps a slot from running longer than the slowest child of the previous call.
- The second cap bounds the stretch even if co-resident blocks slow each other (throughput-bound case): deep work per call is at most ρ × shallow work.
- Slicing does not change outcomes: a slot's step sequence does not depend on how launches are sliced, and the existing slicing tests keep holding.

REBASE WHEN THE CUTOFF MOVES
- set_cutoff rewrites d_ru[m] and the rhs rows of the scale table. After that, a persistent slot's violated list and F are stale.
- `fj_init` gets a mode argument and a block-to-slot map (bmap):
  - mode 0 = today's init, used when a slot is admitted;
  - mode 1 = rebase.
- Rebase keeps W and the counters (moves, steps, lmin, since, work, RNG), recomputes act from x, rebuilds vlist/vpos, sets F = Σ w·v, and sets Fbest = F.
- **`since` carries over.** A moving target therefore does not grant fresh patience, which bounds a slot's life on item_placement, where the incumbent moves almost every generation. A slot is also harvested at age ≥ max_age (30 calls).

ADMISSION (the "selected few"). Free slots are filled after the children are evaluated, in this priority order:
1. Rows in the deep queue:
   - With route_lp (default true), `_lp_arrived` (the 32 LP roundings) and `_lps_use` (4 rows per stream point) push their rows into this queue. They do not append a pending batch with repair=True, so their separate repair calls disappear.
   - The LP-point batch raises D_active once, to min(D_max, queue length).
   - If the queue holds more than 2·D_max rows, the oldest stream rows are dropped and counted.
2. This generation's children that ended STALL or BUDGET (not FEASIBLE), lowest augmented violation first:
   - Vaug = V (the engine's l2-normalised violation) + max(0, key − cut_rhs)/‖c‖₂ while the cutoff is armed;
   - at most one child per parent-A id;
   - exact duplicates of occupied slots are skipped (DupHasher).
- The slot works on a COPY: the shallow child still enters survival as today.
- The slot keeps the child's strategy gene (scale index). There is no new strategy axis, so strategies() and its 4-tuple consumers are untouched.

HARVEST
- A slot is harvested after a call when its status is FEASIBLE (which means feasible and better than the incumbent once the cutoff is armed), STALL or deep BUDGET, or when its age reaches max_age.
- The x row is copied out, the singleton decoder is applied (the same decoder → FJ → decoder order as `_repair`), and the row is appended to `self.pending` as `_Batch(src="deep", repair=False, origin=source origin | ORIGIN_REPAIR, strat=…)`.
- `_with_pending` evaluates it, and it competes in ordinary survival in the same generation.
- There is no free origin bit (uint16 full: ORIGIN_X_AGREE = 1<<15), so provenance goes into a new trace event `deep`: slot, source id, new id, status, age, work, V and key on entry and on harvest.

ADAPTATION OF THE ONE KNOB, D_active (depth per instance and over time)
- Every `epoch` = 10 generations the controller compares credit per unit of work for the two tiers. Its signals are work counters and evaluation results only (no wall clock), so a run with a given seed is reproducible.
- Credit is winner-take-all and weighted by size:
  - when the population's best changes in a generation, the improvement goes to the tier of the individual that made it;
  - children count as shallow, harvested rows as deep; LP, injected and immigrant rows credit neither;
  - in the incumbent phase the improvement is the key decrease divided by max(|best − dual_key|, ε), i.e. gap units;
  - in the feasibility phase it is the relative decrease of best V, and the first feasible solution counts 1.
- Rates: r_d = C_deep/W_deep and r_s = C_shallow/W_shallow over the epoch's work units.
- Update rule:
  - r_d > r_s → D ← min(D_max, ceil(1.5·D));
  - r_d < r_s → D ← max(slots_min, floor(D/1.5));
  - both zero → grow in the feasibility phase (the stagnation signal S4), keep otherwise.
- Bounds: slots_min 4, slots_init 16. D_max = min(slots_max 128, R_max − min(O, chunk), floor(mem_frac 0.25 × local-search budget / bytes per slot)).
- R_max is read at prepare, before t0: cupy_backends.cuda.api.driver.occupancyMaxActiveBlocksPerMultiprocessor(fj_steps.ptr, 128, 0) × the device's MultiProcessorCount.
- Sources: credit as gain per unit of LS effort (Ong & Keane 2004); extreme credit for rare large gains (Fialho et al. 2010); multiplicative success/failure update of a working limit (Chmiela et al. 2023, Hendel 2022).

**why_it_fits.** DATA → DESIGN
- Deep repair's gains on the six come from a few rows, and almost all of them do not need the whole population deep-repaired every generation:
  - ca seed 0: one LP-rounding immigrant became feasible 6 s after the HiGHS LP (35.5 s, gap 0.002), vs V 0.042 at the default depth;
  - sing44: an l2/lp child at gen 38/48;
  - fhnw seed 0: a gen-6 toward_lp child;
  - glass4, loue, neos-960392: a few large jumps per generation (EA steps of at least 0.01 gap: 0.074 per generation deep on glass4; none at the default depth).
- item_placement needs generations: 4.8x more at the default depth, and deep only adds redundant improving children (30 vs 5.7 per generation).
- Two tiers give both: the default depth paces the generation, and the slots carry the LP rows and the most promising stalled children deep.
- The LP/stream routing also removes calls that the code map shows run one after another, each with its own tail and ramp (glass4 lps0: 649 extra calls for 220 generations, 430 of them LP-stream batches).

GPU FACTS USED
- One block per individual, BD = DEFAULT_THREADS = 128.
- I read the cached cubins (~/.cache/itrea/cupy/c9c1eed2….cubin and cebbaea4….cubin, both BD 128; ELF .nv.info parsed on the CPU, no GPU use). fj_steps uses 64 registers per thread, a 1,040 B stack per thread (the candidate arrays, matching the docstring's "~1 KB × resident threads") and 1,856 B static shared memory.
- At 65,536 registers per SM that is 8,192 registers per block → 8 blocks per SM. With the docstring's 48 SMs, at most 384 fj_steps blocks are resident.
- So a 256-child call is one wave (about 5.3 blocks per SM) with about 128 more block slots free. This is derived from the register count plus assumed sm_121 limits; it was not measured with the occupancy API. That is why D_max is computed at runtime.
- ca runs 3 chunks of 117 rows in sequence; each chunk + D_max fits the residency bound, but memory binds first there (about 11.6 MB per slot).
- A separate call on a small batch would add its own serial tail. A second CUDA stream would need a second host launch loop (kernels are sliced at 50 ms for kernelExecTimeoutEnabled), would make harvest timing nondeterministic, and would compete with evaluate/mutation kernels. Deep every k-th generation stretches that generation by the deep tail (glass4 4.4 vs 0.54 s per generation; at k=8 item_placement would lose about 32 % of its generations).
- Co-launching uses capacity the call already holds and adds no wave or tail by construction.

IS THERE SPARE CAPACITY? Whole-run aggregate FJ steps per second, lps0 vs lps1 (summaries; /tmp/itrea-design/twotier/throughput.py), in k steps/s:

| instance | lps0 (default) | lps1 (deep) |
|---|---|---|
| rail01 | 193 | 1,498–1,512 |
| radiationm | 277–278 | 836–863 |
| ca | 24–25 | 45 |
| sing44 | 123–130 | 178–226 |
| item_placement | 930–943 | 1,135–1,146 |
| neos-960392 | 373–487 | 509–588 |
| fhnw | 57–106 | 99–160 |
| glass4 | 80–96 | 89–90 |
| loue | 160–167 | 117–126 |

- So at the default depth the GPU's FJ capacity is underused on most instances, clearly on rail01 and radiationm.
- Per repair-second it is ambiguous: item_placement is about 1.31M vs 1.23M steps/s. Latency-bound and throughput-bound blocks cannot be told apart from the traces, so the ρ cap bounds the stretch in the worse case.

TAIL SHAPE (CPU probe /tmp/itrea-design/twotier/steps_probe.py)
- Setup: Numba FJ, children = uniform crossover of shallow-repaired gen-0 rows + 4 mutated integer genes, cutoff armed only where gen 0 had a feasible row.
- Idle block-step share of a shallow children call (1 − mean/max per-row steps): glass4 0.10, neos-960392 0.39, item_placement 0.64, loue 0.67, sing44 0.68, fhnw 0.71.
- A deep row needs (deep mean steps ÷ shallow max steps) about 1.3 (neos-960392), 1.6 (loue), 2.4 (sing44), 2.7 (fhnw) or 7.8 (glass4) calls, so slots finish within a few generations. glass4's short tail means its slots rely on co-residency, not on idle tail time.
- On item_placement, 253/256 children finished FEASIBLE at the shallow budget, so stalled candidates are scarce. The lane should drift to slots_min there by itself.
- The stall rule ends 90–100 % of repairs at both depths, so the deep slot's stall 1000 is the operative depth lever.

THIN EVIDENCE
- The probe runs on the CPU with gen-0-derived children and mostly no cutoff.
- fhnw rests on one seed and one event; ca on one seed; neos-960392's margin is 0.007.
- glass4 and loue are confounded: A3 default config 0.247 / 0.344 matches or beats lps1.
- lps0 and lps1 ran about an hour apart.
- Residency is derived, not measured.

**implementation.** CONFIG (src/itrea/config.py)
- New `@dataclass DeepLaneCfg`, section `deep_lane`. Fields:
  - enabled False
  - max_steps 20000, stall_steps 1000
  - slots_min 4, slots_init 16, slots_max 128
  - mem_frac 0.25
  - quota_frac 1.0, work_share 0.5
  - max_age 30
  - route_lp True
  - epoch 10, step 1.5
  - per_parent 1
- `EAConfig.deep_lane = field(default_factory=DeepLaneCfg)`.
- Validation: 0 ≤ slots_min ≤ slots_init ≤ slots_max; stall_steps ≥ 1; max_steps ≥ 0; 0 < quota_frac ≤ 1; work_share > 0; 0 < mem_frac ≤ 1; epoch ≥ 1; step > 1; max_age ≥ 1.
- Hash: in `EAConfig.hash` (266-281) add `if not self.deep_lane.enabled: d.pop("deep_lane", None)`, as for strategy and lp_stream. All recorded hashes stay: 9ac4a93bf93b, 584ddee83747, a2ad94e9188d, 4a96b872ffdc.
- Regenerate configs/itrea2-default.yaml; its header hash is unchanged.
- New dev config configs/itrea2-adaptive-lps-deep.yaml = itrea2-adaptive-lps + deep_lane.enabled (or a job override).

KERNEL (src/itrea/ea/ls_fj_gpu.py)
- Constants: S_WORK = 7, ST_PAUSE = 4, ST_EMPTY = 5.
- fj_steps, appended arguments:
  - k_sh;
  - deep base pointers XD, ACTD, WD, VLISTD, VPOSD, STID, STDD;
  - sidxd and qend (int64 per slot);
  - dmax_moves, dstall.
- fj_steps logic:
  - at the top: `deep = b >= k_sh`, slot index `bb = b − k_sh`; pick pointers and caps;
  - the stop rule (377-387 and the end-of-slice re-check 537-543) gets `else if (deep && s_work >= qend[bb]) stt = 4`;
  - s_work is loaded and stored like s_moves, and incremented by thread 0 with the work formula.
- fj_init, appended arguments `mode` and `bmap`:
  - mode 0 = today;
  - mode 1 = rebase: skip `w[i]=1`, keep the counters, rebuild act/vlist/vpos, F = Σ w·v, Fbest = F, `since` kept.
- With no slots, k_sh = kc and the grid is kc. Children arithmetic is unchanged and the work counter is never read, so the output stays bitwise equal to today's.
- Re-read num_regs after the change; D_max follows it automatically.

HOST, GPU
- New class `DeepSlots` in ls_fj_gpu.py. Persistent arrays:
  - X (D_max, n) f64;
  - ACT and W (D_max, m_aug) f64; VLIST and VPOS int32;
  - STI (D_max, 8) and STD (D_max, 2);
  - sidx, qend.
- Host mirror per slot: occupied, age, src_id, origin, strat, V_in, key_in, admit_gen, work.
- Methods: admit(rows, meta), harvest(max_age), rebase_pending, set_quota(Q).
- `FJGpu.run(X, keep_state, should_stop, scale_idx, deep=None)`:
  - before the first chunk: rebase if pending; write status 4 → 0 for the slots;
  - each launch: grid kc + D_max; active = children with status 0 plus slots with status 0;
  - after the call: one D_max × 8 int64 read (STID);
  - new attribute `last_work` (k,) from the existing `st = cp.asnumpy(STI[:kc])` (855), plus per-chunk max work.
- `set_cutoff` sets deep.rebase_pending. Seeding of slots: seed salted with a DEEP constant, idx0 = slot, call = an admission serial number.

HOST, CPU (src/itrea/ea/ls_fj_cpu.py)
- New `@njit _fj_deep_batch`, with explicit state arrays and its own splitmix64/xorshift64 RNG kept in STI[:, S_RNG], the same pause/quota/work rules, and a rebase function.
- `_fj_batch` keeps its semantics and gets only a `lane_work` output.
- `FJCpu.run(..., deep=None)` runs the children rows, then the slots, in the same run() call.
- On the CPU there is no co-residency: deep work adds to the time one row after another. The CPU path is for semantics tests and arm A4.

DISPATCHER (src/itrea/ea/repair.py): LocalSearch.run forwards `deep`; there is a factory for DeepSlots.

ENGINE (src/itrea/ea/engine.py)
- plan_memory (214): new fixed term `deep_lane` = slots_cap × (8n + 24(m+1) + 104). It can lower P on large instances.
- prepare (515-621):
  - compute R_max (occupancy API × SM count; CPU = unbounded) and D_max;
  - allocate the slots before t0;
  - write {regs, local bytes, R_max, D_max, bytes per slot} into the prepare record.
- Warm-up (648-707): the sub-engine uses with_overrides(deep_lane.max_age=1, slots_init=2) with an armed cutoff, so admit, rebase, harvest and the co-launch all run before t0. compiled_on_clock checks this.
- `_step` (931-994):
  - the children call `_repair(C, cs, deep=True)` passes self.deep to `_ls_run`; the children call stays the only abort-guarded call;
  - after `_evaluate` (972) and before `_with_pending`, a new `_deep_cycle`: harvest into pending (src "deep", repair=False), admit (queue first, then Vaug-ranked stalled children), set the quota from `last_work`, and run the controller every epoch generations.
- `_lp_arrived` (1740) and `_lps_use` (1708): when route_lp, push to `self.deep_queue`.
- `_refresh_cutoff` (1417) leads to set_cutoff, which marks the rebase.
- Credit bookkeeping in `_after_survival` (1264).
- The controller is a pure function `deep_slots_next(D, cd, wd, cs, ws, feas_phase, cfg, D_max)`.
- Records:
  - `deep` sub-dict in gen events: D, occupied, admitted, harvested by status, work_deep, work_shallow, credit for each tier;
  - new `deep` event kind;
  - summary `deep_lane` block, and per-strategy sc counters deep_harvested and deep_best. None of these go into SUMMARY_KEYS.
- Update vault/10-design/Trace format and metrics.md and the FJ spec (pause/quota/work/rebase), keeping implementations semantically equal.

TESTS (CPU-runnable unless marked gpu)
- test_configs: the four hashes are unchanged when disabled and change when enabled; the default yaml is regenerated with the same header hash; add deep_lane to the disabled-section test (104-111); validation errors.
- test_fj_cpu:
  - lane_work equals an independent recount;
  - quotas [q, q, q, ∞] vs one unsplit run give bitwise-equal x, act, w and counters;
  - pause status 4, and the work since call start stays within [Q, Q + one step's work);
  - after rebase, vlist equals the violated set (including the cutoff row), F = Σ w·v, Fbest = F, since unchanged;
  - deep caps are honoured, not the shallow ones.
- test_fj_gpu (gpu, later): ref_fj mirrors work and pause; the D=0 path is bitwise equal on the existing cases; split-quota resume equivalence.
- test_strategy: the spy (353-386) still sees ONE run call per children repair; with route_lp there is no call per LP or stream batch.
- test_review_ea: should_stop with slots — slots keep state, and only children rows count as interrupted.
- New tests/test_deep_lane.py (NumPy backend, itrea.benchmarks.toy):
  - two runs with the same seed and a generation limit give identical genealogy and deep events;
  - D stays within [slots_min, D_max];
  - a harvested row enters survival with a `deep` event;
  - controller cases (grow, shrink, both zero by phase, clamps);
  - the memory term is present;
  - warm-up covers admit, rebase and harvest.
- tests/_engine_reference.py (dd31d67) must pass unchanged with the lane disabled.

**cost_and_risk.** GPU COST
- Up to D_max extra blocks per launch; empty, paused and finished ones return at once.
- A second count_nonzero per launch, only once the children are done.
- One read of D_max × 8 int64 per call.
- A D-block fj_init per admission round and per cutoff change. item_placement changes the cutoff almost every generation, but m = 195, so this is cheap.
- Persistent memory per slot = 8n + 24·m_aug + 104 B: glass4 about 13 KB, sing44 about 1.8 MB, ca about 11.6 MB. mem_frac 0.25 of the 1 GiB budget gives about 22 slots on ca. It enters plan_memory and could lower P on very large instances.

CPU COST: an argsort over O for Vaug, DupHasher over D rows, and the controller. All negligible next to a generation.

SLOWEST-INDIVIDUAL EFFECT
- Each slot's quota is at most the previous call's slowest child, counted in work units, so slots are off the critical path by construction.
- ρ = 0.5 caps their share of the work if co-resident blocks slow each other.
- Residual risk: work units are a proxy for time (column lengths vary, plus co-residency contention). Gate 0 and max_generation_s vs lps0 detect it.

ADAPTIVE STRATEGY: inheritance, redraw and the 5 % floor are untouched; slots keep the strategy gene; credit is recorded per strategy but not used for allocation.

LP STREAM
- Routing removes one call per stream point and deep-repairs the points.
- Points enter the population at least one generation later.
- Queue overflow drops the oldest stream rows (counted).
- Shorter generations should keep lps_missed near lps0's 0–17.

EXCHANGE: unchanged (at most one post per generation). Harvested rows that become incumbents are posted as usual.

DETERMINISM: decisions use work counters and evaluation results only. RNG streams change only when the lane is enabled.

WAYS TO FOOL ITSELF
1. Comparing with lps0 and lps1 measured earlier (drift) → rerun a control in the same session.
2. Two seeds; item_placement's default-depth spread is 0.277–0.522 → read the direction across instances, not single values.
3. glass4 and loue are confounded with the adaptive configuration → a result near 0.25 / 0.34 there cannot be credited to the lane.
4. fhnw is a lottery and ca rests on one seed.
5. route_lp is a second change → arm 1b separates it.
6. Winner-take-all credit can starve a tier whose gains are hidden by the same-generation winner. slots_min 4 keeps probing (but Fialho et al. warn that floors slow adaptation).
7. The residency figure of 384 is derived from the register count, not measured.
8. A kernel change may raise registers and lower D_max.
9. Persistent slots chasing a stale cutoff waste work; the carried-over `since` and max_age bound it.
10. The CPU path stretches generations (no co-residency). Never judge A4 timing with the lane on.

**measurement.** GATE 0: co-launch micro-benchmark
- GPU, about 15 min, no SCIP, under a box-window lease, thermal and GPU-health gates.
- Extend tools/bench_ls.py with --deep. For each of the 9 dev instances: build 256 children from seed_population + one crossover/mutation round, then time the children call 5 times each with D ∈ {0, 32, 128} slots loaded with copies of stalled children, quota = previous max.
- Report call wall time ratios, R_max, num_regs and the occupancy result.
- Pass: median ratio ≤ 1.10 at D=32 and ≤ 1.25 at D=128 on every instance. Where it fails, the regime is throughput-bound and ρ is the binding bound; lower its default before gate 1.

GATE 1: coupled runs
- New arm A3@lpsD = itrea2-adaptive-lps (default repair budget) + deep_lane.enabled. 9 dev instances × seeds 0, 1 × 120 s = 18 runs, about 35 min (the lps0 batch took 16:55–17:30).
- Same-session drift control: lps0 on item_placement and sing44 × 2 seeds (8 runs, about 15 min); add lps1 on sing44 if the window allows.
- Report per instance:
  - P(120), seeds averaged, vs lps0 and lps1 (bounded d);
  - generations, max_generation_s, aborted_generations;
  - first-feasible times on sing44, ca and fhnw;
  - D(t) trajectory;
  - admitted and harvested by status;
  - share of new bests and incumbents from harvests;
  - work_deep / work_shallow;
  - lps_missed and queue drops.
- Acceptance:
  - item_placement P(120) within lps0's seed range (≤ about 0.42) and generations ≥ 85 % of lps0's 1,770–1,840;
  - across the six, mean bounded d vs lps1 ≥ −0.05 and vs lps0 > 0;
  - radiationm and rail01 unchanged.
- Diagnostic failure modes:
  - generations below 85 % of lps0 → throughput-bound; lower ρ;
  - D stuck at slots_min on sing44 or ca → the credit signal is wrong;
  - LP-row harvests never feasible on ca → the quota is too small per generation.

GATE 1b, only if gate 1 passes: the same arm with route_lp=false on the 4 instances where LP rows mattered (sing44, ca, glass4, loue), 8 runs, to attribute the LP routing.

All tuning stays on the dev set, as the E4 budget rules require.

**expected_effect.** These are hypotheses with numbers attached, not findings.

item_placement_9903
- Few stalled children (CPU probe: 253/256 FEASIBLE at the shallow budget with the cutoff armed), so D should drift to slots_min = 4.
- Generations should stay within about 10 % of lps0 (1,770–1,840 in 120 s), so P(120) should stay in lps0's range (0.30–0.42; mean 0.358) rather than lps1's 0.569.

sing44 and ca
- LP rows and low-violation stalled children get stall-1000 repair across generations, and D should grow during the feasibility phase.
- Deep throughput estimate for sing44, deep completions per second:
  - co-residency free: up to about 128 slots / 2.4 calls per slot ≈ 50 completions per generation at about 0.27–0.4 s per generation, i.e. roughly 130–190/s;
  - throughput-bound, ρ = 0.5: about 40/s;
  - lps1 for comparison: 256 per 1.5 s ≈ 170/s.
- So the first feasible solution is expected between lps1's (sing44 12.7/16.6 s; ca seed 0 35.5 s) and lps0's (52.9/62.7 s; 58 s), with P(120) between lps1's and lps0's (sing44 0.362–0.551; ca 0.297–0.395). Near lps1 only if co-residency is close to free.
- On ca, LP rows advance only about one shallow tail per generation (about 0.7 s of tail on a 1.5 s generation). The LP batch's feasible result is estimated around 10 s after the LP arrives, vs 6 s in lps1.

glass4, loue, neos-960392
- Harvests supply the large jumps while generations stay fast.
- Expected P(120) between lps1 and lps0 (glass4 0.228–0.323, loue 0.369–0.440, neos-960392 0.032–0.039).
- glass4 slots need about 8 calls each (short shallow tail), so they rely on co-residency. Its result is the most uncertain.
- Because of the confound, results near the A3-default values (0.247 / 0.344) would not show the lane's value.

radiationm, rail01: unchanged (about 0.234 / 1.000). The GPU has the most spare FJ capacity there (deep ran 3–8x more steps/s), so the lane should cost nothing.

fhnw: no prediction. The one success was a single seed and a single event.

Overall target: mean bounded d vs lps1 ≥ −0.05 on the six, with item_placement held near lps0. If gate 0 shows co-resident blocks slow each other, expect the six to land midway and item_placement to lose ≤ ρ (at most 50 %, likely much less since D stays small there) of its generations.

Scratch evidence for these numbers is in /tmp/itrea-design/twotier/:
- throughput.py — aggregate step rates;
- steps_probe.py and steps_*.npy — per-row step distributions;
- fj_c9.cubin and fj_ce.cubin — resource usage of the extracted kernels.

No repository file was touched, the GPU was not used, and every CPU command ran on cores 10–14 with one thread (Numba cache and bytecode redirected outside the repository).
