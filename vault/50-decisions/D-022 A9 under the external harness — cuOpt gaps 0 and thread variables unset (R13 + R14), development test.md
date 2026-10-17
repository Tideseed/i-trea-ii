# D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test (2026-10-12, AgentRA on Berk's word)

**Status:** ADOPTED for A9 (2026-10-13 22:2x, all 74 runs: validity holds, mechanism shows (met only vacuously: 0 of 6 A9 runs in either arm ended on cuOpt's "optimal", as SCIP proves first, so the adoption rests on D-021's fairness and GAMS fidelity with no harm shown; status note 23:3x), no harm on the three; the C1
freeze carries harness ext; C1's hint present 0.827, guard +0.063 — inputs to Berk's go / no-go) — see Result.
Decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run, under Berk's
"Go ahead" instructions and his acceptance of D-021 (2026-10-11); open to a superseding decision on evidence. Required
by [[D-021 Our own external benchmark standard (ITX-1) — equal footprint inside the Spark's thermal envelope]] R14 before
the C1 freeze. Development only (D-010). Also serves as C1's interim A9 result on the 17 instances (D-019 addendum).

## Why
D-021 runs every cuOpt process with both MIP gaps 0 (R13: today cuOpt's "optimal" at its default 1e-4 relative gap is
DEFINITIVE in the runner and ends A9's whole run) and without OMP / OPENBLAS / MKL = 1 (R14). Both change A9's
configuration. R14 is measured inert (M1, 2026-10-11: cuOpt CPU-s per wall-s 5.02 / 4.99 / 5.03 unset vs 5.02 / 4.98 /
5.03 set), so the two are tested together — a stated deviation from one variable, as D-021 R14 says.

## Test — fixed before any run
- **Arms** (`jobs/itx-r13.jsonl`, 102 runs, set `itx-r13`, 120 s, seeds 0–1, the 17 development + class-probe
  instances, both LLM servers stopped): **control** A9@r13 = A9 (`configs/itrea2-dev-rc-3way.yaml`) under harness
  {ext, cuopt_gap_zero false, cuopt_env_unset false} (RMM 24 / host 24, project CUDA cache, D-021's voids — only R13 +
  R14 differ); **treatment** A9@r13x = the same under harness ext; same-day **A6F@r13x** (cuOpt alone on F, ext).
  One block per instance × seed, arms in the pool's Williams order.
- **Validity (must hold):** 0 compile on the clock (CuPy and CUDA), every run completed or a recorded void rerun, equal
  P and gene dtype per block, cuOpt's gaps 0 in every treatment solve and default in every control solve.
- **Mechanism (reported):** the number of control runs ended by cuOpt's "optimal" (exit reason optimal, cuOpt first)
  with a final gap > 0 against the reference, and the treatment's counterparts; cuOpt CPU-s per wall-s per arm.
- **Outcome** (bounded d of treatment vs control on seed means, positive = treatment better; band_i = max(0.05, the
  larger seed-0 vs seed-1 bounded difference of the control or of A6F)):
  - **adopt R13 + R14 for A9** (the C1 freeze carries harness ext) if the mean d over the 17 ≥ −0.02 and no instance is
    worse than −band_i on both seeds — a non-inferiority rule: R13 and R14 are adopted for fairness and GAMS fidelity
    (D-021), and the test only guards against harm;
  - **reject** (A9 keeps the default gaps; D-021's A9 rows run under {ext, gap_zero false, env_unset false}) if the
    mean d ≤ −0.05 or any instance is worse than −band_i on both seeds;
  - otherwise **inconclusive** → seeds 2–3 on the instances worse than −band_i on one seed, same rules.
- **Reported, not criteria (C1's interim result, D-013 terms):** A9@r13x and A9@r13 vs A6F@r13x (equal footprint, same
  day) — better / worse, mean d with 95 % bootstrap interval, Wilcoxon; cuOpt's weak set (A6F P ≥ 0.5 here; selected on
  the same runs — stated); vs SCIP alone with the A0 runs of `dev` / `classprobe` (seeds 0–1, earlier days).
- **Prediction (not a criterion):** d ≈ 0 except on instances where cuOpt reaches its 1e-4 gap early (fhnw-binpack4-48,
  neos-3004026-krka), where the treatment can only gain.
- **Analysis:** `itrea build-runs`, then a short script `tools/d022_report.py` written before the window (it reads only
  the run table).

## Clarifications before any run (2026-10-12 13:1x, AgentRA, after the review of `tools/d022_report.py`; no itx-r13 run exists)
Readings the record left open, fixed in the script and here before the window; one is a rule the record did not have
(the follow-up's terminal case), stated as such and open to Berk.
- **The INCONCLUSIVE follow-up** (`tools/d022_report.py --followup`): runs = the instances the seeds-0–1 report names
  (worse than −band_i on exactly one seed) — or all 17 when it names none — × seeds 2–3 × the same three arms and
  harnesses, 120 s, one block per instance × seed, in their own set `itx-r13f` (`itrea make-jobs itx-r13f --instances
  …|all`). band_i stays the one seeds 0–1 gave. A rerun instance's d is the bounded d on its means over seeds 0–3, and
  it is "worse on both seeds" when d_s < −band_i on both seeds 2 and 3; the other instances keep their seeds-0–1 d; the
  mean is over the 17. Validity as for seeds 0–1, on the follow-up's runs. **Reject** if the mean d ≤ −0.05 or a rerun
  instance is worse on both seeds 2 and 3; **adopt** if the mean d ≥ −0.02; **otherwise R13 + R14 are not adopted**
  for A9 (non-inferiority not shown: the reject consequence) and no further seeds run. That terminal rule was not in
  the record; it follows D-018 / D-023's "otherwise not adopted" and is Berk's to overrule.
- **Each reading reads only its own runs:** the 17 instances × the three arms × its seeds × 120 s, not synthetic, not in
  a voided block — the failed runs and the voids included (a void's key comes from its rerun's row or from the voided
  directory's own run.json; a void whose key cannot be read is counted). Runs of other seeds in `itx-r13` do not
  change the seeds-0–1 verdict.
- **The harness is a validity leg:** every run's resolved harness (run.json `config.harness`) is its arm's — ext for
  A9@r13x and A6F@r13x, {ext, cuopt_gap_zero false, cuopt_env_unset false} for A9@r13 — else INVALID (the gap leg and
  R14 alone cannot tell the control from A9 without any harness).
- **Failed runs read as D-002 §6 keeps crashes (`keep`):** a void whose rerun failed is closed by the kept crash; a kept
  crash that died before writing its EA or cuOpt summary (a SIGKILL writes none — D-021 R29 keeps an EA killed at its
  cap as an A9 outcome) is listed as "crashed before the record" instead of failing the P / gene-dtype or gap leg; its
  harness is still checked. Completed runs without those records still fail the legs.
- **A run without P(120)** on any of the three arms (A6F included: band_i needs its seed pair) makes the verdict
  INCOMPLETE, naming the run.
- **C1's weak-set numbers in D-013 terms:** improved k of |W_B|, mean P before → after, mean d with its bootstrap
  interval and Wilcoxon, worst P — over every weak instance with both arms' runs, those where both arms have P = 1
  included at d = 0 (D-013's own table counts rail01 1.0 → 1.0; `backoff.d013_report` does the same). `stats.compare`,
  which leaves them out, gives only the all-instance line; the instances it leaves out are printed.

## Amendment before any run (2026-10-13 15:1x) — targeted, after Berk's instruction
Berk (15:0x): **"Meanwhile why do we need that many experiments? Shall we do targeted and surgical tests? and only if we
see hints of improvement we can go wider to test the methodology's limits. It will be a more efficient way than blanket
testing."** R13 can act only where cuOpt stops on its own 1e-4 "optimal": in all 92 recorded cuOpt runs at 120 s on
the 17 instances that happened only on **fhnw-binpack4-48 (9/9), neos-3004026-krka (10/10) and neos-960392 (3/4)**;
on the other 14 cuOpt never stopped early, so the gap setting cannot change the run (R14 is measured inert). The test
is therefore run on those three: {A9@r13, A9@r13x, A6F@r13x} × seeds 0–1 (18 runs); the rule reads over the three (mean
d ≥ −0.02 → adopt; ≤ −0.05 or an instance worse than −band_i on both seeds → reject; otherwise the follow-up on them),
the other 14 counted as d = 0 by construction (stated, not run). **C1's interim A9 result** is read from the same set
`itx-r13` on cuOpt's weak development instances only (cuOpt-alone P ≥ 0.5 at 120 s in our A6 runs: doon,
cryptanalysiskb128n5obj16, rail01, crna, loue, sing44, ca, item_placement, neos-960392, toguru, comp21 — 11; comp21
borderline at 0.506; selected on 3-core A6, stated): {A9@r13x, A6F@r13x} × seeds 0–1. `jobs/itx-r13.jsonl` holds
these 58 runs (e1403fa); `tools/d022_report.py` gains the instance selection before its reading.

*Clarification (2026-10-13 15:2x, AgentRA, before any reading of itx-r13; runs in progress, none read):* the mean d is
over the three; the other 14 are not run and do not enter it ("counted as d = 0 by construction" above says why they
need no run, not that they are averaged in — over 17 the mean would dilute any effect 17/3-fold; open to Berk). The
follow-up reruns the three's instances worse than −band_i on exactly one seed, or all three when none is (named in the
verdict: `make-jobs itx-r13f --instances all` means the 17); its mean is over the three. C1's block reads A9@r13x vs
A6F@r13x (and vs A0) on the 11 only, with its own completeness line; it never makes D-022's verdict INCOMPLETE or
INVALID (`tools/d022_report.py --instances / --c1-instances`, defaults the three / the 11).

## Second amendment before any reading (2026-10-13 15:3x) — after the adversarial check of D-024; 7 of 58 runs done, none read
Two verifiers (Claude Code subagents) found the first amendment's rule unsound and the C1 hint without a threshold.
Fixed now, before anyone reads an itx-r13 run:
- **Counts corrected:** cuOpt summaries at 120 s on the 17: 141 runs (153 with M1); cuOpt's own "optimal" stop on
  fhnw-binpack4-48 9/9, neos-3004026-krka 10/10, neos-960392 3/7 (A6 3/3, A9 0/4).
- **The R13 + R14 rule (replaces the bounded-d mean over three, which reads noise: two of the three sit at floor-level P):**
  **adopt** if (1) validity holds (as before: harness legs, gaps 0 in every treatment solve and default in every control
  solve, no compile on the clock), (2) the mechanism shows: no treatment run is ended first by cuOpt's "optimal" with a
  final gap > 1e-9, and (3) no harm: on fhnw-binpack4-48 and neos-3004026-krka |ΔP| ≤ 0.005 (D-018's floor rule) on
  seed means, on neos-960392 d ≥ −band_i; **reject** if (3) fails on both seeds of an instance; otherwise INCONCLUSIVE →
  seeds 2–3 on that instance, same rules; still inconclusive → not adopted (as before).
- **Stated assumption:** the 14 other instances are not run because cuOpt never stopped early there; this assumes cuOpt
  uses the MIP gaps only as a termination test (not as a pruning limit), which is not verified from cuOpt's source or
  documentation. The v1 guard (stage 2 / the C1 guard below) carries the check.
- **C1's interim hint, fixed before any reading:** the weak set is the **fixed 11** named in the first amendment (from
  earlier A6 runs; not re-selected on A6F@r13x). **Hint present** iff the ratio of means of P(120) A9@r13x / A6F@r13x
  over those 11 (seed means, both seeds, A9 failures counted as A9 outcomes, D-002 §6 reading) is **≤ 0.95**. The
  "improved > worse" limb is dropped (under a null it passes about half the time). **Guard:** mean bounded d of
  A9@r13x vs A6F@r13x over all 17 ≥ −0.05 — for which the four instances the first amendment dropped
  (chromaticindex1024-7, neos-3656078-kumeu, glass4, radiationm18-12-05; chromaticindex is A9's known worst case against
  cuOpt alone) are run: {A9@r13x, A6F@r13x} × seeds 0–1 = 16 runs, queued after the 58. Both numbers are inputs to C1's
  go / no-go, which stays Berk's (D-019 addendum); vs SCIP alone reported as before (A0 of dev / classprobe).
- **Clarification (16:0x, before any itx-r13 run is read):** the floor limb is **one-sided** — a no-harm rule: on
  fhnw-binpack4-48 and neos-3004026-krka it fails only when the treatment LOSES more than 0.005 (ΔP = P(A9@r13x) −
  P(A9@r13) > +0.005); a gain never fails it (the second amendment's "|ΔP| ≤ 0.005" quoted D-018's two-sided floor
  value, which here would have rejected R13 for helping). The line appended at 15:2x ("the mean d is over the three")
  is superseded by the second amendment: there is no mean. `tools/d022_report.py` implements both (57 tests).

## Addendum — commit times of the amendments (2026-10-13 17:5x, record-keeping only; no rule changes, no run read)
The headings above give approximate times. From git: the first amendment was committed at 15:10:01 (15a5c81), the
second amendment ("15:3x" above) at 15:28:42 (f52e3e8) and the one-sided floor-limb clarification ("16:0x" above) at
15:44:09 (fd75476). The itx-r13 pool started at 15:16:11; when the second amendment was committed, 7 of its 58 runs were
done and none had been read. The session checkpoint of 2026-10-13 16:00 and the incident log (both kept in the
private operations repository) quote the commit times.

## Result (2026-10-13 22:2x, final; all 74 runs) — ADOPT R13 + R14 for A9; C1: hint present, guard holds
**Runs.** Window W-1013, both LLM servers stopped: pool `20261013-151610-335372` (`jobs/itx-r13.jsonl`, 58 runs,
15:16–18:28; 58 completed, 0 failed, 0 void) and the C1 guard pool `20261013-182821-1425786` (`jobs/itx-r13b.jsonl`,
16 runs in set `itx-r13`, 18:28–19:30; 16 completed). **Analysis:** `itrea build-runs` (run table 22:32), then `itrea
d022-report` with its defaults (sets `itx-r13`; rule set the three; C1 the fixed 11 and the 17); re-run for this record
at 22:4x pinned (`OMP_NUM_THREADS=1 nice -n 19 taskset -c 10-14`), the table not rebuilt; the lines below are quoted
from its output. No run failed, so `--failed both` printed one reading ("exclude" = "keep").

**Reading bug fixed before the reading (038c908; not a run fault).** The first report run on the interim table (17:45)
failed validity on 8 "gap settings wrong": in A9@r13x cuOpt (gaps 0) cannot stop on its own, so after SCIP's proof
(fhnw 13.6 s) the runner stopped it (84 s), and a stopped cuOpt writes no summary. The report now reads the gaps from
the runner's record of cuOpt's command line (`--mip-gap-zero`) when the runner stopped cuOpt; test added (58 pass).
Fixed before any verdict was recorded (journal 2026-10-13 17:45).

**D-022's rule (the three, 18 runs):**
- **Validity holds:** "missing 0; failed 0; compile on the clock 0; harness not the arm's 0; P / gene dtype mismatches 0;
  gap settings wrong 0; voids 0 (without a rerun 0)"; R14: thread variables not as the harness says on 0 runs.
- **Mechanism shows:** no A9@r13x run ended first by cuOpt's "optimal" with a final gap > 1e-9 (0 of 6). Stated plainly:
  no A9@r13 (control) run was ended that way either (0 of 6) — in A9, SCIP proves first on these three, so cuOpt's
  1e-4 stop never ended an A9 run at 120 s; A6F@r13x ended by cuOpt's "optimal" on 5 of 6 runs, every one at final gap
  0. cuOpt CPU-s per wall-s, median: A9@r13 3.10, A9@r13x 3.02, A6F@r13x 4.96. So R13 changed no A9 run's end here;
  its adoption rests on D-021's fairness and GAMS fidelity, with no harm shown.
- **No harm:**

  | instance | limb | P A9@r13 | P A9@r13x | P A6F@r13x | seed means | s0 | s1 | limit | leg |
  |---|---|---|---|---|---|---|---|---|---|
  | fhnw-binpack4-48 | floor (one-sided) | 0.013 | 0.011 | 0.006 | ΔP −0.0016 | −0.0048 | +0.0016 | ΔP ≤ +0.005 | holds |
  | neos-3004026-krka | floor (one-sided) | 0.003 | 0.003 | 0.006 | ΔP −0.0001 | −0.0001 | −0.0001 | ΔP ≤ +0.005 | holds |
  | neos-960392 | band | 0.059 | 0.058 | 0.710 | d +0.0224 | +0.0244 | +0.0199 | d ≥ −band (0.41) | holds |

- **Verdict: ADOPT R13 + R14 for A9** — "validity holds; mechanism shows …; no harm on the 3: the C1 freeze carries
  harness ext". D-023's backoff2 runs use ext on all three arms (D-023's amendment of 2026-10-13 22:5x). The stated
  assumption stands (cuOpt's gaps act only as a termination test; not verified from its source); the C1 guard over the
  17 is its check, and it holds (below).

**C1's interim A9 result (inputs to C1's go / no-go, which is Berk's, D-019 addendum; not part of D-022's rule).**
Completeness: 68 of 68 runs (17 instances × A9@r13x / A6F@r13x × seeds 0–1), A0 (dev / classprobe, the fixed 11) 22 of
22.
- **HINT present:** ratio of means of P(120) A9@r13x / A6F@r13x over the fixed 11 = **0.827** (0.585 / 0.707; present
  iff ≤ 0.95, fixed before the runs).
- **GUARD holds:** mean bounded d of A9@r13x vs A6F@r13x over the 17 = **+0.063** (holds iff ≥ −0.05).
- Per instance, the fixed 11 (seed means, P(120) A6F@r13x → A9@r13x, bounded d; from the report's `--json` rows):

  | instance | A6F@r13x | A9@r13x | d | |
  |---|---|---|---|---|
  | neos-960392 | 0.710 | 0.058 | +0.919 | better |
  | neos-2657525-crna | 0.757 | 0.461 | +0.391 | better |
  | ca__Scenario400_reserves_1 | 0.586 | 0.394 | +0.329 | better |
  | item_placement_9903 | 0.535 | 0.364 | +0.320 | better |
  | neos-3024952-loue | 0.756 | 0.564 | +0.254 | better |
  | sing44 | 0.491 | 0.445 | +0.093 | better |
  | rail01 | 1.000 | 1.000 | 0 | tie (both 1) |
  | cryptanalysiskb128n5obj16 | 1.000 | 1.000 | 0 | tie (both 1) |
  | comp21-2idx | 0.504 | 0.548 | −0.080 | **worse** |
  | neos-4763324-toguru | 0.549 | 0.603 | −0.090 | **worse** |
  | neos-2746589-doon | 0.894 | 1.000 | −0.106 | **worse** |

  The other 6 of the guard's 17: glass4 0.298 → 0.216 (+0.274), neos-3004026-krka 0.006 → 0.003 (+0.465, floor-level
  P), neos-3656078-kumeu 0.095 → 0.103 (−0.076), radiationm18-12-05 0.008 → 0.011 (−0.226, floor-level P),
  fhnw-binpack4-48 0.006 → 0.011 (−0.508, floor-level P), chromaticindex1024-7 0.061 → 0.544 (−0.889, A9's known worst
  case against cuOpt alone — the case D-023's phase Y targets).
- **Reported, not criteria (D-013 terms):** vs cuOpt alone (A6F@r13x, same day, equal footprint) on the 11: improved 6 of
  11, worse 3, mean P 0.707 → 0.585 (−17 %), mean d +0.185 [+0.032, +0.368], Wilcoxon p 0.074, worst P after 1.000
  (rail01 and cryptanalysiskb128n5obj16 both at 1, counted at d = 0). vs SCIP alone (A0 of `dev` / `classprobe`,
  earlier days): improved 8 of 11, worse 0, 0.880 → 0.585 (−34 %), d +0.331 [+0.172, +0.500], p 0.0078; all 11 are in
  SCIP's weak set (A0 P ≥ 0.5).
- **Caveats.** The fixed 11 were chosen on earlier 3-core A6 runs (cuOpt-alone P ≥ 0.5; comp21 borderline at 0.506) and
  not re-selected on A6F@r13x — on these runs sing44 (0.491) would fall just outside. The A0 runs are from earlier days
  (sets `dev` / `classprobe`), not same-day, on SCIP's own harness. The gain is concentrated: neos-960392 alone carries
  48 % of the 11's summed P reduction (0.652 of 1.345); without it the ratio would be 0.902, still ≤ 0.95 (computed from
  the report's rows; a robustness figure, not a rule). The hint is futility-only (D-024 rule 2): it gates whether the line
  widens and declares nothing.
