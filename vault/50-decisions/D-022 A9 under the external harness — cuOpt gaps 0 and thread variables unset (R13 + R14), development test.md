# D-022 A9 under the external harness — cuOpt gaps 0 and thread variables unset (R13 + R14), development test (2026-10-12, AgentRA on Berk's word)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run, under Berk's
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
