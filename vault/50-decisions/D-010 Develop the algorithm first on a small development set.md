# D-010 Develop the algorithm first on a small development set (2026-10-06 08:2x, Berk)

**Berk:** "We shall do fewer test problems and focus on improving the algorithm first."

**Status:** decided; the confirmatory test-set size (point 4) is a proposal until Berk confirms it (issue #11).

## Why this fits the pre-registration
D-002 already puts the algorithm work (E1: the EA alone; E4: tuning) on the non-test side **before** the confirmatory
runs (E2/E3). This decision changes the order and the sizes, not the logic: the algorithm is developed openly on a
small development set; the confirmatory test is still selected later, by the pre-registered rule, from instances the
development never touched.

## Decisions
1. **Development set (frozen now, `datasets/dev_set.csv`, 9 instances):** chosen from the stage-1 screening (seed
   900, one seed) for coverage and fast iteration (nnz ≤ 400k): four MIPLIB optimisation-hard (glass4 — small;
   sing44 — mixed binary/continuous; radiationm18-12-05 — general integers; neos-960392 — pure binary), three MIPLIB
   feasibility-hard without a solution in 300 s (neos-3024952-loue — general integer; fhnw-binpack4-48 — bin packing;
   rail01 — set partitioning, 117k binaries), one PGLib-UC case without a solution (ca__Scenario400_reserves_1) and one
   ML4CO item placement case (item_placement_9903). They were shortlisted after looking at the stage-1 table — fine
   for development, so they are **permanently excluded from the confirmatory test set**.
2. **The development set replaces D-002's tuning side** (E1 and E4 run on it; E4's cap — at most 12 configurations ×
   the set × 120 s — still applies to the final configuration choice, every configuration logged).
3. **Yardstick for every change** (fixed before the first development run): per instance, 120 s, ≥ 2 seeds — SCIP
   alone (A0, harness-matched), the EA alone (A5), the coupled system (A3); cuOpt alone (A6) once per instance as a
   reference. Reported: the EA's own P(120), the portfolio P(120), time to first feasible, EA posts / accepted. Every
   configuration and result goes into [[Development log]] with its config hash. Development comparisons are
   **descriptive** (one-seed screening P; PGLib/ML4CO P self-referenced until reference runs exist).
4. **Fewer test problems (proposal):** the confirmatory test set becomes **16 instances** (MIPLIB optimisation-hard 8,
   feasibility-hard 2, PGLib-UC 3, ML4CO 3), selected after development by D-002 §2's quota rule from the candidates
   minus the development set, using the complete two-seed screening. **Minimum detectable effect at n = 16 ≈ 0.3** on
   the bounded d scale (vs ≈ 0.20–0.25 at 32): only a large effect will read "supported"; a smaller one will most
   likely read "inconclusive".
5. **Stage 2 of the screening is deferred**, not cancelled — it is needed only for the confirmatory pool freeze
   (issue #1 stays open).
6. **Order of the algorithm work** (each gates the next): (a) no CuPy kernel compilation on the clock (#4) — otherwise
   every GPU timing is contaminated; (b) the local-search knobs already measured on synthetic problems (row scale `rhs`
   for UC-shaped models, `row_scan_cap` with the cutoff row, the stall rule); (c) operator mix and population size;
   (d) the exchange (what and when to inject / post).
7. **Resources:** daytime windows with **[local LLM agent] stopped only** ([local LLM agent] stays for chat; Berk does not chat with
   [local LLM agent]), one GPU lane (one coupled run at a time: one SCIP core + the GPU EA, far cooler than 8 SCIP lanes),
   fan-governor rest breaks honoured (D-009). Free memory is checked after stopping llama-qwen38 before any run.

**Addendum 08:2x — the GPU is shared with an idle [local LLM agent].** The first development run was voided by the GPU-health
check: [local LLM agent]'s vLLM engine (`ornith-vllm.service`) holds a GPU context even when idle, and every other GPU process
counts as foreign. For development runs the pool is started with `--allow-serving`: GPU processes of the two model
servers (identified by their systemd cgroup) are recorded as `tolerated_pids` in each run's GPU-health record instead
of voiding the run. A chat with [local LLM agent] during a run adds bandwidth noise — acceptable for descriptive development
numbers; confirmatory and screening runs still require both servers stopped and void on any foreign process.
