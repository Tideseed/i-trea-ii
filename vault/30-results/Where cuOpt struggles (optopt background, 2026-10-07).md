# Where cuOpt struggles — background from optopt's runs (2026-10-07)

**Data:** `berkorbay/optopt` `datasets/runs.parquet` (commit 4dd48395 lineage) — cuOpt 26.08 as shipped (strategy C0)
and SCIP default (S0), budget 60 s, 5 seeds, primal integral P(60); 80 MIPLIB benchmark instances and 33 PGLib-UC
instances with both; static features from optopt `datasets/features.parquet`. Bounded d = (P_SCIP − P_cuOpt) /
max(P_SCIP, P_cuOpt, 10⁻³), positive = cuOpt better. **Background only** — like `hardness_from_optopt.csv`, never a
selection criterion for test instances (D-002 / Datasets).

| group (MIPLIB) | n | cuOpt better / worse (\|d\| > 0.05) | mean d | median P cuOpt / SCIP | cuOpt no solution |
|---|---|---|---|---|---|
| all | 80 | 27 / 42 | −0.14 | 0.146 / 0.139 | 9 |
| continuous share ≥ 0.5 | 22 | 5 / **16** | **−0.35** | 0.135 / 0.085 | 1 |
| continuous share 0.05–0.5 | 20 | 5 / 12 | −0.22 | 0.196 / 0.131 | 1 |
| (almost) no continuous (< 0.05) | 38 | 17 / 14 | +0.03 | 0.153 / 0.213 | 7 |
| pure binary (≥ 0.99) | 28 | **14 / 10** | **+0.08** | 0.140 / 0.180 | 4 |
| general integers ≥ 0.05 | 16 | 4 / 10 | −0.13 | 0.127 / 0.105 | 2 |
| equality-heavy (≥ 0.5 of rows) | 20 | 6 / 8 | −0.04 | 0.161 / 0.491 | 4 |
| PGLib-UC (all) | 33 | 13 / 7 | +0.06 | 0.660 / 1.000 | 0 |

Rank correlations with d over the 80 MIPLIB instances: continuous share ρ = −0.26 (p = 0.02); log nnz ρ = +0.37
(p = 0.0008 — cuOpt is relatively worse on small instances); knapsack-row share ρ = +0.26 (p = 0.02); integer share,
equality share, coefficient range: none (|ρ| ≤ 0.07). Largest absolute cuOpt deficits (P_cuOpt − P_SCIP ≥ 0.2):
dano3_3 1.00 vs 0.21, neos-4722843-widden 1.00 vs 0.18, uccase12 0.99 vs 0.58, snp-02-004-104 0.93 vs 0.62,
supportcase42 0.57 vs 0.09, pk1 0.43 vs 0.16, neos-787933 0.42 vs 0.01, neos-2978193-inde 0.32 vs 0.04, neos17 0.29
vs 0.01, exp-1-500-5-5 0.29 vs 0.01. Caveat: bounded d exaggerates near-zero P (many "worse" cases are 0.01–0.04 vs
0.00); among cases with max P ≥ 0.1 the continuous-heavy group's mean absolute gap is +0.08 against cuOpt.

**Reading for I-TREA II.** cuOpt's weak spots — continuous-heavy / mixed models (fixed-charge network, lot sizing,
big-M on/off), small instances, unit commitment in absolute terms — match where the coupled system wins on the
development set (ca, sing44, item_placement, loue, neos-960392: 51–96 % lower P than cuOpt), while cuOpt's strengths —
pure binary and integer structure that bound propagation handles (fhnw, radiationm) — match our losses. A hypothesis
for the instance-class analysis (issue to follow), not a finding about I-TREA II.
