---
name: itrea-result-note
description: >
  Write the vault/30-results note for an I-TREA II experiment (screening, E0–E6, or any ad-hoc measurement): the
  question, the exact command, the configuration hash, the numbers with intervals, checker coverage, caveats and every
  failure, linked to the derived tables. Use as soon as an experiment's analysis has run — including experiments that
  failed, were aborted or gave a null result — and before quoting any of its numbers anywhere else.
---

# I-TREA II result note

One note per experiment: `vault/30-results/<id> <short title>.md` (e.g. `E1 EA alone vs cuOpt alone.md`), plus its row
in `vault/30-results/Experiment index.md`. A number that is not in a result note does not go into the README,
AGENTS.md, the paper or a message to Berk.

**Development tests (D-010 §3)** are recorded differently: one section in `vault/30-results/Development log.md` (title
with times, arms × instances × seeds and the verdict; pool id, commit, runs completed / expected, void, runs with
`compiled_on_clock`, both LLM servers' state; the numbers; the report that produced them) plus a "Result" section in
the test's decision record with the verdict by its own rules (e.g. *adopted*, *not adopted*, *inconclusive*,
*proceed*) and any stated deviation from the letter of a rule. The numbers come from the test's report, which applies
the rules fixed in the decision record (`tools/d016_report.py`, `tools/d017_report.py`, `itrea d025-report` / `itrea
d026-report` / `itrea d027-report` (the C1 diagnosis's R1 / R2 / R3), `itrea backoff-report --stage 0|1|1y`, `itrea
d022-report [--followup]`, `itrea mipfeas-report --arm A6F`; commit it before the pool ends, as for D-017, D-018 and
D-022), run single-threaded after `itrea build-runs`.
Development results are descriptive; they are never quoted as confirmatory.

**Interim readings** (a report run on part of a run set, e.g. D-021's M4 calibration stopped at 19 of 30 on
2026-10-13: R = 1.390 [1.081, 1.837] on the call clock, late-clock check on 6 of the 10 strata-0–1 instances) carry
**INTERIM** and **INCOMPLETE (n of N)** in the title, the verdict line and wherever they are quoted. An interim reading
may report a check's outcome on what has run, labelled with its n (e.g. "the late-clock check passes on 6 of the 10
strata-0–1 instances run", as D-024 rule 0 records it); it gives no final verdict, no adoption, no footprint or
"inside / outside the band" reading of the final rule. Say which runs are missing and when they run; the verdict comes
from the complete set by the record's own rule. An interim look that a record registers in advance (a
staged test's hint) is reported as that look, not as an interim reading of the final rule.

**Staged results (D-024).** A stage-1 note reports the hint exactly as it was written before the runs (threshold,
fixed instance set, how failures and skipped blocks count), its outcome — *hint present*, *no hint* or *incomplete* —
and what it gated (which stage runs next; v1's guard-only stage runs either way). A hint is futility-only: never
write a hint as a result ("A9 is better"), only as the gate it is. A note on a later stage reports the decisive
reading over all instances, the stage-1 instances included, and beside it the **stage-2-only figure** as the selection
check: conditional on a continuation the pooled estimate is slightly optimistic (by 1.2–2.5 points of the ratio in
D-024's simulation; "better shown" ≈ 5 % instead of ≈ 3 % under no effect) — state that bias in the Caveats. For v1,
every decisive aggregate is also given without R33's exposed instances (`itrea itx-report` prints both).

## Template
```markdown
# <id> <title> (<date>, window <W…>)

**Question.** One sentence; the pre-registered hypothesis or gate it answers (D-002 §…), or "exploratory".
**Verdict.** supported / negative / arm worse / inconclusive / gate passed / gate failed / aborted — by D-002's rule,
not by impression. One sentence with the main number and its interval.

## What ran
- Command(s), verbatim: `python -m itrea.experiments… jobs/<file>.jsonl …`
- Job file and its SHA-256; instances (set, count, tuning or test side); arms; seeds; budget.
- Configuration: file, **config hash**, git revision, `prereg-v1` / `pool-v1` / `frozen-e2` tags in HEAD's ancestry.
- Window: lease, LLM servers' state, cores and caps, MemAvailable minimum, ACPI maximum (from `run.json`).

## Numbers
| comparison | n instances | mean d_i [95 % CI] | Wilcoxon p | wins / losses / ties | ratio of means |
Time-to-target tables and per-stratum rows as pre-registered. Every number with its unit and n; intervals, not only
point estimates; "not significant" is never written as "equal" — give the interval's bounds.

## Integrity
- Checker coverage: incumbents stored / checked / passed / failed, per arm (failed ones listed by run).
- Crashes and voided blocks per arm (cause: GPU health, thermal, memory gate, kernel compiled on the clock), reruns per
  arm; runs whose summary says `compiled_on_clock = true`.
- Post accounting for exchange arms: posted / superseded-untried / tried-accepted / tried-rejected / pending-at-T.
- Runs whose recorded backend did not match the arm (dropped).

## Caveats
What limits the reading (sample, single machine, budgets, deviations from the plan with the decision that allowed them).

## Failures
Everything that went wrong, including runs outside the analysis; link incident entries.

## Files
Derived tables in `datasets/…` (committed), the analysis module that wrote them, figures; raw traces under
`traces/raw/<set>/` (not committed).
```

## Rules
- Report failures and null results with the same care as positive ones; a ✗ row stays in the index.
- Do not round in a way that changes a verdict; keep three significant figures for d and P.
- If a result changes after a correction, add a dated correction block at the top; never overwrite the old numbers
  silently (optopt's PDI correction is the model).
- Exploratory comparisons are labelled exploratory in the note, the index and anywhere they are quoted.
- Interim and staged readings keep their labels (INTERIM / INCOMPLETE (n of N); hint present / no hint / incomplete)
  in the index row and anywhere they are quoted, including messages to Berk; a note is never written for runs that
  have not been read by their record's report.
