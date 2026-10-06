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
- Crashes and voided blocks per arm (cause: GPU health, thermal, memory gate), reruns per arm.
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
