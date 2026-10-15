# D-028 Does D-022's harness (R13 + R14) stop phase Y from restoring cuOpt on chromaticindex — a 4-run diagnostic (2026-10-15, AgentRA)

**Status:** decided for development by AgentRA (Claude Code, Anthropic Claude Opus 5.5) before any run; open to a superseding
decision on evidence. A diagnostic (D-024 rule 1: where the claim lives; rule 4: only what the rule reads): no hint, no adoption.

## Why
- D-027's rule 4 (Result 2026-10-15 09:2x): on chromaticindex1024-7 the EA's presence delays cuOpt's first incumbent (62–66 s inside
  A9 on every run on record; 6.5–9.8 s alone or beside SCIP). Berk (09:3x, question panel): CUDA MPS is **"Not for v1"** — so the
  only v1 route left is the EA's own pacing.
- That route worked once and failed once. With the same phase-Y values (duty ½ until cuOpt's first post or 30 s):
  **old harness** (D-018 stage 1, A9@bo1): the EA was busy 4.90 / 6.24 s of a phase Y that ended when cuOpt posted at 9.54 / 12.47 s;
  **ext** (D-023, A9@boY): the EA was busy 15.21 / 15.21 s of 30.0 s (it did pause ~14.8 s), and cuOpt posted only at 63.47 / 63.41 s.
  At full duty both harnesses give 65 s. So something in ext makes cuOpt slower while it shares the GPU, although ext alone is
  fine (A6@r3 3-thread lane under ext 9.24 / 8.21 s).
- ext differs from the old harness in R13 (cuOpt's MIP gaps 0), R14 (cuOpt's thread variables unset), RMM 24 GB separate from the
  EA's pool, host 24 GB and the project CUDA cache. D-022 adopted R13 + R14 for A9 on 3 instances where cuOpt can stop early, under
  the stated assumption that the gaps act only as a termination test; the C1 freeze carries ext. `r13-off` (`ITX_R13_CONTROL` =
  ext with R13 and R14 off, the rest of ext on) isolates R13 + R14 from the rest.

## Runs — fixed before any run
`jobs/d028.jsonl` (set `d028`, sha256 `ccd274d3944f029418cc2d40ae57560772769a733a9c17fb05e7db60819f2133`): chromaticindex1024-7 ×
seeds 0–1 × {A9@boY under `r13-off` (run hash 36a2c7e5c4e8), A6F under ext (the same-window reference)}, 120 s = **4 runs in 2
blocks** (written by `itrea make-jobs backoff2 --harness r13-off --instances chromaticindex1024-7`, the bo1s rows dropped — no rule
reads them — and the set renamed `d028`). The ext reading of boY is D-023's (`backoff2`, 10-14: 63.47 / 63.41 s); at full duty both
harnesses are known (65 s), so neither is re-run.

## Reading rule (fixed before the runs)
t = cuOpt's first-incumbent time (`cuopt_t_first`; none = 120 s); L = 2 × max(t of A6F in `d028`, seeds 0–1).
- **A9@boY under r13-off: t ≤ L on both seeds → R13 + R14 decide it:** under ext, R13 or R14 stops phase Y from restoring cuOpt on
  chromaticindex. Consequence: a question to Berk before the C1 freeze (D-022's adoption carries ext into frozen-v2; its assumption
  was "gaps act only as a termination test"), with these numbers; a follow-up separating R13 from R14 is written then, not now.
- **t ≥ 50 s on both seeds → not R13 + R14:** the rest of ext (RMM 24 GB separate pool, host 24 GB, CUDA cache) or a day effect.
  Stated; nothing changes for v1 (chromaticindex stays a known loss to cuOpt alone; MPS is v1.1).
- **Otherwise:** inconclusive at n = 2, stated.
Failed or void runs: a missing cell leaves the rule unread (the letter, as D-027).

## Cost
4 runs ≈ 10 min, queued in window W-1015 after D-025's fallback (b); no code.
