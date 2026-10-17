---
name: itrea-journal
description: >
  Keep the I-TREA II research record: the append-only daily research journal (host time: CEST, CET from 2026-10-25)
  and decision records D-NNN with a status and the superseding rule; and send operational notes (session checkpoints,
  the incident log, box windows, leases, server stops, coordination) to the private repository Tideseed/i-trea-ii-ops.
  Use whenever something is decided, measured, started, stopped or found wrong in this repository, when Berk gives an
  instruction or a decision, and before a session ends or is left idle.
---

# I-TREA II journal, decisions and operational notes

The vault (`vault/`, Obsidian) is the record; this repository's chat history is not. Write as you go, not at the end.
This repository is public: its vault holds the research record, and operational notes are kept in a private one.

## What goes where

Berk, 2026-10-17: **"Journal must keep only the developments regarding the problem not procedural stuff."**

| note | where |
|---|---|
| research developments: what was designed, built, measured, found or decided about the method and the problem; Berk's directing words; a defect found in a method, a measurement or a reading | public journal `vault/20-journal/` (below) |
| decisions | public, `vault/50-decisions/` (below) |
| results, failures included (voided, crashed, halted or aborted runs and why) | public, `vault/30-results/` and the experiment index (skill `itrea-result-note`) |
| procedural and operational notes: box windows, leases, server stops and restores (with verification), timers and chains, coordination with other sessions, session checkpoints, incidents and lessons | **private** repository `Tideseed/i-trea-ii-ops` (local clone `~/git_repositories/i-trea-ii-ops`; if it is missing, `gh repo clone Tideseed/i-trea-ii-ops ~/git_repositories/i-trea-ii-ops`) |

- **The test:** would a reader of the paper need the note to understand or check the research? Then it is public. If
  it only says how the box, the session or the servers were run, it is private.
- **A run's validity facts stay public** even when their cause is operational: a block voided, a pool halted by a
  breaker, a crash, a run outside its window. Say what happened to the runs in the result note; the logistics behind it
  go to the operations repository.
- **Never link into the operations repository from a public note** (readers cannot open it). Write "(operational note,
  private operations repository)" instead. Secrets go in neither repository.
- **A lesson that becomes a rule** goes into `AGENTS.md` (public) as the rule, citing the incident by its date only.

## Journal — `vault/20-journal/YYYY-MM-DD.md`
- One file per calendar day, times in **CEST** (the host's local time, Europe/Berlin; **CET from Sun 2026-10-25**,
  when the host's clock changes — write the zone in the file's title, so the v1 entries in November say CET); a session
  past midnight continues in the next day's file (or marks `(10-06)` in a heading when it keeps writing the old one).
- **Append-only.** Never rewrite or delete an entry; a correction is a new entry that names the old one and says what
  was wrong. (One exception, Berk's decision of 2026-10-17: the files up to 2026-10-16 were cut to their research
  content, and each says so in its first line; the complete originals are in the operations repository.)
- Heading per entry: `## HH:MM Title` (use `HH:Mx` when only the ten-minute is known). Content: what happened, the
  numbers with their units, the command or file that produced them, and links (`[[note]]`) to decisions and results.
- Quote Berk's instructions verbatim (with the time) when they set direction; they are the provenance of decisions.
- Name models and harnesses plainly ("Claude Code (Anthropic Claude Opus 5.5)", "llama.cpp serving Qwen3.8-27B");
  no internal pseudonyms. The AI agent is AgentRA (Agent Research Assistant; mainly Claude Opus 5.5).
- **Research developments only.** Box events do not belong here: lease claimed / released, units stopped / restored
  and their verification, chains and timers, messages with other sessions and incidents go to the operations
  repository ("What goes where"). Keep the time and the research content of an entry; leave out the logistics.

## Decision records — `vault/50-decisions/D-NNN <title>.md`
- Next free number; never reuse or renumber. Title says the decision, not the topic.
- First lines: `# D-NNN <title> (date, who)` and `**Status:** …` — one of *draft*, *decided by <who>, open to a
  superseding decision on evidence*, *binding since <tag>*, *narrowed by D-MMM*, *superseded by D-MMM*; a development
  test (D-010) ends as *adopted* or *not adopted* (e.g. D-016, D-017), with its Result section.
- Body: the decision, why (evidence, measurements, Berk's words), consequences, and what it changes elsewhere. A
  development test fixes its rules before any run; a change before the first run is a dated amendment, a departure
  from a rule found after the runs is a stated deviation in the Result (D-017 M1, D-018's sanity rule). Link the
  GitHub issue it belongs to.
- **Superseding rule:** a decision is changed only by a new, dated decision that says what changed and why; the old
  record stays, with its status line pointing at the new one. Berk's decisions are debatable the same way — on evidence,
  through a superseding draft — not by editing them.
- The pre-registration (D-002) is special: after the tag `prereg-v1` it is never edited; changes are new decisions,
  and results under the old plan are still reported. Before the tag, a version bump with a changelog line is allowed
  (`v2 → v3, date, what, why`).
- Link every new decision from the journal entry that made it and from the notes it affects.

## Session checkpoints and the incident log — private operations repository
Both live in `Tideseed/i-trea-ii-ops`, not in this repository (moved there on 2026-10-17 from `vault/90-checkpoints/`
and `vault/60-incidents/`, with the procedural journal entries).

**Checkpoint:** write one at the end of a session, before handing over, and whenever a window opens or closes. Sections:
1. **Where things stand** — what exists, what ran, results (or "no results"), decisions taken.
2. **Box state** — lease, stopped units, restore timer, MemAvailable, anything left running; how it will be restored.
3. **Open items** — known gaps, with the file and the owner.
4. **Next steps (in order)** — concrete, with the gate each one needs.
5. **How to resume** — the commands.
The newest checkpoint is the entry point for the next session, read right after `AGENTS.md` "At a glance".

**Incident log:** one row per mistake or incident — when, what happened, the damage, who caught it and how, the lesson.
A run it affected is also stated in the public result note and the experiment index (validity facts stay public).

## Before you stop
- **Public (this repository):** a journal entry for the last research development, decision records current, the
  Development log section (development tests, D-010) or the experiment index row current.
- **Private (`Tideseed/i-trea-ii-ops`):** checkpoint written, incidents logged, the window closed or its restore
  verified (skill `itrea-window`).
