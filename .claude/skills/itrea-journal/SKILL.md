---
name: itrea-journal
description: >
  Keep the I-TREA II research record: the append-only daily journal (times in CEST), decision records D-NNN with a
  status and the superseding rule, and session checkpoints. Use whenever something is decided, measured, started,
  stopped or found wrong in this repository, when Berk gives an instruction or a decision, and before a session ends
  or is left idle.
---

# I-TREA II journal, decisions and checkpoints

The vault (`vault/`, Obsidian) is the record; this repository's chat history is not. Write as you go, not at the end.

## Journal — `vault/20-journal/YYYY-MM-DD.md`
- One file per calendar day, times in **CEST** (the host's local time, Europe/Berlin); a session past
  midnight continues in the next day's file (or marks `(10-06)` in a heading when it keeps writing the old one).
- **Append-only.** Never rewrite or delete an entry; a correction is a new entry that names the old one and says what
  was wrong.
- Heading per entry: `## HH:MM Title` (use `HH:Mx` when only the ten-minute is known). Content: what happened, the
  numbers with their units, the command or file that produced them, and links (`[[note]]`) to decisions and results.
- Quote Berk's instructions verbatim (with the time) when they set direction; they are the provenance of decisions.
- Name models and harnesses plainly ("Claude Code (Anthropic Claude Opus 5.5)", "vLLM serving Ornith-1.5-35B-A3B");
  no internal pseudonyms. The AI agent is AgentRA (Agent Research Assistant; mainly Claude Opus 5.5).
- Box events belong here too: lease claimed / released, units stopped / restored (with verification), incidents (and
  a line in `vault/60-incidents/Mistakes and incidents.md`).

## Decision records — `vault/50-decisions/D-NNN <title>.md`
- Next free number; never reuse or renumber. Title says the decision, not the topic.
- First lines: `# D-NNN <title> (date, who)` and `**Status:** …` — one of *draft*, *decided by <who>, open to a
  superseding decision on evidence*, *binding since <tag>*, *narrowed by D-MMM*, *superseded by D-MMM*.
- Body: the decision, why (evidence, measurements, Berk's words), consequences, and what it changes elsewhere.
- **Superseding rule:** a decision is changed only by a new, dated decision that says what changed and why; the old
  record stays, with its status line pointing at the new one. Berk's decisions are debatable the same way — on evidence,
  through a superseding draft — not by editing them.
- The pre-registration (D-002) is special: after the tag `prereg-v1` it is never edited; changes are new decisions,
  and results under the old plan are still reported. Before the tag, a version bump with a changelog line is allowed
  (`v2 → v3, date, what, why`).
- Link every new decision from the journal entry that made it and from the notes it affects.

## Checkpoints — `vault/90-checkpoints/YYYY-MM-DD-HHMM.md`
Write one at the end of a session, before handing over, and whenever a window opens or closes. Sections:
1. **Where things stand** — what exists, what ran, results (or "no results"), decisions taken.
2. **Box state** — lease, stopped units, restore timer, MemAvailable, anything left running; how it will be restored.
3. **Open items** — known gaps, with the file and the owner.
4. **Next steps (in order)** — concrete, with the gate each one needs.
5. **How to resume** — the commands.
The newest checkpoint is the entry point for the next session (`vault/00-index/Home.md` says so).

## Before you stop
Journal entry for the last change, checkpoint written, incidents logged, experiment index row current, window closed
or its restore verified (skill `itrea-window`).
