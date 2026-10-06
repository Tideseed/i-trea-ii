# D-009 Honour the fan governor's rest breaks and the night rest (2026-10-06 08:0x)

**Status:** decided by AgentRA (Claude Code) after Berk asked "isn't our fan control script working for this
session?"; the night-rest question is open for Berk (issue #11).

## What happened on the screening night (2026-10-05/06)
The host's `fan-governor` (root service, since 2026-10-02; source in [other repository] `tools/bin/fan-governor`)
**worked**: fan to MAX at 02:52:25 when the screening started (82 °C), GPU clock cap at 02:52:46, back to AUTO at
07:00:49. Even at max fan, 8 X925 cores at 3.9 GHz held the box at 88–91 °C. After 30 min of continuous MAX it
**requested a rest at 03:22:29** (`~/.local/state/thermal/rest`) — Berk's rule of 2026-10-02: "allow the spark to recover
some … fans should not go max for prolonged periods". The I-TREA II runner did **not** read the rest flag, so it kept
launching: the fan ran at MAX for ~4 h 08 min without a break. The night rest (night-rest-start.timer, Berk 2026-10-02:
"take a break after 00:00 for the night") **stepped aside** because this project already held the box-window lease
from 23:46. Both were unknown to this session — they live in the host memory note on overheating, not in the repo's
D-003. A miss, recorded in [[Mistakes and incidents]].

## Decision
- **The runner honours the rest flag** (`gates.rest_due()`, checked in `pool._gate_wait` before every launch): while
  the flag exists no new run starts on any lane; in-flight runs finish; the box idles, the governor returns the fan to
  AUTO and clears the flag after the rest (10–30 min, escalating). No halt timeout on this wait — a rest always clears
  once the box is idle. Same contract as the host's `thermal-rest-wait`.
- **Expected cost:** ~30 min of work, then a 10–30 min rest — the screening and every long run set lose roughly a third
  of their wall clock. Accepted: the box's cooling is the binding constraint (2026-10-01 power-off).
- **Night windows:** no I-TREA II window runs through the night rest without Berk's explicit yes for that night (issue
  #11). Daytime windows (rest breaks honoured) are the default.
- D-003 gains the rest-flag row; the window skill gets a "check fan-governor / rest flag / night rest" step.
