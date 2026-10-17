# D-006 Exchange platform — Redis, with the file mailbox as fallback (2026-10-05)

**Berk, 23:5x:** "for solution sharing one needs a common platform. perhaps redis is good for that?"

**Decision.** Yes — Redis is the common platform; the file mailbox stays as a dependency-free fallback with the same
API (`itrea.exchange.redis_mailbox.open_mailbox(spec, role)`; spec `redis://127.0.0.1:6390/0?run=<id>` or a directory).
The choice for confirmatory runs is fixed by the E0 measurement below, before E2.

**Why Redis fits the question.** A pair (SCIP ‖ EA) needs only two channels, but the study's natural extensions are
N-party: SCIP, HiGHS and cuOpt and one or more EA islands sharing one pool. Redis gives (1) a **shared solution pool**
(sorted set by objective + vectors), any participant reads the best k; (2) a **history of every exchange** (streams,
MAXLEN 64) — the exchange log comes for free; (3) clients in any language (a future C++/CUDA EA or solver plug-in);
(4) the same platform across machines later.

**Setup (no download).** Container `itrea-redis` from the `redis:7-alpine` image already on the host; loopback port
6390; `--save "" --appendonly no` (no persistence); `maxmemory 1536mb noeviction`; container cap 2 GB; pinned to
A725 core 0 (never a timed X925 core); `--restart no`. Client `redis-py` 8.1.0 installed offline from the uv cache
(hiredis has no cp312 wheel in the cache — the pure-Python parser is used). **Never** the host's other Redis instances
(other services' data) and never another service's remote Redis (read-only).

**Measured 2026-10-05 (loopback, pure-Python client):** round trip 52 µs (a file `stat` is ~1 µs); a dense
364k-variable float64 vector: XADD 2.2 ms, read 7.3 ms; sparse encoding (non-zeros only) shrinks a 1 %-dense
364k-binary solution from 2.9 MB to 44 KB; post + poll 3.3 ms. Idle container 36 MB.

**Consequences.**
- Polls cost ~50 µs, so the SCIP heuristic callback polls on a **timer** (`exchange.poll_s`), not at every node; it
  costs one clock read otherwise.
- **E0 measurement (pre-registered):** SCIP nodes/s and LP iterations/s on 6 tuning instances with (a) no exchange,
  (b) file mailbox idle, (c) Redis idle, (d) Redis with a partner posting at 1 Hz. If (c) or (d) is more than 2 % below
  (a), the confirmatory runs use the file mailbox. Every SCIP arm carries the same (inert where unused) plugin.
- The runner starts/stops the container (`itrea exchange up|down`) and deletes each run's keys at the end; keys also
  expire after a day.

Redacted 2026-10-17 for publication: names of other services/projects replaced by neutral wording; meaning unchanged.
