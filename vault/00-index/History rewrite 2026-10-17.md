# History rewrite 2026-10-17

**On 2026-10-17 the repository's git history was rewritten, so every commit hash changed.** Hashes recorded before
then (in run records, derived tables, notes and issues) name commits that no longer exist under that hash.
`COMMIT_MAP.tsv` at the repository root translates each old hash to its new commit. Start page: [[Home]].

## Why

Berk's decisions of 2026-10-17, taken before the repository becomes public:

- **No third-party names in the public history.** Names of third parties (people outside the project, companies and
  their projects, other projects and services on the host), and with them the host's name, are removed from every
  past commit and every commit message, not only from the current files.
- **Operational notes go to a private repository.** Berk: "Journal must keep only the developments regarding the
  problem not procedural stuff." Session checkpoints, the incident log and the procedural journal entries moved,
  complete, to the private repository `Tideseed/i-trea-ii-ops`.
- **I-TREA (part I) is cited only through the open 2011 thesis:** https://open.metu.edu.tr/handle/11511/20509
  (hdl:11511/20509). Notes on unpublished part-I material were removed from every commit.

## What changed

- **Every commit hash.** A commit's hash covers its files and its parent's hash, so removing files and names from past
  commits gives each of them, and every commit after it, a new hash.
- **Commits that only touched removed paths are gone.** A commit that changed nothing but the journal, the
  checkpoints or the incident log is empty after the rewrite and was pruned. Its files are those of its nearest kept
  ancestor, so `COMMIT_MAP.tsv` maps it to that ancestor's new commit (rule below). A commit whose only change was a
  name the filter replaces is pruned the same way. This matters: of the 111 revisions recorded in the run records
  (1,479 records, void runs excluded), 30 revisions with 487 runs are commits that touched only removed paths.
- **The tag `prereg-v1`** points to the rewritten copy of the same commit (old `4cdc33d7`, a kept commit), with the
  same date (2026-10-06 02:51 CEST). The pre-registration's rule "a run's code descends from `prereg-v1`"
  ([[D-002 Pre-registration — hypotheses, arms, analysis|D-002]]) is checked against the rewritten tag.
- **Recorded hashes were not edited.** Run records (`traces/raw/*/*/run.json`: `env.git.revision`,
  `env.git.prereg_tag`), tables built before the rewrite (`datasets/runs.parquet`) and hashes quoted in notes and
  GitHub issues keep the old values. Translate them as below.

## How to resolve an old hash

1. **Look it up.** `COMMIT_MAP.tsv` has one line per rewritten commit, `<old sha><TAB><new sha>`, both full 40-hex,
   under the header `# History rewrite 2026-10-17: old -> new` and a comment line stating the pruned-commit rule. An
   abbreviated hash: `grep '^<abbrev>' COMMIT_MAP.tsv`.
2. **Or ask the code.** `python -m itrea.analysis.commit_map <hash> [<hash> …]` prints `<hash> -> <new hash>`; in
   Python, `from itrea.analysis.commit_map import translate`. It takes a full old hash or an abbreviation of at least 7
   hex characters (if no other old hash starts with it) and returns the new full hash. Anything else comes back
   unchanged: a new hash, an unknown hash, an ambiguous abbreviation, a branch or tag name. Without `COMMIT_MAP.tsv`
   (a checkout from before the rewrite) it changes nothing.
3. **The analysis translates on read.** `itrea.analysis.build_runs` translates `git_rev` and `prereg_tag` when it reads
   a run record, and `itrea.analysis.prereg` translates a revision before `git merge-base`. With a complete map (every
   recorded revision in it, pruned commits resolved), both a rebuild from the traces and a `runs.parquet` built before
   the rewrite pass the git check. The acceptance gate below is what shows the map is complete.

## How the map is made and checked (the rewrite's steps)

The rule for a pruned commit: **its new hash is the new hash of its nearest kept ancestor.** For an ordinary commit
that is the first kept commit down its parent chain. For a pruned merge it is the one resolved parent that no other
resolved parent contains; if there is no such single parent, building the map stops with an error rather than guess.
Keeping the empty commits instead (git-filter-repo `--prune-empty never`) would also give a complete map, but would
leave about 57 empty procedural commits in the public history.

1. **Before the rewrite**, in the old repository: `git rev-list --parents --all > old-parents.txt`. After the rewrite
   the old commits are gone, and with them the parent links the rule needs. Keep the file with the private
   operations repository.
2. **The rewrite** (git-filter-repo) drops the paths in the table below from every commit and applies the name filter
   to the files (`--replace-text`) and to the commit messages (`--replace-message`), both with the same expressions
   file. A commit whose only change was a name the filter replaces is empty afterwards and is pruned as well; the map
   resolves it like any other pruned commit. git-filter-repo writes `.git/filter-repo/commit-map`, with the column row
   `old new` and **forty zeros as the new hash of every commit it pruned**. Never copy that file to `COMMIT_MAP.tsv`
   as it is: the loader rejects an all-zero hash (`ValueError` with the line number), because forty zeros are no
   commit.
3. **Build the map:** `python -m itrea.analysis.commit_map --from-filter-repo .git/filter-repo/commit-map
   --old-parents old-parents.txt`, run in the rewritten repository. It resolves every pruned commit by the rule above
   and writes `COMMIT_MAP.tsv` with its header.
4. **Check `prereg-v1`.** git-filter-repo rewrites annotated tags itself. Check that
   `git rev-parse 'prereg-v1^{commit}'` equals `translate("4cdc33d7fe3b47f1cf4feb3506ef5ea785db59aa")` and that the
   tag keeps its date. Re-create the tag on that commit, with the old tag's date, only if it is missing or points
   elsewhere. If the gate in step 5 fails on `prereg_tag`, look here first.
5. **Acceptance gate:** `python -m itrea.analysis.commit_map --check-traces` must exit 0. Run it with the raw traces
   present (`traces/raw/`, unpacked from the archive): without run records it exits 2, which is not a pass. It reads
   every run record that build_runs reads (`traces/raw/**/run.json`, void runs excluded), translates each recorded
   revision and checks that it is a commit in the repository and descends from `prereg-v1`, and that each recorded
   `prereg_tag` translates to the commit the tag points to. Before the rewrite (no map, 2026-10-17) it reads 111
   revisions and 1 prereg tag from 1,479 run records, all ok. A read-only emulation of the path pruning on the old
   graph gave the same result; the name filter can prune more commits, so only this run on the real rewrite counts.
6. **Name check.** No filtered term may remain: a `git grep` for every term over `git rev-list --all`, and a `grep`
   over `git log --all --format=%B`, must both print nothing. The command lists the terms, so it is kept with the
   expressions file in the operations repository.
7. **The first commit after the rewrite** adds `COMMIT_MAP.tsv` and the files written for the rewrite and the
   publication, among them `src/itrea/analysis/commit_map.py`, `tests/test_commit_map.py`, this note, the research-only
   journals, the rebuilt `paper/main.pdf`, the website (`docs/`), `datasets/README.md` and `LICENSE-CC-BY-4.0`, which
   `README.md` and the website link to. Check with
   `git ls-files COMMIT_MAP.tsv paper/main.pdf docs datasets/README.md LICENSE-CC-BY-4.0`.
8. **Before the repository is made public.** GitHub keeps commits that a force-push removed reachable by hash, in
   the web view and the API, until it garbage-collects them, and this repository lists those hashes
   (`COMMIT_MAP.tsv`, the run records, notes). After the push, `gh api repos/Tideseed/i-trea-ii/commits/<any old
   hash from COMMIT_MAP.tsv>` must return 404. A 200 means the removed content is still served: push the rewritten
   history to a new repository instead (rename the old one and keep it private; move the issues with
   `gh issue transfer`), or have GitHub Support purge the cached views and unreachable commits; then check again.
   Only then change the visibility, and enable GitHub Pages from `main`, folder `/docs`: until then
   https://tideseed.github.io/i-trea-ii/ and its social-card image do not resolve.

## What was removed from the public history

| path | what it held | where it is now |
|---|---|---|
| `vault/20-journal/` | the daily journal, procedural entries included | replaced by research-only journals (the developments on the problem); the full originals are in `Tideseed/i-trea-ii-ops` |
| `vault/90-checkpoints/` | session checkpoints | `Tideseed/i-trea-ii-ops` |
| `vault/60-incidents/` | the incident log | `Tideseed/i-trea-ii-ops` |
| three notes in `vault/40-literature/` on unpublished part-I material | notes and an audit record | not public; I-TREA (part I) is cited only through the 2011 thesis above |
| `vault/50-decisions/D-004 Mapping I-TREA (2015) to I-TREA II.md` | D-004 as first written | dropped from every past commit; D-004 now lives at `vault/50-decisions/D-004 Mapping I-TREA (2011) to I-TREA II.md`, which maps the 2011 thesis and appears from the commit that added it |
| `paper/main.pdf` as first committed (2026-10-16) | the interim paper's PDF; its reference list carried a third-party name, and the text filter cannot edit a binary file | dropped from every past commit; the PDF rebuilt from the cleaned sources is added by the first commit after the rewrite |

The exact list of dropped paths is kept with the rewrite's scripts in the operations repository.

**Names.** A text filter (git-filter-repo `--replace-text`) replaces third-party names in every past file: people
outside the project, companies and their projects, other projects, agents and services on the host, a journal's
submission number, and the host's name. The same expressions filter every commit message (git-filter-repo
`--replace-message`). The terms themselves are not listed in this repository; the expressions file is kept in the
operations repository. Outside the dropped paths above, past versions of these files carried such a term:

- documentation and skills: `README.md`, `AGENTS.md`, `.claude/skills/itrea-literature/SKILL.md`,
  `.claude/skills/itrea-window/SKILL.md`;
- the paper and configuration: `paper/refs.bib` (the 2011 thesis entry's note field),
  `paper/sections/background.tex`, `configs/itrea1-faithful.yaml`;
- code comments and docstrings, log and message text: `src/itrea/config.py`, `src/itrea/ea/ranks.py`,
  `src/itrea/ea/population.py`, `src/itrea/exchange/redis_mailbox.py`, `src/itrea/experiments/exchange_ctl.py`,
  `src/itrea/experiments/gates.py`, `src/itrea/experiments/pool.py`, `tools/serving-window.sh`;
- resource records: the 15 files `traces/resources/pool-20261006-*.json` (the lease's `why` text and the `hostname`
  field, which reads `dgx-spark (redacted)`);
- vault notes in `vault/`: `00-index/Project background.md`, `00-index/Research question and spec.md`,
  `10-design/Datasets.md`, `10-design/Process interface.md`,
  `30-results/2026-10-11 D-021 measurements M1–M3 and the interim calibration.md`, `30-results/Development log.md`,
  `30-results/Experiment index.md`, `40-literature/prior-art/E Novelty check 2026-10-05.md`, and the decision records
  D-003, D-006, D-009, D-010 and D-021;
- reviews in `vault/80-reviews/`: `2026-10-05 design critique.md`,
  `2026-10-06 Feasibility targets consult (GPT-6.1 Sol).md`, `2026-10-06 LP stream design consult (GPT-6.1 Sol).md`,
  `2026-10-07 cuOpt restart design (workflow).md`.

The filter changes text only (comments, docstrings, notes, log and message strings, free-text record fields; a
configuration's hash covers its parsed values, not its comments), so a commit's code behaves as before. It skips
binary files; the only binary file that carried such a term, the first `paper/main.pdf`, is dropped instead (table
above).

**Links.** Notes in the current tree that refer to a checkpoint, the incident log or a procedural journal entry name
the private repository `Tideseed/i-trea-ii-ops`. Only files in past commits may still hold links into the removed
paths, and those do not resolve in the public repository.
