"""Old commit hash -> new commit hash, after the history rewrite of 2026-10-17.

The repository's history was rewritten on 2026-10-17 (vault `00-index/History rewrite 2026-10-17.md`), so every
commit hash changed. Run records written before the rewrite (`run.json`: `git_rev`, `env.git.revision`,
`env.git.prereg_tag`) and tables built from them (`datasets/runs.parquet`) still name the OLD commits, which no longer
exist in the repository. `COMMIT_MAP.tsv` at the repository root lists every rewritten commit as one line
`<old sha> <tab> <new sha>` (full 40-hex hashes, header `# History rewrite 2026-10-17: old -> new`); `translate`
turns an old hash into the new one, so `git merge-base`, `git show` and readers can find the commit again.

    translate(<old 40-hex hash>)          -> the new 40-hex hash of that commit
    translate(<its first 7+ hex characters>) -> the same, if no other old hash starts with them
    translate(anything else)              -> unchanged: a new hash, an unknown hash, "", a branch or tag name

Without `COMMIT_MAP.tsv` (a checkout from before the rewrite) `translate` is the identity. A malformed data line is
an error (`ValueError`), not a silent no-op: a map that quietly fails to translate would make the pre-registration's
git check fail on hashes that are fine. So is a line whose new hash is all zeros: that is git-filter-repo's mark for a
commit it pruned, and the zeros are no commit.

**Pruned commits.** The rewrite drops whole folders (the journal, the checkpoints, the incident log), so a commit that
changed only those paths becomes empty and git-filter-repo prunes it; about a third of the run records name such a
commit. Its files equal those of its nearest kept ancestor, so `COMMIT_MAP.tsv` maps it to that ancestor's new hash.
`from_filter_repo` builds the map that way from filter-repo's own `commit-map` and the OLD repository's parent list
(`git rev-list --parents --all`, saved BEFORE the rewrite: afterwards the old commits are gone). For a pruned merge
the nearest kept ancestor is the one resolved parent that is not an ancestor of another; if there is none, it is an
error, never a guess.

    python -m itrea.analysis.commit_map <rev> [<rev> …]      prints `<rev> -> <translated>` per argument
    python -m itrea.analysis.commit_map --from-filter-repo .git/filter-repo/commit-map --old-parents old-parents.txt
        [--out COMMIT_MAP.tsv] [--repo .]                    writes the complete map (pruned commits resolved)
    python -m itrea.analysis.commit_map --check-traces [--traces traces/raw] [--tag prereg-v1] [--map COMMIT_MAP.tsv]
        the post-rewrite acceptance gate: every revision recorded in a run record (void runs excluded, as in
        build_runs) translates to a commit that exists and descends from the tag, and every recorded prereg tag
        translates to the tag's commit; exit 0 = all pass, 1 = a failure, 2 = no run records
"""
from __future__ import annotations

import argparse
import bisect
import json
import os
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

from itrea.paths import ROOT

PATH = ROOT / "COMMIT_MAP.tsv"
HEADER = "# History rewrite 2026-10-17: old -> new"
PRUNED_NOTE = ("# A commit the rewrite pruned (it changed only removed paths) maps to its nearest kept ancestor, "
               "which holds the same files.")
TAG = "prereg-v1"
ZERO = "0" * 40                                  # git-filter-repo's new hash for a pruned commit
MIN_ABBREV = 7
_FULL = re.compile(r"[0-9a-f]{40}")
_REV = re.compile(r"[0-9a-fA-F]{%d,40}" % MIN_ABBREV)
_COLUMN_ROWS = (["old_sha", "new_sha"], ["old", "new"])   # ours, if a writer adds one; git-filter-repo's commit-map
_cache: dict[Path, tuple[tuple[int, int], dict[str, str], list[str]]] = {}


def load(path: Path | str | None = None) -> dict[str, str]:
    """The map {old sha: new sha} from `path` (default `COMMIT_MAP.tsv` at the repository root); {} if the file does
    not exist. Blank lines, lines starting with '#' and a column-name row (`old_sha new_sha`, or git-filter-repo's
    `old new`) are skipped."""
    return dict(_load(Path(path) if path is not None else PATH)[0])


def _pairs(path: Path, allow_zero: bool) -> dict[str, str]:
    """{old: new} from a two-column file of full hashes; `ValueError` with the line number on a malformed line, a
    commit mapped twice, or (unless `allow_zero`) an all-zero new hash."""
    m: dict[str, str] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        parts = s.lower().split()
        if parts in _COLUMN_ROWS:                  # a column-name row
            continue
        if len(parts) != 2 or not all(_FULL.fullmatch(p) for p in parts):
            raise ValueError(f"{path}:{n}: expected '<old 40-hex sha> <tab> <new 40-hex sha>', got {line!r}")
        old, new = parts
        if new == ZERO and not allow_zero:
            raise ValueError(f"{path}:{n}: {old} maps to the all-zero hash, git-filter-repo's mark for a pruned "
                             "commit, which is no commit; map it to its nearest kept ancestor instead "
                             "(python -m itrea.analysis.commit_map --from-filter-repo)")
        if m.get(old, new) != new:
            raise ValueError(f"{path}:{n}: {old} is mapped twice ({m[old]} and {new})")
        m[old] = new
    return m


def _load(path: Path) -> tuple[dict[str, str], list[str]]:
    """(map, sorted old hashes), cached per file and invalidated when its mtime or size changes."""
    try:
        st = path.stat()
    except FileNotFoundError:
        return {}, []
    key = (st.st_mtime_ns, st.st_size)
    hit = _cache.get(path)
    if hit is not None and hit[0] == key:
        return hit[1], hit[2]
    m = _pairs(path, allow_zero=False)
    olds = sorted(m)
    _cache[path] = (key, m, olds)
    return m, olds


def translate(rev, path: Path | str | None = None):
    """The new hash for an old full or abbreviated (>= 7 hex) commit hash; `rev` unchanged otherwise (not a hash, not
    in the map, an ambiguous abbreviation, or no map file)."""
    if not isinstance(rev, str):
        return rev
    s = rev.strip()
    if not _REV.fullmatch(s):
        return rev
    m, olds = _load(Path(path) if path is not None else PATH)
    if not m:
        return rev
    s = s.lower()
    if len(s) == 40:
        return m.get(s, rev)
    i = bisect.bisect_left(olds, s)
    hits = []
    while i < len(olds) and olds[i].startswith(s) and len(hits) < 2:
        hits.append(olds[i])
        i += 1
    return m[hits[0]] if len(hits) == 1 else rev


# ------------------------------------------------------------------------------- building the map after a rewrite
def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, timeout=60)


def _is_ancestor(a: str, b: str, repo: Path) -> bool:
    """True iff commit `a` is an ancestor of (or equal to) commit `b` in `repo`; `ValueError` if git cannot tell."""
    r = _git(repo, "merge-base", "--is-ancestor", a, b)
    if r.returncode in (0, 1):
        return r.returncode == 0
    raise ValueError(f"git merge-base --is-ancestor {a} {b} failed in {repo}: {r.stderr.strip()}")


def _old_parents(path: Path) -> dict[str, list[str]]:
    """{commit: [parents]} from `git rev-list --parents --all` output."""
    out: dict[str, list[str]] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        parts = line.strip().lower().split()
        if not parts:
            continue
        if not all(_FULL.fullmatch(p) for p in parts):
            raise ValueError(f"{path}:{n}: expected '<commit> [<parent> …]' (git rev-list --parents --all), got {line!r}")
        out[parts[0]] = parts[1:]
    return out


def from_filter_repo(commit_map: Path | str, old_parents: Path | str, repo: Path | str = ROOT) -> dict[str, str]:
    """The complete map {old: new}: git-filter-repo's `commit-map` with every pruned commit (new hash all zeros)
    resolved to the new hash of its nearest kept ancestor. `old_parents` is the OLD repository's
    `git rev-list --parents --all`; `repo` is the rewritten repository, used only for ancestry between new commits."""
    commit_map, old_parents, repo = Path(commit_map), Path(old_parents), Path(repo)
    fr = _pairs(commit_map, allow_zero=True)
    parents = _old_parents(old_parents)
    memo = {o: n for o, n in fr.items() if n != ZERO}
    for start in sorted(o for o, n in fr.items() if n == ZERO):
        stack = [start]
        while stack:                               # depth-first, without recursion: chains of pruned commits are long
            c = stack[-1]
            if c in memo:
                stack.pop()
                continue
            if c not in parents:
                raise ValueError(f"{c} was pruned but is not in {old_parents}; save the parent list before the rewrite")
            ps = parents[c]
            if not ps:
                raise ValueError(f"{c} is a pruned root commit: it has no kept ancestor")
            todo = [p for p in ps if p not in memo]
            for p in todo:
                if p not in fr:
                    raise ValueError(f"{p}, a parent of the pruned commit {c}, is not in {commit_map}")
            if todo:
                stack.extend(todo)
                continue
            stack.pop()
            cands = list(dict.fromkeys(memo[p] for p in ps))
            if len(cands) > 1:                     # a pruned merge: keep the parent no other parent contains
                cands = [a for a in cands if not any(b != a and _is_ancestor(a, b, repo) for b in cands)]
            if len(cands) != 1:
                raise ValueError(f"{c} was pruned, but its parents map to {sorted(cands)} and none of them contains "
                                 "the others: no single nearest kept ancestor")
            memo[c] = cands[0]
    return {o: memo[o] for o in fr}


def write(m: dict[str, str], path: Path | str = PATH) -> Path:
    """Write `m` as COMMIT_MAP.tsv (header, the pruned-commit rule, one sorted line per commit), atomically."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("\n".join([HEADER, PRUNED_NOTE, *(f"{o}\t{m[o]}" for o in sorted(m))]) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


# -------------------------------------------------------------------------------- the post-rewrite acceptance gate
def _recorded(traces: Path) -> tuple[Counter, Counter, int]:
    """(revision -> runs, prereg tag -> runs, run records) over the run records build_runs reads (void runs out)."""
    revs: Counter = Counter()
    tags: Counter = Counter()
    n = 0
    for f in sorted(traces.rglob("run.json")):
        if "void" in f.parent.relative_to(traces).parts[:-1]:
            continue
        rj = json.loads(f.read_text(encoding="utf-8"))
        git = ((rj.get("env") or {}).get("git") or {}) if isinstance(rj.get("env"), dict) else {}
        revs[str(rj.get("git_rev") or rj.get("git_revision") or git.get("revision") or "")] += 1
        tag = str(rj.get("prereg_tag") or git.get("prereg_tag") or "")
        if tag:
            tags[tag] += 1
        n += 1
    return revs, tags, n


def check_traces(traces: Path | str = ROOT / "traces" / "raw", tag: str = TAG, repo: Path | str = ROOT,
                 path: Path | str | None = None, out=None) -> int:
    """Print one verdict per distinct recorded revision and prereg tag; 0 if all pass, 1 on any failure, 2 if there
    are no run records."""
    out = out or sys.stdout
    traces, repo = Path(traces), Path(repo)
    revs, tags, n = _recorded(traces) if traces.is_dir() else (Counter(), Counter(), 0)
    if not n:
        print(f"no run records under {traces}", file=out)
        return 2
    r = _git(repo, "rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    if r.returncode != 0:
        print(f"FAIL: the tag {tag} is not in {repo}", file=out)
        return 1
    tag_commit = r.stdout.strip()
    print(f"{tag} = {tag_commit}", file=out)
    bad = 0
    for rev, k in sorted(revs.items()):
        new = translate(rev, path)
        if not rev:
            why = "no revision recorded"
        elif _git(repo, "cat-file", "-e", f"{new}^{{commit}}").returncode != 0:
            why = "not a commit in this repository"
        elif _git(repo, "merge-base", "--is-ancestor", tag_commit, new).returncode != 0:
            why = f"does not descend from {tag}"
        else:
            why = ""
        bad += bool(why)
        print(f"revision {rev or '-'} -> {new or '-'}: {'FAIL ' + why if why else 'ok'} ({k} runs)", file=out)
    for t, k in sorted(tags.items()):
        new = translate(t, path)
        r = _git(repo, "rev-parse", "--verify", "--quiet", f"{new}^{{commit}}")
        ok = r.returncode == 0 and r.stdout.strip() == tag_commit
        bad += not ok
        print(f"prereg_tag {t} -> {new}: {'ok' if ok else 'FAIL not the commit ' + tag + ' points to'} ({k} runs)",
              file=out)
    print(f"{len(revs)} revisions and {len(tags)} prereg tags from {n} run records: "
          f"{'all ok' if not bad else f'{bad} FAIL'}", file=out)
    return 1 if bad else 0


def _tool(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="python -m itrea.analysis.commit_map")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--from-filter-repo", metavar="COMMIT_MAP", help="git-filter-repo's .git/filter-repo/commit-map")
    mode.add_argument("--check-traces", action="store_true", help="the post-rewrite acceptance gate")
    ap.add_argument("--old-parents", metavar="FILE", help="the OLD repository's `git rev-list --parents --all`")
    ap.add_argument("--out", default=str(PATH), help="where to write the map (default: COMMIT_MAP.tsv at the root)")
    ap.add_argument("--repo", default=str(ROOT), help="the rewritten repository (default: this one)")
    ap.add_argument("--traces", default=str(ROOT / "traces" / "raw"))
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--map", default=None, help="the map to translate with (default: COMMIT_MAP.tsv at the root)")
    a = ap.parse_args(argv)
    if a.check_traces:
        return check_traces(a.traces, tag=a.tag, repo=a.repo, path=a.map)
    if not a.old_parents:
        ap.error("--from-filter-repo needs --old-parents")
    m = from_filter_repo(a.from_filter_repo, a.old_parents, repo=a.repo)
    pruned = sum(1 for n in _pairs(Path(a.from_filter_repo), allow_zero=True).values() if n == ZERO)
    print(f"{len(m)} commits ({pruned} pruned, mapped to their nearest kept ancestor) -> {write(m, a.out)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if args and args[0].startswith("--"):
        return _tool(args)
    if not args:
        print("usage: python -m itrea.analysis.commit_map <rev> [<rev> ...]\n"
              "       python -m itrea.analysis.commit_map --from-filter-repo COMMIT_MAP --old-parents FILE [--out PATH]\n"
              "       python -m itrea.analysis.commit_map --check-traces [--traces DIR] [--tag TAG] [--map PATH]",
              file=sys.stderr)
        return 2
    for a in args:
        print(f"{a} -> {translate(a)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
