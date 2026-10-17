"""itrea.analysis.commit_map: old -> new commit hashes after the history rewrite of 2026-10-17, and its two users
(build_runs' identity columns, prereg's git check). Every map here is a file under tmp_path: the real
`COMMIT_MAP.tsv` at the repository root is never read or written by these tests."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from itrea.analysis import build_runs as B
from itrea.analysis import commit_map as CM
from itrea.analysis import prereg as PR

HEADER = "# History rewrite 2026-10-17: old -> new"
OLD1, NEW1 = "1" * 7 + "a" * 33, "f" * 40                     # 1111111aaa… -> fff…
OLD2, NEW2 = "1" * 7 + "b" * 33, "e" * 40                     # shares its first 7 characters with OLD1
OLD3, NEW3 = "2abcdef" + "0" * 33, "d" * 40


def write_map(path: Path, pairs, header: str = HEADER) -> Path:
    path.write_text("\n".join([header, *(f"{o}\t{n}" for o, n in pairs)]) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def cmap(tmp_path, monkeypatch):
    """A map with three commits, installed as the default map."""
    p = write_map(tmp_path / "COMMIT_MAP.tsv", [(OLD1, NEW1), (OLD2, NEW2), (OLD3, NEW3)])
    monkeypatch.setattr(CM, "PATH", p)
    return p


def test_map_present_full_hash(cmap):
    assert CM.translate(OLD1) == NEW1 and CM.translate(OLD3) == NEW3
    assert CM.translate(OLD3.upper()) == NEW3                     # hex case does not matter
    assert CM.load() == {OLD1: NEW1, OLD2: NEW2, OLD3: NEW3}
    assert CM.translate(OLD1, path=cmap) == NEW1                  # an explicit path reads the same file


def test_map_absent_is_identity(tmp_path, monkeypatch):
    monkeypatch.setattr(CM, "PATH", tmp_path / "no-such-map.tsv")
    for rev in (OLD1, OLD1[:7], "", "HEAD", "prereg-v1"):
        assert CM.translate(rev) == rev
    assert CM.load() == {}


def test_abbreviated_hash(cmap):
    assert CM.translate(OLD3[:7]) == NEW3                         # the git-default 7 characters
    assert CM.translate(OLD3[:12]) == NEW3
    assert CM.translate(OLD1[:8]) == NEW1                         # 8 characters tell OLD1 and OLD2 apart
    assert CM.translate(OLD1[:7]) == OLD1[:7]                     # ambiguous: unchanged, never a guess
    assert CM.translate(OLD3[:6]) == OLD3[:6]                     # shorter than 7: not treated as a hash


def test_unknown_and_non_hash_inputs_are_unchanged(cmap):
    unknown = "3" * 40
    for rev in (unknown, unknown[:9], NEW1, "", "HEAD", "prereg-v1", "rev5", "abc123", f" {OLD1}x", "g" * 40):
        assert CM.translate(rev) == rev
    assert CM.translate(None) is None


def test_comments_and_blank_lines_are_skipped(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text(f"{HEADER}\nold_sha\tnew_sha\n\n# another comment\n{OLD3}  {NEW3}\n\n", encoding="utf-8")
    assert CM.load(p) == {OLD3: NEW3}
    p.write_text(f"{'old':<40} new\n{OLD1} {NEW1}\n", encoding="utf-8")   # git-filter-repo's own commit-map header
    assert CM.load(p) == {OLD1: NEW1}


@pytest.mark.parametrize("line", ["deadbeef\tcafebabe", f"{OLD1}", f"{OLD1}\t{NEW1}\textra", f"{OLD1}\t{'z' * 40}",
                                  f"{OLD1}\t{'0' * 40}"])     # the last: filter-repo's mark for a pruned commit
def test_malformed_line_is_an_error(tmp_path, line):
    p = tmp_path / "m.tsv"
    p.write_text(f"{HEADER}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"m\.tsv:2"):
        CM.translate(OLD1, path=p)


def test_conflicting_entries_are_an_error(tmp_path):
    p = write_map(tmp_path / "m.tsv", [(OLD1, NEW1), (OLD1, NEW2)])
    with pytest.raises(ValueError, match="mapped twice"):
        CM.load(p)


def test_a_rewritten_map_is_reread(tmp_path):
    p = write_map(tmp_path / "m.tsv", [(OLD1, NEW1)])
    assert CM.translate(OLD1, path=p) == NEW1
    write_map(p, [(OLD1, NEW2), (OLD3, NEW3)])
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))   # a different mtime even on a coarse clock
    assert CM.translate(OLD1, path=p) == NEW2 and CM.translate(OLD3, path=p) == NEW3


def test_cli(cmap, capsys):
    assert CM.main([OLD3[:7], "HEAD"]) == 0
    assert capsys.readouterr().out.splitlines() == [f"{OLD3[:7]} -> {NEW3}", "HEAD -> HEAD"]
    assert CM.main([]) == 2


# ------------------------------------------------------------------------------------------- the two users
def _run(tmp_path, run_json) -> B.Run:
    return B.Run(path=tmp_path, set="s", run_id="x__A3__s1__10s__cfg", instance="x.mps", name="x", arm="A3", seed=1,
                 budget=10.0, config_hash="cfg", run_json=run_json)


def test_build_runs_identity_translates_git_rev_and_prereg_tag(cmap, tmp_path):
    rj = dict(env=dict(git=dict(revision=OLD1, prereg_tag=OLD3)))
    ident = B._identity(_run(tmp_path, rj))
    assert (ident["git_rev"], ident["prereg_tag"]) == (NEW1, NEW3)
    ident = B._identity(_run(tmp_path, dict(git_rev=OLD3[:7])))     # top-level key, abbreviated
    assert ident["git_rev"] == NEW3 and ident["prereg_tag"] == ""
    ident = B._identity(_run(tmp_path, dict(git_rev="abc123")))     # not in the map: unchanged
    assert ident["git_rev"] == "abc123"


def _git(cwd: Path, *args: str) -> str:
    env = dict(os.environ, GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_SYSTEM=os.devnull)
    r = subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@example.invalid", "-c", "commit.gpgsign=false",
                        *args], cwd=cwd, env=env, capture_output=True, text=True, check=True)
    return r.stdout.strip()


def test_prereg_git_check_translates_before_merge_base(tmp_path, monkeypatch):
    """A toy repository: tag t on commit A, commit B after it. An old hash mapped to B descends from t; without the
    map git does not know it (None: 'git or the tag unavailable')."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    for name in ("a", "b"):
        (repo / name).write_text(name)
        _git(repo, "add", name)
        _git(repo, "commit", "-q", "-m", name)
        if name == "a":
            _git(repo, "tag", "t")
    head = _git(repo, "rev-parse", "HEAD")
    old = "4" * 40
    monkeypatch.setattr(CM, "PATH", tmp_path / "absent.tsv")
    assert PR._git_descends(old, tag="t", cwd=repo) is None
    assert PR._git_descends(head, tag="t", cwd=repo) is True        # a new hash passes as before
    monkeypatch.setattr(CM, "PATH", write_map(tmp_path / "COMMIT_MAP.tsv", [(old, head)]))
    assert PR._git_descends(old, tag="t", cwd=repo) is True
    assert PR._git_descends(old[:7], tag="t", cwd=repo) is True
    assert PR._git_descends("", tag="t", cwd=repo) is False


# ------------------------------------------------------------- building the map from git-filter-repo's output
def _commit(repo: Path, name: str) -> str:
    (repo / name).write_text(name)
    _git(repo, "add", name)
    _git(repo, "commit", "-q", "-m", name)
    return _git(repo, "rev-parse", "HEAD")


def _new_repo(tmp_path) -> tuple[Path, dict[str, str]]:
    """The rewritten repository: n1 -> n2 -> n3, and n4 on n2 beside n3 (neither of n3 / n4 contains the other)."""
    repo = tmp_path / "new"
    repo.mkdir()
    _git(repo, "init", "-q")
    n = {k: _commit(repo, k) for k in ("n1", "n2", "n3")}
    n["n4"] = _git(repo, "commit-tree", f"{n['n2']}^{{tree}}", "-p", n["n2"], "-m", "n4")
    return repo, n


def _old(i: int) -> str:
    """A distinct old hash whose first 7 characters are distinct too: a00010000…, a00020000…"""
    return format(0xA0000 + i, "05x").ljust(40, "0")


# the old graph: o2, o4, o5 and o7 changed only removed paths; o8 merges o7 (journal only) with o6 (code)
OLD_PARENTS = {1: [], 2: [1], 3: [2], 4: [3], 5: [4], 6: [3], 7: [3], 8: [7, 6], 9: [3]}
KEPT = {1: "n1", 3: "n2", 6: "n3", 9: "n4"}


def _filter_repo_files(tmp_path, n, extra_parents=None, extra_pruned=()):
    parents = {**OLD_PARENTS, **(extra_parents or {})}
    fr = tmp_path / "commit-map"
    rows = [f"{_old(i)} {n[KEPT[i]] if i in KEPT else CM.ZERO}" for i in sorted(parents)]
    fr.write_text("\n".join([f"{'old':<40} new", *rows]) + "\n", encoding="utf-8")
    op = tmp_path / "old-parents.txt"
    op.write_text("\n".join(" ".join([_old(c), *map(_old, ps)]) for c, ps in sorted(parents.items()) if c not in
                            extra_pruned) + "\n", encoding="utf-8")
    return fr, op


def test_from_filter_repo_maps_pruned_commits_to_their_nearest_kept_ancestor(tmp_path):
    repo, n = _new_repo(tmp_path)
    fr, op = _filter_repo_files(tmp_path, n)
    m = CM.from_filter_repo(fr, op, repo=repo)
    assert m == {_old(1): n["n1"], _old(2): n["n1"], _old(3): n["n2"], _old(4): n["n2"], _old(5): n["n2"],
                 _old(6): n["n3"], _old(7): n["n2"], _old(9): n["n4"],
                 _old(8): n["n3"]}        # the merge: n3 contains n2; a first-parent walk would wrongly give n2
    out = CM.write(m, tmp_path / "COMMIT_MAP.tsv")
    assert out.read_text(encoding="utf-8").splitlines()[:2] == [CM.HEADER, CM.PRUNED_NOTE]
    assert CM.load(out) == m and CM.translate(_old(5), path=out) == n["n2"]
    with pytest.raises(ValueError, match="all-zero"):
        CM.load(fr)                       # filter-repo's raw map is never usable as COMMIT_MAP.tsv


def test_from_filter_repo_refuses_to_guess(tmp_path):
    repo, n = _new_repo(tmp_path)
    fr, op = _filter_repo_files(tmp_path, n, extra_parents={10: [6, 9]})     # pruned merge of n3 and n4
    with pytest.raises(ValueError, match="no single nearest kept ancestor"):
        CM.from_filter_repo(fr, op, repo=repo)
    fr, op = _filter_repo_files(tmp_path, n, extra_parents={10: [5]}, extra_pruned=(10,))   # no parent list
    with pytest.raises(ValueError, match="save the parent list before the rewrite"):
        CM.from_filter_repo(fr, op, repo=repo)
    fr, op = _filter_repo_files(tmp_path, n, extra_parents={10: []})        # a pruned root
    with pytest.raises(ValueError, match="pruned root"):
        CM.from_filter_repo(fr, op, repo=repo)


def test_from_filter_repo_cli(tmp_path, capsys):
    repo, n = _new_repo(tmp_path)
    fr, op = _filter_repo_files(tmp_path, n)
    out = tmp_path / "COMMIT_MAP.tsv"
    assert CM.main(["--from-filter-repo", str(fr), "--old-parents", str(op), "--out", str(out), "--repo", str(repo)]) == 0
    assert "9 commits (5 pruned" in capsys.readouterr().out
    assert CM.load(out)[_old(8)] == n["n3"]


# ------------------------------------------------------------------------------ the post-rewrite acceptance gate
def _record(traces: Path, rel: str, **rj) -> None:
    d = traces / rel
    d.mkdir(parents=True)
    (d / "run.json").write_text(json.dumps(rj), encoding="utf-8")


def test_check_traces(tmp_path, capsys):
    """A toy rewritten repository: a, then b tagged t, then c. Old hashes map onto them."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    a = _commit(repo, "a")
    b = _commit(repo, "b")
    _git(repo, "tag", "t")
    c = _commit(repo, "c")
    oa, ob, oc = _old(1), _old(2), _old(3)
    cmap = write_map(tmp_path / "COMMIT_MAP.tsv", [(oa, a), (ob, b), (oc, c)])
    traces = tmp_path / "raw"

    def gate():
        rc = CM.check_traces(traces, tag="t", repo=repo, path=cmap)
        return rc, capsys.readouterr().out

    assert gate()[0] == 2                                                    # no run records
    _record(traces, "s1/r1", env=dict(git=dict(revision=oc, prereg_tag=ob)))
    _record(traces, "s1/r2", git_rev=ob[:7], prereg_tag=ob)                 # top-level keys, abbreviated
    _record(traces, "s1/void/r3__x", env=dict(git=dict(revision="5" * 40)))  # void runs are not read
    rc, out = gate()
    assert rc == 0 and out.splitlines()[-1].endswith("all ok") and "5555555" not in out
    assert f"revision {oc} -> {c}: ok (1 runs)" in out and f"prereg_tag {ob} -> {b}: ok (2 runs)" in out
    _record(traces, "s2/r4", env=dict(git=dict(revision=oa, prereg_tag=oa)))  # before the tag
    _record(traces, "s2/r5", env=dict(git=dict(revision="6" * 40)))           # unknown, not a commit here
    _record(traces, "s2/r6", env=dict(git=dict()))                           # nothing recorded
    rc, out = gate()
    assert rc == 1 and out.splitlines()[-1].endswith("4 FAIL")
    assert f"revision {oa} -> {a}: FAIL does not descend from t" in out
    assert f"revision {'6' * 40} -> {'6' * 40}: FAIL not a commit in this repository" in out
    assert "revision - -> -: FAIL no revision recorded" in out
    assert f"prereg_tag {oa} -> {a}: FAIL not the commit t points to" in out
    assert CM.check_traces(traces, tag="no-such-tag", repo=repo, path=cmap) == 1
    assert CM.main(["--check-traces", "--traces", str(traces), "--tag", "t", "--repo", str(repo),
                    "--map", str(cmap)]) == 1
