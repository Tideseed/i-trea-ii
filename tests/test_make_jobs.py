"""Job files: deterministic output, the D-002 phase rules (screening stages, confirmatory blocks, E4 caps) and the
candidate list built from a fake data directory (no real instance is read)."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from itrea.experiments import make_jobs as mj
from itrea.experiments.pool import make_blocks


def _fake_data(root):
    inst = root / "miplib" / "inst"
    inst.mkdir(parents=True)
    for n in ("air05", "neos-123-abc", "neos859080", "supportcase22", "bnatt500", "rail01"):
        (inst / f"{n}.mps.gz").write_bytes(b"")
    (root / "miplib" / "solu.txt").write_text("=opt=  air05  26374\n=inf=  neos859080\n=inf=  bnatt500\n"
                                              "=inf=  not_on_disk\n=best= rail01 174.2\n")
    mps = root / "pglib_uc" / "mps"
    mps.mkdir(parents=True)
    for n in ("ca__2014-09-01_reserves_0", "ferc__2015-01-01_lw", "rts_gmlc__2020-01-27"):
        (mps / f"{n}.mps.gz").write_bytes(b"")
    ip = root / "ml4co" / "instances" / "1_item_placement"
    (ip / "valid").mkdir(parents=True)
    (ip / "train").mkdir(parents=True)
    for i in (9900, 9924, 9925):
        (ip / "valid" / f"item_placement_{i}.mps.gz").write_bytes(b"")
    for i in (0, 34, 35):
        (ip / "train" / f"item_placement_{i}.mps.gz").write_bytes(b"")
    return root


def test_candidates_follow_d002(tmp_path):
    c = mj.candidates(_fake_data(tmp_path / "data"))
    names = [r["name"] for r in c]
    # =inf= instances and supportcase22 excluded; ferc excluded; ML4CO only valid 9900–9924 and train 0–34
    assert set(names) == {"air05", "neos-123-abc", "rail01", "ca__2014-09-01_reserves_0", "rts_gmlc__2020-01-27",
                          "item_placement_9900", "item_placement_9924", "item_placement_0", "item_placement_34"}
    strata = {r["name"]: r["stratum"] for r in c}
    assert strata["air05"] == "miplib" and strata["rts_gmlc__2020-01-27"] == "pglib_uc"
    assert strata["item_placement_0"] == "ml4co_ip"
    assert {r["family"] for r in c if r["stratum"] == "miplib"} == {"air", "neos", "rail"}


def test_stage1_jobs_and_fields(tmp_path):
    jobs = mj.screening_stage1(mj.candidates(_fake_data(tmp_path / "data")))
    assert len(jobs) == 9
    for j in jobs:
        assert list(j)[:11] == list(mj.FIELDS)
        assert j["arm"] == "SCREEN" and j["seed"] == 900 and j["budget"] == 300.0 and j["split"] == "screen"
        assert j["block"] == f"{j['name']}__s900"


def test_files_are_byte_identical(tmp_path):
    cands = mj.candidates(_fake_data(tmp_path / "data"))
    a = mj.write_jobs(mj.screening_stage1(cands), tmp_path / "a.jsonl").read_bytes()
    b = mj.write_jobs(mj.screening_stage1(list(reversed(cands))), tmp_path / "b.jsonl").read_bytes()
    assert a == b
    pool = _pool_rows()
    c1 = mj.write_jobs(mj.confirmatory(pool, "configs/frozen-e2.yaml", emphasis="aggressive"), tmp_path / "c1.jsonl")
    c2 = mj.write_jobs(mj.confirmatory(list(reversed(pool)), "configs/frozen-e2.yaml", emphasis="aggressive"),
                       tmp_path / "c2.jsonl")
    assert c1.read_bytes() == c2.read_bytes()
    assert all(json.loads(line) for line in c1.read_text().splitlines())


def test_stage2_rule():
    res = [dict(name="a", p300="0.049", feasible="1"),      # solved / easy: no second seed
           dict(name="b", p300="0.05", feasible="1"),       # at the threshold: kept
           dict(name="c", p300="", feasible="0"),           # no solution: kept
           dict(name="d", p300="0.8", feasible="true"),
           dict(name="e", p300=None, feasible=False)]
    assert mj.stage2_selected(res) == ["b", "c", "d", "e"]
    cands = [dict(instance=f"data/x/{n}.mps.gz", name=n, stratum="miplib") for n in "abcde"]
    jobs = mj.screening_stage2(res, cands)
    assert [j["name"] for j in jobs] == ["b", "c", "d", "e"] and {j["seed"] for j in jobs} == {901}
    with pytest.raises(mj.JobError):
        mj.screening_stage2([dict(name="zz", p300="0.5", feasible="1")], cands)


def _pool_rows():
    rows = []
    for i in range(10):
        rows.append(dict(instance=f"data/miplib/inst/m{i}.mps.gz", name=f"m{i}", stratum="miplib_opt", split="test"))
    for i in range(4):
        rows.append(dict(instance=f"data/pglib/p{i}.mps.gz", name=f"p{i}", stratum="pglib_uc", split="test"))
    for i in range(3):
        rows.append(dict(instance=f"data/ml4co/i{i}.mps.gz", name=f"i{i}", stratum="ml4co_ip", split="test"))
    return rows


def test_confirmatory_blocks():
    pool = _pool_rows()
    jobs = mj.confirmatory(pool, "configs/frozen-e2.yaml", emphasis="aggressive", ablations=("x0",))
    blocks = make_blocks(jobs)
    assert len(blocks) == len(pool) * 2                     # instance x {1000, 1001}
    a3r = set(mj.a3r_subset(pool))
    assert len(a3r) == 8 and {r["stratum"] for r in pool if r["name"] in a3r} == {"miplib_opt", "pglib_uc", "ml4co_ip"}
    for b in blocks:
        labels = Counter(r.label for r in b.runs)
        assert all(v == 1 for v in labels.values())
        assert set(mj.CONFIRM_ARMS) <= set(labels) and "A3~x0" in labels
        assert ("A3r" in labels) == (b.name in a3r)
        assert ("A6" in labels) == (b.seed == 1000)          # A6 runs one seed
        assert {r.job["seed"] for r in b.runs} == {b.seed}   # common random numbers within a block
        assert all(r.job["config"] == "configs/frozen-e2.yaml" and r.job["split"] == "test" for r in b.runs)
    a0s = [j for j in jobs if j["arm"] == "A0star"]
    assert a0s and all(j["emphasis"] == "aggressive" for j in a0s)
    assert all("emphasis" not in j for j in jobs if j["arm"] != "A0star")
    with pytest.raises(mj.JobError, match="emphasis"):
        mj.confirmatory(pool, "configs/frozen-e2.yaml")
    with pytest.raises(mj.JobError, match="frozen"):
        mj.confirmatory(pool, None, emphasis="x")


def test_e1_and_e4_caps():
    tuning = [dict(instance=f"data/t{i}.mps.gz", name=f"t{i}", stratum="s", split="tuning") for i in range(16)]
    e1 = mj.e1_jobs(tuning)
    assert Counter(j["arm"] for j in e1) == {"A5": 16, "A6": 16} and {j["seed"] for j in e1} == {0}
    cfgs = [f"configs/e4/c{i:02d}.yaml" for i in range(12)]
    e4 = mj.e4_jobs(tuning, cfgs)
    assert len(e4) == 16 * 13 and {j["budget"] for j in e4} == {120.0}
    blocks = make_blocks(e4)
    assert all(len(b.runs) == 13 for b in blocks)           # 12 candidates + the default, one label each
    with pytest.raises(mj.JobError, match="configurations"):
        mj.e4_jobs(tuning, cfgs + ["configs/e4/c12.yaml"])
    with pytest.raises(mj.JobError, match="instances"):
        mj.e4_jobs(tuning + [dict(instance="x", name="x", stratum="s", split="tuning")], cfgs)
    with pytest.raises(mj.JobError, match="budget"):
        mj.e4_jobs(tuning, cfgs, budget=300)


def test_job_rejects_unknown_arm_and_fields():
    with pytest.raises(mj.JobError):
        mj.job("x.mps", "s", "A99", 0, 10)
    with pytest.raises(mj.JobError):
        mj.job("x.mps", "s", "A0", 0, 10, colour="red")


def test_read_pool_csv(tmp_path):
    p = tmp_path / "pool.csv"
    p.write_text("instance,name,stratum,side\ndata/a.mps.gz,a,miplib_opt,test\ndata/b.mps.gz,b,pglib_uc,tuning\n")
    assert [r["name"] for r in mj.read_pool(p, "test")] == ["a"]
    assert mj.read_pool(p, "tuning")[0]["split"] == "tuning"


def test_cli_smoke(tmp_path):
    out = tmp_path / "s.jsonl"
    assert mj.main(["smoke", "--instance", str(tmp_path / "toy.mps"), "--arms", "A0,A1", "--out", str(out)]) == 0
    jobs = [json.loads(line) for line in out.read_text().splitlines()]
    assert [j["arm"] for j in jobs] == ["A0", "A1"] and all(j["split"] == "smoke" for j in jobs)


def test_backoff_stage1_jobs():
    from itrea.analysis.backoff import STAGE1
    from itrea.experiments import make_jobs as MJ
    jobs = MJ.backoff_stage1(seeds=(0, 1))
    assert len(jobs) == len(STAGE1) * 2 * 3 == 84
    assert {(j["arm"], j.get("variant")) for j in jobs} == {("A9", "bo1s"), ("A9", "bo1"), ("A6", None)}
    assert {j["seed"] for j in jobs} == {0, 1} and all(j["budget"] == 120.0 and j["set"] == "backoff1" for j in jobs)
    blocks = {}
    for j in jobs:
        blocks.setdefault(j["block"], set()).add((j["arm"], j.get("variant")))
    assert len(blocks) == 28 and all(len(v) == 3 for v in blocks.values())
    assert len(MJ.backoff_stage1(seeds=(0,))) == 42


def _mipfeas_data(root):
    from itrea.analysis.mipfeas import SAMPLE
    inst = root / "miplib" / "inst"
    inst.mkdir(parents=True)
    for n in SAMPLE:
        (inst / f"{n}.mps.gz").write_bytes(b"")
    return root


def test_mipfeas_calibration_jobs(tmp_path):
    """D-020's calibration: the 30 sampled instances in D-020's order x A6m, seed 0, 600 s, one block each, stratum
    "0"–"5" (a string: 0 would read as no stratum), split dev (development, never test)."""
    from itrea.analysis.mipfeas import SAMPLE, STRATA
    jobs = mj.mipfeas_calibration(data=_mipfeas_data(tmp_path / "data"))
    assert [j["name"] for j in jobs] == list(SAMPLE) and len(jobs) == 30
    assert all(j["arm"] == "A6m" and j["seed"] == 0 and j["budget"] == 600.0 and j["set"] == "mipfeas" and
               j["split"] == "dev" and j["config"] is None and j["instance"].endswith(f"inst/{j['name']}.mps.gz")
               for j in jobs)
    assert [j["stratum"] for j in jobs] == [str(k) for k, s in enumerate(STRATA) for _ in s]
    blocks = make_blocks(jobs)
    assert len(blocks) == 30 and {b.stratum for b in blocks} == {"0", "1", "2", "3", "4", "5"}
    (tmp_path / "data" / "miplib" / "inst" / "sing44.mps.gz").unlink()
    with pytest.raises(mj.JobError, match="sing44"):
        mj.mipfeas_calibration(data=tmp_path / "data")


def test_itx_job_files_are_the_writers_output(tmp_path):
    """jobs/itx-*.jsonl (D-021 phase 1) are exactly what `make-jobs itx-*` writes; D-020's A6m writer stays (its file
    left jobs/ when M4 replaced it) (skipped without the instances)."""
    from itrea import paths
    from itrea.analysis.mipfeas import SAMPLE
    need = set(SAMPLE) | set(mj.ITX_M3_INSTANCES)
    if not all((paths.DATA / "miplib" / "inst" / f"{n}.mps.gz").is_file() for n in need) or \
            not (paths.DATA / "miplib" / "benchmark-v2.test").is_file():
        pytest.skip("the MIPLIB 2017 instances are not under data/")
    for k in mj.ITX_OUT:
        out = tmp_path / f"{k}.jsonl"
        assert mj.main([f"itx-{k}", "--out", str(out)]) == 0
        assert out.read_bytes() == (paths.ROOT / mj.ITX_OUT[k]).read_bytes(), k
    assert mj.main(["mipfeas-cal", "--out", str(tmp_path / "a6m.jsonl")]) == 0
    assert all(json.loads(x)["arm"] == "A6m" for x in (tmp_path / "a6m.jsonl").read_text().splitlines())


def _itx_data(root):
    """A fake data/ for the ITX writers: D-020's sample, M3's 8 and the development instances on disk, and a
    benchmark-v2.test of 240 names whose 7 =inf= leave 233."""
    from itrea.analysis.mipfeas import SAMPLE
    inst = root / "miplib" / "inst"
    inst.mkdir(parents=True)
    names = list(SAMPLE) + list(mj.ITX_M3_INSTANCES) + [n for n in mj.development_names()
                                                        if not n.startswith(("ca__", "item_"))]
    names = list(dict.fromkeys(names))
    names += [f"filler{i}" for i in range(240 - len(names))]
    for n in names:
        (inst / f"{n}.mps.gz").write_bytes(b"")
    (root / "miplib" / "benchmark-v2.test").write_text("".join(f"{n}.mps.gz\n" for n in names))
    (root / "miplib" / "solu.txt").write_text("".join(f"=inf= filler{i}\n" for i in range(7)))
    return root


def test_itx_writers(tmp_path):
    """D-021 phase 1: run counts, arms, harnesses, budgets and seeds of M1–M5 and R13; distinct run ids within and
    across the sets (each set has its own variant; M4 keeps the plain A6F); M2's AB / BA order; M5 = the 15
    development / class-probe instances among the 233; R13 one block per instance x seed."""
    from itrea.analysis.mipfeas import SAMPLE
    from itrea.experiments import pair
    data = _itx_data(tmp_path / "data")
    sets = dict(m1=mj.itx_m1(), m2=mj.itx_m2(), m3=mj.itx_m3(data=data), m4=mj.itx_m4(data=data),
                m5=mj.itx_m5(data=data), r13=mj.itx_r13())
    assert {k: len(v) for k, v in sets.items()} == dict(m1=12, m2=4, m3=8, m4=30, m5=15, r13=102)
    lab = {k: Counter(f"{j['arm']}@{j.get('variant')}" for j in v) for k, v in sets.items()}
    assert lab["m1"] == {"A9@m1": 3, "A6F@m1": 3, "A2F@m1": 3, "A6F@m1env": 3}
    assert lab["r13"] == {"A9@r13": 34, "A9@r13x": 34, "A6F@r13x": 34}
    assert lab["m4"] == {"A6F@None": 30} and [j["name"] for j in sets["m4"]] == list(SAMPLE)
    assert all(j["set"] == "mipfeas" and j["harness"] == "ext" for j in sets["m4"])
    assert {j["budget"] for k in ("m2", "m3", "m4", "m5") for j in sets[k]} == {600.0}
    assert {j["budget"] for k in ("m1", "r13") for j in sets[k]} == {120.0}
    assert {j["seed"] for k in sets for j in sets[k] if k != "r13"} == {0} and \
        {j["seed"] for j in sets["r13"]} == {0, 1}
    assert all(j["config"] == mj.ITX_A9_CONFIG for v in sets.values() for j in v if j["arm"] == "A9")
    assert all(j["split"] == "dev" for v in sets.values() for j in v)
    assert all(pair.resolve_harness(j["harness"])["cuopt_rmm_gb"] == 40.0 and
               pair.resolve_harness(j["harness"])["cuopt_memory_void"] is False for j in sets["m3"])
    env = [j for j in sets["m1"] if j.get("variant") == "m1env"]
    assert all(pair.resolve_harness(j["harness"])["cuopt_env_unset"] is False for j in env)
    # R13's control: the ext harness with R13 and R14 off, so only they differ from A9@r13x (same caps, cache, voids)
    ctl = [pair.resolve_harness(j["harness"]) for j in sets["r13"] if j["variant"] == "r13"]
    ext = pair.resolve_harness("ext")
    assert len(ctl) == 34 and all({k for k in ext if c[k] != ext[k]} == {"cuopt_gap_zero", "cuopt_env_unset"}
                                  for c in ctl)
    assert all(j.get("harness") is not None for v in sets.values() for j in v)
    m2 = {b.name: [r.label for r in b.runs] for b in make_blocks(sets["m2"])}
    assert m2 == {"comp21-2idx": ["A9@m2", "A6F@m2"], "neos-4763324-toguru": ["A6F@m2", "A9@m2"]}
    assert len({j["name"] for j in sets["m5"]}) == 15 and "ca__Scenario400_reserves_1" not in \
        {j["name"] for j in sets["m5"]}
    assert len(make_blocks(sets["r13"])) == 34
    ids = [pair.run_id_for(j) for v in sets.values() for j in v]
    assert len(ids) == len(set(ids)) == 171
    with pytest.raises(mj.JobError, match="unknown harness"):
        mj.job("x.mps", "s", "A6F", 0, 1.0, harness="nope")
    (tmp_path / "data" / "miplib" / "solu.txt").write_text("=inf= filler0\n")
    with pytest.raises(mj.JobError, match="233"):
        mj.mipfeas_233(tmp_path / "data")
