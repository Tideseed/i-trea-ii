"""Job files: deterministic output, the D-002 phase rules (screening stages, confirmatory blocks, E4 caps) and the
candidate list built from a fake data directory (no real instance is read)."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from itrea import paths
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


def _skip_without_miplib():
    from itrea import paths
    from itrea.analysis.itx_strata import read_strata
    from itrea.analysis.mipfeas import SAMPLE
    need = set(SAMPLE) | set(mj.ITX_M3_INSTANCES) | set(read_strata()["name"])
    if not all((paths.DATA / "miplib" / "inst" / f"{n}.mps.gz").is_file() for n in need) or \
            not (paths.DATA / "miplib" / "benchmark-v2.test").is_file():
        pytest.skip("the MIPLIB 2017 instances are not under data/")


# One case per file, so a stale file does not hide the others (until 2026-10-13 one loop stopped at r13 and never
# compared a0, m6 and m6a). jobs/itx-r13.jsonl was narrowed by hand to 58 runs in e1403fa (D-022's amendment) while
# `make-jobs itx-r13` still writes 102: a known failure, strict so it is noticed once the writer is fixed (code owed,
# #37).
_R13_STALE = pytest.mark.xfail(strict=True, reason="jobs/itx-r13.jsonl narrowed by hand to 58 runs (e1403fa); "
                                                   "the writer still makes 102 — code owed, #37")


@pytest.mark.parametrize("k", [pytest.param(k, marks=_R13_STALE) if k == "r13" else k for k in mj.ITX_OUT])
def test_itx_job_files_are_the_writers_output(tmp_path, k):
    """jobs/itx-*.jsonl (D-021 phase 1) are exactly what `make-jobs itx-*` writes (skipped without the instances)."""
    from itrea import paths
    _skip_without_miplib()
    out = tmp_path / f"{k}.jsonl"
    assert mj.main([f"itx-{k}", "--out", str(out)]) == 0
    assert out.read_bytes() == (paths.ROOT / mj.ITX_OUT[k]).read_bytes(), k


def test_mipfeas_cal_writer_stays(tmp_path):
    """D-020's A6m writer stays (its file left jobs/ when M4 replaced it) (skipped without the instances)."""
    _skip_without_miplib()
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


def _strata_data(root):
    """A fake data/ holding an (empty) instance file for every name of the committed strata table."""
    from itrea.analysis.itx_strata import read_strata
    inst = root / "miplib" / "inst"
    inst.mkdir(parents=True)
    for n in read_strata()["name"]:
        (inst / f"{n}.mps.gz").write_bytes(b"")
    return root


def test_itx_v1_tiers(tmp_path):
    """D-021 phase 2's tier writer on the committed strata: T1 64 / T2 57 / T3 109 blocks of {A9, A6F} at seed 2000,
    T4 the 20 panel blocks at 2001, T5 live A2F on the panel at 2000 (one arm per block); the itx1 harness, 600 s,
    split ext; the cuOpt version in the variant (and in the set unless 26.10), so a 26.08 and a 26.10 run never share
    an id; AB / BA by the block hash; the does-not-fit list in no tier."""
    from itrea.analysis.itx_strata import DOES_NOT_FIT
    from itrea.experiments import pair
    from itrea.experiments.pool import order_blocks
    data = _strata_data(tmp_path / "data")
    cfg = mj.ITX_A9_CONFIG                     # a stand-in: configs/frozen-v2.yaml (the C1 freeze) does not exist yet
    tiers = {t: mj.itx_v1(t, a9_config=cfg, data=data) for t in mj.ITX_TIERS}
    blocks = {t: make_blocks(v) for t, v in tiers.items()}
    assert {t: (len(v), len(blocks[t])) for t, v in tiers.items()} == \
        dict(T1=(128, 64), T2=(114, 57), T3=(218, 109), T4=(40, 20), T5=(20, 20))
    every = [j for v in tiers.values() for j in v]
    assert {j["harness"] for j in every} == {"itx1"} and {j["split"] for j in every} == {"ext"}
    assert {j["budget"] for j in every} == {600.0} and {j["variant"] for j in every} == {"v1c2610"}
    assert {t: {j["set"] for j in v} for t, v in tiers.items()} == {t: {f"itx-t{t[1]}"} for t in mj.ITX_TIERS}
    assert {j["seed"] for t in ("T1", "T2", "T3", "T5") for j in tiers[t]} == {2000} and \
        {j["seed"] for j in tiers["T4"]} == {2001}
    assert {j["arm"] for j in tiers["T5"]} == {"A2F"} and \
        all(sorted(r.label for r in b.runs) == ["A6F@v1c2610", "A9@v1c2610"] for b in blocks["T1"])
    assert all(j["config"] == mj.ITX_A9_CONFIG for j in every if j["arm"] == "A9")
    assert all(j["config"] is None for j in every if j["arm"] != "A9")
    assert not {j["name"] for j in every} & set(DOES_NOT_FIT)
    assert {j["name"] for j in tiers["T4"]} == {j["name"] for j in tiers["T5"]}
    first = [b.runs[0].label.split("@")[0] for b in blocks["T1"] + blocks["T2"] + blocks["T3"]]
    assert 0.35 < first.count("A9") / len(first) < 0.65                           # AB / BA by the block hash
    assert len(order_blocks(blocks["T2"])) == 57 and \
        len({b.stratum for b in blocks["T2"]}) == 5                               # W-only + 4 E quartiles
    ids = [pair.run_id_for(j) for j in every]
    old = mj.itx_v1("T1", cuopt="26.08", a9_config=cfg, data=data)
    assert {j["set"] for j in old} == {"itx-t1-c2608"} and {j["variant"] for j in old} == {"v1c2608"}
    ids_old = [pair.run_id_for(j) for j in old]
    assert len(set(ids)) == len(ids) == 520 and not set(ids) & set(ids_old)
    assert mj.cuopt_tag("26.10") == "2610"
    with pytest.raises(mj.JobError, match="YY.MM"):
        mj.cuopt_tag("26.1")
    with pytest.raises(mj.JobError, match="tier"):
        mj.itx_v1("T6", data=data)
    # review finding (sets): no development default — a tier with A9 needs the registered file; T5 (A2F only) does not
    assert mj.ITX_A9_FROZEN == "configs/frozen-v2.yaml"
    if not paths.resolve(mj.ITX_A9_FROZEN).exists():
        for t in ("T1", "T2", "T3", "T4"):
            with pytest.raises(mj.JobError, match="frozen-v2.yaml does not exist"):
                mj.itx_v1(t, data=data)
        assert len(mj.itx_v1("T5", data=data)) == 20
        assert mj.main(["itx-v1", "--tier", "T1", "--data", str(data), "--out", str(tmp_path / "t1.jsonl")]) == 2
    with pytest.raises(mj.JobError, match="does not exist"):
        mj.itx_v1("T1", a9_config="configs/nothere.yaml", data=data)
    # memory per run under itx1 (D-021: A9 97, A2F 78, A6F 69 GB with the file mailbox)
    need = {j["arm"]: pair.job_budget(j).need_gb for j in tiers["T1"][:2] + tiers["T5"][:1]}
    assert need == dict(A9=97.0, A6F=69.0, A2F=78.0)
    out = tmp_path / "t4.jsonl"
    assert mj.main(["itx-v1", "--tier", "T4", "--cuopt", "26.08", "--a9-config", cfg, "--data", str(data), "--out",
                    str(out)]) == 0
    assert len(out.read_text().splitlines()) == 40


def test_itx_a0_and_m6(tmp_path):
    """A0 on the 233 minus the list (230) under itx1 at seed 2000, H_cu ∪ W_SCIP unseen first (priority 0: 81, then
    149); M6: 3 loads, then the 3 measured runs, then 11 loads (priorities 0 / 1 / 2), and its 3 runs alone."""
    from itrea.analysis.itx_strata import DOES_NOT_FIT, read_strata
    from itrea.experiments import pair
    from itrea.experiments.pool import order_blocks
    data = _strata_data(tmp_path / "data")
    a0 = mj.itx_a0(data=data)
    df = read_strata().set_index("name")
    assert len(a0) == 230 and not {j["name"] for j in a0} & set(DOES_NOT_FIT)
    assert {(j["arm"], j["variant"], j["seed"], j["budget"], j["harness"], j["split"], j["set"]) for j in a0} == \
        {("A0", "v1", 2000, 600.0, "itx1", "ext", "itx-a0")}
    p0 = {j["name"] for j in a0 if j["priority"] == 0}
    want = {n for n, r in df.iterrows() if r["unseen"] and (r["H_cu"] or r["W_SCIP"]) and not r["does_not_fit"]}
    assert p0 == want and len(p0) == 81 and sum(j["priority"] == 1 for j in a0) == 149
    order = order_blocks(make_blocks(a0))
    assert [b.priority for b in order] == [0] * 81 + [1] * 149
    assert pair.job_budget(a0[0]).need_gb == 29.0                                   # 9 GiB SCIP + the 20 GB reserve
    m6, m6a = mj.itx_m6(), mj.itx_m6a()
    assert len(m6) == 17 and len(m6a) == 3 and {j["harness"] for j in m6 + m6a} == {"ext"}
    order = order_blocks(make_blocks(m6))
    assert [b.priority for b in order] == [0] * 3 + [1] * 3 + [2] * 11
    assert {b.name for b in order[3:6]} == set(mj.ITX_M6_MEASURED) and \
        {b.runs[0].job["budget"] for b in order[3:6]} == {60.0}
    assert {j["budget"] for j in m6 if j["variant"] == "m6L"} == {600.0}
    assert {j["name"] for j in m6a} == set(mj.ITX_M6_MEASURED) and {j["variant"] for j in m6a} == {"m6a"}
    ids = [pair.run_id_for(j) for j in a0 + m6 + m6a]
    assert len(set(ids)) == len(ids)


def test_itx_a0_d024_selection(tmp_path):
    """D-024 (f) (Berk 2026-10-13: "Yes, 81"): `itx-a0 --select d024` = only H_cu ∪ W_SCIP unseen (81), v1's stage-1
    instances first — priority 0: the 20-instance panel + the 5 H_cu-unseen instances where cuOpt finds no solution
    (D-024 (a)), priority 1: the other 56 — the same jobs (so run ids) as those 81 lines of the 230; `--select stage1`
    the 25 alone; the 230 unchanged."""
    from itrea.analysis.itx_strata import read_strata
    from itrea.experiments import pair
    from itrea.experiments.pool import order_blocks
    data = _strata_data(tmp_path / "data")
    df = read_strata()
    st1 = set(df[mj.v1_stage1_mask(df)]["name"])
    five = {"bnatt400", "neos-3216931-puriri", "peg-solitaire-a3", "supportcase19", "supportcase22"}
    assert len(st1) == 25 and five <= st1 and st1 - five == set(df[df["panel"]]["name"])
    every = mj.itx_a0(data=data)
    a81 = mj.itx_a0(data=data, select="d024")
    assert len(a81) == 81 and {j["priority"] for j in every if j["name"] in {x["name"] for x in a81}} == {0}
    assert [{k: v for k, v in j.items() if k != "priority"} for j in a81] == \
        [{k: v for k, v in j.items() if k != "priority"} for j in every if j["priority"] == 0]
    assert {j["name"] for j in a81 if j["priority"] == 0} == st1 and sum(j["priority"] == 1 for j in a81) == 56
    assert [b.priority for b in order_blocks(make_blocks(a81))] == [0] * 25 + [1] * 56
    assert {pair.run_id_for(j) for j in a81} == {pair.run_id_for(j) for j in every if j["priority"] == 0}
    a25 = mj.itx_a0(data=data, select="stage1")
    assert {j["name"] for j in a25} == st1 and {j["priority"] for j in a25} == {0}
    assert len(every) == 230 and sum(j["priority"] == 0 for j in every) == 81            # "all" unchanged
    with pytest.raises(mj.JobError, match="select"):
        mj.itx_a0(data=data, select="81")
    out = tmp_path / "a81.jsonl"
    assert mj.main(["itx-a0", "--select", "d024", "--data", str(data), "--out", str(out)]) == 0
    assert [json.loads(x) for x in out.read_text().splitlines()] == a81
    assert mj.ITX_A0_SELECT["d024"] == "jobs/itx-a0-81.jsonl" and "a0-81" not in mj.ITX_OUT


def test_itx_a0_81_file_is_the_writers_output(tmp_path):
    """jobs/itx-a0-81.jsonl (D-024 (f), first written 2026-10-13 by a throwaway script from jobs/itx-a0.jsonl) is
    exactly `make-jobs itx-a0 --select d024` (skipped without the instances)."""
    from itrea.analysis.itx_strata import read_strata
    if not all((paths.DATA / "miplib" / "inst" / f"{n}.mps.gz").is_file() for n in read_strata()["name"]):
        pytest.skip("the MIPLIB 2017 instances are not under data/")
    out = tmp_path / "a81.jsonl"
    assert mj.main(["itx-a0", "--select", "d024", "--out", str(out)]) == 0
    assert out.read_bytes() == (paths.ROOT / "jobs" / "itx-a0-81.jsonl").read_bytes()


# ================================================================== R1 / R2 of the C1 diagnosis (2026-10-14)
_PLACE = ("set", "stratum", "block", "variant", "position")


def _without_place(j: dict) -> dict:
    return {k: v for k, v in j.items() if k not in _PLACE}


def _committed_ids(skip=()) -> set[str]:
    """Every run id of every committed job file (a job whose configuration is gone is left out)."""
    from itrea.experiments import pair
    out = set()
    for f in sorted((paths.ROOT / "jobs").glob("*.jsonl")):
        if f.name in skip:
            continue
        for line in f.read_text().splitlines():
            try:
                out.add(pair.run_id_for(json.loads(line)))
            except Exception:           # noqa: BLE001
                continue
    return out


def _positions_alternate(jobs: list[dict], control: str) -> None:
    """D-025 / D-026 (position clarification): every block carries explicit positions, the control first at even
    seeds and the treatment first at odd ones; the pool keeps them."""
    for b in make_blocks(jobs):
        order = [r.label for r in sorted(b.runs, key=lambda r: r.job["position"])]
        assert sorted(r.job["position"] for r in b.runs) == [0, 1], b.id
        assert (order[0] == f"A9@{control}") == (b.seed % 2 == 0), (b.id, order)


@pytest.mark.parametrize("control", ["r13x", "boY"])
@pytest.mark.parametrize("variant", ["bsync", "bnice", "ysync"])
def test_r1_bsync_jobs(control, variant):
    """D-025's test: 5 instances x seeds 0-1 x {treatment, control}, 120 s, ext, one block per instance x seed (20 runs,
    set r1-<variant>). The control is the same-day development default's existing job — D-022's A9@r13x or D-023's
    A9@boY — re-run under this set's own label A9@r1ctl (A9@r1ctl-<variant> for a fallback set): every field but set /
    stratum / block / label / position equal, the same run HASH, a new run id; the treatment is that control's
    configuration plus the variant. No run id of the set is an id of any committed job file."""
    from itrea.experiments import pair
    jobs = mj.r1_bsync(control, variant)
    assert len(jobs) == 20 and len(make_blocks(jobs)) == 10
    assert Counter(j["name"] for j in jobs) == {n: 4 for n in mj.R1_INSTANCES}
    assert {j["seed"] for j in jobs} == {0, 1} and {j["budget"] for j in jobs} == {120.0}
    assert {j["harness"] for j in jobs} == {"ext"} and {j["arm"] for j in jobs} == {"A9"}
    assert {j["set"] for j in jobs} == {f"r1-{variant}"} and {j["split"] for j in jobs} == {"dev"}
    t_label = ("boY-" if control == "boY" else "") + variant
    c_label = "r1ctl" if variant == "bsync" else f"r1ctl-{variant}"
    assert Counter(j["variant"] for j in jobs) == {t_label: 10, c_label: 10}
    assert all((paths.ROOT / j["config"]).is_file() for j in jobs)
    _positions_alternate(jobs, c_label)
    ctl = [j for j in jobs if j["variant"] == c_label]
    if control == "r13x":
        old = {(j["name"], j["seed"]): j for j in mj.itx_r13() if j.get("variant") == "r13x" and j["arm"] == "A9"}
        want_hash = "28e5a5e9c725"
    else:
        old = {(j["name"], j["seed"]): j for j in mj.backoff2(harness="ext", instances=list(mj.R1_INSTANCES))
               if j.get("variant") == "boY"}
        want_hash = "86b1e527db68"
    for j in ctl:
        o = old[(j["name"], j["seed"])]
        assert _without_place(j) == _without_place(o)
        assert pair.run_hash(j) == pair.run_hash(o) == want_hash
        assert pair.run_id_for(j) != pair.run_id_for(o)
    trt = [j for j in jobs if j["variant"] == t_label]
    assert {j["config"] for j in trt} == {f"configs/itrea2-dev-rc-3way-{t_label}.yaml"}
    ids = {pair.run_id_for(j) for j in jobs}
    own = mj.r1_out(variant).rsplit("/", 1)[1]            # the set's own file, committed once D-023 / D-025 decided
    assert len(ids) == 20 and not ids & _committed_ids(skip=(own,))


def test_r1_fallback_controls_do_not_repeat_r1_bsync_ids():
    from itrea.experiments import pair
    ids = {v: {pair.run_id_for(j) for j in mj.r1_bsync("r13x", v)} for v in mj.R1_VARIANTS}
    assert not ids["bsync"] & ids["bnice"] and not ids["bsync"] & ids["ysync"] and not ids["bnice"] & ids["ysync"]


def test_r1_bsync_refuses_unknown_controls_and_variants():
    with pytest.raises(mj.JobError, match="control"):
        mj.r1_bsync("bo1s")
    with pytest.raises(mj.JobError, match="variant"):
        mj.r1_bsync("r13x", "bsyncn")


@pytest.mark.parametrize("base, nice, t_stem, c_stem", [
    ("bsync", 10, "bsync-ld", "bsync"), ("bsync", 0, "bsync-ld0", "bsync"),
    ("boY-bsync", 10, "boY-bsync-ld", "boY-bsync"), ("boY-bsync", 0, "boY-bsync-ld0", "boY-bsync"),
    ("r13x", None, "ld0", ""), ("boY", 0, "boY-ld0", "boY")])
def test_r2_ld_jobs(base, nice, t_stem, c_stem):
    """D-026's test: {A9@r2ld, A9@r2ctl} x item_placement_9903 and sing44 at seeds 0-3 + ca, glass4, kumeu at seeds 0-1
    (28 runs, 14 blocks), 120 s, ext, set r2-ld. The control runs the base's configuration file (with R1 adopted:
    R1's treatment, the same run hash as A9@bsync / A9@boY-bsync), the treatment the base + the gated decode at the
    stated nice; every run id new."""
    from itrea.experiments import pair
    jobs = mj.r2_ld(base, nice)
    assert len(jobs) == 28 and len(make_blocks(jobs)) == 14 and {j["set"] for j in jobs} == {"r2-ld"}
    seeds = {n: sorted({j["seed"] for j in jobs if j["name"] == n}) for n in {j["name"] for j in jobs}}
    assert seeds == {"item_placement_9903": [0, 1, 2, 3], "sing44": [0, 1, 2, 3], "ca__Scenario400_reserves_1": [0, 1],
                     "glass4": [0, 1], "neos-3656078-kumeu": [0, 1]}
    assert Counter(j["variant"] for j in jobs) == {"r2ld": 14, "r2ctl": 14}
    assert {j["harness"] for j in jobs} == {"ext"} and {j["budget"] for j in jobs} == {120.0}
    cfg = lambda s: f"configs/itrea2-dev-rc-3way{'-' + s if s else ''}.yaml"    # noqa: E731
    assert {j["config"] for j in jobs if j["variant"] == "r2ld"} == {cfg(t_stem)}
    assert {j["config"] for j in jobs if j["variant"] == "r2ctl"} == {cfg(c_stem)}
    assert all((paths.ROOT / j["config"]).is_file() for j in jobs)
    _positions_alternate(jobs, "r2ctl")
    for j in jobs:
        pair.job_budget(j)                                        # the runner plans both arms (gated lp_decode)
    if base in ("bsync", "boY-bsync"):                            # the control = R1's treatment configuration
        r1 = {(j["name"], j["seed"]): j for j in mj.r1_bsync("boY" if base.startswith("boY") else "r13x")
              if j["variant"] == base}
        for j in (j for j in jobs if j["variant"] == "r2ctl" and (j["name"], j["seed"]) in r1):
            o = r1[(j["name"], j["seed"])]
            assert pair.run_hash(j) == pair.run_hash(o) and pair.run_id_for(j) != pair.run_id_for(o)
    ids = {pair.run_id_for(j) for j in jobs}
    own = mj.R2_OUT.rsplit("/", 1)[1]                      # the set's own file, committed once D-025 decided
    assert len(ids) == 28 and not ids & _committed_ids(skip=(own,))


def test_r2_nice_rule_and_refusals():
    assert mj.r2_nice(0.12) == 10 and mj.r2_nice(0.4999) == 10 and mj.r2_nice(0.5) == 0 and mj.r2_nice(1.0) == 0
    with pytest.raises(mj.JobError):
        mj.r2_nice(float("nan"))
    with pytest.raises(mj.JobError, match="nice 0"):
        mj.r2_ld("r13x", 10)                                      # the fallback runs the decoder at nice 0
    with pytest.raises(mj.JobError, match="--nice"):
        mj.r2_ld("bsync", 5)
    with pytest.raises(mj.JobError, match="required"):
        mj.r2_ld("bsync")                                         # no default before D-025's measurement
    with pytest.raises(mj.JobError, match="--base"):
        mj.r2_ld("bnice")                                         # a fallback base: its -ld files come after D-025


def test_r1_r2_cli_default_paths_and_set(tmp_path):
    out = tmp_path / "r1.jsonl"
    assert mj.main(["r1-bsync", "--control", "boY", "--variant", "ysync", "--out", str(out)]) == 0
    jobs = [json.loads(x) for x in out.read_text().splitlines()]
    assert len(jobs) == 20 and {j["variant"] for j in jobs} == {"boY-ysync", "r1ctl-ysync"}
    assert {j["set"] for j in jobs} == {"r1-ysync"}
    assert mj.r1_out("bsync") == "jobs/r1-bsync.jsonl" and mj.R2_OUT == "jobs/r2-ld.jsonl"
    assert mj.main(["r1-bsync", "--control", "r13x", "--set", "r1-bsync-2", "--out", str(out)]) == 0
    jobs = [json.loads(x) for x in out.read_text().splitlines()]
    assert {j["set"] for j in jobs} == {"r1-bsync-2"} and all(j["block"].endswith("__r1-bsync-2") for j in jobs)
    out2 = tmp_path / "r2.jsonl"
    assert mj.main(["r2-ld", "--base", "bsync", "--ea-cpu", "0.7", "--out", str(out2)]) == 0
    assert {json.loads(x)["config"] for x in out2.read_text().splitlines()} == {
        "configs/itrea2-dev-rc-3way-bsync-ld0.yaml", "configs/itrea2-dev-rc-3way-bsync.yaml"}
    assert mj.main(["r2-ld", "--base", "bsync", "--out", str(out2)]) == 2          # neither --nice nor --ea-cpu
    assert mj.main(["r2-ld", "--base", "r13x", "--nice", "10", "--out", str(out2)]) == 2
    assert mj.main(["r2-ld", "--base", "r13x", "--set", "r2-ld-2", "--out", str(out2)]) == 0
    assert {json.loads(x)["set"] for x in out2.read_text().splitlines()} == {"r2-ld-2"}


# ------------------------------------------------------------------------------------- C1 diagnosis R3 (r3-diag)
def test_r3_diag_jobs():
    """R3 (C1 diagnosis note §5, D-027 as amended 2026-10-14: only what a rule reads): comp21, toguru, doon x {A6@r3,
    A2@r3} + toguru, doon x A6F@r3 + chromaticindex1024-7 x {A6@r3, A6F@r3, A2@r3} (rule 4, the 18:3x amendment),
    seeds 0-1, 120 s, ext on every arm, split dev, set r3-diag, one block per instance x seed: 22 runs in 8 blocks,
    every run id distinct, the run hashes of the 16 earlier jobs unchanged."""
    from itrea.experiments import pair
    jobs = mj.r3_diag()
    assert len(jobs) == 22
    assert Counter(pair.arm_label(j) for j in jobs) == {"A6@r3": 8, "A6F@r3": 6, "A2@r3": 8}
    chrom = [j for j in jobs if j["name"] == "chromaticindex1024-7"]
    assert sorted((pair.arm_label(j), j["seed"]) for j in chrom) == sorted(
        (a, s) for a in ("A6@r3", "A6F@r3", "A2@r3") for s in (0, 1))
    assert {j["instance"] for j in chrom} == {"data/miplib/inst/chromaticindex1024-7.mps.gz"}
    assert {j["set"] for j in jobs} == {"r3-diag"} and {j["split"] for j in jobs} == {"dev"}
    assert {j["budget"] for j in jobs} == {120.0} and {j["seed"] for j in jobs} == {0, 1}
    assert all(j["harness"] == "ext" and j["config"] is None and j["overrides"] == {} for j in jobs)
    blocks = {b.id: sorted(r.label for r in b.runs) for b in make_blocks(jobs)}
    assert len(blocks) == 8
    for n in mj.R3_INSTANCES:
        want = sorted(["A6@r3"] + (["A6F@r3"] if n in mj.R3_A6F_INSTANCES else [])
                      + (["A2@r3"] if n in mj.R3_A2_INSTANCES else []))
        assert blocks[f"{n}__s0__r3-diag"] == blocks[f"{n}__s1__r3-diag"] == want, n
    ids = [pair.run_id_for(j) for j in jobs]
    assert len(set(ids)) == 22
    assert {i.rsplit("__", 1)[1] for i in ids} == {"e54170304dad", "82a371305f88", "9c3031af075f"}   # A6, A6F, A2


def test_r3_diag_file_is_the_writers_output(tmp_path):
    """jobs/r3-diag.jsonl is exactly what `make-jobs r3-diag` writes (the instance tables only; no instance read)."""
    out = tmp_path / "r3.jsonl"
    assert mj.main(["r3-diag", "--out", str(out)]) == 0
    assert out.read_bytes() == (paths.ROOT / mj.R3_OUT).read_bytes()


def test_r3_a6f_is_the_r13x_job_under_another_variant():
    """A6F@r3 re-runs A6F@r13x like for like: each job equals the A6F@r13x line of the committed jobs/itx-r13.jsonl
    (the file that ran on 2026-10-13) or, for chromaticindex1024-7, of jobs/itx-r13b.jsonl (the C1 guard, the same set
    itx-r13) on every field but set, block, stratum and variant, so its run hash is the same
    (82a371305f88), while its run id and its directory (traces/raw/r3-diag/) differ — the pool's resume never skips
    an R3 run for an itx-r13 run, and the run table never needs a `#<set>` copy. No R3 run id is the id of any job of
    another committed job file."""
    from itrea.experiments import pair
    r13 = {(j["name"], j["seed"]): j for f in ("itx-r13.jsonl", "itx-r13b.jsonl")
           for j in map(json.loads, (paths.ROOT / "jobs" / f).read_text().splitlines())
           if pair.arm_label(j) == "A6F@r13x"}
    differ = {"set", "block", "stratum", "variant"}
    r3 = [j for j in mj.r3_diag() if j["arm"] == "A6F"]
    assert len(r3) == 6
    for j in r3:
        old = r13[(j["name"], j["seed"])]
        assert {k: v for k, v in j.items() if k not in differ} == {k: v for k, v in old.items() if k not in differ}
        assert pair.run_hash(j) == pair.run_hash(old) == "82a371305f88"
        assert pair.run_id_for(j) != pair.run_id_for(old)
        assert pair.run_dir_for(j).parent == pair.PairOptions().traces_dir / "raw" / "r3-diag"
    mine = {pair.run_id_for(j) for j in mj.r3_diag()}
    for f in sorted((paths.ROOT / "jobs").glob("*.jsonl")):
        if f.name == "r3-diag.jsonl":
            continue
        for line in f.read_text().splitlines():
            j = json.loads(line)
            try:
                rid = pair.run_id_for(j)
            except Exception:           # noqa: BLE001 — a job whose config is gone cannot share an id with R3 anyway
                continue
            assert rid not in mine, (f.name, rid)


def test_r3_diag_layout_and_memory_under_ext(tmp_path):
    """The runner lays R3's arms out as the note says, on the GPU lane g0 (F = X925 5-6 + A725 1-3): A6@r3 = cuOpt
    alone on A725 1-3 with 3 threads; A2@r3 = SCIP on X925 5 + that cuOpt (nothing on X925 6); A6F@r3 = cuOpt on all of
    F with 5 threads — every cuOpt with gaps 0, no thread variables, RMM 24 / host 24 GB. D-003's gate: A6 69, A2 78,
    A6F 69 GB (file mailbox); the pool's pair layout runs every R3 job on g0 and nowhere else."""
    from itrea.experiments import pair, pool
    o = pair.PairOptions(traces_dir=tmp_path, use_systemd=True, pin=True)
    want = {"A6": {"cuopt": ([1, 2, 3], "3")}, "A6F": {"cuopt": ([1, 2, 3, 5, 6], "5")},
            "A2": {"solver": ([5], None), "cuopt": ([1, 2, 3], "3")}}
    need = {}
    for j in mj.r3_diag():
        cfg, _ = pair.effective_config(j, o)
        procs = {p.role: p for p in pair.plan(j, o, tmp_path / pair.run_id_for(j, o), "S", cfg)}
        assert {r: p.cores for r, p in procs.items()} == {r: c for r, (c, _) in want[j["arm"]].items()}, j
        assert 6 not in {c for p in procs.values() for c in p.cores} or j["arm"] == "A6F"
        c = procs["cuopt"]
        assert c.argv[c.argv.index("--cpu-threads") + 1] == want[j["arm"]]["cuopt"][1]
        assert "--mip-gap-zero" in c.argv and c.env_unset == pair.THREAD_VARS and c.mem_gb == c.gpu_gb == 24.0
        assert "--send" not in c.argv                                     # no EA: cuOpt posts nothing
        need[j["arm"]] = pair.job_budget(j, o).need_gb
    assert need == dict(A6=69.0, A6F=69.0, A2=78.0)
    p = pool.Pool(mj.r3_diag(), pool.PoolOptions(layout="pair", governor=False, logs_dir=tmp_path / "logs",
                                                 pair_opts=pair.PairOptions(traces_dir=tmp_path)))
    p.lanes, p.blocks = p._lanes(), make_blocks(mj.r3_diag())
    assert all([lane.name for lane in p.lanes if p._accepts(lane, r)] == ["g0"] for b in p.blocks for r in b.runs)
    lines = p.memory_arm_lines()
    assert len(lines) == 3 and all(" on g0 (harness \"ext\"): " in x for x in lines)
    assert any(x.startswith("memory per run: A2@r3 ") and x.endswith("= 78.0 GB") for x in lines)
