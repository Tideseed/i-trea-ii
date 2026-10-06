"""Hand-launched smoke run of the D-011 LP stream: SCIP (--lp-stream) + the NumPy EA through a file mailbox, on cores
12/13 (the pool holds the runner lock and the measured lane). Usage (repo root): python tools/lps_smoke.py <instance> <budget s> <new out dir>."""
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from itrea.config import EAConfig
from itrea.exchange.mailbox import Mailbox
from itrea.problem import load

inst, budget, out = sys.argv[1], sys.argv[2], Path(sys.argv[3])
out.mkdir(parents=True, exist_ok=False)
run = out / "run"
run.mkdir()
p = load(inst)
mb = Mailbox(out / "mb", "runner", create=True, run_id="smoke")
mb.write_order(p.names)
cfg = EAConfig.from_yaml("configs/itrea2-adaptive-lps.yaml").with_overrides(
    {"repair.max_steps": 20000, "repair.stall_steps": 1000, "backend": "numpy", "population.size": 64,
     "population.offspring_per_gen": 64})
cfg.validate()
(run / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict()))
env = dict(os.environ, OMP_NUM_THREADS="1", PYTHONPATH="src")
common = ["--instance", inst, "--seed", "0", "--budget", budget, "--run-dir", str(run), "--mailbox", str(out / "mb"),
          "--run-id", "smoke"]
s = subprocess.Popen(["taskset", "-c", "12", sys.executable, "-m", "itrea.solvers.scip_run", *common, "--recv", "--send",
                      "--inject-band", "0.01", "--inject-k", "8", "--lp-stream", "--mem-mb", "6000"],
                     env=env, stdout=open(run / "solver.log", "w"), stderr=subprocess.STDOUT)
e = subprocess.Popen(["taskset", "-c", "13", sys.executable, "-m", "itrea.ea.run", *common, "--config",
                      str(run / "config.yaml"), "--backend", "numpy"],
                     env=env, stdout=open(run / "ea.log", "w"), stderr=subprocess.STDOUT)
end = time.monotonic() + 300
while not mb.all_ready():
    assert time.monotonic() < end and s.poll() is None and e.poll() is None, (s.poll(), e.poll())
    time.sleep(0.05)
mb.release()
time.sleep(float(budget) + 3)
mb.stop("budget")
print("rc", s.wait(60), e.wait(60))
