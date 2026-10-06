# Tutorials — a GPU evolutionary algorithm beside a MIP solver, from zero (planned)

**Status: planned, not written.** The chapters below will be written once the build passes its tests, each with a
figure made from this repository's own runs (after the experiments) and a hands-on example that runs in about a minute
on a toy model from `itrea.benchmarks.toy` — no instance downloads, no GPU required (the examples run on the NumPy
backend; the GPU is optional).

| # | chapter (planned) | you will learn | hands-on (planned) |
|---|---|---|---|
| 1 | What a MIP solver and an evolutionary algorithm each do while they run | incumbents, bounds, gaps; populations, operators, survival | `examples/ex01_solver_and_ea.py` |
| 2 | Measuring "good solutions sooner" | the primal integral, time to target, why only checked incumbents count | `examples/ex02_primal_integral.py` |
| 3 | Three ranks and good infeasible solutions | Rank 0 / Rank 1 / Rank 2, the x cap and the protected elite, what an ablation can and cannot show | `examples/ex03_three_ranks.py` |
| 4 | Feasibility jump on a GPU | weighted violation, jump values, one thread block per individual, why memory bandwidth decides | `examples/ex04_feasibility_jump.py` |
| 5 | Two-way exchange | the mailbox and Redis transports, the start barrier, echoes, post accounting | `examples/ex05_exchange.py` |
| 6 | Reading a pre-registration | primary vs secondary hypotheses, the intersection–union test, margins, "inconclusive" | reading `vault/50-decisions/D-002` |

**Background reading:** the predecessor's tutorials on MIP solvers in motion, the primal integral, portfolios and
parallel copies (`berkorbay/optopt`, `tutorials/`) cover chapters 1–2 in depth; these chapters will link to them rather
than repeat them.

## Setup (planned)
```bash
cd i-trea-ii
uv venv .venv && source .venv/bin/activate
uv pip install -e ".[dev,plots]"
python tutorials/examples/ex01_solver_and_ea.py
```
