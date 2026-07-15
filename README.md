# RND Prediction Error as a Convergence Indicator in RL

Project for the Reinforcement Learning lecture (LUH, 2026) by **Jannis Kastner** and **Tom Sommerfeld**.

## Research Question

> Can the prediction error of Random Network Distillation (RND) serve as a reliable indicator of
> reinforcement learning convergence, and how does it compare to state visitation entropy?

Instead of using the RND prediction error as an intrinsic reward for exploration, we reinterpret it
as a measure of state novelty over the course of training and evaluate whether it reliably reflects
learning progress — including a simple early-stopping criterion compared against a fixed training
budget. See [`docs/proposal.pdf`](docs/proposal.pdf) for the full proposal.

## Repository Structure

```
rnd_convergence/          Python package: agents, RND networks, convergence metrics
rnd_convergence/configs/  Hydra configs (base + per-agent/per-env)
scripts/                  Entry points for training runs and plotting
tests/                    Unit tests
docs/                     Proposal and report material
```

Raw experiment outputs (`outputs/`, `results/`, model weights) are git-ignored; curated figures and
aggregated metrics for the report are committed under `docs/`.

## Setup

Requires Python 3.11 and [uv](https://docs.astral.sh/uv/).

```bash
uv venv --python 3.11
source .venv/bin/activate
make install
```

## Development

- `make format` — format code with ruff
- `make check` — lint & format check
- `make test` — run unit tests

## Attribution

- Burda et al., 2018 — [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- Schulman et al., 2017 — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Bellemare et al., 2016 — [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)
