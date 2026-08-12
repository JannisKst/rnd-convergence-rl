# RND vs. State-Visitation Entropy: Convergence Signals as State Spaces Become Continuous

Project for the Reinforcement Learning lecture (LUH, 2026) by **Jannis Kastner** and **Tom Sommerfeld**.

## Research Question

> Does RND prediction error's advantage over state-visitation entropy (SVE) as a
> convergence-monitoring signal grow as the state space becomes more continuous and
> higher-dimensional — because SVE's discretization degrades while RND's does not?

RND prediction error is a learned, continuous-valued novelty signal that requires no discretization
of the state space. SVE requires one: exact in a small discrete environment, increasingly crude as
dimensionality grows and histogram bins outnumber visited states. We test whether that structural
difference produces a measurable gap in how well each signal tracks policy convergence.

## Study Design

**Independent variable — the environment ladder.** Environments ordered by how continuous and
high-dimensional the state space is.

**Dependent variable — signal accuracy.** At each rung, how closely does each signal's plateau track
actual policy convergence? Measured as the signed lag `Δ = t_plateau − t_conv`.

**Deliverable.** One table: environments as rows (ordered down the ladder), signals as columns, `Δ`
in the cells. The hypothesis is visible as the top-to-bottom trend — SVE's `Δ` should deteriorate
and become bin-count-dependent as the ladder is descended, while RND's stays comparatively stable.

Measuring against a ground-truth convergence point is essential rather than incidental. "Advantage
over SVE" is undefined without naming the task the signals are judged at, and without a reference
`t_conv` a stable signal cannot be distinguished from a *stably wrong* one — a detector that always
fires at step 50k is perfectly robust and perfectly useless.

## Environment Ladder

| Env | State representation | Dim | SVE status |
| --- | --- | --- | --- |
| MarsRover | discrete integer | 1 | exact — anchor rung, both signals saturate almost immediately |
| MiniGrid-Empty / DoorKey-5x5 | `(x, y, dir)` | 3, discrete | exact |
| MiniGrid-DoorKey-8x8 | `(x, y, dir)` | 3, discrete, sparse reward | exact |
| CartPole-v1 | continuous | 4 | binning strained |
| LunarLander-v3 (discrete) | continuous | 8 | binning breaks (curse of dimensionality) |

All rungs use **discrete action spaces**, so a single PPO implementation covers the ladder and no
algorithmic difference confounds the comparison.

The two MiniGrid rungs serve different purposes: the easy variant is expected to converge and yield a
well-defined `t_conv`; the hard sparse-reward variant is expected to defeat PPO without an
exploration bonus, giving a case where coverage saturates while return never converges at all.

## Signals Compared

1. **RND prediction error** — `‖f(s) − f̂(s)‖²` for a frozen random target `f` and a trained
   predictor `f̂`. The predictor is trained in a single streaming pass over the visited-state stream.
   Continuous-valued, no discretization. Running observation normalisation is applied throughout
   (per Burda et al.), otherwise the signal tracks observation scale drift rather than novelty.
2. **SVE, fixed grid** — Shannon entropy of the visitation histogram, swept over bins-per-dimension
   ∈ {5, 10, 20} on running-normalised observations. The bin count is treated as an independent
   variable rather than a fixed hyperparameter: the sweep is how discretization sensitivity is
   demonstrated.
3. **SVE, kNN (Kozachenko–Leonenko)** — differential entropy estimated over a buffer of visited
   states, without discretization. Differential entropy is not scale-invariant and may be negative,
   so it is compared only *within* an environment.

## Measurement

### Convergence ground truth (`t_conv`)

Computed offline over the completed run; it deliberately uses information from the full training
curve, because it is the target the online signals must approximate.

- Evaluate every `5_000` environment steps, `10` episodes, greedy action selection → curve `R(t)`.
- `R_ref` = mean of the final 5 evaluations; `R_0` = mean return of a random policy in that
  environment.
- `t_conv` = first `t` at which smoothed `R` stays ≥ `R_0 + 0.95 · (R_ref − R_0)` for **5 consecutive
  evaluations**.

Normalising against the random-policy baseline rather than a percentage of the final value keeps the
criterion well-defined for negative returns and for 0–1 sparse-reward ranges alike.

### Signal plateau (`t_plateau`)

Identical functional form, so the comparison is fair across signals: the signal is smoothed with a
trailing mean, and `t_plateau` is the first `t` at which the change between consecutive points falls
below `τ` times the **largest change seen so far** and stays there for **5 consecutive windows**.
Applied unchanged to RND error and to every SVE variant.

Normalising against the signal's own fastest observed rate of change, rather than against its
magnitude or its accumulated range, is what makes one `τ` valid across signals. It is dimensionless,
it is defined for the negative values that differential entropy routinely takes (a log-derivative is
not), and a signal still descending at a constant rate never satisfies it — whereas normalising by
accumulated range would eventually declare a steady decline "flat" merely because it had already
travelled a long way, manufacturing exactly the false early stop the study sets out to measure.

### Reported quantities

`Δ = t_plateau − t_conv` per environment and signal, aggregated over seeds with bootstrap confidence
intervals. `Δ < 0` means the signal fires early and stopping on it costs performance; `Δ ≈ 0` means
it tracks convergence; `Δ > 0` means it fires late and saves no compute. Reported alongside:
sensitivity of `Δ` to `τ` and to bin count, and retained performance `R(t_plateau) / R_ref` as the
practical cost of acting on each signal.

## Controls

1. **Frozen random-policy run.** The RND predictor accumulates gradient steps whether or not the
   agent discovers anything, so prediction error decays even under a policy that never learns. This
   control establishes how much of the observed plateau is attributable to predictor training rather
   than to exhausted novelty.
2. **Coverage-vs-convergence check.** Both candidate signals measure state-space coverage, not policy
   quality. We report explicitly where exploration saturates before policy improvement finishes —
   i.e. the regime in which neither signal is a safe stopping criterion.

## Experimental Protocol

RND is used here as a monitoring signal, not as an intrinsic reward, so the predictor is not part of
the training loop:

1. Training logs the visited-state stream — step index, observation, episode boundary — to disk.
2. All signals are computed post-hoc. The RND predictor is trained in a single streaming pass over
   the logged states **in visit order**, which reproduces the online signal exactly; SVE variants are
   computed over the same stream.

One PPO run per `(environment, seed)` therefore supports every bin-count sweep, every `τ` sweep and
every detector variant without retraining. Matrix: 5 environments × 10 seeds, plus the frozen-policy
control. Runs are independent and executed as a job array on the cluster.

## Out of Scope

- **DQN and continuous-action environments** (Pendulum, BipedalWalker). The ladder is spanned by
  discrete-action environments from 1 to 8 dimensions; adding a continuous-control algorithm would
  introduce an algorithmic confound without extending the axis under study.
- **NovelD** as a third signal — a possible extension, not part of the present study.

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

All code in this repository was written for this project. Where the architecture of a component
follows an exercise scaffold from the course repository (`automl-edu/RL-exercises`) — the RND
target/predictor network pair of week 7, and the PPO agent of week 6 — the corresponding module
docstring records it.

- Burda et al., 2018 — [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- Schulman et al., 2017 — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Bellemare et al., 2016 — [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)
