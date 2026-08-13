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
algorithmic difference confounds the comparison. Every rung is built by `rnd_convergence.envs.make_env`,
which applies the per-rung treatment needed to make the *Dim* column true: MarsRover has no Gymnasium
id and is implemented in `rnd_convergence/mars_rover.py`, and its integer position is logged as 1-D
rather than as the 5-wide one-hot the policy consumes; MiniGrid's default observation is a `Dict` of a
7×7×3 image and a mission string, which `MiniGridStateWrapper` replaces with the underlying
`(x, y, dir)`. A test asserts the table's dimensions against what `make_env` actually produces.

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
   demonstrated. Reported alongside **grid occupancy** (occupied cells ÷ samples) and swept over the
   clip range, for the reason below.
3. **SVE, kNN (Kozachenko–Leonenko)** — differential entropy estimated over a buffer of visited
   states, without discretization. Differential entropy is not scale-invariant and may be negative,
   so it is compared only *within* an environment.

## Measurement

### Convergence ground truth (`t_conv`)

Computed offline over the completed run; it deliberately uses information from the full training
curve, because it is the target the online signals must approximate.

- Evaluate every `5_000` environment steps, `10` episodes, greedy action selection → curve `R(t)`.
  Evaluation can only run on a rollout boundary, so each point is timestamped with the step it
  actually ran at rather than the grid point that triggered it: the gap is up to one rollout and
  always in the same direction, which would otherwise be a systematic offset in every `Δ`.
- `R_ref` = mean of the final 5 evaluations; `R_0` = mean return of a random policy in that
  environment.
- `t_conv` = first `t` at which smoothed `R` stays ≥ `R_0 + 0.95 · (R_ref − R_0)` for **5 consecutive
  evaluations**.

Normalising against the random-policy baseline rather than a percentage of the final value keeps the
criterion well-defined for negative returns and for 0–1 sparse-reward ranges alike.

`t_conv` is undefined in two opposite cases, which are recorded separately and never pooled: the
agent never beat a random policy (`not_learned` — the expected outcome on DoorKey-8x8), or it was
still improving when the budget ran out (`still_improving` — a statement about the budget). Dropping
both from a cell's mean would bias it towards the fastest-converging seeds, so the rates are reported
with the means.

### Signal plateau (`t_plateau`)

Identical functional form, so the comparison is fair across signals: the signal is smoothed with a
trailing mean, and `t_plateau` is the first `t` at which the change between consecutive points falls
below `τ` times the **median change seen so far** and stays there for **5 consecutive windows**.
Applied unchanged to RND error and to every SVE variant, on observations standardised by the same
running normaliser the RND replay uses — "equal footing" covers preprocessing, not just the detector.

Normalising against the signal's own characteristic rate of change, rather than against its magnitude
or its accumulated range, is what makes one `τ` valid across signals. It is dimensionless, it is
defined for the negative values that differential entropy routinely takes (a log-derivative is not),
and a signal still descending at a constant rate never satisfies it — whereas normalising by
accumulated range would eventually declare a steady decline "flat" merely because it had already
travelled a long way, manufacturing exactly the false early stop the study sets out to measure.

The reference is the running median, not the running maximum. A maximum is set by the single largest
change anywhere in the run, so one transient spike — exactly what RND error does on reaching a new
region — relaxes the threshold for every later point; on synthetic curves that moves `t_plateau` by
tens of thousands of steps, always earlier. `τ` is bounded below by the signal's noise floor: below
roughly `τ = 0.1` the criterion stops firing at all on noisy signals, and "never fired" is
indistinguishable from "never plateaued".

Two settings, not one, therefore have to be swept and reported: **`τ` × `smooth_window`**. On a pilot
100k-step CartPole run, RND's `Δ` moved 13 616 → 33 616 → 48 616 purely from `smooth_window` at fixed
`τ`. Since `Δ` is the number the project reports, the detector settings are pinned per signal and
stated with the table, with `reference_quantile = 1.0` (the maximum) as a sensitivity check.

**Curve resolution.** The detector needs at least `min_points + patience` curve points — 10 at the
defaults — because differencing costs one point and the leading `min_points − 1` are suppressed. A
100k-step run measured at `window = 10_000` yields exactly 10 and can only ever answer at the final
point; `plateau_time` raises rather than returning `None` below the bound, so this cannot be mistaken
for "no plateau". Runs are therefore sized so that `total_steps / window ≥ 40`.

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
3. **SVE-plateau feasibility.** On a pilot 100k-step CartPole run the grid SVE curve did not plateau
   in any of 18 detector configurations (window ∈ {5k, 10k} × `smooth_window` ∈ {1, 5, 9} × `τ` ∈
   {0.1, 0.3, 0.5}), nor did the kNN variant; cumulative mode fired in only 2 of 9. Occupancy peaked
   at 0.077, so this is *not* the saturation case below — the curve simply wobbles at an amplitude
   comparable to its own drift, which is the noise-floor limit. That is a legitimate finding about
   the baseline, but it means the SVE column may be empty on a rung where SVE is supposed to still
   work. It is settled on a pilot run per rung *before* the full matrix is launched, since an empty
   baseline column is not something to discover after 50 cluster runs.
4. **Grid-saturation check.** Once cells outnumber samples badly enough that nearly every sample owns
   a cell, the histogram is uniform over `N` cells and grid SVE returns `log N` *identically*,
   whatever the policy does — at 8-D with 20 bins and a 5 000-step window it is already within `1e-3`
   of `log 5000`. A flat SVE curve there is arithmetic about sample counts, not evidence about
   exploration. Occupancy is therefore reported in every cell of the table so that measurements and
   ceilings are distinguishable, and Miller–Madow bias correction is available as a sensitivity
   check. This is a real limit of the baseline rather than a bug, but the study only earns the claim
   "SVE degrades with dimensionality" if it can show the degradation is not just the estimator
   running out of samples.

## Experimental Protocol

RND is used here as a monitoring signal, not as an intrinsic reward, so the predictor is not part of
the training loop:

1. Training logs the visited-state stream — step index, observation, episode boundary — to disk.
2. All signals are computed post-hoc. The RND predictor is trained in a single streaming pass over
   the logged states **in visit order**, matching what an online predictor consuming the same state
   sequence would have produced; SVE variants are computed over the same stream. (RND is never run
   online here, so there is no recorded signal to be identical to — the replay is faithful to the
   visit order and the update rule, not to a measurement that was taken.)

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
docstring records it. `mars_rover.py` follows the *specification* of the week 2 environment, which
that repository provides complete rather than as a stub; the implementation here is our own and the
exercise repo is not a dependency.

- Burda et al., 2018 — [Exploration by Random Network Distillation](https://arxiv.org/abs/1810.12894)
- Schulman et al., 2017 — [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347)
- Bellemare et al., 2016 — [Unifying Count-Based Exploration and Intrinsic Motivation](https://arxiv.org/abs/1606.01868)
