"""Random Network Distillation networks and offline replay of the prediction-error signal.

The network architecture (a frozen randomly-initialised target MLP and a trainable
predictor MLP of identical shape) follows the ``week_7/rnd_utils.py`` scaffold of the
course exercise repo (automl-edu/RL-exercises).

Departure from the usual RND setup: the prediction error is *not* consumed as an
intrinsic reward here, so the predictor never has to run inside the training loop.
:func:`streaming_rnd_error` replays a logged :class:`~rnd_convergence.streams.StateStream`
in visit order, taking exactly one gradient step per minibatch of consecutive states. That
matches what an online predictor consuming the same state sequence under the same update
rule would have produced, while remaining fully deterministic and re-runnable for any
hyperparameter choice. It is not a reconstruction of a measurement that was actually
taken: RND never runs inside the training loop in this project, so there is no recorded
online signal for the replay to be identical to.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from rnd_convergence.streams import StateStream
from rnd_convergence.windows import iter_windows


class RunningNormalizer:
    """Running mean/std normaliser over observations (Chan's parallel variance update).

    Burda et al. (2018) note that RND is highly sensitive to observation scale: without
    normalisation the prediction error tracks drift in the magnitude of the observations
    rather than their novelty. Statistics are updated from the incoming batch *before*
    that batch is normalised, mirroring the online setting where only past data is
    available.
    """

    def __init__(self, n_features: int, epsilon: float = 1e-8) -> None:
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.var = np.ones(n_features, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon

    def update(self, batch: np.ndarray) -> None:
        """Fold a batch of observations, shape ``(B, D)``, into the running statistics."""
        batch_count = batch.shape[0]
        if batch_count == 0:
            return
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)

        total = self.count + batch_count
        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * (self.count * batch_count / total)

        self.mean = new_mean
        self.var = m2 / total
        self.count = total

    def normalize(self, batch: np.ndarray, clip: float | None = 5.0) -> np.ndarray:
        """Standardise a batch with the current statistics, optionally clipping outliers.

        ``clip=None`` leaves outliers in place. The kNN entropy estimator needs that:
        clipping collapses every outlier onto the boundary, and coincident points give
        zero nearest-neighbour distances, which drags the estimate towards minus
        infinity. The grid estimator does its own clipping, since its bin edges have to
        be finite.
        """
        normalized = (batch - self.mean) / np.sqrt(self.var + self.epsilon)
        return normalized if clip is None else np.clip(normalized, -clip, clip)


def _build_mlp(input_dim: int, hidden_dim: int, output_dim: int, n_layers: int) -> nn.Sequential:
    """``input -> [Linear, ReLU] * n_layers -> Linear`` stack shared by both networks."""
    if n_layers < 1:
        raise ValueError(f"n_layers must be >= 1, got {n_layers}")
    layers: list[nn.Module] = []
    in_dim = input_dim
    for _ in range(n_layers):
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, output_dim))
    return nn.Sequential(*layers)


class TargetNetwork(nn.Module):
    """Randomly initialised MLP, frozen for the whole run.

    Its output is an arbitrary but *fixed* function of the state. Novelty is measured as
    how badly the predictor reproduces it.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dim, output_dim, n_layers)
        for param in self.parameters():
            param.requires_grad_(False)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map states to the fixed random embedding."""
        return self.net(x)


class PredictorNetwork(nn.Module):
    """Trainable MLP of identical shape to :class:`TargetNetwork`.

    Trained to regress onto the target's output on visited states; the residual is the
    RND prediction error.
    """

    def __init__(
        self, input_dim: int, hidden_dim: int = 128, output_dim: int = 64, n_layers: int = 2
    ) -> None:
        super().__init__()
        self.net = _build_mlp(input_dim, hidden_dim, output_dim, n_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Map states to the predicted embedding."""
        return self.net(x)


def streaming_rnd_error(
    stream: StateStream,
    *,
    hidden_dim: int = 128,
    output_dim: int = 64,
    n_layers: int = 2,
    lr: float = 1e-4,
    batch_size: int = 128,
    window: int = 5_000,
    stride: int | None = None,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Replay a state stream and return the RND prediction-error curve.

    States are consumed in visit order in minibatches of consecutive steps. For each
    minibatch the per-state error is recorded *before* the gradient step, so the value
    reflects the novelty of those states at the moment they were visited, and one Adam
    step is then taken on the same batch.

    Parameters
    ----------
    stream
        The logged run to replay.
    hidden_dim, output_dim, n_layers
        Architecture shared by target and predictor.
    lr
        Predictor learning rate.
    batch_size
        Number of consecutive states per gradient step.
    window
        Width, in environment steps, of the window the per-state errors are averaged
        over. Sets how much data each curve point summarises.
    stride
        Spacing of curve points in environment steps; defaults to ``window``, which makes
        consecutive windows adjacent and non-overlapping.
    seed
        Seeds target initialisation and predictor initialisation, so a replay is
        reproducible.
    device
        Torch device. These networks are small; CPU is normally the right choice.

    Returns
    -------
    steps, errors
        ``steps`` holds the upper edge of each window in environment steps; ``errors``
        the mean squared prediction error of the states falling in it.

    The curve is laid out by :func:`~rnd_convergence.windows.iter_windows`, the same
    function :func:`~rnd_convergence.entropy.entropy_curve` uses, so passing both the same
    ``window`` and ``stride`` puts them on an identical step axis. That matters because the
    study compares ``t_plateau`` between the two signals: measuring one on overlapping
    windows and the other on disjoint buckets would put part of the difference between them
    down to how each curve was built. Averaging per-state errors over a sliding window costs
    nothing extra here --- unlike the entropy estimators, the expensive part (the replay)
    has already happened by this point and does not depend on the grid.

    There is deliberately no ``mode`` parameter: this curve is always sliding. A cumulative
    one would average every per-state error since the start of the run, so it would stay
    dominated by the high-novelty beginning long after the current policy had stopped
    finding anything --- a lagging integral of the quantity actually being asked about, which
    is whether novelty is exhausted *now*. It also barely fires: on synthetic decay-then-flat
    signals :func:`~rnd_convergence.convergence.plateau_time` locates the plateau on the
    sliding curve and returns ``None`` on the cumulative transform of the very same signal.
    Since :func:`~rnd_convergence.windows.iter_windows` takes its points from ``stride``
    alone, a cumulative SVE curve nevertheless shares this curve's step axis --- so equal
    steps must not be read as licence to compare the two ``t_plateau`` values directly.
    """
    if stream.n_steps == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)

    torch.manual_seed(seed)
    target = TargetNetwork(stream.obs_dim, hidden_dim, output_dim, n_layers).to(device)
    predictor = PredictorNetwork(stream.obs_dim, hidden_dim, output_dim, n_layers).to(device)
    optimizer = torch.optim.Adam(predictor.parameters(), lr=lr)
    normalizer = RunningNormalizer(stream.obs_dim)

    observations = stream.observations.astype(np.float64)
    per_state_error = np.empty(stream.n_steps, dtype=np.float64)

    for start in range(0, stream.n_steps, batch_size):
        end = min(start + batch_size, stream.n_steps)
        raw = observations[start:end]
        normalizer.update(raw)
        batch = torch.as_tensor(normalizer.normalize(raw), dtype=torch.float32, device=device)

        with torch.no_grad():
            target_out = target(batch)
        predicted = predictor(batch)
        squared_error = (predicted - target_out).pow(2).mean(dim=1)
        per_state_error[start:end] = squared_error.detach().cpu().numpy()

        optimizer.zero_grad(set_to_none=True)
        squared_error.mean().backward()
        optimizer.step()

    grid = list(iter_windows(stream.steps, mode="sliding", window=window, stride=stride))
    steps = np.asarray([point for point, _, _ in grid], dtype=np.int64)
    errors = np.asarray([per_state_error[start:end].mean() for _, start, end in grid])
    return steps, errors
