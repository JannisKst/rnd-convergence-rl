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
        Width, in environment steps, of the buckets the per-state errors are averaged
        into. Determines the resolution of the returned curve.
    seed
        Seeds target initialisation and predictor initialisation, so a replay is
        reproducible.
    device
        Torch device. These networks are small; CPU is normally the right choice.

    Returns
    -------
    steps, errors
        ``steps`` holds the upper edge of each window in environment steps; ``errors``
        the mean squared prediction error of the states falling in it. Windows
        containing no states are dropped.
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

    return _bucket_by_step(stream.steps, per_state_error, window)


def _bucket_by_step(
    steps: np.ndarray, values: np.ndarray, window: int
) -> tuple[np.ndarray, np.ndarray]:
    """Average ``values`` into fixed-width buckets of the environment-step axis.

    Returns the upper edge of each non-empty bucket and the mean of the values in it.
    """
    if window < 1:
        raise ValueError(f"window must be >= 1, got {window}")
    bucket = steps // window
    unique, inverse = np.unique(bucket, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=unique.size)
    counts = np.bincount(inverse, minlength=unique.size)
    return (unique + 1) * window, sums / counts
