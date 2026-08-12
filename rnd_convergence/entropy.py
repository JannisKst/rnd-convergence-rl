"""State-visitation entropy estimators — the baseline RND is compared against.

Two estimators, because the discretization question is the point of the study rather
than an implementation detail:

``grid_entropy``
    Shannon entropy of a visitation histogram over a regular grid. Exact on discrete
    state spaces; increasingly meaningless as dimensionality grows and bins outnumber
    visited states. The bin count is swept rather than fixed.

``knn_entropy``
    Kozachenko-Leonenko differential entropy, which needs no discretization but is not
    scale-invariant, may be negative, and is undefined for atomic (discrete)
    distributions — so it is only compared *within* an environment.

Both are wrapped by :func:`entropy_curve`, which turns a logged state stream into a
signal over training steps that the plateau detector can consume, exactly like the RND
error curve.

Observations are standardised using statistics accumulated *up to* the current point in
the run, never the whole run. That keeps the entropy signal causal in the same way the
RND signal is, so the two are compared on equal footing.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from rnd_convergence.streams import StateStream

Estimator = Literal["grid", "knn"]
CurveMode = Literal["sliding", "cumulative"]

_MAX_SAFE_CELLS = 2**62


def grid_entropy(observations: np.ndarray, bins_per_dim: int, clip: float = 5.0) -> float:
    """Shannon entropy (nats) of the visitation histogram over a regular grid.

    Observations are assumed already standardised; the grid spans ``[-clip, clip]`` in
    every dimension, with out-of-range values folded into the edge bins.

    Only occupied cells are materialised, so this stays tractable when ``bins_per_dim **
    n_dims`` is astronomically larger than the number of visited states — which is
    precisely the regime the study is about.
    """
    if bins_per_dim < 2:
        raise ValueError(f"bins_per_dim must be >= 2, got {bins_per_dim}")
    if observations.ndim != 2:
        raise ValueError(f"observations must be 2-D (N, D), got shape {observations.shape}")
    n_samples, n_dims = observations.shape
    if n_samples == 0:
        return 0.0

    scaled = (np.clip(observations, -clip, clip) + clip) / (2 * clip)
    indices = np.minimum((scaled * bins_per_dim).astype(np.int64), bins_per_dim - 1)

    if bins_per_dim**n_dims < _MAX_SAFE_CELLS:
        flat = np.ravel_multi_index(tuple(indices.T), (bins_per_dim,) * n_dims)
        counts = np.unique(flat, return_counts=True)[1]
    else:
        counts = np.unique(indices, axis=0, return_counts=True)[1]

    probabilities = counts / n_samples
    return float(-np.sum(probabilities * np.log(probabilities)))


def knn_entropy(
    observations: np.ndarray,
    k: int = 4,
    *,
    max_samples: int = 4_000,
    seed: int = 0,
    epsilon: float = 1e-10,
) -> float:
    """Kozachenko-Leonenko differential entropy estimate (nats).

    ``H = psi(N) - psi(k) + log(V_d) + (d / N) * sum_i log(r_i)``, where ``r_i`` is the
    distance from sample ``i`` to its ``k``-th nearest neighbour and ``V_d`` the volume
    of the unit ``d``-ball.

    Cost is ``O(N log N)`` via a k-d tree, but the constant matters over a full run, so
    inputs longer than ``max_samples`` are randomly subsampled with a fixed seed.

    Duplicate states give ``r_i = 0`` and a divergent estimate. Distances are floored at
    ``epsilon`` to keep the value finite, but note what this means: on genuinely discrete
    state spaces (MarsRover, MiniGrid) differential entropy is not the right object at
    all, and the exact grid estimator should be used there instead.
    """
    if observations.ndim != 2:
        raise ValueError(f"observations must be 2-D (N, D), got shape {observations.shape}")
    n_samples, n_dims = observations.shape
    if n_samples <= k:
        return float("nan")

    if n_samples > max_samples:
        rng = np.random.default_rng(seed)
        observations = observations[rng.choice(n_samples, max_samples, replace=False)]
        n_samples = max_samples

    tree = cKDTree(observations)
    # The first neighbour of a point is the point itself, hence k + 1.
    distances = tree.query(observations, k=k + 1)[0][:, -1]
    distances = np.maximum(distances, epsilon)

    log_unit_ball = (n_dims / 2) * np.log(np.pi) - gammaln(n_dims / 2 + 1)
    return float(
        digamma(n_samples)
        - digamma(k)
        + log_unit_ball
        + (n_dims / n_samples) * np.sum(np.log(distances))
    )


def entropy_curve(
    stream: StateStream,
    *,
    estimator: Estimator = "grid",
    mode: CurveMode = "sliding",
    window: int = 5_000,
    stride: int | None = None,
    bins_per_dim: int = 10,
    k: int = 4,
    max_samples: int = 4_000,
    clip: float = 5.0,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Turn a logged state stream into a state-visitation-entropy curve over training.

    Parameters
    ----------
    stream
        The logged run to analyse.
    estimator
        ``"grid"`` for the histogram estimator, ``"knn"`` for Kozachenko-Leonenko.
    mode
        ``"sliding"`` measures the spread of the states visited in the last ``window``
        environment steps — the entropy of the current policy's state distribution.
        ``"cumulative"`` measures the spread of everything seen so far, which is the
        closer analogue of "has the agent stopped finding new regions".
    window
        Width in environment steps of the sliding window (ignored when cumulative,
        except that it still sets the evaluation grid).
    stride
        Spacing of evaluation points in environment steps; defaults to ``window``.
    bins_per_dim
        Grid resolution, swept across ``{5, 10, 20}`` in the experiments.
    k, max_samples
        Passed through to :func:`knn_entropy`.
    clip
        Standardised observations are clipped to ``[-clip, clip]``.
    seed
        Seeds subsampling inside the kNN estimator.

    Returns
    -------
    steps, values
        Evaluation points in environment steps and the entropy estimated at each.
    """
    if stream.n_steps == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    stride = window if stride is None else stride
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    observations = stream.observations.astype(np.float64)
    steps = stream.steps
    means, stds = _causal_moments(observations)

    last_step = int(steps[-1])
    eval_points = np.arange(stride, last_step + stride, stride, dtype=np.int64)

    out_steps: list[int] = []
    out_values: list[float] = []
    for point in eval_points:
        end = int(np.searchsorted(steps, point, side="right"))
        if end == 0:
            continue
        start = (
            int(np.searchsorted(steps, point - window, side="right")) if mode == "sliding" else 0
        )
        if end - start < 2:
            continue

        # Standardise with statistics available at this point in the run, not the whole run.
        normalized = (observations[start:end] - means[end - 1]) / stds[end - 1]
        if estimator == "grid":
            value = grid_entropy(normalized, bins_per_dim=bins_per_dim, clip=clip)
        elif estimator == "knn":
            value = knn_entropy(normalized, k=k, max_samples=max_samples, seed=seed)
        else:
            raise ValueError(f"unknown estimator {estimator!r}, expected 'grid' or 'knn'")

        out_steps.append(int(point))
        out_values.append(value)

    return np.asarray(out_steps, dtype=np.int64), np.asarray(out_values, dtype=np.float64)


def _causal_moments(
    observations: np.ndarray, epsilon: float = 1e-8
) -> tuple[np.ndarray, np.ndarray]:
    """Prefix mean and std of ``observations``, i.e. row ``i`` covers ``[0 .. i]``."""
    counts = np.arange(1, observations.shape[0] + 1, dtype=np.float64)[:, None]
    means = np.cumsum(observations, axis=0) / counts
    mean_squares = np.cumsum(observations**2, axis=0) / counts
    variances = np.maximum(mean_squares - means**2, 0.0)
    return means, np.sqrt(variances + epsilon)
