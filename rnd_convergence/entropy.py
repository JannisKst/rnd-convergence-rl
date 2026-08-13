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

Observations are standardised with :class:`~rnd_convergence.rnd.RunningNormalizer`, the
same class the RND replay uses, updated from the states seen up to the current point in
the run and never from the whole run. Sharing the implementation is deliberate: the study
claims the two signals are compared on equal footing, and that claim covers preprocessing,
not just the plateau detector.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Literal

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln

from rnd_convergence.rnd import RunningNormalizer
from rnd_convergence.streams import StateStream

Estimator = Literal["grid", "knn"]
CurveMode = Literal["sliding", "cumulative"]
Correction = Literal["none", "miller_madow"]

_MAX_SAFE_CELLS = 2**62


def _grid_counts(observations: np.ndarray, bins_per_dim: int, clip: float) -> np.ndarray:
    """Occupancy counts of the non-empty cells of the regular grid.

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
        return np.empty(0, dtype=np.int64)

    scaled = (np.clip(observations, -clip, clip) + clip) / (2 * clip)
    indices = np.minimum((scaled * bins_per_dim).astype(np.int64), bins_per_dim - 1)

    if bins_per_dim**n_dims < _MAX_SAFE_CELLS:
        flat = np.ravel_multi_index(tuple(indices.T), (bins_per_dim,) * n_dims)
        return np.unique(flat, return_counts=True)[1]
    return np.unique(indices, axis=0, return_counts=True)[1]


def grid_entropy(
    observations: np.ndarray,
    bins_per_dim: int,
    clip: float = 5.0,
    correction: Correction = "none",
) -> float:
    """Shannon entropy (nats) of the visitation histogram over a regular grid.

    Observations are assumed already standardised; the grid spans ``[-clip, clip]`` in
    every dimension, with out-of-range values folded into the edge bins. ``clip`` is a
    sweep parameter, not a constant: the fraction of samples pushed into the edge bins
    grows with dimensionality, so leaving it fixed would let clipping masquerade as the
    binning degradation the study is trying to measure.

    ``correction="miller_madow"`` adds the ``(K - 1) / (2N)`` bias correction for the
    ``K`` occupied cells. It compensates the plug-in estimator's downward bias when cells
    are undersampled, but note that it does *not* rescue the saturated regime — see
    :func:`grid_occupancy` for detecting that.
    """
    counts = _grid_counts(observations, bins_per_dim, clip)
    n_samples = observations.shape[0]
    if n_samples == 0:
        return 0.0

    probabilities = counts / n_samples
    entropy = float(-np.sum(probabilities * np.log(probabilities)))
    if correction == "miller_madow":
        entropy += (counts.size - 1) / (2 * n_samples)
    elif correction != "none":
        raise ValueError(f"unknown correction {correction!r}, expected 'none' or 'miller_madow'")
    return entropy


def grid_occupancy(observations: np.ndarray, bins_per_dim: int, clip: float = 5.0) -> float:
    """Occupied cells divided by samples — the grid estimator's saturation diagnostic.

    As dimensionality grows, cells come to outnumber samples so heavily that almost every
    sample lands in a cell of its own. The histogram is then uniform over ``N`` cells and
    ``grid_entropy`` returns ``log N`` *identically*, whatever the policy is doing: at 8-D
    with 20 bins per dimension and a 5000-step window the estimate is already within
    ``1e-3`` of ``log 5000``.

    That matters for how the result is read. A flat SVE curve on the bottom rung would
    otherwise look like evidence about exploration when it is arithmetic about sample
    counts. This function returns 1.0 exactly in that regime, so the report can state
    which cells of the table are measurements and which are ceilings.
    """
    counts = _grid_counts(observations, bins_per_dim, clip)
    if observations.shape[0] == 0:
        return 0.0
    return float(counts.size / observations.shape[0])


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
    if estimator not in ("grid", "knn"):
        raise ValueError(f"unknown estimator {estimator!r}, expected 'grid' or 'knn'")

    def measure(normalized: np.ndarray) -> float:
        if estimator == "grid":
            return grid_entropy(normalized, bins_per_dim=bins_per_dim, clip=clip)
        # No clipping is applied for kNN; see RunningNormalizer.normalize.
        return knn_entropy(normalized, k=k, max_samples=max_samples, seed=seed)

    return _windowed_curve(stream, measure, mode=mode, window=window, stride=stride)


def occupancy_curve(
    stream: StateStream,
    *,
    mode: CurveMode = "sliding",
    window: int = 5_000,
    stride: int | None = None,
    bins_per_dim: int = 10,
    clip: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    """:func:`grid_occupancy` over training, on the same grid as :func:`entropy_curve`.

    Report this next to every grid-SVE curve. Where it sits at 1.0 the entropy curve is
    pinned to ``log N`` and carries no information about the policy.
    """
    return _windowed_curve(
        stream,
        lambda normalized: grid_occupancy(normalized, bins_per_dim=bins_per_dim, clip=clip),
        mode=mode,
        window=window,
        stride=stride,
    )


def _windowed_curve(
    stream: StateStream,
    measure: Callable[[np.ndarray], float],
    *,
    mode: CurveMode,
    window: int,
    stride: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply ``measure`` to each evaluation window of causally standardised observations."""
    if stream.n_steps == 0:
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float64)
    stride = window if stride is None else stride
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")

    observations = stream.observations.astype(np.float64)
    normalizer = RunningNormalizer(stream.obs_dim)

    out_steps: list[int] = []
    out_values: list[float] = []
    fed = 0
    for point, start, end in _iter_windows(stream.steps, mode=mode, window=window, stride=stride):
        # Fold in everything newly visible at this point, then standardise with those
        # statistics: causal, and the same update rule the RND replay uses.
        normalizer.update(observations[fed:end])
        fed = end
        out_steps.append(point)
        out_values.append(measure(normalizer.normalize(observations[start:end], clip=None)))

    return np.asarray(out_steps, dtype=np.int64), np.asarray(out_values, dtype=np.float64)


def _iter_windows(
    steps: np.ndarray, *, mode: CurveMode, window: int, stride: int
) -> Iterator[tuple[int, int, int]]:
    """Yield ``(step, start, end)`` index slices for each evaluation point of the curve."""
    if mode not in ("sliding", "cumulative"):
        raise ValueError(f"unknown mode {mode!r}, expected 'sliding' or 'cumulative'")
    last_step = int(steps[-1])
    for point in range(stride, last_step + stride, stride):
        end = int(np.searchsorted(steps, point, side="right"))
        if end == 0:
            continue
        start = (
            int(np.searchsorted(steps, point - window, side="right")) if mode == "sliding" else 0
        )
        if end - start < 2:
            continue
        yield point, start, end
