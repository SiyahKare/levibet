"""Pure functions: scoring rates -> scoreline distribution -> market probabilities.

Everything is vectorised over a leading "draw" axis so posterior samples can be
pushed through in one call.
"""

import numpy as np

MAX_GOALS = 10


def poisson_pmf(lam, max_goals: int = MAX_GOALS) -> np.ndarray:
    """pmf for 0..max_goals goals. `lam` has shape (n,) -> result (n, max_goals+1)."""
    lam = np.atleast_1d(np.asarray(lam, dtype=float))
    pmf = np.empty((lam.shape[0], max_goals + 1))
    pmf[:, 0] = np.exp(-lam)
    for k in range(1, max_goals + 1):
        pmf[:, k] = pmf[:, k - 1] * lam / k
    return pmf


def score_matrix(lam_home, lam_away, max_goals: int = MAX_GOALS) -> np.ndarray:
    """Independent-Poisson scoreline probabilities, shape (n, G+1, G+1), rows = home goals."""
    ph = poisson_pmf(lam_home, max_goals)
    pa = poisson_pmf(lam_away, max_goals)
    mat = ph[:, :, None] * pa[:, None, :]
    # Renormalise the (tiny) mass cut off by max_goals.
    return mat / mat.sum(axis=(1, 2), keepdims=True)


def outcome_probs(mat: np.ndarray) -> np.ndarray:
    """(n, 3) array of [home win, draw, away win]."""
    g = mat.shape[1]
    i, j = np.indices((g, g))
    home = mat[:, i > j].sum(axis=1)
    draw = mat[:, i == j].sum(axis=1)
    away = mat[:, i < j].sum(axis=1)
    return np.stack([home, draw, away], axis=1)


def over_prob(mat: np.ndarray, line: float = 2.5) -> np.ndarray:
    g = mat.shape[1]
    i, j = np.indices((g, g))
    return mat[:, (i + j) > line].sum(axis=1)


def btts_prob(mat: np.ndarray) -> np.ndarray:
    return mat[:, 1:, 1:].sum(axis=(1, 2))
