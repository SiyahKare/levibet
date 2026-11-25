"""Turn bookmaker odds into fair (margin-free) probabilities."""

from typing import Sequence


def _check(odds: Sequence[float]) -> None:
    if len(odds) < 2:
        raise ValueError("need at least two outcomes")
    if any(o is None or o <= 1.0 for o in odds):
        raise ValueError("decimal odds must be greater than 1.0")


def implied_probs(odds: Sequence[float]) -> list[float]:
    """Raw implied probabilities (they sum to more than 1 because of the margin)."""
    _check(odds)
    return [1.0 / o for o in odds]


def overround(odds: Sequence[float]) -> float:
    return sum(implied_probs(odds)) - 1.0


def devig_proportional(odds: Sequence[float]) -> list[float]:
    raw = implied_probs(odds)
    total = sum(raw)
    return [p / total for p in raw]


def devig_power(odds: Sequence[float]) -> list[float]:
    """Power method: find k so that sum(p_i ** k) == 1.

    It shrinks long shots more than favourites, which matches how margins are
    usually distributed (favourite-longshot bias).
    """
    raw = implied_probs(odds)
    lo, hi = 0.05, 10.0
    for _ in range(80):
        mid = (lo + hi) / 2
        if sum(p**mid for p in raw) > 1.0:
            lo = mid
        else:
            hi = mid
    k = (lo + hi) / 2
    probs = [p**k for p in raw]
    total = sum(probs)
    return [p / total for p in probs]


def devig(odds: Sequence[float], method: str = "power") -> list[float]:
    if method == "proportional":
        return devig_proportional(odds)
    if method == "power":
        return devig_power(odds)
    raise ValueError(f"unknown devig method: {method}")


def edge(prob: float, odds: float) -> float:
    """Expected profit per unit stake if `prob` is the true probability."""
    return prob * odds - 1.0
