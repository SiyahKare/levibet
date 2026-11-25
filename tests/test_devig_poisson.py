import numpy as np
import pytest

from app.services.bayesian import poisson
from app.services.odds.devig import devig, devig_power, devig_proportional, edge, overround


def test_overround_and_sums():
    odds = [2.0, 3.4, 4.0]
    assert overround(odds) == pytest.approx(0.5 + 1 / 3.4 + 0.25 - 1)
    for fn in (devig_power, devig_proportional):
        assert sum(fn(odds)) == pytest.approx(1.0)


def test_power_shrinks_longshots_more():
    odds = [1.5, 4.0, 8.0]
    prop, pw = devig_proportional(odds), devig_power(odds)
    assert pw[2] < prop[2] and pw[0] > prop[0]


def test_fair_odds_unchanged():
    assert devig([2.0, 4.0, 4.0]) == pytest.approx([0.5, 0.25, 0.25])


def test_bad_odds():
    with pytest.raises(ValueError):
        devig([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        devig([2.0])


def test_edge():
    assert edge(0.5, 2.2) == pytest.approx(0.1)


def test_score_matrix_and_markets():
    mat = poisson.score_matrix([1.5, 1.0], [1.1, 1.0])
    assert mat.sum(axis=(1, 2)) == pytest.approx([1, 1])
    p = poisson.outcome_probs(mat)
    assert p.sum(axis=1) == pytest.approx([1, 1])
    assert p[0, 0] > p[0, 2]  # stronger home side
    # equal rates -> symmetric
    assert p[1, 0] == pytest.approx(p[1, 2])
    assert np.all(poisson.over_prob(mat) < 1) and np.all(poisson.btts_prob(mat) < 1)
