"""Tests for ferro_ta.analysis.overfit (PSR, DSR, PBO, purged k-fold).

Rust results are cross-checked against independent NumPy/SciPy reference
implementations of the published formulas.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from ferro_ta.analysis.overfit import (
    PBOResult,
    deflated_sharpe_ratio,
    expected_max_sharpe,
    pbo,
    per_period_sharpe,
    probabilistic_sharpe_ratio,
    purged_kfold,
    purged_kfold_indices,
)
from ferro_ta.core.exceptions import FerroTAInputError, FerroTAValueError

EULER_GAMMA = 0.5772156649015329


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _ref_psr(r: np.ndarray, sr_star: float) -> float:
    stats = pytest.importorskip("scipy.stats")
    sr = r.mean() / r.std(ddof=1)
    skew = stats.skew(r)
    kurt = stats.kurtosis(r, fisher=False)
    denom = math.sqrt(1 - skew * sr + (kurt - 1) / 4 * sr**2)
    return float(stats.norm.cdf((sr - sr_star) * math.sqrt(len(r) - 1) / denom))


def _ref_expected_max(n: int, var: float) -> float:
    stats = pytest.importorskip("scipy.stats")
    z1 = stats.norm.ppf(1 - 1 / n)
    z2 = stats.norm.ppf(1 - 1 / (n * math.e))
    return math.sqrt(var) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def _ref_pbo(m: np.ndarray, s: int) -> float:
    n_bars, n = m.shape
    blocks = [m[b * n_bars // s : (b + 1) * n_bars // s] for b in range(s)]
    overfit = total = 0
    for is_ids in itertools.combinations(range(s), s // 2):
        is_rows = np.vstack([blocks[b] for b in is_ids])
        oos_rows = np.vstack([blocks[b] for b in range(s) if b not in is_ids])
        best = int(np.argmax(per_period_sharpe(is_rows)))
        oos = per_period_sharpe(oos_rows)
        less = np.sum(oos < oos[best])
        ties = np.sum(oos == oos[best]) - 1
        omega = (less + 0.5 * ties + 1) / (n + 1)
        overfit += math.log(omega / (1 - omega)) <= 0
        total += 1
    return overfit / total


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ---------------------------------------------------------------------------
# PSR / DSR
# ---------------------------------------------------------------------------


class TestProbabilisticSharpe:
    def test_matches_reference_on_skewed_returns(self, rng):
        r = rng.lognormal(0, 0.5, 800) - 1.05
        for sr_star in (0.0, 0.05, -0.1):
            got = probabilistic_sharpe_ratio(r, sr_star)
            assert got == pytest.approx(_ref_psr(r, sr_star), abs=2e-7)

    def test_half_when_benchmark_equals_sample_sharpe(self, rng):
        r = rng.normal(0.001, 0.01, 500)
        sr = r.mean() / r.std(ddof=1)
        assert probabilistic_sharpe_ratio(r, sr) == pytest.approx(0.5, abs=1e-7)

    def test_rejects_short_constant_or_nan_input(self):
        with pytest.raises(FerroTAValueError):
            probabilistic_sharpe_ratio([0.1, 0.2])
        with pytest.raises(FerroTAValueError):
            probabilistic_sharpe_ratio([0.1, 0.1, 0.1, 0.1])
        with pytest.raises(FerroTAValueError):
            probabilistic_sharpe_ratio([0.1, np.nan, 0.2, 0.3])

    def test_rejects_2d_input(self):
        with pytest.raises(FerroTAInputError):
            probabilistic_sharpe_ratio(np.ones((3, 3)))


class TestDeflatedSharpe:
    def test_expected_max_matches_reference(self):
        for n, var in ((2, 0.01), (10, 1.0), (1000, 0.25)):
            assert expected_max_sharpe(n, var) == pytest.approx(
                _ref_expected_max(n, var), rel=1e-7
            )

    def test_single_trial_equals_psr(self, rng):
        r = rng.normal(0.001, 0.01, 400)
        assert deflated_sharpe_ratio(
            r, n_trials=1, trial_sharpe_variance=0.3
        ) == pytest.approx(probabilistic_sharpe_ratio(r, 0.0))

    def test_trial_sharpes_path_matches_explicit_path(self, rng):
        trials = rng.normal(0.0, 0.01, (600, 40))
        sharpes = per_period_sharpe(trials)
        best = trials[:, int(np.argmax(sharpes))]
        a = deflated_sharpe_ratio(best, trial_sharpes=sharpes)
        b = deflated_sharpe_ratio(
            best, n_trials=40, trial_sharpe_variance=float(sharpes.var(ddof=1))
        )
        assert a == pytest.approx(b)
        assert a == pytest.approx(
            _ref_psr(best, _ref_expected_max(40, float(sharpes.var(ddof=1)))),
            abs=2e-7,
        )

    def test_best_of_noise_is_deflated_to_a_coin_flip(self):
        psrs, dsrs = [], []
        for seed in range(10):
            trials = np.random.default_rng(seed).normal(0.0, 0.01, (1000, 200))
            sharpes = per_period_sharpe(trials)
            best = trials[:, int(np.argmax(sharpes))]
            psrs.append(probabilistic_sharpe_ratio(best))
            dsrs.append(deflated_sharpe_ratio(best, trial_sharpes=sharpes))
        assert min(psrs) > 0.9  # naive PSR is fooled by selection
        assert 0.3 < np.mean(dsrs) < 0.7  # DSR is not

    def test_argument_validation(self, rng):
        r = rng.normal(0, 1, 50)
        with pytest.raises(FerroTAInputError):
            deflated_sharpe_ratio(r)
        with pytest.raises(FerroTAInputError):
            deflated_sharpe_ratio(r, trial_sharpes=[0.1], n_trials=3)
        with pytest.raises(FerroTAInputError):
            deflated_sharpe_ratio(r, trial_sharpes=[0.1, np.nan])
        with pytest.raises(FerroTAValueError):
            deflated_sharpe_ratio(r, n_trials=0, trial_sharpe_variance=0.1)
        with pytest.raises(FerroTAValueError):
            expected_max_sharpe(5, -1.0)


# ---------------------------------------------------------------------------
# PBO
# ---------------------------------------------------------------------------


class TestPBO:
    def test_matches_bruteforce_reference(self, rng):
        m = rng.normal(0.0, 0.01, (240, 12))
        m[:, 4] += 0.001  # a weak but real edge
        for s in (4, 6, 8):
            assert pbo(m, n_splits=s).pbo == pytest.approx(_ref_pbo(m, s))

    def test_result_shape_and_fields(self, rng):
        res = pbo(rng.normal(0, 0.01, (400, 20)), n_splits=8)
        assert isinstance(res, PBOResult)
        assert res.n_combinations == math.comb(8, 4)
        for arr in (res.logits, res.is_sharpe, res.oos_sharpe, res.best_index):
            assert arr.shape == (res.n_combinations,)
        assert 0.0 <= res.prob_oos_loss <= 1.0
        assert np.all(res.is_sharpe >= res.oos_sharpe.min())
        with pytest.raises(ValueError):
            res.logits[0] = 0.0

    def test_dominant_strategy_never_overfits(self, rng):
        m = rng.normal(0.0, 0.01, (500, 15))
        m[:, 7] += 0.01
        res = pbo(m, n_splits=10)
        assert res.pbo == 0.0
        assert np.all(res.best_index == 7)

    def test_noise_is_coin_flip_on_average(self):
        values = [
            pbo(np.random.default_rng(seed).normal(0, 0.01, (1000, 50)), 10).pbo
            for seed in range(10)
        ]
        assert 0.35 < np.mean(values) < 0.65

    def test_accepts_non_contiguous_input(self, rng):
        m = rng.normal(0, 0.01, (12, 200)).T  # Fortran-ordered view
        assert pbo(m, n_splits=4).pbo == pytest.approx(_ref_pbo(np.array(m), 4))

    @pytest.mark.parametrize("splits", [3, 0, 22])
    def test_rejects_bad_split_counts(self, rng, splits):
        with pytest.raises(FerroTAValueError):
            pbo(rng.normal(0, 0.01, (400, 5)), n_splits=splits)

    def test_rejects_bad_shapes(self, rng):
        with pytest.raises(FerroTAInputError):
            pbo(rng.normal(0, 0.01, 100))
        with pytest.raises(FerroTAValueError):
            pbo(rng.normal(0, 0.01, (100, 1)), n_splits=4)
        with pytest.raises(FerroTAValueError):
            pbo(rng.normal(0, 0.01, (10, 4)), n_splits=8)


# ---------------------------------------------------------------------------
# Purged k-fold
# ---------------------------------------------------------------------------


class TestPurgedKFold:
    def test_boundaries(self):
        b = purged_kfold_indices(100, 4, purge_bars=3, embargo_bars=2)
        assert b.dtype == np.int64
        assert b.tolist() == [
            [0, 0, 25, 27],
            [22, 25, 50, 52],
            [47, 50, 75, 77],
            [72, 75, 100, 100],
        ]

    def test_folds_cover_everything_and_never_leak(self):
        n, purge, embargo = 103, 4, 3
        folds = purged_kfold(n, 5, purge_bars=purge, embargo_bars=embargo)
        tested = np.concatenate([test for _, test in folds])
        assert np.array_equal(np.sort(tested), np.arange(n))
        for train, test in folds:
            assert np.intersect1d(train, test).size == 0
            lo, hi = test.min(), test.max()
            gap = train[(train >= lo - purge) & (train <= hi + embargo)]
            assert gap.size == 0

    def test_no_gaps_is_plain_kfold(self):
        for train, test in purged_kfold(10, 5):
            assert len(train) + len(test) == 10

    def test_validation(self):
        with pytest.raises(FerroTAValueError):
            purged_kfold_indices(10, 1)
        with pytest.raises(FerroTAValueError):
            purged_kfold_indices(3, 5)
        with pytest.raises(FerroTAValueError):
            purged_kfold_indices(10, 2, purge_bars=-1)
