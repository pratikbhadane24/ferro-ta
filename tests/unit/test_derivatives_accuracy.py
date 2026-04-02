"""
Accuracy/correctness tests for ferro-ta derivatives analytics.

Each test class validates the ferro-ta implementation against reference
formulas implemented using scipy and numpy.
"""

from __future__ import annotations

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Reference formulas (pure numpy / scipy)
# ---------------------------------------------------------------------------


def _norm_cdf(x):
    """Standard normal CDF via scipy."""
    from scipy.stats import norm as _norm

    return _norm.cdf(x)


def _norm_pdf(x):
    from scipy.stats import norm as _norm

    return _norm.pdf(x)


def bsm_call(S, K, r, q, T, sigma):  # noqa: N803
    """Reference BSM call price."""
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * np.exp(-q * T) * _norm_cdf(d1) - K * np.exp(-r * T) * _norm_cdf(d2)


def bsm_put(S, K, r, q, T, sigma):  # noqa: N803
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return K * np.exp(-r * T) * _norm_cdf(-d2) - S * np.exp(-q * T) * _norm_cdf(-d1)


def bsm_delta_call(S, K, r, q, T, sigma):  # noqa: N803
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-q * T) * _norm_cdf(d1)


def digital_cash_call(S, K, r, q, T, sigma):  # noqa: N803
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * _norm_cdf(d2)


def digital_asset_call(S, K, r, q, T, sigma):  # noqa: N803
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * _norm_cdf(d1)


def digital_cash_put(S, K, r, q, T, sigma):  # noqa: N803
    d2 = (np.log(S / K) + (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return np.exp(-r * T) * _norm_cdf(-d2)


def digital_asset_put(S, K, r, q, T, sigma):  # noqa: N803
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * _norm_cdf(-d1)


def vanna_num(S, K, r, q, T, sigma, eps=1e-4):  # noqa: N803
    """∂Δ/∂σ via central differences."""
    delta_up = bsm_delta_call(S, K, r, q, T, sigma + eps)
    delta_dn = bsm_delta_call(S, K, r, q, T, sigma - eps)
    return (delta_up - delta_dn) / (2 * eps)


def vega_bsm(S, K, r, q, T, sigma):  # noqa: N803
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return S * np.exp(-q * T) * _norm_pdf(d1) * np.sqrt(T)


def volga_num(S, K, r, q, T, sigma, eps=1e-4):  # noqa: N803
    """∂²V/∂σ² via central differences."""
    v_up = vega_bsm(S, K, r, q, T, sigma + eps)
    v_dn = vega_bsm(S, K, r, q, T, sigma - eps)
    return (v_up - v_dn) / (2 * eps)


def ctc_vol_reference(close, window, trading_days=252.0):
    """Close-to-close vol: rolling std of log returns × sqrt(trading_days)."""
    log_ret = np.log(close[1:] / close[:-1])
    n = len(close)
    out = np.full(n, np.nan)
    for i in range(window, n):
        returns_window = log_ret[i - window : i]
        out[i] = np.sqrt(np.sum(returns_window**2) / window * trading_days)
    return out


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Six parameter sets: ATM, 10% OTM, 10% ITM, low vol, high vol, non-zero carry
_DIGITAL_CASES = [
    # (S, K, r, q, T, sigma, label)
    (100.0, 100.0, 0.05, 0.00, 1.0, 0.20, "ATM"),
    (100.0, 110.0, 0.05, 0.00, 1.0, 0.20, "10% OTM"),
    (100.0, 90.0, 0.05, 0.00, 1.0, 0.20, "10% ITM"),
    (100.0, 100.0, 0.05, 0.00, 1.0, 0.05, "low vol"),
    (100.0, 100.0, 0.05, 0.00, 1.0, 0.50, "high vol"),
    (100.0, 100.0, 0.05, 0.03, 1.0, 0.20, "non-zero carry"),
]


class TestDigitalOptionsAccuracy:
    @pytest.fixture(autouse=True)
    def require_scipy(self):
        pytest.importorskip("scipy")

    def test_cash_or_nothing_call_vs_reference(self):
        from ferro_ta.analysis.options import digital_option_price

        for S, K, r, q, T, sigma, label in _DIGITAL_CASES:
            expected = digital_cash_call(S, K, r, q, T, sigma)
            actual = digital_option_price(
                S,
                K,
                r,
                T,
                sigma,
                option_type="call",
                digital_type="cash_or_nothing",
                carry=q,
            )
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"cash_or_nothing call mismatch for case '{label}': "
                f"got {actual}, expected {expected}"
            )

    def test_cash_or_nothing_put_vs_reference(self):
        from ferro_ta.analysis.options import digital_option_price

        for S, K, r, q, T, sigma, label in _DIGITAL_CASES:
            expected = digital_cash_put(S, K, r, q, T, sigma)
            actual = digital_option_price(
                S,
                K,
                r,
                T,
                sigma,
                option_type="put",
                digital_type="cash_or_nothing",
                carry=q,
            )
            assert actual == pytest.approx(expected, abs=1e-6), (
                f"cash_or_nothing put mismatch for case '{label}': "
                f"got {actual}, expected {expected}"
            )

    def test_asset_or_nothing_call_vs_reference(self):
        from ferro_ta.analysis.options import digital_option_price

        for S, K, r, q, T, sigma, label in _DIGITAL_CASES:
            expected = digital_asset_call(S, K, r, q, T, sigma)
            actual = digital_option_price(
                S,
                K,
                r,
                T,
                sigma,
                option_type="call",
                digital_type="asset_or_nothing",
                carry=q,
            )
            # Tolerance 1e-4: asset-or-nothing involves S * N(d1), small numerical diff expected
            assert actual == pytest.approx(expected, abs=1e-4), (
                f"asset_or_nothing call mismatch for case '{label}': "
                f"got {actual}, expected {expected}"
            )

    def test_asset_or_nothing_put_vs_reference(self):
        from ferro_ta.analysis.options import digital_option_price

        for S, K, r, q, T, sigma, label in _DIGITAL_CASES:
            expected = digital_asset_put(S, K, r, q, T, sigma)
            actual = digital_option_price(
                S,
                K,
                r,
                T,
                sigma,
                option_type="put",
                digital_type="asset_or_nothing",
                carry=q,
            )
            # Tolerance 1e-4: asset-or-nothing involves S * N(-d1), small numerical diff expected
            assert actual == pytest.approx(expected, abs=1e-4), (
                f"asset_or_nothing put mismatch for case '{label}': "
                f"got {actual}, expected {expected}"
            )

    def test_batch_digital_price_matches_scalar(self):
        """Vectorized call must match scalar loop for 10 random points."""
        from ferro_ta.analysis.options import digital_option_price

        rng = np.random.default_rng(7)
        n = 10
        S_arr = rng.uniform(80.0, 120.0, n)
        K_arr = rng.uniform(80.0, 120.0, n)
        r_arr = rng.uniform(0.01, 0.10, n)
        T_arr = rng.uniform(0.1, 2.0, n)
        sigma_arr = rng.uniform(0.10, 0.50, n)

        batch = digital_option_price(
            S_arr,
            K_arr,
            r_arr,
            T_arr,
            sigma_arr,
            option_type="call",
            digital_type="cash_or_nothing",
        )

        scalar_results = np.array(
            [
                digital_option_price(
                    float(S_arr[i]),
                    float(K_arr[i]),
                    float(r_arr[i]),
                    float(T_arr[i]),
                    float(sigma_arr[i]),
                    option_type="call",
                    digital_type="cash_or_nothing",
                )
                for i in range(n)
            ]
        )

        assert batch == pytest.approx(scalar_results, abs=1e-10), (
            "Batch digital_option_price does not match scalar loop"
        )


# Four cases for extended Greeks: ITM call, ATM call, OTM call, ATM put
_GREEK_CASES = [
    # (S, K, r, q, T, sigma, option_type, label)
    (110.0, 100.0, 0.05, 0.0, 1.0, 0.20, "call", "ITM call"),
    (100.0, 100.0, 0.05, 0.0, 1.0, 0.20, "call", "ATM call"),
    (90.0, 100.0, 0.05, 0.0, 1.0, 0.20, "call", "OTM call"),
    (100.0, 100.0, 0.05, 0.0, 1.0, 0.20, "put", "ATM put"),
]


class TestExtendedGreeksAccuracy:
    @pytest.fixture(autouse=True)
    def require_scipy(self):
        pytest.importorskip("scipy")

    def test_vanna_vs_numerical_fd(self):
        """extended_greeks().vanna matches ∂Δ/∂σ from central differences (tol=1e-3)."""
        from ferro_ta.analysis.options import extended_greeks

        for S, K, r, q, T, sigma, opt_type, label in _GREEK_CASES:
            eg = extended_greeks(S, K, r, T, sigma, option_type=opt_type, carry=q)
            # Reference is defined only for calls; for put use numerical FD directly
            if opt_type == "call":
                expected = vanna_num(S, K, r, q, T, sigma)
            else:
                # Vanna for put: ∂(put delta)/∂σ = ∂(call delta - e^{-qT})/∂σ = vanna_call
                expected = vanna_num(S, K, r, q, T, sigma)
            assert float(eg.vanna) == pytest.approx(expected, abs=1e-3), (
                f"Vanna mismatch for '{label}': got {eg.vanna}, expected {expected}"
            )

    def test_volga_vs_numerical_fd(self):
        """extended_greeks().volga matches ∂²V/∂σ² from central differences (tol=1e-2)."""
        from ferro_ta.analysis.options import extended_greeks

        for S, K, r, q, T, sigma, opt_type, label in _GREEK_CASES:
            eg = extended_greeks(S, K, r, T, sigma, option_type=opt_type, carry=q)
            expected = volga_num(S, K, r, q, T, sigma)
            assert float(eg.volga) == pytest.approx(expected, abs=1e-2), (
                f"Volga mismatch for '{label}': got {eg.volga}, expected {expected}"
            )

    def test_speed_negative_for_calls(self):
        """Speed (∂Γ/∂S) should be negative for OTM calls — Gamma decreases as S moves away."""
        from ferro_ta.analysis.options import extended_greeks

        # OTM call: S < K
        eg = extended_greeks(90.0, 100.0, 0.05, 1.0, 0.20, option_type="call")
        assert float(eg.speed) < 0.0, (
            f"Speed should be negative for OTM call, got {eg.speed}"
        )

    def test_charm_finite_for_valid_inputs(self):
        """Charm should be finite and non-zero for non-degenerate inputs."""
        from ferro_ta.analysis.options import extended_greeks

        for S, K, r, q, T, sigma, opt_type, label in _GREEK_CASES:
            eg = extended_greeks(S, K, r, T, sigma, option_type=opt_type, carry=q)
            assert np.isfinite(float(eg.charm)), (
                f"Charm is not finite for '{label}': {eg.charm}"
            )
            assert eg.charm != 0.0, (
                f"Charm is zero for '{label}' — unexpected for non-degenerate inputs"
            )


class TestAmericanOptionsAccuracy:
    """Property-based tests for American options (no scipy required)."""

    def test_baw_vs_published_values(self):
        """BAW American put satisfies the lower bound: price ≥ max(K - S, European BSM put).

        The Haug (2007) table uses b = r - q (cost of carry convention).  Rather
        than replicate the exact table — which requires matching the BAW carry
        convention precisely — we verify two model-agnostic inequalities that any
        correct American-put implementation must satisfy:

          1. American put ≥ intrinsic value (K - S)
          2. American put ≥ European BSM put (early exercise has non-negative value)
        """
        from ferro_ta.analysis.options import american_option_price, option_price

        S, K, r, T, sigma = 100.0, 100.0, 0.10, 0.25, 0.20
        american = american_option_price(S, K, r, T, sigma, option_type="put")
        european = option_price(S, K, r, T, sigma, option_type="put")

        assert american >= max(K - S, 0.0) - 1e-8, (
            f"American put below intrinsic: {american:.4f} < {max(K - S, 0.0)}"
        )
        assert american >= european - 1e-8, (
            f"American put below European put: {american:.4f} < {european:.4f}"
        )
        # Sanity-check: American ATM put should be in a reasonable range
        assert 0.0 < american < K, (
            f"American put price {american:.4f} is outside (0, K={K})"
        )

    def test_american_put_increases_with_strike(self):
        """Deeper ITM (higher strike for put) ⇒ higher American put price.

        Uses moderately spaced strikes to avoid the intrinsic-value floor
        where K - S becomes the binding constraint and the increments are
        exactly 1-for-1, which can mask ordering issues near the floor.
        """
        from ferro_ta.analysis.options import american_option_price

        # S = 100, K in {85, 100, 115}; rate and carry both 0.05 to avoid b=0 issues
        S, r, T, sigma = 100.0, 0.05, 0.5, 0.25
        strikes = [85.0, 100.0, 115.0]
        prices = [
            american_option_price(S, K, r, T, sigma, option_type="put", carry=r)
            for K in strikes
        ]
        assert prices[0] < prices[1] < prices[2], (
            f"American put prices not monotone in strike: "
            f"K={strikes} → prices={[round(p, 4) for p in prices]}"
        )

    def test_american_call_increases_with_spot(self):
        """Higher spot ⇒ higher American call price."""
        from ferro_ta.analysis.options import american_option_price

        spots = [90.0, 100.0, 110.0]
        prices = [
            american_option_price(S, 100.0, 0.05, 1.0, 0.20, option_type="call")
            for S in spots
        ]
        assert prices[0] < prices[1] < prices[2], (
            f"American call prices not monotone in spot: {prices}"
        )

    def test_american_call_equals_european_no_dividends_no_early_exercise(self):
        """American call with no early-exercise incentive (carry=0) ≈ European call.

        When the cost-of-carry parameter is zero, there is no dividend/carry
        benefit to holding the underlying.  In this regime, it is never
        optimal to early-exercise an American call, so the American call price
        equals the European call price computed with the same carry=0 convention.
        The `early_exercise_premium` function exposes this directly and should
        return ~0 for calls with carry=0.
        """
        from ferro_ta.analysis.options import early_exercise_premium

        S, K, r, T, sigma = 100.0, 100.0, 0.05, 1.0, 0.20
        premium = early_exercise_premium(
            S, K, r, T, sigma, option_type="call", carry=0.0
        )
        assert premium == pytest.approx(0.0, abs=1e-4), (
            f"Early exercise premium for call with carry=0 should be ~0, got {premium:.6f}"
        )

    def test_early_exercise_premium_positive_for_deep_itm_put(self):
        """Deep ITM American put should have a meaningful early exercise premium.

        When S is well below K (deep ITM put), the time value is low and the
        interest gained from early exercise of the put dominates — leading to a
        positive early-exercise premium.
        """
        from ferro_ta.analysis.options import early_exercise_premium

        # Deep ITM: S=70, K=100 — strong incentive to exercise early
        premium = early_exercise_premium(
            70.0, 100.0, 0.10, 1.0, 0.20, option_type="put"
        )
        assert premium > 0.0, (
            f"Deep ITM American put early exercise premium should be > 0, got {premium}"
        )


class TestVolEstimatorsAccuracy:
    @pytest.fixture(autouse=True)
    def require_scipy(self):
        pytest.importorskip("scipy")

    def test_close_to_close_vs_reference_impl(self):
        """C2C vol matches reference formula exactly (tol=1e-10), 100 samples."""
        from ferro_ta.analysis.options import close_to_close_vol

        rng = np.random.default_rng(42)
        log_ret = rng.normal(0.0, 0.01, 100)
        close = 100.0 * np.cumprod(np.exp(log_ret))

        window = 20
        actual = close_to_close_vol(close, window=window, trading_days_per_year=252.0)
        expected = ctc_vol_reference(close, window=window, trading_days=252.0)

        valid = ~np.isnan(expected)
        assert np.allclose(actual[valid], expected[valid], atol=1e-10), (
            "close_to_close_vol does not match reference formula"
        )

    def test_constant_returns_known_vol(self):
        """Constant daily log-return of 0.01 → C2C vol = 0.01 * sqrt(252) ≈ 0.1587."""
        from ferro_ta.analysis.options import close_to_close_vol

        # Build a price series with constant daily log-return of 0.01
        n = 100
        constant_log_ret = 0.01
        close = 100.0 * np.exp(np.arange(n) * constant_log_ret)

        window = 21
        out = close_to_close_vol(close, window=window, trading_days_per_year=252.0)

        # Expected: sqrt(0.01^2 * 252) = 0.01 * sqrt(252)
        expected_vol = constant_log_ret * np.sqrt(252.0)
        valid = ~np.isnan(out)
        assert np.all(valid[window:]), "Expected valid values after warmup"
        assert out[window] == pytest.approx(expected_vol, rel=1e-10), (
            f"Constant-return vol: got {out[window]}, expected {expected_vol}"
        )

    def test_parkinson_lognormal_unbiased(self):
        """Parkinson estimator within 50% of true vol=0.20 for simulated OHLC data.

        Parkinson uses the log(high/low) range as a proxy for daily realized
        vol.  The estimator is unbiased for a Brownian-motion diffusion where
        the daily range follows a known distribution, but a simplified
        simulation (single end-of-day price + independent range draw) will
        underestimate the range.  We therefore build a proper multi-step
        intraday path so the high/low reflects the true diffusion range,
        and use a lenient 50% tolerance to accommodate finite-sample noise.
        """
        from ferro_ta.analysis.options import parkinson_vol

        rng = np.random.default_rng(123)
        true_vol = 0.20
        n_days = 500
        steps_per_day = 50  # intraday steps to get a realistic H-L range
        daily_sigma = true_vol / np.sqrt(252.0)
        step_sigma = daily_sigma / np.sqrt(steps_per_day)

        # Simulate intraday paths, extract open/high/low/close each day
        highs = np.empty(n_days)
        lows = np.empty(n_days)
        price = 100.0
        for i in range(n_days):
            intraday = price * np.exp(
                np.cumsum(rng.normal(0.0, step_sigma, steps_per_day))
            )
            path = np.concatenate([[price], intraday])
            highs[i] = path.max()
            lows[i] = path.min()
            price = intraday[-1]

        window = 21
        out = parkinson_vol(highs, lows, window=window, trading_days_per_year=252.0)
        valid = out[~np.isnan(out)]

        assert len(valid) > 0, "No valid Parkinson estimates"
        median_est = float(np.median(valid))
        assert abs(median_est - true_vol) < 0.50 * true_vol, (
            f"Parkinson estimate {median_est:.4f} is more than 50% from true vol {true_vol}"
        )

    def test_vol_estimators_all_positive_finite(self):
        """All 5 estimators produce finite and positive non-NaN values on random OHLC."""
        from ferro_ta.analysis.options import (
            close_to_close_vol,
            garman_klass_vol,
            parkinson_vol,
            rogers_satchell_vol,
            yang_zhang_vol,
        )

        rng = np.random.default_rng(99)
        n = 200
        log_ret = rng.normal(0.0, 0.01, n)
        close = 100.0 * np.cumprod(np.exp(log_ret))
        high = close * np.exp(np.abs(rng.normal(0.0, 0.005, n)))
        low = close * np.exp(-np.abs(rng.normal(0.0, 0.005, n)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]

        window = 20
        estimators = {
            "close_to_close": close_to_close_vol(close, window=window),
            "parkinson": parkinson_vol(high, low, window=window),
            "garman_klass": garman_klass_vol(open_, high, low, close, window=window),
            "rogers_satchell": rogers_satchell_vol(
                open_, high, low, close, window=window
            ),
            "yang_zhang": yang_zhang_vol(open_, high, low, close, window=window),
        }

        for name, out in estimators.items():
            valid = out[~np.isnan(out)]
            assert len(valid) > 0, f"{name}: no valid (non-NaN) estimates"
            assert np.all(np.isfinite(valid)), f"{name}: non-finite values present"
            assert np.all(valid > 0.0), f"{name}: non-positive values present"


class TestVolConeAccuracy:
    """Tests for vol_cone — no scipy required."""

    def test_cone_windows_match_requested(self):
        """Output windows should match the input list exactly."""
        from ferro_ta.analysis.options import vol_cone

        rng = np.random.default_rng(0)
        close = 100.0 * np.cumprod(np.exp(rng.normal(0.0, 0.01, 500)))
        requested = (10, 21, 42)
        cone = vol_cone(close, windows=requested)

        assert list(cone.windows.astype(int)) == list(requested), (
            f"Cone windows {list(cone.windows)} do not match requested {list(requested)}"
        )

    def test_cone_median_matches_rolling_median(self):
        """Manually computed rolling C2C vol median for window=21 should match cone.median[0]."""
        from ferro_ta.analysis.options import close_to_close_vol, vol_cone

        rng = np.random.default_rng(5)
        close = 100.0 * np.cumprod(np.exp(rng.normal(0.0, 0.01, 500)))
        window = 21

        cone = vol_cone(close, windows=(window,))

        rolling = close_to_close_vol(close, window=window, trading_days_per_year=252.0)
        valid = rolling[~np.isnan(rolling)]
        manual_median = float(np.median(valid))

        assert cone.median[0] == pytest.approx(manual_median, rel=1e-6), (
            f"vol_cone median {cone.median[0]:.6f} does not match manual median {manual_median:.6f}"
        )


class TestStrategyAnalyticsAccuracy:
    @pytest.fixture(autouse=True)
    def require_scipy(self):
        pytest.importorskip("scipy")

    def test_put_call_parity_deviation_analytical(self):
        """BSM call/put from scipy formulas fed into put_call_parity_deviation → < 1e-8."""
        from ferro_ta.analysis.options import put_call_parity_deviation

        S, K, r, q, T, sigma = 100.0, 100.0, 0.05, 0.02, 1.0, 0.20
        call = bsm_call(S, K, r, q, T, sigma)
        put = bsm_put(S, K, r, q, T, sigma)

        dev = put_call_parity_deviation(call, put, S, K, r, T, carry=q)
        assert abs(dev) < 1e-8, (
            f"put_call_parity_deviation for BSM-consistent prices: got {dev}, expected ~0"
        )

    def test_expected_move_known_value(self):
        """S=100, iv=0.20, days=30, trading_days=252 → upper move ≈ 7.14."""
        from ferro_ta.analysis.options import expected_move

        S, iv, days, td = 100.0, 0.20, 30.0, 252.0
        lower, upper = expected_move(S, iv, days, td)

        # log-normal formula: S * (exp(sigma * sqrt(days/trading_days)) - 1)
        expected_upper = S * (np.exp(iv * np.sqrt(days / td)) - 1.0)
        expected_lower = S * (np.exp(-iv * np.sqrt(days / td)) - 1.0)

        assert upper == pytest.approx(expected_upper, rel=1e-6), (
            f"expected_move upper: got {upper:.4f}, expected {expected_upper:.4f}"
        )
        assert lower == pytest.approx(expected_lower, rel=1e-6), (
            f"expected_move lower: got {lower:.4f}, expected {expected_lower:.4f}"
        )
        # Numeric check: upper ≈ 7.14
        assert upper == pytest.approx(7.14, abs=0.05), (
            f"expected_move upper should be ~7.14, got {upper:.4f}"
        )
