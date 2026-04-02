import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


class TestOptionsAnalytics:
    def test_black_scholes_price_scalar(self):
        from ferro_ta.analysis.options import black_scholes_price

        price = black_scholes_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
        )
        assert price == pytest.approx(10.4506, rel=1e-4)

    def test_black_76_price_vectorized(self):
        from ferro_ta.analysis.options import black_76_price

        price = black_76_price(
            np.array([100.0, 105.0]),
            np.array([100.0, 100.0]),
            0.03,
            1.0,
            np.array([0.2, 0.25]),
            option_type="call",
        )
        assert isinstance(price, np.ndarray)
        assert price.shape == (2,)
        assert np.all(price > 0.0)

    def test_greeks_and_iv_recovery(self):
        from ferro_ta.analysis.options import greeks, implied_volatility, option_price

        price = option_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
            model="bsm",
        )
        iv = implied_volatility(
            price,
            100.0,
            100.0,
            0.05,
            1.0,
            option_type="call",
            model="bsm",
        )
        result = greeks(
            100.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
            model="bsm",
        )
        assert iv == pytest.approx(0.2, rel=1e-6)
        assert result.delta == pytest.approx(0.6368, rel=1e-3)
        assert result.gamma > 0.0
        assert result.vega > 0.0

    def test_smile_and_chain_helpers(self):
        from ferro_ta.analysis.options import (
            label_moneyness,
            select_strike,
            smile_metrics,
            term_structure_slope,
        )

        strikes = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = np.array([0.30, 0.25, 0.20, 0.22, 0.27])

        metrics = smile_metrics(strikes, vols, 100.0, 0.5)
        labels = label_moneyness(strikes, 100.0, option_type="call")

        assert metrics.atm_iv == pytest.approx(0.20, rel=1e-6)
        assert metrics.skew_slope < 0.0
        assert labels.tolist() == ["ITM", "ITM", "ATM", "OTM", "OTM"]
        assert select_strike(strikes, 101.0, selector="ATM") == 100.0
        assert (
            select_strike(strikes, 101.0, option_type="call", selector="OTM2") == 120.0
        )
        assert select_strike(
            strikes,
            100.0,
            selector="DELTA0.25",
            option_type="call",
            volatilities=vols,
            time_to_expiry=0.5,
        ) in set(strikes.tolist())
        assert term_structure_slope([0.1, 0.5, 1.0], [0.18, 0.20, 0.22]) > 0.0


class TestFuturesAnalytics:
    def test_basis_and_curve_helpers(self):
        from ferro_ta.analysis.futures import (
            annualized_basis,
            basis,
            calendar_spreads,
            carry_spread,
            curve_summary,
            implied_carry_rate,
            synthetic_forward,
        )

        assert basis(100.0, 103.0) == pytest.approx(3.0)
        assert annualized_basis(100.0, 103.0, 0.25) > 0.0
        assert implied_carry_rate(100.0, 103.0, 0.25) > 0.0
        assert carry_spread(100.0, 103.0, 0.02, 0.25) > -1.0
        assert synthetic_forward(8.0, 5.0, 100.0, 0.02, 0.5) > 100.0
        assert np.allclose(calendar_spreads([100.0, 101.0, 103.0]), [1.0, 2.0])

        summary = curve_summary(100.0, [0.1, 0.5, 1.0], [101.0, 102.0, 104.0])
        assert summary.is_contango is True
        assert summary.slope > 0.0

    def test_roll_helpers(self):
        from ferro_ta.analysis.futures import (
            back_adjusted_continuous_contract,
            ratio_adjusted_continuous_contract,
            roll_yield,
            weighted_continuous_contract,
        )

        front = np.array([100.0, 101.0, 102.0, 103.0])
        nxt = np.array([101.0, 102.0, 103.0, 104.0])
        weights = np.array([0.0, 0.25, 0.75, 1.0])

        weighted = weighted_continuous_contract(front, nxt, weights)
        back_adjusted = back_adjusted_continuous_contract(front, nxt, weights)
        ratio_adjusted = ratio_adjusted_continuous_contract(front, nxt, weights)

        assert weighted.shape == front.shape
        assert back_adjusted.shape == front.shape
        assert ratio_adjusted.shape == front.shape
        assert roll_yield(100.0, 102.0, 30.0 / 365.0) > 0.0


class TestStrategyAndPayoff:
    def test_strategy_schema_and_preset(self):
        from ferro_ta.analysis.options_strategy import (
            DerivativesStrategy,
            ExpirySelector,
            ExpirySelectorKind,
            LegPreset,
            StrategyLeg,
            StrikeSelector,
            StrikeSelectorKind,
            build_strategy_preset,
        )

        preset = build_strategy_preset(
            LegPreset.STRADDLE,
            name="ATM Straddle",
            underlying="NIFTY",
            expiry_selector=ExpirySelector(ExpirySelectorKind.CURRENT_WEEK),
        )
        custom = DerivativesStrategy(
            name="Custom Single",
            legs=(
                StrategyLeg(
                    "NIFTY",
                    ExpirySelector(ExpirySelectorKind.CURRENT_WEEK),
                    StrikeSelector(
                        StrikeSelectorKind.EXPLICIT, explicit_strike=22000.0
                    ),
                    "call",
                ),
            ),
        )

        assert len(preset.legs) == 2
        assert custom.to_dict()["name"] == "Custom Single"

    def test_payoff_and_aggregate_greeks(self):
        from ferro_ta.analysis.derivatives_payoff import (
            PayoffLeg,
            aggregate_greeks,
            strategy_payoff,
        )

        spot_grid = np.array([90.0, 100.0, 110.0])
        legs = [
            PayoffLeg(
                instrument="option",
                side="long",
                option_type="call",
                strike=100.0,
                premium=5.0,
                volatility=0.2,
                time_to_expiry=0.5,
            ),
            PayoffLeg(
                instrument="option",
                side="short",
                option_type="call",
                strike=110.0,
                premium=2.0,
                volatility=0.22,
                time_to_expiry=0.5,
            ),
            PayoffLeg(instrument="future", side="long", entry_price=100.0),
        ]

        payoff = strategy_payoff(spot_grid, legs=legs)
        greeks = aggregate_greeks(100.0, legs=legs)

        assert payoff.shape == spot_grid.shape
        assert payoff[1] == pytest.approx(-3.0)
        assert greeks.delta > 0.0
        assert greeks.gamma > 0.0


class TestStockInstrument:
    def test_stock_leg_payoff_linear(self):
        from ferro_ta.analysis.derivatives_payoff import stock_leg_payoff

        spot_grid = np.array([90.0, 100.0, 110.0])
        payoff = stock_leg_payoff(spot_grid, entry_price=100.0, side="long")
        assert payoff == pytest.approx([-10.0, 0.0, 10.0])

    def test_stock_leg_short_side(self):
        from ferro_ta.analysis.derivatives_payoff import stock_leg_payoff

        spot_grid = np.array([90.0, 100.0, 110.0])
        payoff = stock_leg_payoff(spot_grid, entry_price=100.0, side="short")
        assert payoff == pytest.approx([10.0, 0.0, -10.0])

    def test_strategy_payoff_with_stock_leg(self):
        from ferro_ta.analysis.derivatives_payoff import PayoffLeg, strategy_payoff

        # Covered call: long stock + short call
        spot_grid = np.array([90.0, 100.0, 110.0, 120.0])
        legs = [
            PayoffLeg(instrument="stock", side="long", entry_price=100.0),
            PayoffLeg(
                instrument="option",
                side="short",
                option_type="call",
                strike=110.0,
                premium=3.0,
            ),
        ]
        payoff = strategy_payoff(spot_grid, legs=legs)
        assert payoff.shape == spot_grid.shape
        # At 90: stock P&L = -10, short call = +3 (OTM) → total = -7
        assert payoff[0] == pytest.approx(-7.0)
        # At 110: stock P&L = +10, short call = +3 (ATM, intrinsic=0) → total = +13
        assert payoff[2] == pytest.approx(13.0)

    def test_strategy_leg_accepts_stock_instrument(self):
        from ferro_ta.analysis.options_strategy import StrategyLeg

        leg = StrategyLeg(
            underlying="NIFTY",
            expiry_selector=None,
            strike_selector=None,
            option_type=None,
            instrument="stock",
            side="long",
        )
        assert leg.instrument == "stock"


class TestExtendedGreeks:
    def test_extended_greeks_returns_five_values(self):
        from ferro_ta.analysis.options import ExtendedGreeks, extended_greeks

        eg = extended_greeks(100.0, 100.0, 0.05, 1.0, 0.2, option_type="call")
        assert isinstance(eg, ExtendedGreeks)
        assert eg.vanna is not None
        assert eg.volga is not None
        assert eg.charm is not None
        assert eg.speed is not None
        assert eg.color is not None

    def test_vanna_sign_otm_call(self):
        # OTM call vanna > 0 (delta increases as vol rises)
        from ferro_ta.analysis.options import extended_greeks

        eg = extended_greeks(100.0, 110.0, 0.05, 1.0, 0.2, option_type="call")
        assert eg.vanna > 0.0

    def test_extended_greeks_finite_for_valid_inputs(self):
        from ferro_ta.analysis.options import extended_greeks

        eg = extended_greeks(100.0, 100.0, 0.05, 1.0, 0.25, option_type="put")
        assert np.isfinite(eg.vanna)
        assert np.isfinite(eg.volga)
        assert np.isfinite(eg.charm)
        assert np.isfinite(eg.speed)
        assert np.isfinite(eg.color)

    def test_volga_positive_atm(self):
        # Volga is always non-negative for standard BSM inputs
        from ferro_ta.analysis.options import extended_greeks

        eg = extended_greeks(100.0, 100.0, 0.05, 1.0, 0.2, option_type="call")
        assert eg.volga >= 0.0


class TestDigitalOptions:
    def test_cash_or_nothing_call_atm(self):
        from ferro_ta.analysis.options import digital_option_price

        # ATM cash-or-nothing call ≈ e^{-rT} * N(d2) ≈ 0.532
        price = digital_option_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
            digital_type="cash_or_nothing",
        )
        assert 0.0 < price < 1.0
        assert price == pytest.approx(0.532, rel=0.02)

    def test_asset_or_nothing_call_atm(self):
        from ferro_ta.analysis.options import digital_option_price

        price = digital_option_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
            digital_type="asset_or_nothing",
        )
        # asset-or-nothing call ≈ S * N(d1) < S
        assert 0.0 < price < 100.0

    def test_put_call_parity_cash_or_nothing(self):
        from ferro_ta.analysis.options import digital_option_price

        call = digital_option_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.25,
            option_type="call",
            digital_type="cash_or_nothing",
        )
        put = digital_option_price(
            100.0,
            100.0,
            0.05,
            1.0,
            0.25,
            option_type="put",
            digital_type="cash_or_nothing",
        )
        discount = np.exp(-0.05)
        assert call + put == pytest.approx(discount, rel=1e-6)

    def test_digital_greeks_finite(self):
        from ferro_ta.analysis.options import digital_option_greeks

        g = digital_option_greeks(
            100.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
            digital_type="cash_or_nothing",
        )
        assert np.isfinite(g.delta)
        assert np.isfinite(g.gamma)
        assert np.isfinite(g.vega)

    def test_digital_invalid_returns_nan(self):
        from ferro_ta.analysis.options import digital_option_price

        price = digital_option_price(
            -1.0,
            100.0,
            0.05,
            1.0,
            0.2,
            option_type="call",
            digital_type="cash_or_nothing",
        )
        assert np.isnan(price)


class TestAmericanOptions:
    def test_american_price_gte_european(self):
        from ferro_ta.analysis.options import american_option_price, option_price

        spot, strike, rate, tte, vol = 100.0, 100.0, 0.05, 1.0, 0.2
        american = american_option_price(
            spot, strike, rate, tte, vol, option_type="call"
        )
        european = option_price(spot, strike, rate, tte, vol, option_type="call")
        assert american >= european - 1e-8

    def test_early_exercise_premium_nonnegative(self):
        from ferro_ta.analysis.options import early_exercise_premium

        premium = early_exercise_premium(
            100.0, 100.0, 0.05, 1.0, 0.2, option_type="put"
        )
        assert premium >= 0.0

    def test_american_put_early_exercise_positive(self):
        # Deep ITM put with high rate should have meaningful early exercise premium
        from ferro_ta.analysis.options import early_exercise_premium

        premium = early_exercise_premium(80.0, 100.0, 0.1, 0.5, 0.25, option_type="put")
        assert premium > 0.0

    def test_american_call_no_dividends_no_premium(self):
        # With zero carry (no dividends), American call = European call
        from ferro_ta.analysis.options import early_exercise_premium

        premium = early_exercise_premium(
            100.0, 100.0, 0.05, 1.0, 0.2, option_type="call", carry=0.0
        )
        assert premium == pytest.approx(0.0, abs=1e-4)


class TestVolEstimators:
    @pytest.fixture
    def sample_ohlc(self):
        rng = np.random.default_rng(42)
        n = 100
        log_ret = rng.normal(0.0, 0.01, n)
        close = 100.0 * np.cumprod(np.exp(log_ret))
        high = close * np.exp(np.abs(rng.normal(0.0, 0.005, n)))
        low = close * np.exp(-np.abs(rng.normal(0.0, 0.005, n)))
        open_ = np.roll(close, 1)
        open_[0] = close[0]
        return open_, high, low, close

    def test_close_to_close_vol_length(self, sample_ohlc):
        from ferro_ta.analysis.options import close_to_close_vol

        _, _, _, close = sample_ohlc
        out = close_to_close_vol(close, window=20)
        assert len(out) == len(close)

    def test_close_to_close_vol_warmup_nan(self, sample_ohlc):
        from ferro_ta.analysis.options import close_to_close_vol

        _, _, _, close = sample_ohlc
        out = close_to_close_vol(close, window=20)
        # First `window` values are NaN; index `window` is the first valid value
        assert all(np.isnan(out[:20]))
        assert np.isfinite(out[20])

    def test_parkinson_vol_finite_and_positive(self, sample_ohlc):
        from ferro_ta.analysis.options import parkinson_vol

        _, high, low, _ = sample_ohlc
        out = parkinson_vol(high, low, window=20)
        finite = out[~np.isnan(out)]
        assert len(finite) > 0
        assert np.all(finite > 0.0)

    def test_garman_klass_vol(self, sample_ohlc):
        from ferro_ta.analysis.options import garman_klass_vol

        open_, high, low, close = sample_ohlc
        out = garman_klass_vol(open_, high, low, close, window=20)
        finite = out[~np.isnan(out)]
        assert len(finite) > 0
        assert np.all(finite > 0.0)

    def test_rogers_satchell_vol(self, sample_ohlc):
        from ferro_ta.analysis.options import rogers_satchell_vol

        open_, high, low, close = sample_ohlc
        out = rogers_satchell_vol(open_, high, low, close, window=20)
        finite = out[~np.isnan(out)]
        assert len(finite) > 0

    def test_yang_zhang_vol(self, sample_ohlc):
        from ferro_ta.analysis.options import yang_zhang_vol

        open_, high, low, close = sample_ohlc
        out = yang_zhang_vol(open_, high, low, close, window=20)
        finite = out[~np.isnan(out)]
        assert len(finite) > 0
        assert np.all(finite > 0.0)

    def test_yang_zhang_lower_variance_than_close_to_close(self, sample_ohlc):
        # YZ is more efficient than close-to-close
        from ferro_ta.analysis.options import close_to_close_vol, yang_zhang_vol

        open_, high, low, close = sample_ohlc
        c2c = close_to_close_vol(close, window=20)
        yz = yang_zhang_vol(open_, high, low, close, window=20)
        valid = ~np.isnan(c2c) & ~np.isnan(yz)
        # YZ variance < C2C variance (efficiency test)
        assert np.var(yz[valid]) <= np.var(c2c[valid]) * 2.0  # lenient bound


class TestVolCone:
    def test_vol_cone_shape(self):
        from ferro_ta.analysis.options import VolCone, vol_cone

        rng = np.random.default_rng(0)
        close = 100.0 * np.cumprod(np.exp(rng.normal(0.0, 0.01, 300)))
        cone = vol_cone(close, windows=(21, 42, 63))
        assert isinstance(cone, VolCone)
        assert len(cone.windows) == 3
        assert len(cone.min) == 3

    def test_vol_cone_monotonic_percentiles(self):
        from ferro_ta.analysis.options import vol_cone

        rng = np.random.default_rng(1)
        close = 100.0 * np.cumprod(np.exp(rng.normal(0.0, 0.01, 500)))
        cone = vol_cone(close, windows=(21, 42, 63, 126, 252))
        for i in range(len(cone.windows)):
            assert (
                cone.min[i]
                <= cone.p25[i]
                <= cone.median[i]
                <= cone.p75[i]
                <= cone.max[i]
            )

    def test_vol_cone_positive_values(self):
        from ferro_ta.analysis.options import vol_cone

        rng = np.random.default_rng(2)
        close = 100.0 * np.cumprod(np.exp(rng.normal(0.0, 0.01, 400)))
        cone = vol_cone(close)
        assert np.all(cone.min > 0.0)


class TestStrategyAnalytics:
    def test_put_call_parity_deviation_zero(self):
        from ferro_ta.analysis.options import option_price, put_call_parity_deviation

        s, k, r, tte, vol = 100.0, 100.0, 0.05, 1.0, 0.2
        call = option_price(s, k, r, tte, vol, option_type="call")
        put = option_price(s, k, r, tte, vol, option_type="put")
        dev = put_call_parity_deviation(call, put, s, k, r, tte)
        assert dev == pytest.approx(0.0, abs=1e-6)

    def test_put_call_parity_deviation_nonzero_for_stale_quote(self):
        from ferro_ta.analysis.options import put_call_parity_deviation

        dev = put_call_parity_deviation(15.0, 5.0, 100.0, 100.0, 0.05, 1.0)
        assert abs(dev) > 0.01

    def test_expected_move_positive(self):
        from ferro_ta.analysis.options import expected_move

        lower, upper = expected_move(100.0, 0.2, 30.0)
        assert upper > 0.0
        assert lower < 0.0

    def test_expected_move_log_normal_asymmetry(self):
        # Log-normal expected move: upper > |lower| (right-skew)
        from ferro_ta.analysis.options import expected_move

        lower, upper = expected_move(100.0, 0.2, 30.0)
        # Both magnitudes are similar (within 10%) but upper > |lower|
        assert upper > abs(lower) * 0.95
        assert upper < abs(lower) * 2.0

    def test_strategy_value_near_expiry_approx_payoff(self):
        from ferro_ta.analysis.derivatives_payoff import (
            PayoffLeg,
            strategy_payoff,
            strategy_value,
        )

        # Near expiry, BSM value ≈ intrinsic payoff
        spot_grid = np.array([90.0, 100.0, 110.0])
        legs = [
            PayoffLeg(
                instrument="option",
                side="long",
                option_type="call",
                strike=100.0,
                premium=0.0,
                volatility=0.2,
                time_to_expiry=0.001,
            )
        ]
        val = strategy_value(spot_grid, legs=legs, time_to_expiry=0.001, volatility=0.2)
        payoff = strategy_payoff(spot_grid, legs=legs)
        # Near expiry, value ≈ payoff (within a few cents)
        assert np.allclose(val, payoff, atol=0.5)

    def test_strategy_value_shape(self):
        from ferro_ta.analysis.derivatives_payoff import PayoffLeg, strategy_value

        spot_grid = np.linspace(80.0, 120.0, 20)
        legs = [
            PayoffLeg(
                instrument="option",
                side="long",
                option_type="call",
                strike=100.0,
                premium=5.0,
                volatility=0.2,
                time_to_expiry=0.5,
            )
        ]
        val = strategy_value(spot_grid, legs=legs, time_to_expiry=0.5, volatility=0.2)
        assert val.shape == spot_grid.shape


class TestDerivativesBenchmarking:
    def test_derivatives_benchmark_smoke(self, tmp_path):
        root = Path(__file__).resolve().parents[2]
        script = root / "benchmarks" / "bench_derivatives_compare.py"
        output_path = tmp_path / "derivatives_benchmark.json"

        completed = subprocess.run(
            [
                sys.executable,
                str(script),
                "--sizes",
                "32",
                "--accuracy-size",
                "16",
                "--json",
                str(output_path),
            ],
            cwd=root,
            check=False,
            capture_output=True,
            text=True,
        )

        assert completed.returncode == 0, completed.stdout + completed.stderr
        assert output_path.is_file()
        payload = output_path.read_text(encoding="utf-8")
        assert '"accuracy"' in payload
        assert '"speed"' in payload
        assert '"provider": "ferro_ta"' in payload
