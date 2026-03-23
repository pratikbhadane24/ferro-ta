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
