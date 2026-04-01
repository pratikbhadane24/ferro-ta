"""Integration tests for pandas and polars DataFrame/Series support.

Verifies that ferro_ta indicators transparently accept pandas Series and
polars Series inputs, returning correctly shaped results with preserved
index/name metadata.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ferro_ta import BBANDS, EMA, MACD, RSI, SMA

# ---------------------------------------------------------------------------
# Pandas Series tests
# ---------------------------------------------------------------------------


class TestPandasSeries:
    """Indicators accept pd.Series and return pd.Series with index."""

    def test_sma_returns_series(self, ohlcv_500):
        s = pd.Series(ohlcv_500["close"])
        result = SMA(s, timeperiod=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_ema_returns_series(self, ohlcv_500):
        s = pd.Series(ohlcv_500["close"])
        result = EMA(s, timeperiod=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_rsi_returns_series(self, ohlcv_500):
        s = pd.Series(ohlcv_500["close"])
        result = RSI(s, timeperiod=14)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_bbands_returns_tuple_of_series(self, ohlcv_500):
        s = pd.Series(ohlcv_500["close"])
        upper, middle, lower = BBANDS(s, timeperiod=5)
        for band in (upper, middle, lower):
            assert isinstance(band, pd.Series)
            assert len(band) == len(s)

    def test_macd_returns_tuple_of_series(self, ohlcv_500):
        s = pd.Series(ohlcv_500["close"])
        macd, signal, hist = MACD(s)
        for arr in (macd, signal, hist):
            assert isinstance(arr, pd.Series)
            assert len(arr) == len(s)

    def test_index_preserved(self, ohlcv_500):
        """Resulting Series should carry the same index as the input."""
        idx = pd.date_range("2020-01-01", periods=len(ohlcv_500["close"]), freq="D")
        s = pd.Series(ohlcv_500["close"], index=idx)
        result = SMA(s, timeperiod=14)
        assert isinstance(result, pd.Series)
        pd.testing.assert_index_equal(result.index, idx)

    def test_named_series(self, ohlcv_500):
        """Named Series should still work (name is not necessarily preserved,
        but the call should not error)."""
        s = pd.Series(ohlcv_500["close"], name="close_price")
        result = EMA(s, timeperiod=10)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_series_with_nan_values(self):
        """NaN values in the input should not crash the indicator."""
        data = np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s = pd.Series(data)
        result = SMA(s, timeperiod=3)
        assert isinstance(result, pd.Series)
        assert len(result) == len(s)

    def test_bbands_index_preserved(self, ohlcv_500):
        idx = pd.date_range("2020-01-01", periods=len(ohlcv_500["close"]), freq="D")
        s = pd.Series(ohlcv_500["close"], index=idx)
        upper, middle, lower = BBANDS(s, timeperiod=5)
        for band in (upper, middle, lower):
            pd.testing.assert_index_equal(band.index, idx)

    def test_macd_index_preserved(self, ohlcv_500):
        idx = pd.date_range("2020-01-01", periods=len(ohlcv_500["close"]), freq="D")
        s = pd.Series(ohlcv_500["close"], index=idx)
        macd, signal, hist = MACD(s)
        for arr in (macd, signal, hist):
            pd.testing.assert_index_equal(arr.index, idx)


# ---------------------------------------------------------------------------
# Polars Series tests
# ---------------------------------------------------------------------------


class TestPolarsSeries:
    """Indicators accept polars.Series and return polars.Series."""

    @pytest.fixture(autouse=True)
    def _require_polars(self):
        self.pl = pytest.importorskip("polars")

    def test_sma_returns_polars_series(self, ohlcv_500):
        s = self.pl.Series("close", ohlcv_500["close"])
        result = SMA(s, timeperiod=14)
        assert isinstance(result, self.pl.Series)
        assert len(result) == len(s)

    def test_ema_returns_polars_series(self, ohlcv_500):
        s = self.pl.Series("close", ohlcv_500["close"])
        result = EMA(s, timeperiod=14)
        assert isinstance(result, self.pl.Series)
        assert len(result) == len(s)

    def test_rsi_returns_polars_series(self, ohlcv_500):
        s = self.pl.Series("close", ohlcv_500["close"])
        result = RSI(s, timeperiod=14)
        assert isinstance(result, self.pl.Series)
        assert len(result) == len(s)

    def test_bbands_returns_tuple_of_polars_series(self, ohlcv_500):
        s = self.pl.Series("close", ohlcv_500["close"])
        upper, middle, lower = BBANDS(s, timeperiod=5)
        for band in (upper, middle, lower):
            assert isinstance(band, self.pl.Series)
            assert len(band) == len(s)

    def test_macd_returns_tuple_of_polars_series(self, ohlcv_500):
        s = self.pl.Series("close", ohlcv_500["close"])
        macd, signal, hist = MACD(s)
        for arr in (macd, signal, hist):
            assert isinstance(arr, self.pl.Series)
            assert len(arr) == len(s)

    def test_series_name_preserved(self, ohlcv_500):
        """The polars Series name from the first input should be carried through."""
        s = self.pl.Series("my_close", ohlcv_500["close"])
        result = SMA(s, timeperiod=14)
        assert isinstance(result, self.pl.Series)
        assert result.name == "my_close"


# ---------------------------------------------------------------------------
# DataFrame workflow tests
# ---------------------------------------------------------------------------


class TestDataFrameWorkflow:
    """End-to-end workflow: build a DataFrame, compute indicators, add columns."""

    def test_pandas_dataframe_workflow(self, ohlcv_500):
        df = pd.DataFrame(ohlcv_500)

        # Compute indicators from DataFrame columns
        df["sma_14"] = SMA(df["close"], timeperiod=14)
        df["ema_14"] = EMA(df["close"], timeperiod=14)
        df["rsi_14"] = RSI(df["close"], timeperiod=14)

        upper, middle, lower = BBANDS(df["close"], timeperiod=5)
        df["bb_upper"] = upper
        df["bb_middle"] = middle
        df["bb_lower"] = lower

        macd, signal, hist = MACD(df["close"])
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist

        # All new columns should exist and have correct length
        new_cols = [
            "sma_14",
            "ema_14",
            "rsi_14",
            "bb_upper",
            "bb_middle",
            "bb_lower",
            "macd",
            "macd_signal",
            "macd_hist",
        ]
        for col in new_cols:
            assert col in df.columns
            assert len(df[col]) == 500

        # SMA leading values should be NaN
        assert np.isnan(df["sma_14"].iloc[0])
        # Non-NaN values should exist after warmup
        assert not np.isnan(df["sma_14"].iloc[-1])

    def test_pandas_dataframe_index_consistency(self, ohlcv_500):
        """Indicator columns should align with the original DataFrame index."""
        idx = pd.date_range("2020-01-01", periods=500, freq="D")
        df = pd.DataFrame(ohlcv_500, index=idx)

        df["sma_14"] = SMA(df["close"], timeperiod=14)
        pd.testing.assert_index_equal(df["sma_14"].dropna().index, idx[13:])

    def test_polars_dataframe_workflow(self, ohlcv_500):
        pl = pytest.importorskip("polars")
        df = pl.DataFrame(ohlcv_500)

        sma_result = SMA(df["close"], timeperiod=14)
        ema_result = EMA(df["close"], timeperiod=14)
        rsi_result = RSI(df["close"], timeperiod=14)

        # Results are polars Series of correct length
        for result in (sma_result, ema_result, rsi_result):
            assert isinstance(result, pl.Series)
            assert len(result) == 500

        # Can add back to a polars DataFrame via with_columns
        df2 = df.with_columns(
            sma_result.alias("sma_14"),
            ema_result.alias("ema_14"),
            rsi_result.alias("rsi_14"),
        )
        assert "sma_14" in df2.columns
        assert "ema_14" in df2.columns
        assert "rsi_14" in df2.columns
        assert df2.shape[0] == 500
