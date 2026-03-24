"""
Cross-library wrapper registry — comprehensive indicator coverage.

Unified interface: execute_indicator(library, indicator, data, df=None, **kwargs)
Supported libraries: ferro_ta, talib, pandas_ta, ta, tulipy, finta
50+ indicators across all categories.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def _try_import(name):
    try:
        import importlib

        return importlib.import_module(name)
    except ImportError:
        return None


_talib = _try_import("talib")
_pta = _try_import("pandas_ta")
_ta = _try_import("ta")
_tl = _try_import("tulipy")
_fi_m = _try_import("finta")
_fi = getattr(_fi_m, "TA", None) if _fi_m else None


def available_libraries():
    libs = ["ferro_ta"]
    if _talib:
        libs.append("talib")
    if _pta:
        libs.append("pandas_ta")
    if _ta:
        libs.append("ta")
    if _tl:
        libs.append("tulipy")
    if _fi:
        libs.append("finta")
    return libs


def is_supported(library: str, indicator: str) -> bool:
    """Return True if a wrapper exists for the given (library, indicator) pair."""
    if library not in available_libraries():
        return False
    return (library, indicator) in REGISTRY


def _strip_nan(arr):
    a = np.asarray(arr, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def _c64(a):
    return np.ascontiguousarray(a, dtype=np.float64)


def _empty():
    return np.array([], dtype=np.float64)


def _first_col(df, prefix):
    col = next((c for c in df.columns if c.startswith(prefix)), None)
    return _strip_nan(df[col].values) if col is not None else _empty()


# ============================================================
# OVERLAP
# ============================================================
def _sma_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.SMA(d["close"], timeperiod=timeperiod))


def _sma_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.SMA(d["close"], timeperiod=timeperiod))


def _sma_pt(d, df, timeperiod=20, **_):
    return _strip_nan(_pta.sma(df["close"], length=timeperiod).values)


def _sma_ta(d, df, timeperiod=20, **_):
    from ta.trend import SMAIndicator

    return _strip_nan(
        SMAIndicator(df["close"], window=timeperiod).sma_indicator().values
    )


def _sma_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.sma(_c64(d["close"]), period=timeperiod))


def _sma_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.SMA(df, timeperiod).values)


def _ema_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.EMA(d["close"], timeperiod=timeperiod))


def _ema_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.EMA(d["close"], timeperiod=timeperiod))


def _ema_pt(d, df, timeperiod=20, **_):
    return _strip_nan(_pta.ema(df["close"], length=timeperiod).values)


def _ema_ta(d, df, timeperiod=20, **_):
    from ta.trend import EMAIndicator

    return _strip_nan(
        EMAIndicator(df["close"], window=timeperiod).ema_indicator().values
    )


def _ema_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.ema(_c64(d["close"]), period=timeperiod))


def _ema_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.EMA(df, timeperiod).values)


def _wma_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.WMA(d["close"], timeperiod=timeperiod))


def _wma_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.WMA(d["close"], timeperiod=timeperiod))


def _wma_pt(d, df, timeperiod=14, **_):
    return _strip_nan(_pta.wma(df["close"], length=timeperiod).values)


def _wma_ta(d, df, **_):
    return _empty()


_wma_ta._stub = True


def _wma_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.wma(_c64(d["close"]), period=timeperiod))


def _wma_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.WMA(df, timeperiod).values)


def _dema_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.DEMA(d["close"], timeperiod=timeperiod))


def _dema_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.DEMA(d["close"], timeperiod=timeperiod))


def _dema_pt(d, df, timeperiod=20, **_):
    return _strip_nan(_pta.dema(df["close"], length=timeperiod).values)


def _dema_ta(d, df, **_):
    return _empty()


_dema_ta._stub = True


def _dema_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.dema(_c64(d["close"]), period=timeperiod))


def _dema_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.DEMA(df, timeperiod).values)


def _tema_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.TEMA(d["close"], timeperiod=timeperiod))


def _tema_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.TEMA(d["close"], timeperiod=timeperiod))


def _tema_pt(d, df, timeperiod=20, **_):
    return _strip_nan(_pta.tema(df["close"], length=timeperiod).values)


def _tema_ta(d, df, **_):
    return _empty()


_tema_ta._stub = True


def _tema_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.tema(_c64(d["close"]), period=timeperiod))


def _tema_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.TEMA(df, timeperiod).values)


def _t3_ft(d, df, timeperiod=5, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.T3(d["close"], timeperiod=timeperiod))


def _t3_tl(d, df, timeperiod=5, **_):
    return _strip_nan(_talib.T3(d["close"], timeperiod=timeperiod))


def _t3_pt(d, df, timeperiod=5, **_):
    return _strip_nan(_pta.t3(df["close"], length=timeperiod).values)


def _t3_ta(d, df, **_):
    return _empty()


_t3_ta._stub = True


def _t3_tu(d, df, **_):
    return _empty()


_t3_tu._stub = True


def _t3_fi(d, df, **_):
    return _empty()


_t3_fi._stub = True


def _trima_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.TRIMA(d["close"], timeperiod=timeperiod))


def _trima_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.TRIMA(d["close"], timeperiod=timeperiod))


def _trima_pt(d, df, timeperiod=20, **_):
    return _strip_nan(_pta.trima(df["close"], length=timeperiod).values)


def _trima_ta(d, df, **_):
    return _empty()


_trima_ta._stub = True


def _trima_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.trima(_c64(d["close"]), period=timeperiod))


def _trima_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.TRIMA(df, timeperiod).values)


def _kama_ft(d, df, timeperiod=10, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.KAMA(d["close"], timeperiod=timeperiod))


def _kama_tl(d, df, timeperiod=10, **_):
    return _strip_nan(_talib.KAMA(d["close"], timeperiod=timeperiod))


def _kama_pt(d, df, timeperiod=10, **_):
    return _strip_nan(_pta.kama(df["close"], length=timeperiod).values)


def _kama_ta(d, df, **_):
    return _empty()


_kama_ta._stub = True


def _kama_tu(d, df, timeperiod=10, **_):
    return _strip_nan(_tl.kama(_c64(d["close"]), period=timeperiod))


def _kama_fi(d, df, **_):
    return _empty()


_kama_fi._stub = True


def _hma_ft(d, df, timeperiod=16, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.HULL_MA(d["close"], timeperiod=timeperiod))


def _hma_tl(d, df, **_):
    return _empty()


_hma_tl._stub = True


def _hma_pt(d, df, timeperiod=16, **_):
    return _strip_nan(_pta.hma(df["close"], length=timeperiod).values)


def _hma_ta(d, df, **_):
    return _empty()


_hma_ta._stub = True


def _hma_tu(d, df, timeperiod=16, **_):
    return _strip_nan(_tl.hma(_c64(d["close"]), period=timeperiod))


def _hma_fi(d, df, timeperiod=16, **_):
    return _strip_nan(_fi.HMA(df, timeperiod).values)


def _vwma_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.VWMA(d["close"], d["volume"], timeperiod=timeperiod))


def _vwma_tl(d, df, **_):
    return _empty()


_vwma_tl._stub = True


def _vwma_pt(d, df, timeperiod=20, **_):
    r = _pta.vwma(df["close"], df["volume"], length=timeperiod)
    return _strip_nan(r.values) if r is not None else _empty()


def _vwma_ta(d, df, **_):
    return _empty()


_vwma_ta._stub = True


def _vwma_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.vwma(_c64(d["close"]), _c64(d["volume"]), period=timeperiod))


def _vwma_fi(d, df, **_):
    return _empty()


_vwma_fi._stub = True


def _midpoint_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.MIDPOINT(d["close"], timeperiod=timeperiod))


def _midpoint_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.MIDPOINT(d["close"], timeperiod=timeperiod))


def _midpoint_pt(d, df, **_):
    return _empty()


_midpoint_pt._stub = True


def _midpoint_ta(d, df, **_):
    return _empty()


_midpoint_ta._stub = True


def _midpoint_tu(d, df, **_):
    return _empty()


_midpoint_tu._stub = True


def _midpoint_fi(d, df, **_):
    return _empty()


_midpoint_fi._stub = True


def _midprice_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.MIDPRICE(d["high"], d["low"], timeperiod=timeperiod))


def _midprice_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.MIDPRICE(d["high"], d["low"], timeperiod=timeperiod))


def _midprice_pt(d, df, **_):
    return _empty()


_midprice_pt._stub = True


def _midprice_ta(d, df, **_):
    return _empty()


_midprice_ta._stub = True


def _midprice_tu(d, df, **_):
    return _empty()


_midprice_tu._stub = True


def _midprice_fi(d, df, **_):
    return _empty()


_midprice_fi._stub = True


# ============================================================
# MOMENTUM
# ============================================================
def _rsi_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.RSI(d["close"], timeperiod=timeperiod))


def _rsi_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.RSI(d["close"], timeperiod=timeperiod))


def _rsi_pt(d, df, timeperiod=14, **_):
    return _strip_nan(_pta.rsi(df["close"], length=timeperiod).values)


def _rsi_ta(d, df, timeperiod=14, **_):
    from ta.momentum import RSIIndicator

    return _strip_nan(RSIIndicator(df["close"], window=timeperiod).rsi().values)


def _rsi_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.rsi(_c64(d["close"]), period=timeperiod))


def _rsi_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.RSI(df, timeperiod).values)


def _macd_ft(d, df, fastperiod=12, slowperiod=26, signalperiod=9, **_):
    import ferro_ta

    m, s, h = ferro_ta.MACD(
        d["close"],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    return _strip_nan(m)


def _macd_tl(d, df, fastperiod=12, slowperiod=26, signalperiod=9, **_):
    m, s, h = _talib.MACD(
        d["close"],
        fastperiod=fastperiod,
        slowperiod=slowperiod,
        signalperiod=signalperiod,
    )
    return _strip_nan(m)


def _macd_pt(d, df, fastperiod=12, slowperiod=26, signalperiod=9, **_):
    r = _pta.macd(df["close"], fast=fastperiod, slow=slowperiod, signal=signalperiod)
    return _first_col(r, "MACD_")


def _macd_ta(d, df, fastperiod=12, slowperiod=26, signalperiod=9, **_):
    from ta.trend import MACD

    return _strip_nan(
        MACD(
            df["close"],
            window_fast=fastperiod,
            window_slow=slowperiod,
            window_sign=signalperiod,
        )
        .macd()
        .values
    )


def _macd_tu(d, df, fastperiod=12, slowperiod=26, signalperiod=9, **_):
    m, s, h = _tl.macd(
        _c64(d["close"]),
        short_period=fastperiod,
        long_period=slowperiod,
        signal_period=signalperiod,
    )
    return _strip_nan(m)


def _macd_fi(d, df, fastperiod=12, slowperiod=26, signalperiod=9, **_):
    return _strip_nan(_fi.MACD(df, fastperiod, slowperiod, signalperiod)["MACD"].values)


def _stoch_ft(d, df, fastk_period=14, slowk_period=3, slowd_period=3, **_):
    import ferro_ta

    k, dd = ferro_ta.STOCH(
        d["high"],
        d["low"],
        d["close"],
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
    return _strip_nan(k)


def _stoch_tl(d, df, fastk_period=14, slowk_period=3, slowd_period=3, **_):
    k, dd = _talib.STOCH(
        d["high"],
        d["low"],
        d["close"],
        fastk_period=fastk_period,
        slowk_period=slowk_period,
        slowd_period=slowd_period,
    )
    return _strip_nan(k)


def _stoch_pt(d, df, fastk_period=14, slowk_period=3, slowd_period=3, **_):
    r = _pta.stoch(df["high"], df["low"], df["close"], k=fastk_period, d=slowd_period)
    return _first_col(r, "STOCHk_") if r is not None else _empty()


def _stoch_ta(d, df, fastk_period=14, **_):
    from ta.momentum import StochasticOscillator

    return _strip_nan(
        StochasticOscillator(df["high"], df["low"], df["close"], window=fastk_period)
        .stoch()
        .values
    )


def _stoch_tu(d, df, fastk_period=14, slowk_period=3, slowd_period=3, **_):
    k, dd = _tl.stoch(
        _c64(d["high"]),
        _c64(d["low"]),
        _c64(d["close"]),
        pct_k_period=fastk_period,
        pct_k_slowing_period=slowk_period,
        pct_d_period=slowd_period,
    )
    return _strip_nan(k)


def _stoch_fi(d, df, fastk_period=14, **_):
    return _strip_nan(_fi.STOCH(df, fastk_period).values)


def _cci_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.CCI(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _cci_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.CCI(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _cci_pt(d, df, timeperiod=14, **_):
    return _strip_nan(
        _pta.cci(df["high"], df["low"], df["close"], length=timeperiod).values
    )


def _cci_ta(d, df, timeperiod=14, **_):
    from ta.trend import CCIIndicator

    return _strip_nan(
        CCIIndicator(df["high"], df["low"], df["close"], window=timeperiod).cci().values
    )


def _cci_tu(d, df, timeperiod=14, **_):
    return _strip_nan(
        _tl.cci(_c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod)
    )


def _cci_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.CCI(df, timeperiod).values)


def _willr_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.WILLR(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _willr_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.WILLR(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _willr_pt(d, df, timeperiod=14, **_):
    return _strip_nan(
        _pta.willr(df["high"], df["low"], df["close"], length=timeperiod).values
    )


def _willr_ta(d, df, timeperiod=14, **_):
    from ta.momentum import WilliamsRIndicator

    return _strip_nan(
        WilliamsRIndicator(df["high"], df["low"], df["close"], lbp=timeperiod)
        .williams_r()
        .values
    )


def _willr_tu(d, df, timeperiod=14, **_):
    return _strip_nan(
        _tl.willr(_c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod)
    )


def _willr_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.WILLIAMS(df, timeperiod).values)


def _aroon_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    dn, up = ferro_ta.AROON(d["high"], d["low"], timeperiod=timeperiod)
    return _strip_nan(up)


def _aroon_tl(d, df, timeperiod=14, **_):
    dn, up = _talib.AROON(d["high"], d["low"], timeperiod=timeperiod)
    return _strip_nan(up)


def _aroon_pt(d, df, timeperiod=14, **_):
    r = _pta.aroon(df["high"], df["low"], length=timeperiod)
    return _first_col(r, "AROONU_") if r is not None else _empty()


def _aroon_ta(d, df, timeperiod=14, **_):
    from ta.trend import AroonIndicator

    return _strip_nan(
        AroonIndicator(df["high"], df["low"], window=timeperiod).aroon_up().values
    )


def _aroon_tu(d, df, timeperiod=14, **_):
    dn, up = _tl.aroon(_c64(d["high"]), _c64(d["low"]), period=timeperiod)
    return _strip_nan(up)


def _aroon_fi(d, df, **_):
    return _empty()


_aroon_fi._stub = True


def _aroonosc_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.AROONOSC(d["high"], d["low"], timeperiod=timeperiod))


def _aroonosc_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.AROONOSC(d["high"], d["low"], timeperiod=timeperiod))


def _aroonosc_pt(d, df, **_):
    return _empty()


_aroonosc_pt._stub = True


def _aroonosc_ta(d, df, **_):
    return _empty()


_aroonosc_ta._stub = True


def _aroonosc_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.aroonosc(_c64(d["high"]), _c64(d["low"]), period=timeperiod))


def _aroonosc_fi(d, df, **_):
    return _empty()


_aroonosc_fi._stub = True


def _adx_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.ADX(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _adx_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.ADX(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _adx_pt(d, df, timeperiod=14, **_):
    r = _pta.adx(df["high"], df["low"], df["close"], length=timeperiod)
    return _first_col(r, "ADX_")


def _adx_ta(d, df, timeperiod=14, **_):
    from ta.trend import ADXIndicator

    return _strip_nan(
        ADXIndicator(df["high"], df["low"], df["close"], window=timeperiod).adx().values
    )


def _adx_tu(d, df, timeperiod=14, **_):
    return _strip_nan(
        _tl.adx(_c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod)
    )


def _adx_fi(d, df, **_):
    return _empty()


_adx_fi._stub = True


def _mom_ft(d, df, timeperiod=10, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.MOM(d["close"], timeperiod=timeperiod))


def _mom_tl(d, df, timeperiod=10, **_):
    return _strip_nan(_talib.MOM(d["close"], timeperiod=timeperiod))


def _mom_pt(d, df, timeperiod=10, **_):
    return _strip_nan(_pta.mom(df["close"], length=timeperiod).values)


def _mom_ta(d, df, **_):
    return _empty()


_mom_ta._stub = True


def _mom_tu(d, df, timeperiod=10, **_):
    return _strip_nan(_tl.mom(_c64(d["close"]), period=timeperiod))


def _mom_fi(d, df, timeperiod=10, **_):
    return _strip_nan(_fi.MOM(df, timeperiod).values)


def _roc_ft(d, df, timeperiod=10, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.ROC(d["close"], timeperiod=timeperiod))


def _roc_tl(d, df, timeperiod=10, **_):
    return _strip_nan(_talib.ROC(d["close"], timeperiod=timeperiod))


def _roc_pt(d, df, timeperiod=10, **_):
    return _strip_nan(_pta.roc(df["close"], length=timeperiod).values)


def _roc_ta(d, df, timeperiod=10, **_):
    from ta.momentum import ROCIndicator

    return _strip_nan(ROCIndicator(df["close"], window=timeperiod).roc().values)


def _roc_tu(d, df, timeperiod=10, **_):
    return _strip_nan(_tl.roc(_c64(d["close"]), period=timeperiod) * 100.0)


def _roc_fi(d, df, timeperiod=10, **_):
    return _strip_nan(_fi.ROC(df, timeperiod).values)


def _cmo_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.CMO(d["close"], timeperiod=timeperiod))


def _cmo_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.CMO(d["close"], timeperiod=timeperiod))


def _cmo_pt(d, df, timeperiod=14, **_):
    return _strip_nan(_pta.cmo(df["close"], length=timeperiod).values)


def _cmo_ta(d, df, **_):
    return _empty()


_cmo_ta._stub = True


def _cmo_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.cmo(_c64(d["close"]), period=timeperiod))


def _cmo_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.CMO(df, timeperiod).values)


def _ppo_ft(d, df, fastperiod=12, slowperiod=26, **_):
    import ferro_ta

    ppo, sig, hist = ferro_ta.PPO(
        d["close"], fastperiod=fastperiod, slowperiod=slowperiod
    )
    return _strip_nan(ppo)


def _ppo_tl(d, df, fastperiod=12, slowperiod=26, **_):
    return _strip_nan(
        _talib.PPO(d["close"], fastperiod=fastperiod, slowperiod=slowperiod)
    )


def _ppo_pt(d, df, fastperiod=12, slowperiod=26, **_):
    r = _pta.ppo(df["close"], fast=fastperiod, slow=slowperiod)
    return _strip_nan(r.iloc[:, 0].values) if r is not None else _empty()


def _ppo_ta(d, df, **_):
    return _empty()


_ppo_ta._stub = True


def _ppo_tu(d, df, fastperiod=12, slowperiod=26, **_):
    return _strip_nan(
        _tl.ppo(_c64(d["close"]), short_period=fastperiod, long_period=slowperiod)
    )


def _ppo_fi(d, df, fastperiod=12, slowperiod=26, **_):
    return _strip_nan(_fi.PPO(df, fastperiod, slowperiod).values)


def _trix_ft(d, df, timeperiod=18, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.TRIX(d["close"], timeperiod=timeperiod))


def _trix_tl(d, df, timeperiod=18, **_):
    return _strip_nan(_talib.TRIX(d["close"], timeperiod=timeperiod))


def _trix_pt(d, df, timeperiod=18, **_):
    r = _pta.trix(df["close"], length=timeperiod)
    return _strip_nan(r.iloc[:, 0].values) if r is not None else _empty()


def _trix_ta(d, df, timeperiod=18, **_):
    from ta.trend import TRIXIndicator

    return _strip_nan(TRIXIndicator(df["close"], window=timeperiod).trix().values)


def _trix_tu(d, df, timeperiod=18, **_):
    return _strip_nan(_tl.trix(_c64(d["close"]), period=timeperiod))


def _trix_fi(d, df, timeperiod=18, **_):
    return _strip_nan(_fi.TRIX(df, timeperiod).values)


def _tsf_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.TSF(d["close"], timeperiod=timeperiod))


def _tsf_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.TSF(d["close"], timeperiod=timeperiod))


def _tsf_pt(d, df, **_):
    return _empty()


_tsf_pt._stub = True


def _tsf_ta(d, df, **_):
    return _empty()


_tsf_ta._stub = True


def _tsf_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.tsf(_c64(d["close"]), period=timeperiod))


def _tsf_fi(d, df, **_):
    return _empty()


_tsf_fi._stub = True


def _ultosc_ft(d, df, timeperiod1=7, timeperiod2=14, timeperiod3=28, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.ULTOSC(
            d["high"],
            d["low"],
            d["close"],
            timeperiod1=timeperiod1,
            timeperiod2=timeperiod2,
            timeperiod3=timeperiod3,
        )
    )


def _ultosc_tl(d, df, timeperiod1=7, timeperiod2=14, timeperiod3=28, **_):
    return _strip_nan(
        _talib.ULTOSC(
            d["high"],
            d["low"],
            d["close"],
            timeperiod1=timeperiod1,
            timeperiod2=timeperiod2,
            timeperiod3=timeperiod3,
        )
    )


def _ultosc_pt(d, df, **_):
    return _empty()


_ultosc_pt._stub = True


def _ultosc_ta(d, df, timeperiod1=7, timeperiod2=14, timeperiod3=28, **_):
    from ta.momentum import UltimateOscillator

    return _strip_nan(
        UltimateOscillator(
            df["high"],
            df["low"],
            df["close"],
            window1=timeperiod1,
            window2=timeperiod2,
            window3=timeperiod3,
        )
        .ultimate_oscillator()
        .values
    )


def _ultosc_tu(d, df, timeperiod1=7, timeperiod2=14, timeperiod3=28, **_):
    return _strip_nan(
        _tl.ultosc(
            _c64(d["high"]),
            _c64(d["low"]),
            _c64(d["close"]),
            short_period=timeperiod1,
            medium_period=timeperiod2,
            long_period=timeperiod3,
        )
    )


def _ultosc_fi(d, df, **_):
    return _empty()


_ultosc_fi._stub = True


def _bop_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.BOP(d["open"], d["high"], d["low"], d["close"]))


def _bop_tl(d, df, **_):
    return _strip_nan(_talib.BOP(d["open"], d["high"], d["low"], d["close"]))


def _bop_pt(d, df, **_):
    r = _pta.bop(df["open"], df["high"], df["low"], df["close"])
    return _strip_nan(r.values) if r is not None else _empty()


def _bop_ta(d, df, **_):
    return _empty()


_bop_ta._stub = True


def _bop_tu(d, df, **_):
    return _strip_nan(
        _tl.bop(_c64(d["open"]), _c64(d["high"]), _c64(d["low"]), _c64(d["close"]))
    )


def _bop_fi(d, df, **_):
    return _empty()


_bop_fi._stub = True


def _plusdi_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.PLUS_DI(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _plusdi_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.PLUS_DI(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _plusdi_pt(d, df, timeperiod=14, **_):
    r = _pta.adx(df["high"], df["low"], df["close"], length=timeperiod)
    return _first_col(r, "DMP_") if r is not None else _empty()


def _plusdi_ta(d, df, **_):
    return _empty()


_plusdi_ta._stub = True


def _plusdi_tu(d, df, timeperiod=14, **_):
    pdi, mdi = _tl.di(
        _c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod
    )
    return _strip_nan(pdi)


def _plusdi_fi(d, df, **_):
    return _empty()


_plusdi_fi._stub = True


def _minusdi_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.MINUS_DI(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _minusdi_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.MINUS_DI(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _minusdi_pt(d, df, **_):
    return _empty()


_minusdi_pt._stub = True


def _minusdi_ta(d, df, **_):
    return _empty()


_minusdi_ta._stub = True


def _minusdi_tu(d, df, timeperiod=14, **_):
    pdi, mdi = _tl.di(
        _c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod
    )
    return _strip_nan(mdi)


def _minusdi_fi(d, df, **_):
    return _empty()


_minusdi_fi._stub = True


# ============================================================
# VOLATILITY
# ============================================================
def _bb_ft(d, df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, **_):
    import ferro_ta

    u, m, l = ferro_ta.BBANDS(
        d["close"], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn
    )
    return _strip_nan(u)


def _bb_tl(d, df, timeperiod=20, nbdevup=2.0, nbdevdn=2.0, **_):
    u, m, l = _talib.BBANDS(
        d["close"], timeperiod=timeperiod, nbdevup=nbdevup, nbdevdn=nbdevdn
    )
    return _strip_nan(u)


def _bb_pt(d, df, timeperiod=20, nbdevup=2.0, **_):
    r = _pta.bbands(df["close"], length=timeperiod, std=nbdevup)
    return _first_col(r, "BBU_")


def _bb_ta(d, df, timeperiod=20, nbdevup=2.0, **_):
    from ta.volatility import BollingerBands

    return _strip_nan(
        BollingerBands(df["close"], window=timeperiod, window_dev=nbdevup)
        .bollinger_hband()
        .values
    )


def _bb_tu(d, df, timeperiod=20, nbdevup=2.0, **_):
    lo, mi, up = _tl.bbands(_c64(d["close"]), period=timeperiod, stddev=nbdevup)
    return _strip_nan(up)


def _bb_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.BBANDS(df, timeperiod)["BB_UPPER"].values)


def _atr_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.ATR(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _atr_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.ATR(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _atr_pt(d, df, timeperiod=14, **_):
    return _strip_nan(
        _pta.atr(df["high"], df["low"], df["close"], length=timeperiod).values
    )


def _atr_ta(d, df, timeperiod=14, **_):
    from ta.volatility import AverageTrueRange

    return _strip_nan(
        AverageTrueRange(df["high"], df["low"], df["close"], window=timeperiod)
        .average_true_range()
        .values
    )


def _atr_tu(d, df, timeperiod=14, **_):
    return _strip_nan(
        _tl.atr(_c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod)
    )


def _atr_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.ATR(df, timeperiod).values)


def _natr_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.NATR(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _natr_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.NATR(d["high"], d["low"], d["close"], timeperiod=timeperiod)
    )


def _natr_pt(d, df, timeperiod=14, **_):
    return _strip_nan(
        _pta.natr(df["high"], df["low"], df["close"], length=timeperiod).values
    )


def _natr_ta(d, df, **_):
    return _empty()


_natr_ta._stub = True


def _natr_tu(d, df, timeperiod=14, **_):
    return _strip_nan(
        _tl.natr(_c64(d["high"]), _c64(d["low"]), _c64(d["close"]), period=timeperiod)
    )


def _natr_fi(d, df, **_):
    return _empty()


_natr_fi._stub = True


def _trange_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.TRANGE(d["high"], d["low"], d["close"]))


def _trange_tl(d, df, **_):
    return _strip_nan(_talib.TRANGE(d["high"], d["low"], d["close"]))


def _trange_pt(d, df, **_):
    r = _pta.true_range(df["high"], df["low"], df["close"])
    return _strip_nan(r.values) if r is not None else _empty()


def _trange_ta(d, df, **_):
    return _empty()


_trange_ta._stub = True


def _trange_tu(d, df, **_):
    return _strip_nan(_tl.tr(_c64(d["high"]), _c64(d["low"]), _c64(d["close"])))


def _trange_fi(d, df, **_):
    return _strip_nan(_fi.TR(df).values)


def _stddev_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.STDDEV(d["close"], timeperiod=timeperiod))


def _stddev_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.STDDEV(d["close"], timeperiod=timeperiod))


def _stddev_pt(d, df, timeperiod=20, **_):
    r = _pta.stdev(df["close"], length=timeperiod)
    return _strip_nan(r.values) if r is not None else _empty()


def _stddev_ta(d, df, **_):
    return _empty()


_stddev_ta._stub = True


def _stddev_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.stddev(_c64(d["close"]), period=timeperiod))


def _stddev_fi(d, df, timeperiod=20, **_):
    return _strip_nan(_fi.MSD(df, timeperiod).values)


def _var_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.VAR(d["close"], timeperiod=timeperiod))


def _var_tl(d, df, timeperiod=20, **_):
    return _strip_nan(_talib.VAR(d["close"], timeperiod=timeperiod))


def _var_pt(d, df, timeperiod=20, **_):
    r = _pta.variance(df["close"], length=timeperiod)
    return _strip_nan(r.values) if r is not None else _empty()


def _var_ta(d, df, **_):
    return _empty()


_var_ta._stub = True


def _var_tu(d, df, timeperiod=20, **_):
    return _strip_nan(_tl.var(_c64(d["close"]), period=timeperiod))


def _var_fi(d, df, **_):
    return _empty()


_var_fi._stub = True


def _sar_ft(d, df, acceleration=0.02, maximum=0.2, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.SAR(d["high"], d["low"], acceleration=acceleration, maximum=maximum)
    )


def _sar_tl(d, df, acceleration=0.02, maximum=0.2, **_):
    return _strip_nan(
        _talib.SAR(d["high"], d["low"], acceleration=acceleration, maximum=maximum)
    )


def _sar_pt(d, df, **_):
    return _empty()


_sar_pt._stub = True


def _sar_ta(d, df, **_):
    return _empty()


_sar_ta._stub = True


def _sar_tu(d, df, acceleration=0.02, maximum=0.2, **_):
    return _strip_nan(
        _tl.psar(
            _c64(d["high"]),
            _c64(d["low"]),
            acceleration_factor_step=acceleration,
            acceleration_factor_maximum=maximum,
        )
    )


def _sar_fi(d, df, **_):
    return _empty()


_sar_fi._stub = True


def _kc_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    u, m, l = ferro_ta.KELTNER_CHANNELS(
        d["high"], d["low"], d["close"], timeperiod=timeperiod
    )
    return _strip_nan(u)


def _kc_tl(d, df, **_):
    return _empty()


_kc_tl._stub = True


def _kc_pt(d, df, timeperiod=20, **_):
    r = _pta.kc(df["high"], df["low"], df["close"], length=timeperiod)
    if r is None:
        return _empty()
    col = next(
        (c for c in r.columns if "UCe" in c or "UB" in c or c.endswith("U")), None
    )
    return _strip_nan(r[col].values) if col else _first_col(r, "KC")


def _kc_ta(d, df, timeperiod=20, **_):
    from ta.volatility import KeltnerChannel

    return _strip_nan(
        KeltnerChannel(df["high"], df["low"], df["close"], window=timeperiod)
        .keltner_channel_hband()
        .values
    )


def _kc_tu(d, df, **_):
    return _empty()


_kc_tu._stub = True


def _kc_fi(d, df, **_):
    return _empty()


_kc_fi._stub = True


def _donchian_ft(d, df, timeperiod=20, **_):
    import ferro_ta

    u, m, l = ferro_ta.DONCHIAN(d["high"], d["low"], timeperiod=timeperiod)
    return _strip_nan(u)


def _donchian_tl(d, df, **_):
    return _empty()


_donchian_tl._stub = True


def _donchian_pt(d, df, timeperiod=20, **_):
    r = _pta.donchian(
        df["high"], df["low"], lower_length=timeperiod, upper_length=timeperiod
    )
    return _first_col(r, "DCU_") if r is not None else _empty()


def _donchian_ta(d, df, timeperiod=20, **_):
    from ta.volatility import DonchianChannel

    return _strip_nan(
        DonchianChannel(df["high"], df["low"], df["close"], window=timeperiod)
        .donchian_channel_hband()
        .values
    )


def _donchian_tu(d, df, **_):
    return _empty()


_donchian_tu._stub = True


def _donchian_fi(d, df, **_):
    return _empty()


_donchian_fi._stub = True


def _supertrend_ft(d, df, timeperiod=7, **_):
    import ferro_ta

    st, dir_ = ferro_ta.SUPERTREND(
        d["high"], d["low"], d["close"], timeperiod=timeperiod
    )
    return _strip_nan(st)


def _supertrend_tl(d, df, **_):
    return _empty()


_supertrend_tl._stub = True


def _supertrend_pt(d, df, timeperiod=7, **_):
    r = _pta.supertrend(df["high"], df["low"], df["close"], length=timeperiod)
    return _first_col(r, "SUPERT_") if r is not None else _empty()


def _supertrend_ta(d, df, **_):
    return _empty()


_supertrend_ta._stub = True


def _supertrend_tu(d, df, **_):
    return _empty()


_supertrend_tu._stub = True


def _supertrend_fi(d, df, **_):
    return _empty()


_supertrend_fi._stub = True


def _chop_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.CHOPPINESS_INDEX(
            d["high"], d["low"], d["close"], timeperiod=timeperiod
        )
    )


def _chop_tl(d, df, **_):
    return _empty()


_chop_tl._stub = True


def _chop_pt(d, df, timeperiod=14, **_):
    r = _pta.chop(df["high"], df["low"], df["close"], length=timeperiod)
    return _strip_nan(r.values) if r is not None else _empty()


def _chop_ta(d, df, **_):
    return _empty()


_chop_ta._stub = True


def _chop_tu(d, df, **_):
    return _empty()


_chop_tu._stub = True


def _chop_fi(d, df, **_):
    return _empty()


_chop_fi._stub = True


# ============================================================
# VOLUME
# ============================================================
def _obv_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.OBV(d["close"], d["volume"]))


def _obv_tl(d, df, **_):
    return _strip_nan(_talib.OBV(d["close"], d["volume"]))


def _obv_pt(d, df, **_):
    return _strip_nan(_pta.obv(df["close"], df["volume"]).values)


def _obv_ta(d, df, **_):
    from ta.volume import OnBalanceVolumeIndicator

    return _strip_nan(
        OnBalanceVolumeIndicator(df["close"], df["volume"]).on_balance_volume().values
    )


def _obv_tu(d, df, **_):
    return _strip_nan(_tl.obv(_c64(d["close"]), _c64(d["volume"])))


def _obv_fi(d, df, **_):
    return _strip_nan(_fi.OBV(df).values)


def _ad_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.AD(d["high"], d["low"], d["close"], d["volume"]))


def _ad_tl(d, df, **_):
    return _strip_nan(_talib.AD(d["high"], d["low"], d["close"], d["volume"]))


def _ad_pt(d, df, **_):
    return _strip_nan(_pta.ad(df["high"], df["low"], df["close"], df["volume"]).values)


def _ad_ta(d, df, **_):
    from ta.volume import AccDistIndexIndicator

    return _strip_nan(
        AccDistIndexIndicator(df["high"], df["low"], df["close"], df["volume"])
        .acc_dist_index()
        .values
    )


def _ad_tu(d, df, **_):
    return _strip_nan(
        _tl.ad(_c64(d["high"]), _c64(d["low"]), _c64(d["close"]), _c64(d["volume"]))
    )


def _ad_fi(d, df, **_):
    return _empty()


_ad_fi._stub = True


def _adosc_ft(d, df, fastperiod=3, slowperiod=10, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.ADOSC(
            d["high"],
            d["low"],
            d["close"],
            d["volume"],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
        )
    )


def _adosc_tl(d, df, fastperiod=3, slowperiod=10, **_):
    return _strip_nan(
        _talib.ADOSC(
            d["high"],
            d["low"],
            d["close"],
            d["volume"],
            fastperiod=fastperiod,
            slowperiod=slowperiod,
        )
    )


def _adosc_pt(d, df, fastperiod=3, slowperiod=10, **_):
    return _strip_nan(
        _pta.adosc(
            df["high"],
            df["low"],
            df["close"],
            df["volume"],
            fast=fastperiod,
            slow=slowperiod,
        ).values
    )


def _adosc_ta(d, df, **_):
    return _empty()


_adosc_ta._stub = True


def _adosc_tu(d, df, fastperiod=3, slowperiod=10, **_):
    return _strip_nan(
        _tl.adosc(
            _c64(d["high"]),
            _c64(d["low"]),
            _c64(d["close"]),
            _c64(d["volume"]),
            short_period=fastperiod,
            long_period=slowperiod,
        )
    )


def _adosc_fi(d, df, **_):
    return _empty()


_adosc_fi._stub = True


def _mfi_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.MFI(
            d["high"], d["low"], d["close"], d["volume"], timeperiod=timeperiod
        )
    )


def _mfi_tl(d, df, timeperiod=14, **_):
    return _strip_nan(
        _talib.MFI(d["high"], d["low"], d["close"], d["volume"], timeperiod=timeperiod)
    )


def _mfi_pt(d, df, timeperiod=14, **_):
    return _strip_nan(
        _pta.mfi(
            df["high"], df["low"], df["close"], df["volume"], length=timeperiod
        ).values
    )


def _mfi_ta(d, df, timeperiod=14, **_):
    from ta.volume import MFIIndicator

    return _strip_nan(
        MFIIndicator(
            df["high"], df["low"], df["close"], df["volume"], window=timeperiod
        )
        .money_flow_index()
        .values
    )


def _mfi_tu(d, df, timeperiod=14, **_):
    return _strip_nan(
        _tl.mfi(
            _c64(d["high"]),
            _c64(d["low"]),
            _c64(d["close"]),
            _c64(d["volume"]),
            period=timeperiod,
        )
    )


def _mfi_fi(d, df, timeperiod=14, **_):
    return _strip_nan(_fi.MFI(df, timeperiod).values)


def _vwap_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.VWAP(d["high"], d["low"], d["close"], d["volume"]))


def _vwap_tl(d, df, **_):
    return _empty()


_vwap_tl._stub = True


def _vwap_pt(d, df, **_):
    r = _pta.vwap(df["high"], df["low"], df["close"], df["volume"])
    return _strip_nan(r.values) if r is not None else _empty()


def _vwap_ta(d, df, **_):
    return _empty()


_vwap_ta._stub = True


def _vwap_tu(d, df, **_):
    return _empty()


_vwap_tu._stub = True


def _vwap_fi(d, df, **_):
    return _strip_nan(_fi.VWAP(df).values)


# ============================================================
# PRICE TRANSFORM
# ============================================================
def _avgprice_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.AVGPRICE(d["open"], d["high"], d["low"], d["close"]))


def _avgprice_tl(d, df, **_):
    return _strip_nan(_talib.AVGPRICE(d["open"], d["high"], d["low"], d["close"]))


def _avgprice_pt(d, df, **_):
    return _empty()


_avgprice_pt._stub = True


def _avgprice_ta(d, df, **_):
    return _empty()


_avgprice_ta._stub = True


def _avgprice_tu(d, df, **_):
    return _strip_nan(
        _tl.avgprice(_c64(d["open"]), _c64(d["high"]), _c64(d["low"]), _c64(d["close"]))
    )


def _avgprice_fi(d, df, **_):
    return _empty()


_avgprice_fi._stub = True


def _medprice_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.MEDPRICE(d["high"], d["low"]))


def _medprice_tl(d, df, **_):
    return _strip_nan(_talib.MEDPRICE(d["high"], d["low"]))


def _medprice_pt(d, df, **_):
    return _empty()


_medprice_pt._stub = True


def _medprice_ta(d, df, **_):
    return _empty()


_medprice_ta._stub = True


def _medprice_tu(d, df, **_):
    return _strip_nan(_tl.medprice(_c64(d["high"]), _c64(d["low"])))


def _medprice_fi(d, df, **_):
    return _empty()


_medprice_fi._stub = True


def _typprice_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.TYPPRICE(d["high"], d["low"], d["close"]))


def _typprice_tl(d, df, **_):
    return _strip_nan(_talib.TYPPRICE(d["high"], d["low"], d["close"]))


def _typprice_pt(d, df, **_):
    return _empty()


_typprice_pt._stub = True


def _typprice_ta(d, df, **_):
    return _empty()


_typprice_ta._stub = True


def _typprice_tu(d, df, **_):
    return _strip_nan(_tl.typprice(_c64(d["high"]), _c64(d["low"]), _c64(d["close"])))


def _typprice_fi(d, df, **_):
    return _empty()


_typprice_fi._stub = True


def _wclprice_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.WCLPRICE(d["high"], d["low"], d["close"]))


def _wclprice_tl(d, df, **_):
    return _strip_nan(_talib.WCLPRICE(d["high"], d["low"], d["close"]))


def _wclprice_pt(d, df, **_):
    return _empty()


_wclprice_pt._stub = True


def _wclprice_ta(d, df, **_):
    return _empty()


_wclprice_ta._stub = True


def _wclprice_tu(d, df, **_):
    return _strip_nan(_tl.wcprice(_c64(d["high"]), _c64(d["low"]), _c64(d["close"])))


def _wclprice_fi(d, df, **_):
    return _empty()


_wclprice_fi._stub = True


# ============================================================
# MATH
# ============================================================
def _sqrt_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.SQRT(d["close"]))


def _sqrt_tl(d, df, **_):
    return _strip_nan(_talib.SQRT(d["close"]))


def _sqrt_pt(d, df, **_):
    return _empty()


_sqrt_pt._stub = True


def _sqrt_ta(d, df, **_):
    return _empty()


_sqrt_ta._stub = True


def _sqrt_tu(d, df, **_):
    return _strip_nan(_tl.sqrt(_c64(d["close"])))


def _sqrt_fi(d, df, **_):
    return _empty()


_sqrt_fi._stub = True


def _log10_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.LOG10(d["close"]))


def _log10_tl(d, df, **_):
    return _strip_nan(_talib.LOG10(d["close"]))


def _log10_pt(d, df, **_):
    return _empty()


_log10_pt._stub = True


def _log10_ta(d, df, **_):
    return _empty()


_log10_ta._stub = True


def _log10_tu(d, df, **_):
    return _strip_nan(_tl.log10(_c64(d["close"])))


def _log10_fi(d, df, **_):
    return _empty()


_log10_fi._stub = True


def _add_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.ADD(d["high"], d["low"]))


def _add_tl(d, df, **_):
    return _strip_nan(_talib.ADD(d["high"], d["low"]))


def _add_pt(d, df, **_):
    return _empty()


_add_pt._stub = True


def _add_ta(d, df, **_):
    return _empty()


_add_ta._stub = True


def _add_tu(d, df, **_):
    return _strip_nan(_tl.add(_c64(d["high"]), _c64(d["low"])))


def _add_fi(d, df, **_):
    return _empty()


_add_fi._stub = True


# ============================================================
# STATISTICS
# ============================================================
def _linearreg_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.LINEARREG(d["close"], timeperiod=timeperiod))


def _linearreg_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.LINEARREG(d["close"], timeperiod=timeperiod))


def _linearreg_pt(d, df, **_):
    return _empty()


_linearreg_pt._stub = True


def _linearreg_ta(d, df, **_):
    return _empty()


_linearreg_ta._stub = True


def _linearreg_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.linreg(_c64(d["close"]), period=timeperiod))


def _linearreg_fi(d, df, **_):
    return _empty()


_linearreg_fi._stub = True


def _linreg_slope_ft(d, df, timeperiod=14, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.LINEARREG_SLOPE(d["close"], timeperiod=timeperiod))


def _linreg_slope_tl(d, df, timeperiod=14, **_):
    return _strip_nan(_talib.LINEARREG_SLOPE(d["close"], timeperiod=timeperiod))


def _linreg_slope_pt(d, df, **_):
    return _empty()


_linreg_slope_pt._stub = True


def _linreg_slope_ta(d, df, **_):
    return _empty()


_linreg_slope_ta._stub = True


def _linreg_slope_tu(d, df, timeperiod=14, **_):
    return _strip_nan(_tl.linregslope(_c64(d["close"]), period=timeperiod))


def _linreg_slope_fi(d, df, **_):
    return _empty()


_linreg_slope_fi._stub = True


def _correl_ft(d, df, timeperiod=30, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.CORREL(d["high"], d["low"], timeperiod=timeperiod))


def _correl_tl(d, df, timeperiod=30, **_):
    return _strip_nan(_talib.CORREL(d["high"], d["low"], timeperiod=timeperiod))


def _correl_pt(d, df, **_):
    return _empty()


_correl_pt._stub = True


def _correl_ta(d, df, **_):
    return _empty()


_correl_ta._stub = True


def _correl_tu(d, df, **_):
    return _empty()


_correl_tu._stub = True


def _correl_fi(d, df, **_):
    return _empty()


_correl_fi._stub = True


def _beta_ft(d, df, timeperiod=5, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.BETA(d["high"], d["low"], timeperiod=timeperiod))


def _beta_tl(d, df, timeperiod=5, **_):
    return _strip_nan(_talib.BETA(d["high"], d["low"], timeperiod=timeperiod))


def _beta_pt(d, df, **_):
    return _empty()


_beta_pt._stub = True


def _beta_ta(d, df, **_):
    return _empty()


_beta_ta._stub = True


def _beta_tu(d, df, **_):
    return _empty()


_beta_tu._stub = True


def _beta_fi(d, df, **_):
    return _empty()


_beta_fi._stub = True


# ============================================================
# CYCLE
# ============================================================
def _ht_dcperiod_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.HT_DCPERIOD(d["close"]))


def _ht_dcperiod_tl(d, df, **_):
    return _strip_nan(_talib.HT_DCPERIOD(d["close"]))


def _ht_dcperiod_pt(d, df, **_):
    return _empty()


_ht_dcperiod_pt._stub = True


def _ht_dcperiod_ta(d, df, **_):
    return _empty()


_ht_dcperiod_ta._stub = True


def _ht_dcperiod_tu(d, df, **_):
    return _empty()


_ht_dcperiod_tu._stub = True


def _ht_dcperiod_fi(d, df, **_):
    return _empty()


_ht_dcperiod_fi._stub = True


def _ht_trendmode_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(ferro_ta.HT_TRENDMODE(d["close"]).astype(float))


def _ht_trendmode_tl(d, df, **_):
    return _strip_nan(_talib.HT_TRENDMODE(d["close"]).astype(float))


def _ht_trendmode_pt(d, df, **_):
    return _empty()


_ht_trendmode_pt._stub = True


def _ht_trendmode_ta(d, df, **_):
    return _empty()


_ht_trendmode_ta._stub = True


def _ht_trendmode_tu(d, df, **_):
    return _empty()


_ht_trendmode_tu._stub = True


def _ht_trendmode_fi(d, df, **_):
    return _empty()


_ht_trendmode_fi._stub = True


# ============================================================
# CANDLESTICK PATTERNS
# ============================================================
def _cdlengulfing_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.CDLENGULFING(d["open"], d["high"], d["low"], d["close"]).astype(float)
    )


def _cdlengulfing_tl(d, df, **_):
    return _strip_nan(
        _talib.CDLENGULFING(d["open"], d["high"], d["low"], d["close"]).astype(float)
    )


def _cdlengulfing_pt(d, df, **_):
    return _empty()


_cdlengulfing_pt._stub = True


def _cdlengulfing_ta(d, df, **_):
    return _empty()


_cdlengulfing_ta._stub = True


def _cdlengulfing_tu(d, df, **_):
    return _empty()


_cdlengulfing_tu._stub = True


def _cdlengulfing_fi(d, df, **_):
    return _empty()


_cdlengulfing_fi._stub = True


def _cdldoji_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.CDLDOJI(d["open"], d["high"], d["low"], d["close"]).astype(float)
    )


def _cdldoji_tl(d, df, **_):
    return _strip_nan(
        _talib.CDLDOJI(d["open"], d["high"], d["low"], d["close"]).astype(float)
    )


def _cdldoji_pt(d, df, **_):
    return _empty()


_cdldoji_pt._stub = True


def _cdldoji_ta(d, df, **_):
    return _empty()


_cdldoji_ta._stub = True


def _cdldoji_tu(d, df, **_):
    return _empty()


_cdldoji_tu._stub = True


def _cdldoji_fi(d, df, **_):
    return _empty()


_cdldoji_fi._stub = True


def _cdlhammer_ft(d, df, **_):
    import ferro_ta

    return _strip_nan(
        ferro_ta.CDLHAMMER(d["open"], d["high"], d["low"], d["close"]).astype(float)
    )


def _cdlhammer_tl(d, df, **_):
    return _strip_nan(
        _talib.CDLHAMMER(d["open"], d["high"], d["low"], d["close"]).astype(float)
    )


def _cdlhammer_pt(d, df, **_):
    return _empty()


_cdlhammer_pt._stub = True


def _cdlhammer_ta(d, df, **_):
    return _empty()


_cdlhammer_ta._stub = True


def _cdlhammer_tu(d, df, **_):
    return _empty()


_cdlhammer_tu._stub = True


def _cdlhammer_fi(d, df, **_):
    return _empty()


_cdlhammer_fi._stub = True

# ============================================================
# REGISTRY BUILD
# ============================================================
REGISTRY: dict[tuple[str, Any], Any] = {}


def _reg(ind, ft, tl, pt, ta_, tu, fi):
    """
    Register wrappers for a given indicator across all libraries.

    Wrappers marked ._stub = True (no-op return _empty()) are not registered,
    so execute_indicator raises KeyError for unsupported (lib, ind). Speed
    benchmarks then skip those pairs and the table shows N/A.
    """
    for lib, fn in [
        ("ferro_ta", ft),
        ("talib", tl),
        ("pandas_ta", pt),
        ("ta", ta_),
        ("tulipy", tu),
        ("finta", fi),
    ]:
        if getattr(fn, "_stub", False):
            continue
        REGISTRY[(lib, ind)] = fn


_reg("SMA", _sma_ft, _sma_tl, _sma_pt, _sma_ta, _sma_tu, _sma_fi)
_reg("EMA", _ema_ft, _ema_tl, _ema_pt, _ema_ta, _ema_tu, _ema_fi)
_reg("WMA", _wma_ft, _wma_tl, _wma_pt, _wma_ta, _wma_tu, _wma_fi)
_reg("DEMA", _dema_ft, _dema_tl, _dema_pt, _dema_ta, _dema_tu, _dema_fi)
_reg("TEMA", _tema_ft, _tema_tl, _tema_pt, _tema_ta, _tema_tu, _tema_fi)
_reg("T3", _t3_ft, _t3_tl, _t3_pt, _t3_ta, _t3_tu, _t3_fi)
_reg("TRIMA", _trima_ft, _trima_tl, _trima_pt, _trima_ta, _trima_tu, _trima_fi)
_reg("KAMA", _kama_ft, _kama_tl, _kama_pt, _kama_ta, _kama_tu, _kama_fi)
_reg("HULL_MA", _hma_ft, _hma_tl, _hma_pt, _hma_ta, _hma_tu, _hma_fi)
_reg("VWMA", _vwma_ft, _vwma_tl, _vwma_pt, _vwma_ta, _vwma_tu, _vwma_fi)
_reg(
    "MIDPOINT",
    _midpoint_ft,
    _midpoint_tl,
    _midpoint_pt,
    _midpoint_ta,
    _midpoint_tu,
    _midpoint_fi,
)
_reg(
    "MIDPRICE",
    _midprice_ft,
    _midprice_tl,
    _midprice_pt,
    _midprice_ta,
    _midprice_tu,
    _midprice_fi,
)
_reg("RSI", _rsi_ft, _rsi_tl, _rsi_pt, _rsi_ta, _rsi_tu, _rsi_fi)
_reg("MACD", _macd_ft, _macd_tl, _macd_pt, _macd_ta, _macd_tu, _macd_fi)
_reg("STOCH", _stoch_ft, _stoch_tl, _stoch_pt, _stoch_ta, _stoch_tu, _stoch_fi)
_reg("CCI", _cci_ft, _cci_tl, _cci_pt, _cci_ta, _cci_tu, _cci_fi)
_reg("WILLR", _willr_ft, _willr_tl, _willr_pt, _willr_ta, _willr_tu, _willr_fi)
_reg("AROON", _aroon_ft, _aroon_tl, _aroon_pt, _aroon_ta, _aroon_tu, _aroon_fi)
_reg(
    "AROONOSC",
    _aroonosc_ft,
    _aroonosc_tl,
    _aroonosc_pt,
    _aroonosc_ta,
    _aroonosc_tu,
    _aroonosc_fi,
)
_reg("ADX", _adx_ft, _adx_tl, _adx_pt, _adx_ta, _adx_tu, _adx_fi)
_reg("MOM", _mom_ft, _mom_tl, _mom_pt, _mom_ta, _mom_tu, _mom_fi)
_reg("ROC", _roc_ft, _roc_tl, _roc_pt, _roc_ta, _roc_tu, _roc_fi)
_reg("CMO", _cmo_ft, _cmo_tl, _cmo_pt, _cmo_ta, _cmo_tu, _cmo_fi)
_reg("PPO", _ppo_ft, _ppo_tl, _ppo_pt, _ppo_ta, _ppo_tu, _ppo_fi)
_reg("TRIX", _trix_ft, _trix_tl, _trix_pt, _trix_ta, _trix_tu, _trix_fi)
_reg("TSF", _tsf_ft, _tsf_tl, _tsf_pt, _tsf_ta, _tsf_tu, _tsf_fi)
_reg("ULTOSC", _ultosc_ft, _ultosc_tl, _ultosc_pt, _ultosc_ta, _ultosc_tu, _ultosc_fi)
_reg("BOP", _bop_ft, _bop_tl, _bop_pt, _bop_ta, _bop_tu, _bop_fi)
_reg("PLUS_DI", _plusdi_ft, _plusdi_tl, _plusdi_pt, _plusdi_ta, _plusdi_tu, _plusdi_fi)
_reg(
    "MINUS_DI",
    _minusdi_ft,
    _minusdi_tl,
    _minusdi_pt,
    _minusdi_ta,
    _minusdi_tu,
    _minusdi_fi,
)
_reg("BBANDS", _bb_ft, _bb_tl, _bb_pt, _bb_ta, _bb_tu, _bb_fi)
_reg("ATR", _atr_ft, _atr_tl, _atr_pt, _atr_ta, _atr_tu, _atr_fi)
_reg("NATR", _natr_ft, _natr_tl, _natr_pt, _natr_ta, _natr_tu, _natr_fi)
_reg("TRANGE", _trange_ft, _trange_tl, _trange_pt, _trange_ta, _trange_tu, _trange_fi)
_reg("STDDEV", _stddev_ft, _stddev_tl, _stddev_pt, _stddev_ta, _stddev_tu, _stddev_fi)
_reg("VAR", _var_ft, _var_tl, _var_pt, _var_ta, _var_tu, _var_fi)
_reg("SAR", _sar_ft, _sar_tl, _sar_pt, _sar_ta, _sar_tu, _sar_fi)
_reg("KELTNER_CHANNELS", _kc_ft, _kc_tl, _kc_pt, _kc_ta, _kc_tu, _kc_fi)
_reg(
    "DONCHIAN",
    _donchian_ft,
    _donchian_tl,
    _donchian_pt,
    _donchian_ta,
    _donchian_tu,
    _donchian_fi,
)
_reg(
    "SUPERTREND",
    _supertrend_ft,
    _supertrend_tl,
    _supertrend_pt,
    _supertrend_ta,
    _supertrend_tu,
    _supertrend_fi,
)
_reg("CHOPPINESS_INDEX", _chop_ft, _chop_tl, _chop_pt, _chop_ta, _chop_tu, _chop_fi)
_reg("OBV", _obv_ft, _obv_tl, _obv_pt, _obv_ta, _obv_tu, _obv_fi)
_reg("AD", _ad_ft, _ad_tl, _ad_pt, _ad_ta, _ad_tu, _ad_fi)
_reg("ADOSC", _adosc_ft, _adosc_tl, _adosc_pt, _adosc_ta, _adosc_tu, _adosc_fi)
_reg("MFI", _mfi_ft, _mfi_tl, _mfi_pt, _mfi_ta, _mfi_tu, _mfi_fi)
_reg("VWAP", _vwap_ft, _vwap_tl, _vwap_pt, _vwap_ta, _vwap_tu, _vwap_fi)
_reg(
    "AVGPRICE",
    _avgprice_ft,
    _avgprice_tl,
    _avgprice_pt,
    _avgprice_ta,
    _avgprice_tu,
    _avgprice_fi,
)
_reg(
    "MEDPRICE",
    _medprice_ft,
    _medprice_tl,
    _medprice_pt,
    _medprice_ta,
    _medprice_tu,
    _medprice_fi,
)
_reg(
    "TYPPRICE",
    _typprice_ft,
    _typprice_tl,
    _typprice_pt,
    _typprice_ta,
    _typprice_tu,
    _typprice_fi,
)
_reg(
    "WCLPRICE",
    _wclprice_ft,
    _wclprice_tl,
    _wclprice_pt,
    _wclprice_ta,
    _wclprice_tu,
    _wclprice_fi,
)
_reg("SQRT", _sqrt_ft, _sqrt_tl, _sqrt_pt, _sqrt_ta, _sqrt_tu, _sqrt_fi)
_reg("LOG10", _log10_ft, _log10_tl, _log10_pt, _log10_ta, _log10_tu, _log10_fi)
_reg("ADD", _add_ft, _add_tl, _add_pt, _add_ta, _add_tu, _add_fi)
_reg(
    "LINEARREG",
    _linearreg_ft,
    _linearreg_tl,
    _linearreg_pt,
    _linearreg_ta,
    _linearreg_tu,
    _linearreg_fi,
)
_reg(
    "LINEARREG_SLOPE",
    _linreg_slope_ft,
    _linreg_slope_tl,
    _linreg_slope_pt,
    _linreg_slope_ta,
    _linreg_slope_tu,
    _linreg_slope_fi,
)
_reg("CORREL", _correl_ft, _correl_tl, _correl_pt, _correl_ta, _correl_tu, _correl_fi)
_reg("BETA", _beta_ft, _beta_tl, _beta_pt, _beta_ta, _beta_tu, _beta_fi)
_reg(
    "HT_DCPERIOD",
    _ht_dcperiod_ft,
    _ht_dcperiod_tl,
    _ht_dcperiod_pt,
    _ht_dcperiod_ta,
    _ht_dcperiod_tu,
    _ht_dcperiod_fi,
)
_reg(
    "HT_TRENDMODE",
    _ht_trendmode_ft,
    _ht_trendmode_tl,
    _ht_trendmode_pt,
    _ht_trendmode_ta,
    _ht_trendmode_tu,
    _ht_trendmode_fi,
)
_reg(
    "CDLENGULFING",
    _cdlengulfing_ft,
    _cdlengulfing_tl,
    _cdlengulfing_pt,
    _cdlengulfing_ta,
    _cdlengulfing_tu,
    _cdlengulfing_fi,
)
_reg(
    "CDLDOJI",
    _cdldoji_ft,
    _cdldoji_tl,
    _cdldoji_pt,
    _cdldoji_ta,
    _cdldoji_tu,
    _cdldoji_fi,
)
_reg(
    "CDLHAMMER",
    _cdlhammer_ft,
    _cdlhammer_tl,
    _cdlhammer_pt,
    _cdlhammer_ta,
    _cdlhammer_tu,
    _cdlhammer_fi,
)

# ============================================================
# METADATA
# ============================================================
INDICATOR_DEFAULTS: dict[str, dict] = {
    "SMA": {"timeperiod": 20},
    "EMA": {"timeperiod": 20},
    "WMA": {"timeperiod": 14},
    "DEMA": {"timeperiod": 20},
    "TEMA": {"timeperiod": 20},
    "T3": {"timeperiod": 5},
    "TRIMA": {"timeperiod": 20},
    "KAMA": {"timeperiod": 10},
    "HULL_MA": {"timeperiod": 16},
    "VWMA": {"timeperiod": 20},
    "MIDPOINT": {"timeperiod": 14},
    "MIDPRICE": {"timeperiod": 14},
    "RSI": {"timeperiod": 14},
    "MACD": {"fastperiod": 12, "slowperiod": 26, "signalperiod": 9},
    "STOCH": {"fastk_period": 14, "slowk_period": 3, "slowd_period": 3},
    "CCI": {"timeperiod": 14},
    "WILLR": {"timeperiod": 14},
    "AROON": {"timeperiod": 14},
    "AROONOSC": {"timeperiod": 14},
    "ADX": {"timeperiod": 14},
    "MOM": {"timeperiod": 10},
    "ROC": {"timeperiod": 10},
    "CMO": {"timeperiod": 14},
    "PPO": {"fastperiod": 12, "slowperiod": 26},
    "TRIX": {"timeperiod": 18},
    "TSF": {"timeperiod": 14},
    "ULTOSC": {"timeperiod1": 7, "timeperiod2": 14, "timeperiod3": 28},
    "BOP": {},
    "PLUS_DI": {"timeperiod": 14},
    "MINUS_DI": {"timeperiod": 14},
    "BBANDS": {"timeperiod": 20, "nbdevup": 2.0, "nbdevdn": 2.0},
    "ATR": {"timeperiod": 14},
    "NATR": {"timeperiod": 14},
    "TRANGE": {},
    "STDDEV": {"timeperiod": 20},
    "VAR": {"timeperiod": 20},
    "SAR": {"acceleration": 0.02, "maximum": 0.2},
    "KELTNER_CHANNELS": {"timeperiod": 20},
    "DONCHIAN": {"timeperiod": 20},
    "SUPERTREND": {"timeperiod": 7},
    "CHOPPINESS_INDEX": {"timeperiod": 14},
    "OBV": {},
    "AD": {},
    "ADOSC": {"fastperiod": 3, "slowperiod": 10},
    "MFI": {"timeperiod": 14},
    "VWAP": {},
    "AVGPRICE": {},
    "MEDPRICE": {},
    "TYPPRICE": {},
    "WCLPRICE": {},
    "SQRT": {},
    "LOG10": {},
    "ADD": {},
    "LINEARREG": {"timeperiod": 14},
    "LINEARREG_SLOPE": {"timeperiod": 14},
    "CORREL": {"timeperiod": 30},
    "BETA": {"timeperiod": 5},
    "HT_DCPERIOD": {},
    "HT_TRENDMODE": {},
    "CDLENGULFING": {},
    "CDLDOJI": {},
    "CDLHAMMER": {},
}

INDICATOR_NAMES = list(INDICATOR_DEFAULTS.keys())
LIBRARY_NAMES = ["ferro_ta", "talib", "pandas_ta", "ta", "tulipy", "finta"]

INDICATOR_CATEGORIES: dict[str, list[str]] = {
    "Overlap": [
        "SMA",
        "EMA",
        "WMA",
        "DEMA",
        "TEMA",
        "T3",
        "TRIMA",
        "KAMA",
        "HULL_MA",
        "VWMA",
        "MIDPOINT",
        "MIDPRICE",
    ],
    "Momentum": [
        "RSI",
        "MACD",
        "STOCH",
        "CCI",
        "WILLR",
        "AROON",
        "AROONOSC",
        "ADX",
        "MOM",
        "ROC",
        "CMO",
        "PPO",
        "TRIX",
        "TSF",
        "ULTOSC",
        "BOP",
        "PLUS_DI",
        "MINUS_DI",
    ],
    "Volatility": [
        "BBANDS",
        "ATR",
        "NATR",
        "TRANGE",
        "STDDEV",
        "VAR",
        "SAR",
        "KELTNER_CHANNELS",
        "DONCHIAN",
        "SUPERTREND",
        "CHOPPINESS_INDEX",
    ],
    "Volume": ["OBV", "AD", "ADOSC", "MFI", "VWAP"],
    "Price Transform": ["AVGPRICE", "MEDPRICE", "TYPPRICE", "WCLPRICE"],
    "Math": ["SQRT", "LOG10", "ADD"],
    "Statistics": ["LINEARREG", "LINEARREG_SLOPE", "CORREL", "BETA"],
    "Cycle": ["HT_DCPERIOD", "HT_TRENDMODE"],
    "Pattern": ["CDLENGULFING", "CDLDOJI", "CDLHAMMER"],
}

# Cumulative: compare first-differences not absolute values
CUMULATIVE_INDICATORS = {"OBV", "AD", "ADOSC"}
# Binary output: use agreement rate not allclose
BINARY_INDICATORS = {"CDLENGULFING", "CDLDOJI", "CDLHAMMER", "HT_TRENDMODE"}


def execute_indicator(library, indicator, data, df=None, **kwargs):
    """Run indicator from library on data dict, return 1-D float64 array."""
    if library not in available_libraries():
        raise KeyError(f"Library not available in this environment: {library!r}")

    key = (library, indicator)
    if key not in REGISTRY:
        raise KeyError(f"No wrapper for {key!r}")
    if df is None:
        from benchmarks.data_generator import get_pandas_ohlcv

        df = get_pandas_ohlcv(data)
    params = {**INDICATOR_DEFAULTS.get(indicator, {}), **kwargs}
    return REGISTRY[key](data, df, **params)
