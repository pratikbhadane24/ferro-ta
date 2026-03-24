"""
ferro-ta REST API
============================

A minimal FastAPI service that exposes ferro-ta indicators and backtest
over HTTP so that any client can compute technical analysis via REST.

Endpoints
---------
GET  /health                 — readiness / liveness probe
POST /indicators/sma         — Simple Moving Average
POST /indicators/ema         — Exponential Moving Average
POST /indicators/rsi         — Relative Strength Index
POST /indicators/macd        — MACD (line, signal, histogram)
POST /indicators/bbands      — Bollinger Bands
POST /backtest               — Vectorized backtest

Request / Response format
-------------------------
All indicator endpoints accept JSON:
    {
      "close": [1.0, 2.0, ...],   // required; array of floats
      "timeperiod": 14            // optional parameter
    }

And return:
    {
      "result": [null, null, ..., 14.0, ...]  // null for NaN warm-up
    }

Or for multi-output indicators (MACD, BBANDS):
    {
      "result": {
        "macd": [...],
        "signal": [...],
        "hist": [...]
      }
    }

For the backtest endpoint the request is:
    {
      "close": [1.0, 2.0, ...],
      "strategy": "rsi_30_70",     // or "sma_crossover", "macd_crossover"
      "commission_per_trade": 0.0,
      "slippage_bps": 0.0
    }

And the response is:
    {
      "final_equity": 1.123,
      "n_trades": 7,
      "equity": [1.0, ...]
    }

Running
-------
Development::

    uvicorn api.main:app --reload --port 8000

Production::

    uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

Docker::

    docker build -t ferro-ta-api ./api
    docker run -p 8000:8000 ferro-ta-api

Environment variables
---------------------
MAX_SERIES_LENGTH : int — maximum number of data points per request
  (default 100 000). Requests exceeding this limit return HTTP 413.
"""

from __future__ import annotations

import math
import os
from typing import Any, Dict, List, Optional

import numpy as np

try:
    from fastapi import FastAPI, HTTPException
    from pydantic import BaseModel, Field, field_validator
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "The ferro-ta API requires fastapi and pydantic.\n"
        "Install with: pip install 'ferro_ta[api]'"
    ) from exc

import ferro_ta as ft
from ferro_ta.analysis.backtest import backtest as _backtest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_SERIES_LENGTH = int(os.environ.get("MAX_SERIES_LENGTH", "100000"))

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="ferro-ta API",
    description="REST API for ferro-ta technical analysis indicators and backtesting.",
    version=ft.__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _nan_to_none(arr: np.ndarray) -> List[Optional[float]]:
    """Convert numpy array to list, replacing NaN/Inf with None."""
    return [None if not math.isfinite(v) else float(v) for v in arr]


def _validate_series(close: List[float]) -> np.ndarray:
    if len(close) > MAX_SERIES_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Series length {len(close)} exceeds maximum {MAX_SERIES_LENGTH}.",
        )
    if len(close) < 2:
        raise HTTPException(
            status_code=422,
            detail="Series must contain at least 2 values.",
        )
    return np.asarray(close, dtype=np.float64)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class IndicatorRequest(BaseModel):
    close: List[float] = Field(..., description="Close price series")
    timeperiod: int = Field(default=14, ge=1, description="Look-back period")

    @field_validator("close")
    @classmethod
    def close_must_be_finite(cls, v: List[float]) -> List[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("close series must contain only finite values")
        return v


class MACDRequest(BaseModel):
    close: List[float] = Field(..., description="Close price series")
    fastperiod: int = Field(default=12, ge=1)
    slowperiod: int = Field(default=26, ge=1)
    signalperiod: int = Field(default=9, ge=1)

    @field_validator("close")
    @classmethod
    def close_must_be_finite(cls, v: List[float]) -> List[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("close series must contain only finite values")
        return v


class BBANDSRequest(BaseModel):
    close: List[float] = Field(..., description="Close price series")
    timeperiod: int = Field(default=5, ge=2)
    nbdevup: float = Field(default=2.0, gt=0)
    nbdevdn: float = Field(default=2.0, gt=0)

    @field_validator("close")
    @classmethod
    def close_must_be_finite(cls, v: List[float]) -> List[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("close series must contain only finite values")
        return v


class BacktestRequest(BaseModel):
    close: List[float] = Field(..., description="Close price series")
    strategy: str = Field(default="rsi_30_70")
    commission_per_trade: float = Field(default=0.0, ge=0.0)
    slippage_bps: float = Field(default=0.0, ge=0.0)

    @field_validator("close")
    @classmethod
    def close_must_be_finite(cls, v: List[float]) -> List[float]:
        if not all(math.isfinite(x) for x in v):
            raise ValueError("close series must contain only finite values")
        return v


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/health", summary="Health check")
def health() -> Dict[str, str]:
    """Readiness / liveness probe."""
    return {"status": "ok", "version": app.version}


@app.post("/indicators/sma", summary="Simple Moving Average")
def compute_sma(req: IndicatorRequest) -> Dict[str, Any]:
    """Compute Simple Moving Average (SMA).

    Returns ``result``: list of floats (null for warm-up bars).
    """
    c = _validate_series(req.close)
    out = np.asarray(ft.SMA(c, timeperiod=req.timeperiod), dtype=np.float64)
    return {"result": _nan_to_none(out)}


@app.post("/indicators/ema", summary="Exponential Moving Average")
def compute_ema(req: IndicatorRequest) -> Dict[str, Any]:
    """Compute Exponential Moving Average (EMA)."""
    c = _validate_series(req.close)
    out = np.asarray(ft.EMA(c, timeperiod=req.timeperiod), dtype=np.float64)
    return {"result": _nan_to_none(out)}


@app.post("/indicators/rsi", summary="Relative Strength Index")
def compute_rsi(req: IndicatorRequest) -> Dict[str, Any]:
    """Compute Relative Strength Index (RSI)."""
    c = _validate_series(req.close)
    out = np.asarray(ft.RSI(c, timeperiod=req.timeperiod), dtype=np.float64)
    return {"result": _nan_to_none(out)}


@app.post("/indicators/macd", summary="MACD")
def compute_macd(req: MACDRequest) -> Dict[str, Any]:
    """Compute MACD (line, signal, histogram).

    Returns ``result`` with keys ``macd``, ``signal``, ``hist``.
    """
    c = _validate_series(req.close)
    macd, signal, hist = ft.MACD(
        c,
        fastperiod=req.fastperiod,
        slowperiod=req.slowperiod,
        signalperiod=req.signalperiod,
    )
    return {
        "result": {
            "macd": _nan_to_none(np.asarray(macd, dtype=np.float64)),
            "signal": _nan_to_none(np.asarray(signal, dtype=np.float64)),
            "hist": _nan_to_none(np.asarray(hist, dtype=np.float64)),
        }
    }


@app.post("/indicators/bbands", summary="Bollinger Bands")
def compute_bbands(req: BBANDSRequest) -> Dict[str, Any]:
    """Compute Bollinger Bands (upper, middle, lower).

    Returns ``result`` with keys ``upper``, ``middle``, ``lower``.
    """
    c = _validate_series(req.close)
    upper, middle, lower = ft.BBANDS(
        c,
        timeperiod=req.timeperiod,
        nbdevup=req.nbdevup,
        nbdevdn=req.nbdevdn,
    )
    return {
        "result": {
            "upper": _nan_to_none(np.asarray(upper, dtype=np.float64)),
            "middle": _nan_to_none(np.asarray(middle, dtype=np.float64)),
            "lower": _nan_to_none(np.asarray(lower, dtype=np.float64)),
        }
    }


@app.post("/backtest", summary="Vectorized backtest")
def run_backtest(req: BacktestRequest) -> Dict[str, Any]:
    """Run a vectorized backtest using a named strategy.

    Strategies: ``rsi_30_70``, ``sma_crossover``, ``macd_crossover``.
    Returns ``final_equity``, ``n_trades``, and the full ``equity`` curve.
    """
    c = _validate_series(req.close)
    valid_strategies = {"rsi_30_70", "sma_crossover", "macd_crossover"}
    if req.strategy not in valid_strategies:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown strategy '{req.strategy}'. "
            f"Available: {sorted(valid_strategies)}",
        )
    result = _backtest(
        c,
        strategy=req.strategy,
        commission_per_trade=req.commission_per_trade,
        slippage_bps=req.slippage_bps,
    )
    return {
        "final_equity": float(result.final_equity),
        "n_trades": int(result.n_trades),
        "equity": _nan_to_none(result.equity),
    }
