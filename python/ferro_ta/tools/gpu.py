"""
ferro_ta.gpu — Optional GPU-accelerated indicator backend via PyTorch.

When the caller passes a PyTorch Tensor as input, the GPU path is used and the
result is returned as a PyTorch Tensor. When a NumPy array (or plain Python
sequence) is passed, the standard CPU path is used — there is **no behaviour
change** for existing CPU-only code.

Install the optional GPU extra to enable this feature:

    pip install "ferro-ta[gpu]"

Or install PyTorch manually:

    pip install torch

Usage
-----
>>> import torch
>>> from ferro_ta.tools.gpu import sma, ema, rsi
>>>
>>> close_gpu = torch.tensor([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10], device='cuda') # or 'mps'
>>> result = sma(close_gpu, timeperiod=3)
>>> type(result)   # torch.Tensor
>>> result_cpu = result.cpu().numpy()

See ``docs/gpu-backend.md`` for design notes, limitations, and benchmark data.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np

# ---------------------------------------------------------------------------
# PyTorch detection
# ---------------------------------------------------------------------------

try:
    import torch as _torch

    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False


def _is_torch(arr: object) -> bool:
    """Return True when *arr* is a PyTorch Tensor."""
    return (
        _TORCH_AVAILABLE is True
        and _torch is not None
        and isinstance(arr, _torch.Tensor)
    )


def _to_cpu(arr: object) -> np.ndarray:
    """Convert a PyTorch Tensor to a NumPy array; pass NumPy arrays through."""
    if _is_torch(arr):
        return cast(Any, arr).cpu().numpy()
    return np.asarray(arr, dtype=np.float64)


def _to_gpu(arr: np.ndarray, device: Any = None) -> Any:
    """Move a NumPy array to the GPU (returns torch.Tensor)."""
    assert _torch is not None
    return _torch.tensor(arr, device=device)


# ---------------------------------------------------------------------------
# GPU implementations
# ---------------------------------------------------------------------------


def _sma_gpu(close, timeperiod: int):
    """SMA on a PyTorch Tensor using cumsum-based rolling mean."""
    if _torch is None:
        raise RuntimeError("PyTorch is not installed")
    torch = _torch
    n = close.shape[0]
    result = torch.full((n,), float("nan"), dtype=close.dtype, device=close.device)
    if timeperiod < 1 or n < timeperiod:
        return result
    # cumsum-based O(n) rolling sum
    cs = torch.cumsum(close, dim=0)
    # window sum for index i: cs[i] - cs[i - timeperiod]  (i >= timeperiod-1)
    win = cs[timeperiod - 1 :]
    win = win.clone()
    win[1:] -= cs[: len(win) - 1]
    result[timeperiod - 1 :] = win / timeperiod
    return result


def _ema_gpu(close, timeperiod: int):
    """EMA on a PyTorch Tensor — SMA-seeded, element-wise loop in Python/PyTorch."""
    if _torch is None:
        raise RuntimeError("PyTorch is not installed")
    torch = _torch
    n = close.shape[0]
    result = torch.full((n,), float("nan"), dtype=close.dtype, device=close.device)
    if timeperiod < 1 or n < timeperiod:
        return result
    k = 2.0 / (timeperiod + 1.0)
    # Seed with SMA of first window (already on GPU)
    seed = float(torch.mean(close[:timeperiod]).item())
    result[timeperiod - 1] = seed
    # Recurrence on CPU for numerical correctness then move back
    close_cpu = close.cpu().numpy()
    res_cpu = np.full(n, np.nan)
    res_cpu[timeperiod - 1] = seed
    prev = seed
    for i in range(timeperiod, n):
        val = float(close_cpu[i]) * k + prev * (1.0 - k)
        res_cpu[i] = val
        prev = val
    return torch.tensor(res_cpu, dtype=close.dtype, device=close.device)


def _rsi_gpu(close, timeperiod: int):
    """RSI on a PyTorch Tensor — compute diffs on GPU, finish on CPU."""
    if _torch is None:
        raise RuntimeError("PyTorch is not installed")
    torch = _torch
    n = close.shape[0]
    result = torch.full((n,), float("nan"), dtype=close.dtype, device=close.device)
    if timeperiod < 1 or n <= timeperiod:
        return result
    # Compute price diffs on GPU
    diffs = torch.diff(close).cpu().numpy()  # (n-1,) numpy array
    # CPU recurrence (Wilder smoothing)
    res_cpu = np.full(n, np.nan)
    avg_gain = np.mean(np.maximum(diffs[:timeperiod], 0.0))
    avg_loss = np.mean(np.maximum(-diffs[:timeperiod], 0.0))
    rs = avg_gain / avg_loss if avg_loss != 0.0 else np.inf
    res_cpu[timeperiod] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(timeperiod + 1, n):
        d = diffs[i - 1]
        gain = d if d > 0.0 else 0.0
        loss = -d if d < 0.0 else 0.0
        avg_gain = (avg_gain * (timeperiod - 1) + gain) / timeperiod
        avg_loss = (avg_loss * (timeperiod - 1) + loss) / timeperiod
        rs = avg_gain / avg_loss if avg_loss != 0.0 else np.inf
        res_cpu[i] = 100.0 - 100.0 / (1.0 + rs)
    return torch.tensor(res_cpu, dtype=close.dtype, device=close.device)


# ---------------------------------------------------------------------------
# Public API — PyTorch in → PyTorch out; NumPy in → NumPy out
# ---------------------------------------------------------------------------


def sma(close, timeperiod: int = 30):
    """Simple Moving Average — GPU-accelerated when *close* is a PyTorch Tensor.

    Parameters
    ----------
    close : numpy.ndarray or torch.Tensor
        Close price array.
    timeperiod : int, default 30
        Look-back window.

    Returns
    -------
    numpy.ndarray or torch.Tensor
        Same type as *close*.  First ``timeperiod - 1`` values are NaN.
    """
    if _is_torch(close):
        if not close.is_floating_point():
            close = close.float()
        return _sma_gpu(close, timeperiod)
    # CPU fallback
    from ferro_ta import SMA  # noqa: PLC0415

    return SMA(np.asarray(close, dtype=np.float64), timeperiod=timeperiod)


def ema(close, timeperiod: int = 30):
    """Exponential Moving Average — GPU-accelerated when *close* is a PyTorch Tensor.

    Parameters
    ----------
    close : numpy.ndarray or torch.Tensor
    timeperiod : int, default 30

    Returns
    -------
    numpy.ndarray or torch.Tensor — same type as *close*.
    """
    if _is_torch(close):
        if not close.is_floating_point():
            close = close.float()
        return _ema_gpu(close, timeperiod)
    from ferro_ta import EMA  # noqa: PLC0415

    return EMA(np.asarray(close, dtype=np.float64), timeperiod=timeperiod)


def rsi(close, timeperiod: int = 14):
    """Relative Strength Index — GPU-accelerated when *close* is a PyTorch Tensor.

    Parameters
    ----------
    close : numpy.ndarray or torch.Tensor
    timeperiod : int, default 14

    Returns
    -------
    numpy.ndarray or torch.Tensor — same type as *close*.  Values in [0, 100].
    """
    if _is_torch(close):
        if not close.is_floating_point():
            close = close.float()
        return _rsi_gpu(close, timeperiod)
    from ferro_ta import RSI  # noqa: PLC0415

    return RSI(np.asarray(close, dtype=np.float64), timeperiod=timeperiod)


__all__ = [
    "sma",
    "ema",
    "rsi",
]
