Batch Execution API
===================

The batch API lets you run indicators on multiple price series in a single
call. This reduces Python overhead compared to calling the 1-D function in a
loop and naturally maps to multi-asset / multi-symbol workflows.

All batch functions accept a 2-D array of shape ``(n_samples, n_series)`` and
return a 2-D array of the same shape. Passing a 1-D array falls back to the
single-series behaviour.

Usage
-----

.. code-block:: python

    import numpy as np
    from ferro_ta.batch import batch_sma, batch_ema, batch_rsi, batch_apply

    # 100 bars, 5 symbols
    close = np.random.rand(100, 5) + 50.0

    sma = batch_sma(close, timeperiod=14)   # shape (100, 5)
    ema = batch_ema(close, timeperiod=14)   # shape (100, 5)
    rsi = batch_rsi(close, timeperiod=14)   # shape (100, 5)

    # Apply any indicator using batch_apply
    from ferro_ta import MACD
    # MACD returns a tuple so we wrap it
    def macd_line(c, **kw):
        return MACD(c, **kw)[0]

    macd = batch_apply(close, macd_line)    # shape (100, 5)

API Reference
-------------

.. automodule:: ferro_ta.batch
   :no-index:
   :members:
   :undoc-members:
   :show-inheritance:
