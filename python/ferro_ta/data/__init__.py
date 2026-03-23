"""
ferro_ta.data — Data ingestion, streaming, batch, and resampling utilities.

Sub-modules
-----------
* :mod:`ferro_ta.data.streaming`   — Streaming / incremental indicator state machines
* :mod:`ferro_ta.data.batch`       — Batch execution across multiple series (2-D arrays)
* :mod:`ferro_ta.data.chunked`     — Chunked / windowed processing for large datasets
* :mod:`ferro_ta.data.resampling`  — OHLCV resampling and multi-timeframe support
* :mod:`ferro_ta.data.aggregation` — Tick / trade aggregation pipelines
* :mod:`ferro_ta.data.adapters`    — DataFrame adapters (pandas, polars, numpy)

Example usage::

    from ferro_ta.data.streaming import StreamingSMA
    from ferro_ta.data.batch import batch_sma
"""
