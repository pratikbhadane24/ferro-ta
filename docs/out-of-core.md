# Out-of-Core and Distributed Execution

ferro-ta is designed to work efficiently on large datasets that do not fit
in memory by supporting **chunked execution** with warm-up overlap.  This
document explains the problem, the recommended approach, and current
limitations.

---

## Problem statement

Technical analysis indicators are typically stateful: they require a
look-back window of historical bars to produce a valid value.  When a price
dataset is larger than available memory (e.g. tick data, multiple years of
1-second bars), or when it needs to be processed in a distributed cluster
(Spark, Dask), the data must be split into chunks.

The challenges are:

1. **Warm-up / border effects** — the first `period - 1` bars of each chunk
   will produce NaN because the indicator has not yet accumulated enough
   history.
2. **Partition stitching** — after computing an indicator on each partition
   independently, the partial results must be assembled into a single
   coherent output.
3. **Indicators that need full history** — some indicators (e.g. Hilbert
   Transform cycle indicators) cannot be decomposed into partitions; they
   require the full series.

---

## Chunk boundaries and warm-up overlap

The `ferro_ta.chunked` module provides Rust-backed helpers for chunk-based
execution:

```python
from ferro_ta.chunked import make_chunk_ranges, trim_overlap, stitch_chunks, chunk_apply
from ferro_ta import SMA

import numpy as np

data = np.random.rand(1_000_000)   # large price series
period = 20
overlap = period - 1               # warm-up bars needed

ranges = make_chunk_ranges(len(data), chunk_size=50_000, overlap=overlap)
chunks_out = []
for start, end in ranges:
    chunk = data[start:end]
    out = SMA(chunk, timeperiod=period)
    chunks_out.append(out)

result = stitch_chunks(chunks_out, overlap=overlap)
```

### Key concepts

| Concept | Description |
|---------|-------------|
| `chunk_size` | Number of bars per chunk (excluding overlap). |
| `overlap` | Warm-up bars prepended to each chunk from the previous chunk. |
| `trim_overlap` | Strips the warm-up prefix from a chunk result. |
| `stitch_chunks` | Concatenates trimmed chunk results into the final output. |
| `chunk_apply` | Convenience wrapper: runs a callable on each chunk and stitches. |

---

## Options for distributed / out-of-core execution

### Option A: Chunked pandas with overlap (single-machine, recommended)

Use `chunk_apply` or `make_chunk_ranges` + manual loop.  Suitable for
datasets up to ~10 GB that fit on a single machine with streaming reads.

```python
from ferro_ta.chunked import chunk_apply
from ferro_ta import EMA

result = chunk_apply(data, EMA, chunk_size=100_000, overlap=50, timeperiod=50)
```

### Option B: Dask `map_partitions` (distributed)

Dask can partition a large array and apply a function to each partition.
To handle warm-up correctly, use overlapping partitions via
`dask.array.overlap.overlap`:

```python
import dask.array as da
from dask.array.overlap import overlap as da_overlap
from ferro_ta import SMA

x = da.from_array(price_array, chunks=100_000)
depth = 20 - 1  # warm-up depth

x_ov = da_overlap(x, depth={0: depth}, boundary={0: "none"})
result = x_ov.map_blocks(lambda blk: SMA(blk, timeperiod=20))
# trim overlap from each block
result_trimmed = da.map_blocks(
    lambda blk: blk[depth:],
    result,
    dtype=float,
)
```

### Option C: Apache Spark (brief)

Spark does not natively support overlapping windows for time-series
indicators.  You would need to:

1. Repartition data by time range with explicit padding.
2. Apply the indicator via a Pandas UDF.
3. Filter out warm-up rows in a post-processing step.

This approach is feasible but complex.  For most use-cases, Dask
(Option B) is simpler.

---

## Recommended path

| Scale | Recommendation |
|-------|---------------|
| Single machine, fits in RAM | Use ferro_ta directly on the full array. |
| Single machine, does not fit in RAM | `chunk_apply` with overlap (Option A). |
| Multi-machine cluster | Dask `map_partitions` with `dask.array.overlap` (Option B). |

---

## Which indicators are safe for partition-wise execution

Indicators that depend only on a fixed-length window are **safe** for
chunked/partition-wise execution (with correct overlap):

- All overlap studies: SMA, EMA, WMA, DEMA, TEMA, BBANDS, etc.
- Momentum: RSI, MACD, STOCH, ADX, CCI, WILLR, etc.
- Volatility: ATR, NATR.
- Most volume indicators: OBV, AD (cumulative; use `stitch_chunks` carefully).

Indicators that are **not safe** for partition-wise execution without
special handling:

- Hilbert Transform cycle indicators (`HT_*`) — require full history.
- Adaptive indicators with unbounded look-back (e.g. KAMA with long
  adaptation period).
- Streaming state-machine indicators when state must be preserved across
  chunks (use `ferro_ta.streaming` classes instead).

---

## Limitations

- **Volume-weighted indicators** (e.g. VWAP, OBV) accumulate across all
  bars; resetting at chunk boundaries changes their semantics.  Use
  `streaming.StreamingVWAP` for bar-by-bar accumulation instead.
- **SAR and MAMA** have path-dependent state; chunk results will differ
  from full-series results unless the prior state is passed across chunks.
- Current `chunk_apply` does not propagate indicator state across chunks;
  all indicators restart at each chunk boundary (modulo the overlap
  warm-up).

---

## See also

- `ferro_ta.chunked` — API reference for chunk helpers.
- `ferro_ta.streaming` — Stateful streaming classes for live bar-by-bar use.
- Dask documentation: <https://docs.dask.org/en/stable/>
