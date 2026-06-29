# Dynamic Time Warping

ferro-ta ships three DTW entry points. Pick the one that matches your
workload — the distance-only path is measurably faster than the one that
reconstructs the warping path, and `BATCH_DTW` parallelises over rows.

## Quick reference

| Function | Returns | When to use |
|---|---|---|
| `DTW_DISTANCE(a, b, window=None)` | `float` | You only need the distance. Fastest. |
| `DTW(a, b, window=None)` | `(float, ndarray[N, 2])` | You need the alignment path for plotting or downstream analysis. |
| `BATCH_DTW(matrix, reference, window=None)` | `ndarray[N]` | You have N candidate series and one reference; uses rayon. |

## Distance convention

ferro-ta's DTW uses squared-Euclidean local cost accumulated along the
optimal path, with a single `sqrt()` applied at the end. This matches
`dtaidistance.dtw.distance()` to within floating-point tolerance (parity
tests assert numerical agreement, not bitwise identity). Example:

```python
>>> import ferro_ta as fta
>>> fta.DTW_DISTANCE([0.0, 1.0, 2.0], [1.0, 2.0, 3.0])
1.4142135623730951  # == sqrt(2), same as dtaidistance
```

If you are migrating from a library that uses absolute-difference local
cost without the final sqrt (e.g. `fastdtw`'s default), your numbers will
not line up. That is a choice ferro-ta made for parity with the
scientific-Python ecosystem.

## Window constraint (Sakoe-Chiba band)

Passing `window=w` constrains the DP to cells where `|i - j| < w`. This
turns the O(n·m) cost into O(n·w), which is typically a 5–20× speedup for
realistic `w`. A narrower band can only *increase* the distance, so
`window=` is safe to use whenever your series are roughly aligned.

```python
# Unconstrained
fta.DTW_DISTANCE(a, b)

# Constrained: warping may shift up to 5 positions
fta.DTW_DISTANCE(a, b, window=5)
```

## Batch usage

`BATCH_DTW` compares each row of a 2-D matrix against one reference
series, in parallel:

```python
import numpy as np
import ferro_ta as fta

reference = np.random.random(500)
candidates = np.random.random((1000, 500))

distances = fta.BATCH_DTW(candidates, reference, window=20)
nearest = int(np.argmin(distances))
```

Parallelism is via rayon; no thread-pool configuration is needed on the
Python side. For the sequence lengths ferro-ta targets (thousands of
bars, hundreds to low thousands of candidates), batch-parallel classic
DTW beats FastDTW-style approximations.

## Edge cases

- **Empty input:** raises `FerroTAInputError`.
- **NaN in input:** propagates to the output (matches IEEE 754). Call
  `ferro_ta.core.exceptions.check_finite()` first if you want to fail
  loudly instead.
- **Different-length series:** fully supported. The path array length
  is bounded by `max(n, m) <= len(path) <= n + m - 1`.

## See also

- `tests/unit/indicators/test_statistic.py` — parity tests against `dtaidistance`.
