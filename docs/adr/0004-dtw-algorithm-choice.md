# ADR 0004: Classic DP dynamic time warping, not FastDTW

**Status:** Accepted
**Date:** 2026-04-14
**Related:** PR #9 (`Dtw algo`), commit `fd1bb13`

## Context

Dynamic Time Warping has several well-known variants:

1. **Classic DP** — O(n·m) time and space, returns the exact optimal
   alignment. Used by `dtaidistance` and TA-Lib-style libraries.
2. **FastDTW** (Salvador & Chan, 2007) — O(n) approximation via multi-scale
   refinement. Fast but approximate; reported error grows on high-frequency
   data.
3. **PrunedDTW / UCR Suite** — O(n·m) worst case but with aggressive
   early-abandon pruning for nearest-neighbour search.

## Decision

ferro-ta implements the classic DP formulation with Euclidean (squared
difference, final sqrt) local cost and an optional Sakoe-Chiba band
(`window=` parameter). This matches `dtaidistance.dtw.distance()` byte-for-byte,
which is the reference Python users expect to compare against.

## Rationale

- **Correctness over speed by default.** Quants using DTW in backtesting
  need reproducible, exact distances. An approximation hidden behind the
  default call would break parity tests against `dtaidistance`.
- **Window constraint gives the speedup path.** `window=w` reduces the
  DP band to O(n·w), which is usually enough for the speedups FastDTW
  users are after, without approximation.
- **Parallelism is the real batch speedup.** `BATCH_DTW` parallelises
  over rows with rayon, which for realistic workloads (thousands of
  series against one reference) dominates any per-pair algorithmic
  improvement.

## Trade-offs accepted

- No FastDTW support. Users who need O(n) approximate DTW should stick
  with `fastdtw` or `dtaidistance`'s approximate mode.
- O(n·m) memory for the `DTW()` path variant. Acceptable for the
  sequence lengths ferro-ta targets (thousands of bars, not millions).

## Verification

Parity tests in `tests/unit/indicators/test_statistic.py` compare every
DTW variant against `dtaidistance` on random inputs. Rust-level tests in
`crates/ferro_ta_core/src/statistic.rs` check identity, symmetry,
triangle inequality, NaN propagation, and path monotonicity.
