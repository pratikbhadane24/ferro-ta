# ADR 0005: TA-Lib parity is graded, not binary

**Status:** Accepted
**Date:** 2026-04-14

## Context

ferro-ta advertises TA-Lib-compatible APIs for 162 indicators. "Compatible"
isn't a single thing: some ferro-ta functions match TA-Lib to machine
epsilon, others differ by ~1 ULP due to summation order, and a handful
diverge in documented ways (e.g. SAR's seed value).

Users need a clear signal about which category a given indicator falls into
so they can decide whether ferro-ta is a drop-in replacement for their
existing TA-Lib pipeline.

## Decision

Every indicator is classified into one of three compatibility classes,
documented in `TA_LIB_COMPATIBILITY.md`:

| Class | Meaning | Threshold |
|---|---|---|
| **Exact** | Bitwise-equal output for identical inputs. | `abs(diff) == 0.0` |
| **Close** | Equal to TA-Lib within relative tolerance. | `rel_err < 1e-10` |
| **Non-exact** | Documented, intentional divergence. | Covered in the compat doc with reasoning. |

The classification is determined by a dedicated parity-test suite that runs
every indicator against `ta-lib` on a fixed synthetic OHLCV dataset. The
threshold used is explicit in the test code — changing it requires updating
this ADR.

## Why not just "matches TA-Lib"?

- **Floating-point summation order** varies between implementations; Kahan
  summation vs naive can legitimately differ in the last bit without either
  being "wrong".
- **Seed values** (e.g. for EMA/SAR) can be defined differently without
  affecting steady-state results.
- **NaN placement** at the warmup period can differ between implementations
  and is not a correctness issue.

Collapsing these into "matches" would mislead users. Collapsing them into
"doesn't match" would also mislead them.

## Consequences

- Users can grep `TA_LIB_COMPATIBILITY.md` for their indicator and get a
  clear answer before migrating.
- Parity tests act as regression detectors — a previously "Exact" indicator
  that becomes "Close" is a potentially-breaking change.
- The three classes map directly to CI test assertions (`assert_equal`,
  `assert_allclose(rtol=1e-10)`, `# divergence documented, no test`).
