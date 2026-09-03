# TA-Lib Compatibility

`ferro-ta` implements **156 of TA-Lib 0.6.4's 161 functions**, plus 10 extended
indicators and 9 streaming classes that TA-Lib does not provide. This file keeps
the full GitHub-facing parity matrix in one place so the root `README.md` can
stay product-focused.

The five not yet implemented are `ACCBANDS`, `IMI`, `AVGDEV`, `MINMAX` and
`MINMAXINDEX`.

See also:

- [docs/migration_talib.rst](docs/migration_talib.rst)
- [docs/compatibility/talib.md](docs/compatibility/talib.md)
- [docs/support_matrix.rst](docs/support_matrix.rst)

## Legend


| Symbol      | Meaning                                                                                                                                                                      |
| ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ✅ Exact     | A test in `tests/integration/test_vs_talib.py` asserts value equality against TA-Lib over the whole valid region (`atol` between `1e-6` and `1e-10`)                          |
| ✅ Close     | A test asserts value agreement, but only after a convergence window, or only within a documented tolerance rather than bit-for-bit. The note names the tolerance and the test |
| ⚠️ Corr     | A test asserts only a **correlation** threshold against TA-Lib, not values. The note gives the asserted threshold                                                             |
| ⚠️ Shape    | A test asserts only output length and/or NaN structure. **Values are never compared for equality.** Where a weaker gate exists (a correlation or sign-agreement threshold below ⚠️ Corr's 0.95) the note names it |
| ❓ Untested  | Implemented, but `tests/integration/test_vs_talib.py` contains no TA-Lib comparison for it at all. The note says what *is* known                                              |
| ❌           | Not yet implemented                                                                                                                                                          |
| † (dagger)  | Takes a `matype` / `*_matype` argument. TA-Lib enum parity holds for `0`–`6` and `8` but **not** `7` — see [The `matype` enum](#the-matype-enum)                              |

Function counts below are measured against TA-Lib **0.6.4**; the `matype` value
gates cited in the notes were run against a real TA-Lib **0.7.1** install.

Every rating in the tables below was re-derived from `tests/integration/test_vs_talib.py`
in the session that wrote this file; a rating is never carried forward on the
strength of the wording that preceded it. Where a rating names a tolerance or a
threshold, that number is the one the test actually asserts.

> A rating describes agreement on finite input, with **arguments matched on
> both sides**. Where ferro-ta's default arguments differ from TA-Lib's —
> `APO`, `PPO` and `MACDEXT` all default to `matype = 1` (EMA) where TA-Lib
> defaults to `0` (SMA) — the note says so, because the default is what an
> unqualified call gets. For those three the distinction is the whole rating:
> the **values are exact at a matched `matype`**, and an **unqualified call
> still does not match TA-Lib** because the two libraries pick different MAs.
> Both halves are asserted — `TestAPOMatype::test_values_match` /
> `TestPPOMatype::test_ppo_line_values_match` for the first, and each class's
> `test_default_matype_is_ema_not_talib_sma` (which asserts agreement with
> `talib.…(matype=1)` *and* disagreement with `matype=0`) for the second.

## The `matype` enum

`MA`, `MACDEXT`, `MAVP`, `APO`, `PPO`, `STOCH`, `STOCHF` and `STOCHRSI` accept a
`matype` (or `slowk_matype` / `slowd_matype` / `fastd_matype`) selecting the
moving average applied inside the kernel. Valid range is `0..=8`; the
authoritative table lives in the dispatcher's module docs
(`crates/ferro_ta_core/src/overlap/dispatch.rs`).

**The enum is not TA-Lib-compatible at `7`.**

| `matype` | ferro-ta        | TA-Lib `TA_MAType` | Compatible? |
| -------- | --------------- | ------------------ | ----------- |
| 0        | SMA             | SMA                | yes         |
| 1        | EMA             | EMA                | yes         |
| 2        | WMA             | WMA                | yes         |
| 3        | DEMA            | DEMA               | yes         |
| 4        | TEMA            | TEMA               | yes         |
| 5        | TRIMA           | TRIMA              | yes         |
| 6        | KAMA            | KAMA               | yes         |
| 7        | **T3**          | **MAMA**           | **NO**      |
| 8        | T3 (alias of 7) | T3                 | yes         |
| ≥ 9      | out of range    | out of range       | yes         |

- Code ported from TA-Lib that passes `7` expecting MAMA silently gets T3 — no
  error, no warning, a different indicator. **Pass `8` for T3**: it means T3 in
  both libraries.
- This is measured, not inferred. `TestMatypeSevenDivergence` asserts
  `talib.MA(x, 10, matype=7)` equals `talib.MAMA(x, 0.5, 0.05)[0]` to
  `atol=1e-12`, and `ferro_ta.MA(x, 10, matype=7)` equals both `matype=8` and
  `T3(vfactor=0.7)` to the same tolerance (`7` and `8` are asserted
  bit-identical). It then asserts **non**-agreement with TA-Lib at `7` for
  `MA`, `APO`, `PPO`, `STOCHF` %D and `STOCHRSI` %D, so the divergence cannot
  quietly become a passing test.
- **MAMA has no `matype` value at all.** TA-Lib's `TA_MA` special-cases `7` into
  `TA_MAMA(…, 0.5, 0.05, …)`, ignoring the period and discarding the FAMA
  output; ferro-ta's dispatcher has no such arm. Call `MAMA` directly — it takes
  both limits explicitly and returns both MAMA and FAMA.
- `9` and above are out of range and are **never silently treated as SMA**. The
  Python wrappers raise `FerroTAValueError` (a `ValueError`) — asserted for
  `MA`, `APO`, `PPO`, `STOCHF`, `STOCHRSI`, `MAVP` and all three `MACDEXT`
  matypes by `TestMatypeOutOfRange` and
  `TestMACDEXTMatype::test_matype_9_is_rejected`, with a cross-check that
  TA-Lib rejects `9` too. The Rust core, which has no error type, reports the
  same condition as an **all-`NaN` output**
  (`crates/ferro_ta_core/src/momentum/price_osc.rs`,
  `crates/ferro_ta_core/src/overlap/dispatch.rs`).
- `MACDEXT`'s wrappers briefly capped `matype` at `7` while the rest of the
  crate had moved to `8`. That is fixed: all three of `fastmatype`,
  `slowmatype` and `signalmatype` accept `0`–`8` like their siblings, guarded
  by `TestMACDEXTMatype::test_matype_8_is_accepted`.

Because of the `7` divergence, no `matype`-taking function is marked ✅ Exact
without the † marker, whatever its rating at the default `matype`.

### Where the MA is seeded: whole-series vs. converged-tail agreement

For `MAVP` and `MACDEXT`, agreement with TA-Lib splits cleanly by MA family,
and the reason is a real semantic difference rather than floating-point drift:

**TA-Lib computes each leg's MA over a sub-range that begins at the output
start index; ferro-ta seeds from bar 0.**

- **Window MAs — `matype` `0` (SMA), `2` (WMA), `5` (TRIMA):** the value at bar
  *i* depends only on the last *p* bars, so where the MA started is irrelevant
  and the two libraries agree over the **whole valid region**
  (`TestMAVPMatype::test_window_matypes_match_exactly`,
  `TestMACDEXTMatype::test_window_matypes_match_exactly`).
- **Recursive MAs — `matype` `1` (EMA), `3` (DEMA), `4` (TEMA), `6` (KAMA), `8`
  (T3):** the seed differs, so **early bars differ** and the gap decays. Only
  the converged tail is asserted — the last 30% of bars, ≥ 120 compared bars
  (`test_recursive_matypes_match_on_converged_tail` in both classes).
- Practically: expect divergence in the warmup-adjacent bars of a recursive
  `MAVP` / `MACDEXT` and agreement thereafter. The mechanism was confirmed by
  showing `talib.MACDEXT(matype=1)` is bit-identical to `talib.MACD`, which
  itself differs from a full-series `talib.EMA` by the same offset.

### `KAMA` on zero-volatility input — fixed

`ferro_ta.KAMA` and `talib.KAMA` now agree on input containing zero-volatility
windows — long runs of an identical value, where the efficiency-ratio
denominator empties across the whole window. This was a real ferro-ta defect
and it is repaired; the section is kept because the *reason* is not intuitive
and because a TA-Lib upgrade will disturb it.

**What was wrong.** ferro-ta tested the denominator with `volatility > 0.0`.
TA-Lib's `ta_KAMA.c` (v0.6.4) tests

```c
if( (sumROC1 <= periodROC) || TA_IS_ZERO(sumROC1) )   /* TA_EPSILON = 1e-14 */
   tempReal = 1.0;
```

Two defects, each independently sufficient:

- `TA_IS_ZERO` is a **±`1e-14` band**, not a strict comparison. Both libraries
  carry `volatility` as a subtract-then-add rolling sum, so a window that has
  gone flat holds ~`1e-14` of rounding residue rather than an exact zero.
  Testing `> 0.0` read that residue as real volatility and produced
  `ER = 0 / residue = 0` — the *slowest* smoothing constant, the exact opposite
  of what the kernel's own comment claimed. Because the ratio was wrong on
  every bar of a plateau the error held instead of decaying.
- The signed `sumROC1 <= periodROC` clamp was **missing entirely**
  (`periodROC` is `close[i] - close[i-p]`, not its magnitude).

**Evidence.** Validated against the installed `libta-lib` over 1400
series/period combinations: with both halves present the worst relative
deviation is `3e-16`; dropping either one pushes it to ~`0.28`. Plateau input
went from ~16 absolute error to `2.8e-14`, and ordinary random-walk input is
unchanged. Asserted by `test_kama_matches_talib_on_zero_volatility_input`
(`atol=1e-9`, > 400 compared bars), `test_kama_matches_talib_on_flat_series`
(`atol=1e-12`, and the output pinned to the flat price) and
`test_kama_matches_talib_on_plateau_heavy_input[2,3,5,14,30]` (`atol=1e-9`,
> 500 compared bars per period).

**One caveat worth recording rather than burying.** TA-Lib's own flat-window
behaviour is **residue-dependent**, not idealized: negative residue, or residue
inside the `±1e-14` band, takes `ER = 1` (snap to price); positive residue
above the band takes `ER = 0` (crawl). ferro-ta reproduces that faithfully —
including TA-Lib's exact `volatility` update order, so the residue matches bit
for bit. That fidelity is what ✅ Exact means for this row; it is not a claim
that either library is well behaved here.

**On a TA-Lib upgrade, revisit this.** TA-Lib `main` has replaced the
`TA_IS_ZERO` band with a counter of consecutive exactly-zero 1-day changes,
which zeroes `sumROC1` outright once a whole window is flat (upstream issue
#253; the fixed band was scale-inconsistent — it declared every window flat for
an instrument quoted below `1e-14`). That is a *behaviour* change on
residue-carrying plateaus, so `crates/ferro_ta_core/src/overlap/kama.rs` and
this section both need re-deriving when the pinned TA-Lib moves past 0.6.4.

### `STOCHRSI` %D at `matype = 6` — upstream conditioning, not a ferro-ta defect

`STOCHRSI`'s %D *value* gate excludes `fastd_matype = 6` (KAMA) —
`STOCHRSI_D_MATYPES` in `tests/integration/test_vs_talib.py`. **The exclusion
is not a consequence of the `KAMA` bug above**: it survives the fix, at the same
magnitude of 19.2. Any note attributing it to that bug is wrong.

The disproof, all four numbers measured on the same fixture:

| Comparison | Max abs diff |
| ---------- | ------------ |
| `talib.KAMA(ferro's %K)` vs `talib.KAMA(talib's %K)` — **TA-Lib against itself** | **19.21** |
| `ferro.KAMA` vs `talib.KAMA`, both on ferro's %K | `2.1e-14` |
| `ferro.KAMA` vs `talib.KAMA`, both on talib's %K | `2.8e-14` |
| `ferro %K` vs `talib %K` (the *input* difference) | `5.8e-13` |

`TA_KAMA` is catastrophically ill-conditioned on plateau input: the *sign* of a
`1e-14` rolling-sum residue selects between `ER = 1` and `ER = 0`, and KAMA is
recursive, so ferro-ta's `5.8e-13` rounding difference in %K amplifies to 19 in
**TA-Lib's own output**. StochRSI %K is pinned at exactly `0` or `100` for long
stretches (136 and 116 bars in the fixture), which is what makes it the
pathological case; raw-price %K has no such plateaus, so `STOCH` and `STOCHF`
are unaffected.

The `matype = 6` exclusion therefore stays, as a **documented upstream
divergence** rather than a bug. It is pinned by
`test_talib_kama_is_ill_conditioned_on_stochrsi_fastk`, which asserts the
conditioning itself (TA-Lib against itself > 1.0, and the two %K inputs both
within `1e-9` and *not* bit-identical) and that ferro-ta tracks TA-Lib to
`atol=1e-9` on **each input individually**.

### `MACDEXT` warm-up — fixed

`macdext` computed its shared start index as `slowperiod - 1`, which is the
lookback of the SMA family only. With a `matype` argument each leg warms up at
`ma_lookback(period, matype)` — `2(p-1)` for DEMA, `3(p-1)` for TEMA, `6(p-1)`
for T3, `p` for KAMA — and the *fast* leg's lookback can exceed the slow leg's
(T3 at `fastperiod = 12` needs 66 bars where SMA at `slowperiod = 26` needs
25), so neither leg alone bounds it. The correct expression, and now the code's:

```text
macd_start = max(ma_lookback(fastperiod, fastmatype),
                 ma_lookback(slowperiod, slowmatype))
first      = macd_start + ma_lookback(signalperiod, signalmatype)
```

Starting too early was not merely a cosmetic index error: the signal leg is an
MA of `macd_line[macd_start..]`, so an early `macd_start` handed that MA a slice
with a leading `NaN`. `MACDEXT(fastmatype=6, slowmatype=6, signalmatype=6)`
returned **all `NaN`** as a result — `kama` propagates a `NaN` for the whole
series. That failure had been masked by a separate `kama` defect (the
zero-volatility branch above) which happened to swallow the `NaN`; fixing KAMA
exposed it. `first` is exactly `TA_MACDEXT_Lookback`, so NaN counts now agree
with TA-Lib *exactly* (measured: 33/33/33/66/99/33/35/198 at `matype`
`0`/`1`/`2`/`3`/`4`/`5`/`6`/`8` for 12/26/9 — identical on both sides), where
before the fix the suite could only assert ±1. Pinned by
`macdext_warmup_is_max_leg_lookback_plus_signal_lookback` in
`crates/ferro_ta_core/src/overlap/dispatch.rs`, which walks all 729
`(fastmatype, slowmatype, signalmatype)` combinations and asserts the `NaN`
prefix length, a finite value at `first`, no interior `NaN`, and the
`hist = macd - signal` identity; plus
`macdext_kama_on_all_three_legs_is_not_all_nan` for the original symptom and
`macdext_warmup_is_not_slowperiod_minus_one_for_long_lookback_legs` against a
regression to the old expression.

The same enum reaches the extended (non-TA-Lib) indicators `MA_ENVELOPES`,
`OBV_SMOOTHED` and `PVI_WITH_SIGNAL`.


## Overlap Studies


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                 |
| --------------- | -------- | -------- | ----------------------------------------------------- |
| `BBANDS`        | ✅        | ✅ Exact  | Bollinger Bands. All three bands verified (`TestBBANDS`, `TestParitySuite::test_bbands_values_match_talib`). ferro-ta takes **no** `matype`: it computes only TA-Lib's default `0` (SMA) middle band |
| `DEMA`          | ✅        | ✅ Close  | SMA-seeded composed EMA (`2*EMA − EMA(EMA)`); mid-series convergence asserted at `atol=1e-2` |
| `EMA`           | ✅        | ✅ Close  | SMA-seeded (`k = 2/(n+1)`). `TestEMA` asserts NaN-count parity (warmup identical to TA-Lib) and tail convergence at `atol=1e-5`; full-series equality is not asserted |
| `KAMA`          | ✅        | ✅ Exact  | First output at index `timeperiod` (seed not emitted). `TestKAMA::test_values_match` asserts equality at `atol=1e-6` on the random-walk price fixture, and NaN counts match. **Upgraded from ✅ Close:** the zero-volatility divergence that held it back is fixed — the efficiency ratio now reproduces `TA_KAMA`'s `sumROC1 <= periodROC || TA_IS_ZERO(sumROC1)` verbatim, both the signed clamp and the `±1e-14` band. Parity now holds across plateau input too: `test_kama_matches_talib_on_zero_volatility_input` (`atol=1e-9`; residual `2.9e-14` where it was ~16), `test_kama_matches_talib_on_flat_series` (`atol=1e-12`) and `test_kama_matches_talib_on_plateau_heavy_input[2,3,5,14,30]` (`atol=1e-9`). The `xfail(strict=True)` test the old ✅ Close cited no longer exists. Note TA-Lib's flat-window behaviour is itself residue-dependent and ferro-ta reproduces that, rather than idealizing it — see [`KAMA` on zero-volatility input](#kama-on-zero-volatility-input--fixed) |
| `MA`            | ✅        | ✅ Exact † | Generic type-selectable MA. `TestMAMatype` now compares the **dispatcher itself** against `talib.MA` at every compatible `matype` (`0`–`6`, `8`): values at `atol=1e-6` over ≥ 440 compared bars (worst observed `2.6e-12`) and NaN counts exactly `ma_lookback(timeperiod, matype)`. `matype = 7` is excluded and separately asserted to *disagree* |
| `MAMA`          | ✅        | ⚠️ Corr  | MESA Adaptive MA. NaN counts match; MAMA gated at `corr > 0.95`, FAMA at `corr > 0.80`. **Not reachable through any `matype` value** — see [The `matype` enum](#the-matype-enum) |
| `MAVP`          | ✅        | ✅ Close † | MA with variable period. `TestMAVPMatype` compares against `TA_MAVP` argument-for-argument. **Window MAs (`0` SMA, `2` WMA, `5` TRIMA) match exactly** over the whole valid region at `atol=1e-6` (≥ 400 bars compared); the **recursive family (`1`, `3`, `4`, `6`, `8`) matches only on the converged tail** (last 30% of bars, ≥ 120 compared) because TA-Lib computes each leg's MA from the output start index while ferro-ta seeds from bar 0 — so expect early-bar divergence there, not drift everywhere. NaN counts are exactly `ma_lookback(maxperiod, matype)` at every `matype`, matching `TA_MAVP_Lookback`. Defaults to `matype = 0`, which *is* TA-Lib's default, so an unqualified call lands on the exact path. The `matype = 0` path keeps the original per-bar window sum, bit-identical to the pre-`matype` kernel but not to the streaming recurrence in `SMA` |
| `MIDPOINT`      | ✅        | ✅ Exact  | Midpoint over period. **Deliberate divergence on non-finite input:** a window containing a `NaN` now yields `NaN`. TA-Lib's C loop skips `NaN` — an artifact of `<`/`>` being false for `NaN`, not a specified behaviour — so the two disagree there. Parity is asserted on finite series only |
| `MIDPRICE`      | ✅        | ✅ Exact  | Midpoint price over period. Same deliberate `NaN`-propagation divergence as `MIDPOINT` (`crates/ferro_ta_core/src/overlap/midpoint.rs`) |
| `SAR`           | ✅        | ⚠️ Shape | Parabolic SAR. Length, NaN count and positivity match; the only value gate is `corr > 0.90` — reversal history diverges from floating-point accumulation in the early bars |
| `SAREXT`        | ✅        | ⚠️ Shape | Parabolic SAR Extended. Length and NaN count only; no value or correlation gate |
| `SMA`           | ✅        | ✅ Exact  | Simple Moving Average                                 |
| `T3`            | ✅        | ✅ Close  | Six SMA-seeded EMAs (TA-Lib cascade); long warmup. Tail convergence asserted at `atol=1e-3` |
| `TEMA`          | ✅        | ✅ Close  | SMA-seeded composed EMA (`3*E1 − 3*E2 + E3`); mid-series convergence at `atol=1e-2` |
| `TRIMA`         | ✅        | ✅ Exact  | Triangular Moving Average                             |
| `WMA`           | ✅        | ✅ Exact  | Weighted Moving Average                               |


## Momentum Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                                        |
| --------------- | -------- | -------- | ---------------------------------------------------------------------------- |
| `ADX`           | ✅        | ⚠️ Corr  | Avg Directional Movement Index. NaN count and length match; only `corr > 0.99` is asserted — values are not compared |
| `ADXR`          | ✅        | ⚠️ Corr  | ADX Rating (inherits ADX). NaN count within ±1; only `corr > 0.95` is asserted |
| `APO`           | ✅        | ✅ Exact † | Absolute Price Oscillator. **Upgraded from ⚠️ Shape:** `TestAPOMatype::test_values_match` asserts value equality against `talib.APO` at every compatible `matype` (`atol=1e-6`, ≥ 300 bars compared; worst observed `4.9e-12`, at `matype=2`/WMA), and `test_nan_count_match` pins the warmup to `ma_lookback(slowperiod, matype)` exactly. The old `TestAPO` docstring — "values differ (EMA-based when matype != SMA)" — is **false**: the values do not differ, the *defaults* do. ferro-ta defaults `matype = 1` (EMA) where TA-Lib defaults `0` (SMA), so an unqualified call still does not match (`test_default_matype_is_ema_not_talib_sma` asserts exactly that, in both directions). `TestAPO`'s NaN-count-only check is now redundant |
| `AROON`         | ✅        | ✅ Exact  | Aroon Up/Down                                                                |
| `AROONOSC`      | ✅        | ✅ Exact  | Aroon Oscillator                                                             |
| `BOP`           | ✅        | ✅ Exact  | Balance Of Power                                                             |
| `CCI`           | ✅        | ✅ Exact  | Commodity Channel Index (TA-Lib-compatible MAD formula). Values verified at `atol=1e-6` in `TestNumericalParity`; `TestCCI` alone would only establish `corr > 0.99` |
| `CMO`           | ✅        | ✅ Exact  | Chande Momentum Oscillator (Wilder smoothing, same seed as RSI)              |
| `DX`            | ✅        | ⚠️ Corr  | Directional Movement Index. NaN count and range match; only `corr > 0.99` is asserted |
| `MACD`          | ✅        | ✅ Close  | MACD (EMA-based). NaN counts match; tail convergence asserted at `atol=1e-2`. The `hist = macd - signal` identity holds to `1e-10` |
| `MACDEXT`       | ✅        | ✅ Close † | MACD with controllable MA type. `TestMACDEXTMatype` compares all three outputs against `talib.MACDEXT` with the matypes matched on both sides. **Window MAs (`0`, `2`, `5`) match exactly** (`atol=1e-6`, ≥ 400 bars); the **recursive family (`1`, `3`, `4`, `6`, `8`) matches on the converged tail only** (last 30%, ≥ 120 bars) — TA-Lib seeds each leg's MA at the output start index, ferro-ta seeds at bar 0, confirmed by `talib.MACDEXT(matype=1)` being bit-identical to `talib.MACD`. NaN counts now agree **exactly** at every `matype` (measured 33/33/33/66/99/33/35/198 on both sides for 12/26/9 at `matype` `0`–`6` and `8`): a warm-up bug is fixed — `macd_start` was `slowperiod - 1`, the SMA-family lookback, where it must be `max(ma_lookback(fastperiod, fastmatype), ma_lookback(slowperiod, slowmatype))` before the signal leg's own `ma_lookback` stacks on top. That made `MACDEXT(fastmatype=6, slowmatype=6, signalmatype=6)` return **all `NaN`**, which this document should not have been silent about; see [`MACDEXT` warm-up](#macdext-warm-up--fixed) for the derivation and the 729-combination Rust test that pins it (`macdext_warmup_is_max_leg_lookback_plus_signal_lookback`). The Python suite's own gate is still the weaker `TestMACDEXTMatype::test_nan_count_match` (±1), whose "KAMA's off-by-one seed can shift the signal leg by a single bar" docstring no longer describes anything observed. Defaults are `fastmatype = slowmatype = signalmatype = 1` (EMA) against TA-Lib's `0` (SMA) — verified from the runtime signature by `test_runtime_defaults_are_ema`, not read off a stub — so an unqualified call does not match TA-Lib. All three arguments accept `0`–`8`; the old cap at `7` is gone (`test_matype_8_is_accepted`) |
| `MACDFIX`       | ✅        | ⚠️ Shape | MACD Fixed 12/26 (EMA-based). `TestMACDFIX` asserts only NaN count and length; values are not compared |
| `MFI`           | ✅        | ✅ Exact  | Money Flow Index                                                             |
| `MINUS_DI`      | ✅        | ⚠️ Corr  | Minus Directional Indicator. NaN count matches; only `corr > 0.99` is asserted |
| `MINUS_DM`      | ✅        | ⚠️ Corr  | Minus Directional Movement. NaN count within ±1 (Wilder seed); only `corr > 0.99` is asserted |
| `MOM`           | ✅        | ✅ Exact  | Momentum                                                                     |
| `PLUS_DI`       | ✅        | ⚠️ Corr  | Plus Directional Indicator. NaN count matches; only `corr > 0.99` is asserted |
| `PLUS_DM`       | ✅        | ⚠️ Corr  | Plus Directional Movement. Length matches; only `corr > 0.99` is asserted |
| `PPO`           | ✅        | ✅ Exact † | Percentage Price Oscillator — the rating covers the **PPO line only**. `TA_PPO` returns a single array; ferro-ta added a `signalperiod = 9` and returns `(ppo, signal, hist)`, so only `result[0]` has a TA-Lib counterpart — signal and histogram are ferro-ta extensions with nothing to compare against. **Upgraded from ⚠️ Corr:** `TestPPOMatype::test_ppo_line_values_match` asserts the PPO line at every compatible `matype` (`atol=1e-6`, ≥ 300 bars; worst observed `1.2e-11` at `matype=2`/WMA), with NaN counts exactly `ma_lookback(slowperiod, matype)`. As with `APO`, the values are exact at a matched `matype` while the **defaults differ** (`1`/EMA here, `0`/SMA in TA-Lib), so an unqualified call does not match — `test_default_matype_is_ema_not_talib_sma` asserts both halves. `TestPPO`'s `corr > 0.85` gate is now redundant |
| `ROC`           | ✅        | ✅ Exact  | Rate of Change                                                               |
| `ROCP`          | ✅        | ✅ Exact  | Rate of Change Percentage                                                    |
| `ROCR`          | ✅        | ✅ Exact  | Rate of Change Ratio                                                         |
| `ROCR100`       | ✅        | ✅ Exact  | Rate of Change Ratio × 100                                                   |
| `RSI`           | ✅        | ✅ Close  | Relative Strength Index (TA-Lib Wilder seeding). NaN count matches and `TestNumericalParity::test_rsi_values_allclose` asserts values at `atol=1e-8` over the valid region |
| `STOCH`         | ✅        | ✅ Exact † | Stochastic. `TestSTOCHVsTalib` asserts matching NaN counts and slow %K / %D values at `atol=1e-8`. `slowk_matype` / `slowd_matype` default to `0` (SMA), TA-Lib's own defaults, and follow TA-Lib's **interleaved** argument order — each matype immediately after the period it types |
| `STOCHF`        | ✅        | ✅ Exact † | Fast Stochastic. %K and %D both value-verified at `atol=1e-6` against `talib.STOCHF(..., fastd_matype=0)`; `fastd_matype` defaults to `0` (SMA), TA-Lib's default. Both outputs NaN until %D is valid |
| `STOCHRSI`      | ✅        | ✅ Exact † | Stochastic RSI (Wilder-seeded RSI). %K value-verified at `atol=1e-8`; NaN count may differ by up to 2 — the RSI seed differs slightly from TA-Lib's, so ferro-ta can emit values **up to two bars sooner** (`crates/ferro_ta_core/src/momentum/stoch.rs`). `fastd_matype` defaults to `0` (SMA). The %D value gate covers every compatible `matype` **except `6`** (KAMA, `STOCHRSI_D_MATYPES`), and that exclusion is an **upstream divergence, not a ferro-ta defect**: `TA_KAMA` is ill-conditioned on %K's `0`/`100` plateaus, so ferro-ta's `5.8e-13` rounding difference in %K moves *TA-Lib's own* KAMA by 19.2, while ferro-ta's KAMA tracks TA-Lib to `2.8e-14` on each input individually (`test_talib_kama_is_ill_conditioned_on_stochrsi_fastk`; see [`STOCHRSI` %D at `matype = 6`](#stochrsi-d-at-matype--6--upstream-conditioning-not-a-ferro-ta-defect)) |
| `TRIX`          | ✅        | ⚠️ Shape | 1-day ROC of Triple EMA. `TestTRIX` asserts only NaN count and length; values are not compared |
| `ULTOSC`        | ✅        | ✅ Exact  | Ultimate Oscillator                                                          |
| `WILLR`         | ✅        | ✅ Exact  | Williams' %R                                                                 |


## Volume Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                              |
| --------------- | -------- | -------- | ------------------------------------------------------------------ |
| `AD`            | ✅        | ✅ Exact  | Chaikin A/D Line                                                   |
| `ADOSC`         | ✅        | ✅ Exact  | Chaikin A/D Oscillator                                             |
| `OBV`           | ✅        | ✅ Exact  | On Balance Volume; bar 0 is `volume[0]` (TA-Lib)                   |


## Volatility Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                                   |
| --------------- | -------- | -------- | ----------------------------------------------------------------------- |
| `ATR`           | ✅        | ✅ Close  | Average True Range (TA-Lib Wilder seeding). NaN count within ±1; values verified at `atol=1e-8` over the valid region (`TestNumericalParity`) |
| `NATR`          | ✅        | ✅ Close  | Normalized ATR (TA-Lib Wilder seeding). NaN count within ±1; values verified at `atol=1e-6` |
| `TRANGE`        | ✅        | ✅ Exact  | True Range; bar 0 is NaN (TA-Lib: no previous close)                    |


## Cycle Indicators


| TA-Lib Function | ferro-ta | Accuracy | Notes                                                      |
| --------------- | -------- | -------- | ---------------------------------------------------------- |
| `HT_DCPERIOD`   | ✅        | ⚠️ Shape | Hilbert Transform Dominant Cycle Period (Ehlers algorithm). Length matches; NaN count within ±35; range-checked only |
| `HT_DCPHASE`    | ✅        | ⚠️ Shape | Hilbert Transform Dominant Cycle Phase. NaN count matches; the only value gate is ≥ 40% sign agreement |
| `HT_PHASOR`     | ✅        | ⚠️ Shape | Hilbert Transform Phasor Components (inphase, quadrature). NaN count within ±35; inphase gated at ≥ 80% sign agreement |
| `HT_SINE`       | ✅        | ⚠️ Shape | Hilbert Transform SineWave (sine, leadsine). NaN counts match; range-checked only, no value gate |
| `HT_TRENDLINE`  | ✅        | ⚠️ Shape | Hilbert Transform Instantaneous Trendline. NaN count matches; the only value gate is `corr > 0.90` |
| `HT_TRENDMODE`  | ✅        | ⚠️ Shape | Hilbert Transform Trend vs Cycle Mode (1=trend, 0=cycle). NaN count matches and output is binary; agreement with TA-Lib gated at ≥ 50% only |


## Price Transformations


| TA-Lib Function | ferro-ta | Accuracy | Notes                |
| --------------- | -------- | -------- | -------------------- |
| `AVGPRICE`      | ✅        | ✅ Exact  | Average Price        |
| `MEDPRICE`      | ✅        | ✅ Exact  | Median Price         |
| `TYPPRICE`      | ✅        | ✅ Exact  | Typical Price        |
| `WCLPRICE`      | ✅        | ✅ Exact  | Weighted Close Price |


## Statistic Functions


| TA-Lib Function       | ferro-ta | Accuracy | Notes                                                       |
| --------------------- | -------- | -------- | ----------------------------------------------------------- |
| `BETA`                | ✅        | ✅ Close  | Beta coefficient (returns-based regression). NaN count matches; values verified at `atol=1e-8` |
| `CORREL`              | ✅        | ✅ Exact  | Pearson Correlation Coefficient                             |
| `LINEARREG`           | ✅        | ✅ Exact  | Linear Regression                                           |
| `LINEARREG_ANGLE`     | ✅        | ✅ Exact  | Linear Regression Angle                                     |
| `LINEARREG_INTERCEPT` | ✅        | ✅ Exact  | Linear Regression Intercept                                 |
| `LINEARREG_SLOPE`     | ✅        | ✅ Exact  | Linear Regression Slope                                     |
| `STDDEV`              | ✅        | ✅ Close  | Standard Deviation. Verified at `atol=1e-6`, not bit-for-bit — and the gap is **deliberate**. Both kernels were rewritten onto a rolling Welford accumulator (`crates/ferro_ta_core/src/statistic.rs`, `rolling.rs`); TA-Lib's own `TA_VAR` uses the naive `Σx²/N − mean²`, which was measured at **18% relative error** on a mean-1e5 / σ-0.035 series where Welford errs `1.2e-8`. ferro-ta is therefore deliberately *more accurate than* the reference on ill-conditioned input, so bit-agreement is not expected and the tolerance is doing real work |
| `TSF`                 | ✅        | ✅ Exact  | Time Series Forecast                                        |
| `VAR`                 | ✅        | ✅ Close  | Variance. Same rolling-Welford rewrite and the same deliberate divergence from TA-Lib's naive second moment as `STDDEV`; verified at `atol=1e-6` |


## Pattern Recognition

`ferro-ta` implements all 61 candlestick patterns. All return the same
`{-100, 0, 100}` convention as TA-Lib, and every pattern is length-checked and
value-domain-checked against TA-Lib.

Agreement is measured per bar, not asserted as equality:
`TestCandlestickPatternAgreement` gates most patterns at **> 80% bar-for-bar
agreement** with TA-Lib. Five patterns carry lower, explicitly documented gates
because their body/shadow thresholds differ from TA-Lib's —
`CDLDOJI` (0.85), `CDLSPINNINGTOP` (0.75), `CDLLONGLEGGEDDOJI` (0.70),
`CDLHIGHWAVE` (0.65), and `CDLSHORTLINE` (**0.20**, where the body-size cutoff
definition differs outright and only ~25% of bars agree). `CDLENGULFING` is the
one pattern asserted as an exact array match
(`TestPatternShapeCompatibility::test_cdlengulfing_values_match`). So "may differ
slightly" holds for most of the set but not for `CDLSHORTLINE`. Thresholds live
in `CDL_AGREEMENT_THRESHOLDS` in `tests/integration/test_vs_talib.py`.


| TA-Lib Function       | ferro-ta | Notes                                               |
| --------------------- | -------- | --------------------------------------------------- |
| `CDL2CROWS`           | ✅        | Two Crows                                           |
| `CDL3BLACKCROWS`      | ✅        | Three Black Crows                                   |
| `CDL3INSIDE`          | ✅        | Three Inside Up/Down                                |
| `CDL3LINESTRIKE`      | ✅        | Three-Line Strike                                   |
| `CDL3OUTSIDE`         | ✅        | Three Outside Up/Down                               |
| `CDL3STARSINSOUTH`    | ✅        | Three Stars In The South                            |
| `CDL3WHITESOLDIERS`   | ✅        | Three Advancing White Soldiers                      |
| `CDLABANDONEDBABY`    | ✅        | Abandoned Baby                                      |
| `CDLADVANCEBLOCK`     | ✅        | Advance Block                                       |
| `CDLBELTHOLD`         | ✅        | Belt-hold                                           |
| `CDLBREAKAWAY`        | ✅        | Breakaway                                           |
| `CDLCLOSINGMARUBOZU`  | ✅        | Closing Marubozu                                    |
| `CDLCONCEALBABYSWALL` | ✅        | Concealing Baby Swallow                             |
| `CDLCOUNTERATTACK`    | ✅        | Counterattack                                       |
| `CDLDARKCLOUDCOVER`   | ✅        | Dark Cloud Cover                                    |
| `CDLDOJI`             | ✅        | Doji                                                |
| `CDLDOJISTAR`         | ✅        | Doji Star                                           |
| `CDLDRAGONFLYDOJI`    | ✅        | Dragonfly Doji                                      |
| `CDLENGULFING`        | ✅        | Engulfing Pattern                                   |
| `CDLEVENINGDOJISTAR`  | ✅        | Evening Doji Star                                   |
| `CDLEVENINGSTAR`      | ✅        | Evening Star                                        |
| `CDLGAPSIDESIDEWHITE` | ✅        | Up/Down-gap side-by-side white lines                |
| `CDLGRAVESTONEDOJI`   | ✅        | Gravestone Doji                                     |
| `CDLHAMMER`           | ✅        | Hammer                                              |
| `CDLHANGINGMAN`       | ✅        | Hanging Man                                         |
| `CDLHARAMI`           | ✅        | Harami Pattern                                      |
| `CDLHARAMICROSS`      | ✅        | Harami Cross Pattern                                |
| `CDLHIGHWAVE`         | ✅        | High-Wave Candle                                    |
| `CDLHIKKAKE`          | ✅        | Hikkake Pattern                                     |
| `CDLHIKKAKEMOD`       | ✅        | Modified Hikkake Pattern                            |
| `CDLHOMINGPIGEON`     | ✅        | Homing Pigeon                                       |
| `CDLIDENTICAL3CROWS`  | ✅        | Identical Three Crows                               |
| `CDLINNECK`           | ✅        | In-Neck Pattern                                     |
| `CDLINVERTEDHAMMER`   | ✅        | Inverted Hammer                                     |
| `CDLKICKING`          | ✅        | Kicking                                             |
| `CDLKICKINGBYLENGTH`  | ✅        | Kicking by the longer Marubozu                      |
| `CDLLADDERBOTTOM`     | ✅        | Ladder Bottom                                       |
| `CDLLONGLEGGEDDOJI`   | ✅        | Long Legged Doji                                    |
| `CDLLONGLINE`         | ✅        | Long Line Candle                                    |
| `CDLMARUBOZU`         | ✅        | Marubozu                                            |
| `CDLMATCHINGLOW`      | ✅        | Matching Low                                        |
| `CDLMATHOLD`          | ✅        | Mat Hold                                            |
| `CDLMORNINGDOJISTAR`  | ✅        | Morning Doji Star                                   |
| `CDLMORNINGSTAR`      | ✅        | Morning Star                                        |
| `CDLONNECK`           | ✅        | On-Neck Pattern                                     |
| `CDLPIERCING`         | ✅        | Piercing Pattern                                    |
| `CDLRICKSHAWMAN`      | ✅        | Rickshaw Man                                        |
| `CDLRISEFALL3METHODS` | ✅        | Rising/Falling Three Methods                        |
| `CDLSEPARATINGLINES`  | ✅        | Separating Lines                                    |
| `CDLSHOOTINGSTAR`     | ✅        | Shooting Star                                       |
| `CDLSHORTLINE`        | ✅        | Short Line Candle                                   |
| `CDLSPINNINGTOP`      | ✅        | Spinning Top                                        |
| `CDLSTALLEDPATTERN`   | ✅        | Stalled Pattern                                     |
| `CDLSTICKSANDWICH`    | ✅        | Stick Sandwich                                      |
| `CDLTAKURI`           | ✅        | Takuri (Dragonfly Doji with very long lower shadow) |
| `CDLTASUKIGAP`        | ✅        | Tasuki Gap                                          |
| `CDLTHRUSTING`        | ✅        | Thrusting Pattern                                   |
| `CDLTRISTAR`          | ✅        | Tristar Pattern                                     |
| `CDLUNIQUE3RIVER`     | ✅        | Unique 3 River                                      |
| `CDLUPSIDEGAP2CROWS`  | ✅        | Upside Gap Two Crows                                |
| `CDLXSIDEGAP3METHODS` | ✅        | Upside/Downside Gap Three Methods                   |


## Math Operators / Math Transforms

`ferro-ta` provides TA-Lib-compatible wrappers for all arithmetic and
math-transform functions. Rolling functions (`SUM`, `MAX`, `MIN`) produce `NaN`
for the first `timeperiod - 1` bars.

`TestMathOperatorsVsTalib` asserts exact agreement (`np.allclose(..., equal_nan=True)`)
for `ADD`, `SUB`, `MULT`, `DIV`, `SUM`, `MAX`, `MIN`, `SIN`, `COS`, `SQRT`,
`EXP`, `LN` and `LOG10`. The remaining entries below — `MAXINDEX`, `MININDEX`,
`ACOS`, `ASIN`, `ATAN`, `CEIL`, `FLOOR`, `TAN`, `COSH`, `SINH`, `TANH` — have
**no** TA-Lib comparison in the suite and are ❓ Untested in the sense of the
legend, even though they are thin wrappers over `f64` intrinsics.


| TA-Lib Function          | ferro-ta | Notes                         |
| ------------------------ | -------- | ----------------------------- |
| `ADD`                    | ✅        | Element-wise addition         |
| `SUB`                    | ✅        | Element-wise subtraction      |
| `MULT`                   | ✅        | Element-wise multiplication   |
| `DIV`                    | ✅        | Element-wise division         |
| `SUM`                    | ✅        | Rolling sum over *timeperiod* |
| `MAX` / `MAXINDEX`       | ✅        | Rolling maximum / index. `MAX` verified vs TA-Lib; `MAXINDEX` ❓ Untested |
| `MIN` / `MININDEX`       | ✅        | Rolling minimum / index. `MIN` verified vs TA-Lib; `MININDEX` ❓ Untested |
| `ACOS` / `ASIN` / `ATAN` | ✅        | Arc trig transforms. ❓ Untested vs TA-Lib |
| `CEIL` / `FLOOR`         | ✅        | Round up / down. ❓ Untested vs TA-Lib |
| `COS` / `SIN` / `TAN`    | ✅        | Trig transforms. `COS` / `SIN` verified vs TA-Lib; `TAN` ❓ Untested |
| `COSH` / `SINH` / `TANH` | ✅        | Hyperbolic transforms. ❓ Untested vs TA-Lib |
| `EXP` / `LN` / `LOG10`   | ✅        | Exponential / log transforms  |
| `SQRT`                   | ✅        | Square root                   |


## Implementation Coverage Summary


Counts are of distinct TA-Lib function names, measured against
`talib.get_functions()` for TA-Lib 0.6.4.

| Category                    | Implemented | Not Implemented         |
| --------------------------- | ----------- | ----------------------- |
| Overlap Studies             | 16          | 1 (`ACCBANDS`)          |
| Momentum Indicators         | 30          | 1 (`IMI`)               |
| Volume Indicators           | 3           | 0                       |
| Volatility Indicators       | 3           | 0                       |
| Cycle Indicators            | 6           | 0                       |
| Price Transforms            | 4           | 1 (`AVGDEV`)            |
| Statistic Functions         | 9           | 0                       |
| Pattern Recognition         | 61          | 0                       |
| Math Operators / Transforms | 24          | 2 (`MINMAX`, `MINMAXINDEX`) |
| **TA-Lib subtotal**         | **156**     | **5**                   |
| Extended Indicators         | 10          | – (not in TA-Lib)       |
| Streaming Classes           | 9           | – (not in TA-Lib)       |
| **Total implemented**       | **175**     |                         |


> `ferro-ta` implements 156 of TA-Lib's 161 functions. NaN values are placed
> at the beginning of each output array for the warmup period.

Implemented is not the same as verified, and this file has now been audited
twice against `tests/integration/test_vs_talib.py`.

**First pass** (against the suite as it then stood): of the 62 rows previously
marked ✅ Exact or ✅ Close, 45 held their rating, 3 (`EMA`, `STDDEV`, `VAR`)
moved from ✅ Exact to ✅ Close, and 14 dropped out of ✅ altogether — 8 to
⚠️ Corr, 3 to ⚠️ Shape and 3 to ❓ Untested. One row, `STOCH`, was
*strengthened* to ✅ Exact. That left **48 ✅ / 20 ⚠️ / 3 ❓** across 71 rated
rows.

**Second pass** (after 156 new `matype` tests were added against TA-Lib 0.7.1):
six rows changed, five of them **upwards**, on the strength of value gates that
did not exist before.

| Row       | Was          | Now       | Substantiating test |
| --------- | ------------ | --------- | ------------------- |
| `APO`     | ⚠️ Shape     | ✅ Exact † | `TestAPOMatype::test_values_match` / `test_nan_count_match` / `test_default_matype_is_ema_not_talib_sma` |
| `PPO`     | ⚠️ Corr      | ✅ Exact † | `TestPPOMatype::test_ppo_line_values_match` (PPO line only) / `test_ppo_line_nan_count_match` / `test_default_matype_is_ema_not_talib_sma` |
| `MA`      | ❓ Untested  | ✅ Exact † | `TestMAMatype::test_values_match` / `test_nan_count_match` |
| `MAVP`    | ❓ Untested  | ✅ Close † | `TestMAVPMatype::test_window_matypes_match_exactly` / `test_recursive_matypes_match_on_converged_tail` / `test_nan_count_match` |
| `MACDEXT` | ❓ Untested  | ✅ Close † | `TestMACDEXTMatype::test_window_matypes_match_exactly` / `test_recursive_matypes_match_on_converged_tail` / `test_runtime_defaults_are_ema` / `test_matype_8_is_accepted` |
| `KAMA`    | ✅ Exact     | ✅ Close  | `TestKAMA::test_values_match` (exact on the price fixture) *against* `test_kama_diverges_on_zero_volatility_input` (`xfail(strict=True)`, ~16.5 max abs diff) — **since reversed, see the third pass below** |

The `APO` and `PPO` upgrades reverse first-pass downgrades that were made for a
reason **which does not exist**: the old `TestAPO` docstring claimed the values
differ when only the *defaults* do. The `KAMA` downgrade goes the other way —
its previous ✅ Exact was unsupported for plateau-heavy input.

Running total after this pass: **53 ✅ (39 Exact + 14 Close) / 18 ⚠️ (8 Corr +
10 Shape) / 0 ❓** across the same 71 rated rows.

**Third pass** (after the `KAMA` zero-volatility branch and the `MACDEXT`
warm-up were fixed). One rating moved; the other two rows changed their
*evidence* without changing their rating.

| Row        | Was      | Now       | Substantiating test |
| ---------- | -------- | --------- | ------------------- |
| `KAMA`     | ✅ Close | ✅ Exact  | `test_kama_matches_talib_on_zero_volatility_input` (`atol=1e-9`) / `test_kama_matches_talib_on_flat_series` (`atol=1e-12`) / `test_kama_matches_talib_on_plateau_heavy_input[2,3,5,14,30]` (`atol=1e-9`), alongside the pre-existing `TestKAMA::test_values_match` |
| `STOCHRSI` | ✅ Exact † | ✅ Exact † | Unchanged rating, corrected attribution: the `fastd_matype = 6` %D exclusion is upstream `TA_KAMA` conditioning, not the `KAMA` bug — it survives the fix at 19.2 (`test_talib_kama_is_ill_conditioned_on_stochrsi_fastk`) |
| `MACDEXT`  | ✅ Close † | ✅ Close † | Unchanged rating, stronger evidence: the `macd_start` warm-up bug (all-`NaN` at `matype = 6` on all three legs) is fixed and NaN counts now match `TA_MACDEXT_Lookback` exactly (`macdext_warmup_is_max_leg_lookback_plus_signal_lookback`, 729 matype combinations). It stays ✅ Close because the recursive family is still only asserted on the converged tail — the seeding difference is unrelated to the warm-up bug |

The `KAMA` upgrade reverses the second pass's downgrade, and it needed the same
class of evidence the downgrade did: three value assertions across flat,
plateau-heavy and StochRSI-shaped input, not the absence of a failing test. The
`xfail(strict=True)` the ✅ Close cited is gone, so nothing would have flagged a
stale rating here — which is why the two ratings that did *not* move are listed
too.

Running total after this pass: **53 ✅ (40 Exact + 13 Close) / 18 ⚠️ (8 Corr +
10 Shape) / 0 ❓** across the same 71 rated rows. No ❓ Untested rows remain in
the tables; the only ❓ ratings left are the math-transform wrappers called out
in prose above. Where a rating names a tolerance or a threshold, that number is
the one the test actually asserts, and where a rating could not be
substantiated the note says what *is* verified rather than a
weaker-but-still-confident claim about values.
