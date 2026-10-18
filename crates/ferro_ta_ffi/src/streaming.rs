//! Opaque-handle exports for `ferro_ta_core::streaming`.
//!
//! Handles are heap-allocated `Live<core struct>` (`Box::into_raw`), so NaN
//! inputs are skipped without poisoning state and `_peek` can preview the
//! forming bar. A handle is not
//! thread-safe: callers must not use one handle from two threads at once.
//! Every handle from `*_new` must be released exactly once with `*_free`.

use crate::marshal::WriteScalars;
use crate::status::Status;
use ferro_ta_core::streaming::{
    Live, StreamValue, StreamingATR, StreamingBBands, StreamingCooldown, StreamingCross,
    StreamingEMA, StreamingError, StreamingIndicator, StreamingMACD, StreamingOptionGreeks,
    StreamingRSI, StreamingSMA, StreamingStoch, StreamingStreak, StreamingSupertrend,
    StreamingVWAP,
};

/// Copy a snapshot into a caller buffer; see the `*_save` contract.
///
/// # Safety
/// `out_len` valid for one write; `buf` valid for `cap` bytes if `cap > 0`.
#[doc(hidden)]
pub unsafe fn write_snapshot(
    json: &str,
    buf: *mut std::ffi::c_char,
    cap: usize,
    out_len: *mut usize,
) -> Result<(), Status> {
    // SAFETY: caller contract.
    unsafe { out_len.write(json.len()) };
    if cap < json.len() {
        return Err(Status::LengthMismatch);
    }
    if buf.is_null() {
        return Err(Status::NullPtr);
    }
    // SAFETY: `buf` holds at least `json.len()` bytes; regions are distinct.
    unsafe { std::ptr::copy_nonoverlapping(json.as_ptr(), buf.cast::<u8>(), json.len()) };
    Ok(())
}

/// Borrow a caller snapshot as UTF-8; see the `*_load` contract.
///
/// # Safety
/// `json` valid for `len` bytes (may be NULL only when `len == 0`).
#[doc(hidden)]
pub unsafe fn read_snapshot<'a>(
    json: *const std::ffi::c_char,
    len: usize,
) -> Result<&'a str, Status> {
    if json.is_null() {
        return if len == 0 {
            Ok("")
        } else {
            Err(Status::NullPtr)
        };
    }
    // SAFETY: caller contract.
    let bytes = unsafe { std::slice::from_raw_parts(json.cast::<u8>(), len) };
    std::str::from_utf8(bytes).map_err(|_| Status::InvalidParam)
}

/// Normalise core constructors (fallible or not) to `Result<T, Status>`.
pub trait IntoCtorResult<T> {
    fn into_ctor_result(self) -> Result<T, Status>;
}

impl<T> IntoCtorResult<T> for Result<T, StreamingError> {
    fn into_ctor_result(self) -> Result<T, Status> {
        self.map_err(|_| Status::InvalidParam)
    }
}

/// Bar-by-bar backtester handle: wraps the core `StreamingBacktest` so each
/// update also reports the running summary.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
#[serde(transparent)]
pub struct BacktestStream(ferro_ta_core::backtest::StreamingBacktest);

impl BacktestStream {
    pub fn new(commission_per_trade: f64, slippage_bps: f64) -> Self {
        Self(ferro_ta_core::backtest::StreamingBacktest::new(
            commission_per_trade,
            slippage_bps,
        ))
    }

    /// `(position, bar_return, equity, n_trades, total_commission, win_rate,
    /// avg_win, avg_loss, kelly_fraction)` after this bar.
    #[allow(clippy::type_complexity)]
    pub fn update(
        &mut self,
        close: f64,
        signal: f64,
    ) -> (f64, f64, f64, i64, f64, f64, f64, f64, f64) {
        let bar = self.0.on_bar(close, signal);
        let s = self.0.summary();
        (
            bar.position,
            bar.bar_return,
            bar.equity,
            bar.n_trades as i64,
            s.total_commission,
            s.win_rate,
            s.avg_win,
            s.avg_loss,
            s.kelly_fraction,
        )
    }

    pub fn reset(&mut self) {
        self.0.reset();
    }
}

/// Backtest stream input. A NaN `signal` means "flat", so only a non-finite
/// close (or an infinite signal) makes a tick unusable for the `Live` guard.
#[derive(Clone, Copy, Debug)]
pub struct BacktestBar {
    pub close: f64,
    pub signal: f64,
}

impl From<(f64, f64)> for BacktestBar {
    fn from((close, signal): (f64, f64)) -> Self {
        Self { close, signal }
    }
}

impl StreamValue for BacktestBar {
    fn has_nan(&self) -> bool {
        self.close.is_nan()
    }
    fn all_finite(&self) -> bool {
        self.close.is_finite() && !self.signal.is_infinite()
    }
    fn nan() -> Self {
        Self {
            close: f64::NAN,
            signal: f64::NAN,
        }
    }
    fn to_parts(&self) -> Vec<Option<f64>> {
        [self.close, self.signal]
            .map(|v| (!v.is_nan()).then_some(v))
            .to_vec()
    }
    fn from_parts(parts: &[Option<f64>]) -> Option<Self> {
        match parts {
            [c, s] => Some(Self {
                close: c.unwrap_or(f64::NAN),
                signal: s.unwrap_or(f64::NAN),
            }),
            _ => None,
        }
    }
}

/// `(position, bar_return, equity, n_trades, total_commission, win_rate,
/// avg_win, avg_loss, kelly_fraction)`.
pub type BacktestTuple = (f64, f64, f64, i64, f64, f64, f64, f64, f64);

/// Backtest stream output. Readiness tracks equity: the trade statistics
/// are legitimately NaN until the first closed trade.
#[derive(Clone, Copy, Debug)]
pub struct BacktestOut(pub BacktestTuple);

impl StreamValue for BacktestOut {
    fn has_nan(&self) -> bool {
        self.0 .2.is_nan()
    }
    fn all_finite(&self) -> bool {
        let (a, b, c, _, e, f, g, h, i) = self.0;
        [a, b, c, e, f, g, h, i].iter().all(|v| v.is_finite())
    }
    fn nan() -> Self {
        let n = f64::NAN;
        Self((n, n, n, 0, n, n, n, n, n))
    }
    fn to_parts(&self) -> Vec<Option<f64>> {
        let (a, b, c, d, e, f, g, h, i) = self.0;
        [a, b, c, d as f64, e, f, g, h, i]
            .map(|v| (!v.is_nan()).then_some(v))
            .to_vec()
    }
    fn from_parts(parts: &[Option<f64>]) -> Option<Self> {
        let v = |p: &Option<f64>| p.unwrap_or(f64::NAN);
        match parts {
            [a, b, c, Some(d), e, f, g, h, i] if d.fract() == 0.0 => Some(Self((
                v(a),
                v(b),
                v(c),
                *d as i64,
                v(e),
                v(f),
                v(g),
                v(h),
                v(i),
            ))),
            _ => None,
        }
    }
}

impl<P> WriteScalars<P> for BacktestOut
where
    BacktestTuple: WriteScalars<P>,
{
    unsafe fn write_to(self, ptrs: P) {
        // SAFETY: forwarded caller contract.
        unsafe { self.0.write_to(ptrs) }
    }
}

impl StreamingIndicator for BacktestStream {
    const KIND: &'static str = "backtest";
    type Input = BacktestBar;
    type Output = BacktestOut;

    fn update(&mut self, input: BacktestBar) -> BacktestOut {
        BacktestOut(BacktestStream::update(self, input.close, input.signal))
    }

    fn reset(&mut self) {
        BacktestStream::reset(self)
    }

    fn validate(&self) -> Result<(), StreamingError> {
        let s = &self.0;
        let finite = [
            s.commission_per_trade,
            s.slippage_bps,
            s.equity,
            s.total_commission,
        ];
        if finite.iter().any(|v| !v.is_finite()) || s.n_wins + s.n_losses > s.n_trades {
            return Err(StreamingError("inconsistent backtest state".to_string()));
        }
        Ok(())
    }
}

impl IntoCtorResult<BacktestStream> for BacktestStream {
    fn into_ctor_result(self) -> Result<BacktestStream, Status> {
        Ok(self)
    }
}

impl IntoCtorResult<StreamingVWAP> for StreamingVWAP {
    fn into_ctor_result(self) -> Result<StreamingVWAP, Status> {
        Ok(self)
    }
}

/// Emits one `extern "C"` feed function (`_update` or `_peek`).
macro_rules! ffi_stream_feed {
    (
        $(#[doc = $doc:literal])*
        $fn_name:ident, $ty:ident, closed = $closed:literal,
        ($($input:ident),+) -> ($($out:ident : $elem:ident),+)
    ) => {
        $(#[doc = $doc])*
        ///
        /// # Safety
        /// `handle` must come from the matching `_new` and not be freed;
        /// each output pointer must be valid for one write.
        #[no_mangle]
        pub unsafe extern "C" fn $fn_name(
            handle: *mut Live<$ty>,
            $($input: f64,)+
            $($out: *mut $elem,)+
        ) -> i32 {
            $crate::status::guard(|| {
                if handle.is_null() $(|| $out.is_null())+ {
                    return Err(Status::NullPtr);
                }
                // SAFETY: live, exclusively-used handle per contract.
                let state = unsafe { &mut *handle };
                let raw = ($($input),+);
                let input = <$ty as StreamingIndicator>::Input::from(raw);
                let result = state.update(input, $closed);
                // SAFETY: outputs checked non-null above.
                unsafe { WriteScalars::write_to(result, ($($out,)+)) };
                Ok(())
            })
        }
    };
}

macro_rules! ffi_streams {
    (
        specs = $specs:ident;
        $(
            $(#[doc = $doc:literal])*
            $name:ident: $ty:ident as $handle:ident {
                $new_fn:ident ( $($param:ident : $kind:ident $($min:literal)?),* ),
                $update_fn:ident ( $($input:ident),+ ) -> [ $($out:ident : $elem:ident),+ ],
                $peek_fn:ident,
                $reset_fn:ident,
                $save_fn:ident,
                $load_fn:ident,
                $free_fn:ident $(,)?
            }
        )+
    ) => {
        $(
            $(#[doc = $doc])*
            ///
            /// Creates a handle in `*out_handle` (set to NULL on failure).
            ///
            /// # Safety
            /// `out_handle` must be valid for one write.
            #[no_mangle]
            pub unsafe extern "C" fn $new_fn(
                $($param: $crate::ffi_cty!($kind),)*
                out_handle: *mut *mut Live<$ty>,
            ) -> i32 {
                if out_handle.is_null() {
                    return $crate::status::FT_ERR_NULL_PTR;
                }
                // SAFETY: checked non-null; valid for one write per contract.
                unsafe { out_handle.write(std::ptr::null_mut()) };
                $crate::status::guard(|| {
                    $(
                        let $param: $crate::ffi_rty!($kind) = $crate::marshal::Param::from_raw(
                            $param,
                            $crate::ffi_kind!($kind $($min)?),
                        )?;
                    )*
                    let state = IntoCtorResult::into_ctor_result($ty::new($($param),*))?;
                    let live = Box::new(Live::new(state));
                    // SAFETY: as above.
                    unsafe { out_handle.write(Box::into_raw(live)) };
                    Ok(())
                })
            }

            ffi_stream_feed! {
                /// Commit one bar; writes this bar's value(s) (NaN during
                /// warm-up). A NaN/±inf input writes NaN and leaves state untouched.
                $update_fn, $ty, closed = true, ($($input),+) -> ($($out: $elem),+)
            }

            ffi_stream_feed! {
                /// Preview the still-forming bar: writes the value(s) `update`
                /// would write for these inputs, without changing state.
                $peek_fn, $ty, closed = false, ($($input),+) -> ($($out: $elem),+)
            }

            /// Clear all state; the next bars warm up again.
            ///
            /// # Safety
            /// `handle` must come from the matching `_new` and not be freed.
            #[no_mangle]
            pub unsafe extern "C" fn $reset_fn(handle: *mut Live<$ty>) -> i32 {
                $crate::status::guard(|| {
                    if handle.is_null() {
                        return Err(Status::NullPtr);
                    }
                    // SAFETY: live, exclusively-used handle per contract.
                    unsafe { &mut *handle }.reset();
                    Ok(())
                })
            }

            /// Write a versioned JSON snapshot (UTF-8, not NUL-terminated) of the
            /// full state. `*out_len` always receives the snapshot length; when
            /// `cap` is smaller (pass `buf = NULL, cap = 0` to query) nothing is
            /// copied and `FT_ERR_LENGTH_MISMATCH` is returned.
            ///
            /// # Safety
            /// `handle` must come from the matching `_new`/`_load` and not be
            /// freed; `buf` must be valid for `cap` bytes when `cap > 0`;
            /// `out_len` must be valid for one write.
            #[no_mangle]
            pub unsafe extern "C" fn $save_fn(
                handle: *mut Live<$ty>,
                buf: *mut std::ffi::c_char,
                cap: usize,
                out_len: *mut usize,
            ) -> i32 {
                $crate::status::guard(|| {
                    if handle.is_null() || out_len.is_null() {
                        return Err(Status::NullPtr);
                    }
                    // SAFETY: live handle per contract; read-only use.
                    let json = unsafe { &*handle }.save().map_err(|_| Status::Internal)?;
                    // SAFETY: checked non-null; valid for one write.
                    unsafe { $crate::streaming::write_snapshot(&json, buf, cap, out_len) }
                })
            }

            /// Create a handle from a snapshot made by `_save` of this same
            /// stream. Wrong stream, unsupported format or malformed JSON
            /// returns `FT_ERR_INVALID_PARAM` with `*out_handle` = NULL.
            ///
            /// # Safety
            /// `json` must be valid for `len` bytes; `out_handle` must be
            /// valid for one write.
            #[no_mangle]
            pub unsafe extern "C" fn $load_fn(
                json: *const std::ffi::c_char,
                len: usize,
                out_handle: *mut *mut Live<$ty>,
            ) -> i32 {
                if out_handle.is_null() {
                    return $crate::status::FT_ERR_NULL_PTR;
                }
                // SAFETY: checked non-null; valid for one write per contract.
                unsafe { out_handle.write(std::ptr::null_mut()) };
                $crate::status::guard(|| {
                    // SAFETY: valid for `len` bytes per contract.
                    let text = unsafe { $crate::streaming::read_snapshot(json, len) }?;
                    let live = Live::<$ty>::load(text).map_err(|_| Status::InvalidParam)?;
                    // SAFETY: as above.
                    unsafe { out_handle.write(Box::into_raw(Box::new(live))) };
                    Ok(())
                })
            }

            /// Release a handle. NULL is a no-op.
            ///
            /// # Safety
            /// `handle` must come from the matching `_new` and must not be
            /// used (or freed) again afterwards.
            #[no_mangle]
            pub unsafe extern "C" fn $free_fn(handle: *mut Live<$ty>) {
                if !handle.is_null() {
                    // SAFETY: produced by Box::into_raw in `_new`, freed once.
                    drop(unsafe { Box::from_raw(handle) });
                }
            }
        )+

        /// Signature metadata for the streams of this invocation.
        pub const $specs: &[$crate::spec::StreamSpec] = &[
            $(
                $crate::spec::StreamSpec {
                    name: stringify!($name),
                    handle: stringify!($handle),
                    doc: concat!($($doc, "\n",)*),
                    new_fn: stringify!($new_fn),
                    update_fn: stringify!($update_fn),
                    peek_fn: stringify!($peek_fn),
                    reset_fn: stringify!($reset_fn),
                    save_fn: stringify!($save_fn),
                    load_fn: stringify!($load_fn),
                    free_fn: stringify!($free_fn),
                    params: &[$(
                        $crate::spec::ParamSpec {
                            name: stringify!($param),
                            kind: $crate::ffi_kind!($kind $($min)?),
                        }
                    ),*],
                    inputs: &[$(stringify!($input)),+],
                    outputs: &[$(
                        $crate::spec::OutputSpec {
                            name: stringify!($out),
                            elem: $crate::ffi_elem!($elem),
                        }
                    ),+],
                }
            ),+
        ];
    };
}

ffi_streams! {
    specs = LEGACY_STREAM_SPECS;
    /// Streaming simple moving average.
    sma: StreamingSMA as FtStreamSma {
        ft_stream_sma_new(period: period),
        ft_stream_sma_update(value) -> [out: f64],
        ft_stream_sma_peek,
        ft_stream_sma_reset,
        ft_stream_sma_save,
        ft_stream_sma_load,
        ft_stream_sma_free,
    }
    /// Streaming exponential moving average (SMA-seeded, TA-Lib compatible).
    ema: StreamingEMA as FtStreamEma {
        ft_stream_ema_new(period: period),
        ft_stream_ema_update(value) -> [out: f64],
        ft_stream_ema_peek,
        ft_stream_ema_reset,
        ft_stream_ema_save,
        ft_stream_ema_load,
        ft_stream_ema_free,
    }
    /// Streaming RSI (Wilder smoothing).
    rsi: StreamingRSI as FtStreamRsi {
        ft_stream_rsi_new(period: period),
        ft_stream_rsi_update(value) -> [out: f64],
        ft_stream_rsi_peek,
        ft_stream_rsi_reset,
        ft_stream_rsi_save,
        ft_stream_rsi_load,
        ft_stream_rsi_free,
    }
    /// Streaming average true range.
    atr: StreamingATR as FtStreamAtr {
        ft_stream_atr_new(period: period),
        ft_stream_atr_update(high, low, close) -> [out: f64],
        ft_stream_atr_peek,
        ft_stream_atr_reset,
        ft_stream_atr_save,
        ft_stream_atr_load,
        ft_stream_atr_free,
    }
    /// Streaming Bollinger Bands.
    bbands: StreamingBBands as FtStreamBbands {
        ft_stream_bbands_new(period: period, nbdevup: float, nbdevdn: float),
        ft_stream_bbands_update(value) -> [out_upper: f64, out_middle: f64, out_lower: f64],
        ft_stream_bbands_peek,
        ft_stream_bbands_reset,
        ft_stream_bbands_save,
        ft_stream_bbands_load,
        ft_stream_bbands_free,
    }
    /// Streaming MACD. Requires `fastperiod < slowperiod`.
    macd: StreamingMACD as FtStreamMacd {
        ft_stream_macd_new(fastperiod: period, slowperiod: period, signalperiod: period),
        ft_stream_macd_update(value) -> [out_macd: f64, out_signal: f64, out_hist: f64],
        ft_stream_macd_peek,
        ft_stream_macd_reset,
        ft_stream_macd_save,
        ft_stream_macd_load,
        ft_stream_macd_free,
    }
    /// Streaming slow stochastic.
    stoch: StreamingStoch as FtStreamStoch {
        ft_stream_stoch_new(fastk_period: period, slowk_period: period, slowd_period: period),
        ft_stream_stoch_update(high, low, close) -> [out_slowk: f64, out_slowd: f64],
        ft_stream_stoch_peek,
        ft_stream_stoch_reset,
        ft_stream_stoch_save,
        ft_stream_stoch_load,
        ft_stream_stoch_free,
    }
    /// Streaming cumulative VWAP (session-anchored; `reset` starts a new session).
    vwap: StreamingVWAP as FtStreamVwap {
        ft_stream_vwap_new(),
        ft_stream_vwap_update(high, low, close, volume) -> [out: f64],
        ft_stream_vwap_peek,
        ft_stream_vwap_reset,
        ft_stream_vwap_save,
        ft_stream_vwap_load,
        ft_stream_vwap_free,
    }
    /// Streaming backtester: feed each bar's close and the target position for the next bar (NaN = flat). Reports the position, bar return, equity (starting at 1.0), the number of closed trades and running statistics over closed trades (win rate, average win/loss, Kelly fraction). commission_per_trade is subtracted from equity on each position change; slippage is in basis points.
    backtest: BacktestStream as FtStreamBacktest {
        ft_stream_backtest_new(commission_per_trade: float, slippage_bps: float),
        ft_stream_backtest_update(close, signal) -> [
            out_position: f64,
            out_bar_return: f64,
            out_equity: f64,
            out_n_trades: i64,
            out_total_commission: f64,
            out_win_rate: f64,
            out_avg_win: f64,
            out_avg_loss: f64,
            out_kelly_fraction: f64
        ],
        ft_stream_backtest_peek,
        ft_stream_backtest_reset,
        ft_stream_backtest_save,
        ft_stream_backtest_load,
        ft_stream_backtest_free,
    }
    /// Streaming Supertrend. `out_direction` is 1 (up) / -1 (down), 0 in warm-up.
    supertrend: StreamingSupertrend as FtStreamSupertrend {
        ft_stream_supertrend_new(period: period, multiplier: float),
        ft_stream_supertrend_update(high, low, close) -> [out_line: f64, out_direction: i8],
        ft_stream_supertrend_peek,
        ft_stream_supertrend_reset,
        ft_stream_supertrend_save,
        ft_stream_supertrend_load,
        ft_stream_supertrend_free,
    }
    /// Dead-band cross of `a` over `b`: `out` is 1 (up), -1 (down) or 0.
    /// Inside `±dead_band` the side is unchanged, so noise fires at most once.
    cross: StreamingCross as FtStreamCross {
        ft_stream_cross_new(dead_band: float),
        ft_stream_cross_update(a, b) -> [out: i8],
        ft_stream_cross_peek,
        ft_stream_cross_reset,
        ft_stream_cross_save,
        ft_stream_cross_load,
        ft_stream_cross_free,
    }
    /// "N bars in a row": `out` is 1 while the run of non-zero inputs is >= n.
    streak: StreamingStreak as FtStreamStreak {
        ft_stream_streak_new(n: period),
        ft_stream_streak_update(condition) -> [out: i8],
        ft_stream_streak_peek,
        ft_stream_streak_reset,
        ft_stream_streak_save,
        ft_stream_streak_load,
        ft_stream_streak_free,
    }
    /// Passes the first non-zero signal's sign, then mutes the next `bars` bars.
    cooldown: StreamingCooldown as FtStreamCooldown {
        ft_stream_cooldown_new(bars: count),
        ft_stream_cooldown_update(signal) -> [out: i8],
        ft_stream_cooldown_peek,
        ft_stream_cooldown_reset,
        ft_stream_cooldown_save,
        ft_stream_cooldown_load,
        ft_stream_cooldown_free,
    }
    /// Live implied volatility and greeks for one option contract. The IV
    /// solver is seeded with the previous tick's IV; an unsolvable price
    /// (e.g. below intrinsic) writes NaN everywhere and keeps the seed.
    option_greeks: StreamingOptionGreeks as FtStreamOptionGreeks {
        ft_stream_option_greeks_new(
            model: pricing_model,
            kind: option_kind,
            strike: float,
            rate: float,
            carry: float
        ),
        ft_stream_option_greeks_update(option_price, underlying, time_to_expiry) -> [
            out_iv: f64,
            out_delta: f64,
            out_gamma: f64,
            out_vega: f64,
            out_theta: f64,
            out_rho: f64
        ],
        ft_stream_option_greeks_peek,
        ft_stream_option_greeks_reset,
        ft_stream_option_greeks_save,
        ft_stream_option_greeks_load,
        ft_stream_option_greeks_free,
    }
}

/// C exports for every registry stream (`ferro_ta_core::for_each_stream!`):
/// `ft_stream_<name>_{new,update,peek,reset,save,load,free}` over handle
/// `FtStream<UpperCamel(name)>`, params/inputs/outputs exactly as registered.
macro_rules! ffi_stream_gen {
    ($(
        $name:ident => $Ty:ident, doc $doc:literal,
            params($($p:ident: $kind:ident $(($min:literal))? = $default:tt),* $(,)?),
            inputs($($i:ident),+ $(,)?),
            outputs($($o:ident: $elem:ident),+ $(,)?),
            $imp:ident($($body:tt)*);
    )*) => {
        pastey::paste! {
            ffi_streams! {
                specs = GENERATED_STREAM_SPECS;
                $(
                    #[doc = $doc]
                    $name: $Ty as [<FtStream $name:upper_camel>] {
                        [<ft_stream_ $name _new>]($($p: $kind $($min)?),*),
                        [<ft_stream_ $name _update>]($($i),+) -> [$($o: $elem),+],
                        [<ft_stream_ $name _peek>],
                        [<ft_stream_ $name _reset>],
                        [<ft_stream_ $name _save>],
                        [<ft_stream_ $name _load>],
                        [<ft_stream_ $name _free>],
                    }
                )*
            }
        }
    };
}

mod generated {
    use super::*;
    use ferro_ta_core::streaming::*;

    ferro_ta_core::for_each_stream!(ffi_stream_gen);
}

pub use generated::*;

// Registry streams re-check periods against the core cap on snapshot load;
// keep it equal to the constructor cap.
const _: () =
    assert!(crate::marshal::MAX_PERIOD as usize == ferro_ta_core::streaming::MAX_STREAM_PERIOD);

/// `a` followed by `b` (const, so [`STREAM_SPECS`] stays a plain slice).
const fn concat_specs<const N: usize>(
    a: &[crate::spec::StreamSpec],
    b: &[crate::spec::StreamSpec],
) -> [crate::spec::StreamSpec; N] {
    assert!(a.len() + b.len() == N);
    let mut out = [a[0]; N];
    let mut k = 0;
    while k < a.len() {
        out[k] = a[k];
        k += 1;
    }
    while k < N {
        out[k] = b[k - a.len()];
        k += 1;
    }
    out
}

const ALL_STREAM_SPECS: [crate::spec::StreamSpec;
    LEGACY_STREAM_SPECS.len() + GENERATED_STREAM_SPECS.len()] =
    concat_specs(LEGACY_STREAM_SPECS, GENERATED_STREAM_SPECS);

/// Signature metadata for every streaming export: the hand-written streams,
/// then the registry streams in registry order.
pub const STREAM_SPECS: &[crate::spec::StreamSpec] = &ALL_STREAM_SPECS;
