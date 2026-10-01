//! Opaque-handle exports for `ferro_ta_core::streaming`.
//!
//! Handles are heap-allocated core structs (`Box::into_raw`). A handle is not
//! thread-safe: callers must not use one handle from two threads at once.
//! Every handle from `*_new` must be released exactly once with `*_free`.

use crate::status::Status;
use ferro_ta_core::streaming::{
    StreamingATR, StreamingBBands, StreamingEMA, StreamingError, StreamingMACD, StreamingRSI,
    StreamingSMA, StreamingStoch, StreamingSupertrend, StreamingVWAP,
};

/// Normalise core constructors (fallible or not) to `Result<T, Status>`.
pub trait IntoCtorResult<T> {
    fn into_ctor_result(self) -> Result<T, Status>;
}

impl<T> IntoCtorResult<T> for Result<T, StreamingError> {
    fn into_ctor_result(self) -> Result<T, Status> {
        self.map_err(|_| Status::InvalidParam)
    }
}

impl IntoCtorResult<StreamingVWAP> for StreamingVWAP {
    fn into_ctor_result(self) -> Result<StreamingVWAP, Status> {
        Ok(self)
    }
}

/// Write an `update` result (scalar or tuple) through output pointers.
pub trait WriteScalars<P> {
    /// # Safety
    /// Every pointer in `ptrs` must be valid for one write.
    unsafe fn write_to(self, ptrs: P);
}

impl WriteScalars<(*mut f64,)> for f64 {
    unsafe fn write_to(self, ptrs: (*mut f64,)) {
        // SAFETY: valid for one write per caller contract.
        unsafe { ptrs.0.write(self) };
    }
}

macro_rules! impl_write_scalars {
    ($($t:ident $v:ident $p:ident),+) => {
        impl<$($t),+> WriteScalars<($(*mut $t,)+)> for ($($t,)+) {
            unsafe fn write_to(self, ptrs: ($(*mut $t,)+)) {
                let ($($v,)+) = self;
                let ($($p,)+) = ptrs;
                // SAFETY: each pointer valid for one write per caller contract.
                $( unsafe { $p.write($v) }; )+
            }
        }
    };
}

impl_write_scalars!(A a pa, B b pb);
impl_write_scalars!(A a pa, B b pb, C c pc);

macro_rules! ffi_streams {
    (
        $(
            $(#[doc = $doc:literal])*
            $name:ident: $ty:ident as $handle:ident {
                $new_fn:ident ( $($param:ident : $kind:ident $($min:literal)?),* ),
                $update_fn:ident ( $($input:ident),+ ) -> [ $($out:ident : $elem:ident),+ ],
                $reset_fn:ident,
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
                out_handle: *mut *mut $ty,
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
                    // SAFETY: as above.
                    unsafe { out_handle.write(Box::into_raw(Box::new(state))) };
                    Ok(())
                })
            }

            /// Feed one bar; writes this bar's value(s) (NaN during warm-up).
            ///
            /// # Safety
            /// `handle` must come from the matching `_new` and not be freed;
            /// each output pointer must be valid for one write.
            #[no_mangle]
            pub unsafe extern "C" fn $update_fn(
                handle: *mut $ty,
                $($input: f64,)+
                $($out: *mut $elem,)+
            ) -> i32 {
                $crate::status::guard(|| {
                    if handle.is_null() $(|| $out.is_null())+ {
                        return Err(Status::NullPtr);
                    }
                    // SAFETY: live, exclusively-used handle per contract.
                    let state = unsafe { &mut *handle };
                    let result = state.update($($input),+);
                    // SAFETY: outputs checked non-null above.
                    unsafe { WriteScalars::write_to(result, ($($out,)+)) };
                    Ok(())
                })
            }

            /// Clear all state; the next bars warm up again.
            ///
            /// # Safety
            /// `handle` must come from the matching `_new` and not be freed.
            #[no_mangle]
            pub unsafe extern "C" fn $reset_fn(handle: *mut $ty) -> i32 {
                $crate::status::guard(|| {
                    if handle.is_null() {
                        return Err(Status::NullPtr);
                    }
                    // SAFETY: live, exclusively-used handle per contract.
                    unsafe { &mut *handle }.reset();
                    Ok(())
                })
            }

            /// Release a handle. NULL is a no-op.
            ///
            /// # Safety
            /// `handle` must come from the matching `_new` and must not be
            /// used (or freed) again afterwards.
            #[no_mangle]
            pub unsafe extern "C" fn $free_fn(handle: *mut $ty) {
                if !handle.is_null() {
                    // SAFETY: produced by Box::into_raw in `_new`, freed once.
                    drop(unsafe { Box::from_raw(handle) });
                }
            }
        )+

        /// Signature metadata for every streaming export.
        pub const STREAM_SPECS: &[$crate::spec::StreamSpec] = &[
            $(
                $crate::spec::StreamSpec {
                    name: stringify!($name),
                    handle: stringify!($handle),
                    doc: concat!($($doc, "\n",)*),
                    new_fn: stringify!($new_fn),
                    update_fn: stringify!($update_fn),
                    reset_fn: stringify!($reset_fn),
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
    /// Streaming simple moving average.
    sma: StreamingSMA as FtStreamSma {
        ft_stream_sma_new(period: period),
        ft_stream_sma_update(value) -> [out: f64],
        ft_stream_sma_reset,
        ft_stream_sma_free,
    }
    /// Streaming exponential moving average (SMA-seeded, TA-Lib compatible).
    ema: StreamingEMA as FtStreamEma {
        ft_stream_ema_new(period: period),
        ft_stream_ema_update(value) -> [out: f64],
        ft_stream_ema_reset,
        ft_stream_ema_free,
    }
    /// Streaming RSI (Wilder smoothing).
    rsi: StreamingRSI as FtStreamRsi {
        ft_stream_rsi_new(period: period),
        ft_stream_rsi_update(value) -> [out: f64],
        ft_stream_rsi_reset,
        ft_stream_rsi_free,
    }
    /// Streaming average true range.
    atr: StreamingATR as FtStreamAtr {
        ft_stream_atr_new(period: period),
        ft_stream_atr_update(high, low, close) -> [out: f64],
        ft_stream_atr_reset,
        ft_stream_atr_free,
    }
    /// Streaming Bollinger Bands.
    bbands: StreamingBBands as FtStreamBbands {
        ft_stream_bbands_new(period: period, nbdevup: float, nbdevdn: float),
        ft_stream_bbands_update(value) -> [out_upper: f64, out_middle: f64, out_lower: f64],
        ft_stream_bbands_reset,
        ft_stream_bbands_free,
    }
    /// Streaming MACD. Requires `fastperiod < slowperiod`.
    macd: StreamingMACD as FtStreamMacd {
        ft_stream_macd_new(fastperiod: period, slowperiod: period, signalperiod: period),
        ft_stream_macd_update(value) -> [out_macd: f64, out_signal: f64, out_hist: f64],
        ft_stream_macd_reset,
        ft_stream_macd_free,
    }
    /// Streaming slow stochastic.
    stoch: StreamingStoch as FtStreamStoch {
        ft_stream_stoch_new(fastk_period: period, slowk_period: period, slowd_period: period),
        ft_stream_stoch_update(high, low, close) -> [out_slowk: f64, out_slowd: f64],
        ft_stream_stoch_reset,
        ft_stream_stoch_free,
    }
    /// Streaming cumulative VWAP (session-anchored; `reset` starts a new session).
    vwap: StreamingVWAP as FtStreamVwap {
        ft_stream_vwap_new(),
        ft_stream_vwap_update(high, low, close, volume) -> [out: f64],
        ft_stream_vwap_reset,
        ft_stream_vwap_free,
    }
    /// Streaming Supertrend. `out_direction` is 1 (up) / -1 (down), 0 in warm-up.
    supertrend: StreamingSupertrend as FtStreamSupertrend {
        ft_stream_supertrend_new(period: period, multiplier: float),
        ft_stream_supertrend_update(high, low, close) -> [out_line: f64, out_direction: i8],
        ft_stream_supertrend_reset,
        ft_stream_supertrend_free,
    }
}
