//! C exports for `ferro_ta_core::cycle`.

crate::ffi_exports! {
    group: "cycle",
    /// Hilbert Transform Instantaneous Trendline (Ehlers).
    ft_ht_trendline = cycle::ht_trendline(close)[]
        -> [out: f64];
    /// Hilbert Transform Dominant Cycle Period in bars.
    ft_ht_dcperiod = cycle::ht_dcperiod(close)[]
        -> [out: f64];
    /// Hilbert Transform Dominant Cycle Phase in degrees.
    ft_ht_dcphase = cycle::ht_dcphase(close)[]
        -> [out: f64];
    /// Hilbert Transform Phasor components. Returns `(inphase, quadrature)`.
    ft_ht_phasor = cycle::ht_phasor(close)[]
        -> [out_inphase: f64, out_quadrature: f64];
    /// Hilbert Transform SineWave. Returns `(sine, leadsine)` where leadsine
    ft_ht_sine = cycle::ht_sine(close)[]
        -> [out_sine: f64, out_leadsine: f64];
    /// Hilbert Transform Trend vs Cycle Mode: 1 = trending, 0 = cycling.
    ft_ht_trendmode = cycle::ht_trendmode(close)[]
        -> [out: i32];
}
