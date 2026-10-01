//! C exports for `ferro_ta_core::volume`.

crate::ffi_exports! {
    group: "volume",
    /// Compute On-Balance Volume (OBV).
    ft_obv = volume::obv(close, volume)[]
        -> [out: f64];
    /// Compute the Money Flow Index (MFI).
    ft_mfi = volume::mfi(high, low, close, volume)[timeperiod: period]
        -> [out: f64];
    /// Chaikin Accumulation/Distribution Line.
    ft_ad = volume::ad(high, low, close, volume)[]
        -> [out: f64];
    /// Chaikin A/D Oscillator: fast EMA of AD minus slow EMA of AD.
    ft_adosc = volume::adosc(high, low, close, volume)[fastperiod: period, slowperiod: period]
        -> [out: f64] if fastperiod < slowperiod;
}
