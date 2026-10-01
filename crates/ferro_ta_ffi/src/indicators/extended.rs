//! C exports for `ferro_ta_core::extended`.

crate::ffi_exports! {
    group: "extended",
    /// ATR-based Supertrend indicator.
    ft_supertrend = extended::supertrend(high, low, close)[timeperiod: period, multiplier: float]
        -> [out_line: f64, out_direction: i8];
    /// Donchian Channels — rolling highest high / lowest low.
    ft_donchian = extended::donchian(high, low)[timeperiod: period]
        -> [out_upper: f64, out_middle: f64, out_lower: f64];
    /// Keltner Channels — EMA +/- (multiplier x ATR).
    ft_keltner_channels = extended::keltner_channels(high, low, close)[timeperiod: period, atr_period: period, multiplier: float]
        -> [out_upper: f64, out_middle: f64, out_lower: f64];
    /// Chandelier Exit — ATR-based trailing stop levels.
    ft_chandelier_exit = extended::chandelier_exit(high, low, close)[timeperiod: period, multiplier: float]
        -> [out_long_exit: f64, out_short_exit: f64];
    /// Choppiness Index — measures market choppiness vs trending.
    ft_choppiness_index = extended::choppiness_index(high, low, close)[timeperiod: period]
        -> [out: f64];
    /// Hull Moving Average (HMA).
    ft_hull_ma = extended::hull_ma(close)[timeperiod: period]
        -> [out: f64];
    /// Directional Movement Index as `(PLUS_DI, MINUS_DI, ADX)`.
    ft_dmi = extended::dmi(high, low, close)[timeperiod: period]
        -> [out_plus_di: f64, out_minus_di: f64, out_adx: f64];
    /// Williams Fractals: local swing high / swing low with `timeperiod` bars on each side of the pivot.
    ft_williams_fractals = extended::williams_fractals(high, low)[timeperiod: period]
        -> [out_up: f64, out_down: f64];
    /// Random Walk Index (Poulos): max over lookbacks `2..=timeperiod` of
    ft_rwi = extended::rwi(high, low, close)[timeperiod: period 2]
        -> [out_rwi_high: f64, out_rwi_low: f64];
    /// # Returns `(tenkan, kijun, senkou_a, senkou_b, chikou)` arrays. Mismatched input lengths yield all `NaN`.
    ft_ichimoku = extended::ichimoku(high, low, close)[tenkan_period: period, kijun_period: period, senkou_b_period: period, displacement: count]
        -> [out_tenkan: f64, out_kijun: f64, out_senkou_a: f64, out_senkou_b: f64, out_chikou: f64];
    /// Elder Ray Index: bull power and bear power versus an EMA of close.
    ft_elder_ray = extended::elder_ray(high, low, close)[timeperiod: period]
        -> [out_bull: f64, out_bear: f64];
    /// Ehlers Fisher Transform of median price, plus a 1-bar trigger.
    ft_fisher = extended::fisher(high, low)[timeperiod: period]
        -> [out_fisher: f64, out_signal: f64];
    /// Connors RSI: average of price RSI, streak RSI, and percent rank of ROC.
    ft_crsi = extended::crsi(close)[timeperiod: period, streakperiod: period, rankperiod: period]
        -> [out: f64];
    /// Awesome Oscillator: `SMA(median, fast) − SMA(median, slow)`.
    ft_ao = extended::ao(high, low)[fastperiod: period, slowperiod: period]
        -> [out: f64] if fastperiod < slowperiod;
    /// Accelerator Oscillator: `AO − SMA(AO, timeperiod)`.
    ft_ac = extended::ac(high, low)[fastperiod: period, slowperiod: period, timeperiod: period]
        -> [out: f64] if fastperiod < slowperiod;
    /// Price Oscillator (SMA): `SMA(close, fast) − SMA(close, slow)`.
    ft_po = extended::po(close)[fastperiod: period, slowperiod: period]
        -> [out: f64] if fastperiod < slowperiod;
    /// Detrended Price Oscillator: `close[i − shift] − SMA(close, timeperiod)`, where `shift = timeperiod / 2 + 1`.
    ft_dpo = extended::dpo(close)[timeperiod: period]
        -> [out: f64];
    /// Relative Vigor Index and its 4-bar weighted signal.
    ft_rvi = extended::rvi(open, high, low, close)[timeperiod: period]
        -> [out_rvi: f64, out_signal: f64];
    /// Chaikin Oscillator — same math as [`volume::adosc`].
    ft_cho = extended::cho(high, low, close, volume)[fastperiod: period, slowperiod: period]
        -> [out: f64] if fastperiod < slowperiod;
    /// Know Sure Thing: weighted sum of four ROC SMAs, plus a signal SMA.
    ft_kst = extended::kst(close)[roc1: period, roc2: period, roc3: period, roc4: period, sma1: period, sma2: period, sma3: period, sma4: period, signalperiod: period]
        -> [out_kst: f64, out_signal: f64];
    /// True Strength Index and an EMA signal of that series.
    ft_tsi = extended::tsi(close)[longperiod: period, shortperiod: period, signalperiod: period]
        -> [out_tsi: f64, out_signal: f64];
    /// Vortex Indicator: `+VI` and `−VI` over `timeperiod`.
    ft_vortex = extended::vortex(high, low, close)[timeperiod: period]
        -> [out_plus_vi: f64, out_minus_vi: f64];
    /// Schaff Trend Cycle: stochastic of MACD, double-smoothed (`d1`, `d2`).
    ft_stc = extended::stc(close)[fastperiod: period, slowperiod: period, cycleperiod: period, d1: period, d2: period]
        -> [out: f64] if fastperiod < slowperiod;
    /// Gator Oscillator from the Alligator jaw / teeth / lips.
    ft_gator = extended::gator(high, low)[jaw_period: period, jaw_shift: count, teeth_period: period, teeth_shift: count, lips_period: period, lips_shift: count]
        -> [out_upper: f64, out_lower: f64];
    /// Coppock Curve: `WMA(ROC(roc1) + ROC(roc2), wma_period)`.
    ft_coppock = extended::coppock(close)[wma_period: period, roc1: period, roc2: period]
        -> [out: f64];
    /// Rolling median of `real` over `timeperiod`.
    ft_median = extended::median(real)[timeperiod: period]
        -> [out: f64];
    /// Median bands: rolling median of `(high + low) / 2`, ATR envelopes, and an EMA of the median.
    ft_median_bands = extended::median_bands(high, low, close)[timeperiod: period, atr_period: period, multiplier: float]
        -> [out_median: f64, out_upper: f64, out_lower: f64, out_median_ema: f64];
    /// Rolling mode via equal-width discretization of each window.
    ft_mode = extended::mode(real)[timeperiod: period, bins: period]
        -> [out: f64];
    /// Arnaud Legoux Moving Average.
    ft_alma = extended::alma(close)[timeperiod: period, offset: float, sigma: float]
        -> [out: f64] if sigma > 0.0;
    /// Zero-lag exponential moving average.
    ft_zlema = extended::zlema(close)[timeperiod: period]
        -> [out: f64];
    /// Fractal Adaptive Moving Average (Ehlers).
    ft_frama = extended::frama(close)[timeperiod: period 2]
        -> [out: f64];
    /// McGinley Dynamic.
    ft_mcginley = extended::mcginley(close)[timeperiod: period]
        -> [out: f64];
    /// Variable Index Dynamic Average (Chande).
    ft_vidya = extended::vidya(close)[timeperiod: period, cmo_period: period]
        -> [out: f64];
    /// Bill Williams Alligator (jaw / teeth / lips).
    ft_alligator = extended::alligator(high, low)[jaw_period: period, jaw_shift: count, teeth_period: period, teeth_shift: count, lips_period: period, lips_shift: count]
        -> [out_jaw: f64, out_teeth: f64, out_lips: f64];
    /// Moving-average envelopes: `MA * (1 ± percent / 100)`.
    ft_ma_envelopes = extended::ma_envelopes(close)[timeperiod: period, percent: float, matype: matype]
        -> [out_upper: f64, out_middle: f64, out_lower: f64];
    /// Chande Kroll Stop.
    ft_chande_kroll_stop = extended::chande_kroll_stop(high, low, close)[timeperiod: period, multiplier: float, stop_period: period]
        -> [out_long_stop: f64, out_short_stop: f64];
    /// Chaikin Volatility: rate of change of an EMA of the high–low range.
    ft_chaikin_vol = extended::chaikin_vol(high, low)[timeperiod: period, rocperiod: period]
        -> [out: f64];
    /// Mass Index: rolling sum of the single/double EMA ratio of the high–low range.
    ft_mass = extended::mass(high, low)[timeperiod: period, sumperiod: period]
        -> [out: f64];
    /// Bollinger %B: `(close - lower) / (upper - lower)` from [`overlap::bbands`].
    ft_bbpercent = extended::bbpercent(close)[timeperiod: period, nbdevup: float, nbdevdn: float]
        -> [out: f64];
    /// Bollinger Bandwidth: `(upper - lower) / middle` from [`overlap::bbands`].
    ft_bbwidth = extended::bbwidth(close)[timeperiod: period, nbdevup: float, nbdevdn: float]
        -> [out: f64];
    /// Close-to-close historical volatility, annualized and in percent.
    ft_historical_volatility = extended::historical_volatility(close)[timeperiod: period, annual: float]
        -> [out: f64];
    /// Ulcer Index: RMS of percent drawdowns versus the rolling highest close.
    ft_ulcer_index = extended::ulcer_index(close)[timeperiod: period]
        -> [out: f64];
    /// Stoller Average Range Channels: SMA of close ± `multiplier * ATR`.
    ft_starc = extended::starc(high, low, close)[timeperiod: period, atr_period: period, multiplier: float]
        -> [out_upper: f64, out_middle: f64, out_lower: f64];
    /// On-Balance Volume smoothed with [`overlap::ma`].
    ft_obv_smoothed = extended::obv_smoothed(close, volume)[timeperiod: period, matype: matype]
        -> [out: f64];
    /// Chaikin Money Flow: rolling sum of CLV × volume over rolling volume.
    ft_cmf = extended::cmf(high, low, close, volume)[timeperiod: period]
        -> [out: f64];
    /// Ease of Movement, then SMA.
    ft_emv = extended::emv(high, low, volume)[timeperiod: period, scale: float]
        -> [out: f64];
    /// Force Index: `(close - close[1]) * volume`, optionally EMA-smoothed.
    ft_force_index = extended::force_index(close, volume)[timeperiod: period]
        -> [out: f64];
    /// Negative Volume Index, seeded at 1000.
    ft_nvi = extended::nvi(close, volume)[]
        -> [out: f64];
    /// NVI plus an EMA signal of that series.
    ft_nvi_with_ema = extended::nvi_with_ema(close, volume)[timeperiod: period]
        -> [out_nvi: f64, out_signal: f64];
    /// Positive Volume Index, seeded at 1000.
    ft_pvi = extended::pvi(close, volume)[]
        -> [out: f64];
    /// PVI plus a moving-average signal (`matype` matches `MA`, `0`-`8`).
    ft_pvi_with_signal = extended::pvi_with_signal(close, volume)[timeperiod: period, matype: matype]
        -> [out_pvi: f64, out_signal: f64];
    /// Volume oscillator: `100 * (SMA(vol, fast) - SMA(vol, slow)) / SMA(vol, slow)`.
    ft_volosc = extended::volosc(volume)[fastperiod: period, slowperiod: period]
        -> [out: f64] if fastperiod < slowperiod;
    /// Volume rate of change: `100 * (volume - volume[timeperiod]) / volume[timeperiod]`.
    ft_vroc = extended::vroc(volume)[timeperiod: period]
        -> [out: f64];
    /// Klinger Volume Oscillator and its EMA signal.
    ft_kvo = extended::kvo(high, low, close, volume)[fastperiod: period, slowperiod: period, signalperiod: period]
        -> [out_kvo: f64, out_signal: f64] if fastperiod < slowperiod;
    /// Price Volume Trend: cumulative `volume * (close - close[1]) / close[1]`.
    ft_pvt = extended::pvt(close, volume)[]
        -> [out: f64];
    /// Relative volume: `volume / SMA(volume, timeperiod)`.
    ft_rvol = extended::rvol(volume)[timeperiod: period]
        -> [out: f64];
    /// Volume Weighted Average Price (cumulative or rolling).
    ft_vwap = extended::vwap(high, low, close, volume)[timeperiod: period]
        -> [out: f64];
    /// Volume Weighted Moving Average.
    ft_vwma = extended::vwma(close, volume)[timeperiod: period]
        -> [out: f64];
}
