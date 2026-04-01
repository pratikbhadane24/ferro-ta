//! Candlestick pattern recognition — pure Rust implementations.
//!
//! Each function takes `(open, high, low, close)` as `&[f64]` slices and returns
//! `Vec<i32>` with values -100, 0, or 100 indicating bearish, neutral, or bullish
//! pattern signals respectively.

// ---------------------------------------------------------------------------
// Shared helpers
// ---------------------------------------------------------------------------

/// Epsilon for doji-like candles (body ~ 0) to avoid division by zero.
pub const DOJI_BODY_EPSILON: f64 = 0.0001;

#[inline]
pub fn body_size(open: f64, close: f64) -> f64 {
    (close - open).abs()
}

#[inline]
pub fn upper_shadow(open: f64, high: f64, close: f64) -> f64 {
    high - open.max(close)
}

#[inline]
pub fn lower_shadow(open: f64, low: f64, close: f64) -> f64 {
    open.min(close) - low
}

#[inline]
pub fn candle_range(high: f64, low: f64) -> f64 {
    high - low
}

#[inline]
pub fn is_bullish(open: f64, close: f64) -> bool {
    close >= open
}

#[inline]
pub fn is_bearish(open: f64, close: f64) -> bool {
    close < open
}

/// Validate that all four OHLC slices have the same length. Returns `Err` with
/// a descriptive message on mismatch.
pub fn validate_ohlc(
    open: &[f64],
    high: &[f64],
    low: &[f64],
    close: &[f64],
) -> Result<usize, String> {
    let n = open.len();
    if high.len() != n || low.len() != n || close.len() != n {
        return Err(format!(
            "OHLC length mismatch: open={}, high={}, low={}, close={}",
            n,
            high.len(),
            low.len(),
            close.len()
        ));
    }
    Ok(n)
}

// ---------------------------------------------------------------------------
// 61 candlestick pattern functions
// ---------------------------------------------------------------------------

/// Two Crows (bearish)
pub fn cdl2crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, c1) = (open[i - 2], close[i - 2]);
        let (o2, c2) = (open[i - 1], close[i - 1]);
        let (o3, c3) = (open[i], close[i]);
        if is_bullish(o1, c1)
            && is_bearish(o2, c2)
            && o2 > c1
            && c2 > c1
            && is_bearish(o3, c3)
            && o3 < o2
            && o3 > c2
            && c3 > o1
            && c3 < c1
        {
            result[i] = -100;
        }
    }
    result
}

/// Three Black Crows (bearish)
pub fn cdl3blackcrows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, h2, l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);
        let range3 = candle_range(h3, l3);

        let long_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let long_body2 = range2 > 0.0 && body2 >= range2 * 0.5;
        let long_body3 = range3 > 0.0 && body3 >= range3 * 0.5;

        let open2_in_body1 = o2 < o1 && o2 > c1;
        let open3_in_body2 = o3 < o2 && o3 > c2;

        let small_upper1 = upper_shadow(o1, h1, c1) <= body1 * 0.3;
        let small_upper2 = upper_shadow(o2, h2, c2) <= body2 * 0.3;
        let small_upper3 = upper_shadow(o3, h3, c3) <= body3 * 0.3;

        if is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && is_bearish(o3, c3)
            && long_body1
            && long_body2
            && long_body3
            && open2_in_body1
            && open3_in_body2
            && small_upper1
            && small_upper2
            && small_upper3
            && c2 < c1
            && c3 < c2
            && l3 < l2
            && l2 < l1
        {
            result[i] = -100;
        }
    }
    result
}

/// Three Inside Up/Down
pub fn cdl3inside(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, _h2, _l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let c3 = close[i];

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let range1 = candle_range(h1, l1);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.5;

        let body2_high = o2.max(c2);
        let body2_low = o2.min(c2);
        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        let inside = body2_high <= body1_high && body2_low >= body1_low && body2 < body1 * 0.5;

        if is_bearish(o1, c1) && large_body1 && inside && is_bullish(o2, c2) && c3 > c2 {
            result[i] = 100;
        } else if is_bullish(o1, c1) && large_body1 && inside && is_bearish(o2, c2) && c3 < c2 {
            result[i] = -100;
        }
    }
    result
}

/// Three-Line Strike
pub fn cdl3linestrike(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 3..n {
        let (o0, c0) = (open[i - 3], close[i - 3]);
        let (o1, c1) = (open[i - 2], close[i - 2]);
        let (o2, c2) = (open[i - 1], close[i - 1]);
        let (o3, c3) = (open[i], close[i]);
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && c1 < c0
            && c2 < c1
            && is_bullish(o3, c3)
            && o3 < c2
            && c3 > o0
        {
            result[i] = 100;
        } else if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && c1 > c0
            && c2 > c1
            && is_bearish(o3, c3)
            && o3 > c2
            && c3 < o0
        {
            result[i] = -100;
        }
    }
    result
}

/// Three Outside Up/Down
pub fn cdl3outside(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, c1) = (open[i - 2], close[i - 2]);
        let (o2, c2) = (open[i - 1], close[i - 1]);
        let c3 = close[i];

        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        let body2_high = o2.max(c2);
        let body2_low = o2.min(c2);
        let engulfs = body2_high > body1_high && body2_low < body1_low;

        if is_bearish(o1, c1) && is_bullish(o2, c2) && engulfs && c3 > c2 {
            result[i] = 100;
        } else if is_bullish(o1, c1) && is_bearish(o2, c2) && engulfs && c3 < c2 {
            result[i] = -100;
        }
    }
    result
}

/// Three Stars In The South (bullish)
pub fn cdl3starsinsouth(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && h1 <= h0
            && l1 >= l0
            && h2 <= h1
            && l2 >= l1
            && body_size(o2, c2) <= body_size(o1, c1) * 0.6
            && upper_shadow(o2, h2, c2) <= body_size(o2, c2) * 0.2
        {
            result[i] = 100;
        }
    }
    result
}

/// Three Advancing White Soldiers (bullish)
pub fn cdl3whitesoldiers(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, h2, l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);
        let range3 = candle_range(h3, l3);

        let long_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let long_body2 = range2 > 0.0 && body2 >= range2 * 0.5;
        let long_body3 = range3 > 0.0 && body3 >= range3 * 0.5;

        let open2_in_body1 = o2 > o1 && o2 < c1;
        let open3_in_body2 = o3 > o2 && o3 < c2;

        let small_lower1 = lower_shadow(o1, l1, c1) <= body1 * 0.3;
        let small_lower2 = lower_shadow(o2, l2, c2) <= body2 * 0.3;
        let small_lower3 = lower_shadow(o3, l3, c3) <= body3 * 0.3;

        if is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && is_bullish(o3, c3)
            && long_body1
            && long_body2
            && long_body3
            && open2_in_body1
            && open3_in_body2
            && small_lower1
            && small_lower2
            && small_lower3
            && c2 > c1
            && c3 > c2
            && h3 > h2
            && h2 > h1
        {
            result[i] = 100;
        }
    }
    result
}

/// Abandoned Baby
pub fn cdlabandonedbaby(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let is_doji1 = range1 > 0.0 && body1 / range1 <= 0.1;
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_doji1
            && h1 < l0
            && is_bullish(o2, c2)
            && range2 > 0.0
            && body2 >= range2 * 0.5
            && l2 > h1
        {
            result[i] = 100;
        } else if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_doji1
            && l1 > h0
            && is_bearish(o2, c2)
            && range2 > 0.0
            && body2 >= range2 * 0.5
            && h2 < l1
        {
            result[i] = -100;
        }
    }
    result
}

/// Advance Block (bearish)
pub fn cdladvanceblock(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, _l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, h1, _l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let us0 = upper_shadow(o0, h0, c0);
        let us1 = upper_shadow(o1, h1, c1);
        let us2 = upper_shadow(o2, h2, c2);
        if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && c1 > c0
            && c2 > c1
            && o1 >= o0
            && o1 <= c0
            && o2 >= o1
            && o2 <= c1
            && (body1 < body0 || body2 < body1 || us2 > us1 || us1 > us0)
        {
            result[i] = -100;
        }
    }
    result
}

/// Belt-hold
pub fn cdlbelthold(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        let body = body_size(o, c);
        if range == 0.0 {
            continue;
        }
        let long_body = body >= range * 0.6;
        if is_bullish(o, c) && long_body && (o - l).abs() <= range * 0.01 {
            result[i] = 100;
        } else if is_bearish(o, c) && long_body && (h - o).abs() <= range * 0.01 {
            result[i] = -100;
        }
    }
    result
}

/// Breakaway
pub fn cdlbreakaway(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 4..n {
        let (o0, h0, l0, c0) = (open[i - 4], high[i - 4], low[i - 4], close[i - 4]);
        let c1 = close[i - 3];
        let c2 = close[i - 2];
        let c3 = close[i - 1];
        let (o4, c4) = (open[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && c1 < l0
            && c2 < c1
            && c3 < c2
            && is_bullish(o4, c4)
            && c4 > c1
            && c4 < c0
        {
            result[i] = 100;
        } else if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && c1 > h0
            && c2 > c1
            && c3 > c2
            && is_bearish(o4, c4)
            && c4 < c1
            && c4 > c0
        {
            result[i] = -100;
        }
    }
    result
}

/// Closing Marubozu
pub fn cdlclosingmarubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        if body < range * 0.4 {
            continue;
        }
        if is_bullish(o, c) && (h - c).abs() <= range * 0.01 {
            result[i] = 100;
        } else if is_bearish(o, c) && (c - l).abs() <= range * 0.01 {
            result[i] = -100;
        }
    }
    result
}

/// Concealing Baby Swallow (bullish)
pub fn cdlconcealbabyswall(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 3..n {
        let (o0, h0, l0, c0) = (open[i - 3], high[i - 3], low[i - 3], close[i - 3]);
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, h2, l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let maru0 = range0 > 0.0
            && upper_shadow(o0, h0, c0) <= range0 * 0.02
            && lower_shadow(o0, l0, c0) <= range0 * 0.02;
        let maru1 = range1 > 0.0
            && upper_shadow(o1, h1, c1) <= range1 * 0.02
            && lower_shadow(o1, l1, c1) <= range1 * 0.02;
        let gap_down = o2 < c1;
        let shadow_into = h2 >= c1;
        let engulfs = o3 >= o2 && c3 <= c2 && h3 >= h2 && l3 <= l2;
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && maru0
            && maru1
            && is_bearish(o2, c2)
            && gap_down
            && shadow_into
            && is_bearish(o3, c3)
            && engulfs
        {
            result[i] = 100;
        }
    }
    result
}

/// Counterattack
pub fn cdlcounterattack(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, h1, l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let long0 = range0 > 0.0 && body0 >= range0 * 0.5;
        let long1 = range1 > 0.0 && body1 >= range1 * 0.5;
        let same_close = (c1 - c0).abs() <= range0 * 0.02;
        if is_bearish(o0, c0) && long0 && is_bullish(o1, c1) && long1 && same_close {
            result[i] = 100;
        } else if is_bullish(o0, c0) && long0 && is_bearish(o1, c1) && long1 && same_close {
            result[i] = -100;
        }
    }
    result
}

/// Dark Cloud Cover (bearish)
pub fn cdldarkcloudcover(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let body0 = body_size(o0, c0);
        let range0 = candle_range(h0, l0);
        let midpoint0 = (o0 + c0) / 2.0;
        if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_bearish(o1, c1)
            && o1 > h0
            && c1 < midpoint0
            && c1 > o0
        {
            result[i] = -100;
        }
    }
    result
}

/// Doji
pub fn cdldoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let body = body_size(open[i], close[i]);
        let range = candle_range(high[i], low[i]);
        if range > 0.0 && body / range <= 0.1 {
            result[i] = 100;
        }
    }
    result
}

/// Doji Star
pub fn cdldojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let is_doji2 = range2 > 0.0 && body2 / range2 <= 0.1;

        let gap_down = o2.max(c2) < l1;
        if is_bearish(o1, c1) && large_body1 && is_doji2 && gap_down {
            result[i] = 100;
        }
        let gap_up = o2.min(c2) > h1;
        if is_bullish(o1, c1) && large_body1 && is_doji2 && gap_up {
            result[i] = -100;
        }
    }
    result
}

/// Dragonfly Doji (bullish)
pub fn cdldragonflydoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if body / range <= 0.1 && us / range <= 0.1 && ls >= range * 0.6 {
            result[i] = 100;
        }
    }
    result
}

/// Engulfing
pub fn cdlengulfing(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let prev_o = open[i - 1];
        let prev_c = close[i - 1];
        let curr_o = open[i];
        let curr_c = close[i];

        let prev_body_high = prev_o.max(prev_c);
        let prev_body_low = prev_o.min(prev_c);
        let curr_body_high = curr_o.max(curr_c);
        let curr_body_low = curr_o.min(curr_c);

        if is_bearish(prev_o, prev_c)
            && is_bullish(curr_o, curr_c)
            && curr_body_high > prev_body_high
            && curr_body_low < prev_body_low
        {
            result[i] = 100;
        } else if is_bullish(prev_o, prev_c)
            && is_bearish(curr_o, curr_c)
            && curr_body_high > prev_body_high
            && curr_body_low < prev_body_low
        {
            result[i] = -100;
        }
    }
    result
}

/// Evening Doji Star (bearish)
pub fn cdleveningdojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, _h2, _l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(o2.min(c2) - DOJI_BODY_EPSILON, o2.max(c2));
        let range3 = candle_range(h3, l3);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let is_doji2 = range2 > 0.0 && body2 / range2 <= 0.1;
        let large_body3 = range3 > 0.0 && body3 >= range3 * 0.6;

        if is_bullish(o1, c1)
            && large_body1
            && is_doji2
            && is_bearish(o3, c3)
            && large_body3
            && c3 < (o1 + c1) / 2.0
        {
            result[i] = -100;
        }
    }
    result
}

/// Evening Star (bearish)
pub fn cdleveningstar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, _h2, _l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range3 = candle_range(h3, l3);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let small_body2 = range1 > 0.0 && body2 < body1 * 0.3;
        let large_body3 = range3 > 0.0 && body3 >= range3 * 0.6;

        if is_bullish(o1, c1)
            && large_body1
            && small_body2
            && is_bearish(o3, c3)
            && large_body3
            && c3 < (o1 + c1) / 2.0
        {
            result[i] = -100;
        }
    }
    result
}

/// Up/Down-gap side-by-side white lines
pub fn cdlgapsidesidewhite(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, _h0, _l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, _h1, _l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let both_bullish = is_bullish(o1, c1) && is_bullish(o2, c2);
        let similar_size = body1 > 0.0 && (body2 - body1).abs() / body1 <= 0.3;
        let similar_open = body1 > 0.0 && (o2 - o1).abs() / body1 <= 0.3;
        if is_bullish(o0, c0) && both_bullish && similar_size && similar_open && o1 > c0 {
            result[i] = 100;
        } else if is_bearish(o0, c0) && both_bullish && similar_size && similar_open && c1 < o0 {
            result[i] = -100;
        }
    }
    result
}

/// Gravestone Doji (bearish)
pub fn cdlgravestonedoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if body / range <= 0.1 && ls / range <= 0.1 && us >= range * 0.6 {
            result[i] = -100;
        }
    }
    result
}

/// Hammer (bullish)
pub fn cdlhammer(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let body = body_size(open[i], close[i]);
        let range = candle_range(high[i], low[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        if range > 0.0 && body > 0.0 && body <= range / 3.0 && lower >= 2.0 * body && upper <= body
        {
            result[i] = 100;
        }
    }
    result
}

/// Hanging Man (bearish)
pub fn cdlhangingman(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if range > 0.0 && body > 0.0 && ls >= body * 2.0 && us <= body && body / range <= 0.4 {
            result[i] = -100;
        }
    }
    result
}

/// Harami
pub fn cdlharami(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let range1 = candle_range(h1, l1);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.5;

        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        let body2_high = o2.max(c2);
        let body2_low = o2.min(c2);

        let inside = body2_high <= body1_high && body2_low >= body1_low && body2 < body1 * 0.6;

        if is_bearish(o1, c1) && large_body1 && inside && is_bullish(o2, c2) {
            result[i] = 100;
        } else if is_bullish(o1, c1) && large_body1 && inside && is_bearish(o2, c2) {
            result[i] = -100;
        }
    }
    result
}

/// Harami Cross
pub fn cdlharamicross(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.5;
        let is_doji2 = range2 > 0.0 && body2 / range2 <= 0.1;

        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        let doji_mid = (o2 + c2) / 2.0;
        let inside = doji_mid <= body1_high && doji_mid >= body1_low;

        if is_bearish(o1, c1) && large_body1 && is_doji2 && inside {
            result[i] = 100;
        } else if is_bullish(o1, c1) && large_body1 && is_doji2 && inside {
            result[i] = -100;
        }
    }
    result
}

/// High-Wave Candle
pub fn cdlhighwave(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if body / range <= 0.3 && us >= range * 0.3 && ls >= range * 0.3 {
            if is_bullish(o, c) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    result
}

/// Hikkake Pattern
pub fn cdlhikkake(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let h1 = high[i - 1];
        let l1 = low[i - 1];
        let h2 = high[i];
        let l2 = low[i];
        let inside = h1 <= h0 && l1 >= l0;
        if !inside {
            continue;
        }
        if is_bearish(o0, c0) && h2 > h1 && l2 > l1 {
            result[i] = 100;
        } else if is_bullish(o0, c0) && l2 < l1 && h2 < h1 {
            result[i] = -100;
        }
    }
    result
}

/// Modified Hikkake Pattern
pub fn cdlhikkakemod(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 3..n {
        let (o0, h0, l0, c0) = (open[i - 3], high[i - 3], low[i - 3], close[i - 3]);
        let h1 = high[i - 2];
        let l1 = low[i - 2];
        let h2 = high[i - 1];
        let l2 = low[i - 1];
        let h3 = high[i];
        let l3 = low[i];
        let inside = h1 <= h0 && l1 >= l0;
        if !inside {
            continue;
        }
        if is_bearish(o0, c0) && l2 < l1 && h3 > h1 && l3 > l1 {
            result[i] = 100;
        } else if is_bullish(o0, c0) && h2 > h1 && l3 < l1 && h3 < h1 {
            result[i] = -100;
        }
    }
    result
}

/// Homing Pigeon (bullish)
pub fn cdlhomingpigeon(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, _h0, _l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let body0_high = o0.max(c0);
        let body0_low = o0.min(c0);
        let body1_high = o1.max(c1);
        let body1_low = o1.min(c1);
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && body1_high <= body0_high
            && body1_low >= body0_low
        {
            result[i] = 100;
        }
    }
    result
}

/// Identical Three Crows (bearish)
pub fn cdlidentical3crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let tol0 = range0 * 0.03;
        let tol1 = range1 * 0.03;
        if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && c1 < c0
            && c2 < c1
            && (o1 - c0).abs() <= tol0
            && (o2 - c1).abs() <= tol1
            && range0 > 0.0
            && range1 > 0.0
            && candle_range(h2, l2) > 0.0
        {
            result[i] = -100;
        }
    }
    result
}

/// In-Neck Pattern (bearish)
pub fn cdlinneck(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bullish(o1, c1)
            && o1 < l0
            && (c1 - c0).abs() <= range0 * 0.03
        {
            result[i] = -100;
        }
    }
    result
}

/// Inverted Hammer (bullish)
pub fn cdlinvertedhammer(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if body > 0.0 && us >= body * 2.0 && ls <= body && body / range <= 0.4 {
            result[i] = 100;
        }
    }
    result
}

/// Kicking
pub fn cdlkicking(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, h1, l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let maru0 = range0 > 0.0
            && upper_shadow(o0, h0, c0) <= range0 * 0.02
            && lower_shadow(o0, l0, c0) <= range0 * 0.02;
        let maru1 = range1 > 0.0
            && upper_shadow(o1, h1, c1) <= range1 * 0.02
            && lower_shadow(o1, l1, c1) <= range1 * 0.02;
        if is_bearish(o0, c0) && maru0 && is_bullish(o1, c1) && maru1 && o1 > o0 {
            result[i] = 100;
        } else if is_bullish(o0, c0) && maru0 && is_bearish(o1, c1) && maru1 && o1 < o0 {
            result[i] = -100;
        }
    }
    result
}

/// Kicking — bull/bear determined by longer of the two marubozu
pub fn cdlkickingbylength(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, h1, l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let maru0 = range0 > 0.0
            && upper_shadow(o0, h0, c0) <= range0 * 0.02
            && lower_shadow(o0, l0, c0) <= range0 * 0.02;
        let maru1 = range1 > 0.0
            && upper_shadow(o1, h1, c1) <= range1 * 0.02
            && lower_shadow(o1, l1, c1) <= range1 * 0.02;
        let opposite = (is_bearish(o0, c0) && is_bullish(o1, c1))
            || (is_bullish(o0, c0) && is_bearish(o1, c1));
        let has_gap = (o1 - c0).abs() > 0.0;
        if maru0 && maru1 && opposite && has_gap {
            if range1 >= range0 {
                if is_bullish(o1, c1) {
                    result[i] = 100;
                } else {
                    result[i] = -100;
                }
            } else if is_bullish(o0, c0) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    result
}

/// Ladder Bottom (bullish)
pub fn cdlladderbottom(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 4..n {
        let (o0, _h0, _l0, c0) = (open[i - 4], high[i - 4], low[i - 4], close[i - 4]);
        let (o1, _h1, _l1, c1) = (open[i - 3], high[i - 3], low[i - 3], close[i - 3]);
        let (o2, _h2, _l2, c2) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o3, h3, _l3, c3) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o4, h4, l4, c4) = (open[i], high[i], low[i], close[i]);
        let three_bear = is_bearish(o0, c0) && is_bearish(o1, c1) && is_bearish(o2, c2);
        let descend = c1 < c0 && c2 < c1;
        let us3 = upper_shadow(o3, h3, c3);
        let body3 = body_size(o3, c3);
        let inv_hammer = us3 >= body3 * 1.5;
        let range4 = candle_range(h4, l4);
        let body4 = body_size(o4, c4);
        let large_bull = is_bullish(o4, c4) && range4 > 0.0 && body4 >= range4 * 0.5;
        if three_bear && descend && inv_hammer && large_bull && c4 > c2 {
            result[i] = 100;
        }
    }
    result
}

/// Long Legged Doji
pub fn cdllongleggeddoji(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        if body / range <= 0.1 && us >= range * 0.3 && ls >= range * 0.3 {
            result[i] = 100;
        }
    }
    result
}

/// Long Line Candle
pub fn cdllongline(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        if body >= range * 0.7 {
            if is_bullish(o, c) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    result
}

/// Marubozu
pub fn cdlmarubozu(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let body = body_size(open[i], close[i]);
        let range = candle_range(high[i], low[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        if range > 0.0 && body >= range * 0.95 && upper <= range * 0.025 && lower <= range * 0.025 {
            if is_bullish(open[i], close[i]) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    result
}

/// Matching Low (bullish)
pub fn cdlmatchinglow(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let tol = range0 * 0.02;
        if is_bearish(o0, c0) && is_bearish(o1, c1) && (c1 - c0).abs() <= tol {
            result[i] = 100;
        }
    }
    result
}

/// Mat Hold (bullish)
pub fn cdlmathold(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 4..n {
        let (o0, h0, l0, c0) = (open[i - 4], high[i - 4], low[i - 4], close[i - 4]);
        let (o1, _h1, l1, c1) = (open[i - 3], high[i - 3], low[i - 3], close[i - 3]);
        let (o2, _h2, l2, c2) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o3, _h3, l3, c3) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o4, h4, l4, c4) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        let range4 = candle_range(h4, l4);
        let body4 = body_size(o4, c4);
        let large_bull0 = is_bullish(o0, c0) && range0 > 0.0 && body0 >= range0 * 0.5;
        let small_bears = is_bearish(o1, c1) && is_bearish(o2, c2) && is_bearish(o3, c3);
        let stay_above = l1 >= o0 && l2 >= o0 && l3 >= o0;
        let large_bull4 = is_bullish(o4, c4) && range4 > 0.0 && body4 >= range4 * 0.5 && c4 > c0;
        if large_bull0 && small_bears && stay_above && large_bull4 {
            result[i] = 100;
        }
    }
    result
}

/// Morning Doji Star (bullish)
pub fn cdlmorningdojistar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, _h2, _l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(o2.min(c2) - DOJI_BODY_EPSILON, o2.max(c2));
        let range3 = candle_range(h3, l3);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let is_doji2 = range2 > 0.0 && body2 / range2 <= 0.1;
        let large_body3 = range3 > 0.0 && body3 >= range3 * 0.6;

        if is_bearish(o1, c1)
            && large_body1
            && is_doji2
            && is_bullish(o3, c3)
            && large_body3
            && c3 > (o1 + c1) / 2.0
        {
            result[i] = 100;
        }
    }
    result
}

/// Morning Star (bullish)
pub fn cdlmorningstar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o1, h1, l1, c1) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o2, _h2, _l2, c2) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o3, h3, l3, c3) = (open[i], high[i], low[i], close[i]);

        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let body3 = body_size(o3, c3);
        let range1 = candle_range(h1, l1);
        let range3 = candle_range(h3, l3);

        let large_body1 = range1 > 0.0 && body1 >= range1 * 0.6;
        let small_body2 = range1 > 0.0 && body2 < body1 * 0.3;
        let large_body3 = range3 > 0.0 && body3 >= range3 * 0.6;

        if is_bearish(o1, c1)
            && large_body1
            && small_body2
            && is_bullish(o3, c3)
            && large_body3
            && c3 > (o1 + c1) / 2.0
        {
            result[i] = 100;
        }
    }
    result
}

/// On-Neck Pattern (bearish)
pub fn cdlonneck(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bullish(o1, c1)
            && o1 < l0
            && (c1 - l0).abs() <= range0 * 0.03
        {
            result[i] = -100;
        }
    }
    result
}

/// Piercing Pattern (bullish)
pub fn cdlpiercing(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let body0 = body_size(o0, c0);
        let range0 = candle_range(h0, l0);
        let midpoint0 = (o0 + c0) / 2.0;
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bullish(o1, c1)
            && o1 < l0
            && c1 > midpoint0
            && c1 < o0
        {
            result[i] = 100;
        }
    }
    result
}

/// Rickshaw Man
pub fn cdlrickshawman(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        let us = upper_shadow(o, h, c);
        let ls = lower_shadow(o, l, c);
        let body_mid = (o + c) / 2.0;
        let range_mid = (h + l) / 2.0;
        let is_doji = body / range <= 0.1;
        let long_shadows = us >= range * 0.3 && ls >= range * 0.3;
        let near_center = (body_mid - range_mid).abs() <= range * 0.15;
        if is_doji && long_shadows && near_center {
            result[i] = 100;
        }
    }
    result
}

/// Rising/Falling Three Methods
pub fn cdlrisefall3methods(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 4..n {
        let (o0, h0, l0, c0) = (open[i - 4], high[i - 4], low[i - 4], close[i - 4]);
        let (o1, h1, l1, c1) = (open[i - 3], high[i - 3], low[i - 3], close[i - 3]);
        let (o2, h2, l2, c2) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o3, h3, l3, c3) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o4, h4, l4, c4) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        let range4 = candle_range(h4, l4);
        let body4 = body_size(o4, c4);
        if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_bearish(o1, c1)
            && is_bearish(o2, c2)
            && is_bearish(o3, c3)
            && h1 <= h0
            && l1 >= l0
            && h2 <= h0
            && l2 >= l0
            && h3 <= h0
            && l3 >= l0
            && is_bullish(o4, c4)
            && range4 > 0.0
            && body4 >= range4 * 0.5
            && c4 > c0
            && o4 > c3
        {
            result[i] = 100;
        } else if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.5
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && is_bullish(o3, c3)
            && h1 <= h0
            && l1 >= l0
            && h2 <= h0
            && l2 >= l0
            && h3 <= h0
            && l3 >= l0
            && is_bearish(o4, c4)
            && range4 > 0.0
            && body4 >= range4 * 0.5
            && c4 < c0
            && o4 < c3
        {
            result[i] = -100;
        }
    }
    result
}

/// Separating Lines
pub fn cdlseparatinglines(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, h1, l1, c1) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body1 = body_size(o1, c1);
        let range1 = candle_range(h1, l1);
        let same_open = range0 > 0.0 && (o1 - o0).abs() <= range0 * 0.02;
        let long1 = range1 > 0.0 && body1 >= range1 * 0.5;
        if is_bearish(o0, c0) && is_bullish(o1, c1) && same_open && long1 {
            result[i] = 100;
        } else if is_bullish(o0, c0) && is_bearish(o1, c1) && same_open && long1 {
            result[i] = -100;
        }
    }
    result
}

/// Shooting Star (bearish)
pub fn cdlshootingstar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let body = body_size(open[i], close[i]);
        let range = candle_range(high[i], low[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        if range > 0.0 && body > 0.0 && body <= range / 3.0 && upper >= 2.0 * body && lower <= body
        {
            result[i] = -100;
        }
    }
    result
}

/// Short Line Candle
pub fn cdlshortline(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c);
        if body > 0.0 && body <= range * 0.3 {
            if is_bullish(o, c) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    result
}

/// Spinning Top
pub fn cdlspinningtop(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let body = body_size(open[i], close[i]);
        let range = candle_range(high[i], low[i]);
        let lower = lower_shadow(open[i], low[i], close[i]);
        let upper = upper_shadow(open[i], high[i], close[i]);
        if range > 0.0 && body > 0.0 && body <= range / 3.0 && upper > body && lower > body {
            if is_bullish(open[i], close[i]) {
                result[i] = 100;
            } else {
                result[i] = -100;
            }
        }
    }
    result
}

/// Stalled Pattern (bearish)
pub fn cdlstalledpattern(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, _h1, _l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && is_bullish(o2, c2)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && c1 > c0
            && c2 > c1
            && o1 >= o0
            && o1 <= c0
            && o2 >= c1 * 0.99
            && body2 < body1 * 0.7
        {
            result[i] = -100;
        }
    }
    result
}

/// Stick Sandwich (bullish)
pub fn cdlsticksandwich(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, _h1, _l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let tol = range0 * 0.02;
        if is_bearish(o0, c0)
            && is_bullish(o1, c1)
            && is_bearish(o2, c2)
            && (c2 - c0).abs() <= tol
            && o1 >= c0
            && c1 <= o0
        {
            result[i] = 100;
        }
    }
    result
}

/// Takuri (Dragonfly Doji with very long lower shadow)
pub fn cdltakuri(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 0..n {
        let (o, h, l, c) = (open[i], high[i], low[i], close[i]);
        let range = candle_range(h, l);
        if range == 0.0 {
            continue;
        }
        let body = body_size(o, c) + DOJI_BODY_EPSILON;
        let ls = lower_shadow(o, l, c);
        let us = upper_shadow(o, h, c);
        if ls >= body * 3.0 && us <= range * 0.1 {
            result[i] = 100;
        }
    }
    result
}

/// Tasuki Gap
pub fn cdltasukigap(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, _h0, _l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && o1 > c0
            && is_bearish(o2, c2)
            && o2 >= l1
            && o2 <= c1
            && c2 > c0
            && c2 < o1
        {
            result[i] = 100;
        } else if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && o1 < c0
            && is_bullish(o2, c2)
            && o2 >= c1
            && o2 <= h1
            && c2 < c0
            && c2 > o1
        {
            result[i] = -100;
        }
    }
    result
}

/// Thrusting Pattern (bearish)
pub fn cdlthrusting(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 1..n {
        let (o0, h0, l0, c0) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o1, _h1, _l1, c1) = (open[i], high[i], low[i], close[i]);
        let body0 = body_size(o0, c0);
        let range0 = candle_range(h0, l0);
        let midpoint0 = (o0 + c0) / 2.0;
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bullish(o1, c1)
            && o1 < l0
            && c1 > c0
            && c1 < midpoint0
        {
            result[i] = -100;
        }
    }
    result
}

/// Tristar Pattern
pub fn cdltristar(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let range1 = candle_range(h1, l1);
        let range2 = candle_range(h2, l2);
        let body0 = body_size(o0, c0);
        let body1 = body_size(o1, c1);
        let body2 = body_size(o2, c2);
        let doji0 = range0 > 0.0 && body0 / range0 <= 0.1;
        let doji1 = range1 > 0.0 && body1 / range1 <= 0.1;
        let doji2 = range2 > 0.0 && body2 / range2 <= 0.1;
        if doji0 && doji1 && doji2 {
            if l1 < l0 && h1 < h0 && c2 > c1 {
                result[i] = 100;
            } else if l1 > l0 && h1 > h0 && c2 < c1 {
                result[i] = -100;
            }
        }
    }
    result
}

/// Unique 3 River (bullish)
pub fn cdlunique3river(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, _h1, l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, h2, l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        let body2 = body_size(o2, c2);
        let range2 = candle_range(h2, l2);
        if is_bearish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bearish(o1, c1)
            && l1 < l0
            && lower_shadow(o1, l1, c1) > 0.0
            && is_bullish(o2, c2)
            && range2 > 0.0
            && body2 <= range2 * 0.5
            && c2 < c1
            && c2 > l1
        {
            result[i] = 100;
        }
    }
    result
}

/// Upside Gap Two Crows (bearish)
pub fn cdlupsidegap2crows(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, h0, l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, _h1, _l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        let range0 = candle_range(h0, l0);
        let body0 = body_size(o0, c0);
        if is_bullish(o0, c0)
            && range0 > 0.0
            && body0 >= range0 * 0.4
            && is_bearish(o1, c1)
            && o1 > c0
            && is_bearish(o2, c2)
            && o2 > o1
            && c2 < o1
            && c2 > c0
        {
            result[i] = -100;
        }
    }
    result
}

/// Upside/Downside Gap Three Methods
pub fn cdlxsidegap3methods(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<i32> {
    let n = validate_ohlc(open, high, low, close).expect("OHLC length mismatch");
    let mut result = vec![0i32; n];
    for i in 2..n {
        let (o0, _h0, _l0, c0) = (open[i - 2], high[i - 2], low[i - 2], close[i - 2]);
        let (o1, _h1, _l1, c1) = (open[i - 1], high[i - 1], low[i - 1], close[i - 1]);
        let (o2, _h2, _l2, c2) = (open[i], high[i], low[i], close[i]);
        if is_bullish(o0, c0)
            && is_bullish(o1, c1)
            && o1 > c0
            && is_bearish(o2, c2)
            && o2 <= c1
            && o2 >= o1
            && c2 >= c0
            && c2 <= o1
        {
            result[i] = 100;
        } else if is_bearish(o0, c0)
            && is_bearish(o1, c1)
            && o1 < c0
            && is_bullish(o2, c2)
            && o2 >= c1
            && o2 <= o1
            && c2 <= c0
            && c2 >= o1
        {
            result[i] = -100;
        }
    }
    result
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_doji_basic() {
        let open = vec![10.0];
        let high = vec![11.0];
        let low = vec![9.0];
        let close = vec![10.0];
        let r = cdldoji(&open, &high, &low, &close);
        assert_eq!(r, vec![100]);
    }

    #[test]
    fn test_doji_not_detected() {
        let open = vec![9.0];
        let high = vec![11.0];
        let low = vec![9.0];
        let close = vec![11.0];
        let r = cdldoji(&open, &high, &low, &close);
        assert_eq!(r, vec![0]);
    }

    #[test]
    fn test_engulfing_bullish() {
        // prev: bearish (open=10, close=8), body_high=10, body_low=8
        // curr: bullish (open=7.5, close=11), body_high=11, body_low=7.5
        // curr engulfs prev: 11>10 && 7.5<8
        let open = vec![10.0, 7.5];
        let high = vec![10.5, 11.5];
        let low = vec![7.5, 7.0];
        let close = vec![8.0, 11.0];
        let r = cdlengulfing(&open, &high, &low, &close);
        assert_eq!(r[1], 100);
    }

    #[test]
    fn test_engulfing_bearish() {
        let open = vec![8.0, 11.5];
        let high = vec![11.0, 12.0];
        let low = vec![7.5, 7.0];
        let close = vec![11.0, 7.5];
        let r = cdlengulfing(&open, &high, &low, &close);
        assert_eq!(r[1], -100);
    }

    #[test]
    fn test_hammer() {
        let open = vec![10.0];
        let high = vec![10.2];
        let low = vec![7.0];
        let close = vec![10.1];
        let r = cdlhammer(&open, &high, &low, &close);
        assert_eq!(r[0], 100);
    }

    #[test]
    fn test_marubozu_bullish() {
        let open = vec![10.0];
        let high = vec![12.0];
        let low = vec![10.0];
        let close = vec![12.0];
        let r = cdlmarubozu(&open, &high, &low, &close);
        assert_eq!(r[0], 100);
    }

    #[test]
    fn test_empty_input() {
        let empty: Vec<f64> = vec![];
        let r = cdldoji(&empty, &empty, &empty, &empty);
        assert!(r.is_empty());
    }

    #[test]
    fn test_morning_star() {
        let open = vec![20.0, 14.5, 15.0];
        let high = vec![20.5, 15.0, 19.5];
        let low = vec![14.0, 14.0, 14.5];
        let close = vec![14.5, 14.6, 19.0];
        let r = cdlmorningstar(&open, &high, &low, &close);
        assert_eq!(r[2], 100);
    }

    #[test]
    fn test_validate_ohlc_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(validate_ohlc(&a, &b, &a, &a).is_err());
    }
}
