//! C exports for `ferro_ta_core::pattern`.

crate::ffi_exports! {
    group: "pattern",
    /// Two Crows (bearish)
    ft_cdl2crows = pattern::cdl2crows(open, high, low, close)[]
        -> [out: i32];
    /// Three Black Crows (bearish)
    ft_cdl3blackcrows = pattern::cdl3blackcrows(open, high, low, close)[]
        -> [out: i32];
    /// Three Inside Up/Down
    ft_cdl3inside = pattern::cdl3inside(open, high, low, close)[]
        -> [out: i32];
    /// Three-Line Strike
    ft_cdl3linestrike = pattern::cdl3linestrike(open, high, low, close)[]
        -> [out: i32];
    /// Three Outside Up/Down
    ft_cdl3outside = pattern::cdl3outside(open, high, low, close)[]
        -> [out: i32];
    /// Three Stars In The South (bullish)
    ft_cdl3starsinsouth = pattern::cdl3starsinsouth(open, high, low, close)[]
        -> [out: i32];
    /// Three Advancing White Soldiers (bullish)
    ft_cdl3whitesoldiers = pattern::cdl3whitesoldiers(open, high, low, close)[]
        -> [out: i32];
    /// Abandoned Baby
    ft_cdlabandonedbaby = pattern::cdlabandonedbaby(open, high, low, close)[]
        -> [out: i32];
    /// Advance Block (bearish)
    ft_cdladvanceblock = pattern::cdladvanceblock(open, high, low, close)[]
        -> [out: i32];
    /// Belt-hold
    ft_cdlbelthold = pattern::cdlbelthold(open, high, low, close)[]
        -> [out: i32];
    /// Breakaway
    ft_cdlbreakaway = pattern::cdlbreakaway(open, high, low, close)[]
        -> [out: i32];
    /// Closing Marubozu
    ft_cdlclosingmarubozu = pattern::cdlclosingmarubozu(open, high, low, close)[]
        -> [out: i32];
    /// Concealing Baby Swallow (bullish)
    ft_cdlconcealbabyswall = pattern::cdlconcealbabyswall(open, high, low, close)[]
        -> [out: i32];
    /// Counterattack
    ft_cdlcounterattack = pattern::cdlcounterattack(open, high, low, close)[]
        -> [out: i32];
    /// Dark Cloud Cover (bearish)
    ft_cdldarkcloudcover = pattern::cdldarkcloudcover(open, high, low, close)[]
        -> [out: i32];
    /// Doji
    ft_cdldoji = pattern::cdldoji(open, high, low, close)[]
        -> [out: i32];
    /// Doji Star
    ft_cdldojistar = pattern::cdldojistar(open, high, low, close)[]
        -> [out: i32];
    /// Dragonfly Doji (bullish)
    ft_cdldragonflydoji = pattern::cdldragonflydoji(open, high, low, close)[]
        -> [out: i32];
    /// Engulfing
    ft_cdlengulfing = pattern::cdlengulfing(open, high, low, close)[]
        -> [out: i32];
    /// Evening Doji Star (bearish)
    ft_cdleveningdojistar = pattern::cdleveningdojistar(open, high, low, close)[]
        -> [out: i32];
    /// Evening Star (bearish)
    ft_cdleveningstar = pattern::cdleveningstar(open, high, low, close)[]
        -> [out: i32];
    /// Up/Down-gap side-by-side white lines
    ft_cdlgapsidesidewhite = pattern::cdlgapsidesidewhite(open, high, low, close)[]
        -> [out: i32];
    /// Gravestone Doji (bearish)
    ft_cdlgravestonedoji = pattern::cdlgravestonedoji(open, high, low, close)[]
        -> [out: i32];
    /// Hammer (bullish)
    ft_cdlhammer = pattern::cdlhammer(open, high, low, close)[]
        -> [out: i32];
    /// Hanging Man (bearish)
    ft_cdlhangingman = pattern::cdlhangingman(open, high, low, close)[]
        -> [out: i32];
    /// Harami
    ft_cdlharami = pattern::cdlharami(open, high, low, close)[]
        -> [out: i32];
    /// Harami Cross
    ft_cdlharamicross = pattern::cdlharamicross(open, high, low, close)[]
        -> [out: i32];
    /// High-Wave Candle
    ft_cdlhighwave = pattern::cdlhighwave(open, high, low, close)[]
        -> [out: i32];
    /// Hikkake Pattern
    ft_cdlhikkake = pattern::cdlhikkake(open, high, low, close)[]
        -> [out: i32];
    /// Modified Hikkake Pattern
    ft_cdlhikkakemod = pattern::cdlhikkakemod(open, high, low, close)[]
        -> [out: i32];
    /// Homing Pigeon (bullish)
    ft_cdlhomingpigeon = pattern::cdlhomingpigeon(open, high, low, close)[]
        -> [out: i32];
    /// Identical Three Crows (bearish)
    ft_cdlidentical3crows = pattern::cdlidentical3crows(open, high, low, close)[]
        -> [out: i32];
    /// In-Neck Pattern (bearish)
    ft_cdlinneck = pattern::cdlinneck(open, high, low, close)[]
        -> [out: i32];
    /// Inverted Hammer (bullish)
    ft_cdlinvertedhammer = pattern::cdlinvertedhammer(open, high, low, close)[]
        -> [out: i32];
    /// Kicking
    ft_cdlkicking = pattern::cdlkicking(open, high, low, close)[]
        -> [out: i32];
    /// Kicking — bull/bear determined by longer of the two marubozu
    ft_cdlkickingbylength = pattern::cdlkickingbylength(open, high, low, close)[]
        -> [out: i32];
    /// Ladder Bottom (bullish)
    ft_cdlladderbottom = pattern::cdlladderbottom(open, high, low, close)[]
        -> [out: i32];
    /// Long Legged Doji
    ft_cdllongleggeddoji = pattern::cdllongleggeddoji(open, high, low, close)[]
        -> [out: i32];
    /// Long Line Candle
    ft_cdllongline = pattern::cdllongline(open, high, low, close)[]
        -> [out: i32];
    /// Marubozu
    ft_cdlmarubozu = pattern::cdlmarubozu(open, high, low, close)[]
        -> [out: i32];
    /// Matching Low (bullish)
    ft_cdlmatchinglow = pattern::cdlmatchinglow(open, high, low, close)[]
        -> [out: i32];
    /// Mat Hold (bullish)
    ft_cdlmathold = pattern::cdlmathold(open, high, low, close)[]
        -> [out: i32];
    /// Morning Doji Star (bullish)
    ft_cdlmorningdojistar = pattern::cdlmorningdojistar(open, high, low, close)[]
        -> [out: i32];
    /// Morning Star (bullish)
    ft_cdlmorningstar = pattern::cdlmorningstar(open, high, low, close)[]
        -> [out: i32];
    /// On-Neck Pattern (bearish)
    ft_cdlonneck = pattern::cdlonneck(open, high, low, close)[]
        -> [out: i32];
    /// Piercing Pattern (bullish)
    ft_cdlpiercing = pattern::cdlpiercing(open, high, low, close)[]
        -> [out: i32];
    /// Rickshaw Man
    ft_cdlrickshawman = pattern::cdlrickshawman(open, high, low, close)[]
        -> [out: i32];
    /// Rising/Falling Three Methods
    ft_cdlrisefall3methods = pattern::cdlrisefall3methods(open, high, low, close)[]
        -> [out: i32];
    /// Separating Lines
    ft_cdlseparatinglines = pattern::cdlseparatinglines(open, high, low, close)[]
        -> [out: i32];
    /// Shooting Star (bearish)
    ft_cdlshootingstar = pattern::cdlshootingstar(open, high, low, close)[]
        -> [out: i32];
    /// Short Line Candle
    ft_cdlshortline = pattern::cdlshortline(open, high, low, close)[]
        -> [out: i32];
    /// Spinning Top
    ft_cdlspinningtop = pattern::cdlspinningtop(open, high, low, close)[]
        -> [out: i32];
    /// Stalled Pattern (bearish)
    ft_cdlstalledpattern = pattern::cdlstalledpattern(open, high, low, close)[]
        -> [out: i32];
    /// Stick Sandwich (bullish)
    ft_cdlsticksandwich = pattern::cdlsticksandwich(open, high, low, close)[]
        -> [out: i32];
    /// Takuri (Dragonfly Doji with very long lower shadow)
    ft_cdltakuri = pattern::cdltakuri(open, high, low, close)[]
        -> [out: i32];
    /// Tasuki Gap
    ft_cdltasukigap = pattern::cdltasukigap(open, high, low, close)[]
        -> [out: i32];
    /// Thrusting Pattern (bearish)
    ft_cdlthrusting = pattern::cdlthrusting(open, high, low, close)[]
        -> [out: i32];
    /// Tristar Pattern
    ft_cdltristar = pattern::cdltristar(open, high, low, close)[]
        -> [out: i32];
    /// Unique 3 River (bullish)
    ft_cdlunique3river = pattern::cdlunique3river(open, high, low, close)[]
        -> [out: i32];
    /// Upside Gap Two Crows (bearish)
    ft_cdlupsidegap2crows = pattern::cdlupsidegap2crows(open, high, low, close)[]
        -> [out: i32];
    /// Upside/Downside Gap Three Methods
    ft_cdlxsidegap3methods = pattern::cdlxsidegap3methods(open, high, low, close)[]
        -> [out: i32];
}
