//! Candlestick pattern streams: stateless rules over the last 1-5 bars, each a
//! windowed replay of its `pattern::cdl*` kernel. `lookback` is the furthest
//! bar the kernel reads (`for i in K..n` => `K + 1` bars); `0` ("no
//! pattern") is a real output, as in batch, so these streams are ready from
//! the first bar.

/// Registry entries for the 61 `cdl*` pattern streams.
#[macro_export]
#[doc(hidden)]
macro_rules! stream_family_pattern {
    ($($ctx:tt)*) => {
        $crate::__stream_chain! {
            [
                cdl2crows => StreamingCdl2crows, doc "Streaming Two Crows (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdl2crows(open, high, low, close));
                cdl3blackcrows => StreamingCdl3blackcrows, doc "Streaming Three Black Crows (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdl3blackcrows(open, high, low, close));
                cdl3inside => StreamingCdl3inside, doc "Streaming Three Inside Up/Down candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdl3inside(open, high, low, close));
                cdl3linestrike => StreamingCdl3linestrike, doc "Streaming Three-Line Strike candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 4, batch = $crate::pattern::cdl3linestrike(open, high, low, close));
                cdl3outside => StreamingCdl3outside, doc "Streaming Three Outside Up/Down candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdl3outside(open, high, low, close));
                cdl3starsinsouth => StreamingCdl3starsinsouth, doc "Streaming Three Stars In The South (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdl3starsinsouth(open, high, low, close));
                cdl3whitesoldiers => StreamingCdl3whitesoldiers, doc "Streaming Three Advancing White Soldiers (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdl3whitesoldiers(open, high, low, close));
                cdlabandonedbaby => StreamingCdlabandonedbaby, doc "Streaming Abandoned Baby candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlabandonedbaby(open, high, low, close));
                cdladvanceblock => StreamingCdladvanceblock, doc "Streaming Advance Block (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdladvanceblock(open, high, low, close));
                cdlbelthold => StreamingCdlbelthold, doc "Streaming Belt-hold candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlbelthold(open, high, low, close));
                cdlbreakaway => StreamingCdlbreakaway, doc "Streaming Breakaway candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 5, batch = $crate::pattern::cdlbreakaway(open, high, low, close));
                cdlclosingmarubozu => StreamingCdlclosingmarubozu, doc "Streaming Closing Marubozu candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlclosingmarubozu(open, high, low, close));
                cdlconcealbabyswall => StreamingCdlconcealbabyswall, doc "Streaming Concealing Baby Swallow (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 4, batch = $crate::pattern::cdlconcealbabyswall(open, high, low, close));
                cdlcounterattack => StreamingCdlcounterattack, doc "Streaming Counterattack candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlcounterattack(open, high, low, close));
                cdldarkcloudcover => StreamingCdldarkcloudcover, doc "Streaming Dark Cloud Cover (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdldarkcloudcover(open, high, low, close));
                cdldoji => StreamingCdldoji, doc "Streaming Doji candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdldoji(open, high, low, close));
                cdldojistar => StreamingCdldojistar, doc "Streaming Doji Star candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdldojistar(open, high, low, close));
                cdldragonflydoji => StreamingCdldragonflydoji, doc "Streaming Dragonfly Doji (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdldragonflydoji(open, high, low, close));
                cdlengulfing => StreamingCdlengulfing, doc "Streaming Engulfing candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlengulfing(open, high, low, close));
                cdleveningdojistar => StreamingCdleveningdojistar, doc "Streaming Evening Doji Star (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdleveningdojistar(open, high, low, close));
                cdleveningstar => StreamingCdleveningstar, doc "Streaming Evening Star (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdleveningstar(open, high, low, close));
                cdlgapsidesidewhite => StreamingCdlgapsidesidewhite, doc "Streaming Up/Down-gap side-by-side white lines candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlgapsidesidewhite(open, high, low, close));
                cdlgravestonedoji => StreamingCdlgravestonedoji, doc "Streaming Gravestone Doji (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlgravestonedoji(open, high, low, close));
                cdlhammer => StreamingCdlhammer, doc "Streaming Hammer (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlhammer(open, high, low, close));
                cdlhangingman => StreamingCdlhangingman, doc "Streaming Hanging Man (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlhangingman(open, high, low, close));
                cdlharami => StreamingCdlharami, doc "Streaming Harami candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlharami(open, high, low, close));
                cdlharamicross => StreamingCdlharamicross, doc "Streaming Harami Cross candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlharamicross(open, high, low, close));
                cdlhighwave => StreamingCdlhighwave, doc "Streaming High-Wave Candle candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlhighwave(open, high, low, close));
                cdlhikkake => StreamingCdlhikkake, doc "Streaming Hikkake Pattern candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlhikkake(open, high, low, close));
                cdlhikkakemod => StreamingCdlhikkakemod, doc "Streaming Modified Hikkake Pattern candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 4, batch = $crate::pattern::cdlhikkakemod(open, high, low, close));
                cdlhomingpigeon => StreamingCdlhomingpigeon, doc "Streaming Homing Pigeon (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlhomingpigeon(open, high, low, close));
                cdlidentical3crows => StreamingCdlidentical3crows, doc "Streaming Identical Three Crows (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlidentical3crows(open, high, low, close));
                cdlinneck => StreamingCdlinneck, doc "Streaming In-Neck Pattern (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlinneck(open, high, low, close));
                cdlinvertedhammer => StreamingCdlinvertedhammer, doc "Streaming Inverted Hammer (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlinvertedhammer(open, high, low, close));
                cdlkicking => StreamingCdlkicking, doc "Streaming Kicking candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlkicking(open, high, low, close));
                cdlkickingbylength => StreamingCdlkickingbylength, doc "Streaming Kicking — bull/bear determined by longer of the two marubozu candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlkickingbylength(open, high, low, close));
                cdlladderbottom => StreamingCdlladderbottom, doc "Streaming Ladder Bottom (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 5, batch = $crate::pattern::cdlladderbottom(open, high, low, close));
                cdllongleggeddoji => StreamingCdllongleggeddoji, doc "Streaming Long Legged Doji candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdllongleggeddoji(open, high, low, close));
                cdllongline => StreamingCdllongline, doc "Streaming Long Line Candle candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdllongline(open, high, low, close));
                cdlmarubozu => StreamingCdlmarubozu, doc "Streaming Marubozu candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlmarubozu(open, high, low, close));
                cdlmatchinglow => StreamingCdlmatchinglow, doc "Streaming Matching Low (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlmatchinglow(open, high, low, close));
                cdlmathold => StreamingCdlmathold, doc "Streaming Mat Hold (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 5, batch = $crate::pattern::cdlmathold(open, high, low, close));
                cdlmorningdojistar => StreamingCdlmorningdojistar, doc "Streaming Morning Doji Star (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlmorningdojistar(open, high, low, close));
                cdlmorningstar => StreamingCdlmorningstar, doc "Streaming Morning Star (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlmorningstar(open, high, low, close));
                cdlonneck => StreamingCdlonneck, doc "Streaming On-Neck Pattern (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlonneck(open, high, low, close));
                cdlpiercing => StreamingCdlpiercing, doc "Streaming Piercing Pattern (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlpiercing(open, high, low, close));
                cdlrickshawman => StreamingCdlrickshawman, doc "Streaming Rickshaw Man candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlrickshawman(open, high, low, close));
                cdlrisefall3methods => StreamingCdlrisefall3methods, doc "Streaming Rising/Falling Three Methods candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 5, batch = $crate::pattern::cdlrisefall3methods(open, high, low, close));
                cdlseparatinglines => StreamingCdlseparatinglines, doc "Streaming Separating Lines candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlseparatinglines(open, high, low, close));
                cdlshootingstar => StreamingCdlshootingstar, doc "Streaming Shooting Star (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlshootingstar(open, high, low, close));
                cdlshortline => StreamingCdlshortline, doc "Streaming Short Line Candle candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlshortline(open, high, low, close));
                cdlspinningtop => StreamingCdlspinningtop, doc "Streaming Spinning Top candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdlspinningtop(open, high, low, close));
                cdlstalledpattern => StreamingCdlstalledpattern, doc "Streaming Stalled Pattern (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlstalledpattern(open, high, low, close));
                cdlsticksandwich => StreamingCdlsticksandwich, doc "Streaming Stick Sandwich (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlsticksandwich(open, high, low, close));
                cdltakuri => StreamingCdltakuri, doc "Streaming Takuri (Dragonfly Doji with very long lower shadow) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 1, batch = $crate::pattern::cdltakuri(open, high, low, close));
                cdltasukigap => StreamingCdltasukigap, doc "Streaming Tasuki Gap candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdltasukigap(open, high, low, close));
                cdlthrusting => StreamingCdlthrusting, doc "Streaming Thrusting Pattern (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 2, batch = $crate::pattern::cdlthrusting(open, high, low, close));
                cdltristar => StreamingCdltristar, doc "Streaming Tristar Pattern candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdltristar(open, high, low, close));
                cdlunique3river => StreamingCdlunique3river, doc "Streaming Unique 3 River (bullish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlunique3river(open, high, low, close));
                cdlupsidegap2crows => StreamingCdlupsidegap2crows, doc "Streaming Upside Gap Two Crows (bearish) candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlupsidegap2crows(open, high, low, close));
                cdlxsidegap3methods => StreamingCdlxsidegap3methods, doc "Streaming Upside/Downside Gap Three Methods candlestick pattern: 100 bullish, -100 bearish, 0 none.",
                    params(), inputs(open, high, low, close), outputs(out: i32),
                    windowed(lookback = 3, batch = $crate::pattern::cdlxsidegap3methods(open, high, low, close));
            ]
            $($ctx)*
        }
    };
}

crate::stream_family_pattern! {
    gen = crate::__stream_define;
    acc = [];
    rest = [];
}
