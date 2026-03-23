//! Candlestick pattern recognition (CDL*). One module per pattern for maintainability.

mod common;

mod cdl2crows;
mod cdl3blackcrows;
mod cdl3inside;
mod cdl3linestrike;
mod cdl3outside;
mod cdl3starsinsouth;
mod cdl3whitesoldiers;
mod cdlabandonedbaby;
mod cdladvanceblock;
mod cdlbelthold;
mod cdlbreakaway;
mod cdlclosingmarubozu;
mod cdlconcealbabyswall;
mod cdlcounterattack;
mod cdldarkcloudcover;
mod cdldoji;
mod cdldojistar;
mod cdldragonflydoji;
mod cdlengulfing;
mod cdleveningdojistar;
mod cdleveningstar;
mod cdlgapsidesidewhite;
mod cdlgravestonedoji;
mod cdlhammer;
mod cdlhangingman;
mod cdlharami;
mod cdlharamicross;
mod cdlhighwave;
mod cdlhikkake;
mod cdlhikkakemod;
mod cdlhomingpigeon;
mod cdlidentical3crows;
mod cdlinneck;
mod cdlinvertedhammer;
mod cdlkicking;
mod cdlkickingbylength;
mod cdlladderbottom;
mod cdllongleggeddoji;
mod cdllongline;
mod cdlmarubozu;
mod cdlmatchinglow;
mod cdlmathold;
mod cdlmorningdojistar;
mod cdlmorningstar;
mod cdlonneck;
mod cdlpiercing;
mod cdlrickshawman;
mod cdlrisefall3methods;
mod cdlseparatinglines;
mod cdlshootingstar;
mod cdlshortline;
mod cdlspinningtop;
mod cdlstalledpattern;
mod cdlsticksandwich;
mod cdltakuri;
mod cdltasukigap;
mod cdlthrusting;
mod cdltristar;
mod cdlunique3river;
mod cdlupsidegap2crows;
mod cdlxsidegap3methods;

use pyo3::prelude::*;

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(pyo3::wrap_pyfunction!(self::cdldoji::cdldoji, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlengulfing::cdlengulfing, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlhammer::cdlhammer, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlshootingstar::cdlshootingstar,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlmarubozu::cdlmarubozu, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlspinningtop::cdlspinningtop,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlmorningstar::cdlmorningstar,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdleveningstar::cdleveningstar,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdl2crows::cdl2crows, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdl3blackcrows::cdl3blackcrows,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdl3whitesoldiers::cdl3whitesoldiers,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdl3inside::cdl3inside, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdl3outside::cdl3outside, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdldojistar::cdldojistar, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlmorningdojistar::cdlmorningdojistar,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdleveningdojistar::cdleveningdojistar,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlharami::cdlharami, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlharamicross::cdlharamicross,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdl3linestrike::cdl3linestrike,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdl3starsinsouth::cdl3starsinsouth,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlabandonedbaby::cdlabandonedbaby,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdladvanceblock::cdladvanceblock,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlbelthold::cdlbelthold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlbreakaway::cdlbreakaway, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlclosingmarubozu::cdlclosingmarubozu,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlconcealbabyswall::cdlconcealbabyswall,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlcounterattack::cdlcounterattack,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdldarkcloudcover::cdldarkcloudcover,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdldragonflydoji::cdldragonflydoji,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlgapsidesidewhite::cdlgapsidesidewhite,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlgravestonedoji::cdlgravestonedoji,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlhangingman::cdlhangingman,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlhighwave::cdlhighwave, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlhikkake::cdlhikkake, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlhikkakemod::cdlhikkakemod,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlhomingpigeon::cdlhomingpigeon,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlidentical3crows::cdlidentical3crows,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlinneck::cdlinneck, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlinvertedhammer::cdlinvertedhammer,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlkicking::cdlkicking, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlkickingbylength::cdlkickingbylength,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlladderbottom::cdlladderbottom,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdllongleggeddoji::cdllongleggeddoji,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdllongline::cdllongline, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlmatchinglow::cdlmatchinglow,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlmathold::cdlmathold, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlonneck::cdlonneck, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlpiercing::cdlpiercing, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlrickshawman::cdlrickshawman,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlrisefall3methods::cdlrisefall3methods,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlseparatinglines::cdlseparatinglines,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlshortline::cdlshortline, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlstalledpattern::cdlstalledpattern,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlsticksandwich::cdlsticksandwich,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdltakuri::cdltakuri, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdltasukigap::cdltasukigap, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdlthrusting::cdlthrusting, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(self::cdltristar::cdltristar, m)?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlunique3river::cdlunique3river,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlupsidegap2crows::cdlupsidegap2crows,
        m
    )?)?;
    m.add_function(pyo3::wrap_pyfunction!(
        self::cdlxsidegap3methods::cdlxsidegap3methods,
        m
    )?)?;
    Ok(())
}
