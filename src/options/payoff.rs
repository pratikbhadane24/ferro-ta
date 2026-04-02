use numpy::{IntoPyArray, PyArray1, PyReadonlyArray1};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyAny, PyTuple};

#[derive(Clone, Copy)]
enum Instrument {
    Option,
    Future,
    Stock,
}

#[derive(Clone, Copy)]
enum Side {
    Long,
    Short,
}

#[derive(Clone, Copy)]
enum OptionType {
    Call,
    Put,
}

impl Side {
    fn sign(self) -> f64 {
        match self {
            Side::Long => 1.0,
            Side::Short => -1.0,
        }
    }
}

fn parse_instrument(v: i64) -> PyResult<Instrument> {
    match v {
        0 => Ok(Instrument::Option),
        1 => Ok(Instrument::Future),
        2 => Ok(Instrument::Stock),
        _ => Err(PyValueError::new_err(
            "instrument must be 0 (option), 1 (future), or 2 (stock)",
        )),
    }
}

fn parse_side(v: i64) -> PyResult<Side> {
    match v {
        1 => Ok(Side::Long),
        -1 => Ok(Side::Short),
        _ => Err(PyValueError::new_err("side must be 1 (long) or -1 (short)")),
    }
}

fn parse_option_type(v: i64) -> PyResult<OptionType> {
    match v {
        1 => Ok(OptionType::Call),
        -1 => Ok(OptionType::Put),
        _ => Err(PyValueError::new_err(
            "option_type must be 1 (call) or -1 (put)",
        )),
    }
}

fn parse_instrument_label(v: &str) -> PyResult<Instrument> {
    match v.to_ascii_lowercase().as_str() {
        "option" => Ok(Instrument::Option),
        "future" => Ok(Instrument::Future),
        "stock" => Ok(Instrument::Stock),
        _ => Err(PyValueError::new_err(
            "instrument must be 'option', 'future', or 'stock'",
        )),
    }
}

fn parse_side_label(v: &str) -> PyResult<Side> {
    match v.to_ascii_lowercase().as_str() {
        "long" => Ok(Side::Long),
        "short" => Ok(Side::Short),
        _ => Err(PyValueError::new_err("side must be 'long' or 'short'")),
    }
}

fn parse_option_type_label(v: &str) -> PyResult<OptionType> {
    match v.to_ascii_lowercase().as_str() {
        "call" => Ok(OptionType::Call),
        "put" => Ok(OptionType::Put),
        _ => Err(PyValueError::new_err("option_type must be 'call' or 'put'")),
    }
}

fn leg_attr_string(leg: &Bound<'_, PyAny>, name: &str) -> PyResult<String> {
    let value = leg
        .getattr(name)
        .map_err(|_| PyValueError::new_err(format!("leg missing '{name}' attribute")))?;
    value.extract::<String>().map_err(|_| {
        PyValueError::new_err(format!(
            "leg field '{name}' has invalid type; expected string"
        ))
    })
}

fn leg_attr_f64(leg: &Bound<'_, PyAny>, name: &str) -> PyResult<f64> {
    let value = leg
        .getattr(name)
        .map_err(|_| PyValueError::new_err(format!("leg missing '{name}' attribute")))?;
    value.extract::<f64>().map_err(|_| {
        PyValueError::new_err(format!(
            "leg field '{name}' has invalid type; expected float"
        ))
    })
}

fn leg_attr_optional_string(leg: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<String>> {
    let value = leg
        .getattr(name)
        .map_err(|_| PyValueError::new_err(format!("leg missing '{name}' attribute")))?;
    if value.is_none() {
        return Ok(None);
    }
    value.extract::<String>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "leg field '{name}' has invalid type; expected string or None"
        ))
    })
}

fn leg_attr_optional_f64(leg: &Bound<'_, PyAny>, name: &str) -> PyResult<Option<f64>> {
    let value = leg
        .getattr(name)
        .map_err(|_| PyValueError::new_err(format!("leg missing '{name}' attribute")))?;
    if value.is_none() {
        return Ok(None);
    }
    value.extract::<f64>().map(Some).map_err(|_| {
        PyValueError::new_err(format!(
            "leg field '{name}' has invalid type; expected float or None"
        ))
    })
}

/// Compute aggregate strategy payoff over a spot grid.
///
/// Encoded arrays (same length = n_legs):
/// - `instruments`: 0=option, 1=future
/// - `sides`: 1=long, -1=short
/// - `option_types`: 1=call, -1=put (ignored for futures)
/// - `strikes`: strike for options, ignored for futures
/// - `premiums`: premium for options, ignored for futures
/// - `entry_prices`: entry price for futures, ignored for options
/// - `quantities`, `multipliers`: applied to both instruments
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn strategy_payoff_dense<'py>(
    py: Python<'py>,
    spot_grid: PyReadonlyArray1<'py, f64>,
    instruments: PyReadonlyArray1<'py, i64>,
    sides: PyReadonlyArray1<'py, i64>,
    option_types: PyReadonlyArray1<'py, i64>,
    strikes: PyReadonlyArray1<'py, f64>,
    premiums: PyReadonlyArray1<'py, f64>,
    entry_prices: PyReadonlyArray1<'py, f64>,
    quantities: PyReadonlyArray1<'py, f64>,
    multipliers: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let grid = spot_grid.as_slice()?;
    let inst = instruments.as_slice()?;
    let side = sides.as_slice()?;
    let opt_t = option_types.as_slice()?;
    let strike = strikes.as_slice()?;
    let premium = premiums.as_slice()?;
    let entry = entry_prices.as_slice()?;
    let qty = quantities.as_slice()?;
    let mult = multipliers.as_slice()?;

    let n_legs = inst.len();
    if side.len() != n_legs
        || opt_t.len() != n_legs
        || strike.len() != n_legs
        || premium.len() != n_legs
        || entry.len() != n_legs
        || qty.len() != n_legs
        || mult.len() != n_legs
    {
        return Err(PyValueError::new_err(
            "All leg arrays must have the same length",
        ));
    }

    let mut total = vec![0.0_f64; grid.len()];

    for leg_idx in 0..n_legs {
        let instrument = parse_instrument(inst[leg_idx])?;
        let side_sign = parse_side(side[leg_idx])?.sign();
        let leg_scale = side_sign * qty[leg_idx] * mult[leg_idx];

        match instrument {
            Instrument::Option => {
                let otype = parse_option_type(opt_t[leg_idx])?;
                let k = strike[leg_idx];
                let p = premium[leg_idx];
                for (i, &s) in grid.iter().enumerate() {
                    let intrinsic = match otype {
                        OptionType::Call => (s - k).max(0.0),
                        OptionType::Put => (k - s).max(0.0),
                    };
                    total[i] += leg_scale * (intrinsic - p);
                }
            }
            Instrument::Future | Instrument::Stock => {
                let e = entry[leg_idx];
                for (i, &s) in grid.iter().enumerate() {
                    total[i] += leg_scale * (s - e);
                }
            }
        }
    }

    Ok(total.into_pyarray(py))
}

/// Compute aggregate strategy payoff from Python leg objects.
///
/// `legs` is expected to be a sequence of `PayoffLeg`-like objects
/// with attributes used by `ferro_ta.analysis.derivatives_payoff`.
#[pyfunction]
pub fn strategy_payoff_legs<'py>(
    py: Python<'py>,
    spot_grid: PyReadonlyArray1<'py, f64>,
    legs: Bound<'py, PyTuple>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let grid = spot_grid.as_slice()?;
    let mut total = vec![0.0_f64; grid.len()];

    for leg in legs.iter() {
        let instrument = parse_instrument_label(&leg_attr_string(&leg, "instrument")?)?;
        let side_sign = parse_side_label(&leg_attr_string(&leg, "side")?)?.sign();
        let quantity = leg_attr_f64(&leg, "quantity")?;
        let multiplier = leg_attr_f64(&leg, "multiplier")?;
        let leg_scale = side_sign * quantity * multiplier;

        match instrument {
            Instrument::Option => {
                let otype_raw =
                    leg_attr_optional_string(&leg, "option_type")?.ok_or_else(|| {
                        PyValueError::new_err("Option payoff legs require option_type.")
                    })?;
                let otype = parse_option_type_label(&otype_raw)?;
                let strike = leg_attr_optional_f64(&leg, "strike")?
                    .ok_or_else(|| PyValueError::new_err("Option payoff legs require strike."))?;
                let premium = leg_attr_f64(&leg, "premium")?;

                for (i, &s) in grid.iter().enumerate() {
                    let intrinsic = match otype {
                        OptionType::Call => (s - strike).max(0.0),
                        OptionType::Put => (strike - s).max(0.0),
                    };
                    total[i] += leg_scale * (intrinsic - premium);
                }
            }
            Instrument::Future | Instrument::Stock => {
                let entry_price = leg_attr_optional_f64(&leg, "entry_price")?.ok_or_else(|| {
                    PyValueError::new_err("Futures/stock payoff legs require entry_price.")
                })?;
                for (i, &s) in grid.iter().enumerate() {
                    total[i] += leg_scale * (s - entry_price);
                }
            }
        }
    }

    Ok(total.into_pyarray(py))
}

/// Aggregate Greeks over multiple legs.
///
/// Encodings match `strategy_payoff_dense`.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn aggregate_greeks_dense(
    spot: f64,
    instruments: PyReadonlyArray1<'_, i64>,
    sides: PyReadonlyArray1<'_, i64>,
    option_types: PyReadonlyArray1<'_, i64>,
    strikes: PyReadonlyArray1<'_, f64>,
    volatilities: PyReadonlyArray1<'_, f64>,
    time_to_expiries: PyReadonlyArray1<'_, f64>,
    rates: PyReadonlyArray1<'_, f64>,
    carries: PyReadonlyArray1<'_, f64>,
    quantities: PyReadonlyArray1<'_, f64>,
    multipliers: PyReadonlyArray1<'_, f64>,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let inst = instruments.as_slice()?;
    let side = sides.as_slice()?;
    let opt_t = option_types.as_slice()?;
    let strike = strikes.as_slice()?;
    let vol = volatilities.as_slice()?;
    let tte = time_to_expiries.as_slice()?;
    let rate = rates.as_slice()?;
    let carry = carries.as_slice()?;
    let qty = quantities.as_slice()?;
    let mult = multipliers.as_slice()?;

    let n_legs = inst.len();
    if side.len() != n_legs
        || opt_t.len() != n_legs
        || strike.len() != n_legs
        || vol.len() != n_legs
        || tte.len() != n_legs
        || rate.len() != n_legs
        || carry.len() != n_legs
        || qty.len() != n_legs
        || mult.len() != n_legs
    {
        return Err(PyValueError::new_err(
            "All leg arrays must have the same length",
        ));
    }

    let mut delta = 0.0_f64;
    let mut gamma = 0.0_f64;
    let mut vega = 0.0_f64;
    let mut theta = 0.0_f64;
    let mut rho = 0.0_f64;

    for i in 0..n_legs {
        let instrument = parse_instrument(inst[i])?;
        let side_sign = parse_side(side[i])?.sign();
        let leg_scale = side_sign * qty[i] * mult[i];
        match instrument {
            Instrument::Future | Instrument::Stock => {
                delta += leg_scale;
            }
            Instrument::Option => {
                if vol[i].is_nan() || tte[i].is_nan() {
                    return Err(PyValueError::new_err(
                        "Option legs require strike, volatility, and time_to_expiry for Greeks aggregation.",
                    ));
                }
                let kind = match parse_option_type(opt_t[i])? {
                    OptionType::Call => ferro_ta_core::options::OptionKind::Call,
                    OptionType::Put => ferro_ta_core::options::OptionKind::Put,
                };
                let greeks = ferro_ta_core::options::greeks::model_greeks(
                    ferro_ta_core::options::OptionEvaluation {
                        contract: ferro_ta_core::options::OptionContract {
                            model: ferro_ta_core::options::PricingModel::BlackScholes,
                            underlying: spot,
                            strike: strike[i],
                            rate: rate[i],
                            carry: carry[i],
                            time_to_expiry: tte[i],
                            kind,
                        },
                        volatility: vol[i],
                    },
                );
                delta += leg_scale * greeks.delta;
                gamma += leg_scale * greeks.gamma;
                vega += leg_scale * greeks.vega;
                theta += leg_scale * greeks.theta;
                rho += leg_scale * greeks.rho;
            }
        }
    }

    Ok((delta, gamma, vega, theta, rho))
}

/// Aggregate Greeks from Python leg objects.
#[pyfunction]
pub fn aggregate_greeks_legs(
    spot: f64,
    legs: Bound<'_, PyTuple>,
) -> PyResult<(f64, f64, f64, f64, f64)> {
    let mut delta = 0.0_f64;
    let mut gamma = 0.0_f64;
    let mut vega = 0.0_f64;
    let mut theta = 0.0_f64;
    let mut rho = 0.0_f64;

    for leg in legs.iter() {
        let instrument = parse_instrument_label(&leg_attr_string(&leg, "instrument")?)?;
        let side_sign = parse_side_label(&leg_attr_string(&leg, "side")?)?.sign();
        let quantity = leg_attr_f64(&leg, "quantity")?;
        let multiplier = leg_attr_f64(&leg, "multiplier")?;
        let leg_scale = side_sign * quantity * multiplier;

        match instrument {
            Instrument::Future | Instrument::Stock => {
                delta += leg_scale;
            }
            Instrument::Option => {
                let otype_raw =
                    leg_attr_optional_string(&leg, "option_type")?.ok_or_else(|| {
                        PyValueError::new_err(
                            "Option legs require option_type for Greeks aggregation.",
                        )
                    })?;
                let otype = parse_option_type_label(&otype_raw)?;
                let strike = leg_attr_optional_f64(&leg, "strike")?.ok_or_else(|| {
                    PyValueError::new_err(
                        "Option legs require strike, volatility, and time_to_expiry for Greeks aggregation.",
                    )
                })?;
                let volatility = leg_attr_optional_f64(&leg, "volatility")?.ok_or_else(|| {
                    PyValueError::new_err(
                        "Option legs require strike, volatility, and time_to_expiry for Greeks aggregation.",
                    )
                })?;
                let time_to_expiry =
                    leg_attr_optional_f64(&leg, "time_to_expiry")?.ok_or_else(|| {
                        PyValueError::new_err(
                            "Option legs require strike, volatility, and time_to_expiry for Greeks aggregation.",
                        )
                    })?;
                let rate = leg_attr_f64(&leg, "rate")?;
                let carry = leg_attr_f64(&leg, "carry")?;

                let kind = match otype {
                    OptionType::Call => ferro_ta_core::options::OptionKind::Call,
                    OptionType::Put => ferro_ta_core::options::OptionKind::Put,
                };
                let greeks = ferro_ta_core::options::greeks::model_greeks(
                    ferro_ta_core::options::OptionEvaluation {
                        contract: ferro_ta_core::options::OptionContract {
                            model: ferro_ta_core::options::PricingModel::BlackScholes,
                            underlying: spot,
                            strike,
                            rate,
                            carry,
                            time_to_expiry,
                            kind,
                        },
                        volatility,
                    },
                );
                delta += leg_scale * greeks.delta;
                gamma += leg_scale * greeks.gamma;
                vega += leg_scale * greeks.vega;
                theta += leg_scale * greeks.theta;
                rho += leg_scale * greeks.rho;
            }
        }
    }

    Ok((delta, gamma, vega, theta, rho))
}

/// Compute BSM-based strategy value over a spot grid (pre-expiry mark-to-market).
///
/// Unlike `strategy_payoff_dense` (which uses intrinsic at expiry), this function
/// values each option leg using the Black-Scholes model price. Futures and stock
/// legs are valued the same as in `strategy_payoff_dense`.
///
/// Delegates to `ferro_ta_core::options::payoff::strategy_value_grid`.
///
/// NOTE: `crates/ferro_ta_core/src/options/mod.rs` must declare `pub mod payoff;`
/// for this function to compile.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
pub fn strategy_value_dense<'py>(
    py: Python<'py>,
    spot_grid: PyReadonlyArray1<'py, f64>,
    instruments: PyReadonlyArray1<'py, i64>,
    sides: PyReadonlyArray1<'py, i64>,
    option_types: PyReadonlyArray1<'py, i64>,
    strikes: PyReadonlyArray1<'py, f64>,
    premiums: PyReadonlyArray1<'py, f64>,
    entry_prices: PyReadonlyArray1<'py, f64>,
    quantities: PyReadonlyArray1<'py, f64>,
    multipliers: PyReadonlyArray1<'py, f64>,
    time_to_expiries: PyReadonlyArray1<'py, f64>,
    volatilities: PyReadonlyArray1<'py, f64>,
    rates_per_leg: PyReadonlyArray1<'py, f64>,
    carries_per_leg: PyReadonlyArray1<'py, f64>,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let grid = spot_grid.as_slice()?;
    let inst = instruments.as_slice()?;
    let side = sides.as_slice()?;
    let opt_t = option_types.as_slice()?;
    let strike = strikes.as_slice()?;
    let premium = premiums.as_slice()?;
    let entry = entry_prices.as_slice()?;
    let qty = quantities.as_slice()?;
    let mult = multipliers.as_slice()?;
    let tte = time_to_expiries.as_slice()?;
    let vol = volatilities.as_slice()?;
    let rate = rates_per_leg.as_slice()?;
    let carry = carries_per_leg.as_slice()?;

    let result = ferro_ta_core::options::payoff::strategy_value_grid(
        grid, inst, side, opt_t, strike, premium, entry, qty, mult, tte, vol, rate, carry,
    );

    Ok(result.into_pyarray(py))
}
