//! PyO3 wrapper for `ferro_ta_core::options::strategy::Strategy`.

use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use ferro_ta_core::options::strategy::{Extreme, Strategy, StrategyError};

fn to_py_err(e: StrategyError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// `(value, spot)`; unlimited is `(±inf, None)`.
fn extreme_tuple(e: Extreme, unlimited: f64) -> (f64, Option<f64>) {
    match e {
        Extreme::Finite { value, spot } => (value, Some(spot)),
        Extreme::Unlimited => (unlimited, None),
    }
}

/// Multi-leg option strategy: exact breakevens, max profit / loss (with
/// unlimited detection), probability of profit, greeks and scenario grids.
///
/// Build with ``OptionStrategy(legs)`` or ``option_strategy(legs)`` (which also
/// accepts dicts and a ``DerivativesStrategy``).
#[pyclass(module = "ferro_ta._ferro_ta", frozen)]
pub struct OptionStrategy {
    inner: Strategy,
}

#[pymethods]
impl OptionStrategy {
    /// Build from an iterable of ``PayoffLeg``-like objects (use
    /// ``option_strategy`` to also accept dicts or a ``DerivativesStrategy``).
    #[new]
    fn new(legs: &Bound<'_, PyAny>) -> PyResult<Self> {
        let legs = legs
            .try_iter()?
            .map(|leg| crate::options::payoff::core_leg_from_py(&leg?))
            .collect::<PyResult<Vec<_>>>()?;
        Ok(Self {
            inner: Strategy::new(legs).map_err(to_py_err)?,
        })
    }

    /// Expiry P&L over ``spot_grid``.
    fn payoff<'py>(
        &self,
        py: Python<'py>,
        spot_grid: Vec<f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let values = self.inner.payoff(&spot_grid).map_err(to_py_err)?;
        Ok(values.into_pyarray(py))
    }

    /// Spots where expiry P&L crosses zero, ascending.
    fn breakevens(&self) -> PyResult<Vec<f64>> {
        self.inner.breakevens().map_err(to_py_err)
    }

    /// ``(value, spot)`` of the best expiry P&L; ``(inf, None)`` if unlimited.
    fn max_profit(&self) -> PyResult<(f64, Option<f64>)> {
        let e = self.inner.max_profit().map_err(to_py_err)?;
        Ok(extreme_tuple(e, f64::INFINITY))
    }

    /// ``(value, spot)`` of the worst expiry P&L; ``(-inf, None)`` if unlimited.
    fn max_loss(&self) -> PyResult<(f64, Option<f64>)> {
        let e = self.inner.max_loss().map_err(to_py_err)?;
        Ok(extreme_tuple(e, f64::NEG_INFINITY))
    }

    /// Risk-neutral probability that expiry P&L > 0 (lognormal underlying).
    #[pyo3(signature = (spot, volatility, time_to_expiry, rate = 0.0, carry = 0.0))]
    fn probability_of_profit(
        &self,
        spot: f64,
        volatility: f64,
        time_to_expiry: f64,
        rate: f64,
        carry: f64,
    ) -> PyResult<f64> {
        self.inner
            .probability_of_profit(spot, volatility, time_to_expiry, rate, carry)
            .map_err(to_py_err)
    }

    /// ``(delta, gamma, vega, theta, rho)`` summed over legs.
    #[pyo3(signature = (spot, rate = 0.0, carry = 0.0))]
    fn greeks(&self, spot: f64, rate: f64, carry: f64) -> PyResult<(f64, f64, f64, f64, f64)> {
        let g = self.inner.greeks(spot, rate, carry).map_err(to_py_err)?;
        Ok((g.delta, g.gamma, g.vega, g.theta, g.rho))
    }

    /// Mark-to-model P&L, shape ``(len(vol_shifts), len(spots))``.
    #[pyo3(signature = (spots, vol_shifts, days_forward = 0.0, rate = 0.0, carry = 0.0))]
    fn scenario_grid<'py>(
        &self,
        py: Python<'py>,
        spots: Vec<f64>,
        vol_shifts: Vec<f64>,
        days_forward: f64,
        rate: f64,
        carry: f64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let (spots, shifts) = (spots.as_slice(), vol_shifts.as_slice());
        let flat = self
            .inner
            .scenario_grid(spots, shifts, days_forward, rate, carry);
        let grid = numpy::ndarray::Array2::from_shape_vec((shifts.len(), spots.len()), flat)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(grid.into_pyarray(py))
    }

    fn __len__(&self) -> usize {
        self.inner.legs().len()
    }

    fn __repr__(&self) -> String {
        format!("OptionStrategy(legs={})", self.inner.legs().len())
    }
}
