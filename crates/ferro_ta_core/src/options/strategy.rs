//! Multi-leg option strategy analytics on a leg/strategy struct API.
//!
//! Expiry P&L of a strategy is piecewise linear in the spot price with kinks
//! only at option strikes, so breakevens and the max profit / max loss are
//! solved exactly from the kink values and the slope beyond the last strike
//! (no grid sampling). Spot is confined to `[0, ∞)`: the downside is always
//! bounded, and a non-zero right-tail slope means unlimited profit or loss.

use super::greeks::black_scholes_greeks;
use super::normal::cdf;
use super::pricing::black_scholes_price;
use super::Greeks;
pub use super::OptionKind;

const DAYS_PER_YEAR: f64 = 365.0;

/// Invalid leg, strategy or analysis input.
#[derive(Clone, Debug, PartialEq)]
pub struct StrategyError(pub String);

impl std::fmt::Display for StrategyError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for StrategyError {}

fn err<T>(msg: impl Into<String>) -> Result<T, StrategyError> {
    Err(StrategyError(msg.into()))
}

/// What a leg holds.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum LegKind {
    Option {
        right: OptionKind,
        strike: f64,
        /// Premium paid (long) or received (short) per unit.
        premium: f64,
        expiry_years: f64,
        volatility: f64,
    },
    Future {
        entry_price: f64,
    },
    Stock {
        entry_price: f64,
    },
}

/// One position: `quantity` is signed (+ long, − short).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Leg {
    pub kind: LegKind,
    pub quantity: f64,
    pub multiplier: f64,
}

impl Leg {
    #[allow(clippy::too_many_arguments)]
    pub fn option(
        quantity: f64,
        right: OptionKind,
        strike: f64,
        premium: f64,
        expiry_years: f64,
        volatility: f64,
        multiplier: f64,
    ) -> Self {
        Self {
            kind: LegKind::Option {
                right,
                strike,
                premium,
                expiry_years,
                volatility,
            },
            quantity,
            multiplier,
        }
    }

    pub fn future(quantity: f64, entry_price: f64, multiplier: f64) -> Self {
        Self {
            kind: LegKind::Future { entry_price },
            quantity,
            multiplier,
        }
    }

    pub fn stock(quantity: f64, entry_price: f64, multiplier: f64) -> Self {
        Self {
            kind: LegKind::Stock { entry_price },
            quantity,
            multiplier,
        }
    }

    /// Signed position size in underlying units.
    fn scale(&self) -> f64 {
        self.quantity * self.multiplier
    }

    fn validate(&self) -> Result<(), StrategyError> {
        if !self.quantity.is_finite() || self.quantity == 0.0 {
            return err(format!(
                "quantity must be finite and non-zero, got {}",
                self.quantity
            ));
        }
        if !self.multiplier.is_finite() || self.multiplier <= 0.0 {
            return err(format!(
                "multiplier must be finite and > 0, got {}",
                self.multiplier
            ));
        }
        match self.kind {
            LegKind::Option {
                strike,
                premium,
                expiry_years,
                volatility,
                ..
            } => {
                if !strike.is_finite() || strike <= 0.0 {
                    return err(format!("strike must be finite and > 0, got {strike}"));
                }
                if !premium.is_finite() || premium < 0.0 {
                    return err(format!("premium must be finite and >= 0, got {premium}"));
                }
                if !expiry_years.is_finite() || expiry_years < 0.0 {
                    return err(format!(
                        "expiry_years must be finite and >= 0, got {expiry_years}"
                    ));
                }
                if !volatility.is_finite() || volatility < 0.0 {
                    return err(format!(
                        "volatility must be finite and >= 0, got {volatility}"
                    ));
                }
            }
            LegKind::Future { entry_price } | LegKind::Stock { entry_price } => {
                if !entry_price.is_finite() || entry_price <= 0.0 {
                    return err(format!(
                        "entry_price must be finite and > 0, got {entry_price}"
                    ));
                }
            }
        }
        Ok(())
    }

    /// P&L of this leg if the underlying is at `spot` and each option is
    /// worth `option_value(right, strike, expiry, vol)`.
    fn pnl(&self, spot: f64, option_value: impl Fn(OptionKind, f64, f64, f64) -> f64) -> f64 {
        let price = match self.kind {
            LegKind::Option {
                right,
                strike,
                premium,
                expiry_years,
                volatility,
            } => option_value(right, strike, expiry_years, volatility) - premium,
            LegKind::Future { entry_price } | LegKind::Stock { entry_price } => spot - entry_price,
        };
        self.scale() * price
    }
}

fn intrinsic(right: OptionKind, spot: f64, strike: f64) -> f64 {
    match right {
        OptionKind::Call => (spot - strike).max(0.0),
        OptionKind::Put => (strike - spot).max(0.0),
    }
}

/// Max profit or max loss of a strategy at expiry.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Extreme {
    /// Attained P&L and the lowest spot where it is attained.
    Finite {
        value: f64,
        spot: f64,
    },
    Unlimited,
}

/// A validated multi-leg strategy.
#[derive(Clone, Debug, PartialEq)]
pub struct Strategy {
    legs: Vec<Leg>,
}

impl Strategy {
    pub fn new(legs: Vec<Leg>) -> Result<Self, StrategyError> {
        if legs.is_empty() {
            return err("a strategy needs at least one leg");
        }
        for (i, leg) in legs.iter().enumerate() {
            leg.validate()
                .map_err(|e| StrategyError(format!("leg {i}: {}", e.0)))?;
        }
        Ok(Self { legs })
    }

    pub fn legs(&self) -> &[Leg] {
        &self.legs
    }

    /// Shared option expiry; expiry analytics are undefined across expiries.
    fn require_single_expiry(&self) -> Result<(), StrategyError> {
        let mut expiries = self.legs.iter().filter_map(|leg| match leg.kind {
            LegKind::Option { expiry_years, .. } => Some(expiry_years),
            _ => None,
        });
        if let Some(first) = expiries.next() {
            if expiries.any(|e| e != first) {
                return err(
                    "mixed expiries: expiry payoff, breakevens, max profit/loss and \
                     probability of profit need every option leg to share one expiry",
                );
            }
        }
        Ok(())
    }

    fn expiry_pnl(&self, spot: f64) -> f64 {
        self.legs.iter().fold(0.0, |total, leg| {
            total + leg.pnl(spot, |right, strike, _, _| intrinsic(right, spot, strike))
        })
    }

    /// P&L at expiry with the underlying at `spot`.
    pub fn payoff_at(&self, spot: f64) -> Result<f64, StrategyError> {
        self.require_single_expiry()?;
        Ok(self.expiry_pnl(spot))
    }

    /// Expiry P&L over a spot grid.
    pub fn payoff(&self, spots: &[f64]) -> Result<Vec<f64>, StrategyError> {
        self.require_single_expiry()?;
        Ok(spots.iter().map(|&s| self.expiry_pnl(s)).collect())
    }

    /// `0` plus every strike, sorted and unique: the expiry P&L is linear
    /// between consecutive points.
    fn kinks(&self) -> Vec<f64> {
        let mut points: Vec<f64> = std::iter::once(0.0)
            .chain(self.legs.iter().filter_map(|leg| match leg.kind {
                LegKind::Option { strike, .. } => Some(strike),
                _ => None,
            }))
            .collect();
        points.sort_by(f64::total_cmp);
        points.dedup();
        points
    }

    /// d(P&L)/d(spot) beyond the highest strike.
    fn right_tail_slope(&self) -> f64 {
        self.legs
            .iter()
            .map(|leg| match leg.kind {
                LegKind::Option {
                    right: OptionKind::Put,
                    ..
                } => 0.0,
                _ => leg.scale(),
            })
            .sum()
    }

    /// Spots where the expiry P&L crosses or touches zero, ascending.
    /// A stretch where the P&L is identically zero contributes no points.
    pub fn breakevens(&self) -> Result<Vec<f64>, StrategyError> {
        self.require_single_expiry()?;
        let xs = self.kinks();
        let ys: Vec<f64> = xs.iter().map(|&x| self.expiry_pnl(x)).collect();
        let slope = self.right_tail_slope();
        let n = xs.len();
        let flat_zero = |i: usize| ys[i] == 0.0 && ys[i + 1] == 0.0;
        let mut roots = Vec::new();
        for i in 0..n {
            let touches_flat = (i > 0 && flat_zero(i - 1))
                || (i + 1 < n && flat_zero(i))
                || (i + 1 == n && ys[i] == 0.0 && slope == 0.0);
            if ys[i] == 0.0 && !touches_flat {
                roots.push(xs[i]);
            }
            if i + 1 < n && ys[i] * ys[i + 1] < 0.0 {
                let (a, b, fa, fb) = (xs[i], xs[i + 1], ys[i], ys[i + 1]);
                roots.push(a + fa * (b - a) / (fa - fb));
            }
        }
        let (last_x, last_y) = (xs[n - 1], ys[n - 1]);
        if last_y != 0.0 && slope != 0.0 && last_y.signum() != slope.signum() {
            roots.push(last_x - last_y / slope);
        }
        roots.sort_by(f64::total_cmp);
        roots.dedup();
        Ok(roots)
    }

    /// Best expiry P&L over `[0, ∞)`.
    pub fn max_profit(&self) -> Result<Extreme, StrategyError> {
        self.extreme(1.0)
    }

    /// Worst expiry P&L over `[0, ∞)` (a P&L value, usually negative).
    pub fn max_loss(&self) -> Result<Extreme, StrategyError> {
        self.extreme(-1.0)
    }

    /// `direction` = +1 for the maximum, −1 for the minimum.
    fn extreme(&self, direction: f64) -> Result<Extreme, StrategyError> {
        self.require_single_expiry()?;
        if self.right_tail_slope() * direction > 0.0 {
            return Ok(Extreme::Unlimited);
        }
        let best = self
            .kinks()
            .into_iter()
            .map(|x| (self.expiry_pnl(x), x))
            .fold(None, |best: Option<(f64, f64)>, (y, x)| match best {
                Some((by, _)) if y * direction <= by * direction => best,
                _ => Some((y, x)),
            });
        let (value, spot) = best.expect("kinks always contain 0");
        Ok(Extreme::Finite { value, spot })
    }

    /// Risk-neutral probability that the expiry P&L is strictly positive,
    /// with a lognormal underlying (drift `rate − carry`).
    pub fn probability_of_profit(
        &self,
        spot: f64,
        volatility: f64,
        time_years: f64,
        rate: f64,
        carry: f64,
    ) -> Result<f64, StrategyError> {
        self.require_single_expiry()?;
        if !spot.is_finite() || spot <= 0.0 {
            return err(format!("spot must be finite and > 0, got {spot}"));
        }
        if !volatility.is_finite()
            || volatility < 0.0
            || !time_years.is_finite()
            || time_years < 0.0
        {
            return err("volatility and time_years must be finite and >= 0");
        }
        if !rate.is_finite() || !carry.is_finite() {
            return err("rate and carry must be finite");
        }
        let drift = (rate - carry - 0.5 * volatility * volatility) * time_years;
        let sd = volatility * time_years.sqrt();
        if sd == 0.0 {
            let forward = spot * drift.exp();
            return Ok(if self.expiry_pnl(forward) > 0.0 {
                1.0
            } else {
                0.0
            });
        }
        // P(S_T <= x) for the lognormal terminal price.
        let below = |x: f64| {
            if x <= 0.0 {
                0.0
            } else if x.is_infinite() {
                1.0
            } else {
                cdf(((x / spot).ln() - drift) / sd)
            }
        };
        let mut edges = vec![0.0];
        edges.extend(self.breakevens()?);
        edges.push(f64::INFINITY);
        let probability = edges
            .windows(2)
            .filter(|w| {
                let probe = if w[1].is_infinite() {
                    w[0] * 2.0 + 1.0
                } else {
                    0.5 * (w[0] + w[1])
                };
                self.expiry_pnl(probe) > 0.0
            })
            .map(|w| below(w[1]) - below(w[0]))
            .sum::<f64>();
        Ok(probability.clamp(0.0, 1.0))
    }

    /// Black-Scholes greeks summed over legs (each option at its own expiry
    /// and volatility); futures and stock contribute delta only.
    pub fn greeks(&self, spot: f64, rate: f64, carry: f64) -> Greeks {
        let mut total = Greeks {
            delta: 0.0,
            gamma: 0.0,
            vega: 0.0,
            theta: 0.0,
            rho: 0.0,
        };
        for leg in &self.legs {
            let scale = leg.scale();
            match leg.kind {
                LegKind::Option {
                    right,
                    strike,
                    expiry_years,
                    volatility,
                    ..
                } => {
                    let g = black_scholes_greeks(
                        spot,
                        strike,
                        rate,
                        carry,
                        expiry_years,
                        volatility,
                        right,
                    );
                    total.delta += scale * g.delta;
                    total.gamma += scale * g.gamma;
                    total.vega += scale * g.vega;
                    total.theta += scale * g.theta;
                    total.rho += scale * g.rho;
                }
                LegKind::Future { .. } | LegKind::Stock { .. } => total.delta += scale,
            }
        }
        total
    }

    /// Mark-to-model P&L at `spot` after `days_forward` days with every
    /// option's volatility shifted by `vol_shift` (both floored at 0).
    pub fn value_at(
        &self,
        spot: f64,
        vol_shift: f64,
        days_forward: f64,
        rate: f64,
        carry: f64,
    ) -> f64 {
        let years_forward = days_forward / DAYS_PER_YEAR;
        self.legs.iter().fold(0.0, |total, leg| {
            total
                + leg.pnl(spot, |right, strike, expiry, vol| {
                    let t = (expiry - years_forward).max(0.0);
                    let sigma = (vol + vol_shift).max(0.0);
                    black_scholes_price(spot, strike, rate, carry, t, sigma, right)
                })
        })
    }

    /// What-if grid: row-major `[vol_shift][spot]` of [`Strategy::value_at`].
    pub fn scenario_grid(
        &self,
        spots: &[f64],
        vol_shifts: &[f64],
        days_forward: f64,
        rate: f64,
        carry: f64,
    ) -> Vec<f64> {
        vol_shifts
            .iter()
            .flat_map(|&shift| {
                spots
                    .iter()
                    .map(move |&s| self.value_at(s, shift, days_forward, rate, carry))
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use OptionKind::{Call, Put};

    const T: f64 = 0.25;
    const VOL: f64 = 0.2;

    fn option(qty: f64, right: OptionKind, strike: f64, premium: f64) -> Leg {
        Leg::option(qty, right, strike, premium, T, VOL, 1.0)
    }

    fn finite(e: Extreme) -> (f64, f64) {
        match e {
            Extreme::Finite { value, spot } => (value, spot),
            Extreme::Unlimited => panic!("expected finite extreme"),
        }
    }

    fn approx(a: f64, b: f64) -> bool {
        (a - b).abs() < 1e-9
    }

    #[test]
    fn long_call_breakeven_and_extremes() {
        let s = Strategy::new(vec![option(1.0, Call, 100.0, 5.0)]).unwrap();
        assert_eq!(s.breakevens().unwrap(), vec![105.0]);
        assert_eq!(s.max_profit().unwrap(), Extreme::Unlimited);
        let (loss, _) = finite(s.max_loss().unwrap());
        assert!(approx(loss, -5.0));
        assert!(approx(s.payoff_at(110.0).unwrap(), 5.0));
    }

    #[test]
    fn naked_short_call_loss_unlimited() {
        let s = Strategy::new(vec![option(-1.0, Call, 100.0, 5.0)]).unwrap();
        assert_eq!(s.max_loss().unwrap(), Extreme::Unlimited);
        let (profit, _) = finite(s.max_profit().unwrap());
        assert!(approx(profit, 5.0));
    }

    #[test]
    fn covered_call_bounded_both_sides() {
        let s = Strategy::new(vec![
            Leg::stock(1.0, 100.0, 1.0),
            option(-1.0, Call, 110.0, 3.0),
        ])
        .unwrap();
        let (profit, profit_spot) = finite(s.max_profit().unwrap());
        assert!(approx(profit, 13.0) && approx(profit_spot, 110.0));
        let (loss, loss_spot) = finite(s.max_loss().unwrap());
        assert!(
            approx(loss, -97.0) && approx(loss_spot, 0.0),
            "worst case at S=0"
        );
        assert_eq!(s.breakevens().unwrap(), vec![97.0]);
    }

    fn iron_condor() -> Strategy {
        // Credit = 3 + 3 - 1 - 1 = 4; wings 10 wide.
        Strategy::new(vec![
            option(1.0, Put, 80.0, 1.0),
            option(-1.0, Put, 90.0, 3.0),
            option(-1.0, Call, 110.0, 3.0),
            option(1.0, Call, 120.0, 1.0),
        ])
        .unwrap()
    }

    #[test]
    fn iron_condor_two_breakevens_and_extremes() {
        let s = iron_condor();
        assert_eq!(s.breakevens().unwrap(), vec![86.0, 114.0]);
        let (profit, _) = finite(s.max_profit().unwrap());
        assert!(approx(profit, 4.0), "max profit = net credit");
        let (loss, _) = finite(s.max_loss().unwrap());
        assert!(approx(loss, -6.0), "max loss = width - credit");
    }

    #[test]
    fn straddle_two_breakevens() {
        let s = Strategy::new(vec![
            option(1.0, Call, 100.0, 4.0),
            option(1.0, Put, 100.0, 6.0),
        ])
        .unwrap();
        assert_eq!(s.breakevens().unwrap(), vec![90.0, 110.0]);
        let (loss, spot) = finite(s.max_loss().unwrap());
        assert!(approx(loss, -10.0) && approx(spot, 100.0));
        assert_eq!(s.max_profit().unwrap(), Extreme::Unlimited);
    }

    #[test]
    fn flat_zero_segment_has_no_breakeven_points() {
        // Long 100 call + short 100 call cancels exactly: payoff identically 0.
        let s = Strategy::new(vec![
            option(1.0, Call, 100.0, 2.0),
            option(-1.0, Call, 100.0, 2.0),
        ])
        .unwrap();
        assert!(s.breakevens().unwrap().is_empty());
        assert_eq!(finite(s.max_profit().unwrap()).0, 0.0);
    }

    #[test]
    fn mixed_expiries_rejected_for_expiry_analytics() {
        let calendar = Strategy::new(vec![
            Leg::option(-1.0, Call, 100.0, 3.0, 0.1, VOL, 1.0),
            Leg::option(1.0, Call, 100.0, 5.0, 0.3, VOL, 1.0),
        ])
        .unwrap();
        assert!(calendar.breakevens().is_err());
        assert!(calendar.max_profit().is_err());
        assert!(calendar
            .probability_of_profit(100.0, VOL, 0.1, 0.0, 0.0)
            .is_err());
        let g = calendar.greeks(100.0, 0.0, 0.0);
        assert!(g.vega > 0.0, "long back month dominates vega");
        assert_eq!(
            calendar
                .scenario_grid(&[95.0, 105.0], &[0.0], 0.0, 0.0, 0.0)
                .len(),
            2
        );
    }

    #[test]
    fn invalid_legs_rejected() {
        let bad = [
            option(0.0, Call, 100.0, 1.0),
            option(1.0, Call, -100.0, 1.0),
            option(1.0, Call, f64::NAN, 1.0),
            option(1.0, Call, 100.0, -1.0),
            Leg::option(1.0, Call, 100.0, 1.0, -0.1, VOL, 1.0),
            Leg::option(1.0, Call, 100.0, 1.0, T, -0.2, 1.0),
            Leg::option(1.0, Call, 100.0, 1.0, T, VOL, 0.0),
            Leg::future(1.0, f64::INFINITY, 1.0),
        ];
        for leg in bad {
            assert!(Strategy::new(vec![leg]).is_err(), "{leg:?}");
        }
        assert!(Strategy::new(vec![]).is_err(), "empty strategy");
    }

    #[test]
    fn payoff_matches_slice_api() {
        let s = iron_condor();
        let grid: Vec<f64> = (60..=140).map(f64::from).collect();
        let slice = super::super::payoff::strategy_payoff_dense(
            &grid,
            &[0, 0, 0, 0],
            &[1, -1, -1, 1],
            &[-1, -1, 1, 1],
            &[80.0, 90.0, 110.0, 120.0],
            &[1.0, 3.0, 3.0, 1.0],
            &[0.0; 4],
            &[1.0; 4],
            &[1.0; 4],
        );
        assert_eq!(s.payoff(&grid).unwrap(), slice);
    }

    #[test]
    fn greeks_match_aggregate_greeks_dense() {
        let s = iron_condor();
        let g = s.greeks(100.0, 0.03, 0.01);
        let (d, ga, v, th, rh) = super::super::payoff::aggregate_greeks_dense(
            100.0,
            &[0, 0, 0, 0],
            &[1, -1, -1, 1],
            &[-1, -1, 1, 1],
            &[80.0, 90.0, 110.0, 120.0],
            &[VOL; 4],
            &[T; 4],
            &[0.03; 4],
            &[0.01; 4],
            &[1.0; 4],
            &[1.0; 4],
        );
        let close = |a: f64, b: f64| (a - b).abs() <= 1e-12 * b.abs().max(1.0);
        assert!(close(g.delta, d) && close(g.gamma, ga) && close(g.vega, v));
        assert!(close(g.theta, th) && close(g.rho, rh));
    }

    #[test]
    fn pop_long_call_matches_closed_form() {
        // P(S_T > 105) under risk-neutral lognormal = N(d2) at strike 105.
        let s = Strategy::new(vec![option(1.0, Call, 100.0, 5.0)]).unwrap();
        let (spot, r, q) = (100.0, 0.05, 0.0);
        let pop = s.probability_of_profit(spot, VOL, T, r, q).unwrap();
        let d2 = ((spot / 105.0_f64).ln() + (r - q - 0.5 * VOL * VOL) * T) / (VOL * T.sqrt());
        assert!((pop - super::super::normal::cdf(d2)).abs() < 1e-12, "{pop}");
    }

    #[test]
    fn pop_iron_condor_between_breakevens() {
        let s = iron_condor();
        let (spot, r, q) = (100.0, 0.0, 0.0);
        let pop = s.probability_of_profit(spot, VOL, T, r, q).unwrap();
        let sd = VOL * T.sqrt();
        let p_below =
            |x: f64| super::super::normal::cdf(((x / spot).ln() + 0.5 * VOL * VOL * T) / sd);
        let expected = p_below(114.0) - p_below(86.0);
        assert!((pop - expected).abs() < 1e-12, "{pop} vs {expected}");
        assert!(s.probability_of_profit(spot, -0.1, T, r, q).is_err());
    }

    #[test]
    fn scenario_grid_shape_and_zero_shift_matches_value() {
        let s = iron_condor();
        let spots = [90.0, 100.0, 110.0];
        let shifts = [-0.05, 0.0, 0.03];
        let grid = s.scenario_grid(&spots, &shifts, 0.0, 0.0, 0.0);
        assert_eq!(grid.len(), shifts.len() * spots.len());
        for (j, &sp) in spots.iter().enumerate() {
            let direct = s.value_at(sp, 0.0, 0.0, 0.0, 0.0);
            assert_eq!(grid[spots.len() + j], direct, "row 1 is the zero shift");
        }
        // Past expiry the grid equals the expiry payoff.
        let expired = s.scenario_grid(&spots, &[0.0], 365.0, 0.0, 0.0);
        assert_eq!(expired, s.payoff(&spots).unwrap());
        // Short premium strategy loses value when vol rises.
        assert!(grid[2 * spots.len() + 1] < grid[spots.len() + 1]);
    }
}
