//! Option chain analytics helpers.

use super::greeks::model_greeks;
use super::{ChainGreeksContext, OptionContract, OptionEvaluation, OptionKind};

/// Return the index of the strike closest to the reference price.
pub fn atm_index(strikes: &[f64], reference_price: f64) -> Option<usize> {
    if strikes.is_empty() || !reference_price.is_finite() {
        return None;
    }
    strikes
        .iter()
        .enumerate()
        .filter(|(_, strike)| strike.is_finite())
        .min_by(|(_, a), (_, b)| {
            (*a - reference_price)
                .abs()
                .partial_cmp(&(*b - reference_price).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(idx, _)| idx)
}

/// Label strikes as ITM (1), ATM (0), or OTM (-1).
pub fn label_moneyness(strikes: &[f64], reference_price: f64, kind: OptionKind) -> Vec<i8> {
    let mut labels = Vec::with_capacity(strikes.len());
    let atm_idx = atm_index(strikes, reference_price);
    for (idx, &strike) in strikes.iter().enumerate() {
        if Some(idx) == atm_idx {
            labels.push(0);
            continue;
        }
        let label = match kind {
            OptionKind::Call => {
                if strike < reference_price {
                    1
                } else {
                    -1
                }
            }
            OptionKind::Put => {
                if strike > reference_price {
                    1
                } else {
                    -1
                }
            }
        };
        labels.push(label);
    }
    labels
}

/// Select a strike relative to the ATM strike by offset steps.
pub fn select_strike_by_offset(
    strikes: &[f64],
    reference_price: f64,
    offset: isize,
) -> Option<f64> {
    let idx = atm_index(strikes, reference_price)? as isize + offset;
    if idx < 0 || idx >= strikes.len() as isize {
        None
    } else {
        Some(strikes[idx as usize])
    }
}

/// Select the strike whose delta is closest to the requested target.
pub fn select_strike_by_delta(
    strikes: &[f64],
    vols: &[f64],
    context: ChainGreeksContext,
    target_delta: f64,
) -> Option<f64> {
    if strikes.len() != vols.len() || strikes.is_empty() {
        return None;
    }
    strikes
        .iter()
        .zip(vols.iter())
        .filter(|(strike, vol)| strike.is_finite() && vol.is_finite())
        .min_by(|(strike_a, vol_a), (strike_b, vol_b)| {
            let delta_a = model_greeks(OptionEvaluation {
                contract: OptionContract {
                    model: context.model,
                    underlying: context.reference_price,
                    strike: **strike_a,
                    rate: context.rate,
                    carry: context.carry,
                    time_to_expiry: context.time_to_expiry,
                    kind: context.kind,
                },
                volatility: **vol_a,
            })
            .delta;
            let delta_b = model_greeks(OptionEvaluation {
                contract: OptionContract {
                    model: context.model,
                    underlying: context.reference_price,
                    strike: **strike_b,
                    rate: context.rate,
                    carry: context.carry,
                    time_to_expiry: context.time_to_expiry,
                    kind: context.kind,
                },
                volatility: **vol_b,
            })
            .delta;
            (delta_a - target_delta)
                .abs()
                .partial_cmp(&(delta_b - target_delta).abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(strike, _)| *strike)
}

#[cfg(test)]
mod tests {
    use super::{atm_index, label_moneyness, select_strike_by_delta, select_strike_by_offset};
    use crate::options::{ChainGreeksContext, OptionKind, PricingModel};

    #[test]
    fn atm_index_finds_nearest() {
        let strikes = [90.0, 100.0, 110.0];
        assert_eq!(atm_index(&strikes, 103.0), Some(1));
    }

    #[test]
    fn moneyness_labels_calls() {
        let strikes = [90.0, 100.0, 110.0];
        assert_eq!(
            label_moneyness(&strikes, 100.0, OptionKind::Call),
            vec![1, 0, -1]
        );
    }

    #[test]
    fn offset_selects_expected_strike() {
        let strikes = [90.0, 100.0, 110.0];
        assert_eq!(select_strike_by_offset(&strikes, 101.0, 1), Some(110.0));
    }

    #[test]
    fn delta_selection_returns_a_strike() {
        let strikes = [80.0, 90.0, 100.0, 110.0, 120.0];
        let vols = [0.28, 0.24, 0.20, 0.22, 0.26];
        let strike = select_strike_by_delta(
            &strikes,
            &vols,
            ChainGreeksContext {
                model: PricingModel::BlackScholes,
                reference_price: 100.0,
                rate: 0.01,
                carry: 0.0,
                time_to_expiry: 0.5,
                kind: OptionKind::Call,
            },
            0.25,
        );
        assert!(strike.is_some());
    }
}
