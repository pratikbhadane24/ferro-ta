//! Parabolic SAR and SAREXT.

// ---------------------------------------------------------------------------
// SAR — Parabolic SAR
// ---------------------------------------------------------------------------

/// Parabolic SAR.
///
/// The recurrence is inherently serial with two data-dependent branches per
/// bar, so there is no algorithmic improvement available. The only thing done
/// here is to carry `high[i-1]`, `high[i-2]`, `low[i-1]` and `low[i-2]` in
/// registers instead of re-loading (and re-bounds-checking) them every
/// iteration. The `min`/`max` chains keep their original operand order, so the
/// output is bit-identical.
pub fn sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> Vec<f64> {
    let n = high.len();
    if n < 2 {
        return vec![f64::NAN; n];
    }
    let mut is_rising = high[1] >= high[0];
    let mut af = acceleration;
    let (mut ep, mut sar_val) = if is_rising {
        (high[1], low[0])
    } else {
        (low[1], high[0])
    };

    // Indexed stores over a pre-filled `NaN` buffer rather than `push`. Only
    // the output write moved: the carried `high[i-1]`/`low[i-2]` registers
    // below are untouched, and the `min`/`max` operand order is unchanged.
    let mut result = vec![f64::NAN; n];
    result[1] = sar_val;
    if n == 2 {
        return result;
    }

    // Carried predecessors, shifted at the end of each iteration.
    let (mut high_1, mut high_2) = (high[1], high[0]);
    let (mut low_1, mut low_2) = (low[1], low[0]);

    for i in 2..n {
        let high_i = high[i];
        let low_i = low[i];
        let prev_sar = sar_val;
        sar_val = prev_sar + af * (ep - prev_sar);
        if is_rising {
            sar_val = sar_val.min(low_1).min(low_2);
            if low_i < sar_val {
                is_rising = false;
                sar_val = ep;
                ep = low_i;
                af = acceleration;
            } else if high_i > ep {
                ep = high_i;
                af = (af + acceleration).min(maximum);
            }
        } else {
            sar_val = sar_val.max(high_1).max(high_2);
            if high_i > sar_val {
                is_rising = true;
                sar_val = ep;
                ep = high_i;
                af = acceleration;
            } else if low_i < ep {
                ep = low_i;
                af = (af + acceleration).min(maximum);
            }
        }
        result[i] = sar_val;
        high_2 = high_1;
        high_1 = high_i;
        low_2 = low_1;
        low_1 = low_i;
    }
    result
}

// ---------------------------------------------------------------------------
// SAREXT — Extended Parabolic SAR
// ---------------------------------------------------------------------------

/// Parabolic SAR Extended with configurable acceleration factors.
#[allow(clippy::too_many_arguments)]
pub fn sarext(
    high: &[f64],
    low: &[f64],
    startvalue: f64,
    offsetonreverse: f64,
    accelerationinitlong: f64,
    accelerationlong: f64,
    accelerationmaxlong: f64,
    accelerationinitshort: f64,
    accelerationshort: f64,
    accelerationmaxshort: f64,
) -> Vec<f64> {
    let n = high.len();
    if n < 2 {
        return vec![f64::NAN; n];
    }
    let mut is_rising = high[1] >= high[0];
    let (mut af, mut af_step_cur, mut af_max_cur) = if is_rising {
        (accelerationinitlong, accelerationlong, accelerationmaxlong)
    } else {
        (
            accelerationinitshort,
            accelerationshort,
            accelerationmaxshort,
        )
    };
    let (mut ep, mut sar_val) = if is_rising {
        (
            high[1],
            if startvalue != 0.0 {
                startvalue
            } else {
                low[0]
            },
        )
    } else {
        (
            low[1],
            if startvalue != 0.0 {
                -startvalue
            } else {
                high[0]
            },
        )
    };
    let mut result = vec![f64::NAN; n];
    result[1] = sar_val;
    for i in 2..n {
        let prev_sar = sar_val;
        sar_val = prev_sar + af * (ep - prev_sar);
        if is_rising {
            sar_val = sar_val.min(low[i - 1]).min(low[i - 2]);
            if low[i] < sar_val {
                is_rising = false;
                sar_val = ep + sar_val.abs() * offsetonreverse;
                ep = low[i];
                af = accelerationinitshort;
                af_step_cur = accelerationshort;
                af_max_cur = accelerationmaxshort;
            } else if high[i] > ep {
                ep = high[i];
                af = (af + af_step_cur).min(af_max_cur);
            }
        } else {
            sar_val = sar_val.max(high[i - 1]).max(high[i - 2]);
            if high[i] > sar_val {
                is_rising = true;
                sar_val = ep - sar_val.abs() * offsetonreverse;
                ep = high[i];
                af = accelerationinitlong;
                af_step_cur = accelerationlong;
                af_max_cur = accelerationmaxlong;
            } else if low[i] < ep {
                ep = low[i];
                af = (af + af_step_cur).min(af_max_cur);
            }
        }
        result[i] = sar_val;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::overlap::test_support::*;

    fn reference_sar(high: &[f64], low: &[f64], acceleration: f64, maximum: f64) -> Vec<f64> {
        let n = high.len();
        if n < 2 {
            return vec![f64::NAN; n];
        }
        let mut result = vec![f64::NAN; n];
        let mut is_rising = high[1] >= high[0];
        let mut af = acceleration;
        let (mut ep, mut sar_val) = if is_rising {
            (high[1], low[0])
        } else {
            (low[1], high[0])
        };
        result[1] = sar_val;
        for i in 2..n {
            let prev_sar = sar_val;
            sar_val = prev_sar + af * (ep - prev_sar);
            if is_rising {
                sar_val = sar_val.min(low[i - 1]).min(low[i - 2]);
                if low[i] < sar_val {
                    is_rising = false;
                    sar_val = ep;
                    ep = low[i];
                    af = acceleration;
                } else if high[i] > ep {
                    ep = high[i];
                    af = (af + acceleration).min(maximum);
                }
            } else {
                sar_val = sar_val.max(high[i - 1]).max(high[i - 2]);
                if high[i] > sar_val {
                    is_rising = true;
                    sar_val = ep;
                    ep = high[i];
                    af = acceleration;
                } else if low[i] < ep {
                    ep = low[i];
                    af = (af + acceleration).min(maximum);
                }
            }
            result[i] = sar_val;
        }
        result
    }

    // -- SAR / SAREXT ------------------------------------------------------

    #[test]
    fn sar_matches_reference_bitwise() {
        for n in [0usize, 1, 2, 3, 4, 17, 5000] {
            let (high, low) = synthetic_hl(n);
            let got = sar(&high, &low, 0.02, 0.2);
            let want = reference_sar(&high, &low, 0.02, 0.2);
            assert_bits_eq(&got, &want, &format!("sar n={n}"));
        }
        // A descending open (is_rising == false at bar 1) takes the other seed.
        let high: Vec<f64> = (0..64).map(|i| 100.0 - i as f64).collect();
        let low: Vec<f64> = high.iter().map(|h| h - 1.0).collect();
        assert_bits_eq(
            &sar(&high, &low, 0.02, 0.2),
            &reference_sar(&high, &low, 0.02, 0.2),
            "sar falling",
        );
    }

    #[test]
    fn sarext_reduces_to_sar_bitwise() {
        // With `startvalue = 0`, `offsetonreverse = 0` and one shared
        // acceleration triple, SAREXT is definitionally SAR. This pins the
        // SAREXT output-construction change against the bitwise-verified SAR.
        let (high, low) = synthetic_hl(5000);
        let got = sarext(&high, &low, 0.0, 0.0, 0.02, 0.02, 0.2, 0.02, 0.02, 0.2);
        let want = sar(&high, &low, 0.02, 0.2);
        assert_bits_eq(&got, &want, "sarext == sar");
    }
}
