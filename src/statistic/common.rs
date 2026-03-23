/// Rolling linear regression: returns (slope, intercept) for the given window.
pub(super) fn linreg(window: &[f64]) -> (f64, f64) {
    let n = window.len() as f64;
    let sum_x: f64 = (0..window.len()).map(|i| i as f64).sum();
    let sum_y: f64 = window.iter().sum();
    let sum_xy: f64 = window.iter().enumerate().map(|(i, &y)| i as f64 * y).sum();
    let sum_x2: f64 = (0..window.len()).map(|i| (i as f64).powi(2)).sum();
    let denom = n * sum_x2 - sum_x * sum_x;
    let slope = if denom != 0.0 {
        (n * sum_xy - sum_x * sum_y) / denom
    } else {
        0.0
    };
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

pub(crate) fn rolling_linreg_apply<F>(prices: &[f64], timeperiod: usize, mut map: F) -> Vec<f64>
where
    F: FnMut(f64, f64) -> f64,
{
    let n = prices.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod == 0 || n < timeperiod {
        return result;
    }

    if prices.iter().any(|value| !value.is_finite()) {
        for end in (timeperiod - 1)..n {
            let window = &prices[(end + 1 - timeperiod)..=end];
            let (slope, intercept) = linreg(window);
            result[end] = map(slope, intercept);
        }
        return result;
    }

    let period = timeperiod as f64;
    let last_x = (timeperiod - 1) as f64;
    let sum_x = last_x * period / 2.0;
    let sum_x2 = last_x * period * (2.0 * period - 1.0) / 6.0;
    let denom = period * sum_x2 - sum_x * sum_x;

    let mut sum_y = prices[..timeperiod].iter().sum::<f64>();
    let mut sum_xy = prices[..timeperiod]
        .iter()
        .enumerate()
        .map(|(idx, &value)| idx as f64 * value)
        .sum::<f64>();

    for end in (timeperiod - 1)..n {
        let slope = if denom != 0.0 {
            (period * sum_xy - sum_x * sum_y) / denom
        } else {
            0.0
        };
        let intercept = (sum_y - slope * sum_x) / period;
        result[end] = map(slope, intercept);

        if end + 1 < n {
            let outgoing = prices[end + 1 - timeperiod];
            let incoming = prices[end + 1];
            let prev_sum_y = sum_y;

            sum_y = prev_sum_y - outgoing + incoming;
            sum_xy = sum_xy - (prev_sum_y - outgoing) + last_x * incoming;
        }
    }

    result
}
