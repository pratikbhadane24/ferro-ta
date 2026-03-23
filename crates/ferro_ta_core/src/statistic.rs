//! Statistic functions.

/// Standard deviation — population (`ddof = 0`).
pub fn stddev(real: &[f64], timeperiod: usize, nbdev: f64) -> Vec<f64> {
    let n = real.len();
    let mut result = vec![f64::NAN; n];
    if timeperiod < 1 || n < timeperiod {
        return result;
    }
    for i in (timeperiod - 1)..n {
        let window = &real[i + 1 - timeperiod..=i];
        let mean: f64 = window.iter().sum::<f64>() / timeperiod as f64;
        let var: f64 = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / timeperiod as f64;
        result[i] = var.sqrt() * nbdev;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stddev_constant() {
        let prices = vec![5.0; 5];
        let result = stddev(&prices, 3, 1.0);
        for v in result.iter().filter(|v| !v.is_nan()) {
            assert!(v.abs() < 1e-10);
        }
    }
}
