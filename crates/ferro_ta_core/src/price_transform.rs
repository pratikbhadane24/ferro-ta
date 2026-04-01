//! Price transformations — synthesize OHLC arrays into single price arrays.

/// Average Price: (open + high + low + close) / 4.
pub fn avgprice(open: &[f64], high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    open.iter()
        .zip(high.iter())
        .zip(low.iter())
        .zip(close.iter())
        .map(|(((&o, &h), &l), &c)| (o + h + l + c) / 4.0)
        .collect()
}

/// Median Price: (high + low) / 2.
pub fn medprice(high: &[f64], low: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .map(|(&h, &l)| (h + l) / 2.0)
        .collect()
}

/// Typical Price: (high + low + close) / 3.
pub fn typprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c) / 3.0)
        .collect()
}

/// Weighted Close Price: (high + low + close * 2) / 4.
pub fn wclprice(high: &[f64], low: &[f64], close: &[f64]) -> Vec<f64> {
    high.iter()
        .zip(low.iter())
        .zip(close.iter())
        .map(|((&h, &l), &c)| (h + l + c * 2.0) / 4.0)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avgprice() {
        let o = vec![1.0, 2.0, 3.0];
        let h = vec![4.0, 5.0, 6.0];
        let l = vec![0.5, 1.5, 2.5];
        let c = vec![2.5, 3.5, 4.5];
        let result = avgprice(&o, &h, &l, &c);
        assert_eq!(result.len(), 3);
        assert!((result[0] - 2.0).abs() < 1e-10); // (1+4+0.5+2.5)/4 = 2.0
    }

    #[test]
    fn test_medprice() {
        let h = vec![10.0, 20.0];
        let l = vec![6.0, 12.0];
        let result = medprice(&h, &l);
        assert!((result[0] - 8.0).abs() < 1e-10);
        assert!((result[1] - 16.0).abs() < 1e-10);
    }

    #[test]
    fn test_typprice() {
        let h = vec![10.0];
        let l = vec![6.0];
        let c = vec![8.0];
        let result = typprice(&h, &l, &c);
        assert!((result[0] - 8.0).abs() < 1e-10); // (10+6+8)/3 = 8.0
    }

    #[test]
    fn test_wclprice() {
        let h = vec![10.0];
        let l = vec![6.0];
        let c = vec![8.0];
        let result = wclprice(&h, &l, &c);
        assert!((result[0] - 8.0).abs() < 1e-10); // (10+6+16)/4 = 8.0
    }

    #[test]
    fn test_empty_inputs() {
        let empty: Vec<f64> = vec![];
        assert!(avgprice(&empty, &empty, &empty, &empty).is_empty());
        assert!(medprice(&empty, &empty).is_empty());
        assert!(typprice(&empty, &empty, &empty).is_empty());
        assert!(wclprice(&empty, &empty, &empty).is_empty());
    }
}
