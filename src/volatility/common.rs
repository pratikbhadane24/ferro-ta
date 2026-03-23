/// Compute True Range for all bars.
/// Bar 0 uses H-L; subsequent bars use TA-Lib formula.
pub(super) fn compute_tr(highs: &[f64], lows: &[f64], closes: &[f64]) -> Vec<f64> {
    let n = highs.len();
    let mut tr = vec![0.0_f64; n];
    if n == 0 {
        return tr;
    }
    tr[0] = highs[0] - lows[0];
    for i in 1..n {
        let hl = highs[i] - lows[i];
        let hpc = (highs[i] - closes[i - 1]).abs();
        let lpc = (lows[i] - closes[i - 1]).abs();
        tr[i] = hl.max(hpc).max(lpc);
    }
    tr
}
