//! Chunked / out-of-core execution helpers.
//!
//! - `trim_overlap`       — remove the first N elements from a slice
//! - `stitch_chunks`      — concatenate trimmed chunk results
//! - `make_chunk_ranges`  — compute (start, end) index pairs for chunked processing
//! - `forward_fill_nan`   — forward-fill NaN values

/// Remove the first `overlap` elements from a slice.
pub fn trim_overlap(chunk_out: &[f64], overlap: usize) -> Vec<f64> {
    if overlap > chunk_out.len() {
        return vec![];
    }
    chunk_out[overlap..].to_vec()
}

/// Concatenate a list of slices into a single Vec.
pub fn stitch_chunks(chunks: &[&[f64]]) -> Vec<f64> {
    let mut out = Vec::new();
    for &chunk in chunks {
        out.extend_from_slice(chunk);
    }
    out
}

/// Compute (start, end) index pairs for chunked processing.
///
/// Returns a flat Vec of pairs: [start0, end0, start1, end1, ...].
/// `chunk_size` is the desired output bars per chunk, `overlap` is the warm-up prefix.
pub fn make_chunk_ranges(n: usize, chunk_size: usize, overlap: usize) -> Vec<i64> {
    if chunk_size == 0 || n == 0 {
        return vec![];
    }
    let mut ranges: Vec<i64> = Vec::new();
    let mut start: usize = 0;
    loop {
        let end = (start + chunk_size + overlap).min(n);
        ranges.push(start as i64);
        ranges.push(end as i64);
        if end >= n {
            break;
        }
        start = end.saturating_sub(overlap);
    }
    ranges
}

/// Forward-fill NaN values in a 1-D array.
/// Leading NaN values are preserved until the first non-NaN value appears.
pub fn forward_fill_nan(values: &[f64]) -> Vec<f64> {
    let mut out = Vec::with_capacity(values.len());
    let mut last = f64::NAN;
    for &value in values {
        if value.is_nan() {
            out.push(last);
        } else {
            last = value;
            out.push(value);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_trim_overlap() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let result = trim_overlap(&data, 2);
        assert_eq!(result, vec![3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_trim_overlap_zero() {
        let data = vec![1.0, 2.0, 3.0];
        assert_eq!(trim_overlap(&data, 0), data);
    }

    #[test]
    fn test_trim_overlap_exceeds() {
        let data = vec![1.0, 2.0];
        assert!(trim_overlap(&data, 5).is_empty());
    }

    #[test]
    fn test_stitch_chunks() {
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0, 5.0];
        let chunks: Vec<&[f64]> = vec![&a, &b];
        let result = stitch_chunks(&chunks);
        assert_eq!(result, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_make_chunk_ranges() {
        let ranges = make_chunk_ranges(10, 4, 2);
        // Expected: [0,6], [4,10]
        assert_eq!(ranges.len() % 2, 0);
        assert!(ranges.len() >= 4);
        assert_eq!(ranges[0], 0);
    }

    #[test]
    fn test_forward_fill_nan() {
        let data = vec![f64::NAN, 1.0, f64::NAN, f64::NAN, 2.0, f64::NAN];
        let result = forward_fill_nan(&data);
        assert!(result[0].is_nan()); // leading NaN preserved
        assert!((result[1] - 1.0).abs() < 1e-10);
        assert!((result[2] - 1.0).abs() < 1e-10); // filled
        assert!((result[3] - 1.0).abs() < 1e-10); // filled
        assert!((result[4] - 2.0).abs() < 1e-10);
        assert!((result[5] - 2.0).abs() < 1e-10); // filled
    }

    #[test]
    fn test_empty() {
        assert!(trim_overlap(&[], 0).is_empty());
        assert!(stitch_chunks(&[]).is_empty());
        assert!(make_chunk_ranges(0, 4, 2).is_empty());
        assert!(forward_fill_nan(&[]).is_empty());
    }
}
