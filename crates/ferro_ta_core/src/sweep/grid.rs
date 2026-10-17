//! Parameter-grid helpers: Cartesian expansion and plateau-robust scoring.

/// Largest grid [`expand_grid`] will materialise.
pub const MAX_GRID_COMBOS: usize = 1_000_000;

/// Cartesian product of `axes`, one row per combination.
///
/// The last axis varies fastest (same order as Python's
/// `itertools.product`). Returns a row-major `(n_combos, n_axes)` buffer.
pub fn expand_grid(axes: &[Vec<f64>]) -> Result<Vec<f64>, String> {
    if axes.is_empty() {
        return Err("grid must have at least one axis".to_string());
    }
    if let Some(k) = axes.iter().position(|a| a.is_empty()) {
        return Err(format!("grid axis {k} has no values"));
    }
    let n_combos = axes
        .iter()
        .try_fold(1usize, |acc, a| acc.checked_mul(a.len()))
        .filter(|&n| n <= MAX_GRID_COMBOS)
        .ok_or_else(|| format!("grid exceeds {MAX_GRID_COMBOS} combinations"))?;
    let lens: Vec<usize> = axes.iter().map(Vec::len).collect();
    Ok((0..n_combos)
        .flat_map(|c| {
            grid_index(c, &lens)
                .into_iter()
                .zip(axes)
                .map(|(i, axis)| axis[i])
                .collect::<Vec<_>>()
        })
        .collect())
}

/// Per-axis indices of combination `c` in a grid with axis lengths `lens`
/// (last axis fastest).
pub fn grid_index(c: usize, lens: &[usize]) -> Vec<usize> {
    let mut rest = c;
    let mut idx = vec![0; lens.len()];
    for (k, &len) in lens.iter().enumerate().rev() {
        idx[k] = rest % len;
        rest /= len;
    }
    idx
}

/// Mean of `values` over each combination's grid neighbourhood.
///
/// `coords` is a row-major `(n_combos, n_axes)` matrix of per-axis grid
/// indices. Neighbours are combinations within `radius` steps on every axis
/// (Chebyshev distance), including the combination itself. NaN values are
/// ignored; a neighbourhood that is all NaN scores NaN.
pub fn plateau_scores(
    values: &[f64],
    coords: &[usize],
    n_axes: usize,
    radius: usize,
) -> Result<Vec<f64>, String> {
    if n_axes == 0 {
        return Err("n_axes must be >= 1".to_string());
    }
    if coords.len() != values.len() * n_axes {
        return Err(format!(
            "coords length ({}) must be values length ({}) * n_axes ({n_axes})",
            coords.len(),
            values.len()
        ));
    }
    let rows: Vec<&[usize]> = coords.chunks_exact(n_axes).collect();
    Ok(rows
        .iter()
        .map(|here| {
            let (sum, count) = rows
                .iter()
                .zip(values)
                .filter(|(there, v)| !v.is_nan() && within(here, there, radius))
                .fold((0.0, 0usize), |(s, c), (_, &v)| (s + v, c + 1));
            if count == 0 {
                f64::NAN
            } else {
                sum / count as f64
            }
        })
        .collect())
}

fn within(a: &[usize], b: &[usize], radius: usize) -> bool {
    a.iter().zip(b).all(|(&x, &y)| x.abs_diff(y) <= radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expand_grid_last_axis_fastest() {
        let g = expand_grid(&[vec![1.0, 2.0], vec![10.0, 20.0, 30.0]]).unwrap();
        assert_eq!(
            g,
            vec![1.0, 10.0, 1.0, 20.0, 1.0, 30.0, 2.0, 10.0, 2.0, 20.0, 2.0, 30.0]
        );
    }

    #[test]
    fn expand_grid_rejects_empty_and_huge() {
        assert!(expand_grid(&[]).is_err());
        assert!(expand_grid(&[vec![1.0], vec![]]).is_err());
        let big = vec![0.0; 1001];
        assert!(expand_grid(&[big.clone(), big]).is_err());
    }

    #[test]
    fn grid_index_roundtrip() {
        assert_eq!(grid_index(0, &[2, 3]), vec![0, 0]);
        assert_eq!(grid_index(4, &[2, 3]), vec![1, 1]);
        assert_eq!(grid_index(5, &[2, 3]), vec![1, 2]);
    }

    #[test]
    fn plateau_prefers_broad_region_over_spike() {
        // 1-D grid: spike at 0, plateau around 3.
        let values = [10.0, -5.0, 3.0, 3.0, 3.0, 0.0];
        let coords: Vec<usize> = (0..6).collect();
        let s = plateau_scores(&values, &coords, 1, 1).unwrap();
        let best = s
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .unwrap()
            .0;
        assert_eq!(best, 3);
        assert_eq!(s[0], 2.5);
    }

    #[test]
    fn plateau_radius_zero_is_identity_and_nan_aware() {
        let values = [1.0, f64::NAN, 3.0];
        let coords = [0, 1, 2];
        let s0 = plateau_scores(&values, &coords, 1, 0).unwrap();
        assert_eq!(s0[0], 1.0);
        assert!(s0[1].is_nan());
        let s1 = plateau_scores(&values, &coords, 1, 1).unwrap();
        assert_eq!(s1[1], 2.0);
    }

    #[test]
    fn plateau_validates_shapes() {
        assert!(plateau_scores(&[1.0, 2.0], &[0, 1, 2], 1, 1).is_err());
        assert!(plateau_scores(&[1.0], &[0], 0, 1).is_err());
    }
}
