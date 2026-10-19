//! `Windowed` — a streaming indicator that replays a batch kernel over the
//! last `lookback` bars.
//!
//! For any kernel whose value at bar `i` depends only on bars
//! `i + 1 - lookback ..= i`, the last element of the batch result on that
//! window *is* the batch value at `i`. While fewer than `lookback` bars have
//! arrived the kernel sees the whole history, so warm-up NaNs land exactly
//! where the batch function puts them.
//!
//! Each input keeps its own contiguous buffer of at most `2 · lookback` bars;
//! when full, the oldest `lookback` are drained in one move, so the cost per
//! bar stays amortised O(1) on top of the O(lookback) kernel call.

use super::{validate_buffer_len, StreamValue, StreamingError, StreamingIndicator};
use std::marker::PhantomData;

/// Most inputs a windowed stream may take.
pub const MAX_WINDOW_INPUTS: usize = 8;

/// A streaming input that splits into one `f64` per buffer column.
pub trait WindowInput: StreamValue {
    /// Number of input columns.
    const WIDTH: usize;
    /// Append each component to its column (`cols.len() == WIDTH`).
    fn push_to(self, cols: &mut [Vec<f64>]);
    /// Component `k` (`k < WIDTH`).
    fn component(&self, k: usize) -> f64;
}

impl WindowInput for f64 {
    const WIDTH: usize = 1;
    fn push_to(self, cols: &mut [Vec<f64>]) {
        cols[0].push(self);
    }
    fn component(&self, _k: usize) -> f64 {
        *self
    }
}

macro_rules! impl_window_input_tuple {
    ($width:literal: $($v:ident $k:tt),+) => {
        impl WindowInput for ($(impl_window_input_tuple!(@f64 $v)),+) {
            const WIDTH: usize = $width;
            fn push_to(self, cols: &mut [Vec<f64>]) {
                $( cols[$k].push(self.$k); )+
            }
            fn component(&self, k: usize) -> f64 {
                match k {
                    $( $k => self.$k, )+
                    _ => unreachable!("input component {k} of {}", $width),
                }
            }
        }
    };
    (@f64 $v:ident) => { f64 };
}

impl_window_input_tuple!(2: a 0, b 1);
impl_window_input_tuple!(3: a 0, b 1, c 2);
impl_window_input_tuple!(4: a 0, b 1, c 2, d 3);

/// A batch result read bar by bar as a streaming output `O`.
pub trait BatchOutput<O> {
    /// Number of bars in the result.
    fn bars(&self) -> usize;
    /// Output at bar `i`.
    fn at(&self, i: usize) -> O;
    /// Output at the last bar.
    fn last_bar(&self) -> O {
        self.at(self.bars() - 1)
    }
}

impl<T: Copy> BatchOutput<T> for Vec<T> {
    fn bars(&self) -> usize {
        self.len()
    }
    fn at(&self, i: usize) -> T {
        self[i]
    }
}

macro_rules! impl_batch_output_tuple {
    ($($k:tt),+) => {
        impl BatchOutput<($(impl_batch_output_tuple!(@f64 $k)),+)>
            for ($(impl_batch_output_tuple!(@vec $k)),+)
        {
            fn bars(&self) -> usize {
                self.0.len()
            }
            fn at(&self, i: usize) -> ($(impl_batch_output_tuple!(@f64 $k)),+) {
                ($(self.$k[i]),+)
            }
        }
    };
    (@f64 $k:tt) => { f64 };
    (@vec $k:tt) => { Vec<f64> };
}

impl_batch_output_tuple!(0, 1);
impl_batch_output_tuple!(0, 1, 2);
impl_batch_output_tuple!(0, 1, 2, 3);

/// Parameter struct of a [`WindowKernel`] (serializable with `serde`).
#[cfg(feature = "serde")]
pub trait KernelParams:
    Clone + std::fmt::Debug + serde::Serialize + serde::de::DeserializeOwned
{
}
#[cfg(feature = "serde")]
impl<T: Clone + std::fmt::Debug + serde::Serialize + serde::de::DeserializeOwned> KernelParams
    for T
{
}
/// Parameter struct of a [`WindowKernel`].
#[cfg(not(feature = "serde"))]
pub trait KernelParams: Clone + std::fmt::Debug {}
#[cfg(not(feature = "serde"))]
impl<T: Clone + std::fmt::Debug> KernelParams for T {}

/// A batch kernel replayed by [`Windowed`]. Registry `windowed` entries
/// generate the implementation; see `streaming::registry`.
pub trait WindowKernel: Clone + std::fmt::Debug {
    /// Snapshot type tag (the stream name).
    const KIND: &'static str;
    type Params: KernelParams;
    type Input: WindowInput;
    type Output: StreamValue;
    /// Bars the value at the last bar depends on (`>= 1`).
    fn lookback(params: &Self::Params) -> usize;
    /// Reject invalid parameters.
    fn validate(params: &Self::Params) -> Result<(), StreamingError>;
    /// Batch value at the last bar of `window` (one slice per input).
    fn last(params: &Self::Params, window: &[&[f64]]) -> Self::Output;
}

/// The window a preview sees, column after column in one buffer: each
/// column's last `lookback - 1` committed bars plus that column's `parts`
/// value. Returns the buffer and the per-column length. Not generic, so the
/// ~150 windowed streams share one copy of it.
fn peek_buffer(cols: &[Vec<f64>], lookback: usize, parts: &[f64]) -> (Vec<f64>, usize) {
    let len = cols[0].len();
    let start = (len + 1).saturating_sub(lookback);
    let span = len - start + 1;
    let mut flat = Vec::with_capacity(span * cols.len());
    for (col, &part) in cols.iter().zip(parts) {
        flat.extend_from_slice(&col[start..]);
        flat.push(part);
    }
    (flat, span)
}

/// `width` consecutive `span`-long slices of `flat`, as kernel inputs.
fn window_slots(flat: &[f64], span: usize, width: usize) -> [&[f64]; MAX_WINDOW_INPUTS] {
    let mut window: [&[f64]; MAX_WINDOW_INPUTS] = [&[]; MAX_WINDOW_INPUTS];
    for (slot, chunk) in window.iter_mut().zip(flat.chunks_exact(span)).take(width) {
        *slot = chunk;
    }
    window
}

/// Streaming adapter over a [`WindowKernel`].
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "serde",
    serde(bound(
        serialize = "K::Params: serde::Serialize",
        deserialize = "K::Params: serde::de::DeserializeOwned"
    ))
)]
pub struct Windowed<K: WindowKernel> {
    params: K::Params,
    /// One buffer per input, all the same length, at most `2 · lookback`.
    cols: Vec<Vec<f64>>,
    #[cfg_attr(feature = "serde", serde(skip))]
    kernel: PhantomData<K>,
}

impl<K: WindowKernel> Windowed<K> {
    /// Build from validated parameters. Registry types also get a
    /// positional `new(params...)`.
    pub fn from_params(params: K::Params) -> Result<Self, StreamingError> {
        K::validate(&params)?;
        if K::lookback(&params) == 0 {
            return Err(StreamingError(format!(
                "{}: lookback must be >= 1",
                K::KIND
            )));
        }
        Ok(Self {
            params,
            cols: vec![Vec::new(); K::Input::WIDTH],
            kernel: PhantomData,
        })
    }

    pub fn params(&self) -> &K::Params {
        &self.params
    }

    /// Commit one bar and return the kernel value for it.
    pub fn update(&mut self, input: K::Input) -> K::Output {
        let lookback = K::lookback(&self.params);
        if self.cols[0].len() >= lookback.saturating_mul(2) {
            for col in &mut self.cols {
                col.drain(..col.len() - lookback);
            }
        }
        input.push_to(&mut self.cols);
        let start = self.cols[0].len().saturating_sub(lookback);
        let mut window: [&[f64]; MAX_WINDOW_INPUTS] = [&[]; MAX_WINDOW_INPUTS];
        for (slot, col) in window.iter_mut().zip(&self.cols) {
            *slot = &col[start..];
        }
        K::last(&self.params, &window[..self.cols.len()])
    }

    /// Value `update(input)` would return, without changing state: the
    /// kernel sees the same last `lookback` bars as on commit, copied into
    /// one scratch buffer (a clone would copy up to `2 · lookback` bars per
    /// input and then reallocate on the push).
    pub fn peek(&self, input: K::Input) -> K::Output {
        let mut parts = [0.0; MAX_WINDOW_INPUTS];
        for (k, part) in parts.iter_mut().enumerate().take(self.cols.len()) {
            *part = input.component(k);
        }
        let width = self.cols.len();
        let (flat, span) = peek_buffer(&self.cols, K::lookback(&self.params), &parts[..width]);
        let window = window_slots(&flat, span, width);
        K::last(&self.params, &window[..width])
    }

    pub fn reset(&mut self) {
        for col in &mut self.cols {
            col.clear();
        }
    }

    /// Re-check parameters and buffer invariants (used when restoring).
    pub fn validate(&self) -> Result<(), StreamingError> {
        K::validate(&self.params)?;
        let lookback = K::lookback(&self.params);
        if lookback == 0 || self.cols.len() != K::Input::WIDTH {
            return Err(StreamingError(format!("{}: malformed window", K::KIND)));
        }
        let len = self.cols[0].len();
        if self.cols.iter().any(|c| c.len() != len) {
            return Err(StreamingError(format!(
                "{}: input buffers differ in length",
                K::KIND
            )));
        }
        validate_buffer_len(len, lookback.saturating_mul(2), "window buffer")
    }
}

impl<K: WindowKernel> StreamingIndicator for Windowed<K> {
    const KIND: &'static str = K::KIND;
    type Input = K::Input;
    type Output = K::Output;

    fn update(&mut self, input: K::Input) -> K::Output {
        Windowed::update(self, input)
    }

    fn reset(&mut self) {
        Windowed::reset(self)
    }

    fn validate(&self) -> Result<(), StreamingError> {
        Windowed::validate(self)
    }

    fn peek(&self, input: K::Input) -> K::Output {
        Windowed::peek(self, input)
    }
}
