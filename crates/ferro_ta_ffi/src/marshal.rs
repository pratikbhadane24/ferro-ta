//! Pointer → slice conversion and result copy-out shared by every export.

use crate::spec::ParamKind;
use crate::status::Status;

/// View a caller-owned input array as a slice.
///
/// `len == 0` yields an empty slice without touching `ptr`, so callers may pass
/// NULL for empty inputs.
///
/// # Safety
/// When `len > 0`, `ptr` must be valid for `len` reads of `f64` for the
/// lifetime of the returned slice.
pub unsafe fn input<'a>(ptr: *const f64, len: usize) -> Result<&'a [f64], Status> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null() {
        return Err(Status::NullPtr);
    }
    // SAFETY: non-null and valid for `len` reads per this function's contract.
    Ok(unsafe { std::slice::from_raw_parts(ptr, len) })
}

/// Reject NULL output pointers up front so nothing is computed (or partially
/// written) for a call that cannot succeed.
pub fn check_output<T>(ptr: *mut T, len: usize) -> Result<(), Status> {
    if len > 0 && ptr.is_null() {
        return Err(Status::NullPtr);
    }
    Ok(())
}

/// Validate a raw scalar parameter and convert it to the core's Rust type.
pub trait Param<Raw>: Sized {
    fn from_raw(raw: Raw, kind: ParamKind) -> Result<Self, Status>;
}

impl Param<i64> for usize {
    fn from_raw(raw: i64, kind: ParamKind) -> Result<Self, Status> {
        let min = match kind {
            ParamKind::Period { min } => min,
            ParamKind::Count => 0,
            ParamKind::MaType | ParamKind::Float | ParamKind::Enum(_) => {
                unreachable!("not an integer period")
            }
        };
        if raw < min {
            return Err(Status::InvalidParam);
        }
        usize::try_from(raw).map_err(|_| Status::InvalidParam)
    }
}

impl Param<i32> for u8 {
    fn from_raw(raw: i32, _kind: ParamKind) -> Result<Self, Status> {
        const MAX_MATYPE: i32 = 8;
        if !(0..=MAX_MATYPE).contains(&raw) {
            return Err(Status::InvalidParam);
        }
        Ok(raw as u8)
    }
}

impl Param<f64> for f64 {
    fn from_raw(raw: f64, _kind: ParamKind) -> Result<Self, Status> {
        crate::validation::check_float(raw)
    }
}

/// Copy a core result into caller-allocated output buffers.
///
/// Every output's length is checked against `len` before anything is written,
/// so a core function returning a different length (a core bug) yields
/// `Status::Internal` with the caller's buffers untouched, never a partially
/// initialised buffer reported as success.
///
/// `std::ptr::copy` (memmove semantics) is used rather than
/// `copy_nonoverlapping` so callers may pass an input buffer as the output
/// (in-place computation): inputs are no longer borrowed once the core
/// function has returned its owned `Vec`s.
pub trait WriteOutputs<P> {
    /// # Safety
    /// Each pointer in `ptrs` must be valid for `len` writes (or `len == 0`).
    unsafe fn write_to(self, ptrs: P, len: usize) -> Result<(), Status>;
}

/// # Safety
/// `dst` must be valid for `src.len()` writes.
unsafe fn copy_out<T: Copy>(src: &[T], dst: *mut T) {
    if !src.is_empty() {
        // SAFETY: `dst` valid for `src.len()` writes per caller contract.
        unsafe { std::ptr::copy(src.as_ptr(), dst, src.len()) };
    }
}

impl<A: Copy> WriteOutputs<(*mut A,)> for Vec<A> {
    unsafe fn write_to(self, ptrs: (*mut A,), len: usize) -> Result<(), Status> {
        if self.len() != len {
            return Err(Status::Internal);
        }
        // SAFETY: length verified; pointer validity per caller contract.
        unsafe { copy_out(&self, ptrs.0) };
        Ok(())
    }
}

macro_rules! impl_write_tuple {
    ($($t:ident $v:ident $p:ident),+) => {
        impl<$($t: Copy),+> WriteOutputs<($(*mut $t,)+)> for ($(Vec<$t>,)+) {
            unsafe fn write_to(self, ptrs: ($(*mut $t,)+), len: usize) -> Result<(), Status> {
                let ($($v,)+) = self;
                let ($($p,)+) = ptrs;
                if $( $v.len() != len )||+ {
                    return Err(Status::Internal);
                }
                // SAFETY: lengths verified; pointer validity per caller contract.
                $( unsafe { copy_out(&$v, $p) }; )+
                Ok(())
            }
        }
    };
}

impl_write_tuple!(A a pa, B b pb);
impl_write_tuple!(A a pa, B b pb, C c pc);
impl_write_tuple!(A a pa, B b pb, C c pc, D d pd);
impl_write_tuple!(A a pa, B b pb, C c pc, D d pd, E e pe);

/// Write a scalar result (single value or tuple) through output pointers.
pub trait WriteScalars<P> {
    /// # Safety
    /// Every pointer in `ptrs` must be valid for one write.
    unsafe fn write_to(self, ptrs: P);
}

impl WriteScalars<(*mut f64,)> for f64 {
    unsafe fn write_to(self, ptrs: (*mut f64,)) {
        // SAFETY: valid for one write per caller contract.
        unsafe { ptrs.0.write(self) };
    }
}

macro_rules! impl_write_scalars {
    ($($t:ident $v:ident $p:ident),+) => {
        impl<$($t),+> WriteScalars<($(*mut $t,)+)> for ($($t,)+) {
            unsafe fn write_to(self, ptrs: ($(*mut $t,)+)) {
                let ($($v,)+) = self;
                let ($($p,)+) = ptrs;
                // SAFETY: each pointer valid for one write per caller contract.
                $( unsafe { $p.write($v) }; )+
            }
        }
    };
}

impl_write_scalars!(A a pa, B b pb);
impl_write_scalars!(A a pa, B b pb, C c pc);
impl_write_scalars!(A a pa, B b pb, C c pc, D d pd);
impl_write_scalars!(A a pa, B b pb, C c pc, D d pd, E e pe);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_mismatch_is_internal_error_and_writes_nothing() {
        let mut a = [7.0; 4];
        let mut b = [7.0; 4];
        let result = unsafe {
            WriteOutputs::write_to(
                (vec![1.0; 4], vec![2.0; 3]),
                (a.as_mut_ptr(), b.as_mut_ptr()),
                4,
            )
        };
        assert_eq!(result, Err(Status::Internal));
        assert_eq!(a, [7.0; 4], "first output must stay untouched");
        assert_eq!(b, [7.0; 4]);
    }

    #[test]
    fn single_output_length_mismatch() {
        let mut a = [7.0; 4];
        let result = unsafe { WriteOutputs::write_to(vec![1.0; 5], (a.as_mut_ptr(),), 4) };
        assert_eq!(result, Err(Status::Internal));
        assert_eq!(a, [7.0; 4]);
    }
}
