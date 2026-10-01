// Package ferrota provides Go bindings for ferro-ta, a Rust technical
// analysis library with TA-Lib compatible indicators.
//
// Every function wraps ferro_ta_core through its C ABI (crates/ferro_ta_ffi)
// via cgo; no indicator is reimplemented in Go.
//
// Conventions:
//   - Outputs have the same length as the inputs; leading warm-up values are NaN.
//   - Functions taking several series (high, low, close, ...) require equal lengths
//     and return ErrLengthMismatch otherwise.
//   - Periods below their minimum, MA types outside 0-8 and non-finite float
//     parameters return ErrInvalidParam.
//   - Streaming indicators (StreamSma, StreamRsi, ...) process one bar at a
//     time and must be closed with Close.
package ferrota
