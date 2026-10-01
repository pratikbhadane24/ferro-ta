package ferrota

/*
#include "ferro_ta.h"
*/
import "C"

import "unsafe"

// inPtr passes a Go slice to C for the duration of one call. The slice holds
// no Go pointers, so this satisfies the cgo pointer-passing rules; the
// native side never retains it. Empty slices map to NULL, which the ABI
// accepts when len == 0.
func inPtr(s []float64) *C.double {
	if len(s) == 0 {
		return nil
	}
	return (*C.double)(unsafe.Pointer(&s[0]))
}

// outPtr is inPtr for caller-allocated output buffers of any element type.
func outPtr[T float64 | int32 | int64 | int8](s []T) unsafe.Pointer {
	if len(s) == 0 {
		return nil
	}
	return unsafe.Pointer(&s[0])
}
