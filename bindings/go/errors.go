package ferrota

/*
#include "ferro_ta.h"
*/
import "C"

import (
	"fmt"
	"strings"
)

// Error is returned by every function in this package when the native call
// fails. Compare with the sentinel errors using errors.Is.
type Error struct {
	Code int    // FT_ERR_* status code from the C ABI
	Op   string // Go function that failed, e.g. "Sma"
	Msg  string
}

func (e *Error) Error() string {
	if e.Op == "" {
		return "ferrota: " + e.Msg
	}
	return fmt.Sprintf("ferrota: %s: %s", e.Op, e.Msg)
}

// Is reports whether target is a sentinel with the same status code.
func (e *Error) Is(target error) bool {
	t, ok := target.(*Error)
	return ok && t.Op == "" && t.Code == e.Code
}

// Sentinel errors, one per C ABI status code.
var (
	ErrNullPointer    = &Error{Code: int(C.FT_ERR_NULL_PTR), Msg: "null pointer argument"}
	ErrInvalidParam   = &Error{Code: int(C.FT_ERR_INVALID_PARAM), Msg: "invalid parameter"}
	ErrLengthMismatch = &Error{Code: int(C.FT_ERR_LENGTH_MISMATCH), Msg: "input arrays must have the same length"}
	ErrPanic          = &Error{Code: int(C.FT_ERR_PANIC), Msg: "internal error in ferro_ta_core"}
	// ErrClosed is returned when a streaming indicator is used after Close.
	ErrClosed = &Error{Code: -1, Msg: "use of closed streaming indicator"}
)

func statusError(op string, status C.int32_t) error {
	if status == C.FT_OK {
		return nil
	}
	return &Error{Code: int(status), Op: op, Msg: C.GoString(C.ft_status_message(status))}
}

func closedError(op string) error {
	return &Error{Code: ErrClosed.Code, Op: op, Msg: ErrClosed.Msg}
}

// checkLengths enforces the C ABI's single shared length up front, so callers
// get a descriptive error instead of the native layer reading out of bounds.
func checkLengths(op string, names []string, lengths []int) error {
	for _, l := range lengths[1:] {
		if l != lengths[0] {
			parts := make([]string, len(names))
			for i, name := range names {
				parts[i] = fmt.Sprintf("%s=%d", name, lengths[i])
			}
			return &Error{
				Code: ErrLengthMismatch.Code,
				Op:   op,
				Msg:  "input arrays must have the same length (" + strings.Join(parts, ", ") + ")",
			}
		}
	}
	return nil
}
