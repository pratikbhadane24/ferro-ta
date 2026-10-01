package ferrota

/*
#include "ferro_ta.h"
*/
import "C"

// nativeVersion reports the version of the linked native library, which must
// match Version (checked by TestVersion).
func nativeVersion() string {
	return C.GoString(C.ft_version())
}
