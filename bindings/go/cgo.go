package ferrota

// The native library is a Rust static archive (crates/ferro_ta_ffi). Release
// tags of this module (bindings/go/vX.Y.Z) ship prebuilt archives under lib/,
// so `go get` needs no Rust toolchain. On a source checkout, build the archive
// for the host first with `make go-lib`.

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/lib/linux_amd64/libferro_ta_ffi.a -lm -ldl -lpthread -lrt -lgcc_s
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/lib/linux_arm64/libferro_ta_ffi.a -lm -ldl -lpthread -lrt -lgcc_s
#cgo darwin,amd64 LDFLAGS: ${SRCDIR}/lib/darwin_amd64/libferro_ta_ffi.a -lm
#cgo darwin,arm64 LDFLAGS: ${SRCDIR}/lib/darwin_arm64/libferro_ta_ffi.a -lm
#cgo windows,amd64 LDFLAGS: ${SRCDIR}/lib/windows_amd64/libferro_ta_ffi.a -lws2_32 -luserenv -lntdll -lbcrypt -ladvapi32 -lkernel32
#include "ferro_ta.h"
*/
import "C"
