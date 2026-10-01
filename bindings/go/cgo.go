package ferrota

// The native library is a Rust static archive (crates/ferro_ta_ffi). Release
// tags of this module (bindings/go/vX.Y.Z) ship prebuilt archives under lib/,
// so `go get` needs no Rust toolchain. On a source checkout, build the archive
// for the host first with `make go-lib`.
//
// The system libraries after each archive are exactly what rustc reports via
// `--print native-static-libs` for that target (CI logs it on every run).

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo linux,amd64 LDFLAGS: ${SRCDIR}/lib/linux_amd64/libferro_ta_ffi.a -lgcc_s -lutil -lrt -lpthread -lm -ldl
#cgo linux,arm64 LDFLAGS: ${SRCDIR}/lib/linux_arm64/libferro_ta_ffi.a -lgcc_s -lutil -lrt -lpthread -lm -ldl
#cgo darwin,amd64 LDFLAGS: ${SRCDIR}/lib/darwin_amd64/libferro_ta_ffi.a -lSystem -lc -lm
#cgo darwin,arm64 LDFLAGS: ${SRCDIR}/lib/darwin_arm64/libferro_ta_ffi.a -lSystem -lc -lm
#cgo windows,amd64 LDFLAGS: ${SRCDIR}/lib/windows_amd64/libferro_ta_ffi.a -lkernel32 -lntdll -luserenv -lws2_32 -ldbghelp
#include "ferro_ta.h"
*/
import "C"
