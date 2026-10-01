# ferrota — Go bindings for ferro-ta

[![Go Reference](https://pkg.go.dev/badge/github.com/pratikbhadane24/ferro-ta/bindings/go.svg)](https://pkg.go.dev/github.com/pratikbhadane24/ferro-ta/bindings/go)

Go bindings for [ferro-ta](https://github.com/pratikbhadane24/ferro-ta), a Rust
technical analysis library with TA-Lib compatible indicators. Every function
calls the Rust core through its C ABI (`crates/ferro_ta_ffi`) via cgo; nothing
is reimplemented in Go.

```bash
go get github.com/pratikbhadane24/ferro-ta/bindings/go
```

Release tags ship prebuilt static libraries for linux/darwin (amd64, arm64)
and windows (amd64), so no Rust toolchain is needed. cgo must be enabled (on
Windows, install a MinGW-w64 `gcc`).

```go
import ferrota "github.com/pratikbhadane24/ferro-ta/bindings/go"

sma, err := ferrota.Sma(closes, 20)
macd, signal, hist, err := ferrota.Macd(closes, 12, 26, 9)
upper, middle, lower, err := ferrota.Bbands(closes, 20, 2, 2, ferrota.MATypeSMA)

rsi, err := ferrota.NewStreamRsi(14) // bar-by-bar
defer rsi.Close()
value, err := rsi.Update(lastPrice)
```

- 229 array functions (all TA indicators, candlestick patterns, IV, realized
  volatility, regime and portfolio series), 24 scalar functions (option
  pricing and greeks, implied volatility, futures, Kelly sizing) and 9
  streaming indicators.
- Outputs match input length with NaN warm-up; errors are `*ferrota.Error`,
  comparable with `errors.Is` (`ErrInvalidParam`, `ErrLengthMismatch`, ...).

Full guide: [docs/languages/go.rst](https://github.com/pratikbhadane24/ferro-ta/blob/main/docs/languages/go.rst).
License: MIT.
