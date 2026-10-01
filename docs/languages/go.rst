Go
==

Go module ``github.com/pratikbhadane24/ferro-ta/bindings/go`` (package
``ferrota``) wraps ``ferro_ta_core`` through the shared C ABI crate
``ferro_ta_ffi`` via cgo. No indicator is reimplemented in Go.

Release tags ship **prebuilt static archives** for Linux and macOS
(amd64/arm64) and Windows (amd64), so ``go get`` needs no Rust toolchain. cgo
must be enabled (the default when a C compiler is installed; on Windows use a
MinGW-w64 ``gcc``).

Installation
------------

.. code-block:: bash

   go get github.com/pratikbhadane24/ferro-ta/bindings/go@latest

Quick start
-----------

.. code-block:: go

   package main

   import (
   	"fmt"

   	ferrota "github.com/pratikbhadane24/ferro-ta/bindings/go"
   )

   func main() {
   	closes := []float64{44.3, 44.1, 44.6, 45.2, 45.6, 45.1, 46.0, 46.4}

   	sma, err := ferrota.Sma(closes, 3)
   	if err != nil {
   		panic(err)
   	}
   	upper, middle, lower, err := ferrota.Bbands(closes, 5, 2, 2, ferrota.MATypeSMA)
   	if err != nil {
   		panic(err)
   	}
   	fmt.Println(sma, upper, middle, lower)
   }

API shape
---------

- **Names** are the core function names in Go CamelCase: ``sma`` → ``Sma``,
  ``linearreg_slope`` → ``LinearregSlope``, ``cdl3blackcrows`` →
  ``Cdl3blackcrows``.
- **Arrays** are ``[]float64`` in; outputs have the same length with NaN for
  warm-up bars. Candle patterns return ``[]int32``; some regime/alert helpers
  return ``[]int8``.
- **Multiple outputs** are named results:
  ``Macd(...) (macd, signal, hist []float64, err error)``.
- **Errors** are ``*ferrota.Error``; compare with ``errors.Is`` against
  ``ErrInvalidParam`` (period below minimum, MA type outside 0–8, non-finite
  float parameter, invalid enum), ``ErrLengthMismatch`` (series of different
  lengths), or ``ErrPanic`` (internal error, contained).
- **Typed constants**: ``MAType`` (``MATypeSMA`` … ``MATypeT3``),
  ``OptionKind`` (``OptionCall``, ``OptionPut``), ``DigitalKind``,
  ``PricingModel``.
- **Concurrency**: every function is safe to call from many goroutines.

Streaming
---------

Stateful bar-by-bar indicators wrap a Rust handle. Methods are safe for
concurrent use; call ``Close`` when done (a finalizer is only a backstop).

.. code-block:: go

   rsi, err := ferrota.NewStreamRsi(14)
   if err != nil {
   	panic(err)
   }
   defer rsi.Close()
   for _, price := range closes {
   	value, err := rsi.Update(price) // NaN during warm-up
   	if err != nil {
   		panic(err)
   	}
   	_ = value
   }

Available: ``StreamSma``, ``StreamEma``, ``StreamRsi``, ``StreamAtr``,
``StreamBbands``, ``StreamMacd``, ``StreamStoch``, ``StreamVwap``,
``StreamSupertrend``.

Options, futures and sizing
---------------------------

.. code-block:: go

   price, _ := ferrota.BlackScholesPrice(100, 105, 0.03, 0.01, 0.5, 0.25, ferrota.OptionCall)
   delta, gamma, vega, theta, rho, _ := ferrota.BlackScholesGreeks(
   	100, 105, 0.03, 0.01, 0.5, 0.25, ferrota.OptionCall)
   iv, _ := ferrota.ImpliedVolatility(price, ferrota.ModelBlackScholes,
   	100, 105, 0.03, 0.02, 0.5, ferrota.OptionCall, 0.2, 1e-8, 100)

Not exported yet
----------------

The C ABI covers same-length array functions, scalar functions and streaming
handles. These core APIs need a different ABI pattern (variable-length or
2-D results, nested structs) and are not in the Go module yet: backtest
engines and performance metrics, batch/2-D portfolio functions, bar
resampling and tick aggregation, volatility cones and smile metrics, option
chain/strategy helpers, chunked processing helpers, ``pivot_points``,
``dtw_*``, ``calendar_spreads``. :doc:`coverage` lists every function per
language.

Performance
-----------

Each call crosses cgo once (about 30 ns), then the Rust kernel runs on the
whole slice. Results are copied into Go-allocated slices, which costs one
extra pass per output: on 1M bars, RSI is within ~6% of calling the Rust core
directly, SMA ~12%, and three-output BBANDS ~35%.

Building from source
--------------------

On a repository checkout, ``lib/`` is not committed. Build the archive for
your machine, then use the module normally:

.. code-block:: bash

   make go-lib              # cargo build -p ferro_ta_ffi --release + copy
   cd bindings/go && go test ./...

Development
-----------

The Go wrappers, the C header and ``ffi_spec.json`` are generated from the
Rust export declarations; never edit ``*_gen.go`` by hand.

.. code-block:: bash

   make ffi-gen     # regenerate after changing crates/ferro_ta_ffi
   make ffi-check   # full gate: Rust tests, clippy, codegen freshness, C, Go

Golden fixtures in ``tests/fixtures/golden/ffi_golden.json`` are replayed
through every Go wrapper; regenerate them with
``python3 scripts/build_golden_fixtures.py`` after a core change that alters
numbers.
