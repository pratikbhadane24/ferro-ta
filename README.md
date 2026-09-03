<div align="center">

# ⚡ ferro-ta

## Rust-core technical analysis with first-class language bindings

**ferro-ta is a Rust-core technical analysis library with first-class bindings for Python, Rust, JavaScript (WASM), and Flutter.**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pratikbhadane24/ferro-ta/HEAD?labpath=examples%2Fquickstart.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pratikbhadane24/ferro-ta/blob/main/examples/quickstart.ipynb)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://pratikbhadane24.github.io/ferro-ta/)

</div>

---

> ferro-ta is a Rust-core technical analysis library with first-class bindings for Python, Rust, JavaScript (WASM), and Flutter.

Python, JavaScript (WASM), and Flutter wrap [`ferro_ta_core`](crates/ferro_ta_core); Rust uses that crate directly. New languages may only wrap the core — reimplementing indicators is out of scope. See [Adding a language](docs/languages/adding.rst).

Python is the most complete *ergonomic* surface (TA-Lib names, pandas/polars, Sphinx autodoc). It is not the only product.

## What ferro-ta is

| | TA-Lib | ferro-ta |
|---|---|---|
| **Core** | C implementations | Pure Rust [`ferro_ta_core`](crates/ferro_ta_core) |
| **Languages** | C API plus community wrappers | First-class Python, Rust, JavaScript (WASM), and Flutter |
| **API shape** | `talib.SMA(close, 20)` | Same indicators in each language (`SMA`, `sma`, `overlap::sma`) |
| **Installation** | Often requires a native/system toolchain | Pre-built packages on supported targets |
| **Scope** | Technical indicators | Technical indicators first; other tooling is optional |

## Install

| Language | Package | Install |
|---|---|---|
| Python | PyPI `ferro-ta` | `pip install ferro-ta` |
| Rust | crates.io `ferro_ta_core` | `cargo add ferro_ta_core` |
| JavaScript | npm `ferro-ta-wasm` | `npm install ferro-ta-wasm` |
| Flutter / Dart | pub.dev `ferro_ta` | `flutter pub add ferro_ta` |

Python extras:

```bash
pip install "ferro-ta[pandas]"   # pandas.Series support
pip install "ferro-ta[polars]"   # polars.Series support
pip install "ferro-ta[gpu]"      # PyTorch-backed GPU helpers
pip install "ferro-ta[options]"  # derivatives analytics helpers
pip install "ferro-ta[mcp]"      # MCP server for agent/tool clients
pip install "ferro-ta[all]"      # most optional extras (excluding gpu)
```

Language guides: [Python](docs/languages/python.rst) · [Rust](docs/languages/rust.rst) · [WASM](docs/languages/wasm.rst) · [Flutter](docs/languages/flutter.rst)

## Quick start

The same four indicators — SMA, RSI, MACD, BBANDS — on every binding.

### Python

```python
import numpy as np
from ferro_ta import SMA, RSI, MACD, BBANDS

close = np.linspace(44.0, 48.0, 40)
sma = SMA(close, timeperiod=5)
rsi = RSI(close, timeperiod=14)
macd_line, signal, histogram = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
upper, middle, lower = BBANDS(close, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
```

### Rust

```rust
use ferro_ta_core::{momentum, overlap};

fn main() {
    let close: Vec<f64> = (0..40).map(|i| 44.0 + i as f64 * 0.1).collect();
    let sma = overlap::sma(&close, 5);
    let rsi = momentum::rsi(&close, 14);
    let (macd, signal, hist) = overlap::macd(&close, 12, 26, 9);
    let (upper, middle, lower) = overlap::bbands(&close, 5, 2.0, 2.0);
}
```

### JavaScript (WASM)

```javascript
const { sma, rsi, macd, bbands } = require('ferro-ta-wasm');

const close = Float64Array.from({ length: 40 }, (_, i) => 44 + i * 0.1);
console.log(sma(close, 5));
console.log(rsi(close, 14));
const [macdLine, signal, hist] = macd(close, 12, 26, 9);
const [upper, middle, lower] = bbands(close, 5, 2.0, 2.0);
```

### Flutter

```dart
import 'dart:typed_data';
import 'package:ferro_ta/ferro_ta.dart';

Future<void> main() async {
  await FerroTa.init();
  final close = Float64List.fromList([
    for (var i = 0; i < 40; i++) 44.0 + i * 0.1,
  ]);
  final smaOut = await sma(close: close, timeperiod: 5);
  final rsiOut = await rsi(close: close, timeperiod: 14);
  final (macdLine, signal, hist) =
      await macd(close: close, fastperiod: 12, slowperiod: 26, signalperiod: 9);
  final (upper, middle, lower) =
      await bbands(close: close, timeperiod: 5, nbdevup: 2, nbdevdn: 2);
}
```

## Benchmark evidence

The latest checked-in TA-Lib comparison artifact uses contiguous `float64`
arrays at 10k and 100k bars on an `Apple M3 Max`, `CPython 3.13.5`, and `Rust
1.91.1`.

- `ferro-ta` achieves competitive parity with TA-Lib, winning on 7 of 12 tested indicators at 100k bars (5 of 12 at 10k bars).
- Strong performance wins at 100k bars include `MFI` (`3.25×`), `WMA` (`2.20×`), `BBANDS` (`1.97×`), and `SMA` (`1.93×`) vs TA-Lib.
- TA-Lib maintains performance advantages on `STOCH` and `ADX`; `EMA`, `ATR`, and `OBV` are statistical ties.
- Compared to pure-Python libraries like Tulipy, `ferro-ta` provides 150-350x speedups through Rust-optimized implementations.

See the benchmark methodology and artifacts:

- [benchmarks/README.md](benchmarks/README.md)
- [benchmarks/artifacts/latest/](benchmarks/artifacts/latest/)
- [docs/benchmarks.rst](docs/benchmarks.rst)

## Capabilities

- 160+ indicators over a shared Rust core.
- Batch and streaming APIs for multi-series and bar-by-bar workloads.
- Python extras: NumPy-first execution with pandas and polars adapters, type stubs, and Sphinx autodoc.
- Pre-built artifacts: Python wheels, crates.io, npm, and Flutter natives (web reuses WASM).
- Reproducible benchmarks instead of blanket speed claims.

Adjacent surfaces — derivatives analytics, MCP, GPU helpers, plugins, and agent wrappers — remain opt-in. See [docs/adjacent_tooling.rst](docs/adjacent_tooling.rst).

## TA-Lib compatibility

- `ferro-ta` implements 156 of TA-Lib 0.6.4's 161 functions, plus 10 extended indicators and 9 streaming classes that TA-Lib does not provide. Not yet implemented: `ACCBANDS`, `IMI`, `AVGDEV`, `MINMAX`, `MINMAXINDEX`.
- Most functions are marked `Exact` or `Close`; the remaining notable non-exact categories are the Hilbert cycle indicators plus `MAMA`, `SAR`, and `SAREXT`.
- The full parity matrix and coverage summary now live in [TA_LIB_COMPATIBILITY.md](TA_LIB_COMPATIBILITY.md).

Migration and compatibility references:

- [docs/migration_talib.rst](docs/migration_talib.rst)
- [docs/compatibility/talib.md](docs/compatibility/talib.md)
- [docs/support_matrix.rst](docs/support_matrix.rst)
- [docs/languages/coverage.rst](docs/languages/coverage.rst) — generated Python / Rust / WASM / Flutter coverage

## Docs map

Languages:

- [docs/languages/index.rst](docs/languages/index.rst)
- [docs/quickstart.rst](docs/quickstart.rst)
- [docs/languages/adding.rst](docs/languages/adding.rst)
- [PLATFORMS.md](PLATFORMS.md)

Python guides:

- [docs/migration_talib.rst](docs/migration_talib.rst)
- [docs/support_matrix.rst](docs/support_matrix.rst)
- [docs/batch.rst](docs/batch.rst)
- [docs/streaming.rst](docs/streaming.rst)
- [docs/derivatives.rst](docs/derivatives.rst)

Evidence and optional tooling:

- [benchmarks/README.md](benchmarks/README.md)
- [docs/mcp.md](docs/mcp.md)
- [docs/adjacent_tooling.rst](docs/adjacent_tooling.rst)
- [docs/plugins.rst](docs/plugins.rst)

Project and release docs:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CHANGELOG.md](CHANGELOG.md)
- [VERSIONING.md](VERSIONING.md)
- [RELEASE.md](RELEASE.md)

## Development

```bash
uv sync --extra dev
uv run pytest tests/unit tests/integration
uv run maturin build --release --out dist
```

More setup details live in [CONTRIBUTING.md](CONTRIBUTING.md). New language bindings must wrap `ferro_ta_core` — see [docs/languages/adding.rst](docs/languages/adding.rst).
