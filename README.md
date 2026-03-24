<div align="center">

# ⚡ ferro-ta

### Rust-powered Python technical analysis with a TA-Lib-compatible API

**Focused on one primary job: fast, reproducible technical analysis for Python users who want TA-Lib-style ergonomics without native build friction.**

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/pratikbhadane24/ferro-ta/HEAD?labpath=examples%2Fquickstart.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pratikbhadane24/ferro-ta/blob/main/examples/quickstart.ipynb)
[![Documentation](https://img.shields.io/badge/docs-github.io-blue)](https://pratikbhadane24.github.io/ferro-ta/)

</div>

---

> `ferro-ta` is a Rust-backed Python technical analysis library for NumPy-first workloads. It keeps TA-Lib-style ergonomics, ships pre-built wheels on supported targets, and publishes reproducible benchmark artifacts instead of blanket speed claims.

## 🚀 What ferro-ta is

| | TA-Lib | ferro-ta |
|---|---|---|
| **Primary product** | C-backed Python TA library | Rust-backed Python TA library |
| **API shape** | `talib.SMA(close, 20)` | `ferro_ta.SMA(close, 20)` |
| **Installation** | Often requires native/system setup | Pre-built wheels on supported targets |
| **Scope** | Technical indicators | Technical indicators first; other tooling is optional and secondary |

## ⚡ Benchmark evidence

The latest checked-in TA-Lib comparison artifact uses contiguous `float64`
arrays at 10k and 100k bars on an `Apple M3 Max`, `CPython 3.13.5`, and `Rust
1.91.1`.

- `ferro-ta` is ahead outside the tie band on 6 of 12 indicators at both 10k and 100k bars.
- Strong public wins in the latest 100k-bar artifact include `SMA` (`2.28x`), `BBANDS` (`2.34x`), `MFI` (`3.04x`), and `WMA` (`2.39x`).
- TA-Lib still wins or ties on parts of the suite, including `STOCH`, `ADX`, and some current `EMA` / `RSI` / `ATR` runs.

See the benchmark methodology and artifacts:

- [benchmarks/README.md](benchmarks/README.md)
- [benchmarks/artifacts/latest/](benchmarks/artifacts/latest/)
- [docs/benchmarks.rst](docs/benchmarks.rst)

## 🎯 Core capabilities

- 160+ indicators with a TA-Lib-style public API.
- Batch and streaming APIs for multi-series and bar-by-bar workloads.
- NumPy-first execution with pandas and polars adapters.
- Pre-built wheels on the supported Python and OS matrix.
- Type stubs, error codes, examples, and reproducible benchmarks.

Adjacent and experimental surfaces such as derivatives analytics, MCP, GPU,
plugins, and WASM remain opt-in and secondary to the core TA library story.

## 📦 Installation

```bash
pip install ferro-ta
```

Optional extras:

```bash
pip install "ferro-ta[pandas]"   # pandas.Series support
pip install "ferro-ta[polars]"   # polars.Series support
pip install "ferro-ta[gpu]"      # PyTorch-backed GPU helpers
pip install "ferro-ta[options]"  # derivatives analytics helpers
pip install "ferro-ta[mcp]"      # MCP server for agent/tool clients
pip install "ferro-ta[all]"      # most optional extras (excluding gpu)
```

## ⚡ Quick start

```python
import numpy as np
from ferro_ta import SMA, EMA, RSI, MACD, BBANDS

close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15,
                  43.61, 44.33, 44.83, 45.10, 45.15, 43.61, 44.33])

sma = SMA(close, timeperiod=5)
ema = EMA(close, timeperiod=5)
rsi = RSI(close, timeperiod=14)
macd_line, signal, histogram = MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
upper, middle, lower = BBANDS(close, timeperiod=5, nbdevup=2.0, nbdevdn=2.0)
```

## 📊 TA-Lib compatibility

- `ferro-ta` implements 100% of TA-Lib's function set (`162+` indicators).
- Most functions are marked `Exact` or `Close`; the remaining notable non-exact categories are the Hilbert cycle indicators plus `MAMA`, `SAR`, and `SAREXT`.
- The full parity matrix and coverage summary now live in [TA_LIB_COMPATIBILITY.md](TA_LIB_COMPATIBILITY.md).

Migration and compatibility references:

- [docs/migration_talib.rst](docs/migration_talib.rst)
- [docs/compatibility/talib.md](docs/compatibility/talib.md)
- [docs/support_matrix.rst](docs/support_matrix.rst)

## 🗺️ Docs map

Core guides:

- [docs/quickstart.rst](docs/quickstart.rst)
- [docs/migration_talib.rst](docs/migration_talib.rst)
- [docs/support_matrix.rst](docs/support_matrix.rst)
- [PLATFORMS.md](PLATFORMS.md)

Evidence and APIs:

- [benchmarks/README.md](benchmarks/README.md)
- [docs/batch.rst](docs/batch.rst)
- [docs/streaming.rst](docs/streaming.rst)
- [docs/derivatives.rst](docs/derivatives.rst)

Optional and experimental surfaces:

- [docs/mcp.md](docs/mcp.md)
- [docs/adjacent_tooling.rst](docs/adjacent_tooling.rst)
- [docs/plugins.rst](docs/plugins.rst)
- [wasm/README.md](wasm/README.md)

Project and release docs:

- [CONTRIBUTING.md](CONTRIBUTING.md)
- [CHANGELOG.md](CHANGELOG.md)
- [VERSIONING.md](VERSIONING.md)
- [RELEASE.md](RELEASE.md)

## 🛠️ Development

```bash
uv sync --extra dev
uv run pytest tests/unit tests/integration
uv run maturin build --release --out dist
```

More setup details live in [CONTRIBUTING.md](CONTRIBUTING.md).
