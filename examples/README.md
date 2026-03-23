# ferro-ta Examples

Jupyter notebooks demonstrating key ferro-ta features.

## Notebooks

| Notebook | Description |
|---|---|
| [`quickstart.ipynb`](quickstart.ipynb) | Core API: moving averages, RSI, MACD, Bollinger Bands, batch API, pipeline, pandas integration |
| [`streaming.ipynb`](streaming.ipynb) | Streaming bar-by-bar API: StreamingSMA, StreamingRSI, StreamingBBands, StreamingMACD, StreamingATR |
| [`backtesting.ipynb`](backtesting.ipynb) | Backtesting harness, indicator pipeline for feature engineering, config defaults |
| [`features_21_30.ipynb`](features_21_30.ipynb) | Multi-timeframe, resampling, portfolio analytics, strategy DSL, feature matrix, viz, adapters |

## Running the Notebooks

```bash
# Install dependencies
pip install ferro-ta jupyter numpy

# Optional: pandas and polars integration
pip install "ferro-ta[pandas]" "ferro_ta[polars]"

# Start Jupyter
jupyter notebook examples/
```

## Links

- [ferro-ta README](../README.md)
- [API documentation](../docs/)
- [CONTRIBUTING.md](../CONTRIBUTING.md)
