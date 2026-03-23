# Compatibility: ferro-ta vs TA-Lib

See the full migration guide at [docs/migration_talib.rst](../migration_talib.rst).

ferro-ta is designed as a **drop-in replacement** for TA-Lib (`talib` Python package) for the most commonly used indicators.

## Quick Reference

```python
# TA-Lib
import talib
result = talib.SMA(close, timeperiod=14)

# ferro-ta (identical API)
import ferro_ta
result = ferro_ta.SMA(close, timeperiod=14)
```

Full migration guide including all indicator mappings, known differences, and step-by-step migration: [migration_talib.rst](../migration_talib.rst)

## Running Cross-Library Tests

```bash
# Requires TA-Lib C library + talib Python package
pip install TA-Lib
pytest tests/integration/test_vs_talib.py -v
```
