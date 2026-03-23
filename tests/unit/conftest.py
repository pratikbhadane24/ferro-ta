"""
Unit test conftest — inherits shared fixtures from tests/conftest.py.

pytest automatically loads parent conftest.py files, so all fixtures
defined in tests/conftest.py (ohlcv_500, ohlcv_100, ohlcv_real) are
available here without any explicit import.
"""
