from __future__ import annotations

from unittest.mock import patch

import numpy as np

from ferro_ta._utils import (
    _optional_pandas_module,
    _optional_polars_module,
    pandas_wrap,
    polars_wrap,
)


def _missing_only(module_name: str):
    real_import = __import__
    attempts: list[str] = []

    def side_effect(name, globals=None, locals=None, fromlist=(), level=0):
        if name == module_name:
            attempts.append(name)
            raise ImportError(f"{module_name} not installed")
        return real_import(name, globals, locals, fromlist, level)

    return attempts, side_effect


def test_pandas_wrap_caches_missing_optional_import() -> None:
    _optional_pandas_module.cache_clear()
    wrapped = pandas_wrap(lambda arr: arr)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    attempts, side_effect = _missing_only("pandas")

    try:
        with patch("builtins.__import__", side_effect=side_effect):
            np.testing.assert_array_equal(wrapped(arr), arr)
            np.testing.assert_array_equal(wrapped(arr), arr)
    finally:
        _optional_pandas_module.cache_clear()

    assert attempts == ["pandas"]


def test_polars_wrap_caches_missing_optional_import() -> None:
    _optional_polars_module.cache_clear()
    wrapped = polars_wrap(lambda arr: arr)
    arr = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    attempts, side_effect = _missing_only("polars")

    try:
        with patch("builtins.__import__", side_effect=side_effect):
            np.testing.assert_array_equal(wrapped(arr), arr)
            np.testing.assert_array_equal(wrapped(arr), arr)
    finally:
        _optional_polars_module.cache_clear()

    assert attempts == ["polars"]
