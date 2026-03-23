"""
ferro_ta.adapters — Market data adapters (pluggable).

Defines an abstract ``DataAdapter`` interface and a concrete
``CsvAdapter`` that loads OHLCV data from a CSV file.  Users can
subclass ``DataAdapter`` to add their own data sources (e.g. Alpaca,
Yahoo Finance, a database, etc.) while keeping the rest of the pipeline
unchanged.

Classes
-------
DataAdapter
    Abstract base class.  Subclasses must implement :meth:`fetch`.

CsvAdapter
    Load OHLCV data from a CSV file.  Requires pandas.

InMemoryAdapter
    Wrap an already-loaded pandas DataFrame or dict of arrays.

Functions
---------
register_adapter(name, adapter_class)
    Register an adapter class under a name for lookup by string.

get_adapter(name)
    Return an adapter class previously registered under *name*.

Examples
--------
>>> from ferro_ta.data.adapters import InMemoryAdapter
>>> import numpy as np
>>> n = 50
>>> rng = np.random.default_rng(0)
>>> close = np.cumprod(1 + rng.normal(0, 0.01, n)) * 100
>>> adapter = InMemoryAdapter({
...     "open": close, "high": close * 1.001,
...     "low": close * 0.999, "close": close,
...     "volume": np.ones(n) * 1000,
... })
>>> ohlcv = adapter.fetch()
>>> "close" in ohlcv
True
"""

from __future__ import annotations

import abc
from typing import Any, Optional

__all__ = [
    "DataAdapter",
    "CsvAdapter",
    "InMemoryAdapter",
    "register_adapter",
    "get_adapter",
]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_ADAPTER_REGISTRY: dict[str, type[DataAdapter]] = {}


def register_adapter(name: str, adapter_class: type[DataAdapter]) -> None:
    """Register *adapter_class* under *name*.

    Parameters
    ----------
    name : str
    adapter_class : type  — must subclass :class:`DataAdapter`

    Examples
    --------
    >>> from ferro_ta.data.adapters import register_adapter, DataAdapter
    >>> class MyAdapter(DataAdapter):
    ...     def fetch(self, **kwargs): return {}
    >>> register_adapter("my_source", MyAdapter)
    """
    if not issubclass(adapter_class, DataAdapter):
        raise TypeError(f"{adapter_class!r} must subclass DataAdapter")
    _ADAPTER_REGISTRY[name] = adapter_class


def get_adapter(name: str) -> type[DataAdapter]:
    """Return the adapter class registered under *name*.

    Parameters
    ----------
    name : str

    Raises
    ------
    KeyError
        If *name* is not registered.
    """
    if name not in _ADAPTER_REGISTRY:
        available = sorted(_ADAPTER_REGISTRY.keys())
        raise KeyError(f"No adapter registered under {name!r}. Available: {available}")
    return _ADAPTER_REGISTRY[name]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class DataAdapter(abc.ABC):
    """Abstract base class for market data adapters.

    Subclasses must implement :meth:`fetch`, which returns OHLCV data as
    a ``pandas.DataFrame`` (preferred) or a ``dict`` of numpy arrays.

    The contract for the returned data:
    - Keys/columns: ``open``, ``high``, ``low``, ``close``, ``volume``
      (additional columns are allowed but not required).
    - Values: numeric (float64-compatible).
    - Index (for DataFrames): ideally a ``DatetimeIndex``; not required.
    """

    @abc.abstractmethod
    def fetch(self, **kwargs: Any) -> Any:
        """Return OHLCV data.

        Returns
        -------
        pandas.DataFrame or dict
            OHLCV data with keys/columns ``open``, ``high``, ``low``,
            ``close``, ``volume``.
        """

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


# ---------------------------------------------------------------------------
# CsvAdapter
# ---------------------------------------------------------------------------


class CsvAdapter(DataAdapter):
    """Load OHLCV data from a CSV file.

    The CSV must have a header row.  Column names are configurable.

    Parameters
    ----------
    path : str
        Path to the CSV file.
    open_col, high_col, low_col, close_col, volume_col : str
        CSV column names for each OHLCV field.
    index_col : str or None
        Column to use as the DataFrame index (e.g. ``'timestamp'``).
    parse_dates : bool
        If ``True`` (default), attempt to parse the index as dates.

    Requires
    --------
    pandas

    Examples
    --------
    >>> from ferro_ta.data.adapters import CsvAdapter
    >>> # adapter = CsvAdapter("data.csv", index_col="date")
    >>> # ohlcv = adapter.fetch()
    """

    def __init__(
        self,
        path: str,
        *,
        open_col: str = "open",
        high_col: str = "high",
        low_col: str = "low",
        close_col: str = "close",
        volume_col: str = "volume",
        index_col: Optional[str] = None,
        parse_dates: bool = True,
    ) -> None:
        self.path = path
        self.open_col = open_col
        self.high_col = high_col
        self.low_col = low_col
        self.close_col = close_col
        self.volume_col = volume_col
        self.index_col = index_col
        self.parse_dates = parse_dates

    def fetch(self, **kwargs: Any) -> Any:
        """Load the CSV and return a pandas DataFrame.

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for CsvAdapter.  Install with: pip install pandas"
            ) from exc
        df = pd.read_csv(
            self.path,
            index_col=self.index_col,
            parse_dates=self.parse_dates if self.index_col is not None else False,
        )
        # Rename columns if they differ from the standard names
        rename = {}
        for src, dst in [
            (self.open_col, "open"),
            (self.high_col, "high"),
            (self.low_col, "low"),
            (self.close_col, "close"),
            (self.volume_col, "volume"),
        ]:
            if src != dst and src in df.columns:
                rename[src] = dst
        if rename:
            df = df.rename(columns=rename)
        return df

    def __repr__(self) -> str:
        return f"CsvAdapter(path={self.path!r})"


# ---------------------------------------------------------------------------
# InMemoryAdapter
# ---------------------------------------------------------------------------


class InMemoryAdapter(DataAdapter):
    """Wrap already-loaded OHLCV data (dict or DataFrame).

    Parameters
    ----------
    data : dict or pandas.DataFrame
        OHLCV data.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.data.adapters import InMemoryAdapter
    >>> n = 10
    >>> close = np.ones(n) * 100.0
    >>> adapter = InMemoryAdapter({"open": close, "high": close,
    ...                            "low": close, "close": close,
    ...                            "volume": close})
    >>> ohlcv = adapter.fetch()
    >>> "close" in ohlcv
    True
    """

    def __init__(self, data: Any) -> None:
        self._data = data

    def fetch(self, **kwargs: Any) -> Any:
        """Return the wrapped data as-is."""
        return self._data

    def __repr__(self) -> str:
        return "InMemoryAdapter(...)"


# ---------------------------------------------------------------------------
# Register built-in adapters
# ---------------------------------------------------------------------------

register_adapter("csv", CsvAdapter)
register_adapter("memory", InMemoryAdapter)
