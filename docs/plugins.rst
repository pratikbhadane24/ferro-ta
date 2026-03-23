Writing a plugin
================

The plugin registry lets you register custom indicator functions and call them by name
alongside built-in indicators. This page describes the **plugin contract**, how to
register and run plugins, and a full example.

Plugin contract
---------------

A plugin is a **callable** (function or callable object) that satisfies:

1. **Signature**
   - At least one positional argument that is array-like (e.g. ``close``, ``high``, ``low``).
   - Optional ``*args`` and ``**kwargs`` for parameters (e.g. ``timeperiod=14``).
   - :func:`ferro_ta.registry.run` forwards all ``*args`` and ``**kwargs`` to the callable.

2. **Return type**
   - A single ``numpy.ndarray``, or
   - A tuple of ``numpy.ndarray`` (for multi-output indicators).
   - Output length should match input length (same number of bars); document any exception.

3. **Behaviour**
   - The callable may use ``ferro_ta`` internally (e.g. call :func:`ferro_ta.RSI` and then apply another transformation).
   - Plugins run with the caller's privileges; there is no sandboxing.

Validation
----------

:func:`ferro_ta.registry.register` checks that the provided object is callable. If not,
it raises ``TypeError``. No strict signature check is performed at registration time
so that valid plugins (e.g. with default arguments) are not rejected.

Step-by-step
------------

1. **Write a function** that accepts at least one array-like and returns one or more
   arrays of the same length as the first argument.

2. **Register it** with :func:`ferro_ta.registry.register`:

   .. code-block:: python

      from ferro_ta.registry import register
      register("MY_INDICATOR", my_indicator_function)

3. **Call it by name** with :func:`ferro_ta.registry.run`:

   .. code-block:: python

      from ferro_ta.registry import run
      result = run("MY_INDICATOR", close, timeperiod=14)

4. **List all indicators** (built-in and registered) with :func:`ferro_ta.registry.list_indicators`.

Full example
------------

The following plugin computes a smoothed RSI (RSI of RSI, or "double RSI") and is
included in the repo as ``examples/custom_indicator.py``:

.. code-block:: python

   """Example plugin: smoothed RSI (RSI applied to RSI values)."""
   import numpy as np
   from ferro_ta.registry import register, run, list_indicators
   from ferro_ta import RSI, SMA

   def SMOOTH_RSI(close, timeperiod=14, smooth=3):
       """Smoothed RSI: RSI then SMA of the RSI series."""
       rsi = RSI(close, timeperiod=timeperiod)
       return SMA(rsi, timeperiod=smooth)

   if __name__ == "__main__":
       register("SMOOTH_RSI", SMOOTH_RSI)
       close = np.array([44.34, 44.09, 44.15, 43.61, 44.33, 44.83, 45.10, 45.15, 44.61, 44.33])
       out = run("SMOOTH_RSI", close, timeperiod=5, smooth=2)
       print("SMOOTH_RSI:", out)
       assert "SMOOTH_RSI" in list_indicators()

API reference
-------------

- :func:`ferro_ta.registry.register` — Register a callable under a name.
- :func:`ferro_ta.registry.unregister` — Remove a registered indicator.
- :func:`ferro_ta.registry.get` — Return the callable for a name.
- :func:`ferro_ta.registry.run` — Look up by name and call with given args/kwargs.
- :func:`ferro_ta.registry.list_indicators` — Sorted list of all registered names.
- :exc:`ferro_ta.registry.FerroTARegistryError` — Raised when a name is not found.

See :mod:`ferro_ta.registry` for full docstrings.
