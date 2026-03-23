Error Handling and Validation
=============================

ferro-ta uses a consistent error model so you can catch and handle failures in a
predictable way.

Exception hierarchy
-------------------

All ferro-ta–specific exceptions inherit from :exc:`ferro_ta.FerroTAError` and the
corresponding built-in type so that existing ``except ValueError`` code keeps
working:

- **FerroTAError** — base for all ferro-ta exceptions
- **FerroTAValueError** — invalid parameter values (e.g. ``timeperiod < 1``,
  ``fastperiod >= slowperiod`` for MACD). Inherits from :exc:`ValueError`.
- **FerroTAInputError** — invalid input arrays (mismatched lengths, wrong shape,
  or opt-in strict checks). Inherits from :exc:`ValueError`.

Example:

.. code-block:: python

   from ferro_ta import SMA, FerroTAValueError, FerroTAInputError

   try:
       SMA(close, timeperiod=0)
   except FerroTAValueError as e:
       print(e)  # "timeperiod must be >= 1, got 0"

   try:
       SMA(open_arr, timeperiod=5)  # if open_arr has different length
   except FerroTAInputError as e:
       print(e)

Validation in wrappers
----------------------

Every indicator wrapper validates parameters and inputs before calling the Rust
engine:

- **Period parameters** (e.g. ``timeperiod``, ``fastperiod``, ``slowperiod``) are
  checked with :func:`ferro_ta.exceptions.check_timeperiod` and must be >= 1
  (or >= 2 where the algorithm requires it, e.g. MAVP ``minperiod``).
- **Multiple arrays** (e.g. open, high, low, close, volume) are checked with
  :func:`ferro_ta.exceptions.check_equal_length` so all have the same length.

Any error raised by the Rust extension (e.g. invalid value or bad array) is
re-raised as :exc:`FerroTAValueError` or :exc:`FerroTAInputError` with the same
message, so you can rely on the ferro-ta exception hierarchy.

NaN and Inf
-----------

By default, ferro-ta **propagates** NaN and Inf in input arrays: output values
that depend on a NaN/Inf input will themselves be NaN/Inf. No exception is
raised for NaN or Inf in the input.

If you need strict behaviour (no NaN/Inf), call
:func:`ferro_ta.exceptions.check_finite` on your arrays before passing them to
an indicator.

Empty and short arrays
----------------------

Indicators that require a minimum number of bars (e.g. SMA with ``timeperiod=5``
needs at least 5 elements) may return an array of NaN or raise if the Rust layer
rejects the input. You can use :func:`ferro_ta.exceptions.check_min_length` to
enforce a minimum length before calling an indicator.

Helper reference
----------------

- :func:`ferro_ta.exceptions.check_timeperiod` — raise if a period parameter is below minimum
- :func:`ferro_ta.exceptions.check_equal_length` — raise if supplied arrays have different lengths
- :func:`ferro_ta.exceptions.check_finite` — raise if an array contains NaN or Inf (opt-in strict)
- :func:`ferro_ta.exceptions.check_min_length` — raise if an array is shorter than required

See the :mod:`ferro_ta.exceptions` API for full signatures and examples.
