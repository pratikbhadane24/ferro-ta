"""
Custom exception hierarchy for ferro_ta.

Exception classes
-----------------
FerroTAError       — Base class for all ferro_ta exceptions.
FerroTAValueError  — Raised for invalid parameter values (e.g. timeperiod < 1).
FerroTAInputError  — Raised for invalid input arrays (e.g. mismatched lengths, wrong dtype, unexpected NaN/Inf when strict mode is used).

All custom exceptions inherit from both the ferro_ta base and the corresponding
built-in exception (ValueError) so that existing ``except ValueError`` clauses
continue to work after upgrading.

Error codes
-----------
Every exception carries a ``code`` attribute (e.g. ``"FTERR001"``) for
programmatic handling:

    FTERR001  — Invalid parameter value (FerroTAValueError)
    FTERR002  — Invalid input array (FerroTAInputError)
    FTERR003  — Input array too short (FerroTAInputError)
    FTERR004  — Input arrays have mismatched lengths (FerroTAInputError)
    FTERR005  — Input array contains NaN or Inf (FerroTAInputError, strict mode)
    FTERR006  — General Rust-bridge error (FerroTAValueError or FerroTAInputError)

Examples
--------
>>> from ferro_ta.core.exceptions import FerroTAError, FerroTAValueError, FerroTAInputError
>>> raise FerroTAValueError("timeperiod must be >= 1, got 0")
Traceback (most recent call last):
    ...
ferro_ta.exceptions.FerroTAValueError: [FTERR001] timeperiod must be >= 1, got 0
>>> try:
...     raise FerroTAValueError("bad value")
... except FerroTAValueError as exc:
...     print(exc.code)
FTERR001

NaN / Inf policy
----------------
By default ferro_ta **propagates** NaN and Inf in input arrays — output values
that depend on a NaN/Inf input will themselves be NaN/Inf.  No exception is
raised for NaN or Inf values in the input data.  If you need strict mode, call
:func:`ferro_ta.exceptions.check_finite` on your arrays before passing them.
"""

from __future__ import annotations

from typing import NoReturn

# ---------------------------------------------------------------------------
# Error code registry
# ---------------------------------------------------------------------------

#: Maps each ``FerroTAError`` subclass to its default error code.
ERROR_CODES: dict[str, str] = {
    "FerroTAError": "FTERR000",
    "FerroTAValueError": "FTERR001",
    "FerroTAInputError": "FTERR002",
}

# Well-known codes for specific error kinds
_CODE_TOO_SHORT = "FTERR003"
_CODE_LENGTH_MISMATCH = "FTERR004"
_CODE_NOT_FINITE = "FTERR005"
_CODE_RUST_BRIDGE = "FTERR006"

# Code descriptions (for reference and programmatic inspection)
ERROR_CODE_DESCRIPTIONS: dict[str, str] = {
    "FTERR000": "General ferro_ta error (base class fallback)",
    "FTERR001": "Invalid parameter value",
    "FTERR002": "Invalid input array",
    "FTERR003": "Input array too short",
    "FTERR004": "Input arrays have mismatched lengths",
    "FTERR005": "Input array contains NaN or Inf (strict mode)",
    "FTERR006": "Rust-bridge error (re-raised from Rust ValueError)",
}


class FerroTAError(Exception):
    """Base class for all ferro_ta exceptions.

    Attributes
    ----------
    code : str
        A short error code string (e.g. ``"FTERR001"``) for programmatic
        handling.  The code is included at the beginning of the exception
        message.
    suggestion : str | None
        Optional human-readable suggestion for how to fix the error.
    """

    code: str = "FTERR000"
    suggestion: str | None = None

    def __init__(
        self,
        message: str,
        *,
        code: str | None = None,
        suggestion: str | None = None,
    ) -> None:
        self.code = code or type(self).code
        self.suggestion = suggestion
        full_msg = f"[{self.code}] {message}"
        if suggestion:
            full_msg = f"{full_msg}\n  Suggestion: {suggestion}"
        super().__init__(full_msg)


class FerroTAValueError(FerroTAError, ValueError):
    """Raised when a parameter value is out of the accepted range.

    Examples: ``timeperiod < 1``, ``fastperiod >= slowperiod`` for MACD.

    Default error code: ``FTERR001``.
    """

    code = "FTERR001"


class FerroTAInputError(FerroTAError, ValueError):
    """Raised when one or more input arrays are invalid.

    Examples: mismatched lengths for open/high/low/close, wrong dtype that
    cannot be coerced to float64.

    Default error code: ``FTERR002``.
    """

    code = "FTERR002"


# ---------------------------------------------------------------------------
# Validation helpers (called by Python wrappers)
# ---------------------------------------------------------------------------


def check_timeperiod(value: int, name: str = "timeperiod", minimum: int = 1) -> None:
    """Raise :class:`FerroTAValueError` if *value* < *minimum*.

    Parameters
    ----------
    value:
        The period parameter to validate.
    name:
        Human-readable parameter name for the error message.
    minimum:
        Minimum acceptable value (default 1).

    Raises
    ------
    FerroTAValueError
        If ``value < minimum``.
    """
    if value < minimum:
        raise FerroTAValueError(
            f"{name} must be >= {minimum}, got {value}",
            suggestion=f"Set {name}={minimum} or higher.",
        )


def check_equal_length(**arrays: object) -> None:
    """Raise :class:`FerroTAInputError` if the supplied arrays differ in length.

    Parameters
    ----------
    **arrays:
        Keyword arguments mapping name → array-like.  At least two arrays
        should be supplied for the check to be meaningful.

    Raises
    ------
    FerroTAInputError
        If any two arrays have different lengths.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.core.exceptions import check_equal_length
    >>> check_equal_length(open=np.array([1.0]), close=np.array([1.0, 2.0]))
    Traceback (most recent call last):
        ...
    ferro_ta.exceptions.FerroTAInputError: ...
    """

    lengths = {}
    for name, arr in arrays.items():
        if hasattr(arr, "__len__"):
            lengths[name] = len(arr)  # type: ignore[arg-type]
        elif hasattr(arr, "shape"):
            lengths[name] = arr.shape[0]  # type: ignore[union-attr]

    if len(set(lengths.values())) > 1:
        detail = ", ".join(f"{k}={v}" for k, v in lengths.items())
        raise FerroTAInputError(
            f"All input arrays must have the same length. Got: {detail}",
            code=_CODE_LENGTH_MISMATCH,
            suggestion="Trim or align your arrays so that open, high, low, close, and volume all have the same number of rows.",
        )


def check_finite(arr: object, name: str = "input") -> None:
    """Raise :class:`FerroTAInputError` if *arr* contains NaN or Inf.

    This is an *opt-in* strict-mode helper.  ferro_ta does **not** call this
    automatically — it is provided for users who want deterministic behaviour
    when their data may contain missing values.

    Parameters
    ----------
    arr:
        Array-like to check.
    name:
        Human-readable name used in the error message.

    Raises
    ------
    FerroTAInputError
        If any element of *arr* is NaN or Inf.
    """
    import numpy as np  # local import

    a = np.asarray(arr, dtype=np.float64)
    if not np.all(np.isfinite(a)):
        raise FerroTAInputError(
            f"{name} contains NaN or Inf values. "
            "ferro_ta propagates NaN by default; call check_finite() only "
            "when you require all-finite inputs.",
            code=_CODE_NOT_FINITE,
            suggestion="Use numpy.nan_to_num() or dropna() to clean your data before passing it to ferro_ta.",
        )


def check_min_length(arr: object, min_len: int, name: str = "input") -> None:
    """Raise :class:`FerroTAInputError` if *arr* has length less than *min_len*.

    Parameters
    ----------
    arr:
        Array-like to check.
    min_len:
        Minimum required length.
    name:
        Human-readable name used in the error message.

    Raises
    ------
    FerroTAInputError
        If ``len(arr) < min_len``.
    """
    length = 0
    if hasattr(arr, "__len__"):
        length = len(arr)  # type: ignore[arg-type]
    elif hasattr(arr, "shape"):
        length = arr.shape[0]  # type: ignore[union-attr]
    if length < min_len:
        raise FerroTAInputError(
            f"{name} must have at least {min_len} elements, got {length}",
            code=_CODE_TOO_SHORT,
            suggestion=f"Provide at least {min_len} data points. Current length: {length}.",
        )


def _normalize_rust_error(err: ValueError) -> NoReturn:
    """Re-raise a Rust-originated ValueError as FerroTAValueError or FerroTAInputError.

    Used by Python wrappers so users can catch FerroTA* exceptions consistently.
    """
    msg = str(err).lower()
    if (
        "length" in msg
        or "same length" in msg
        or "array" in msg
        or "mismatch" in msg
        or "dimension" in msg
        or "1-d" in msg
    ):
        raise FerroTAInputError(str(err), code=_CODE_RUST_BRIDGE) from err
    raise FerroTAValueError(str(err), code=_CODE_RUST_BRIDGE) from err
