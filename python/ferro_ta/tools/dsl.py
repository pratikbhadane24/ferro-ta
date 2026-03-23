"""
ferro_ta.dsl — Strategy expression DSL.

A small domain-specific language that lets users define rule-based trading
strategies as strings (e.g. ``"RSI(14) < 30 and close > SMA(20)"``) and
evaluate them to produce a boolean or integer signal series.

This module provides:
- :func:`parse_expression` — validate and compile an expression string.
- :func:`evaluate` — evaluate a compiled expression against OHLCV data.
- :class:`Strategy` — convenience wrapper around parse + evaluate.

The expression grammar supports:
- Indicator calls: ``RSI(14)``, ``SMA(20)``, ``BBANDS(20, 2)``
- Price series references: ``close``, ``open``, ``high``, ``low``, ``volume``
- Comparison operators: ``<``, ``>``, ``<=``, ``>=``, ``==``, ``!=``
- Logical connectives: ``and``, ``or``, ``not``
- Cross-above/below helpers: ``cross_above(a, b)``, ``cross_below(a, b)``
- Parentheses for grouping

Evaluating an expression returns a 1-D integer array of 1 (signal on) and 0
(signal off), with leading ``0`` values during indicator warm-up.

Examples
--------
>>> import numpy as np
>>> from ferro_ta.tools.dsl import Strategy
>>> rng = np.random.default_rng(0)
>>> close = np.cumprod(1 + rng.normal(0, 0.01, 100)) * 100
>>> ohlcv = {"close": close}
>>> strat = Strategy("RSI(14) < 30")
>>> signal = strat.evaluate(ohlcv)
>>> signal.shape
(100,)
>>> set(signal.tolist()).issubset({0, 1})
True
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any, Optional

import numpy as np
from numpy.typing import NDArray

from ferro_ta._utils import _to_f64
from ferro_ta.core.registry import run as _registry_run

__all__ = [
    "parse_expression",
    "evaluate",
    "Strategy",
]

# ---------------------------------------------------------------------------
# Supported indicator / function names (resolved via registry)
# ---------------------------------------------------------------------------

_PRICE_KEYS = {"close", "open", "high", "low", "volume"}

# ---------------------------------------------------------------------------
# Expression AST (minimal)
# ---------------------------------------------------------------------------


class _Expr:
    """Abstract expression node."""

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        raise NotImplementedError


class _PriceRef(_Expr):
    def __init__(self, name: str) -> None:
        self.name = name

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        if self.name not in ctx:
            raise ValueError(f"Price series '{self.name}' not found in OHLCV data.")
        return ctx[self.name]


class _IndicatorCall(_Expr):
    def __init__(
        self,
        name: str,
        args: list[float],
        output_index: int = 0,
    ) -> None:
        self.name = name
        self.args = args
        self.output_index = output_index

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        close = ctx.get("close")
        high = ctx.get("high")
        low = ctx.get("low")
        volume = ctx.get("volume")
        if close is None:
            raise ValueError("'close' series is required to evaluate indicator calls.")
        kwargs: dict[str, Any] = {}
        if self.args:
            # Heuristic: first numeric arg → timeperiod
            kwargs["timeperiod"] = int(self.args[0])
            # Additional args passed as extra kwargs are not supported in this
            # simple DSL; only the first param is used as timeperiod.

        # Try different signatures
        result = None
        for positional in [
            [close],
            [high, low, close] if high is not None and low is not None else None,
            [high, low, close, volume]
            if volume is not None and high is not None
            else None,
        ]:
            if positional is None:
                continue
            try:
                result = _registry_run(self.name, *positional, **kwargs)
                break
            except Exception:
                continue
        if result is None:
            raise ValueError(
                f"Cannot evaluate indicator '{self.name}' with available data."
            )

        if isinstance(result, tuple):
            arr = result[self.output_index]
        else:
            arr = result
        return np.asarray(arr, dtype=np.float64)


class _Comparison(_Expr):
    _OPS: dict[str, Callable[[Any, Any], Any]] = {
        "<": lambda a, b: a < b,
        ">": lambda a, b: a > b,
        "<=": lambda a, b: a <= b,
        ">=": lambda a, b: a >= b,
        "==": lambda a, b: a == b,
        "!=": lambda a, b: a != b,
    }

    def __init__(self, left: _Expr, op: str, right: _Expr) -> None:
        self.left = left
        self.op = op
        self.right = right

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        lv = self.left.eval(ctx)
        rv = self.right.eval(ctx)
        fn = self._OPS[self.op]
        result = fn(lv, rv)
        return result.astype(np.int32)


class _Logic(_Expr):
    def __init__(self, op: str, operands: list[_Expr]) -> None:
        self.op = op  # 'and' | 'or'
        self.operands = operands

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        result = self.operands[0].eval(ctx).astype(bool)
        for operand in self.operands[1:]:
            v = operand.eval(ctx).astype(bool)
            if self.op == "and":
                result = result & v
            else:
                result = result | v
        return result.astype(np.int32)


class _Not(_Expr):
    def __init__(self, operand: _Expr) -> None:
        self.operand = operand

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        return (~self.operand.eval(ctx).astype(bool)).astype(np.int32)


class _CrossFunc(_Expr):
    def __init__(self, direction: str, a: _Expr, b: _Expr) -> None:
        self.direction = direction  # 'above' | 'below'
        self.a = a
        self.b = b

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        av = self.a.eval(ctx).astype(np.float64)
        bv = self.b.eval(ctx).astype(np.float64)
        n = len(av)
        result = np.zeros(n, dtype=np.int32)
        if self.direction == "above":
            for i in range(1, n):
                if av[i] > bv[i] and av[i - 1] <= bv[i - 1]:
                    result[i] = 1
        else:
            for i in range(1, n):
                if av[i] < bv[i] and av[i - 1] >= bv[i - 1]:
                    result[i] = 1
        return result


class _Scalar(_Expr):
    def __init__(self, value: float) -> None:
        self.value = value

    def eval(self, ctx: dict[str, NDArray[np.float64]]) -> NDArray:
        return np.array([self.value])


# ---------------------------------------------------------------------------
# Tokeniser
# ---------------------------------------------------------------------------

_TOKEN_SPEC = [
    ("NUMBER", r"-?\d+\.?\d*"),
    ("AND", r"\band\b"),
    ("OR", r"\bor\b"),
    ("NOT", r"\bnot\b"),
    ("IDENT", r"[A-Za-z_][A-Za-z0-9_]*"),
    ("OP", r"<=|>=|==|!=|<|>"),
    ("LPAREN", r"\("),
    ("RPAREN", r"\)"),
    ("COMMA", r","),
    ("SKIP", r"\s+"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for name, pattern in _TOKEN_SPEC)
)


def _tokenise(expr: str) -> list[tuple[str, str]]:
    tokens: list[tuple[str, str]] = []
    for m in _TOKEN_RE.finditer(expr):
        kind = m.lastgroup
        value = m.group()
        if kind == "SKIP" or kind is None:
            continue
        tokens.append((kind, value))
    # Check for unmatched characters
    matched_len = sum(len(m.group()) for m in _TOKEN_RE.finditer(expr))
    if matched_len != len(expr.replace(" ", "").replace("\t", "").replace("\n", "")):
        # rough check; just skip
        pass
    return tokens


# ---------------------------------------------------------------------------
# Recursive-descent parser
# ---------------------------------------------------------------------------


class _Parser:
    def __init__(self, tokens: list[tuple[str, str]]) -> None:
        self.tokens = tokens
        self.pos = 0

    def peek(self) -> Optional[tuple[str, str]]:
        if self.pos < len(self.tokens):
            return self.tokens[self.pos]
        return None

    def consume(self, kind: Optional[str] = None) -> tuple[str, str]:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression.")
        if kind and tok[0] != kind:
            raise ValueError(f"Expected {kind}, got {tok[0]!r} ({tok[1]!r}).")
        self.pos += 1
        return tok

    def parse(self) -> _Expr:
        expr = self.parse_or()
        if self.peek() is not None:
            raise ValueError(
                f"Unexpected token at position {self.pos}: {self.peek()!r}"
            )
        return expr

    def parse_or(self) -> _Expr:
        left = self.parse_and()
        operands = [left]
        while self.peek() and self.peek()[0] == "OR":  # type: ignore[index]
            self.consume("OR")
            operands.append(self.parse_and())
        return operands[0] if len(operands) == 1 else _Logic("or", operands)

    def parse_and(self) -> _Expr:
        left = self.parse_not()
        operands = [left]
        while self.peek() and self.peek()[0] == "AND":  # type: ignore[index]
            self.consume("AND")
            operands.append(self.parse_not())
        return operands[0] if len(operands) == 1 else _Logic("and", operands)

    def parse_not(self) -> _Expr:
        if self.peek() and self.peek()[0] == "NOT":  # type: ignore[index]
            self.consume("NOT")
            return _Not(self.parse_not())
        return self.parse_comparison()

    def parse_comparison(self) -> _Expr:
        left = self.parse_atom()
        tok = self.peek()
        if tok and tok[0] == "OP":
            op = tok[1]
            self.consume("OP")
            right = self.parse_atom()
            return _Comparison(left, op, right)
        return left

    def parse_atom(self) -> _Expr:
        tok = self.peek()
        if tok is None:
            raise ValueError("Unexpected end of expression in atom.")

        if tok[0] == "NUMBER":
            self.consume("NUMBER")
            return _Scalar(float(tok[1]))

        if tok[0] == "LPAREN":
            self.consume("LPAREN")
            expr = self.parse_or()
            self.consume("RPAREN")
            return expr

        if tok[0] == "NOT":
            self.consume("NOT")
            return _Not(self.parse_comparison())

        if tok[0] == "IDENT":
            name = tok[1]
            self.consume("IDENT")

            # Check if followed by '('
            if self.peek() and self.peek()[0] == "LPAREN":  # type: ignore[index]
                self.consume("LPAREN")
                # Parse comma-separated args
                args: list[float] = []
                sub_exprs: list[_Expr] = []
                while self.peek() and self.peek()[0] != "RPAREN":  # type: ignore[index]
                    t = self.peek()
                    if t and t[0] == "NUMBER":
                        self.consume("NUMBER")
                        args.append(float(t[1]))
                    elif t and t[0] == "IDENT":
                        # nested indicator or price ref used as sub-expression
                        sub_exprs.append(self.parse_atom())
                    if self.peek() and self.peek()[0] == "COMMA":  # type: ignore[index]
                        self.consume("COMMA")
                self.consume("RPAREN")

                name_upper = name.upper()
                if name_upper == "CROSS_ABOVE":
                    if len(sub_exprs) < 2:
                        raise ValueError("cross_above requires two arguments.")
                    return _CrossFunc("above", sub_exprs[0], sub_exprs[1])
                if name_upper == "CROSS_BELOW":
                    if len(sub_exprs) < 2:
                        raise ValueError("cross_below requires two arguments.")
                    return _CrossFunc("below", sub_exprs[0], sub_exprs[1])
                return _IndicatorCall(name_upper, args)
            else:
                # Price reference or bare indicator name
                name_lower = name.lower()
                if name_lower in _PRICE_KEYS:
                    return _PriceRef(name_lower)
                # Treat as indicator with no args
                return _IndicatorCall(name.upper(), [])

        raise ValueError(f"Unexpected token: {tok!r}")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def parse_expression(expr: str) -> _Expr:
    """Parse and compile an expression string into an AST.

    Parameters
    ----------
    expr : str
        Strategy expression, e.g. ``"RSI(14) < 30 and close > SMA(20)"``.

    Returns
    -------
    Compiled expression object (internal type).

    Raises
    ------
    ValueError
        If the expression cannot be parsed.

    Examples
    --------
    >>> from ferro_ta.tools.dsl import parse_expression
    >>> ast = parse_expression("RSI(14) < 30")
    >>> ast is not None
    True
    """
    if not isinstance(expr, str) or not expr.strip():
        raise ValueError("expr must be a non-empty string.")
    tokens = _tokenise(expr.strip())
    parser = _Parser(tokens)
    return parser.parse()


def evaluate(
    expr: Any,
    ohlcv: Any,
    *,
    close_col: str = "close",
    high_col: str = "high",
    low_col: str = "low",
    open_col: str = "open",
    volume_col: str = "volume",
) -> NDArray[np.int32]:
    """Evaluate a strategy expression against OHLCV data.

    Parameters
    ----------
    expr : str or compiled expression
        Either a strategy expression string or the result of
        :func:`parse_expression`.
    ohlcv : dict of arrays, pandas.DataFrame, or array-like
        OHLCV data.  At minimum ``close`` is required for indicator-only
        expressions.

    Returns
    -------
    numpy.ndarray of dtype int32 (values 0 or 1), same length as input.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools.dsl import evaluate
    >>> rng = np.random.default_rng(1)
    >>> close = np.cumprod(1 + rng.normal(0, 0.01, 60)) * 100
    >>> signal = evaluate("RSI(14) < 40", {"close": close})
    >>> set(signal.tolist()).issubset({0, 1})
    True
    """
    if isinstance(expr, str):
        ast = parse_expression(expr)
    else:
        ast = expr

    # Build context dict
    def _extract(col: str, key: str) -> Optional[NDArray]:
        try:
            import pandas as pd

            if isinstance(ohlcv, pd.DataFrame) and col in ohlcv.columns:
                return _to_f64(ohlcv[col].to_numpy())
        except ImportError:
            pass
        if isinstance(ohlcv, dict) and key in ohlcv:
            return _to_f64(ohlcv[key])
        return None

    ctx: dict[str, NDArray[np.float64]] = {}
    for col, key in [
        (close_col, "close"),
        (high_col, "high"),
        (low_col, "low"),
        (open_col, "open"),
        (volume_col, "volume"),
    ]:
        val = _extract(col, key)
        if val is not None:
            ctx[key] = val

    if "close" not in ctx and isinstance(ohlcv, np.ndarray):
        ctx["close"] = _to_f64(ohlcv)

    result = ast.eval(ctx)
    # Broadcast scalar to full length
    n = len(ctx.get("close", np.array([])))
    if result.shape == (1,) and n > 0:
        result = np.broadcast_to(result, (n,)).copy()

    # Convert to int32 signal while avoiding warnings when casting NaN/inf.
    # For numeric indicator outputs, treat non-finite values as "no signal" (0).
    if np.issubdtype(result.dtype, np.floating):
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
    return result.astype(np.int32)


class Strategy:
    """Convenience class for defining and evaluating a strategy expression.

    Parameters
    ----------
    expr : str
        Strategy expression string.

    Examples
    --------
    >>> import numpy as np
    >>> from ferro_ta.tools.dsl import Strategy
    >>> rng = np.random.default_rng(42)
    >>> close = np.cumprod(1 + rng.normal(0, 0.01, 100)) * 100
    >>> strat = Strategy("RSI(14) < 30")
    >>> signal = strat.evaluate({"close": close})
    >>> signal.shape
    (100,)
    """

    def __init__(self, expr: str) -> None:
        self.expr_str = expr
        self._ast = parse_expression(expr)

    def evaluate(self, ohlcv: Any, **kwargs: Any) -> NDArray[np.int32]:
        """Evaluate this strategy on *ohlcv* data."""
        return evaluate(self._ast, ohlcv, **kwargs)

    def __repr__(self) -> str:
        return f"Strategy({self.expr_str!r})"
