#!/usr/bin/env python3
"""
Check that python/ferro_ta/__init__.pyi matches the runtime ferro_ta package.

The type stub is hand-maintained, so it silently drifts from the runtime
signatures. Drift is not cosmetic: a stub parameter the runtime rejects makes a
type checker bless calls that raise ``TypeError``, and a stub parameter in the
wrong *position* makes a positional argument bind to a different runtime
parameter and return a wrong answer with no error at all.

This script AST-parses the stub, compares each declaration against
``inspect.signature()`` of the corresponding runtime object, and exits non-zero
if anything diverges. Findings are reported as one of:

  missing-at-runtime  the stub declares a name the runtime does not export
  missing-in-stub     the runtime exports a function the stub does not declare
  param-count         different number of parameters
  param-names         same count, different parameter names
  param-order         same names, different order
  param-kind          same names and order, different kind (e.g. keyword-only
                      in one and positional-or-keyword in the other, which
                      changes what a positional call binds to)
  param-default       same parameters, different default value
  param-annotation    same parameter, but the stub's annotated type is
                      incompatible with the runtime's (e.g. ``close: int``
                      against a runtime ``close: ArrayLike``)
  return-shape        one side returns a single value where the other returns
                      a tuple, or both return tuples of different length

Policy — what is deliberately *not* an error
--------------------------------------------
1. ``*args`` / ``**kwargs`` are compared as ordinary ordered parameters, by
   name *and* kind. A stub that faithfully declares ``*args: Any`` therefore
   matches the runtime instead of being reported as a count mismatch (the
   original throwaway audit false-positived on ``log_call`` and ``benchmark``
   for exactly this reason).
2. Stub default elision. In a stub, ``fmt: str = ...`` legitimately means "has
   a default, whose value is not restated here". A stub default of ``...``
   matches *any* runtime default. What is still enforced is default
   *presence*: a parameter with a default in one place and none in the other is
   a ``param-default`` finding, because that changes whether a call may omit it.
3. Class bodies. Only module-level ``def``s are compared. ``ferro_ta``'s
   streaming classes are native pyo3 types whose ``__init__`` reports
   ``(self, /, *args, **kwargs)`` to ``inspect``, so there is no signature to
   compare against and every constructor would be a false positive.
4. Non-routine runtime objects (classes, constants, submodules) and dunder
   names are skipped rather than compared.
5. ``@overload`` groups are skipped with a note, since a single runtime
   signature cannot be matched against several stub declarations.
6. Return *types* are not compared; only the return *shape* is (see below).
7. Parameter annotations are compared by *compatibility class*, not textually,
   and only in one direction (see below).

Policy — return annotations are compared by shape, not by type
--------------------------------------------------------------
Comparing return annotations textually is worse than not comparing them: the
stub is deliberately more precise than the runtime, so ``NDArray[np.float64]``
in the stub against a bare ``np.ndarray`` at runtime is *correct*, and string
equality would emit dozens of false positives. A noisy gate gets switched off,
so the check is deliberately narrow.

What is compared is the shape of the return:

  ``single``   one value (any non-tuple annotation, however written)
  ``tuple``    a tuple, plus its length — ``tuple[X, ...]`` counts as a tuple
               of unknown length and matches any tuple
  ``none``     an explicit ``-> None``

That is the failure mode that actually bites callers, and it is exactly the
one that got through: the stub promised ``PPO`` returned
``NDArray[np.float64]`` while the runtime returns a 3-tuple, so
``line, signal, hist = PPO(...)`` type-checked as an unpack of an array. Shape
comparison catches it and is immune to how richly either side spells its
element types.

Both sides are read *statically*. The stub's annotation comes from the AST;
the runtime's comes from ``inspect.signature().return_annotation``, which for
``ferro_ta`` is a string (the package uses ``from __future__ import
annotations``) and is parsed with ``ast``. Calling each function on synthetic
input to observe its real return shape was considered and rejected: every
runtime routine in ``__all__`` already carries a return annotation, so
execution would buy nothing, and it would make a CI signature linter import
*and run* 237 pieces of library code — needing per-function synthetic inputs
that satisfy each one's period and array-length constraints, and turning any
runtime bug into a parity "failure". If a runtime annotation is ever missing
(``Signature.empty``), that is an absence of information, not drift, and the
function's return is skipped rather than reported.

Names present in exactly one place *are* errors: a stub-only name is a promise
the runtime cannot keep, and a runtime function in ``__all__`` with no stub
declaration is unchecked public API. Runtime coverage is limited to
``__all__`` so that incidental re-exports are not flagged.

Policy — parameter annotations are compared by compatibility class
------------------------------------------------------------------
A stub saying ``close: int`` where the runtime says ``close: ArrayLike`` is the
same class of defect as the ``PPO`` return-type bug above: a type checker
blesses a call that cannot work. But textual comparison is unusable here for
the same reason it is unusable for returns — the stub is *deliberately* richer
than the runtime (``NDArray[np.float64]`` against ``np.ndarray``), and a noisy
gate gets switched off. So each annotation is bucketed into a coarse
compatibility class and only *cross-class* mismatches are reported:

  ``array``     ``ArrayLike``, ``np.ndarray``, ``NDArray[...]``, and
                ``Sequence``/``Iterable``/``list``/``tuple`` of a numeric type
  ``int``       ``int`` and the numpy integer scalar types
  ``float``     ``float`` and the numpy floating scalar types
  ``bool``      ``bool``, ``np.bool_``
  ``str``       ``str``
  ``bytes``     ``bytes``, ``bytearray``
  ``callable``  ``Callable``, ``Callable[...]``
  ``mapping``   ``dict[...]``, ``Mapping[...]``
  ``none``      ``None``
  ``unknown``   anything else

A union (``X | Y``, ``Optional[X]``, ``Union[X, Y]``) becomes the set of its
members' classes. Three rules keep the check sound without making it brittle:

1. **``unknown`` is absence of information, not drift.** ``Any``, type
   variables (the runtime's ``F`` vs. the stub's ``_F``), bare ``list``, and
   anything the bucketing does not recognise all land in ``unknown``, and a
   parameter whose *either* side is ``unknown`` is skipped. Same for a
   parameter carrying no annotation on one side: absence of information again.
2. **Direction matters.** The stub may be *narrower* than the runtime — that is
   a legitimate documentation choice, and the real stub uses it: ``info``
   declares ``func_or_name: Callable[..., Any] | str`` against a runtime
   ``Any``. So the requirement is that every class the stub admits is one the
   runtime also admits; a stub that is *wider* or *incompatible* is the
   finding, since only that direction blesses calls the runtime rejects.
3. **The numeric tower is honoured**, since Python's is: a stub ``int`` is
   accepted against a runtime ``float``, and ``bool`` against ``int`` or
   ``float``. Scalars are deliberately *not* accepted against ``array``, even
   though ``ArrayLike`` technically admits scalars — that is exactly the
   ``close: int`` defect this check exists to catch.

Optionality is folded in as an ordinary class (``none``), with one
concession: a runtime parameter defaulting to ``None`` is treated as admitting
``None`` even if its annotation does not say so, so a stub ``X | None`` against
a runtime ``x: X = None`` is not reported.

Known blind spots
-----------------
What this check still does not see, as of the annotation work above:

* Stub classes are entirely unchecked (12 of them), for the pyo3 reason in
  policy note 3 — their bodies, not just their ``__init__``, are invisible.
* ``@overload`` groups are dropped rather than merely left uncompared: an
  overloaded name is not even checked for existence at runtime. (The stub
  declares none today, so the note is about the next one added.)
* Module-level constants are invisible in both directions (a stub-declared
  constant the runtime lacks, and vice versa) — the stub's only one is
  ``__version__: str``.
* Defaults that are not literals (``= SomeEnum.X``) are treated as elided, so
  only their *presence* is checked, not their value.
* ``async def`` vs. ``def`` is not detected; the two are parsed identically.
* Only ``__init__.pyi`` is compared. Submodule stubs, if any are added, are
  not checked.
* Return comparison remains shape-only, so two same-shape returns with
  different element protocols (a 3-tuple of arrays vs. a 3-tuple of floats)
  still pass. The parameter-annotation classes above are *not* applied to
  return elements.

Closed by the annotation comparison: a parameter whose stub type was
incompatible with the runtime's used to pass silently; it is now a
``param-annotation`` finding.
"""

from __future__ import annotations

import argparse
import ast
import inspect
import sys
from pathlib import Path
from typing import Any, NamedTuple, get_args, get_origin

STUB_REL = Path("python") / "ferro_ta" / "__init__.pyi"

# Sentinel for "this parameter has no default at all".
NO_DEFAULT = object()
# Sentinel for a stub default written as `...` (elided, matches any value).
ELIDED_DEFAULT = object()


class Param(NamedTuple):
    """One parameter, normalised so stub and runtime are comparable."""

    name: str
    kind: str
    default: Any
    # The annotation exactly as written, or None when there is none to compare.
    annotation: str | None = None

    @property
    def has_default(self) -> bool:
        return self.default is not NO_DEFAULT

    def render(self) -> str:
        prefix = {"VAR_POSITIONAL": "*", "VAR_KEYWORD": "**"}.get(self.kind, "")
        if not self.has_default:
            return f"{prefix}{self.name}"
        if self.default is ELIDED_DEFAULT:
            return f"{prefix}{self.name}=..."
        return f"{prefix}{self.name}={self.default!r}"


class ReturnShape(NamedTuple):
    """The shape of a return annotation, normalised for comparison.

    ``kind`` is ``"single"``, ``"tuple"`` or ``"none"``; ``arity`` is the tuple
    length, or ``None`` for a variadic ``tuple[X, ...]`` whose length is
    unknown and therefore matches any tuple. ``text`` is the annotation as
    written, so failure messages can show both sides verbatim.
    """

    kind: str
    arity: int | None
    text: str

    def render(self) -> str:
        if self.kind == "tuple":
            length = "?" if self.arity is None else str(self.arity)
            return f"tuple of {length}  [{self.text}]"
        if self.kind == "none":
            return f"no value  [{self.text}]"
        return f"single value  [{self.text}]"


class Finding(NamedTuple):
    name: str
    kind: str
    detail: str
    stub: str
    runtime: str


def render_signature(params: list[Param]) -> str:
    return "(" + ", ".join(p.render() for p in params) + ")"


# ---------------------------------------------------------------------------
# Return shapes
# ---------------------------------------------------------------------------

TUPLE_NAMES = frozenset({"tuple", "Tuple"})


def _annotation_root(node: ast.expr) -> str:
    """Name a subscript is applied to, ignoring any module qualifier."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


def shape_from_ast(node: ast.expr | None) -> ReturnShape | None:
    """Derive a return shape from an annotation expression.

    ``None`` means "no information" (no annotation at all), which is never a
    finding — see the return-annotation policy in the module docstring.
    """
    if node is None:
        return None
    text = ast.unparse(node)
    if isinstance(node, ast.Constant) and node.value is None:
        return ReturnShape("none", 0, text)
    if isinstance(node, ast.Name) and node.id == "None":
        return ReturnShape("none", 0, text)
    if isinstance(node, ast.Subscript) and _annotation_root(node.value) in TUPLE_NAMES:
        elements = (
            list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        )
        variadic = any(
            isinstance(element, ast.Constant) and element.value is Ellipsis
            for element in elements
        )
        return ReturnShape("tuple", None if variadic else len(elements), text)
    if _annotation_root(node) in TUPLE_NAMES:  # bare `tuple`, length unknown
        return ReturnShape("tuple", None, text)
    return ReturnShape("single", 1, text)


def shape_from_annotation(annotation: Any) -> ReturnShape | None:
    """Derive a return shape from a runtime ``return_annotation``.

    ``ferro_ta`` uses ``from __future__ import annotations``, so these arrive
    as strings and are parsed with ``ast``; real objects are handled too, for
    robustness against a module that does not postpone its annotations.
    """
    if annotation is inspect.Signature.empty:
        return None
    if annotation is None or annotation is type(None):
        return ReturnShape("none", 0, "None")
    if isinstance(annotation, str):
        try:
            return shape_from_ast(ast.parse(annotation, mode="eval").body)
        except SyntaxError:
            return None
    origin = get_origin(annotation)
    if origin is tuple:
        arguments = get_args(annotation)
        variadic = Ellipsis in arguments
        return ReturnShape(
            "tuple",
            None if variadic or not arguments else len(arguments),
            str(annotation),
        )
    return ReturnShape("single", 1, str(annotation))


def compare_returns(
    name: str, stub: ReturnShape | None, runtime: ReturnShape | None
) -> list[Finding]:
    """Compare two return shapes, ignoring how richly either spells its types."""
    if stub is None or runtime is None:
        return []  # absence of information, not drift
    if stub.kind != runtime.kind:
        detail = f"stub returns a {stub.kind}, runtime returns a {runtime.kind}"
    elif stub.kind == "tuple" and not (
        stub.arity is None or runtime.arity is None or stub.arity == runtime.arity
    ):
        detail = (
            f"stub returns a {stub.arity}-tuple, runtime returns a "
            f"{runtime.arity}-tuple"
        )
    else:
        return []
    return [Finding(name, "return-shape", detail, stub.render(), runtime.render())]


# ---------------------------------------------------------------------------
# Parameter annotations: compatibility classes
# ---------------------------------------------------------------------------

UNKNOWN = "unknown"

# Leaf names (module qualifier ignored) mapped to their compatibility class.
LEAF_CLASSES = {
    "ArrayLike": "array",
    "NDArray": "array",
    "ndarray": "array",
    "int": "int",
    "integer": "int",
    "signedinteger": "int",
    "unsignedinteger": "int",
    "intp": "int",
    "int8": "int",
    "int16": "int",
    "int32": "int",
    "int64": "int",
    "uint8": "int",
    "uint16": "int",
    "uint32": "int",
    "uint64": "int",
    "float": "float",
    "floating": "float",
    "double": "float",
    "float16": "float",
    "float32": "float",
    "float64": "float",
    "bool": "bool",
    "bool_": "bool",
    "str": "str",
    "bytes": "bytes",
    "bytearray": "bytes",
    "Callable": "callable",
    "None": "none",
    "NoneType": "none",
}

# Containers that count as array-like when their elements are numeric.
SEQUENCE_ROOTS = frozenset(
    {"Sequence", "MutableSequence", "Iterable", "Collection", "list", "tuple", "List"}
)
MAPPING_ROOTS = frozenset({"dict", "Dict", "Mapping", "MutableMapping"})
NUMERIC_CLASSES = frozenset({"int", "float", "bool"})

# What a stub class may legitimately be, given the runtime's class: Python's
# numeric tower, and nothing else. Scalars are *not* widened into "array".
ACCEPTED_BY = {
    "bool": frozenset({"bool", "int", "float"}),
    "int": frozenset({"int", "float"}),
}


def annotation_text(annotation: Any) -> str | None:
    """Normalise a runtime annotation to source text, or None if absent.

    ``ferro_ta`` postpones its annotations, so these arrive as strings; real
    objects are handled too for robustness against a module that does not.
    """
    if annotation is inspect.Parameter.empty:
        return None
    if annotation is None or annotation is type(None):
        return "None"
    if isinstance(annotation, str):
        return annotation
    return getattr(annotation, "__name__", None) or str(annotation)


def _subscript_elements(node: ast.Subscript) -> list[ast.expr]:
    if isinstance(node.slice, ast.Tuple):
        return list(node.slice.elts)
    return [node.slice]


def _classes_of(node: ast.expr) -> frozenset[str]:
    """Compatibility classes admitted by one annotation expression."""
    if isinstance(node, ast.Constant):
        return frozenset({"none"}) if node.value is None else frozenset({UNKNOWN})
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return _classes_of(node.left) | _classes_of(node.right)
    if isinstance(node, ast.Subscript):
        root = _annotation_root(node.value)
        elements = _subscript_elements(node)
        if root == "Optional":
            return _classes_of(elements[0]) | {"none"}
        if root == "Union":
            return frozenset().union(*(_classes_of(e) for e in elements))
        if root == "Annotated":
            return _classes_of(elements[0])
        if root in SEQUENCE_ROOTS:
            items = [
                _classes_of(element)
                for element in elements
                if not (isinstance(element, ast.Constant) and element.value is Ellipsis)
            ]
            item_classes = frozenset().union(*items) if items else frozenset()
            if item_classes and item_classes <= NUMERIC_CLASSES:
                return frozenset({"array"})
            return frozenset({UNKNOWN})
        if root in MAPPING_ROOTS:
            return frozenset({"mapping"})
        return frozenset({LEAF_CLASSES.get(root, UNKNOWN)})
    root = _annotation_root(node)
    if root in MAPPING_ROOTS:
        return frozenset({"mapping"})
    return frozenset({LEAF_CLASSES.get(root, UNKNOWN)})


def classify_annotation(text: str | None) -> frozenset[str]:
    """Bucket an annotation into compatibility classes.

    An empty or unparsable annotation, and anything the buckets do not
    recognise, yields ``{"unknown"}`` — absence of information, never drift.
    """
    if text is None:
        return frozenset({UNKNOWN})
    try:
        return _classes_of(ast.parse(text, mode="eval").body)
    except SyntaxError:
        return frozenset({UNKNOWN})


def annotations_match(stub: Param, runtime: Param) -> bool:
    """Whether the stub's annotated type is compatible with the runtime's.

    Direction-aware: the stub may admit *fewer* types than the runtime, never
    more. See the annotation policy in the module docstring.
    """
    stub_classes = classify_annotation(stub.annotation)
    runtime_classes = classify_annotation(runtime.annotation)
    if UNKNOWN in stub_classes or UNKNOWN in runtime_classes:
        return True  # absence of information, not drift
    if runtime.default is None:
        runtime_classes = runtime_classes | {"none"}
    return all(
        ACCEPTED_BY.get(cls, frozenset({cls})) & runtime_classes for cls in stub_classes
    )


# ---------------------------------------------------------------------------
# Stub parsing
# ---------------------------------------------------------------------------


def _stub_default(node: ast.expr | None) -> Any:
    """Normalise a stub default expression to a comparable value."""
    if node is None:
        return NO_DEFAULT
    if isinstance(node, ast.Constant) and node.value is Ellipsis:
        return ELIDED_DEFAULT
    try:
        return ast.literal_eval(node)
    except (ValueError, SyntaxError):
        # e.g. `= SomeEnum.X`; treat as elided rather than inventing a value.
        return ELIDED_DEFAULT


def _stub_annotation(arg: ast.arg) -> str | None:
    return None if arg.annotation is None else ast.unparse(arg.annotation)


def stub_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[Param]:
    args = node.args
    params: list[Param] = []

    positional = list(args.posonlyargs) + list(args.args)
    pad = len(positional) - len(args.defaults)
    for index, arg in enumerate(positional):
        default = _stub_default(args.defaults[index - pad] if index >= pad else None)
        kind = (
            "POSITIONAL_ONLY"
            if index < len(args.posonlyargs)
            else "POSITIONAL_OR_KEYWORD"
        )
        params.append(Param(arg.arg, kind, default, _stub_annotation(arg)))

    if args.vararg is not None:
        params.append(
            Param(
                args.vararg.arg,
                "VAR_POSITIONAL",
                NO_DEFAULT,
                _stub_annotation(args.vararg),
            )
        )

    for arg, kw_default in zip(args.kwonlyargs, args.kw_defaults):
        params.append(
            Param(
                arg.arg,
                "KEYWORD_ONLY",
                _stub_default(kw_default),
                _stub_annotation(arg),
            )
        )

    if args.kwarg is not None:
        params.append(
            Param(
                args.kwarg.arg, "VAR_KEYWORD", NO_DEFAULT, _stub_annotation(args.kwarg)
            )
        )

    return params


def parse_stub(
    path: Path,
) -> tuple[dict[str, list[Param]], dict[str, ReturnShape | None], list[str]]:
    """Return (functions, return shapes, names skipped as @overload groups)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    functions: dict[str, list[Param]] = {}
    returns: dict[str, ReturnShape | None] = {}
    seen: dict[str, int] = {}
    for node in tree.body:  # module level only; see policy note 3
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        seen[node.name] = seen.get(node.name, 0) + 1
        functions[node.name] = stub_params(node)
        returns[node.name] = shape_from_ast(node.returns)
    overloaded = sorted(name for name, count in seen.items() if count > 1)
    for name in overloaded:
        functions.pop(name, None)
        returns.pop(name, None)
    return functions, returns, overloaded


# ---------------------------------------------------------------------------
# Runtime introspection
# ---------------------------------------------------------------------------


def runtime_params(func: Any) -> list[Param] | None:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    return [
        Param(
            parameter.name,
            parameter.kind.name,
            NO_DEFAULT
            if parameter.default is inspect.Parameter.empty
            else parameter.default,
            annotation_text(parameter.annotation),
        )
        for parameter in signature.parameters.values()
    ]


def runtime_return(func: Any) -> ReturnShape | None:
    try:
        signature = inspect.signature(func)
    except (TypeError, ValueError):
        return None
    return shape_from_annotation(signature.return_annotation)


def runtime_functions(module: Any) -> dict[str, Any]:
    names = getattr(module, "__all__", None) or [
        name for name in dir(module) if not name.startswith("_")
    ]
    functions: dict[str, Any] = {}
    for name in names:
        if name.startswith("__"):
            continue
        obj = getattr(module, name, None)
        if obj is not None and inspect.isroutine(obj):
            functions[name] = obj
    return functions


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------


def defaults_match(stub: Param, runtime: Param) -> bool:
    if stub.has_default != runtime.has_default:
        return False
    if not stub.has_default:
        return True
    if stub.default is ELIDED_DEFAULT:  # policy note 2
        return True
    if stub.default == runtime.default:
        return True
    return repr(stub.default) == repr(runtime.default)


def compare(name: str, stub: list[Param], runtime: list[Param]) -> list[Finding]:
    rendered_stub = render_signature(stub)
    rendered_runtime = render_signature(runtime)

    def finding(kind: str, detail: str) -> Finding:
        return Finding(name, kind, detail, rendered_stub, rendered_runtime)

    stub_names = [p.name for p in stub]
    runtime_names = [p.name for p in runtime]

    if len(stub) != len(runtime):
        return [
            finding(
                "param-count",
                f"stub declares {len(stub)} parameter(s), runtime takes {len(runtime)}",
            )
        ]
    if stub_names != runtime_names:
        if sorted(stub_names) == sorted(runtime_names):
            return [finding("param-order", f"{stub_names} vs {runtime_names}")]
        only_stub = [n for n in stub_names if n not in runtime_names]
        only_runtime = [n for n in runtime_names if n not in stub_names]
        return [
            finding(
                "param-names",
                f"stub-only {only_stub}, runtime-only {only_runtime}",
            )
        ]

    findings: list[Finding] = []
    for stub_param, runtime_param in zip(stub, runtime):
        if stub_param.kind != runtime_param.kind:
            findings.append(
                finding(
                    "param-kind",
                    f"{stub_param.name}: stub is {stub_param.kind}, "
                    f"runtime is {runtime_param.kind}",
                )
            )
        elif not defaults_match(stub_param, runtime_param):
            findings.append(
                finding(
                    "param-default",
                    f"{stub_param.name}: stub default {stub_param.render()}, "
                    f"runtime default {runtime_param.render()}",
                )
            )
        if not annotations_match(stub_param, runtime_param):
            stub_classes = sorted(classify_annotation(stub_param.annotation))
            runtime_classes = sorted(classify_annotation(runtime_param.annotation))
            findings.append(
                Finding(
                    name,
                    "param-annotation",
                    f"{stub_param.name}: stub admits {stub_classes}, "
                    f"runtime admits {runtime_classes}",
                    f"{stub_param.name}: {stub_param.annotation}",
                    f"{runtime_param.name}: {runtime_param.annotation}",
                )
            )
    return findings


def check(stub_path: Path, module: Any) -> tuple[list[Finding], list[str], int]:
    stub_functions, stub_returns, overloaded = parse_stub(stub_path)
    runtime = runtime_functions(module)

    findings: list[Finding] = []
    compared = 0

    for name in sorted(stub_functions):
        func = getattr(module, name, None)
        if func is None:
            findings.append(
                Finding(
                    name,
                    "missing-at-runtime",
                    "declared in stub only",
                    "declared",
                    "absent",
                )
            )
            continue
        if not inspect.isroutine(func):
            continue  # policy note 4
        params = runtime_params(func)
        if params is None:
            continue  # no introspectable signature; nothing to compare
        compared += 1
        findings.extend(compare(name, stub_functions[name], params))
        findings.extend(
            compare_returns(name, stub_returns.get(name), runtime_return(func))
        )

    for name in sorted(runtime):
        if name not in stub_functions and name not in overloaded:
            findings.append(
                Finding(
                    name,
                    "missing-in-stub",
                    "exported at runtime only",
                    "absent",
                    "exported",
                )
            )

    return findings, overloaded, compared


def import_ferro_ta(root: Path) -> Any | None:
    """Import ferro_ta, preferring an installed build over the source tree.

    In CI the wheel is installed and `python/ferro_ta/` holds no compiled
    extension, so putting the source tree first on sys.path would shadow the
    working package with an unimportable one. Locally (maturin develop, or an
    in-tree `.so`) there is no installed package, so the source tree is the
    fallback.
    """
    try:
        import ferro_ta

        return ferro_ta
    except ImportError:
        pass

    python_root = str(root / "python")
    if python_root not in sys.path:
        sys.path.insert(0, python_root)
    try:
        import ferro_ta

        return ferro_ta
    except ImportError:
        return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument(
        "--stub",
        type=Path,
        default=None,
        help=f"path to the type stub (default: {STUB_REL})",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]

    stub_path = args.stub if args.stub is not None else root / STUB_REL
    if not stub_path.exists():
        print(f"Type stub not found: {stub_path}")
        return 1

    ferro_ta = import_ferro_ta(root)
    if ferro_ta is None:
        print(
            "Cannot import ferro_ta, either as an installed package or from "
            f"{root / 'python'}.\n"
            "The compiled extension must be built before this check can run:\n"
            "  maturin develop   # or: pip install ."
        )
        return 1

    findings, overloaded, compared = check(stub_path, ferro_ta)

    if findings:
        print(f"{len(findings)} stub/runtime signature mismatch(es) in {stub_path}:\n")
        for item in sorted(findings, key=lambda f: (f.kind, f.name)):
            print(f"  {item.name} [{item.kind}]: {item.detail}")
            print(f"    stub:    {item.stub}")
            print(f"    runtime: {item.runtime}")
        print(
            f"\nUpdate {STUB_REL} so it matches the runtime signatures "
            "(or fix the runtime, if the stub is right)."
        )
        return 1

    note = f" ({len(overloaded)} @overload group(s) skipped)" if overloaded else ""
    print(
        f"{stub_path} matches the runtime signatures: {compared} function(s) checked{note}."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
