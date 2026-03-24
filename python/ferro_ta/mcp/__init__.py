"""Optional MCP server exposing the public ferro-ta API."""

from __future__ import annotations

import dataclasses
import enum
import importlib
import inspect
import json
import re
from collections.abc import Callable, Mapping
from datetime import date, datetime, time
from functools import lru_cache
from itertools import count
from typing import Any, get_args, get_origin, get_type_hints

import numpy as np

import ferro_ta
from ferro_ta.tools import compute_indicator, run_backtest
from ferro_ta.tools.api_info import methods as api_methods

__all__ = ["create_server", "run_server", "handle_list_tools", "handle_call_tool"]

_MCP_INSTALL_HINT = (
    "The ferro-ta MCP server requires the optional 'mcp' dependency. "
    'Install it with `pip install "ferro-ta[mcp]"` or `uv sync --extra mcp`.'
)

_REFERENCE_HELP = (
    "Use {'instance_id': '<id>'} for stored objects or "
    "{'callable': '<public ferro-ta name>'} for public callables."
)

_JSON_ANY_TYPE: list[str] = [
    "array",
    "boolean",
    "integer",
    "null",
    "number",
    "object",
    "string",
]

_NO_DEFAULT = object()
_INSTANCE_STORE: dict[str, Any] = {}
_INSTANCE_META: dict[str, dict[str, Any]] = {}
_INSTANCE_COUNTER = count(1)


@dataclasses.dataclass(frozen=True)
class _ToolSpec:
    """Resolved metadata and dispatcher for a single MCP tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    wrapper_signature: inspect.Signature
    invoke: Callable[[dict[str, Any]], Any]


def _import_object(module_name: str, name: str) -> Any:
    """Import *name* from *module_name*."""
    module = importlib.import_module(module_name)
    return getattr(module, name)


@lru_cache(maxsize=1)
def _discover_public_callables() -> tuple[list[dict[str, str]], dict[str, Any]]:
    """Return canonical public callable metadata and lookup targets."""
    entries = api_methods()
    top_level = sorted(
        [item for item in entries if item["category"] == "top_level"],
        key=lambda item: item["name"],
    )
    top_level_names = {item["name"] for item in top_level}
    extras = sorted(
        [
            item
            for item in entries
            if item["category"] != "top_level" and item["name"] not in top_level_names
        ],
        key=lambda item: item["name"],
    )

    public_entries: list[dict[str, str]] = []
    targets: dict[str, Any] = {}
    for item in [*top_level, *extras]:
        name = item["name"]
        if name in targets:
            continue
        targets[name] = _import_object(item["module"], name)
        public_entries.append(item)

    return public_entries, targets


def _slugify(value: str) -> str:
    """Convert *value* into a short identifier fragment."""
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "object"


def _instance_ref(value: Any, *, source_tool: str) -> dict[str, Any]:
    """Store *value* and return a serialisable reference payload."""
    identifier = f"{_slugify(type(value).__name__)}-{next(_INSTANCE_COUNTER):04d}"
    _INSTANCE_STORE[identifier] = value
    payload = {
        "instance_id": identifier,
        "type": f"{type(value).__module__}.{type(value).__name__}",
        "repr": repr(value),
        "callable": callable(value),
        "source_tool": source_tool,
    }
    snapshot = _object_snapshot(value)
    if snapshot is not None:
        payload["snapshot"] = snapshot
    _INSTANCE_META[identifier] = payload
    return payload


def _get_instance(identifier: str) -> Any:
    """Return the stored instance for *identifier* or raise a clear error."""
    try:
        return _INSTANCE_STORE[identifier]
    except KeyError as exc:
        raise KeyError(f"Unknown instance_id: {identifier!r}") from exc


def _is_instance_ref(value: Any) -> bool:
    """Return whether *value* is a stored-object reference payload."""
    return isinstance(value, dict) and set(value) == {"instance_id"}


def _is_callable_ref(value: Any) -> bool:
    """Return whether *value* is a public-callable reference payload."""
    return isinstance(value, dict) and set(value) == {"callable"}


def _public_method_summaries(value: Any) -> list[dict[str, str]]:
    """Return public callable methods for *value*."""
    result: list[dict[str, str]] = []
    for method_name, member in inspect.getmembers(value):
        if method_name.startswith("_") or not callable(member):
            continue
        try:
            signature = str(inspect.signature(member))
        except (TypeError, ValueError):
            signature = "()"
        result.append({"name": method_name, "signature": signature})
    return result


def _object_snapshot(value: Any) -> Any:
    """Return a serialisable snapshot for common non-primitive objects."""
    if isinstance(value, enum.Enum):
        return value.value

    if dataclasses.is_dataclass(value):
        return _normalise_json(dataclasses.asdict(value), store_objects=False)

    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            return _normalise_json(value.to_dict(), store_objects=False)
        except TypeError:
            if value.__class__.__module__.startswith("pandas"):
                return _normalise_json(value.to_dict(orient="list"), store_objects=False)
            if value.__class__.__module__.startswith("polars"):
                return _normalise_json(
                    value.to_dict(as_series=False), store_objects=False
                )
        except Exception:
            return None

    if hasattr(value, "__dict__"):
        fields = {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_") and not callable(val)
        }
        if fields:
            return _normalise_json(fields, store_objects=False)

    slots = getattr(type(value), "__slots__", ())
    if slots:
        fields = {}
        for slot in slots:
            if slot.startswith("_") or not hasattr(value, slot):
                continue
            slot_value = getattr(value, slot)
            if callable(slot_value):
                continue
            fields[slot] = slot_value
        if fields:
            return _normalise_json(fields, store_objects=False)

    return None


def _normalise_json(value: Any, *, store_objects: bool = True) -> Any:
    """Convert Python and numpy-rich values into JSON-safe values."""
    if value is None or isinstance(value, (str, bool)):
        return value

    if isinstance(value, (int, np.integer)):
        return int(value)

    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else float(value)

    if isinstance(value, (date, datetime, time)):
        return value.isoformat()

    if isinstance(value, enum.Enum):
        return _normalise_json(value.value, store_objects=store_objects)

    if isinstance(value, np.ndarray):
        return [_normalise_json(item, store_objects=store_objects) for item in value.tolist()]

    if isinstance(value, np.generic):
        return _normalise_json(value.item(), store_objects=store_objects)

    if isinstance(value, Mapping):
        return {
            str(key): _normalise_json(item, store_objects=store_objects)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple, set, frozenset)):
        return [_normalise_json(item, store_objects=store_objects) for item in value]

    if hasattr(value, "tolist") and callable(value.tolist):
        try:
            return _normalise_json(value.tolist(), store_objects=store_objects)
        except Exception:
            pass

    snapshot = _object_snapshot(value)
    if snapshot is not None:
        return snapshot

    if store_objects:
        return _instance_ref(value, source_tool="return_value")

    return repr(value)


def _json_result(
    payload: Any,
    *,
    structured_key: str | None = None,
) -> dict[str, Any]:
    """Wrap *payload* in the helper response shape."""
    structured: dict[str, Any] | None = None
    if isinstance(payload, dict):
        structured = payload
    elif structured_key is not None:
        structured = {structured_key: payload}

    result: dict[str, Any] = {
        "content": [{"type": "text", "text": json.dumps(payload)}],
    }
    if structured is not None:
        result["structuredContent"] = structured
    return result


def _text_result(
    text: str,
    *,
    structured: dict[str, Any] | None = None,
    is_error: bool = False,
) -> dict[str, Any]:
    """Wrap *text* in the helper response shape."""
    result: dict[str, Any] = {
        "content": [{"type": "text", "text": text}],
    }
    if structured is not None:
        result["structuredContent"] = structured
    if is_error:
        result["isError"] = True
    return result


def _response_from_payload(payload: Any) -> dict[str, Any]:
    """Create a helper response from a normalised payload."""
    if isinstance(payload, str):
        return _text_result(payload, structured={"value": payload})
    return _json_result(payload)


def _load_mcp_sdk() -> tuple[type[Any], type[Any], type[Any]]:
    """Import the optional MCP SDK lazily."""
    try:
        fastmcp = importlib.import_module("mcp.server.fastmcp")
        types = importlib.import_module("mcp.types")
    except ImportError as exc:  # pragma: no cover - exercised via tests
        raise RuntimeError(_MCP_INSTALL_HINT) from exc

    return fastmcp.FastMCP, types.CallToolResult, types.TextContent


def _to_call_tool_result(result: dict[str, Any]) -> Any:
    """Convert helper-style results into an MCP SDK CallToolResult."""
    _, call_tool_result_type, text_content_type = _load_mcp_sdk()
    content = [
        text_content_type(type="text", text=item["text"])
        for item in result.get("content", [])
        if item.get("type") == "text"
    ]
    return call_tool_result_type(
        content=content,
        structuredContent=result.get("structuredContent"),
        isError=result.get("isError", False),
    )


def _safe_default(value: Any) -> Any:
    """Return a JSON-safe schema default or a sentinel when unavailable."""
    if value is inspect._empty:
        return _NO_DEFAULT
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if isinstance(value, tuple) and all(
        isinstance(item, (str, bool, int, float)) or item is None for item in value
    ):
        return list(value)
    if isinstance(value, list) and all(
        isinstance(item, (str, bool, int, float)) or item is None for item in value
    ):
        return value
    return _NO_DEFAULT


def _wrapper_default(value: Any) -> Any:
    """Return a lightweight default for generated wrapper signatures."""
    default = _safe_default(value)
    if default is _NO_DEFAULT:
        return None
    return default


def _annotation_label(annotation: Any) -> str:
    """Return a readable label for *annotation*."""
    if annotation is inspect._empty:
        return "Any"
    if isinstance(annotation, str):
        return annotation
    origin = get_origin(annotation)
    if origin is not None:
        return str(annotation).replace("typing.", "")
    return getattr(annotation, "__name__", repr(annotation))


def _annotation_options(annotation: Any) -> list[Any]:
    """Flatten simple union annotations into a list of options."""
    if annotation is inspect._empty:
        return [Any]

    origin = get_origin(annotation)
    if origin in (None,):
        return [annotation]

    if origin in (list, tuple, dict):
        return [annotation]

    if origin in (Callable,):
        return [annotation]

    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    return args or [annotation]


def _is_enum_annotation(annotation: Any) -> type[enum.Enum] | None:
    """Return the enum class inside *annotation*, if any."""
    for option in _annotation_options(annotation):
        if inspect.isclass(option) and issubclass(option, enum.Enum):
            return option
    return None


def _annotation_includes_callable(annotation: Any) -> bool:
    """Return whether *annotation* includes a callable type."""
    label = _annotation_label(annotation)
    if "Callable" in label:
        return True
    for option in _annotation_options(annotation):
        origin = get_origin(option)
        if origin in (Callable,):
            return True
    return False


def _annotation_includes_custom_class(annotation: Any) -> bool:
    """Return whether *annotation* includes a non-builtin class."""
    for option in _annotation_options(annotation):
        if not inspect.isclass(option):
            continue
        if issubclass(option, enum.Enum):
            return True
        if option.__module__ == "builtins":
            continue
        if option in (date, datetime, time):
            continue
        return True
    return False


def _schema_and_py_type(annotation: Any, *, param_name: str) -> tuple[dict[str, Any], Any]:
    """Map Python annotations to JSON Schema and wrapper annotations."""
    enum_type = _is_enum_annotation(annotation)
    if enum_type is not None:
        raw_values = [member.value for member in enum_type]
        if all(isinstance(item, str) for item in raw_values):
            schema = {"type": "string", "enum": list(raw_values)}
            return schema, str
        if all(isinstance(item, int) for item in raw_values):
            schema = {"type": "integer", "enum": list(raw_values)}
            return schema, int

    label = _annotation_label(annotation)
    lower = label.lower()

    if "bool" in lower:
        return {"type": "boolean"}, bool
    if "int" in lower and "point" not in lower:
        return {"type": "integer"}, int
    if (
        "float" in lower
        or "number" in lower
        or "scalar" in lower
        or "ndarray" in lower
        or "arraylike" in lower
    ):
        if "scalarorarray" in lower:
            return {
                "type": _JSON_ANY_TYPE,
                "description": f"Parameter `{param_name}`. { _REFERENCE_HELP }",
            }, Any
        if "ndarray" in lower or "arraylike" in lower:
            return {"type": "array", "items": {}}, list[Any]
        return {"type": "number"}, float
    if "list" in lower or "tuple" in lower or "sequence" in lower or "iterable" in lower:
        return {"type": "array", "items": {}}, list[Any]
    if "dict" in lower or "mapping" in lower:
        return {"type": "object"}, dict[str, Any]
    if "str" in lower or "date" in lower or "datetime" in lower or "time" in lower:
        return {"type": "string"}, str
    if _annotation_includes_callable(annotation) or _annotation_includes_custom_class(annotation):
        return {
            "type": _JSON_ANY_TYPE,
            "description": f"Parameter `{param_name}`. {_REFERENCE_HELP}",
        }, Any
    return {"type": _JSON_ANY_TYPE}, Any


def _build_signature_and_schema(
    signature: inspect.Signature,
    *,
    type_hints: dict[str, Any],
) -> tuple[inspect.Signature, dict[str, Any]]:
    """Build a wrapper signature and JSON schema for *signature*."""
    wrapper_parameters: list[inspect.Parameter] = []
    properties: dict[str, Any] = {}
    required: list[str] = []

    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            properties["args"] = {
                "type": "array",
                "items": {},
                "description": "Extra positional arguments.",
            }
            wrapper_parameters.append(
                inspect.Parameter(
                    "args",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=list[Any],
                    default=None,
                )
            )
            continue

        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            properties["kwargs"] = {
                "type": "object",
                "description": "Extra keyword arguments.",
            }
            wrapper_parameters.append(
                inspect.Parameter(
                    "kwargs",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=dict[str, Any],
                    default=None,
                )
            )
            continue

        annotation = type_hints.get(parameter.name, parameter.annotation)
        schema, py_type = _schema_and_py_type(annotation, param_name=parameter.name)
        default = _safe_default(parameter.default)
        if default is not _NO_DEFAULT:
            schema["default"] = default
        else:
            required.append(parameter.name)

        properties[parameter.name] = schema
        wrapper_parameters.append(
            inspect.Parameter(
                parameter.name,
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=py_type,
                default=(
                    inspect._empty
                    if parameter.default is inspect._empty
                    else _wrapper_default(parameter.default)
                ),
            )
        )

    wrapper_signature = inspect.Signature(parameters=wrapper_parameters)
    input_schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return wrapper_signature, input_schema


def _type_hints_for_target(target: Any) -> dict[str, Any]:
    """Return best-effort type hints for *target*."""
    hinted = target.__init__ if inspect.isclass(target) else target
    try:
        return get_type_hints(hinted, include_extras=True)
    except Exception:
        return {}


def _resolve_public_callable(name: str) -> Any:
    """Resolve a public ferro-ta callable by its exposed MCP name."""
    _, targets = _discover_public_callables()
    if name in targets:
        return targets[name]
    aliases = {
        "sma": ferro_ta.SMA,
        "ema": ferro_ta.EMA,
        "rsi": ferro_ta.RSI,
        "macd": ferro_ta.MACD,
        "backtest": run_backtest,
    }
    try:
        return aliases[name]
    except KeyError as exc:
        raise KeyError(f"Unknown callable reference: {name!r}") from exc


def _coerce_enum(value: Any, enum_type: type[enum.Enum]) -> enum.Enum:
    """Convert *value* into *enum_type*."""
    if isinstance(value, enum_type):
        return value
    if isinstance(value, str):
        if value in enum_type.__members__:
            return enum_type[value]
        for member in enum_type:
            if member.value == value:
                return member
    return enum_type(value)


def _decode_value(value: Any, annotation: Any = Any) -> Any:
    """Resolve stored-object and callable references inside *value*."""
    if _is_instance_ref(value):
        return _get_instance(str(value["instance_id"]))

    if _is_callable_ref(value):
        return _resolve_public_callable(str(value["callable"]))

    enum_type = _is_enum_annotation(annotation)
    if enum_type is not None:
        try:
            return _coerce_enum(value, enum_type)
        except Exception:
            pass

    if _annotation_includes_callable(annotation) and isinstance(value, str):
        try:
            return _resolve_public_callable(value)
        except KeyError:
            pass

    if isinstance(value, list):
        return [_decode_value(item, Any) for item in value]

    if isinstance(value, tuple):
        return tuple(_decode_value(item, Any) for item in value)

    if isinstance(value, dict):
        return {str(key): _decode_value(item, Any) for key, item in value.items()}

    return value


def _invoke_target(
    target: Any,
    *,
    signature: inspect.Signature,
    type_hints: dict[str, Any],
    arguments: dict[str, Any],
) -> Any:
    """Call *target* with decoded arguments."""
    positional_args: list[Any] = []
    keyword_args: dict[str, Any] = {}
    has_varargs = any(
        parameter.kind == inspect.Parameter.VAR_POSITIONAL
        for parameter in signature.parameters.values()
    )
    before_varargs = True

    for parameter in signature.parameters.values():
        if parameter.name in {"self", "cls"}:
            continue

        annotation = type_hints.get(parameter.name, parameter.annotation)

        if parameter.kind == inspect.Parameter.VAR_POSITIONAL:
            before_varargs = False
            extra_args = arguments.get("args", []) or []
            positional_args.extend(_decode_value(item, Any) for item in extra_args)
            continue

        if parameter.kind == inspect.Parameter.VAR_KEYWORD:
            extra_kwargs = arguments.get("kwargs", {}) or {}
            if not isinstance(extra_kwargs, dict):
                raise TypeError("kwargs must be a JSON object")
            keyword_args.update(
                {str(key): _decode_value(item, Any) for key, item in extra_kwargs.items()}
            )
            continue

        expects_positional = (
            before_varargs
            and has_varargs
            and parameter.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
        )

        if parameter.name in arguments:
            decoded = _decode_value(arguments[parameter.name], annotation)
        elif parameter.default is not inspect._empty:
            if expects_positional:
                decoded = parameter.default
            else:
                continue
        else:
            raise KeyError(f"Missing required argument: {parameter.name}")

        if expects_positional or parameter.kind == inspect.Parameter.POSITIONAL_ONLY:
            positional_args.append(decoded)
        else:
            keyword_args[parameter.name] = decoded

    return target(*positional_args, **keyword_args)


def _describe_instance_payload(identifier: str) -> dict[str, Any]:
    """Return a detailed description for a stored instance."""
    value = _get_instance(identifier)
    payload = dict(_INSTANCE_META.get(identifier, {}))
    payload.update(
        {
            "instance_id": identifier,
            "module": type(value).__module__,
            "class_name": type(value).__name__,
            "methods": _public_method_summaries(value),
        }
    )
    return payload


def _list_instances_payload() -> list[dict[str, Any]]:
    """Return current stored-object metadata."""
    return [
        _describe_instance_payload(identifier)
        for identifier in sorted(_INSTANCE_STORE)
    ]


def _build_public_tool_spec(item: dict[str, str], target: Any) -> _ToolSpec:
    """Create a generated tool spec for a public ferro-ta callable."""
    is_enum = inspect.isclass(target) and issubclass(target, enum.Enum)
    type_hints = _type_hints_for_target(target)

    if is_enum:
        signature = inspect.Signature(
            [
                inspect.Parameter(
                    "value",
                    kind=inspect.Parameter.KEYWORD_ONLY,
                    annotation=str,
                    default=inspect._empty,
                )
            ]
        )
        wrapper_signature, input_schema = _build_signature_and_schema(
            signature,
            type_hints={"value": str},
        )
        description = (
            item["doc"]
            or f"Construct a {item['name']} enum member. Returns a stored instance reference."
        )

        def invoke(arguments: dict[str, Any], *, enum_type: type[enum.Enum] = target) -> Any:
            if "value" not in arguments:
                raise KeyError("Missing required argument: value")
            member = _coerce_enum(arguments["value"], enum_type)
            return _instance_ref(member, source_tool=item["name"])

        return _ToolSpec(
            name=item["name"],
            description=description,
            input_schema=input_schema,
            wrapper_signature=wrapper_signature,
            invoke=invoke,
        )

    signature = inspect.signature(target)
    wrapper_signature, input_schema = _build_signature_and_schema(
        signature,
        type_hints=type_hints,
    )
    is_class = inspect.isclass(target)
    description = item["doc"] or f"Call {item['name']}."
    if is_class:
        description = (
            item["doc"]
            or f"Construct a {item['name']} instance. Returns a stored instance reference."
        )

    def invoke(
        arguments: dict[str, Any],
        *,
        raw_target: Any = target,
        raw_signature: inspect.Signature = signature,
        raw_type_hints: dict[str, Any] = type_hints,
        returns_instance: bool = is_class,
        source_tool: str = item["name"],
    ) -> Any:
        result = _invoke_target(
            raw_target,
            signature=raw_signature,
            type_hints=raw_type_hints,
            arguments=arguments,
        )
        if returns_instance:
            return _instance_ref(result, source_tool=source_tool)
        return _normalise_json(result)

    return _ToolSpec(
        name=item["name"],
        description=description,
        input_schema=input_schema,
        wrapper_signature=wrapper_signature,
        invoke=invoke,
    )


def _legacy_series_tool(
    name: str,
    *,
    indicator_name: str,
    description: str,
) -> _ToolSpec:
    """Return a legacy lowercase indicator alias tool."""
    wrapper_signature = inspect.Signature(
        [
            inspect.Parameter(
                "close",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=list[float],
                default=inspect._empty,
            ),
            inspect.Parameter(
                "timeperiod",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=int,
                default=14,
            ),
        ]
    )
    input_schema = {
        "type": "object",
        "properties": {
            "close": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Close price series.",
            },
            "timeperiod": {
                "type": "integer",
                "description": "Look-back period.",
                "default": 14,
            },
        },
        "required": ["close"],
        "additionalProperties": False,
    }

    def invoke(arguments: dict[str, Any]) -> Any:
        close = np.asarray(arguments["close"], dtype=np.float64)
        timeperiod = int(arguments.get("timeperiod", 14))
        return _normalise_json(
            compute_indicator(indicator_name, close, timeperiod=timeperiod)
        )

    return _ToolSpec(
        name=name,
        description=description,
        input_schema=input_schema,
        wrapper_signature=wrapper_signature,
        invoke=invoke,
    )


def _legacy_macd_tool() -> _ToolSpec:
    """Return the legacy lowercase MACD alias tool."""
    wrapper_signature = inspect.Signature(
        [
            inspect.Parameter(
                "close",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=list[float],
                default=inspect._empty,
            ),
            inspect.Parameter(
                "fastperiod",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=int,
                default=12,
            ),
            inspect.Parameter(
                "slowperiod",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=int,
                default=26,
            ),
            inspect.Parameter(
                "signalperiod",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=int,
                default=9,
            ),
        ]
    )
    input_schema = {
        "type": "object",
        "properties": {
            "close": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Close price series.",
            },
            "fastperiod": {
                "type": "integer",
                "description": "Fast EMA period.",
                "default": 12,
            },
            "slowperiod": {
                "type": "integer",
                "description": "Slow EMA period.",
                "default": 26,
            },
            "signalperiod": {
                "type": "integer",
                "description": "Signal EMA period.",
                "default": 9,
            },
        },
        "required": ["close"],
        "additionalProperties": False,
    }

    def invoke(arguments: dict[str, Any]) -> Any:
        close = np.asarray(arguments["close"], dtype=np.float64)
        result = compute_indicator(
            "MACD",
            close,
            fastperiod=int(arguments.get("fastperiod", 12)),
            slowperiod=int(arguments.get("slowperiod", 26)),
            signalperiod=int(arguments.get("signalperiod", 9)),
        )
        return _normalise_json(result)

    return _ToolSpec(
        name="macd",
        description=(
            "Compute MACD (Moving Average Convergence/Divergence). "
            "Returns the line, signal, and histogram."
        ),
        input_schema=input_schema,
        wrapper_signature=wrapper_signature,
        invoke=invoke,
    )


def _legacy_backtest_tool() -> _ToolSpec:
    """Return the legacy lowercase backtest alias tool."""
    wrapper_signature = inspect.Signature(
        [
            inspect.Parameter(
                "close",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=list[float],
                default=inspect._empty,
            ),
            inspect.Parameter(
                "strategy",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
                default="rsi_30_70",
            ),
            inspect.Parameter(
                "commission_per_trade",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=float,
                default=0.0,
            ),
            inspect.Parameter(
                "slippage_bps",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=float,
                default=0.0,
            ),
        ]
    )
    input_schema = {
        "type": "object",
        "properties": {
            "close": {
                "type": "array",
                "items": {"type": "number"},
                "description": "Close price series.",
            },
            "strategy": {
                "type": "string",
                "description": (
                    "Strategy name: 'rsi_30_70', 'sma_crossover', or 'macd_crossover'."
                ),
                "default": "rsi_30_70",
            },
            "commission_per_trade": {
                "type": "number",
                "description": "Fixed commission per trade.",
                "default": 0.0,
            },
            "slippage_bps": {
                "type": "number",
                "description": "Slippage in basis points.",
                "default": 0.0,
            },
        },
        "required": ["close"],
        "additionalProperties": False,
    }

    def invoke(arguments: dict[str, Any]) -> Any:
        close = np.asarray(arguments["close"], dtype=np.float64)
        result = run_backtest(
            str(arguments.get("strategy", "rsi_30_70")),
            close,
            commission_per_trade=float(arguments.get("commission_per_trade", 0.0)),
            slippage_bps=float(arguments.get("slippage_bps", 0.0)),
        )
        return _normalise_json(result)

    return _ToolSpec(
        name="backtest",
        description=(
            "Run a vectorized backtest on close prices using a named strategy. "
            "Returns final equity, trade count, and the equity curve."
        ),
        input_schema=input_schema,
        wrapper_signature=wrapper_signature,
        invoke=invoke,
    )


def _instance_management_specs() -> list[_ToolSpec]:
    """Return generic stored-object management tools."""
    list_signature = inspect.Signature([])
    simple_schema = {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False,
    }

    describe_signature = inspect.Signature(
        [
            inspect.Parameter(
                "instance_id",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
                default=inspect._empty,
            )
        ]
    )
    describe_schema = {
        "type": "object",
        "properties": {
            "instance_id": {
                "type": "string",
                "description": "Stored object identifier.",
            }
        },
        "required": ["instance_id"],
        "additionalProperties": False,
    }

    call_signature = inspect.Signature(
        [
            inspect.Parameter(
                "instance_id",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
                default=inspect._empty,
            ),
            inspect.Parameter(
                "method",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
                default=inspect._empty,
            ),
            inspect.Parameter(
                "args",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=list[Any],
                default=None,
            ),
            inspect.Parameter(
                "kwargs",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=dict[str, Any],
                default=None,
            ),
        ]
    )
    call_schema = {
        "type": "object",
        "properties": {
            "instance_id": {
                "type": "string",
                "description": "Stored object identifier.",
            },
            "method": {
                "type": "string",
                "description": "Public method name.",
            },
            "args": {
                "type": "array",
                "items": {},
                "description": "Optional positional arguments.",
            },
            "kwargs": {
                "type": "object",
                "description": f"Optional keyword arguments. {_REFERENCE_HELP}",
            },
        },
        "required": ["instance_id", "method"],
        "additionalProperties": False,
    }

    callable_signature = inspect.Signature(
        [
            inspect.Parameter(
                "instance_id",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=str,
                default=inspect._empty,
            ),
            inspect.Parameter(
                "args",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=list[Any],
                default=None,
            ),
            inspect.Parameter(
                "kwargs",
                kind=inspect.Parameter.KEYWORD_ONLY,
                annotation=dict[str, Any],
                default=None,
            ),
        ]
    )
    callable_schema = {
        "type": "object",
        "properties": {
            "instance_id": {
                "type": "string",
                "description": "Stored callable identifier.",
            },
            "args": {
                "type": "array",
                "items": {},
                "description": "Optional positional arguments.",
            },
            "kwargs": {
                "type": "object",
                "description": f"Optional keyword arguments. {_REFERENCE_HELP}",
            },
        },
        "required": ["instance_id"],
        "additionalProperties": False,
    }

    def list_instances_tool(arguments: dict[str, Any]) -> Any:
        del arguments
        return _list_instances_payload()

    def describe_instance_tool(arguments: dict[str, Any]) -> Any:
        return _describe_instance_payload(str(arguments["instance_id"]))

    def call_instance_method_tool(arguments: dict[str, Any]) -> Any:
        value = _get_instance(str(arguments["instance_id"]))
        method_name = str(arguments["method"])
        if method_name.startswith("_"):
            raise ValueError("Only public methods can be called")
        method = getattr(value, method_name)
        if not callable(method):
            raise TypeError(f"{method_name!r} is not callable on {arguments['instance_id']!r}")
        args = [_decode_value(item, Any) for item in (arguments.get("args") or [])]
        kwargs = {
            str(key): _decode_value(item, Any)
            for key, item in (arguments.get("kwargs") or {}).items()
        }
        return _normalise_json(method(*args, **kwargs))

    def call_stored_callable_tool(arguments: dict[str, Any]) -> Any:
        value = _get_instance(str(arguments["instance_id"]))
        if not callable(value):
            raise TypeError(f"{arguments['instance_id']!r} is not callable")
        args = [_decode_value(item, Any) for item in (arguments.get("args") or [])]
        kwargs = {
            str(key): _decode_value(item, Any)
            for key, item in (arguments.get("kwargs") or {}).items()
        }
        return _normalise_json(value(*args, **kwargs))

    def delete_instance_tool(arguments: dict[str, Any]) -> Any:
        identifier = str(arguments["instance_id"])
        payload = _describe_instance_payload(identifier)
        _INSTANCE_STORE.pop(identifier)
        _INSTANCE_META.pop(identifier, None)
        return {"deleted": True, **payload}

    return [
        _ToolSpec(
            name="list_instances",
            description="List stored MCP object references created during this session.",
            input_schema=simple_schema,
            wrapper_signature=list_signature,
            invoke=list_instances_tool,
        ),
        _ToolSpec(
            name="describe_instance",
            description="Describe a stored MCP object reference and list its public methods.",
            input_schema=describe_schema,
            wrapper_signature=describe_signature,
            invoke=describe_instance_tool,
        ),
        _ToolSpec(
            name="call_instance_method",
            description="Call a public method on a stored MCP object reference.",
            input_schema=call_schema,
            wrapper_signature=call_signature,
            invoke=call_instance_method_tool,
        ),
        _ToolSpec(
            name="call_stored_callable",
            description="Invoke a stored callable object reference.",
            input_schema=callable_schema,
            wrapper_signature=callable_signature,
            invoke=call_stored_callable_tool,
        ),
        _ToolSpec(
            name="delete_instance",
            description="Delete a stored MCP object reference.",
            input_schema=describe_schema,
            wrapper_signature=describe_signature,
            invoke=delete_instance_tool,
        ),
    ]


@lru_cache(maxsize=1)
def _tool_catalog() -> dict[str, _ToolSpec]:
    """Return the full MCP tool catalog."""
    catalog: dict[str, _ToolSpec] = {}

    legacy_specs = [
        _legacy_series_tool(
            "sma",
            indicator_name="SMA",
            description="Compute the Simple Moving Average (SMA) of a price series.",
        ),
        _legacy_series_tool(
            "ema",
            indicator_name="EMA",
            description="Compute the Exponential Moving Average (EMA) of a price series.",
        ),
        _legacy_series_tool(
            "rsi",
            indicator_name="RSI",
            description="Compute the Relative Strength Index (RSI) of a price series.",
        ),
        _legacy_macd_tool(),
        _legacy_backtest_tool(),
    ]
    for spec in legacy_specs:
        catalog[spec.name] = spec

    public_entries, targets = _discover_public_callables()
    for item in public_entries:
        spec = _build_public_tool_spec(item, targets[item["name"]])
        catalog[spec.name] = spec

    for spec in _instance_management_specs():
        catalog[spec.name] = spec

    return catalog


def _make_fastmcp_wrapper(spec: _ToolSpec) -> Callable[..., Any]:
    """Create a FastMCP-friendly wrapper for *spec*."""

    def wrapper(**kwargs: Any) -> Any:
        return _to_call_tool_result(handle_call_tool(spec.name, dict(kwargs)))

    wrapper.__name__ = f"tool_{_slugify(spec.name).replace('-', '_')}"
    wrapper.__doc__ = spec.description
    wrapper.__signature__ = spec.wrapper_signature
    wrapper.__annotations__ = {
        parameter.name: (
            Any if parameter.annotation is inspect._empty else parameter.annotation
        )
        for parameter in spec.wrapper_signature.parameters.values()
    }
    wrapper.__annotations__["return"] = Any
    return wrapper


def handle_list_tools() -> dict[str, Any]:
    """Return the MCP ListTools response."""
    return {
        "tools": [
            {
                "name": spec.name,
                "description": spec.description,
                "inputSchema": spec.input_schema,
            }
            for spec in _tool_catalog().values()
        ]
    }


def handle_call_tool(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Dispatch an MCP CallTool request and return helper-style content."""
    try:
        spec = _tool_catalog().get(name)
        if spec is None:
            return _text_result(
                f"Unknown tool: {name!r}",
                structured={"error": f"Unknown tool: {name!r}"},
                is_error=True,
            )

        payload = spec.invoke(arguments)
        return _response_from_payload(payload)

    except Exception as exc:
        return _text_result(
            f"Error: {exc}",
            structured={"error": str(exc)},
            is_error=True,
        )


@lru_cache(maxsize=1)
def create_server() -> Any:
    """Create the FastMCP server lazily so MCP stays optional."""
    fast_mcp_type, _, _ = _load_mcp_sdk()
    app = fast_mcp_type(
        "ferro-ta",
        instructions=(
            "Expose ferro-ta's public API over MCP. "
            "Use exact public ferro-ta names such as SMA, RSI, compute_indicator, "
            "TickAggregator, or AlertManager, or the legacy aliases sma, ema, rsi, "
            "macd, and backtest. Use list_instances, describe_instance, "
            "call_instance_method, call_stored_callable, and delete_instance for "
            "stateful objects and stored callables."
        ),
    )

    for spec in _tool_catalog().values():
        app.add_tool(_make_fastmcp_wrapper(spec), name=spec.name, description=spec.description)

    return app


def run_server() -> None:  # pragma: no cover
    """Run the stdio MCP server."""
    try:
        create_server().run(transport="stdio")
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
