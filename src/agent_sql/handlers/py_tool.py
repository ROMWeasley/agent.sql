"""Stateless Python @tool decorator, JSON schema inference, and py.call handler."""

import inspect
import json
import re
import sqlite3
import types
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Annotated, Any, Literal, Union, get_args, get_origin, get_type_hints

SAFE_NAME_RE = re.compile(r"^[a-zA-Z0-9_-]+$")
_NONE_TYPE = type(None)


@dataclass(frozen=True, slots=True)
class ToolSpec:
    """Declared metadata and runtime callable for a single Python tool."""

    name: str
    description: str
    input_schema: dict[str, Any]
    fn: Callable[..., Any]
    annotations: dict[str, Any]
    required: tuple[str, ...]


def _validate_identifier(value: str, label: str) -> str:
    """Ensure SQL-interpolated identifiers match schema CHECK constraints."""
    if not isinstance(value, str) or not SAFE_NAME_RE.fullmatch(value):
        raise ValueError(f"{label} must match ^[a-zA-Z0-9_-]+$, got {value!r}")
    return value


def _unwrap_annotated(annotation: Any) -> tuple[Any, str | None]:
    """Strip Annotated wrappers and return the underlying type plus first string description."""
    description: str | None = None
    current = annotation
    while get_origin(current) is Annotated:
        args = get_args(current)
        current = args[0]
        if description is None:
            for meta in args[1:]:
                if isinstance(meta, str) and meta.strip():
                    description = meta.strip()
                    break
    return current, description


def _literal_schema(values: tuple[Any, ...]) -> dict[str, Any]:
    """Convert Literal[...] members into a JSON Schema enum."""
    if not values:
        raise TypeError("Literal tool annotation must have at least one value")
    types_seen: set[str] = set()
    for val in values:
        if isinstance(val, bool):
            types_seen.add("boolean")
        elif isinstance(val, int):
            types_seen.add("integer")
        elif isinstance(val, float):
            types_seen.add("number")
        elif isinstance(val, str):
            types_seen.add("string")
        elif val is None:
            types_seen.add("null")
        else:
            raise TypeError(f"Unsupported Literal member type: {type(val).__name__}")
    schema: dict[str, Any] = {"enum": list(values)}
    if len(types_seen) == 1:
        schema["type"] = next(iter(types_seen))
    return schema


def _type_to_schema(annotation: Any) -> dict[str, Any]:
    """Map a Python type hint to a JSON Schema subset suitable for LLM function calling."""
    base, param_desc = _unwrap_annotated(annotation)
    schema: dict[str, Any]

    if base is str:
        schema = {"type": "string"}
    elif base is int:
        schema = {"type": "integer"}
    elif base is float:
        schema = {"type": "number"}
    elif base is bool:
        schema = {"type": "boolean"}
    elif base is _NONE_TYPE:
        schema = {"type": "null"}
    elif base is list:
        schema = {"type": "array"}
    elif base is dict:
        schema = {"type": "object"}
    else:
        origin = get_origin(base)
        args = get_args(base)

        if origin is Literal:
            schema = _literal_schema(args)
        elif origin is list:
            if len(args) != 1:
                raise TypeError(f"Unsupported list annotation: {annotation!r}")
            schema = {"type": "array", "items": _type_to_schema(args[0])}
        elif origin is dict:
            if len(args) != 2 or args[0] is not str:
                raise TypeError(f"Only dict[str, V] is supported for tools, got {annotation!r}")
            schema = {"type": "object", "additionalProperties": _type_to_schema(args[1])}
        elif origin in (Union, types.UnionType):
            non_none = [a for a in args if a is not _NONE_TYPE]
            has_none = len(non_none) != len(args)
            if not non_none:
                schema = {"type": "null"}
            elif len(non_none) == 1 and has_none:
                inner = _type_to_schema(non_none[0])
                if "type" in inner and isinstance(inner["type"], str):
                    schema = {**inner, "type": [inner["type"], "null"]}
                else:
                    schema = {"anyOf": [inner, {"type": "null"}]}
            else:
                schema = {"anyOf": [_type_to_schema(a) for a in args]}
        else:
            raise TypeError(f"Unsupported parameter type annotation: {annotation!r}")

    if param_desc is not None:
        schema["description"] = param_desc
    return schema


def _check_runtime_value(value: Any, annotation: Any, path: str) -> str | None:
    """Validate a JSON-decoded argument value strictly against a Python type hint."""
    base, _ = _unwrap_annotated(annotation)

    if base is str:
        return None if isinstance(value, str) else f"{path} must be a string"
    if base is bool:
        return None if isinstance(value, bool) else f"{path} must be a boolean"
    if base is int:
        # Reject bool explicitly because isinstance(True, int) is True in Python.
        if isinstance(value, bool) or not isinstance(value, int):
            return f"{path} must be an integer"
        return None
    if base is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{path} must be a number"
        return None
    if base is _NONE_TYPE:
        return None if value is None else f"{path} must be null"
    if base is list:
        return None if isinstance(value, list) else f"{path} must be an array"
    if base is dict:
        return None if isinstance(value, dict) else f"{path} must be an object"

    origin = get_origin(base)
    args = get_args(base)

    if origin is Literal:
        for member in args:
            if type(value) is type(member) and value == member:
                return None
        return f"{path} must be one of {list(args)!r}"
    if origin is list:
        if not isinstance(value, list):
            return f"{path} must be an array"
        for idx, elem in enumerate(value):
            err = _check_runtime_value(elem, args[0], f"{path}[{idx}]")
            if err is not None:
                return err
        return None
    if origin is dict:
        if not isinstance(value, dict):
            return f"{path} must be an object"
        for k, v in value.items():
            if not isinstance(k, str):
                return f"{path} keys must be strings"
            err = _check_runtime_value(v, args[1], f"{path}.{k}")
            if err is not None:
                return err
        return None
    if origin in (Union, types.UnionType):
        for opt in args:
            if _check_runtime_value(value, opt, path) is None:
                return None
        return f"{path} does not match expected type"

    return f"Unsupported annotation for {path}"


def _build_tool_spec(
    fn: Callable[..., Any],
    *,
    name: str | None = None,
    description: str | None = None,
) -> ToolSpec:
    """Inspect a typed Python function and construct its ToolSpec and JSON Schema."""
    if not callable(fn):
        raise TypeError(f"@tool target must be callable, got {type(fn).__name__}")

    tool_name = _validate_identifier(fn.__name__ if name is None else name, "tool name")
    raw_doc = inspect.getdoc(fn) or ""
    tool_desc = raw_doc.splitlines()[0].strip() if description is None and raw_doc else (
        "" if description is None else description.strip()
    )

    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)

    properties: dict[str, Any] = {}
    required: list[str] = []
    annotations: dict[str, Any] = {}

    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise TypeError(f"Tool {tool_name!r} cannot use *args or **kwargs ({param_name})")
        if param.kind == inspect.Parameter.POSITIONAL_ONLY:
            raise TypeError(f"Tool {tool_name!r} cannot use positional-only param {param_name!r}")
        if param_name not in hints:
            raise TypeError(f"Tool {tool_name!r} parameter {param_name!r} needs a type annotation")

        hint = hints[param_name]
        annotations[param_name] = hint
        properties[param_name] = _type_to_schema(hint)
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    schema: dict[str, Any] = {
        "type": "object",
        "properties": properties,
        "required": required,
        "additionalProperties": False,
    }
    return ToolSpec(
        name=tool_name,
        description=tool_desc,
        input_schema=schema,
        fn=fn,
        annotations=annotations,
        required=tuple(required),
    )


def tool(
    fn: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
) -> Any:
    """Decorate a typed Python function as an agent.sql tool, attaching `__tool_spec__`."""

    def _decorate(target: Callable[..., Any]) -> Callable[..., Any]:
        spec = _build_tool_spec(target, name=name, description=description)
        target.__tool_spec__ = spec  # type: ignore[attr-defined]
        return target

    if fn is not None:
        return _decorate(fn)
    return _decorate


def _validate_arguments(spec: ToolSpec, arguments: Any) -> str | None:
    """Validate `arguments` against `spec` before invoking the Python callable."""
    if not isinstance(arguments, dict):
        return f"Tool arguments for {spec.name} must be an object"

    for req in spec.required:
        if req not in arguments:
            return f"Missing required argument {req!r} for tool {spec.name}"

    for key, val in arguments.items():
        if key not in spec.annotations:
            return f"Unexpected argument {key!r} for tool {spec.name}"
        err = _check_runtime_value(val, spec.annotations[key], key)
        if err is not None:
            return err
    return None


class ToolRegistry:
    """In-memory Python tool catalog that syncs rows to SQLite and handles `py.call` effects."""

    def __init__(self) -> None:
        """Create an empty Python tool registry keyed by `(server_id, tool_name)`."""
        self._tools: dict[tuple[str, str], ToolSpec] = {}

    def register(
        self,
        fn: Callable[..., Any],
        *,
        server_id: str = "python",
        name: str | None = None,
        description: str | None = None,
    ) -> ToolSpec:
        """Register a `@tool`-decorated or plain typed function under `server_id`."""
        srv = _validate_identifier(server_id, "server_id")
        existing = getattr(fn, "__tool_spec__", None)
        if isinstance(existing, ToolSpec) and name is None and description is None:
            spec = existing
        else:
            spec = _build_tool_spec(fn, name=name, description=description)

        key = (srv, spec.name)
        if key in self._tools:
            raise ValueError(f"Duplicate tool registration for {srv}.{spec.name}")
        self._tools[key] = spec
        return spec

    def sync_to_db(self, conn: sqlite3.Connection, *, agent_id: str | None = None) -> None:
        """Upsert `tool_servers` and `tools` rows and optionally bind them to `agent_id`."""
        servers = sorted({srv for srv, _ in self._tools})
        for srv in servers:
            conn.execute(
                """
                INSERT INTO tool_servers (id, kind) VALUES (?, 'python')
                ON CONFLICT(id) DO UPDATE SET kind = 'python'
                """,
                (srv,),
            )

        for (srv, _), spec in sorted(self._tools.items()):
            conn.execute(
                """
                INSERT INTO tools (server_id, name, description, input_schema)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(server_id, name) DO UPDATE SET
                    description = excluded.description,
                    input_schema = excluded.input_schema
                """,
                (srv, spec.name, spec.description, json.dumps(spec.input_schema)),
            )
            if agent_id is not None:
                conn.execute(
                    """
                    INSERT OR IGNORE INTO agent_tools (agent_id, server_id, tool_name)
                    VALUES (?, ?, ?)
                    """,
                    (agent_id, srv, spec.name),
                )

    def _resolve(self, server: Any, name: Any) -> ToolSpec | str:
        """Resolve `(server, name)` to a ToolSpec, or return an error message string."""
        if not isinstance(name, str) or not name:
            return "py.call requires a non-empty tool 'name'"
        if server is not None:
            if not isinstance(server, str) or not server:
                return "py.call 'server' must be a non-empty string when provided"
            spec = self._tools.get((server, name))
            if spec is None:
                return f"Unknown python tool: {server}.{name}"
            return spec

        matches = [spec for (srv, tool_name), spec in self._tools.items() if tool_name == name]
        if not matches:
            return f"Unknown python tool: {name}"
        if len(matches) > 1:
            return f"Ambiguous python tool {name!r}; specify 'server'"
        return matches[0]

    def __call__(self, args: dict[str, Any]) -> dict[str, Any]:
        """Execute a `py.call` effect, returning errors as data and never raising."""
        if not isinstance(args, dict):
            return {"ok": False, "error": "py.call args must be an object"}

        resolved = self._resolve(args.get("server"), args.get("name"))
        if isinstance(resolved, str):
            return {"ok": False, "error": resolved}

        raw_arguments = args.get("arguments", {})
        validation_err = _validate_arguments(resolved, raw_arguments)
        if validation_err is not None:
            return {"ok": False, "error": validation_err}

        try:
            value = resolved.fn(**raw_arguments)
            # Validate JSON serializability inside the handler so sys() never hits encoding errors.
            json.dumps(value)
        except Exception as exc:
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        return {"ok": True, "value": value}


def make_py_call_handler(
    tools: Iterable[Callable[..., Any]],
    *,
    server_id: str = "python",
) -> ToolRegistry:
    """Build a ToolRegistry pre-populated with `tools` under `server_id`."""
    registry = ToolRegistry()
    for fn in tools:
        registry.register(fn, server_id=server_id)
    return registry
