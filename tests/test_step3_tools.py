"""Tests for Step 3: @tool schema inference, py.call handler, and SQL ReAct tool loops."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Annotated, Any, Literal

import pytest

from agent_sql.handlers import ToolRegistry, make_py_call_handler, tool
from agent_sql.host.db import connect_db, open_db
from agent_sql.host.runner import Tick, load_kernel_sql, run_agent, run_kernel
from agent_sql.host.sys import SysDispatcher


def _seed_agent(conn: sqlite3.Connection, agent_id: str = "assistant") -> None:
    """Insert a basic llm agent row for tool tests."""
    conn.execute(
        "INSERT INTO agents (id, kind, model_id, instruction) VALUES (?, 'llm', ?, ?)",
        (agent_id, "demo-model", "Use tools when helpful."),
    )


def test_tool_decorator_infers_json_schema_and_preserves_direct_call() -> None:
    """@tool infers JSON Schema types, Annotated descriptions, optionals, and Literals."""

    @tool
    def search(
        query: Annotated[str, "Search terms"],
        limit: int = 5,
        weight: float = 1.0,
        exact: bool = False,
        tags: list[str] | None = None,
        filters: dict[str, int] | None = None,
        mode: Literal["fast", "deep"] = "fast",
    ) -> dict[str, Any]:
        """Run a structured search."""
        return {"query": query, "limit": limit, "mode": mode}

    assert search("sqlite") == {"query": "sqlite", "limit": 5, "mode": "fast"}
    spec = search.__tool_spec__
    assert spec.name == "search"
    assert spec.description == "Run a structured search."
    assert spec.input_schema == {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search terms"},
            "limit": {"type": "integer"},
            "weight": {"type": "number"},
            "exact": {"type": "boolean"},
            "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            "filters": {
                "type": ["object", "null"],
                "additionalProperties": {"type": "integer"},
            },
            "mode": {"type": "string", "enum": ["fast", "deep"]},
        },
        "required": ["query"],
        "additionalProperties": False,
    }


def test_tool_decorator_rejects_untyped_varargs_and_unsafe_names() -> None:
    """@tool rejects missing annotations, *args/**kwargs, and unsafe identifiers."""
    with pytest.raises(TypeError, match="needs a type annotation"):

        @tool
        def untyped(x) -> str:  # type: ignore[no-untyped-def]
            return str(x)

    with pytest.raises(TypeError, match=r"\*args or \*\*kwargs"):

        @tool
        def varargs(*args: str) -> str:
            return ",".join(args)

    with pytest.raises(ValueError, match="tool name"):

        @tool(name="bad.name")
        def good_fn(x: int) -> int:
            return x


def test_py_call_validates_types_strictly_and_never_raises() -> None:
    """py.call rejects bool-as-int, unknown args, exceptions, and non-JSON return values."""

    @tool
    def add(a: int, b: int) -> int:
        """Add two integers."""
        return a + b

    @tool
    def explode(msg: str) -> str:
        raise ZeroDivisionError(msg)

    @tool
    def unserializable() -> object:
        return object()

    handler = make_py_call_handler([add, explode, unserializable])

    assert handler({"server": "python", "name": "add", "arguments": {"a": 2, "b": 3}}) == {
        "ok": True,
        "value": 5,
    }
    # Python treats bool as int subclass; runtime validation must reject it.
    bool_err = handler({"server": "python", "name": "add", "arguments": {"a": True, "b": 2}})
    assert bool_err["ok"] is False
    assert "integer" in bool_err["error"]

    missing_err = handler({"server": "python", "name": "add", "arguments": {"a": 1}})
    assert missing_err["ok"] is False
    assert "Missing required argument" in missing_err["error"]

    extra_err = handler({"server": "python", "name": "add", "arguments": {"a": 1, "b": 2, "c": 3}})
    assert extra_err["ok"] is False
    assert "Unexpected argument" in extra_err["error"]

    exc_err = handler({"server": "python", "name": "explode", "arguments": {"msg": "boom"}})
    assert exc_err == {"ok": False, "error": "ZeroDivisionError: boom"}

    json_err = handler({"server": "python", "name": "unserializable", "arguments": {}})
    assert json_err["ok"] is False
    assert "TypeError" in json_err["error"]

    unknown_err = handler({"server": "python", "name": "missing", "arguments": {}})
    assert unknown_err["ok"] is False
    assert "Unknown python tool" in unknown_err["error"]


def test_registry_sync_and_agent_tools_constraints() -> None:
    """sync_to_db upserts tools idempotently and schema enforces valid bindings."""

    @tool
    def ping(text: str) -> str:
        """Echo text."""
        return text

    registry = ToolRegistry()
    registry.register(ping)

    with closing(open_db()) as conn:
        _seed_agent(conn)
        registry.sync_to_db(conn, agent_id="assistant")
        registry.sync_to_db(conn, agent_id="assistant")
        assert conn.execute("SELECT count(*) FROM agent_tools").fetchone()[0] == 1

        # Wildcard binding (tool_name IS NULL) is permitted by the composite FK.
        conn.execute(
            "INSERT INTO agent_tools (agent_id, server_id, tool_name)"
            " VALUES ('assistant', 'python', NULL)"
        )
        # Duplicate wildcard binding is blocked by idx_agent_tools_binding.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO agent_tools (agent_id, server_id, tool_name)"
                " VALUES ('assistant', 'python', NULL)"
            )
        # Nonexistent explicit tool name fails FK check.
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO agent_tools (agent_id, server_id, tool_name)"
                " VALUES ('assistant', 'python', 'ghost')"
            )


def test_kernel_with_bound_tools_keeps_coroutine_plan() -> None:
    """EXPLAIN QUERY PLAN still streams `world` via CO-ROUTINE when tools are bound."""

    @tool
    def echo(msg: str) -> str:
        return msg

    registry = make_py_call_handler([echo])

    with closing(open_db()) as conn:
        _seed_agent(conn)
        registry.sync_to_db(conn, agent_id="assistant")
        conn.create_function("sys", 1, lambda _pending: "{}")
        plan = conn.execute(
            "\n".join(("EXPLAIN QUERY PLAN", load_kernel_sql())),
            {"run_id": 1, "agent_id": "assistant", "input": '{"prompt":"hi"}'},
        ).fetchall()

    top_level = "\n".join(row["detail"] for row in plan if row["parent"] == 0)
    assert "CO-ROUTINE" in top_level
    assert "USE TEMP B-TREE" not in top_level
    assert "MATERIALIZE" not in top_level


def test_bound_tool_schemas_deduplicated_across_wildcard_and_explicit() -> None:
    """Overlapping explicit and wildcard bindings deduplicate tools without forking rows."""

    @tool
    def alpha(x: int) -> int:
        """Alpha tool."""
        return x

    @tool
    def beta(s: str) -> str:
        """Beta tool."""
        return s

    registry = make_py_call_handler([alpha, beta])
    seen_tools: list[list[dict[str, Any]]] = []

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        seen_tools.append(effects[0]["args"]["tools"])
        return {effects[0]["id"]: {"ok": True, "value": {"content": "done"}}}

    with closing(open_db()) as conn:
        _seed_agent(conn)
        registry.sync_to_db(conn, agent_id="assistant")
        conn.execute(
            "INSERT INTO agent_tools (agent_id, server_id, tool_name)"
            " VALUES ('assistant', 'python', NULL)"
        )
        ticks = list(
            run_kernel(conn, agent_id="assistant", input_data={"prompt": "hi"}, sys_fn=fake_sys)
        )

    assert len(seen_tools) == 1
    assert [t["function"]["name"] for t in seen_tools[0]] == ["alpha", "beta"]
    assert ticks[-1].status == "done"


def test_react_loop_handles_typed_tool_returns_and_openai_messages() -> None:
    """Multi-turn ReAct loop encodes string, int, dict, and None tool outputs as string content."""

    @tool
    def get_num() -> int:
        return 42

    @tool
    def get_obj() -> dict[str, Any]:
        return {"ok": True, "count": 2}

    @tool
    def get_none() -> None:
        return None

    registry = make_py_call_handler([get_num, get_obj, get_none])
    turn = 0

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal turn
        out: dict[str, Any] = {}
        for eff in effects:
            if eff["op"] == "py.call":
                out[eff["id"]] = registry(eff["args"])
            elif eff["op"] == "llm.chat":
                turn += 1
                if turn == 1:
                    out[eff["id"]] = {
                        "ok": True,
                        "value": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "get_num", "arguments": "{}"},
                                },
                                {
                                    "id": "call_2",
                                    "type": "function",
                                    "function": {"name": "get_obj", "arguments": ""},
                                },
                                {
                                    "id": "call_3",
                                    "type": "function",
                                    "function": {"name": "get_none", "arguments": {}},
                                },
                            ],
                            "usage": {"total_tokens": 20},
                        },
                    }
                else:
                    out[eff["id"]] = {
                        "ok": True,
                        "value": {"content": "all done", "usage": {"total_tokens": 10}},
                    }
        return out

    with closing(open_db()) as conn:
        _seed_agent(conn)
        registry.sync_to_db(conn, agent_id="assistant")
        ticks = list(
            run_kernel(conn, agent_id="assistant", input_data={"prompt": "run"}, sys_fn=fake_sys)
        )

    last = ticks[-1]
    assert last.status == "done"
    assert last.completions == {}
    assert last.usage == {"tokens": 30}
    messages = last.tasks["t1"]["messages"]
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
        "tool",
        "tool",
        "assistant",
    ]
    assert all(isinstance(tc["function"]["arguments"], str) for tc in messages[2]["tool_calls"])
    tool_msgs = messages[3:6]
    assert [m["tool_call_id"] for m in tool_msgs] == ["call_1", "call_2", "call_3"]
    assert all(isinstance(m["content"], str) for m in tool_msgs)
    assert tool_msgs[0]["content"] == "42"
    assert json.loads(tool_msgs[1]["content"]) == {"ok": True, "count": 2}
    assert tool_msgs[2]["content"] == "null"
    assert last.tasks["t1"]["result"] == "all done"


def test_parallel_tool_calls_batch_into_single_effect_and_preserve_order() -> None:
    """Multiple tool_calls in one turn execute in one sys() batch and preserve call order."""

    @tool
    def mul(a: int, b: int) -> int:
        return a * b

    registry = make_py_call_handler([mul])
    batches: list[list[dict[str, Any]]] = []
    turn = 0

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal turn
        batches.append(effects)
        if effects[0]["op"] == "llm.chat":
            turn += 1
            if turn == 1:
                return {
                    effects[0]["id"]: {
                        "ok": True,
                        "value": {
                            "content": "",
                            "tool_calls": [
                                {
                                    "id": "c_first",
                                    "type": "function",
                                    "function": {"name": "mul", "arguments": '{"a":2,"b":3}'},
                                },
                                {
                                    "id": "c_second",
                                    "type": "function",
                                    "function": {"name": "mul", "arguments": '{"a":4,"b":5}'},
                                },
                                {
                                    "id": "c_third",
                                    "type": "function",
                                    "function": {"name": "mul", "arguments": '{"a":6,"b":7}'},
                                },
                            ],
                        },
                    }
                }
            return {effects[0]["id"]: {"ok": True, "value": {"content": "6, 20, 42"}}}
        return {eff["id"]: registry(eff["args"]) for eff in effects}

    with closing(open_db()) as conn:
        _seed_agent(conn)
        registry.sync_to_db(conn, agent_id="assistant")
        ticks = list(
            run_kernel(conn, agent_id="assistant", input_data={"prompt": "calc"}, sys_fn=fake_sys)
        )

    assert [len(b) for b in batches] == [1, 3, 1]
    assert [e["args"]["arguments"] for e in batches[1]] == [
        {"a": 2, "b": 3},
        {"a": 4, "b": 5},
        {"a": 6, "b": 7},
    ]
    tool_msgs = [m for m in ticks[-1].tasks["t1"]["messages"] if m["role"] == "tool"]
    assert [(m["tool_call_id"], m["content"]) for m in tool_msgs] == [
        ("c_first", "6"),
        ("c_second", "20"),
        ("c_third", "42"),
    ]


def test_untrusted_tool_calls_and_handler_errors_feed_back_for_recovery() -> None:
    """Unbound tools, malformed arguments, adversarial call IDs, and tool errors recover cleanly."""

    @tool
    def divide(a: int, b: int) -> float:
        return a / b

    @tool
    def secret_admin() -> str:
        return "secret"

    registry = make_py_call_handler([divide, secret_admin])
    py_calls: list[str] = []
    turn = 0

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal turn
        if effects[0]["op"] == "llm.chat":
            turn += 1
            if turn == 1:
                return {
                    effects[0]["id"]: {
                        "ok": True,
                        "value": {
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call.1$['x']",
                                    "type": "function",
                                    "function": {"name": "secret_admin", "arguments": "{}"},
                                },
                                {
                                    "id": "call.2$['y']",
                                    "type": "function",
                                    "function": {"name": "divide", "arguments": "{oops"},
                                },
                                {
                                    "id": "call.3$['z']",
                                    "type": "function",
                                    "function": {"name": "divide", "arguments": '{"a":10,"b":0}'},
                                },
                            ],
                        },
                    }
                }
            return {effects[0]["id"]: {"ok": True, "value": {"content": "recovered"}}}

        out: dict[str, Any] = {}
        for eff in effects:
            py_calls.append(eff["args"]["name"])
            out[eff["id"]] = registry(eff["args"])
        return out

    with closing(open_db()) as conn:
        _seed_agent(conn)
        # Bind only `divide` — `secret_admin` stays unbound in SQL.
        registry.sync_to_db(conn)
        conn.execute(
            "INSERT INTO agent_tools (agent_id, server_id, tool_name)"
            " VALUES ('assistant', 'python', 'divide')"
        )
        ticks = list(
            run_kernel(conn, agent_id="assistant", input_data={"prompt": "try"}, sys_fn=fake_sys)
        )

    assert py_calls == ["divide"]
    last = ticks[-1]
    assert last.status == "done"
    assert last.tasks["t1"]["result"] == "recovered"
    tool_msgs = [m for m in last.tasks["t1"]["messages"] if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_msgs] == [
        "call.1$['x']",
        "call.2$['y']",
        "call.3$['z']",
    ]
    assert "not bound" in json.loads(tool_msgs[0]["content"])["error"]
    assert "JSON object" in json.loads(tool_msgs[1]["content"])["error"]
    assert "ZeroDivisionError" in json.loads(tool_msgs[2]["content"])["error"]


def test_run_agent_with_tools_persists_definitions_and_replays_identically(tmp_path: Path) -> None:
    """End-to-end ReAct run persists tools snapshot + effects and replays with empty handlers."""

    @tool
    def double(n: int) -> int:
        """Double an integer."""
        return n * 2

    registry = make_py_call_handler([double])
    db_path = tmp_path / "react.db"
    with closing(open_db(db_path)) as conn:
        _seed_agent(conn)
        registry.sync_to_db(conn, agent_id="assistant")
        conn.commit()

    def fake_llm(args: dict[str, Any]) -> dict[str, Any]:
        if args["messages"][-1]["role"] == "user":
            return {
                "ok": True,
                "value": {
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_dbl",
                            "type": "function",
                            "function": {"name": "double", "arguments": '{"n":21}'},
                        }
                    ],
                    "usage": {"total_tokens": 9},
                },
            }
        answer = args["messages"][-1]["content"]
        return {
            "ok": True,
            "value": {"content": f"Result is {answer}", "usage": {"total_tokens": 5}},
        }

    recorded: list[Tick] = []
    outcome = run_agent(
        db_path,
        agent_id="assistant",
        input_data={"prompt": "double 21"},
        handlers={"llm.chat": fake_llm, "py.call": registry},
        on_tick=recorded.append,
    )

    assert outcome.status == "done"
    assert outcome.result == "Result is 42"
    assert outcome.usage == {"tokens": 14}

    wal_path = Path(f"{db_path}-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0

    with closing(connect_db(db_path)) as conn:
        run_row = conn.execute(
            "SELECT definitions FROM runs WHERE id = ?", (outcome.run_id,)
        ).fetchone()
        defs = json.loads(run_row["definitions"])
        assert [t["name"] for t in defs["tools"]] == ["double"]

        ops = [
            r["op"]
            for r in conn.execute(
                "SELECT op FROM effects WHERE run_id = ? ORDER BY effect_id", (outcome.run_id,)
            ).fetchall()
        ]
        assert sorted(ops) == ["llm.chat", "llm.chat", "py.call"]

        replay_sys = SysDispatcher({}, writer=conn, run_id=outcome.run_id)
        replayed = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "double 21"},
                sys_fn=replay_sys,
                run_id=outcome.run_id,
            )
        )

    assert replayed == recorded
