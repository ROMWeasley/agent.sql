"""Tests for idempotent DB setup and the Step 1 single-query kernel heartbeat."""

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from agent_sql.host.db import SCHEMA_VERSION, init_db, open_db, reset_db
from agent_sql.host.runner import load_kernel_sql, run_kernel


def _add_agent(
    conn: sqlite3.Connection,
    agent_id: str = "assistant",
    *,
    kind: str = "llm",
    model_id: str | None = "demo-model",
    instruction: str = "Be concise.",
) -> None:
    """Insert an agent definition row for kernel tests."""
    conn.execute(
        "INSERT INTO agents (id, kind, model_id, instruction) VALUES (?, ?, ?, ?)",
        (agent_id, kind, model_id, instruction),
    )


def test_init_db_is_idempotent(tmp_path: Path) -> None:
    """Re-running open_db/init_db preserves existing agents and customized settings."""
    db_path = tmp_path / "agent.db"
    with closing(open_db(db_path)) as conn:
        _add_agent(conn, "custom-agent")
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('max_ticks', '99')")
        conn.commit()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION

    with closing(open_db(db_path)) as conn:
        init_db(conn)
        assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        row = conn.execute("SELECT id FROM agents WHERE id = 'custom-agent'").fetchone()
        assert row is not None
        max_ticks = conn.execute("SELECT value FROM settings WHERE key = 'max_ticks'").fetchone()[0]
        assert max_ticks == "99"


def test_init_db_rejects_unsupported_user_version() -> None:
    """init_db fails fast instead of touching an unknown schema version."""
    with closing(sqlite3.connect(":memory:")) as conn:
        conn.execute("PRAGMA user_version = 99")
        with pytest.raises(RuntimeError, match="Unsupported schema version 99"):
            init_db(conn)


def test_reset_db_wipes_existing_data(tmp_path: Path) -> None:
    """reset_db deletes the DB file and WAL sidecars, then recreates default rows."""
    db_path = tmp_path / "agent.db"
    with closing(open_db(db_path)) as conn:
        _add_agent(conn, "ephemeral")
        conn.commit()

    with closing(reset_db(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM agents").fetchone()[0] == 0
        max_ticks = conn.execute("SELECT value FROM settings WHERE key = 'max_ticks'").fetchone()[0]
        assert max_ticks == "50"


@pytest.mark.parametrize("bad_id", ["", "bad.id", "bad id", "bad$id"])
def test_agent_id_constraint_rejects_invalid_identifiers(bad_id: str) -> None:
    """agents.id CHECK rejects non-path-safe characters because kernel.sql builds '$.' || id."""
    with closing(open_db()) as conn, pytest.raises(sqlite3.IntegrityError):
        _add_agent(conn, bad_id)


def test_happy_path_heartbeat() -> None:
    """Verify the full 8-tick REDUCE -> EFFECT -> REDUCE cycle and final task state."""
    # 8-tick cycle: anchor -> dispatch -> enqueue -> flush -> sys() -> dispatch -> consume -> halt.
    calls: list[list[dict[str, Any]]] = []

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        calls.append(effects)
        eff = effects[0]
        return {
            eff["id"]: {
                "ok": True,
                "value": {
                    "content": "pong",
                    "usage": {"total_tokens": 12},
                },
            }
        }

    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "ping"},
                sys_fn=fake_sys,
            )
        )

    assert [t.phase for t in ticks] == [
        "sched",
        "llm_step",
        "sched",
        "effect",
        "sched",
        "llm_step",
        "sched",
        "halt",
    ]
    assert len(calls) == 1
    assert calls[0] == [
        {
            "id": "r1_e2",
            "op": "llm.chat",
            "args": {
                "model_id": "demo-model",
                "messages": [
                    {"role": "system", "content": "Be concise."},
                    {"role": "user", "content": "ping"},
                ],
            },
        }
    ]

    last = ticks[-1]
    assert last.status == "done"
    assert last.pending == []
    assert last.completions == {}
    assert last.usage == {"tokens": 12}
    t1 = last.tasks["t1"]
    assert t1["status"] == "done"
    assert t1["result"] == "pong"
    assert t1["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "ping"},
        {"role": "assistant", "content": "pong"},
    ]


@pytest.mark.parametrize("input_data", [None, {}])
def test_empty_input_omits_user_turn(input_data: dict[str, Any] | None) -> None:
    """Omitted or empty input_data seeds only the system turn (D15: prompt is optional)."""
    seen_messages: list[list[dict[str, Any]]] = []

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        seen_messages.append(effects[0]["args"]["messages"])
        return {effects[0]["id"]: {"ok": True, "value": {"content": "ready"}}}

    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(conn, agent_id="assistant", input_data=input_data, sys_fn=fake_sys)
        )

    assert seen_messages == [[{"role": "system", "content": "Be concise."}]]
    assert ticks[-1].tasks["t1"]["input"] == {}


def test_structured_input_without_prompt() -> None:
    """Structured input without a prompt keeps the full JSON payload in t1.input."""
    seen_messages: list[list[dict[str, Any]]] = []

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        seen_messages.append(effects[0]["args"]["messages"])
        return {effects[0]["id"]: {"ok": True, "value": {"content": "ok"}}}

    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"topic": "sqlite"},
                sys_fn=fake_sys,
            )
        )

    assert seen_messages == [[{"role": "system", "content": "Be concise."}]]
    assert ticks[-1].tasks["t1"]["input"] == {"topic": "sqlite"}


def test_empty_instruction_and_empty_input_gives_empty_messages() -> None:
    """Empty instruction and empty input seed an empty messages list without blank turns."""
    seen_messages: list[list[dict[str, Any]]] = []

    def fake_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        seen_messages.append(effects[0]["args"]["messages"])
        return {effects[0]["id"]: {"ok": True, "value": {"content": "ok"}}}

    with closing(open_db()) as conn:
        _add_agent(conn, instruction="")
        list(run_kernel(conn, agent_id="assistant", input_data={}, sys_fn=fake_sys))

    assert seen_messages == [[]]


def test_unsupported_agent_kind_fails_cleanly() -> None:
    """An unimplemented agent kind halts with status='failed' instead of looping."""
    with closing(open_db()) as conn:
        _add_agent(conn, kind="call")
        ticks = list(
            run_kernel(conn, agent_id="assistant", input_data={}, sys_fn=lambda _e: {})
        )

    assert ticks[-1].phase == "halt"
    assert ticks[-1].status == "failed"


def test_non_dict_input_raises_type_error() -> None:
    """run_kernel rejects non-dict input_data eagerly rather than deferring until iteration."""
    with closing(open_db()) as conn, pytest.raises(TypeError, match="input_data must be a dict"):
        run_kernel(
            conn,
            agent_id="assistant",
            input_data="not-a-dict",  # type: ignore[arg-type]
            sys_fn=lambda _e: {},
        )


def test_kernel_streams_via_coroutine_without_temp_btree() -> None:
    """EXPLAIN QUERY PLAN must stream `world` via CO-ROUTINE with no top-level temp B-tree."""
    # Only check top-level nodes (parent=0); inner subqueries (like sched's ORDER BY seq LIMIT 1)
    # may use a per-tick temp B-tree without buffering the outer `world` stream.
    with closing(open_db()) as conn:
        _add_agent(conn)
        conn.create_function("sys", 1, lambda _pending: "{}")
        explain_sql = "\n".join(("EXPLAIN QUERY PLAN", load_kernel_sql()))
        plan = conn.execute(
            explain_sql,
            {"run_id": 1, "agent_id": "assistant", "input": '{"prompt":"hi"}'},
        ).fetchall()

    top_level = "\n".join(row["detail"] for row in plan if row["parent"] == 0)
    assert "CO-ROUTINE" in top_level
    assert "USE TEMP B-TREE" not in top_level
    assert "MATERIALIZE" not in top_level


def test_kernel_streams_ticks_before_effect_runs() -> None:
    """Verify ticks stream incrementally so sys() only fires once the effect row is reached."""
    # CPython sqlite3 stays 1 row ahead (reads tick k then steps to k+1 before returning k),
    # so ticks 0..2 yield with sys() still uncalled, and reading tick 3 triggers tick 4's sys().
    sys_calls = 0

    def counting_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        nonlocal sys_calls
        sys_calls += 1
        return {effects[0]["id"]: {"ok": True, "value": {"content": "ok"}}}

    with closing(open_db()) as conn:
        _add_agent(conn)
        stream = run_kernel(
            conn,
            agent_id="assistant",
            input_data={"prompt": "hi"},
            sys_fn=counting_sys,
        )
        with closing(stream):
            phases = [next(stream).phase for _ in range(3)]
            assert phases == ["sched", "llm_step", "sched"]
            assert sys_calls == 0

            assert next(stream).phase == "effect"
            assert sys_calls == 1


def test_error_completion_fails_task_cleanly() -> None:
    """An ok=false completion settles t1 as failed with the error stored in result."""
    def failing_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        return {effects[0]["id"]: {"ok": False, "error": "rate limited"}}

    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "hi"},
                sys_fn=failing_sys,
            )
        )

    last = ticks[-1]
    assert last.phase == "halt"
    assert last.status == "failed"
    assert last.completions == {}
    assert last.tasks["t1"]["status"] == "failed"
    assert last.tasks["t1"]["result"] == {"error": "rate limited"}


def test_raising_sys_becomes_error_completion() -> None:
    """An exception raised by sys_fn is caught at the UDF boundary and fails t1 as data."""
    def exploding_sys(_effects: list[dict[str, Any]]) -> dict[str, Any]:
        raise RuntimeError("boom")

    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "hi"},
                sys_fn=exploding_sys,
            )
        )

    last = ticks[-1]
    assert last.phase == "halt"
    assert last.status == "failed"
    assert last.tasks["t1"]["result"] == {"error": "boom"}


def test_max_ticks_stops_run_with_limit() -> None:
    """Hitting settings.max_ticks stops the loop cleanly with status='limit'."""
    with closing(open_db()) as conn:
        _add_agent(conn)
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('max_ticks', '3')")
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "hi"},
                sys_fn=lambda e: {e[0]["id"]: {"ok": True, "value": {"content": "ok"}}},
            )
        )

    last = ticks[-1]
    assert last.phase == "halt"
    assert last.status == "limit"
    # max_ticks=3 is checked on sched ticks (0, 2, 4); tick 4 sees 4 >= 3 and halts on tick 5.
    assert last.tick == 5


def test_missing_completion_ends_with_stuck() -> None:
    """If sys() drops an expected completion and pending is empty, sched halts as 'stuck'."""
    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "hi"},
                sys_fn=lambda _effects: {},
            )
        )

    last = ticks[-1]
    assert last.phase == "halt"
    assert last.status == "stuck"


def test_effect_merge_preserves_nested_json_types() -> None:
    """Effect completion merge keeps nested nulls, booleans, arrays, and objects intact."""
    # Guards against json_patch null-stripping and json_each boolean/container coercion.
    payload = {
        "content": "ok",
        "meta": {
            "nullable": None,
            "flag_t": True,
            "flag_f": False,
            "items": [1, None, True, False, {"nested": "v"}],
        },
    }

    def rich_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
        return {effects[0]["id"]: {"ok": True, "value": payload}}

    with closing(open_db()) as conn:
        _add_agent(conn)
        ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "hi"},
                sys_fn=rich_sys,
            )
        )

    post_effect = ticks[4]
    assert ticks[3].phase == "effect"
    assert post_effect.phase == "sched"
    assert post_effect.completions == {"r1_e2": {"ok": True, "value": payload}}


def test_unknown_agent_raises_value_error() -> None:
    """Requesting a missing agent raises ValueError instead of silently yielding zero ticks."""
    with closing(open_db()) as conn, pytest.raises(ValueError, match="Unknown agent: missing"):
        list(
            run_kernel(
                conn,
                agent_id="missing",
                input_data={"prompt": "hi"},
                sys_fn=lambda _e: {},
            )
        )
