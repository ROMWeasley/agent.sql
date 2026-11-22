"""Tests for Step 2: SysDispatcher idempotency, run lifecycle, and WAL isolation."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

import pytest

from agent_sql.host.db import connect_db, open_db
from agent_sql.host.runner import Tick, run_agent, run_kernel
from agent_sql.host.sys import SysDispatcher


def _seed_run(conn: sqlite3.Connection, agent_id: str = "assistant") -> int:
    """Insert a test agent and open a parent runs row, returning its run_id."""
    conn.execute(
        "INSERT INTO agents (id, kind, model_id, instruction)"
        " VALUES (?, 'llm', 'm1', 'Be concise.')",
        (agent_id,),
    )
    run_id = int(
        conn.execute(
            "INSERT INTO runs (agent_id) VALUES (?) RETURNING id",
            (agent_id,),
        ).fetchone()[0]
    )
    conn.commit()
    return run_id


def _seed_agent(
    db_path: Path,
    agent_id: str = "assistant",
    *,
    instruction: str = "Be concise.",
) -> None:
    """Initialize a file-backed DB with a single llm agent definition."""
    with closing(open_db(db_path)) as conn:
        conn.execute(
            "INSERT INTO agents (id, kind, model_id, instruction) VALUES (?, 'llm', ?, ?)",
            (agent_id, "demo-model", instruction),
        )
        conn.commit()


def test_dispatcher_journals_effect_and_adds_meta_ms(tmp_path: Path) -> None:
    """SysDispatcher executes the op handler, adds meta.ms, and commits a row to effects."""
    db_path = tmp_path / "sys.db"
    with closing(open_db(db_path)) as conn:
        run_id = _seed_run(conn)
        seen_args: list[dict[str, Any]] = []

        def fake_llm(args: dict[str, Any]) -> dict[str, Any]:
            seen_args.append(args)
            return {"ok": True, "value": {"content": "hello"}}

        dispatcher = SysDispatcher({"llm.chat": fake_llm}, writer=conn, run_id=run_id)
        out = dispatcher([{"id": "r1_e2", "op": "llm.chat", "args": {"model_id": "m1"}}])

        assert seen_args == [{"model_id": "m1"}]
        comp = out["r1_e2"]
        assert comp["ok"] is True
        assert comp["value"] == {"content": "hello"}
        assert isinstance(comp["meta"]["ms"], int)
        assert comp["meta"]["ms"] >= 0

        row = conn.execute(
            "SELECT run_id, op, args, completion FROM effects WHERE effect_id = 'r1_e2'"
        ).fetchone()
        assert row is not None
        assert row["run_id"] == run_id
        assert row["op"] == "llm.chat"
        assert json.loads(row["completion"]) == comp


def test_dispatcher_idempotency_hit_skips_handler_with_reordered_keys(tmp_path: Path) -> None:
    """Re-ordered dict keys under the same effect_id hit the effects cache without re-running."""
    db_path = tmp_path / "sys.db"
    with closing(open_db(db_path)) as conn:
        run_id = _seed_run(conn)
        calls = 0

        def fake_llm(_args: dict[str, Any]) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {"ok": True, "value": {"content": "cached"}}

        dispatcher = SysDispatcher({"llm.chat": fake_llm}, writer=conn, run_id=run_id)
        first = dispatcher(
            [{"id": "r1_e2", "op": "llm.chat", "args": {"a": 1, "b": {"x": 10, "y": 20}}}]
        )
        second = dispatcher(
            [{"id": "r1_e2", "op": "llm.chat", "args": {"b": {"y": 20, "x": 10}, "a": 1}}]
        )

        assert calls == 1
        assert first == second


def test_dispatcher_idempotency_mismatch_returns_error_without_overwriting(
    tmp_path: Path,
) -> None:
    """Changed args or op for an existing effect_id returns an error without overwriting effects."""
    db_path = tmp_path / "sys.db"
    with closing(open_db(db_path)) as conn:
        run_id = _seed_run(conn)
        calls = 0

        def fake_llm(_args: dict[str, Any]) -> dict[str, Any]:
            nonlocal calls
            calls += 1
            return {"ok": True, "value": {"content": "original"}}

        dispatcher = SysDispatcher({"llm.chat": fake_llm}, writer=conn, run_id=run_id)
        original = dispatcher([{"id": "r1_e2", "op": "llm.chat", "args": {"prompt": "one"}}])

        diff_args = dispatcher([{"id": "r1_e2", "op": "llm.chat", "args": {"prompt": "two"}}])
        diff_op = dispatcher([{"id": "r1_e2", "op": "py.call", "args": {"prompt": "one"}}])

        assert calls == 1
        assert diff_args["r1_e2"]["ok"] is False
        assert "mismatch" in diff_args["r1_e2"]["error"]
        assert diff_op["r1_e2"]["ok"] is False
        assert "mismatch" in diff_op["r1_e2"]["error"]

        stored = conn.execute(
            "SELECT op, completion FROM effects WHERE effect_id = 'r1_e2'"
        ).fetchone()
        assert stored["op"] == "llm.chat"
        assert json.loads(stored["completion"]) == original["r1_e2"]


def test_dispatcher_distinguishes_int_from_bool_in_args(tmp_path: Path) -> None:
    """Canonical JSON check separates 1 from True so Python bool==int never masks a mismatch."""
    db_path = tmp_path / "sys.db"
    with closing(open_db(db_path)) as conn:
        run_id = _seed_run(conn)
        dispatcher = SysDispatcher(
            {"llm.chat": lambda _a: {"ok": True, "value": {"content": "ok"}}},
            writer=conn,
            run_id=run_id,
        )
        dispatcher([{"id": "r1_e2", "op": "llm.chat", "args": {"x": 1}}])
        mismatched = dispatcher([{"id": "r1_e2", "op": "llm.chat", "args": {"x": True}}])

        assert mismatched["r1_e2"]["ok"] is False
        assert "mismatch" in mismatched["r1_e2"]["error"]


def test_dispatcher_unknown_op_and_raising_handler_become_error_completions(
    tmp_path: Path,
) -> None:
    """Unknown ops and handler exceptions become ok=false completions and still get journaled."""
    db_path = tmp_path / "sys.db"
    with closing(open_db(db_path)) as conn:
        run_id = _seed_run(conn)

        def boom(_args: dict[str, Any]) -> dict[str, Any]:
            raise RuntimeError("network down")

        dispatcher = SysDispatcher({"llm.chat": boom}, writer=conn, run_id=run_id)
        out = dispatcher(
            [
                {"id": "r1_e2", "op": "nope.op", "args": {}},
                {"id": "r1_e3", "op": "llm.chat", "args": {}},
            ]
        )

        assert out["r1_e2"] == {"ok": False, "error": "Unknown effect op: nope.op"}
        assert out["r1_e3"]["ok"] is False
        assert out["r1_e3"]["error"] == "network down"
        assert conn.execute("SELECT count(*) FROM effects").fetchone()[0] == 2


def test_journal_replay_with_empty_handlers_matches_original_ticks(tmp_path: Path) -> None:
    """A recorded run replays identically from the effects table with an empty handler map."""
    db_path = tmp_path / "replay.db"
    _seed_agent(db_path)

    recorded_ticks: list[Tick] = []
    outcome = run_agent(
        db_path,
        agent_id="assistant",
        input_data={"prompt": "ping"},
        handlers={
            "llm.chat": lambda _a: {
                "ok": True,
                "value": {"content": "pong", "usage": {"total_tokens": 7}},
            }
        },
        on_tick=recorded_ticks.append,
    )

    with closing(connect_db(db_path)) as conn:
        replay_sys = SysDispatcher({}, writer=conn, run_id=outcome.run_id)
        replayed_ticks = list(
            run_kernel(
                conn,
                agent_id="assistant",
                input_data={"prompt": "ping"},
                sys_fn=replay_sys,
                run_id=outcome.run_id,
            )
        )

    assert replayed_ticks == recorded_ticks


def test_run_agent_persists_runs_effects_trace_and_truncates_wal(tmp_path: Path) -> None:
    """run_agent persists runs, effects, and all trace ticks, then truncates the WAL to 0 bytes."""
    db_path = tmp_path / "agent.db"
    _seed_agent(db_path)

    streamed: list[Tick] = []
    outcome = run_agent(
        db_path,
        agent_id="assistant",
        input_data={"prompt": "hello", "meta": 1},
        session_id="sess-42",
        handlers={
            "llm.chat": lambda _a: {
                "ok": True,
                "value": {"content": "hi there", "usage": {"total_tokens": 15}},
            }
        },
        on_tick=streamed.append,
    )

    assert outcome.status == "done"
    assert outcome.result == "hi there"
    assert outcome.usage == {"tokens": 15}
    assert outcome.ticks == 8
    assert len(streamed) == 8

    wal_path = Path(f"{db_path}-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0

    with closing(connect_db(db_path)) as conn:
        run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (outcome.run_id,)).fetchone()
        assert run_row["agent_id"] == "assistant"
        assert run_row["session_id"] == "sess-42"
        assert run_row["status"] == "done"
        assert json.loads(run_row["input"]) == {"prompt": "hello", "meta": 1}
        assert json.loads(run_row["result"]) == "hi there"
        assert json.loads(run_row["usage"]) == {"tokens": 15}
        assert run_row["started_at"] is not None
        assert run_row["finished_at"] is not None

        defs = json.loads(run_row["definitions"])
        assert defs["agent"]["id"] == "assistant"
        assert defs["agent"]["instruction"] == "Be concise."
        assert defs["settings"]["trace_level"] == "full"

        effects_rows = conn.execute(
            "SELECT * FROM effects WHERE run_id = ?", (outcome.run_id,)
        ).fetchall()
        assert len(effects_rows) == 1

        trace_rows = conn.execute(
            "SELECT tick, phase, status FROM trace WHERE run_id = ? ORDER BY tick",
            (outcome.run_id,),
        ).fetchall()
        assert len(trace_rows) == 8
        assert [r["phase"] for r in trace_rows] == [t.phase for t in streamed]


def test_run_agent_snapshot_isolation_and_live_observer_mid_run(tmp_path: Path) -> None:
    """mode=ro observer sees live trace/effects while kernel_conn ignores mid-run agent edits."""
    db_path = tmp_path / "wal.db"
    _seed_agent(db_path, instruction="v1")

    observed: list[tuple[int, int, int, str]] = []
    seen_system_prompts: list[str] = []

    def fake_llm(args: dict[str, Any]) -> dict[str, Any]:
        seen_system_prompts.append(args["messages"][0]["content"])
        return {"ok": True, "value": {"content": "ok"}}

    def on_tick(tick: Tick) -> None:
        # Mutate the agent table mid-run from a separate writer connection.
        if tick.tick == 1:
            with closing(connect_db(db_path)) as editor:
                editor.execute(
                    "UPDATE agents SET instruction = 'v2_mutated' WHERE id = 'assistant'"
                )
                editor.commit()

        with closing(sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)) as observer:
            observer.execute("PRAGMA busy_timeout = 5000")
            eff_n = observer.execute("SELECT count(*) FROM effects").fetchone()[0]
            tr_n = observer.execute("SELECT count(*) FROM trace").fetchone()[0]
            instr = observer.execute(
                "SELECT instruction FROM agents WHERE id = 'assistant'"
            ).fetchone()[0]
            observed.append((tick.tick, eff_n, tr_n, instr))

    outcome = run_agent(
        db_path,
        agent_id="assistant",
        input_data={"prompt": "hello"},
        handlers={"llm.chat": fake_llm},
        on_tick=on_tick,
    )

    assert seen_system_prompts == ["v1"]
    assert observed[0] == (0, 0, 1, "v1")
    # By tick 4 (post-effect sched), the effect row and tick-4 trace are committed live.
    assert observed[4] == (4, 1, 5, "v2_mutated")
    assert observed[-1] == (7, 1, 8, "v2_mutated")

    wal_path = Path(f"{db_path}-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0

    with closing(connect_db(db_path)) as conn:
        raw_defs = conn.execute(
            "SELECT definitions FROM runs WHERE id = ?", (outcome.run_id,)
        ).fetchone()[0]
        defs = json.loads(raw_defs)
        assert defs["agent"]["instruction"] == "v1"


def test_run_agent_trace_level_off_skips_trace_rows(tmp_path: Path) -> None:
    """Setting trace_level='off' skips trace inserts while still persisting runs and effects."""
    db_path = tmp_path / "quiet.db"
    _seed_agent(db_path)
    with closing(open_db(db_path)) as conn:
        conn.execute("INSERT OR REPLACE INTO settings (key, value) VALUES ('trace_level', 'off')")
        conn.commit()

    outcome = run_agent(
        db_path,
        agent_id="assistant",
        input_data={"prompt": "hi"},
        handlers={"llm.chat": lambda _a: {"ok": True, "value": {"content": "ok"}}},
    )
    assert outcome.status == "done"

    with closing(connect_db(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM trace").fetchone()[0] == 0
        assert conn.execute("SELECT count(*) FROM effects").fetchone()[0] == 1
        status = conn.execute(
            "SELECT status FROM runs WHERE id = ?", (outcome.run_id,)
        ).fetchone()[0]
        assert status == "done"


def test_run_agent_handler_error_persists_failed_run(tmp_path: Path) -> None:
    """A handler returning ok=false finishes the run row with status='failed' and error result."""
    db_path = tmp_path / "fail.db"
    _seed_agent(db_path)

    outcome = run_agent(
        db_path,
        agent_id="assistant",
        input_data={"prompt": "hi"},
        handlers={"llm.chat": lambda _a: {"ok": False, "error": "upstream 502"}},
    )
    assert outcome.status == "failed"
    assert outcome.result == {"error": "upstream 502"}

    with closing(connect_db(db_path)) as conn:
        row = conn.execute(
            "SELECT status, result FROM runs WHERE id = ?", (outcome.run_id,)
        ).fetchone()
        assert row["status"] == "failed"
        assert json.loads(row["result"]) == {"error": "upstream 502"}


def test_run_agent_mid_run_exception_marks_run_failed_and_truncates_wal(tmp_path: Path) -> None:
    """An exception mid-stream finalizes runs as failed and still truncates the WAL file."""
    db_path = tmp_path / "crash.db"
    _seed_agent(db_path)

    def crashing_observer(tick: Tick) -> None:
        if tick.tick == 2:
            raise RuntimeError("observer aborted")

    with pytest.raises(RuntimeError, match="observer aborted"):
        run_agent(
            db_path,
            agent_id="assistant",
            input_data={"prompt": "hi"},
            handlers={"llm.chat": lambda _a: {"ok": True, "value": {"content": "ok"}}},
            on_tick=crashing_observer,
        )

    wal_path = Path(f"{db_path}-wal")
    assert not wal_path.exists() or wal_path.stat().st_size == 0

    with closing(connect_db(db_path)) as conn:
        row = conn.execute("SELECT status, result, finished_at FROM runs").fetchone()
        assert row["status"] == "failed"
        assert json.loads(row["result"]) == {"error": "observer aborted"}
        assert row["finished_at"] is not None


def test_run_agent_unknown_agent_leaves_no_run_row(tmp_path: Path) -> None:
    """Unknown agent_id fails before creating a runs row and raises ValueError."""
    db_path = tmp_path / "empty.db"
    with closing(open_db(db_path)):
        pass

    with pytest.raises(ValueError, match="Unknown agent: ghost"):
        run_agent(db_path, agent_id="ghost", handlers={})

    with closing(connect_db(db_path)) as conn:
        assert conn.execute("SELECT count(*) FROM runs").fetchone()[0] == 0


def test_run_agent_rejects_memory_db() -> None:
    """run_agent rejects ':memory:' because two-connection WAL isolation requires a file DB."""
    with pytest.raises(ValueError, match="file-backed database"):
        run_agent(":memory:", agent_id="assistant", handlers={})
