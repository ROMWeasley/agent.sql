"""Domain-neutral SQLite kernel runner, WAL run lifecycle, and sys() UDF bridge."""

import json
import sqlite3
import tempfile
from collections.abc import Callable, Iterator, Mapping
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent_sql.host.db import KERNEL_DIR, checkpoint_wal, connect_db, open_db
from agent_sql.host.sys import EffectHandler, SysDispatcher

# Maps a batch of pending effect dicts to {effect_id: completion_dict}.
type SysFn = Callable[[list[dict[str, Any]]], dict[str, Any]]


@dataclass(frozen=True, slots=True)
class Tick:
    """Parsed snapshot of a single row emitted by kernel.sql."""

    tick: int
    seq: int
    status: str
    phase: str
    active_task: str | None
    tasks: dict[str, Any]
    pending: list[dict[str, Any]]
    completions: dict[str, Any]
    usage: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RunResult:
    """Final outcome returned by run_agent after finish.sql commits."""

    run_id: int
    status: str
    result: Any
    usage: dict[str, Any]
    ticks: int


def load_sql(name: str) -> str:
    """Read a SQL statement file from the kernel/ directory."""
    return (KERNEL_DIR / name).read_text(encoding="utf-8")


def load_kernel_sql() -> str:
    """Read the main recursive CTE query (kernel.sql)."""
    return load_sql("kernel.sql")


def run_kernel(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    input_data: dict[str, Any] | None = None,
    sys_fn: SysFn,
    run_id: int = 1,
) -> Iterator[Tick]:
    """Low-level CTE streamer on one connection; writes no runs or trace rows.

    Validates `input_data` eagerly and wraps `sys_fn` so exceptions become error completions.
    """
    payload = {} if input_data is None else input_data
    if not isinstance(payload, dict):
        raise TypeError(f"input_data must be a dict, got {type(payload).__name__}")
    input_json = json.dumps(payload)

    def _stream() -> Iterator[Tick]:
        # An uncaught exception in a Python UDF aborts the SQLite query, so convert any
        # failure into per-effect error completions and let SQL handle task failure.
        def _sys(pending_json: str) -> str:
            effects: list[dict[str, Any]] = []
            try:
                effects = json.loads(pending_json)
                result = sys_fn(effects)
                if not isinstance(result, dict):
                    raise TypeError(f"sys_fn must return a dict, got {type(result).__name__}")
                return json.dumps(result)
            except Exception as exc:
                return json.dumps(
                    {
                        eff["id"]: {"ok": False, "error": str(exc)}
                        for eff in effects
                        if isinstance(eff, dict) and "id" in eff
                    }
                )

        conn.create_function("sys", 1, _sys)
        cursor = conn.execute(
            load_kernel_sql(),
            {"run_id": run_id, "agent_id": agent_id, "input": input_json},
        )
        yielded = False
        try:
            for row in cursor:
                yielded = True
                (
                    tick,
                    seq,
                    status,
                    phase,
                    active_task,
                    tasks,
                    pending,
                    completions,
                    usage,
                ) = tuple(row)
                yield Tick(
                    tick=tick,
                    seq=seq,
                    status=status,
                    phase=phase,
                    active_task=active_task,
                    tasks=json.loads(tasks),
                    pending=json.loads(pending),
                    completions=json.loads(completions),
                    usage=json.loads(usage),
                )
        finally:
            # Close promptly if the caller breaks early so the read snapshot isn't held open.
            cursor.close()

        # The anchor SELECT filters on `agents.id = :agent_id`, so 0 rows means unknown agent.
        if not yielded:
            raise ValueError(f"Unknown agent: {agent_id}")

    return _stream()


def run_agent(
    db_path: Path | str,
    *,
    agent_id: str,
    input_data: dict[str, Any] | None = None,
    handlers: Mapping[str, EffectHandler],
    session_id: str | None = None,
    on_tick: Callable[[Tick], None] | None = None,
) -> RunResult:
    """Execute one persisted agent run: start.sql -> kernel.sql -> finish.sql.

    Uses two WAL connections: a serialized `writer` for runs/effects/trace and a
    separate `kernel_conn` whose read snapshot stays isolated for the whole CTE.
    """
    if str(db_path) == ":memory:":
        raise ValueError("run_agent requires a file-backed database (WAL cannot share :memory:)")

    payload = {} if input_data is None else input_data
    if not isinstance(payload, dict):
        raise TypeError(f"input_data must be a dict, got {type(payload).__name__}")
    input_json = json.dumps(payload)

    writer = open_db(db_path)
    kernel_conn: sqlite3.Connection | None = None
    try:
        # 1. Start the run row and freeze agent + settings definitions via start.sql.
        with closing(
            writer.execute(
                load_sql("start.sql"),
                {"agent_id": agent_id, "input": input_json, "session_id": session_id},
            )
        ) as cur:
            start_row = cur.fetchone()
        if start_row is None:
            writer.rollback()
            raise ValueError(f"Unknown agent: {agent_id}")
        run_id = int(start_row["id"])
        writer.commit()

        trace_row = writer.execute(
            "SELECT value FROM settings WHERE key = 'trace_level'"
        ).fetchone()
        trace_level = str(trace_row["value"]) if trace_row is not None else "full"

        # 2. Open kernel_conn after committing start.sql so its snapshot sees the new run row,
        # then stream kernel.sql ticks while committing trace rows live on writer.
        kernel_conn = connect_db(db_path)
        dispatcher = SysDispatcher(handlers, writer=writer, run_id=run_id)

        last_tick: Tick | None = None
        tick_count = 0
        try:
            with closing(
                run_kernel(
                    kernel_conn,
                    agent_id=agent_id,
                    input_data=payload,
                    sys_fn=dispatcher,
                    run_id=run_id,
                )
            ) as stream:
                for tick in stream:
                    tick_count += 1
                    last_tick = tick
                    if trace_level != "off":
                        writer.execute(
                            """
                            INSERT INTO trace (
                                run_id, tick, seq, status, phase, active_task,
                                tasks, pending, completions, usage
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """,
                            (
                                run_id,
                                tick.tick,
                                tick.seq,
                                tick.status,
                                tick.phase,
                                tick.active_task,
                                json.dumps(tick.tasks),
                                json.dumps(tick.pending),
                                json.dumps(tick.completions),
                                json.dumps(tick.usage),
                            ),
                        )
                        writer.commit()
                    if on_tick is not None:
                        on_tick(tick)
        except Exception as exc:
            kernel_conn.close()
            kernel_conn = None
            writer.rollback()
            writer.execute(
                load_sql("finish.sql"),
                {
                    "run_id": run_id,
                    "status": "failed",
                    "result": json.dumps({"error": str(exc)}),
                    "usage": json.dumps(
                        last_tick.usage if last_tick is not None else {"tokens": 0}
                    ),
                },
            )
            writer.commit()
            raise

        # 3. Close kernel_conn before finish/checkpoint so no reader blocks WAL truncation.
        kernel_conn.close()
        kernel_conn = None
        if last_tick is None:
            raise RuntimeError(f"Kernel produced no ticks for run {run_id}")

        t1_result = last_tick.tasks.get("t1", {}).get("result")
        writer.execute(
            load_sql("finish.sql"),
            {
                "run_id": run_id,
                "status": last_tick.status,
                "result": json.dumps(t1_result) if t1_result is not None else None,
                "usage": json.dumps(last_tick.usage),
            },
        )
        writer.commit()

        return RunResult(
            run_id=run_id,
            status=last_tick.status,
            result=t1_result,
            usage=last_tick.usage,
            ticks=tick_count,
        )
    finally:
        if kernel_conn is not None:
            kernel_conn.close()
        checkpoint_wal(writer)
        writer.close()


def format_tick(t: Tick) -> str:
    """Format a single Tick as a compact one-line summary for terminal output."""
    pending_ids = [str(eff.get("id", "?")) for eff in t.pending]
    comp_keys = list(t.completions)
    return (
        f"tick={t.tick:<2d} phase={t.phase:<8s} status={t.status:<7s} "
        f"active={t.active_task or '-':<4s} pending={pending_ids} completions={comp_keys}"
    )


# Standalone offline demo (`PYTHONPATH=src uv run python -m agent_sql.host.runner`).
def _demo_llm(args: dict[str, Any]) -> dict[str, Any]:
    """Stub llm.chat handler for the offline runner demo."""
    user_msg = args["messages"][-1]["content"] if args.get("messages") else "(no prompt)"
    return {
        "ok": True,
        "value": {
            "content": f"Canned reply to: {user_msg}",
            "usage": {"total_tokens": 12},
        },
    }


def main() -> None:
    """Run a self-contained offline demo against a temporary WAL database."""
    with tempfile.TemporaryDirectory(prefix="agent_sql_demo_") as tmp_dir:
        db_path = Path(tmp_dir) / "demo.db"
        with closing(open_db(db_path)) as conn:
            conn.execute(
                "INSERT INTO agents (id, kind, model_id, instruction) VALUES (?, 'llm', ?, ?)",
                ("assistant", "demo-model", "Be concise."),
            )
            conn.commit()

        outcome = run_agent(
            db_path,
            agent_id="assistant",
            input_data={"prompt": "Hello SQL!"},
            handlers={"llm.chat": _demo_llm},
            on_tick=lambda t: print(format_tick(t)),
        )
        with closing(connect_db(db_path)) as conn:
            run_row = conn.execute(
                "SELECT id, status, result, usage FROM runs WHERE id = ?",
                (outcome.run_id,),
            ).fetchone()
            effects_n = conn.execute(
                "SELECT count(*) FROM effects WHERE run_id = ?",
                (outcome.run_id,),
            ).fetchone()[0]
            trace_n = conn.execute(
                "SELECT count(*) FROM trace WHERE run_id = ?",
                (outcome.run_id,),
            ).fetchone()[0]

        print("run:", dict(run_row))
        print(f"persisted: effects={effects_n}, trace={trace_n}")


if __name__ == "__main__":
    main()
