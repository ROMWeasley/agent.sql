"""Domain-neutral SQLite kernel runner and sys() UDF bridge."""

import json
import sqlite3
from collections.abc import Callable, Iterator
from contextlib import closing
from dataclasses import dataclass
from typing import Any

from agent_sql.host.db import KERNEL_DIR, open_db

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


def load_kernel_sql() -> str:
    return (KERNEL_DIR / "kernel.sql").read_text(encoding="utf-8")


def run_kernel(
    conn: sqlite3.Connection,
    *,
    agent_id: str,
    prompt: str,
    sys_fn: SysFn,
    run_id: int = 1,
) -> Iterator[Tick]:
    """Stream ticks from kernel.sql, wrapping sys_fn so exceptions become data."""

    # create sy function
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

    # register sys function
    conn.create_function("sys", 1, _sys)
    cursor = conn.execute(
        load_kernel_sql(),
        {"run_id": run_id, "agent_id": agent_id, "prompt": prompt},
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


def format_tick(t: Tick) -> str:
    pending_ids = [str(eff.get("id", "?")) for eff in t.pending]
    comp_keys = list(t.completions)
    return (
        f"tick={t.tick:<2d} phase={t.phase:<8s} status={t.status:<7s} "
        f"active={t.active_task or '-':<4s} pending={pending_ids} completions={comp_keys}"
    )


# Standalone offline demo for Step 1 (`PYTHONPATH=src uv run python -m agent_sql.host.runner`).
def _demo_sys(effects: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for eff in effects:
        if eff["op"] == "llm.chat":
            user_msg = eff["args"]["messages"][-1]["content"]
            out[eff["id"]] = {
                "ok": True,
                "value": {
                    "content": f"Canned reply to: {user_msg}",
                    "usage": {"total_tokens": 12},
                },
            }
        else:
            out[eff["id"]] = {"ok": False, "error": f"unsupported op: {eff['op']}"}
    return out


def main() -> None:
    with closing(open_db()) as conn:
        conn.execute(
            "INSERT INTO agents (id, kind, model_id, instruction) VALUES (?, 'llm', ?, ?)",
            ("assistant", "demo-model", "Be concise."),
        )
        last: Tick | None = None
        for tick in run_kernel(
            conn,
            agent_id="assistant",
            prompt="Hello SQL!",
            sys_fn=_demo_sys,
        ):
            print(format_tick(tick))
            last = tick
        if last is not None:
            print("result:", last.tasks["t1"]["result"])
            print("usage:", last.usage)


if __name__ == "__main__":
    main()
