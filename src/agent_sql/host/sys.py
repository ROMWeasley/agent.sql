"""Stateless sys() effect dispatcher with journal-backed idempotency."""

import json
import sqlite3
import time
from collections.abc import Callable, Mapping
from contextlib import closing
from typing import Any

# Takes an effect's `args` dict and returns an envelope {"ok": bool, ...}.
type EffectHandler = Callable[[dict[str, Any]], dict[str, Any]]


def _canonical_json(value: Any) -> str:
    """Serialize JSON canonically so dict key order never triggers false mismatches.

    Comparing canonical JSON strings also preserves strict JSON types (for example,
    1 vs true, which Python dict equality treats as equal).
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _with_timing(result: dict[str, Any], elapsed_ms: int) -> dict[str, Any]:
    """Shallow-copy a completion dict and ensure meta.ms is set without mutating handler output."""
    completion = dict(result)
    existing_meta = completion.get("meta")
    meta = dict(existing_meta) if isinstance(existing_meta, dict) else {}
    meta.setdefault("ms", elapsed_ms)
    completion["meta"] = meta
    return completion


class SysDispatcher:
    """Dispatches pending effect batches to op handlers and journals every completion.

    Lookup checks the durable `effects` table before calling a handler:
    - Same id + matching (op, args): return cached completion (replay / crash recovery).
    - Same id + mismatched (op, args): return an error without overwriting the journal.
    - New id: run handler, attach meta.ms, persist immediately to `effects`, and commit.
    """

    def __init__(
        self,
        handlers: Mapping[str, EffectHandler],
        *,
        writer: sqlite3.Connection,
        run_id: int,
    ) -> None:
        """Bind op handlers to the serialized WAL writer connection and active run_id."""
        self._handlers = handlers
        self._writer = writer
        self._run_id = run_id

    def __call__(self, effects: list[dict[str, Any]]) -> dict[str, Any]:
        """Dispatch one pending batch from SQLite into an {effect_id: completion} map."""
        if not isinstance(effects, list):
            raise TypeError(f"effects batch must be a list, got {type(effects).__name__}")

        completions: dict[str, Any] = {}
        for eff in effects:
            if not isinstance(eff, dict):
                continue
            eff_id = eff.get("id")
            if not isinstance(eff_id, str) or not eff_id:
                continue
            completions[eff_id] = self._dispatch_one(eff_id, eff)
        return completions

    def _dispatch_one(self, eff_id: str, eff: dict[str, Any]) -> dict[str, Any]:
        """Validate one effect, serve or reject from the effects journal, or run and persist it."""
        op = eff.get("op")
        if not isinstance(op, str) or not op:
            return {"ok": False, "error": f"Effect {eff_id} is missing a valid op"}

        args = eff.get("args", {})
        if not isinstance(args, dict):
            return {"ok": False, "error": f"Effect args for {eff_id} must be an object"}

        try:
            canonical_args = _canonical_json(args)
        except TypeError as exc:
            return {"ok": False, "error": f"Effect args for {eff_id} are not JSON: {exc}"}

        # Check the journal before dispatching so replay works even with an empty handler map.
        with closing(
            self._writer.execute(
                "SELECT op, args, completion FROM effects WHERE effect_id = ?",
                (eff_id,),
            )
        ) as cur:
            row = cur.fetchone()

        if row is not None:
            stored_op = row["op"]
            stored_args = row["args"]
            stored_completion = row["completion"]
            if stored_op == op and stored_args == canonical_args:
                return json.loads(stored_completion)
            return {
                "ok": False,
                "error": f"Effect idempotency mismatch for {eff_id}: expected {stored_op}",
            }

        completion = self._run_handler(op, args)
        return self._persist(eff_id, op, canonical_args, completion)

    def _run_handler(self, op: str, args: dict[str, Any]) -> dict[str, Any]:
        """Invoke the matching handler, time it, and turn unknown ops or crashes into data."""
        handler = self._handlers.get(op)
        if handler is None:
            return {"ok": False, "error": f"Unknown effect op: {op}"}

        t0 = time.perf_counter()
        try:
            raw = handler(args)
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            if not isinstance(raw, dict) or not isinstance(raw.get("ok"), bool):
                return {
                    "ok": False,
                    "error": f"Handler for {op} must return a dict with boolean 'ok'",
                    "meta": {"ms": elapsed_ms},
                }
            return _with_timing(raw, elapsed_ms)
        except Exception as exc:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            return {"ok": False, "error": str(exc), "meta": {"ms": elapsed_ms}}

    def _persist(
        self,
        eff_id: str,
        op: str,
        canonical_args: str,
        completion: dict[str, Any],
    ) -> dict[str, Any]:
        """Insert and commit a completed effect row immediately for crash-recovery durability."""
        try:
            completion_json = json.dumps(completion)
        except TypeError as exc:
            completion = {"ok": False, "error": f"Non-JSON completion for {op}: {exc}"}
            completion_json = json.dumps(completion)

        # Commit each effect as it finishes so a crash mid-batch preserves completed work.
        try:
            self._writer.execute(
                """
                INSERT INTO effects (effect_id, run_id, op, args, completion)
                VALUES (?, ?, ?, ?, ?)
                """,
                (eff_id, self._run_id, op, canonical_args, completion_json),
            )
            self._writer.commit()
        except Exception as exc:
            self._writer.rollback()
            return {"ok": False, "error": f"Failed to persist effect {eff_id}: {exc}"}
        return completion
