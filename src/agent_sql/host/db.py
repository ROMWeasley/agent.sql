"""SQLite connection setup, idempotent schema initialization, and WAL helpers."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernel"


def _apply_pragmas(conn: sqlite3.Connection) -> None:
    """Apply per-connection pragmas (foreign keys, disk temp spill, lock timeout, NORMAL sync)."""
    conn.execute("PRAGMA foreign_keys = ON")
    # Spill CTE/subquery temp materializations to disk rather than exhausting RAM.
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA busy_timeout = 5000")
    conn.execute("PRAGMA synchronous = NORMAL")


def connect_db(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a SQLite connection with runtime pragmas, without running DDL.

    Separate reader connections (such as the kernel cursor under WAL) use this
    so they never contend for a write lock on an already-initialized database.
    """
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    _apply_pragmas(conn)
    return conn


def init_db(conn: sqlite3.Connection) -> None:
    """Apply pragmas and idempotently ensure the v1 schema and default settings exist."""
    _apply_pragmas(conn)
    # WAL lets readers/writers operate concurrently while the kernel cursor stays open.
    conn.execute("PRAGMA journal_mode = WAL")

    # Fail fast instead of silently mutating an unknown or newer schema version.
    version = int(conn.execute("PRAGMA user_version").fetchone()[0])
    if version not in (0, SCHEMA_VERSION):
        raise RuntimeError(
            f"Unsupported schema version {version} (expected 0 or {SCHEMA_VERSION})"
        )

    schema_sql = (KERNEL_DIR / "schema.sql").read_text(encoding="utf-8")
    conn.executescript(schema_sql)


def open_db(path: str | Path = ":memory:") -> sqlite3.Connection:
    """Open a DB connection, safely initializing new DBs while leaving existing data intact."""
    conn = connect_db(path)
    try:
        init_db(conn)
    except Exception:
        conn.close()
        raise
    return conn


def checkpoint_wal(conn: sqlite3.Connection) -> tuple[int, int, int]:
    """Truncate the WAL back to zero bytes; returns (busy, log_frames, checkpointed_frames)."""
    busy, log, checkpointed = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    return int(busy), int(log), int(checkpointed)


def reset_db(path: Path) -> sqlite3.Connection:
    """Delete a file-backed SQLite database (and WAL sidecars) and recreate it."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    return open_db(path)
