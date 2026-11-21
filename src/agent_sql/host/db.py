"""SQLite connection setup, idempotent schema initialization, and reset helpers."""

import sqlite3
from pathlib import Path

SCHEMA_VERSION = 1
KERNEL_DIR = Path(__file__).resolve().parent.parent / "kernel"


def init_db(conn: sqlite3.Connection) -> None:
    """Apply connection pragmas and ensure the schema is present."""
    conn.execute("PRAGMA foreign_keys = ON")
    # Spill CTE/subquery temp materializations to disk rather than exhausting RAM.
    conn.execute("PRAGMA temp_store = FILE")
    conn.execute("PRAGMA busy_timeout = 5000")
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
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def reset_db(path: Path) -> sqlite3.Connection:
    """Delete a file-backed SQLite database (and WAL sidecars) and recreate it."""
    for suffix in ("", "-wal", "-shm"):
        Path(f"{path}{suffix}").unlink(missing_ok=True)
    return open_db(path)
