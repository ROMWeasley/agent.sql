-- Idempotent v0 schema applied on every open_db() call; safe against existing data
-- because tables use IF NOT EXISTS and defaults use INSERT OR IGNORE.

CREATE TABLE IF NOT EXISTS settings (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Identifiers (`id`, `output_key`) are interpolated directly into SQLite JSON paths
-- (`'$.' || id`), so restrict them to path-safe characters at insert time.
CREATE TABLE IF NOT EXISTS agents (
    id           TEXT PRIMARY KEY
                 CHECK (id != '' AND id NOT GLOB '*[^a-zA-Z0-9_-]*'),
    kind         TEXT NOT NULL
                 CHECK (kind IN ('llm', 'call', 'human', 'sequential', 'parallel', 'loop', 'router', 'map')),
    model_id     TEXT,
    instruction  TEXT NOT NULL DEFAULT '',
    output_key   TEXT
                 CHECK (output_key IS NULL OR (output_key != '' AND output_key NOT GLOB '*[^a-zA-Z0-9_-]*')),
    description  TEXT,
    input_schema TEXT CHECK (input_schema IS NULL OR json_valid(input_schema)),
    config       TEXT NOT NULL DEFAULT '{}' CHECK (json_valid(config))
);

CREATE TABLE IF NOT EXISTS runs (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    agent_id    TEXT    NOT NULL REFERENCES agents(id),
    session_id  TEXT,
    status      TEXT    NOT NULL DEFAULT 'running'
                CHECK (status IN ('running', 'done', 'failed', 'limit', 'stuck', 'suspended', 'cancelled')),
    input       TEXT    NOT NULL DEFAULT '{}'
                CHECK (json_valid(input) AND json_type(input) = 'object'),
    result      TEXT    CHECK (result IS NULL OR json_valid(result)),
    usage       TEXT    NOT NULL DEFAULT '{"tokens":0}' CHECK (json_valid(usage)),
    definitions TEXT    NOT NULL DEFAULT '{}' CHECK (json_valid(definitions)),
    started_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    finished_at TEXT
);

-- Durable effect journal written by sys() as each effect completes; powers idempotency and replay.
CREATE TABLE IF NOT EXISTS effects (
    effect_id  TEXT    PRIMARY KEY
               CHECK (effect_id != '' AND effect_id NOT GLOB '*[^a-zA-Z0-9_-]*'),
    run_id     INTEGER NOT NULL REFERENCES runs(id),
    op         TEXT    NOT NULL,
    args       TEXT    NOT NULL CHECK (json_valid(args)),
    completion TEXT    NOT NULL CHECK (json_valid(completion)),
    created_at TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_effects_run_id ON effects(run_id);

CREATE TABLE IF NOT EXISTS trace (
    run_id      INTEGER NOT NULL REFERENCES runs(id),
    tick        INTEGER NOT NULL,
    seq         INTEGER NOT NULL,
    status      TEXT    NOT NULL,
    phase       TEXT    NOT NULL,
    active_task TEXT,
    tasks       TEXT    CHECK (tasks IS NULL OR json_valid(tasks)),
    pending     TEXT    CHECK (pending IS NULL OR json_valid(pending)),
    completions TEXT    CHECK (completions IS NULL OR json_valid(completions)),
    usage       TEXT    CHECK (usage IS NULL OR json_valid(usage)),
    created_at  TEXT    NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
    PRIMARY KEY (run_id, tick)
);

INSERT OR IGNORE INTO settings (key, value) VALUES ('max_ticks', '50');
INSERT OR IGNORE INTO settings (key, value) VALUES ('trace_level', 'full');

-- Checked by host/db.py on open to fail fast on incompatible schema versions.
PRAGMA user_version = 1;
