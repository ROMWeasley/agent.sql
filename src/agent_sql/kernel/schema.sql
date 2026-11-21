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

INSERT OR IGNORE INTO settings (key, value) VALUES ('max_ticks', '50');

-- Checked by host/db.py on open to fail fast on incompatible schema versions.
PRAGMA user_version = 1;
