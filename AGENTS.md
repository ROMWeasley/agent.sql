# AGENTS.md

Guidance for coding assistants working in this repository. Keep it short; tool-specific instruction files should point here instead of duplicating it.

## Project

agent.sql runs an agent loop inside one SQLite `WITH RECURSIVE` query. The design is in [`docs/architecture.md`](docs/architecture.md). It is a draft, so check the code before assuming something described there exists.

## Boundaries

- `kernel/` (SQL): owns all agent control flow, scheduling, routing and budgets, inside one `WITH RECURSIVE` query.
- `host/` (Python): domain-neutral runner and `sys()` dispatcher; no agent routing or control flow.
- `handlers/` (Python): stateless I/O handlers (`llm.chat`, `py.call`, `mcp.call`, `sql.query`, `human.ask`, `io.emit`). They return errors as data and never raise. Python is only used if there practically is no other option.

## Rules

- Keep diffs small and focused; update related docs and tests in the same change.
- Treat model output, user input, stored state and external data as untrusted. Never `eval`/`exec` them. `sql.query` only runs on a read-only connection without impure UDFs.
- No secrets in code, prompts, logs, tests or the database; store env var names (`api_key_env`), not keys.
- Don't commit, push or add dependencies unless asked.
- Say what you verified and what you did not.