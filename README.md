# agent.sql

An agent framework where the entire agent loop runs inside a single SQLite `WITH RECURSIVE` query.

Most "X in SQL" projects use SQL as a compute step driven by an outer Python or Bash `while` loop. Here, the query *is* the loop. Routing, tool calls, sub-agents, retries, and stop conditions all happen in SQL. Python only handles the things SQL physically cannot do: LLM requests, execution of external tools, and talking to the user.

## Why?

- **Why not?**
- Building yet another Python agent framework would be boring. So since SQLite's `WITH RECURSIVE` is Turing-complete, pushing an entire agent runtime into one query seems like a reasonable thing to do. Especially now that AI assistants make it realistic to actually build and maintain gnarly recursive SQL.

## How it works

An agent run is a deterministic state machine executed by one SQLite `WITH RECURSIVE` query. Each row of the query is one **tick** carrying the run's state (tasks, pending I/O, completions), streamed back to the host as it runs. SQL owns every decision; Python only injects a single `sys()` function for external I/O.

```text
SQLite (one WITH RECURSIVE query)                 Python host (thin I/O shell)
┌───────────────────────────────────────┐
│  REDUCE (pure SQL)                    │
│  pick runnable task, update state,    │
│  queue I/O requests into `pending`    │
└─────┬─────────────────────────────────┘
      │ when tasks        ▲ next tick
      ▼ need I/O          │ (until halt)
┌─────────────────────────┴─────────────┐  batch  ┌───────────────────────────┐
│  EFFECT (1 UDF call)                  ├────────►│  sys(pending)             │
│  merge results into `completions`     │◄────────┤  LLM / tools / MCP / I/O  │
└───────────────────────────────────────┘  JSON   └───────────────────────────┘
```

- **One query per run:** The whole state machine lives in `kernel.sql`. No outer Python loop driving it.
- **One Python boundary:** Python injects a single `sys()` UDF that runs batched side effects concurrently and returns results as JSON. Zero routing or control flow in Python.
- **Agents are rows:** Defining an agent, wiring sub-agents, or attaching tools is just an `INSERT`. The kernel query itself never changes.
- **Durable by accident:** Because the transition step is pure SQL and every `sys()` result can be journaled, deterministic replay, crash recovery, forking, and LLM-free tests fall out of the design.

For the full design (task map, WAL concurrency, replay, and SQLite details), see [`docs/architecture.md`](docs/architecture.md).

## Usage (planned)

Once v0 lands, defining and running an agent will look roughly like:

```sql
INSERT INTO agents (id, kind, model_id, instruction)
VALUES ('assistant', 'llm', 'fast', 'Answer clearly and use tools when needed.');
```

```bash
agent-sql run assistant "Check the latest status and summarize it."
```

## Current stage

Architecture verified

- [x] Core architecture baseline ([`docs/architecture.md`](docs/architecture.md))
- [x] **Spike 0:** verify SQLite assumptions (CTE streaming, nested JSON aggregates, WAL writer concurrency, Ctrl-C abort, JSON/JSONB limits)
- [ ] **v0** *(in progress)*: walking skeleton with a single `llm` agent loop, `sys()`, Python tools, and CLI
