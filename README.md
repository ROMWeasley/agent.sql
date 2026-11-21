# agent.sql

An agent framework where the entire agent loop runs inside a single SQLite `WITH RECURSIVE` query.

Most "X in SQL" projects use SQL as a compute step driven by an outer Python or Bash `while` loop. Here, the query *is* the loop. Routing, tool calls, sub-agents, retries, and stop conditions all happen in SQL. Python only handles the things SQL physically cannot do: LLM requests, execution of external tools, and talking to the user.

## Why?

- **Why not?**
- Building yet another Python agent framework would be boring. So since SQLite's `WITH RECURSIVE` is Turing-complete, pushing an entire agent runtime into one query seems like a reasonable thing to do. Especially now that AI assistants make it realistic to actually build and maintain gnarly recursive SQL.

## How it works

```
REDUCE (pure SQL)   -->  advances runnable agent tasks, collects I/O requests
EFFECT (1 UDF call) -->  calls sys(batch) once; Python runs LLM / tools / I/O concurrently
REDUCE (pure SQL)   -->  feeds results back to waiting tasks, repeats until done
```

- **One query per run:** The whole control flow sits in `kernel.sql`.
- **One Python boundary:** Python exposes a single function, `sys()`, which executes batched side effects and returns results as JSON. No routing or control flow lives in Python.
- **Agents are rows:** Defining an agent, wiring sub-agents, or attaching tools is just an `INSERT`. The kernel query itself never changes.

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

Early planning, no runnable code yet.

- [x] Core architecture draft ([`docs/architecture.md`](docs/architecture.md))
- [ ] **Spike 0:** verify SQLite assumptions (CTE streaming, nested JSON aggregates, WAL writer concurrency)
- [ ] **v0:** walking skeleton with a single `llm` agent loop, `sys()`, Python tools, and CLI

The architecture may still change based on what Spike 0 reveals.