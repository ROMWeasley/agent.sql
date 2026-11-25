# agent.sql

An agent framework where the entire agent loop runs inside a single SQLite `WITH RECURSIVE` query.

Most "X in SQL" projects use SQL as a compute step driven by an outer Python or Bash `while` loop. Here, the query *is* the loop. Routing, tool calls, sub-agents, retries, and stop conditions all happen in SQL. Python only handles the things SQL physically cannot do: LLM requests, execution of external tools, and talking to the user.

## Why?

- **Why not?**
- Building yet another Python agent framework would be boring. So since SQLite's `WITH RECURSIVE` is Turing-complete, pushing an entire agent runtime into one query seems like a reasonable thing to do. Especially now that AI assistants make it realistic to actually build and maintain gnarly recursive SQL.

## How it works

The agent kernel is a deterministic state machine executed by one SQLite `WITH RECURSIVE` statement. Each row of the query is one **tick** carrying the run's state (tasks, pending I/O, completions, etc.), streamed back to the host as it runs:

- **One query per run:** The whole state machine runs inside `kernel.sql`.
- **One Python boundary:** Python injects a single `sys()` UDF that executes batched side effects concurrently and returns their results as JSON. Python only provides access to external I/O and tools; it is **not** involved in routing or the state machine itself.
- **Agents are rows:** Defining an agent, wiring sub-agents, or attaching tools is just an `INSERT` statement. The kernel query itself never changes.

Ticks alternate between pure-SQL **REDUCE** steps and batched **EFFECT** calls to `sys()`:

```text
SQLite (one WITH RECURSIVE query)                 Python host (thin I/O shell)
┌───────────────────────────────────────┐
│  REDUCE (pure SQL: sched, llm_step)   │
│  pick runnable task, update state,    │
│  queue I/O requests into `pending`    │
└─────┬─────────────────────────────────┘
      │ when tasks        ▲ next tick
      ▼ need I/O          │ (until halt)
┌─────────────────────────┴─────────────┐  batch  ┌───────────────────────────┐
│  EFFECT (1 UDF call: effect)          ├────────►│  sys(pending)             │
│  merge results into `completions`     │◄────────┤  LLM / tools / MCP / I/O  │
└───────────────────────────────────────┘  JSON   └───────────────────────────┘
```

### Under the hood

At any point in a run, the entire state sits in a single relational row: the *world* $w_t$. Each row produced by the recursive CTE is one *tick*. The kernel (state machine) is a pure transition function

$$
\delta : (w_t,\ c_t) \mapsto (w_{t+1},\ e_{t+1})
$$

where $w_t$ is the state at tick $t$, $c_t$ are the I/O completions available at tick $t$, and $e_{t+1}$ are the effects requested next.

Since SQL cannot execute LLM calls, run external tools, or interact with the user on its own, the single Python function `sys()` acts as a black-box oracle $\mathcal{O}$ that turns requested effects into completions (I/O results, not to be confused with LLM text completions):

$$
\mathcal{O} : e \mapsto c
$$

Because $\delta$ is pure, recording $c$ in an `effects` table is enough to replay, recover, or test any run deterministically without calling an LLM.

In SQLite, this state machine is expressed as one compound `WITH RECURSIVE` query. Each `world` row carries a `phase` field (`sched`, `llm_step`, `effect`, or `halt`), and every `UNION ALL` branch filters on `WHERE phase = ...` so at most one branch fires per tick:

```sql
WITH RECURSIVE world(tick, phase, active_task, tasks, pending, completions, ...) AS (
    -- Tick 0 (anchor): seed initial world and root task from `agents`
    SELECT 0, 'sched', NULL, init_tasks, '[]', '{}', ...
    FROM agents WHERE id = :agent_id

    UNION ALL

    -- 1. REDUCE (sched): check stop/limit conditions, pick next runnable task, or trigger I/O
    SELECT tick + 1, next_phase, picked_task, tasks, pending, completions, ...
    FROM world WHERE phase = 'sched'

    UNION ALL

    -- 2. REDUCE (llm_step): advance active task, queue llm.chat / tool calls into `pending`,
    --    or consume finished results from `completions`
    SELECT tick + 1, 'sched', NULL, updated_tasks, updated_pending, completions, ...
    FROM world WHERE phase = 'llm_step'

    UNION ALL

    -- 3. EFFECT (the only impure branch): execute all queued effects in one sys() batch
    SELECT tick + 1, 'sched', NULL, tasks, '[]', merge(completions, sys(pending)), ...
    FROM world WHERE phase = 'effect'
)
SELECT * FROM world;
```

How the loop runs step by step:

1. **Anchor (`tick = 0`):** Seeds the initial `world` row from the `agents` table with `phase = 'sched'`.
2. **`sched`:** Evaluates stop conditions and picks the next runnable task (`phase = 'llm_step'`). Once no task can advance in pure SQL and `pending` is non-empty, it switches to `phase = 'effect'`.
3. **`llm_step`:** Pure SQL updates task state/messages, appends new effect requests (`llm.chat`, `py.call`, ...) to `pending` or consumes settled results from `completions`, and returns to `phase = 'sched'`.
4. **`effect`:** Calls `sys(pending)` once for the entire batch, merges the returned JSON results into `completions`, clears `pending`, and returns to `phase = 'sched'`.
5. **`halt`:** Once the root task finishes (or hits an error or `max_ticks`), `sched` emits a row with `phase = 'halt'`. Because no `UNION ALL` branch matches `WHERE phase = 'halt'`, recursion stops automatically and the query ends.

With just this loop and `sys()`, almost everything else (parallel sub-agents, routing, retries, budgets, or tool permissions) can be built in pure SQL.

For the full design (task map, composite agents, WAL concurrency, replay, and SQLite details), see [`docs/architecture.md`](docs/architecture.md).

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

- [x] Core architecture baseline ([`docs/architecture.md`](docs/architecture.md))
- [x] **Spike 0:** verify SQLite assumptions (CTE streaming, nested JSON aggregates, WAL writer concurrency, Ctrl-C abort, JSON/JSONB limits)
- [ ] **v0** *(in progress)*: walking skeleton with a single `llm` agent loop, `sys()`, Python tools, and CLI
