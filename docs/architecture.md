# Architecture

> Status: draft design

## 1. The idea in one paragraph

The control flow of an agent run is a single SQL statement, namely one `WITH RECURSIVE` query on SQLite. Every decision the agent makes happens inside that query. The only things it hands off are the ones SQL cannot do:

- **I/O**: showing output to the user and asking questions, whatever the interface is
- **Tools that live outside SQL**: Python functions, MCP servers
- **LLM inference**

For those, Python registers exactly one impure function (UDF), `sys()`.
Agents, tools, models and policies are rows in tables. They are defined with `INSERT` statements; the kernel query itself never changes.

**Goal:** a small but usable agent runtime: a handful of agents with real tools, a CLI, and runs you can inspect, replay and fork.
**Non-goals:** hundreds of concurrent agents, high throughput, competing with production frameworks.

---

## 2. Execution model

### 2.1 State and transition

The whole run state is one value, the *world* $w$. Each row of the recursive CTE is one *tick*. The kernel is a pure function

$$
\delta : (w_t,\ c_t) \mapsto (w_{t+1},\ e_{t+1})
$$

where $w_t$ is the state at tick $t$, $c_t$ are the I/O completions (results of `sys()` calls) at tick $t$, and $e_{t+1}$ are the effects requested next. `sys()` acts as the oracle $\mathcal{O}$ that turns effects into completions:

$$
\mathcal{O} : e \mapsto c
$$

Since $\delta$ is pure, a run is fully determined by its initial world and the sequence of completions (not to be confused with LLM text completions) $c_1, \dots, c_n$. Recording those gives replay, forking, crash recovery and LLM-free tests (under the conditions in §2.6).

Physically, $w$ is the current row of the recursive CTE, split across columns rather than one blob: scalars `tick`, `seq`, `status`, `phase`, `active_task` and JSON columns `tasks`, `pending`, `completions`, `usage` (text JSON first for simplicity and debuggability; JSONB on SQLite ≥ 3.45 is a drop-in optimization later if profiling calls for it). Transitions only rewrite the columns they change, and the EFFECT tick calls `sys(pending)` and merges the results into `completions`.

### 2.2 Two kinds of ticks

A recursive CTE row is either a **REDUCE** tick or an **EFFECT** tick.

- **REDUCE** is pure SQL (no impure calls). It advances one runnable task by one step and appends whatever effects that step needs to `pending`. Runnable tasks are picked in a fixed order (lowest `seq`), so the schedule is deterministic.
- **EFFECT** happens once no task can move without I/O. It calls `sys(pending)` exactly once, with *all* pending effects as a batch, merges the returned map into `completions` and clears `pending` (merging rather than overwriting so unconsumed results for tasks still waiting on child agents aren't lost; avoid `json_patch`, since RFC 7396 strips `null` keys inside payloads). Subsequent REDUCE ticks hand each completion to its waiting task and remove consumed entries.

```
REDUCE  task A -> wants LLM call
REDUCE  task B -> wants tool call
REDUCE  nothing runnable
EFFECT  sys([llm(A), tool(B)])   -- Python runs both concurrently
REDUCE  deliver results to A and B ...
```

So the query is strictly single-threaded and deterministic, while the I/O runs in parallel. SQL owns the event loop and Python is just the I/O reactor. If no task is runnable, nothing is pending and the root task is not finished, the run ends with status `stuck` instead of spinning.

**Kernel structure.** SQLite allows the recursive table only once in the `FROM` of each recursive `SELECT`, and helper CTEs or views cannot reference it. To keep the query readable, the recursive part is a compound `UNION ALL` of several `SELECT`s (SQLite ≥ 3.34), one per transition. Rather than repeating complex `WHERE` guards on every branch (where an overlap forks rows exponentially and a gap stops the run early), a single `sched` branch (`WHERE phase = 'sched'`) checks limits and deadlocks, picks the next runnable task into `active_task`, and sets `phase` (`'effect'`, `'llm_step'`, `'call_step'`, ...). Every other branch guards on a single scalar check (`WHERE phase = '...'`). Aggregates live in correlated scalar subqueries over `json_each(...)`. Multi-item work (spawning parallel children, consuming multiple completions) micro-steps one item per tick where practical so updates stay single-key `json_set`/`json_remove` calls.

### 2.3 Tasks instead of a call stack

A call stack can only express one thread of execution, but parallel sub-agents need more than that, so the world holds a map of tasks. One row, shown as JSON:

```json
{
  "tick": 41, "seq": 8, "status": "running", "phase": "sched", "active_task": null,
  "tasks": {
    "t1": {"seq": 1, "agent": "pipeline",   "status": "waiting", "waiting_on": ["t2"], "pc": 1,
           "scope": {"topic": "...", "research": null}},
    "t2": {"seq": 2, "agent": "researcher", "status": "waiting", "waiting_on": ["r9_e7"],
           "parent": "t1", "input": {"topic": "..."}, "result": null, "messages": [], "iter": 2}
  },
  "pending":     [],
  "completions": {},
  "usage":       {"tokens": 5123}
}
```

A task is runnable when it is `ready`, or when it is `waiting` and everything in `waiting_on` has arrived (either finished child tasks or keys in `completions`). A composite agent (sequential, parallel, ...) spawns child tasks and waits on their ids. Nesting, recursion and fan-out all fall out of this without mutual recursion between CTEs. `messages` holds previous turns, tool calls and results in the OpenAI chat format.

**Data flow is scoped, not global.** Children never write shared state; each child writes only its own `result`. When a child finishes, its parent stores that result under the child's `output_key` in its own `scope` (for parallel children: on join, in child order). A child receives a snapshot when it is spawned: its `input` is the parent's `scope` patched with its explicit arguments, so a sequential child sees the results of earlier siblings. Because every composite task has its own scope, recursive or repeated sub-agents never overwrite each other's keys. The root task's scope is the run's state and becomes the run result.

### 2.4 Why there is only one impure call

Python is called in exactly one place in the kernel: the `sys()` call in the EFFECT branch. This has the following consequences:

**Results can be reused.** Once an effect completes, its result stays in `completions` until consumed, so later ticks can read it as often as needed without re-calling `sys()`. When rebuilding JSON objects from `json_each(...)`, wrap nested objects, arrays and booleans via `CASE e.type` + `json(...)` rather than passing raw `e.value`, otherwise SQLite double-encodes objects as strings and turns booleans into `0`/`1`.

**Getting a fresh value** (time, random, a second LLM sample). Every effect carries an id (`r<run>_e<seq>`), and `sys()` is idempotent per id: same id returns the cached result, new id makes a new call. `sys()` also remembers `(op, args)` per id and returns an error completion if the same id reappears with different arguments (catching kernel non-determinism). Any key interpolated into SQLite JSON paths (`effect_id`, `task_id`, `agents.id`, `output_key`) must match `[a-zA-Z0-9_-]+`; untrusted strings from models or tools (like `tool_call.id`) are never used as JSON path keys.

**Pure helpers.** Deterministic functions without I/O (such as `{key}` template rendering, which is awkward and prone to nested-placeholder bugs in SQL) may be registered as `deterministic=True` UDFs. Calling them twice is harmless. Each one is listed here (planned: `render()`), and the goal is to keep that list short.

### 2.5 Failures

- **Effect errors are data** (`"ok": false`). Tool errors go back to the model as a tool result so it can recover. Failed `llm.chat` calls are retried with a new effect id up to `max_retries`, then the task fails.
- **A failed task fails its parent** (fail-fast). Siblings and their descendants are cancelled (propagated via parent/ancestor status so grandchildren don't keep running). In-flight effects still finish and their completions are dropped.
- **Run-level stops** end the run with a status instead of an error: `stuck` (deadlock, see §2.2) and `limit` (`max_ticks` or a budget, see §8).
- **`sys()` never raises.** Anything unexpected becomes an error completion, because an exception escaping a UDF aborts the whole statement.

### 2.6 Journal, replay and recovery

- **Journal.** Because $\delta$ is deterministic, the initial world plus the `effects` log (every completed `sys()` call) are enough to reconstruct a run. `trace` separately records ticks and periodic world checkpoints depending on `trace_level`.
- **Replay** runs the kernel with `sys()` served from the journal. It requires the same definitions: `start.sql` stores a JSON snapshot of the rows a run uses in `runs`, and replay refuses to run if they changed. Exact replay also assumes the same SQLite version, since JSON formatting can differ between versions.
- **Crash recovery** restarts from the last checkpoint and feeds the journaled completions back in. Effects that were in flight during the crash run again (at-least-once), so tools with external side effects should be idempotent.
- **Fork** starts a new run from any checkpoint, for example with a different human answer.

---

## 3. The syscall interface

`sys()` takes a JSON array of effects and returns completions keyed by id.

```json
[{"id": "r9_e7", "op": "llm.chat", "args": {}},
 {"id": "r9_e8", "op": "py.call",  "args": {"name": "calculator", "arguments": {}}}]
```
```json
{"r9_e7": {"ok": true,  "value": {}, "meta": {"ms": 812}},
 "r9_e8": {"ok": false, "error": "timeout"}}
```

Rules:

- Handlers never raise. Transport retries (HTTP 429/503 with backoff) stay inside the handler; unrecovered errors come back as data, and SQL decides on logical retries, fallbacks or aborting.
- SQL resolves everything before emitting an effect (which tool server, which model, which arguments). Python only maps `op -> handler`, so there is no routing in Python.
- Every effect has a timeout (default in `settings`, overridable per effect). A timeout is an error completion.
- Version 1 waits for the whole batch, so parallel tasks advance in lockstep and one slow call holds back the others. Later, `sys()` may return only the completions that are already done, in an io_uring style. Tasks already wait on ids, so the kernel does not change.

Initial ops: `llm.chat`, `py.call`, `mcp.call`, `sql.query`, `human.ask`, `io.emit`. Wall-time budgets read timestamps and durations from `meta`. Later: `embed`, `tools.list`.

Handlers run on an asyncio loop in a background thread; synchronous Python tools run in a thread pool so they don't block that loop. `sys()` itself runs on the SQLite thread and blocks further SQL processing until the batch is done.

---

## 4. Data model

**Definitions** (the "program"):

| Table | Purpose |
|---|---|
| `settings` | key/value: `max_ticks`, `trace_level`, default model, timeouts, limits |
| `models` | provider, model name, base URL, `api_key_env` (the *name* of an env var; secrets never go into the DB) |
| `tool_servers` | where tools come from: `python`, `mcp`, `sql` |
| `tools` | tool catalog, one row per tool: name, description, JSON schema, optional SQL body |
| `agents` | `id`, `kind`, `model_id`, `instruction`, `output_key`, `description`, `input_schema`, `config` (JSON) |
| `agent_children` | `(parent, position, child, condition)` for composite agents |
| `agent_tools` | `(agent, server, tool_name)` bindings, where `tool_name = NULL` means every tool of that server |

A `definition_errors` view catches invalid definitions (unknown models, invalid identifier characters, duplicate `output_key`s among parallel children, ...), and runs refuse to start while it is non-empty.

**Runtime:**

| Table | Purpose |
|---|---|
| `runs` | one row per run: agent, status, optional `session_id`, input, result, usage, definitions snapshot |
| `effects` | one row per executed effect (`effect_id`, `op`, `args`, `completion`): durable journal written by `sys()` as each effect completes; powers idempotency, crash recovery and replay |
| `trace` | one row per tick (`run_id`, `tick`, periodic world checkpoints per `trace_level`), written by the host cursor for observability and resume/fork |

Observability is a set of views over `trace` (timeline, spans, token usage, cost). Whether it is on, and how much it records, is a row in `settings`.

**Connections.** Because the kernel holds a read transaction for the whole run, the DB uses WAL mode and all writes (`effects` and `trace`) go through a single serialized writer connection while the kernel cursor is open. That gives the kernel snapshot isolation against definition edits mid-run. The trade-off is that the WAL cannot checkpoint past an active reader, so it grows during a run and shrinks once the run finishes (if long human waits make it large, the journal can move to a separate DB file). WAL requires a local filesystem; on WSL, keep databases on the Linux side rather than `/mnt/c`.

---

## 5. Agents, tools and data flow

### 5.1 Agent kinds

Agents are built from a small set of building blocks, borrowed from Google ADK:

| Kind | What it does |
|---|---|
| `llm` | ReAct loop: call the model, run the tool calls it asks for, repeat until it answers |
| `call` | **deterministic step**: call one tool with arguments taken from its input, store the result |
| `human` | ask the user, store the answer |
| `sequential` | run the children one after another |
| `parallel` | run all children at once and join |
| `loop` | repeat the children until a condition holds or `max_iterations` is reached |
| `router` | pick one child based on a condition over the scope |
| `map` | run one child per element of an array and collect the results |

Any agent can also be used as a tool by an `llm` agent (using its `description` and optional `input_schema`). The tool arguments become the child's input, and its result goes back to the model as the tool result. When the model calls three sub-agents in one turn, three tasks run in parallel. So 0, 1 or *n* sub-agents can come either from the model (dynamic) or from `parallel` / `map` (static). Internally, the 8 kinds lower to 4 kernel engines so `kernel.sql` stays compact: `llm`, `call` (also handles `human` via `human.ask`), `step` (`sequential`/`router`/`loop` via `pc`, `iter`, `condition`, `max_iterations` and all vs first-match), and `fanout` (`parallel`/`map` sharing one-by-one spawning and join).

Free-form graphs (arbitrary nodes and edges, LangGraph style) are not in scope for now. They could later become another kind, `graph`, with its own edges table.

### 5.2 Passing data between steps

Each step's result is stored under its `output_key` in the enclosing composite's scope (§2.3); `map` collects its results into an array under that key. There are two ways to read it:

- **Instructions** use placeholders rendered via `render()`: `"Summarize this: {research}"`.
- **Tool arguments** in `call` steps resolve top-level JSON-path references via `json_extract`: `{"text": {"$ref": "$.research"}, "lang": "en"}` (top-level only, to avoid recursive JSON tree rewriting in SQL).

A typical deterministic chain: the LLM `drafter` writes `draft`, then the `call` step runs `spellcheck(text={"$ref": "$.draft"})` and writes `checked`, then the LLM `editor` reads `{checked}`. The tool can be plain Python. The user writes the function and adds two rows; the kernel is not touched.

Conditions (router, loop exit) are data as well: `(path, op, value)`, for example `("$.review.verdict", "=", "approved")`. They can be evaluated in pure SQL.

### 5.3 Where tools live

Tools are always *declared* in SQL (rows in `tools`). Their *implementation* can live in one of four places:

| Server kind | Implementation | Executed by |
|---|---|---|
| `python` | a function decorated with `@tool`; the schema comes from type hints | `py.call` |
| `mcp` | an external MCP server registered by command or URL | `mcp.call` (the host is the MCP client) |
| `sql` | an SQL query stored in `tools.body` | `sql.query` |
| agent | any row in `agents` | the kernel itself, as a child task |

`sql.query` exists because SQLite has no `eval`: the kernel cannot execute SQL text stored in a table. The handler runs that text on a **separate read-only connection** (`mode=ro`, `PRAGMA query_only`). No impure UDFs are registered on that connection, `sys()` in particular. A `sqlite3.set_authorizer()` callback restricts it to allowlisted tables and views and denies `ATTACH` and `PRAGMA`, with row/byte limits and a `set_progress_handler` timeout. The handler binds the tool arguments as named parameters and returns the rows as JSON. This lets anyone write tools in SQL without touching the kernel. It is *not* a way to run the kernel recursively.

MCP discovery can be written in SQL as a separate statement outside the kernel, because `sys()` is just a function. `tools.list` uses a fresh discovery id and is not part of any run's journal:

```sql
INSERT INTO tools (server_id, name, description, input_schema)
SELECT s.id, t.value->>'name', t.value->>'description', t.value->'inputSchema'
FROM tool_servers s,
     json_each(sys(json_array(json_object('id', 'discover_'||s.id, 'op', 'tools.list',
 'args', json_object('server', s.id))))
               -> ('$.discover_'||s.id||'.value')) t
WHERE s.kind = 'mcp';
```

---

## 6. Outputs, humans and long waits

Anything the user should see goes through `io.emit` (turn-based, not token-by-token streaming), and questions go through `human.ask`. Both are ordinary effects, ordered and recorded like everything else. `sys()` writes each completion to `effects` as it arrives (crash recovery), while the host streams ticks from the `kernel.sql` cursor and writes `trace` rows on the same serialized writer connection (§4).

`human.ask` has two modes:

- **blocking** (CLI): the handler waits for input.
- **suspending** (later, for web or chat): the handler returns `pending`, and the kernel finishes with `status = 'suspended'`. Resuming starts a new kernel statement (a new run segment), anchored on the last checkpoint plus the answer. A run can wait for days without holding a transaction open.

A run segment consists of three statements, all of them SQL: `start.sql` (create the run row and snapshot the definitions it uses), `kernel.sql` (the one recursive query), and `finish.sql` (write the final result into `runs`). Multi-turn sessions use `runs.session_id`, with `start.sql` seeding messages from the previous finished run.

---

## 7. Embeddings and retrieval (after v1)

Embedding is never part of the kernel's logic. From the kernel's point of view it is an effect (`embed`) like any other, and the backend behind it is chosen in `settings`:

| Backend | How | Speed |
|---|---|---|
| `openai` | API call | fast |
| `sql-static` | static token embeddings (Model2Vec style): tokenize, look up, mean-pool, normalize, all in SQL | fast enough for local use |
| `sql-transformer` | a small encoder (MiniLM class) whose forward pass is written in SQL; an experiment, see [ideas.md](ideas.md) | slow; offline only |

The `sql-*` backends are separate statements (`embed.sql`) that the `embed` handler runs on its own connection, so there is SQL behind the syscall too. They stay out of the kernel, because a forward pass inside a correlated subquery would make the kernel unreadable for no gain.

**Indexing** is a separate job (`agent-sql index`) that runs before any agent: load documents, chunk them, embed the chunks, store the vectors. Python only reads files (documents and safetensors weights) and inserts them as rows; all the math is SQL.
**Retrieval** at run time is an ordinary SQL tool. Cosine similarity over vectors stored as `(doc, dim, val)` rows is a join and a `SUM`, which is fine for up to a few thousand chunks (≤ 384 dims). Larger corpora use packed float32 BLOBs (`sqlite-vec` or a small dot-product UDF).

---

## 8. Performance and staying alive

The LLM dominates wall time: model calls take seconds, while a REDUCE tick takes milliseconds. Roughly, if a tick touches the world $k$ times, a tick costs $O(k\,|w|)$ and a run costs $O(T\,k\,|w|)$. With $|w| \approx 100$ KB, $k \approx 50$ and $T \approx 1000$ ticks, that is only a few seconds of CPU, and if profiling ever flags JSON parsing on large worlds, switching CTE columns to JSONB (SQLite ≥ 3.45) is a drop-in change. Long tool outputs are truncated in SQL before entering `messages` (limit in `settings`).

The main trap is large fan-out: with $n$ tasks, roughly $n$ ticks copy a world that grows with $n$, which is $O(n^2)$ total. Fine for 50 sub-agents, bad for 5000. Large fan-out is a non-goal (§1); an escape hatch is sketched in [ideas.md](ideas.md).

Guards that keep the process from falling over:

- `max_ticks` in the recursive `WHERE`, so a runaway loop ends cleanly with status `limit` instead of running forever.
- Budgets (tokens, cost, wall time) checked in REDUCE.
- A hard cap on the world size. If it is exceeded, the run fails with a clear error.
- A turn-aligned message window per task (drops old turns without splitting an assistant `tool_calls` message from its `tool` results).
- `PRAGMA temp_store = FILE`, so anything SQLite materializes spills to disk instead of RAM.
- `conn.interrupt()` on Ctrl-C, plus cancelling active asyncio handler tasks and any blocking `human.ask` (since `conn.interrupt()` does not abort an in-flight Python UDF).

The recursive CTE itself is iterative, so there is no stack to blow. Since every tick produces exactly one row, its queue holds a single row at a time.

---

## 9. Architecture layers

```
host/        Python, knows nothing about agents:
             runner (runs statements, streams rows into trace),
             sys dispatcher, handler registry, replay source, CLI
kernel/      the SQL: schema, kernel.sql, start/finish, views
handlers/    llm (OpenAI-compatible), py tools, mcp, sql.query, human, io
embed/       (later) embed.sql backends, index job
```

Ways to extend the system, from most to least common:

| You want ... | You do ... |
|---|---|
| a new agent / workflow | `INSERT` rows |
| a new Python tool | write a function and add `@tool` |
| a new MCP server | `INSERT` into `tool_servers` |
| an SQL tool | `INSERT` into `tools` with a body |
| a new LLM provider | add a handler (or use an OpenAI-compatible endpoint) |
| a new kind of I/O | add an op and a handler |
| a new agent kind | add one transition branch to the kernel (framework work, not user work) |

**Testing.** Kernel transitions are tested without an LLM: seed the anchor with a fixture world, serve `sys()` from a scripted journal, and compare the resulting trace with a golden file.

**Visualization.** Every view (CLI now, TUI or web later) is a read-only consumer of `trace` and `runs`. Readers use WAL, so they never block a run. The CLI prints the `io.emit` stream live and can render a run's timeline afterwards.

---

## 10. Assumptions to verify (Spike 0)

| Assumption | If it does not hold |
|---|---|
| Rows of a compound (`UNION ALL`) recursive CTE stream to the cursor as they are produced | Live output still works (`io.emit` prints from the handler) and `sys()` still persists completions; only `trace` is written at the end. Alternatively, split runs into segments at EFFECT boundaries. |
| `WHERE phase = 'effect'` prevents `sys()` from running on non-matching ticks, and `sys()` runs once per EFFECT row | Repeats are harmless thanks to the id cache; they only cost time. |
| Correlated scalar subqueries with aggregates (`json_group_object`, ...) work inside the recursive SELECT | Move the affected logic into a listed pure helper UDF (§2.4). |
| Compound recursive SELECTs (≥ 3.34) work with JSON columns | One recursive SELECT with a `CASE` per column. |
| `json_each` rebuilds preserve nested objects/booleans/`null`s, and `json_set`/`json_extract` stay fast for 100 KB–few MB worlds | Move affected transforms to a pure helper UDF (§2.4), or switch CTE columns to JSONB (SQLite ≥ 3.45) with smaller message windows. |
| A second connection can commit to `trace` while the kernel cursor is open (WAL) | Journal in a separate database file. |
| Ctrl-C ends a run cleanly, even during a blocking batch | Run handlers with hard timeouts and check a cancel flag between batches. |
| SQLite ≥ 3.38 (compound recursive CTEs + `->`/`->>` JSON operators) is available on all targets (x86_64 and arm64 Linux, including WSL) | Use a `uv`-managed Python with a bundled SQLite, or `apsw`. |

---

## 11. Staged delivery

**v0, walking skeleton.** It should be runnable from the first day: host, `sys()`, `llm.chat` (OpenAI-compatible), `py.call`, `io.emit` and a blocking `human.ask`, one kind (`llm`) with a tool loop, `trace`, and a CLI that runs an agent and prints what happens. The world already uses the task map, even with a single task, so the full version can grow from here instead of replacing it.

**v1, usable.** In this order: composite kinds and parallel batches, `call` steps and scoped data flow, replay and fork, MCP and SQL tools, budgets and limits, suspend/resume.

After that: embeddings (§7). Further ideas live in [ideas.md](ideas.md) and are not commitments.

---

## 12. Prior art

- **Elm architecture / redux-loop**: `update` returns a new state plus commands; effects are data executed by the runtime. REDUCE/EFFECT is the same split.
- **Temporal, Restate**: durable execution through deterministic replay of a journal of side-effect results (§2.6).
- **LangGraph checkpointing**: resuming and forking runs from saved steps.
- **Google ADK**: the agent kinds, `instruction` templating and `output_key` (§5).
- **SQLite's own recursive CTE examples** (Mandelbrot, Sudoku solver): proof that the dialect carries real computation.
