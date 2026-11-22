-- Single-query agent loop (v0 heartbeat): every tick is one row in `world`.
--
-- Execution alternates between pure REDUCE ticks (`sched`, `llm_step`) and batched
-- impure EFFECT ticks (`effect`, the sole place `sys()` is invoked). Each recursive
-- `UNION ALL` branch guards on a disjoint `phase` so at most one branch fires per tick,
-- and transitioning to `phase = 'halt'` terminates the CTE cleanly.
--
-- Bound parameters: :run_id (int), :agent_id (text), :prompt (text).
WITH RECURSIVE world(tick, seq, status, phase, active_task, tasks, pending, completions, usage) AS (
    -- Tick 0: recursion anchor. Seed root task t1 in OpenAI message format (omit system turn if instruction is empty)
    SELECT
        0,
        1,
        'running',
        'sched',
        NULL,
        json_object(
            't1', json_object(
                'seq', 1,
                'agent', a.id,
                'status', 'ready',
                'waiting_on', json('[]'),
                'input', json_object('prompt', :prompt),
                'messages', CASE
                    WHEN a.instruction != '' THEN json_array(
                        json_object('role', 'system', 'content', a.instruction),
                        json_object('role', 'user', 'content', :prompt)
                    )
                    ELSE json_array(
                        json_object('role', 'user', 'content', :prompt)
                    )
                END,
                'result', NULL
            )
        ),
        '[]',
        '{}',
        '{"tokens":0}'
    FROM agents AS a
    WHERE a.id = :agent_id

    UNION ALL

    -- REDUCE (sched): evaluate stop conditions first, then dispatch lowest-seq runnable task.
    -- Priority: root task terminal -> max_ticks limit -> runnable task -> flush pending -> stuck.
    SELECT
        w.tick + 1,
        w.seq,
        CASE
            WHEN (w.tasks ->> '$.t1.status') IN ('done', 'failed')
                THEN w.tasks ->> '$.t1.status'
            WHEN w.tick >= COALESCE((SELECT CAST(value AS INTEGER) FROM settings WHERE key = 'max_ticks'), 100)
                THEN 'limit'
            WHEN pick.value IS NOT NULL AND a.kind = 'llm'
                THEN 'running'
            WHEN pick.value IS NOT NULL
                THEN 'failed'
            WHEN json_array_length(w.pending) > 0
                THEN 'running'
            ELSE 'stuck'
        END,
        CASE
            WHEN (w.tasks ->> '$.t1.status') IN ('done', 'failed')
                THEN 'halt'
            WHEN w.tick >= COALESCE((SELECT CAST(value AS INTEGER) FROM settings WHERE key = 'max_ticks'), 100)
                THEN 'halt'
            WHEN pick.value IS NOT NULL AND a.kind = 'llm'
                THEN 'llm_step'
            WHEN pick.value IS NOT NULL
                THEN 'halt'
            WHEN json_array_length(w.pending) > 0
                THEN 'effect'
            ELSE 'halt'
        END,
        CASE
            WHEN (w.tasks ->> '$.t1.status') IN ('done', 'failed')
                THEN NULL
            WHEN w.tick >= COALESCE((SELECT CAST(value AS INTEGER) FROM settings WHERE key = 'max_ticks'), 100)
                THEN NULL
            WHEN pick.value IS NOT NULL AND a.kind = 'llm'
                THEN pick.value
            ELSE NULL
        END,
        w.tasks,
        w.pending,
        w.completions,
        w.usage
    FROM world AS w
    -- SQLite forbids correlated subqueries in FROM, so wrap the scalar pick in json_array()
    -- and unwrap via json_each() to bind `pick.value` once across all SELECT expressions
    -- (json_array(NULL) still yields one row with pick.value IS NULL when nothing is runnable).
    -- Runnable = status is 'ready', or 'waiting' with every waiting_on dep settled.
    JOIN json_each(json_array((
        SELECT t.key
        FROM json_each(w.tasks) AS t
        WHERE (t.value ->> '$.status') = 'ready'
           OR (
               (t.value ->> '$.status') = 'waiting'
               AND NOT EXISTS (
                   SELECT 1
                   FROM json_each(t.value -> '$.waiting_on') AS dep
                   WHERE json_type(w.completions, '$.' || dep.value) IS NULL
                     AND COALESCE(w.tasks ->> ('$.' || dep.value || '.status'), '') NOT IN ('done', 'failed')
               )
           )
        ORDER BY CAST(t.value ->> '$.seq' AS INTEGER) ASC
        LIMIT 1
    ))) AS pick
    LEFT JOIN agents AS a ON a.id = (w.tasks ->> ('$.' || pick.value || '.agent'))
    WHERE w.status = 'running' AND w.phase = 'sched'

    UNION ALL

    -- REDUCE (llm_step, ready): assign monotonic r<run>_e<seq> effect id and enqueue llm.chat
    SELECT
        w.tick + 1,
        w.seq + 1,
        'running',
        'sched',
        NULL,
        json_set(
            w.tasks,
            '$.' || w.active_task || '.status', 'waiting',
            '$.' || w.active_task || '.waiting_on', json_array('r' || :run_id || '_e' || (w.seq + 1))
        ),
        json_insert(
            w.pending,
            '$[#]',
            json_object(
                'id', 'r' || :run_id || '_e' || (w.seq + 1),
                'op', 'llm.chat',
                'args', json_object(
                    'model_id', a.model_id,
                    -- Wrap `tasks -> ...` in json() so SQLite embeds the array rather than
                    -- double-encoding it as a JSON string.
                    'messages', json(w.tasks -> ('$.' || w.active_task || '.messages'))
                )
            )
        ),
        w.completions,
        w.usage
    FROM world AS w
    JOIN agents AS a ON a.id = (w.tasks ->> ('$.' || w.active_task || '.agent'))
    WHERE w.status = 'running'
      AND w.phase = 'llm_step'
      AND (w.tasks ->> ('$.' || w.active_task || '.status')) = 'ready'

    UNION ALL

    -- REDUCE (llm_step, waiting): consume completion, append assistant turn, and settle task
    SELECT
        w.tick + 1,
        w.seq,
        'running',
        'sched',
        NULL,
        CASE
            -- json_type(...) = 'true' checks JSON booleans strictly without SQLite 1/0 coercion.
            WHEN json_type(w.completions, '$.' || eff.value || '.ok') = 'true' THEN
                json_set(
                    w.tasks,
                    '$.' || w.active_task || '.status', 'done',
                    '$.' || w.active_task || '.waiting_on', json('[]'),
                    '$.' || w.active_task || '.messages', json_insert(
                        w.tasks -> ('$.' || w.active_task || '.messages'),
                        '$[#]',
                        json_object(
                            'role', 'assistant',
                            'content', w.completions ->> ('$.' || eff.value || '.value.content')
                        )
                    ),
                    '$.' || w.active_task || '.result', w.completions ->> ('$.' || eff.value || '.value.content')
                )
            ELSE
                json_set(
                    w.tasks,
                    '$.' || w.active_task || '.status', 'failed',
                    '$.' || w.active_task || '.waiting_on', json('[]'),
                    '$.' || w.active_task || '.result', json_object(
                        'error', COALESCE(w.completions ->> ('$.' || eff.value || '.error'), 'unknown error')
                    )
                )
        END,
        w.pending,
        -- Drop the consumed completion so `completions` stays bounded across turns.
        json_remove(w.completions, '$.' || eff.value),
        CASE
            WHEN json_type(w.completions, '$.' || eff.value || '.ok') = 'true' THEN
                json_set(
                    w.usage,
                    '$.tokens',
                    COALESCE(CAST(w.usage ->> '$.tokens' AS INTEGER), 0)
                    + COALESCE(CAST(w.completions ->> ('$.' || eff.value || '.value.usage.total_tokens') AS INTEGER), 0)
                )
            ELSE w.usage
        END
    FROM world AS w
    -- Bind waiting_on[0] once so CASE expressions and json_remove share the same effect id.
    JOIN json_each(json_array(w.tasks ->> ('$.' || w.active_task || '.waiting_on[0]'))) AS eff
    WHERE w.status = 'running'
      AND w.phase = 'llm_step'
      AND (w.tasks ->> ('$.' || w.active_task || '.status')) = 'waiting'

    UNION ALL

    -- EFFECT: execute pending batch once via sys(w.pending), clear pending, and merge results.
    -- Avoid json_patch (RFC 7396 strips `null` keys inside payloads); instead merge via
    -- json_each(json_array(old, new)) preferring src.key = 1, re-tagging containers/booleans
    -- with json() so SQLite preserves nested JSON types instead of stringifying or coercing to 0/1.
    SELECT
        w.tick + 1,
        w.seq,
        'running',
        'sched',
        NULL,
        w.tasks,
        '[]',
        COALESCE((
            SELECT json_group_object(
                m.key,
                CASE m.type
                    WHEN 'object' THEN json(m.value)
                    WHEN 'array'  THEN json(m.value)
                    WHEN 'true'   THEN json('true')
                    WHEN 'false'  THEN json('false')
                    ELSE m.value
                END
            )
            FROM json_each(json_array(json(w.completions), json(batch.value))) AS src
            JOIN json_each(src.value) AS m
            WHERE src.key = 1
               OR json_type(batch.value, '$.' || m.key) IS NULL
        ), '{}'),
        w.usage
    FROM world AS w
    JOIN json_each(json_array(sys(w.pending))) AS batch
    WHERE w.status = 'running' AND w.phase = 'effect'
)
-- Keep outer SELECT plain (no ORDER BY or WHERE) so SQLite streams CTE ticks via CO-ROUTINE.
SELECT tick, seq, status, phase, active_task, tasks, pending, completions, usage
FROM world;
