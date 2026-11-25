-- Single-query agent loop (v0 ReAct + tools): every tick is one row in `world`.
--
-- Execution alternates between pure REDUCE ticks (`sched`, `llm_step`) and batched
-- impure EFFECT ticks (`effect`, the sole place `sys()` is invoked). Each recursive
-- `UNION ALL` branch guards on a disjoint `(phase, mode, status)` so at most one branch
-- fires per tick, and transitioning to `phase = 'halt'` terminates the CTE cleanly.
--
-- Bound parameters: :run_id (int), :agent_id (text), :input (JSON object text).
WITH RECURSIVE world(tick, seq, status, phase, active_task, tasks, pending, completions, usage) AS (
    -- Tick 0: recursion anchor. Seed root task t1 from structured :input (prompt is optional).
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
                'input', json(:input),
                'messages', CASE
                    WHEN a.instruction != ''
                     AND json_type(:input, '$.prompt') = 'text'
                     AND (:input ->> '$.prompt') != '' THEN json_array(
                        json_object('role', 'system', 'content', a.instruction),
                        json_object('role', 'user', 'content', :input ->> '$.prompt')
                    )
                    WHEN a.instruction != '' THEN json_array(
                        json_object('role', 'system', 'content', a.instruction)
                    )
                    WHEN json_type(:input, '$.prompt') = 'text'
                     AND (:input ->> '$.prompt') != '' THEN json_array(
                        json_object('role', 'user', 'content', :input ->> '$.prompt')
                    )
                    ELSE json('[]')
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
        decision.value ->> '$.status',
        decision.value ->> '$.phase',
        decision.value ->> '$.active',
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
    JOIN json_each(json_array(
        CASE
            WHEN (w.tasks ->> '$.t1.status') IN ('done', 'failed')
                THEN json_object('status', w.tasks ->> '$.t1.status', 'phase', 'halt', 'active', NULL)
            WHEN w.tick >= COALESCE((SELECT CAST(value AS INTEGER) FROM settings WHERE key = 'max_ticks'), 100)
                THEN json_object('status', 'limit', 'phase', 'halt', 'active', NULL)
            WHEN pick.value IS NOT NULL AND a.kind = 'llm'
                THEN json_object('status', 'running', 'phase', 'llm_step', 'active', pick.value)
            WHEN pick.value IS NOT NULL
                THEN json_object('status', 'failed', 'phase', 'halt', 'active', NULL)
            WHEN json_array_length(w.pending) > 0
                THEN json_object('status', 'running', 'phase', 'effect', 'active', NULL)
            ELSE json_object('status', 'stuck', 'phase', 'halt', 'active', NULL)
        END
    )) AS decision
    WHERE w.status = 'running' AND w.phase = 'sched'

    UNION ALL

    -- REDUCE (llm_step, chat ready): assign monotonic r<run>_e<seq> effect id and enqueue llm.chat.
    -- Includes OpenAI `tools` only when the agent has at least one bound tool.
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
                'args', CASE
                    WHEN json_array_length(bound_tools.value) > 0 THEN json_object(
                        'model_id', a.model_id,
                        -- Wrap `tasks -> ...` in json() so SQLite embeds the array rather than
                        -- double-encoding it as a JSON string.
                        'messages', json(w.tasks -> ('$.' || w.active_task || '.messages')),
                        'tools', json(bound_tools.value)
                    )
                    ELSE json_object(
                        'model_id', a.model_id,
                        'messages', json(w.tasks -> ('$.' || w.active_task || '.messages'))
                    )
                END
            )
        ),
        w.completions,
        w.usage
    FROM world AS w
    JOIN agents AS a ON a.id = (w.tasks ->> ('$.' || w.active_task || '.agent'))
    JOIN json_each(json_array(COALESCE((
        SELECT json_group_array(
            json_object(
                'type', 'function',
                'function', json_object(
                    'name', bt.name,
                    'description', bt.description,
                    'parameters', json(bt.input_schema)
                )
            )
            ORDER BY bt.name ASC
        )
        FROM (
            SELECT
                t.name,
                t.description,
                t.input_schema,
                ROW_NUMBER() OVER (
                    PARTITION BY t.name
                    ORDER BY CASE WHEN at.tool_name IS NOT NULL THEN 0 ELSE 1 END, t.server_id ASC
                ) AS rn
            FROM agent_tools AS at
            JOIN tools AS t
              ON t.server_id = at.server_id
             AND (at.tool_name IS NULL OR at.tool_name = t.name)
            WHERE at.agent_id = a.id
        ) AS bt
        WHERE bt.rn = 1
    ), '[]'))) AS bound_tools
    WHERE w.status = 'running'
      AND w.phase = 'llm_step'
      AND (w.tasks ->> ('$.' || w.active_task || '.status')) = 'ready'
      AND COALESCE(w.tasks ->> ('$.' || w.active_task || '.mode'), 'chat') = 'chat'

    UNION ALL

    -- REDUCE (llm_step, chat waiting): consume llm.chat completion.
    -- Final text settles the task as `done`; `tool_calls` enters `tool_dispatch` mode.
    SELECT
        w.tick + 1,
        w.seq,
        'running',
        'sched',
        NULL,
        CASE
            -- json_type(...) = 'true' checks JSON booleans strictly without SQLite 1/0 coercion.
            WHEN json_type(w.completions, '$.' || eff.value || '.ok') != 'true' THEN
                json_set(
                    w.tasks,
                    '$.' || w.active_task || '.status', 'failed',
                    '$.' || w.active_task || '.waiting_on', json('[]'),
                    '$.' || w.active_task || '.result', json_object(
                        'error', COALESCE(w.completions ->> ('$.' || eff.value || '.error'), 'unknown error')
                    )
                )
            WHEN json_array_length(norm_calls.value) > 0 THEN
                json_set(
                    w.tasks,
                    '$.' || w.active_task || '.status', 'ready',
                    '$.' || w.active_task || '.mode', 'tool_dispatch',
                    '$.' || w.active_task || '.waiting_on', json('[]'),
                    '$.' || w.active_task || '.tool_queue', json(norm_calls.value),
                    '$.' || w.active_task || '.inflight_calls', json('[]'),
                    '$.' || w.active_task || '.messages', json_insert(
                        w.tasks -> ('$.' || w.active_task || '.messages'),
                        '$[#]',
                        json_object(
                            'role', 'assistant',
                            'content', w.completions ->> ('$.' || eff.value || '.value.content'),
                            'tool_calls', json(norm_calls.value)
                        )
                    )
                )
            ELSE
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
    -- Bind waiting_on[0] and normalized tool_calls once across CASE branches and json_remove.
    JOIN json_each(json_array(w.tasks ->> ('$.' || w.active_task || '.waiting_on[0]'))) AS eff
    JOIN json_each(json_array(
        CASE
            WHEN json_type(w.completions, '$.' || eff.value || '.ok') = 'true'
             AND json_type(w.completions, '$.' || eff.value || '.value.tool_calls') = 'array'
             AND json_array_length(w.completions -> ('$.' || eff.value || '.value.tool_calls')) > 0
            THEN (
                SELECT json_group_array(
                    json_object(
                        'id', CASE
                            WHEN json_type(tc.value, '$.id') = 'text' AND (tc.value ->> '$.id') != ''
                                THEN tc.value ->> '$.id'
                            ELSE 'call_' || w.tick || '_' || tc.key
                        END,
                        'type', 'function',
                        'function', json_object(
                            'name', COALESCE(tc.value ->> '$.function.name', ''),
                            -- Strip JSON subtype with `'' ||` so `arguments` stays a JSON string.
                            'arguments', CASE
                                WHEN json_type(tc.value, '$.function.arguments') = 'text'
                                    THEN tc.value ->> '$.function.arguments'
                                WHEN json_type(tc.value, '$.function.arguments') IS NOT NULL
                                    THEN '' || (tc.value -> '$.function.arguments')
                                ELSE '{}'
                            END
                        )
                    )
                    ORDER BY tc.key ASC
                )
                FROM json_each(w.completions -> ('$.' || eff.value || '.value.tool_calls')) AS tc
            )
            ELSE '[]'
        END
    )) AS norm_calls
    WHERE w.status = 'running'
      AND w.phase = 'llm_step'
      AND (w.tasks ->> ('$.' || w.active_task || '.status')) = 'waiting'
      AND COALESCE(w.tasks ->> ('$.' || w.active_task || '.mode'), 'chat') = 'chat'

    UNION ALL

    -- REDUCE (llm_step, tool_dispatch): micro-step tool_queue[0] into pending or local error.
    -- Untrusted model fields (`id`, `name`, `arguments`) are never interpolated into JSON paths.
    SELECT
        w.tick + 1,
        CASE WHEN eff_id.value IS NOT NULL THEN w.seq + 1 ELSE w.seq END,
        'running',
        CASE WHEN more.value = 1 THEN 'llm_step' ELSE 'sched' END,
        CASE WHEN more.value = 1 THEN w.active_task ELSE NULL END,
        json_remove(
            json_set(
                w.tasks,
                '$.' || w.active_task || '.status', CASE WHEN more.value = 1 THEN 'ready' ELSE 'waiting' END,
                '$.' || w.active_task || '.mode', CASE WHEN more.value = 1 THEN 'tool_dispatch' ELSE 'tool_collect' END,
                '$.' || w.active_task || '.waiting_on', json(
                    CASE
                        WHEN eff_id.value IS NOT NULL THEN json_insert(
                            w.tasks -> ('$.' || w.active_task || '.waiting_on'),
                            '$[#]',
                            eff_id.value
                        )
                        ELSE w.tasks -> ('$.' || w.active_task || '.waiting_on')
                    END
                ),
                '$.' || w.active_task || '.inflight_calls', json_insert(
                    w.tasks -> ('$.' || w.active_task || '.inflight_calls'),
                    '$[#]',
                    json_object(
                        'call_id', tc.value ->> '$.id',
                        'dep_id', eff_id.value,
                        'error', CASE
                            WHEN eff_id.value IS NOT NULL THEN NULL
                            WHEN resolved.value IS NULL
                                THEN 'Tool is not bound to this agent: ' || COALESCE(tc.value ->> '$.function.name', '')
                            WHEN (resolved.value ->> '$.server_kind') != 'python'
                                THEN 'Unsupported tool server kind: ' || (resolved.value ->> '$.server_kind')
                            ELSE 'Tool arguments must be a JSON object'
                        END
                    )
                )
            ),
            CASE
                WHEN more.value = 1 THEN '$.' || w.active_task || '.tool_queue[0]'
                ELSE '$.' || w.active_task || '.tool_queue'
            END
        ),
        CASE
            WHEN eff_id.value IS NOT NULL THEN
                json_insert(
                    w.pending,
                    '$[#]',
                    json_object(
                        'id', eff_id.value,
                        'op', 'py.call',
                        'args', json_object(
                            'server', resolved.value ->> '$.server_id',
                            'name', resolved.value ->> '$.name',
                            'arguments', json(
                                CASE
                                    WHEN TRIM(tc.value ->> '$.function.arguments') = '' THEN '{}'
                                    ELSE tc.value ->> '$.function.arguments'
                                END
                            )
                        )
                    )
                )
            ELSE w.pending
        END,
        w.completions,
        w.usage
    FROM world AS w
    JOIN json_each(json_array(w.tasks -> ('$.' || w.active_task || '.tool_queue[0]'))) AS tc
    JOIN json_each(json_array((
        SELECT json_object(
            'server_id', ts.id,
            'server_kind', ts.kind,
            'name', t.name
        )
        FROM agent_tools AS at
        JOIN tools AS t
          ON t.server_id = at.server_id
         AND (at.tool_name IS NULL OR at.tool_name = t.name)
        JOIN tool_servers AS ts ON ts.id = t.server_id
        WHERE at.agent_id = (w.tasks ->> ('$.' || w.active_task || '.agent'))
          AND t.name = (tc.value ->> '$.function.name')
        ORDER BY CASE WHEN at.tool_name IS NOT NULL THEN 0 ELSE 1 END, ts.id ASC
        LIMIT 1
    ))) AS resolved
    JOIN json_each(json_array(json_array_length(w.tasks -> ('$.' || w.active_task || '.tool_queue')) > 1)) AS more
    JOIN json_each(json_array(
        CASE
            WHEN resolved.value IS NOT NULL
             AND (resolved.value ->> '$.server_kind') = 'python'
             AND (
                 TRIM(tc.value ->> '$.function.arguments') = ''
                 OR (
                     json_valid(tc.value ->> '$.function.arguments') = 1
                     AND json_type(tc.value ->> '$.function.arguments') = 'object'
                 )
             )
            THEN 'r' || :run_id || '_e' || (w.seq + 1)
        END
    )) AS eff_id
    WHERE w.status = 'running'
      AND w.phase = 'llm_step'
      AND (w.tasks ->> ('$.' || w.active_task || '.mode')) = 'tool_dispatch'

    UNION ALL

    -- REDUCE (llm_step, tool_collect): micro-step inflight_calls[0] into a `tool` message.
    -- Tool errors (SQL-rejected or handler `ok=false`) become tool messages so the LLM can recover.
    SELECT
        w.tick + 1,
        w.seq,
        'running',
        CASE WHEN more.value = 1 THEN 'llm_step' ELSE 'sched' END,
        CASE WHEN more.value = 1 THEN w.active_task ELSE NULL END,
        CASE
            WHEN more.value = 1 THEN
                json_remove(
                    json_set(
                        w.tasks,
                        '$.' || w.active_task || '.messages', json_insert(
                            w.tasks -> ('$.' || w.active_task || '.messages'),
                            '$[#]',
                            json(tool_msg.value)
                        )
                    ),
                    '$.' || w.active_task || '.inflight_calls[0]'
                )
            ELSE
                json_remove(
                    json_set(
                        w.tasks,
                        '$.' || w.active_task || '.status', 'ready',
                        '$.' || w.active_task || '.waiting_on', json('[]'),
                        '$.' || w.active_task || '.messages', json_insert(
                            w.tasks -> ('$.' || w.active_task || '.messages'),
                            '$[#]',
                            json(tool_msg.value)
                        )
                    ),
                    '$.' || w.active_task || '.mode',
                    '$.' || w.active_task || '.inflight_calls'
                )
        END,
        w.pending,
        CASE
            WHEN (item.value ->> '$.dep_id') IS NOT NULL
                THEN json_remove(w.completions, '$.' || (item.value ->> '$.dep_id'))
            ELSE w.completions
        END,
        w.usage
    FROM world AS w
    JOIN json_each(json_array(w.tasks -> ('$.' || w.active_task || '.inflight_calls[0]'))) AS item
    JOIN json_each(json_array(json_array_length(w.tasks -> ('$.' || w.active_task || '.inflight_calls')) > 1)) AS more
    JOIN json_each(json_array(
        json_object(
            'role', 'tool',
            'tool_call_id', item.value ->> '$.call_id',
            'content', CASE
                WHEN (item.value ->> '$.dep_id') IS NULL THEN
                    '' || json_object('error', COALESCE(item.value ->> '$.error', 'tool call rejected'))
                WHEN json_type(w.completions, '$.' || (item.value ->> '$.dep_id') || '.ok') = 'true' THEN
                    CASE
                        WHEN json_type(w.completions, '$.' || (item.value ->> '$.dep_id') || '.value') = 'text'
                            THEN w.completions ->> ('$.' || (item.value ->> '$.dep_id') || '.value')
                        ELSE '' || COALESCE(w.completions -> ('$.' || (item.value ->> '$.dep_id') || '.value'), 'null')
                    END
                ELSE
                    '' || json_object(
                        'error',
                        COALESCE(w.completions ->> ('$.' || (item.value ->> '$.dep_id') || '.error'), 'unknown tool error')
                    )
            END
        )
    )) AS tool_msg
    WHERE w.status = 'running'
      AND w.phase = 'llm_step'
      AND (w.tasks ->> ('$.' || w.active_task || '.mode')) = 'tool_collect'

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
