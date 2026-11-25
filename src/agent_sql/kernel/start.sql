-- Create a run row and snapshot the agent + settings definitions used by the run.
-- Inserts and returns 0 rows when :agent_id does not exist.
-- Bound parameters: :agent_id (text), :input (JSON object text), :session_id (text or NULL).
INSERT INTO runs (agent_id, session_id, status, input, definitions)
SELECT
    a.id,
    :session_id,
    'running',
    json(:input),
    json_object(
        'agent', json_object(
            'id', a.id,
            'kind', a.kind,
            'model_id', a.model_id,
            'instruction', a.instruction,
            'output_key', a.output_key,
            'description', a.description,
            'input_schema', CASE
                WHEN a.input_schema IS NOT NULL THEN json(a.input_schema)
            END,
            'config', json(a.config)
        ),
        -- Re-wrap COALESCE in json() because COALESCE drops the JSON subtype from json_group_object().
        'settings', json(COALESCE((SELECT json_group_object(s.key, s.value) FROM settings AS s), '{}')),
        'tools', json(COALESCE((
            SELECT json_group_array(
                json_object(
                    'server_id', bt.server_id,
                    'server_kind', bt.server_kind,
                    'name', bt.name,
                    'description', bt.description,
                    'input_schema', json(bt.input_schema)
                )
                ORDER BY bt.name ASC, bt.server_id ASC
            )
            FROM (
                SELECT
                    t.server_id,
                    ts.kind AS server_kind,
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
                JOIN tool_servers AS ts ON ts.id = t.server_id
                WHERE at.agent_id = a.id
            ) AS bt
            WHERE bt.rn = 1
        ), '[]'))
    )
FROM agents AS a
WHERE a.id = :agent_id
RETURNING id;
