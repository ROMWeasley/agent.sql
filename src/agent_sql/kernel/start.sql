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
        'settings', json(COALESCE((SELECT json_group_object(s.key, s.value) FROM settings AS s), '{}'))
    )
FROM agents AS a
WHERE a.id = :agent_id
RETURNING id;
