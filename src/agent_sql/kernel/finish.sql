-- Finalize a run row after the kernel cursor closes.
-- Bound parameters: :run_id (int), :status (text), :result (JSON text or NULL), :usage (JSON text).
UPDATE runs
SET
    status      = :status,
    result      = CASE WHEN :result IS NOT NULL THEN json(:result) END,
    usage       = json(:usage),
    finished_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
WHERE id = :run_id;
