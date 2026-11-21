# Ideas

Things that are fun or might be useful later. None of this is planned work; the committed design is in [architecture.md](architecture.md).

## A transformer forward pass in SQL

The `sql-transformer` embedding backend (architecture §7). Matrices are relations `(i, j, v)`, and matrix multiplication is a join plus an aggregate:

$$
(AB)_{ij} = \sum_k A_{ik} B_{kj}
$$

```sql
SELECT a.i, b.j, SUM(a.v * b.v) AS v
FROM A a
JOIN B b
    ON a.j = b.i
GROUP BY a.i, b.j;
```

LayerNorm is a `GROUP BY` with `AVG` and a variance. Softmax needs `exp`, and GELU needs `tanh`. SQLite has no `erf`, so the `tanh` approximation is used; this requires math functions (`-DSQLITE_ENABLE_MATH_FUNCTIONS`) or short series. The tokenizer (WordPiece, greedy longest match) is a recursive CTE over the vocabulary table. Floats are 64-bit, so results should agree with the reference implementation to about $10^{-6}$. That makes a nice test: cosine $> 0.9999$ against `sentence-transformers`.

**Cost estimate.** A MiniLM-L6 layer has about $1.8 \cdot 10^6$ weights outside the embedding table, which means roughly that many multiply-adds per token per layer. For a 64-token chunk:

$$
64 \times 6 \times 1.8\cdot 10^6 \approx 7 \cdot 10^8 \ \text{joined rows}
$$

CTE intermediates across 48+ matmul `GROUP BY`s are unindexed, so this probably takes 10 to 30+ minutes per chunk on one laptop core. That makes it strictly an offline curiosity; the static backend covers practical local cases. The 22M weights take roughly 1 GB on disk as rows. The embedding lookup table is the largest part of that and costs nothing at inference time, because it is just a join.

## Child runs for large fan-out

`run.spawn`: a `map` over thousands of elements starts child runs with their own world instead of child tasks, which avoids the $O(n^2)$ world copying (architecture §8). The parent waits on the child runs like on any other effect.

## Smaller ideas

- Completion-based I/O: `sys()` returns whatever has finished instead of waiting for the whole batch (architecture §3).
- A `graph` agent kind with its own edges table, for LangGraph-style workflows.
- `inbox.poll` and A2A, so agents can receive messages from other systems while running.
- A TUI or web view over `trace`.
- An agent that analyzes its own past runs with SQL tools.
