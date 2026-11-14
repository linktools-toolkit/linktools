# Trace SQL residual JSON simplification

This changes the SQL adapter's inline event representation, not public StepEvent
or portable snapshot envelopes. Native event references and their joins are
unchanged. Trace keeps its independent lifetime and publication policy. This is
**encoding simplification, not State/Trace single-source completion**.

All already-promoted scalar fields remain in their columns. Remaining metadata
is a JSON object rather than an envelope containing a typed dataclass containing
a typed key/value mapping. Unknown metadata strings and forward-compatible
residual fields survive read/export/restore. The prerelease SQL format is changed
in place; no legacy-format fallback or migration is introduced. The JSON column,
DDL and existing MySQL/PostgreSQL SQL expressions are unchanged.

## Evidence

Published baseline #311 `a0f56f6b` (tree `65ece0ac`), candidate based on the same tree.
Same Python 3.11.17, pydantic-ai 2.68 / harness 0.68, SQLite synthetic two-producer
graph, four model calls, two tools, Native checkpoints, synchronous Trace, live
observation before model completion and replay after reopen. Both passed using
manage.py. This is not production traffic or real MySQL/PostgreSQL acceptance.

Actual stored UTF-8 JSON payload bytes, queried after each run:

| Trace event kind | Rows | Before | After |
|---|---:|---:|---:|
| MODEL_REQUEST_STARTED | 4 | 712 | 308 |
| MODEL_REQUEST_SUCCEEDED | 4 | 1532 | 1096 |
| TOOL_CALL_STARTED | 2 | 274 | 76 |
| TOOL_CALL_SUCCEEDED | 2 | 350 | 148 |
| Inline total | 12 | 2868 | 1628 |
| Native lifecycle references | 4 | 764 | 764 |
| Whole Trace payload total | 16 | 3632 | 2392 |

Inline JSON decreased **1240 bytes (43.2%)**; whole Trace payload decreased **34.1%**.
These are JSON bytes, not physical database pages, index size or network traffic.

Same runtime-window SQL: INSERT 140→140, UPDATE 119→119, DELETE 2→2, WITH 10→10.
SELECT 587→594 and COMMIT 344→348 vary with live observation scheduling; this
single sample is not a stable latency or read-count comparison. No query, JOIN,
transaction boundary, audit publication or owner-revision mechanism was added.

Focused Trace, history, Native backup projection and cross-backend Trace snapshot
roundtrip gate: 47 passed. Architecture gate: 18 passed. Independent read-only
review found no must-fix. No real server plans or timings are claimed.
