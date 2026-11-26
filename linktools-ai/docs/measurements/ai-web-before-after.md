# ai-web turn I/O: paired MySQL measurement

Compare the shipped ai-web at baseline `927cea3` and the exact candidate
commit, using each revision's own web assets and HTTP contract. The candidate's
`tests.ai._measure_web_session_io` command drives the native turn API in-process;
it is useful for the candidate's SQL breakdown, but does not run on the
baseline or count requests the browser actually makes. Do not combine its
numbers with this comparison.

## Fixture

- Start one normal ai-web server per revision on loopback. Give each a clean,
  disposable MySQL database initialized **outside the measurement** with that
  revision's schema. Do not run the candidate against the baseline schema.
  Use the same MySQL version, async driver, SQL mode, database settings, pool
  configuration, hardware and Python/dependency versions where both revisions
  support them. Record any difference rather than pooling unlike runs.
- Give each measured server a dedicated database login that no other process
  uses. Use a different login to read counters; exclude database bootstrap and
  application startup. Do not use production accounts, data, or traffic.
- Configure the same deterministic model fixture for both revisions: one
  successful streaming answer, fixed chunks and output bytes, no tool calls,
  no model retries, no external work. For a tool scenario use the same fixed
  tool invocation and result separately. The candidate's in-process Python
  fixture is not a deployed provider endpoint; a shared model fixture and two
  ready isolated MySQL instances or databases are prerequisites.
- Split runs by history (`empty`, `warm: one prior successful turn`,
  `reopen: same seeded history, new server`), trace (`off`/`on`), Step
  (`off`/`on`) and observers (one sender, or one sender plus one passive
  same-session tab). Certify identical fixture settings instead of assuming
  that similarly named configuration options enable the same behavior. Start
  with `empty, trace off, Step on, one observer` and repeat with two observers.

## Same window on each revision

1. Open the same session in the selected tabs and wait for the initial page
   loads and session SSE snapshots. Verify no other client connects and no
   previous turn or background operation is active. Warm-up and snapshots
   are not included; capture the initial session snapshot separately if needed.
2. In each tab open DevTools Network and preserve the log. Record one baseline
   snapshot of the database counters below. Send the same prompt once from the
   first tab. Do not drive client HTTP by hand or explicitly fetch history:
   the baseline may legitimately fetch an execution or session readback on its
   own, and it must count.
3. Wait until the final answer (not just the last text delta) is settled in
   both selected pages and their send controls are available again. To count
   storage work after publication, keep observing until the server's owned
   turn producer has quiesced. Use the same evidence of producer quiescence for
   both revisions, such as an existing trusted turn-owner probe; if unavailable,
   record the run as UI-terminal only, not producer-drained. Record a second
   counter snapshot before closing tabs. If a reconnect, retry or repair occurs,
   keep its trace as a separate scenario rather than mixing it with the healthy
   path.
4. In the browser logs count requests by method/route: initial SSE connections
   and new SSE requests during the window, message POST, session/history GETs
   (full page versus exact turn) and execution GETs. Count automatic requests
   even if they complete after the model stops. An SSE connection already open
   at the start is one observer, not a newly initiated request. In `927cea3`,
   identify the turn by the POST's `execution_id` and session readback by
   `execution_id`; the candidate uses `turn_id` and its native lifecycle frames.
   Compare the same behavioral endpoint, not identical response envelopes.

## Database snapshot

Use a read-only observer account and existing Performance Schema
instrumentation. Check availability before measuring; a disabled instrument or
missing privilege is not a zero. Do not change server instrumentation or
truncate summary tables as part of this procedure.

```sql
SELECT EVENT_NAME,
       SUM(COUNT_STAR) AS calls,
       SUM(SUM_ROWS_EXAMINED) AS rows_examined,
       SUM(SUM_ROWS_SENT) AS rows_sent,
       SUM(SUM_ROWS_AFFECTED) AS rows_affected,
       SUM(SUM_ERRORS) AS errors
FROM performance_schema.events_statements_summary_by_account_by_event_name
WHERE USER = 'ai_web_bench'
  AND EVENT_NAME LIKE 'statement/sql/%'
GROUP BY EVENT_NAME
ORDER BY EVENT_NAME;
```

For each label subtract the start counters from the end counters. Report
`SELECT`; `INSERT`, `UPDATE`, `DELETE`, `REPLACE` separately and summed;
all other statement types by name; rows examined/sent/affected; and errors.
Treat negative deltas, a server restart/account reset, unrelated activity or
unfinished request work as invalid. These are **server statement counts**, not
SQLAlchemy driver calls: an executemany may result in a different number of
server statements. If transaction instrumentation is also available, report its
source and deltas separately; don't infer transaction counts from statement
totals. Do not compare server counters directly with the ASGI harness's
SQLAlchemy begin/commit/rollback callbacks.

Record one row per sample with both exact revisions, all fixture labels,
window type, observer count, HTTP counts by route, SQL counts above, elapsed
time and validity/exclusion reason. Interleave repeated baseline/candidate
samples under otherwise idle conditions. Report all valid samples (or their
median and range) and the exclusions. Missing browser, MySQL, provider fixture
or producer-drain evidence must be stated explicitly; this procedure does not
supply any measured numbers.
