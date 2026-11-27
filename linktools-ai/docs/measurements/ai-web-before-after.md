# ai-web turn I/O on SQLite

Run the bounded ASGI probe from the repository checkout:

```sh
python -m tests.ai._measure_web_session_io --tabs 1 --samples 2
python -m tests.ai._measure_web_session_io --tabs 2 --samples 2
```

The default is a warmed conversation, with Trace off and Step enabled.
Each sample opens a fresh temporary SQLite database, creates the session,
connects the observers, and completes one setup turn before counting.
Only one subscriber sends a prompt; the optional second subscriber is
passive. The model is an in-process fixture, not a live provider.

The measured window begins immediately before the message POST and ends
after every subscriber sees the terminal frame and the local producer
finishes. Initial snapshots and seeding are excluded from SQL but the
number of opened SSE streams is reported. Healthy publication does not
request session history or issue a terminal GET. Use `--measure-repair`
to record an exact-turn GET separately (it does not simulate a lost
frame), `--cold` for empty history, `--cold-reopen` for reopening seeded
history, or `--step-off` to isolate the effects of Step.

Report SELECT, each DML kind, explicit SQL BEGIN, and SQLAlchemy
transaction begin/commit/rollback events separately. A read-only
transaction rollback is not a failed turn; a SQL BEGIN statement and
a SQLAlchemy begin event are not necessarily the same count. These
are intercepted SQLite driver calls, not filesystem reads or writes.

This acceptance requires SQLite only. The ASGI probe does not drive the
browser or measure production or MySQL I/O. The baseline `927cea3` uses
`execution_id` and its own SSE/readback flow; this probe targets the
candidate's native `turn_id` contract. Do not subtract SDK-only baseline
numbers from the new ASGI count. For a before/after claim, measure each
version's own HTTP flow under the same prompt, model, history, feature
flags, observers, and terminal-plus-producer window.

## Measured candidate results

SQLite 3.53.1, warmed session, Trace off, deterministic model without tools.
Both repetitions agreed in each cell. SQLAlchemy transaction events are
shown as begin / commit / rollback. Explicit SQL BEGIN is separate.

| Observers | Step | SELECT | INSERT | UPDATE | SQL BEGIN | Transaction events |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 | on | 3 | 14 | 6 | 6 | 7 / 6 / 1 |
| 2 | on | 3 | 14 | 6 | 6 | 7 / 6 / 1 |
| 1 | off | 1 | 3 | 3 | 2 | 3 / 2 / 1 |
| 2 | off | 1 | 3 | 3 | 2 | 3 / 2 / 1 |

Every row used one message POST and zero terminal GETs. No DELETE or
PRAGMA occurred during these warm samples. Adding the passive observer
increased the number of open SSE streams, not SQL calls.
