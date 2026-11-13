# Joint model-request history publication

For synchronous Trace and Native checkpoints in the same SQL storage group,
publish the prepared request history and Native request/frontier in one short
transaction before provider dispatch. The immediate built-in model wrapper is
unwrapped for this invocation only. The original wrapper remains the retry entry
point, so each subsequent dispatch still crosses its producer fence.

Custom intervening wrappers retain the original pre-request Trace flush rather
than being unwrapped early. Split storage groups, buffered/terminal Trace, and
other boundaries retain separate publication. No transaction spans a provider
call, tool effect or user hook. No deferred flush credit, cross-request queue,
new cache, schema migration or new durable data owner is introduced.

## Matched SQLite evidence

Baseline: `9e924ff9e2cf952d1b835f2f998f88e3593937ac`. Candidate: this change based
on that commit. Same CPython 3.11.18, SQLite 3.53.1, SQLAlchemy 2.0.51,
pydantic-ai-slim 2.68.0 and harness 0.68.0, with a fresh database for each run.
The candidate's disabled-join control exactly matched the real
baseline; the baseline was also run from a separate checkout, not simulated only
by disabling the new gate.

One request, one offline model response, Native frontier enabled, synchronous
Trace, no tools. Counting starts after Runtime attachment, immediately before
`Agent.start`, and ends after the successful authoritative wait. The provider
checks that both the Native request event and Trace interaction are already
durable; those visibility-check reads are included. Initialization and close are
excluded. This is the first fixture in
`tests/ai/test_joint_model_history_boundary.py`, not the older two-turn workload.

| Counter | Baseline | Joint publication |
| --- | ---: | ---: |
| SELECT statements | 55 | 54 |
| INSERT statements | 33 | 35 |
| UPDATE statements | 24 | 22 |
| SQL BEGIN statements | 33 | 31 |
| SQLAlchemy transaction begins | 35 | 33 |
| Commit events | 31 | 30 |
| Rollback events | 4 | 3 |
| Committed transactions containing writes | 11 | 10 |
| Bound text/blob argument bytes | 78,299 | 78,469 |

The measured saving is **one commit and one committed write transaction**, not
two commits. SQL BEGIN and transaction events are separate measurements. Read
scopes that exit by rollback are included, not silently counted as commits.

The two extra INSERT statements are immutable message/sequence insertion attempts:
both projections prepare before either is confirmed, whereas separate commits
previously let the second preparation reuse confirmed content. Existing key/content
validation retains one durable content owner. This is a real statement/argument
tradeoff, not an all-counters decrease. No generic transaction row cache was added
to hide it. Bound bytes are neither network traffic nor physical database size.

The existing Trace-disabled `test_upstream_runtime_measurement.py` also passed
on the actual baseline and candidate with identical results: 31 SELECT,
129 INSERT, 44 UPDATE, 46 commits, 38 committed write transactions and 200,183
bound text/blob argument bytes. Both retain two outputs, sixteen history
messages, eight model calls, six tool effects and replay without duplicated
effects. This change claims no improvement in that Trace-disabled workload.

## Failure and authority boundaries

Both publication paths acquire the execution-owner boundary before the Native
publication lock. Joint admission checks actual storage-group identity, not just
engine identity. Producer authority remains guarded in the write transaction.

Trace can replace inline execution input with a shared content reference inside
the same transaction. The joint publisher retains that transaction's final
storage revision; it does not assume Native-only revision `+1`. A lost commit
acknowledgement is accepted only after the complete execution value and final
revision, Native records and Trace projection are confirmed. Encoded Native
checkpoints are decoded through their existing codec before semantic comparison.

A readback equal to the exact pre-transaction execution observation proves that
this same-group atomic publication did not commit. That is an atomicity/control
revision proof, not a claim that both sides were separately read and found absent.
Other uncertain outcomes remain fenced. Caller cancellation is propagated only
after the owned durable operation settles and Atomic's pending start/events are
acknowledged. Otherwise a committed start could be replayed incorrectly during
cancellation cleanup. Successful publication does not grant a later request
permission to skip its fence.

## Limits

These are targeted SQLite correctness and statement-count results. They do not
prove lower latency, fewer scanned rows, global minimum IO, whole Trace/State
single-source completion, or MySQL/PostgreSQL server acceptance. Real-server
plans, examined/node work, locking and latency remain separate validation work.
