# Retention owner projection

Retention closure only consumes execution identity, status, conversation ownership,
start time, dependency holds, recovery handoff state and revision metadata. When
an observed execution still has its original stored projection and matching
physical revision, these values have already been decoded. Re-reading immutable
runtime policy or recovery bodies does not add authority: the final owner CAS
rejects an intervening update. Unregistered caller-owned values keep the full
fallback path. Recovery handoffs and dependency holds still prevent closure.

## Matched measurement

The repository `test_upstream_runtime_measurement.py` probe was run before and
after this change with SQLite, memory cache, checkpointed persistence, frontier
capture enabled, coordination disabled, synchronous commits, four-byte tool
outputs and Trace disabled. Both runs execute two turns, eight model calls and
six tool calls, and preserve sixteen history messages plus replay semantics.
Startup/schema checks are not part of the measured turn interval.

| Counter | Before | After |
| --- | ---: | ---: |
| SELECT, first turn | 26 | 25 |
| SELECT, second turn | 29 | 28 |
| SELECT, total | 55 | 53 |
| Transactions / commits | 47 | 47 |
| Write transactions | 38 | 38 |
| INSERT | 130 | 130 |
| UPDATE | 46 | 46 |
| PRAGMA | 2 | 2 |

This removes one immutable policy lookup per closure, not a transaction or a
write. It is a narrow matched workload, not a bound for arbitrary conversations.
It must not be compared directly with measurements using different turn counts,
tool counts, recovery behavior, cache configuration or Trace settings.

Regression coverage includes original observed metadata, equal-object fallback,
stale physical revisions, dependency holds, prepared recovery handoffs and
concurrent dependency acquisition during retention closure.
