# Current owner and fixed content reads

Targeted SQLite probes on 2026-11-12, code through `e49162bd`. The implementation
starts at `720ffeef`; `32347ed9` avoids repeating the owner lookup for each declared
content slot. These are diagnostic probes, not a local formal test-suite run.

## Matched result read

Both paths use the same real writer, SQLite database, dependencies and terminal
result containing `"output " * 40000`. The comparison disables only the new
projection prefetch for the baseline. Counts cover a fresh public repository
result read with an already-open archive, not setup or a complete Runtime turn.

| Metric | Separate reads | Current-owner join |
| --- | ---: | ---: |
| Read statements, counting SELECT and WITH | 3 | 1 |
| Returned rows | 4 | 4 |
| Returned string/BLOB bytes | 5341 | 5349 |
| Returned numeric cells | 12 | 16 |
| One batch of 30 warm reads | 0.3173 s | 0.2858 s |

Bytes are the sum of UTF-8 strings and binary values in returned database rows,
not network packet sizes. The tagged union adds one numeric source discriminator
per row. It does not repeat owner JSON per content body. The timing is one small
paired diagnostic, not evidence of a stable end-to-end speedup. An earlier version
that explicitly compiled every statement was slightly slower than its baseline;
the final version instead uses a bounded batch size verified against each dialect's
parameter expansion.

## Query plan, not a physical-I/O claim

SQLite EXPLAIN QUERY PLAN reports two exact owner index-search nodes: one returns
the owner and one extracts its declared references. A constant slot table and
DISTINCT reference set are small intermediate relations. Generic configuration
content uses the store/kind/sort-prefix index with complete scope and identity
predicates. Typed Step payload content uses the store-locator/SHA-256 unique index
with the complete raw store key checked as well. No business-table scan or full
business-table materialization appears in this fixture's plan.

One SQL statement is not one physical I/O. Each referenced content identity can
still require an index lookup, and this probe does not measure database page reads.
The join reduces round trips and repeated result processing, not the required
source existence and integrity checks.

MySQL 5.7 uses derived subqueries rather than WITH. SQLite, PostgreSQL and MySQL
5.7 compilation passed for 1, 4, 8 and 16 declared paths at their maximum batch
sizes, each below the 999-bind limit. No MySQL or PostgreSQL server was available,
so their execution plans and latency were not measured. In particular, the MySQL
derived-table plan may repeat indexed owner lookups across union arms.

## Integrity and scope

The join follows only explicitly declared typed reference paths from the current
owner in the active read-only transaction. Existing policy, binding and payload
decoders still check source scope, codec, identity and complete body integrity.
Narrow metadata reads do not fetch input/output bodies. Cold archive discovery
restarts the owner read; an owner deleted during discovery returns missing rather
than hydrating the stale owner. Focused probes exercised that race plus missing
policy, foreign binding scope, damaged result template and missing output source.
The full corruption matrix is committed for natural CI.

This result does not establish the cost of child listings, Native message body
reads, snapshots or an entire workload, and does not change authorization or
offline retention semantics.
