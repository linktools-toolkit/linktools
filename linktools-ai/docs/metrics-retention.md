# Metrics retention and replay

`Metrics.prune(before=...)` removes the payload of each active observation before
that UTC timestamp. It retains the original namespace, observation ID, payload
digest, kind and occurrence time in the same unique row. `pruned_at` is the first
backend pruning time; a second prune neither counts nor refreshes that receipt.
Memory, path-backed SQLite and SQL share this contract.

Reads, pagination, scan limits and SQL aggregations omit pruned observations.
The nullable `active_occurred_at` column is a derived query index projection, not
an independent deletion fact. Active records require a valid payload and an
active time matching their occurrence time. Pruned records require both fields
to be SQL NULL. Prune validates the full active envelope and its projections
before destroying payloads, within one transaction even when candidates are
read in bounded pages.

## Exact replay proof

An original-ID replay never reactivates a receipt. Different payloads always
raise `STORAGE_CONFLICT`. A matching payload for a validated pruned identity
raises:

```python
AIError(
    ErrorCode.STORAGE_CONFLICT,
    retryable=False,
    safe_details={"reason": "metric_observation_pruned"},
)
```

Only a **single-item call with the original namespace, ID and payload** can use
that exact reason as a source-confirmed deletion proof. `get_observation() is
None`, generic conflicts, non-retryable failures and a whole-batch pruned error
are not per-item proofs. Explicit writes still fail; pruning is not reported as
a newly successful write. Unknown commit/rollback outcomes must not be relabeled
as that proof. The facade may retry an unknown original write after readback;
a later confirmed pruned error does not prove the earlier attempt rolled back.

Batches remain atomic. A pruned or conflicting item rolls back other new items
in that mutation. Consumers settle each original ID separately after a failed
batch and acknowledge only their contiguous proven-safe prefix. This source
contract does not impose a reliable queue on best-effort automatic metrics.

A first, never-stored late observation remains admissible. Retention is not a
permanent occurrence-time admission floor; expiration is the host's separate
policy. Receipts occupy space proportional to retained IDs. Remove receipts
only through an authorized maintenance process after every allowed producer
retry, broker replay and manual redrive window has ended. Zero lag alone does
not establish that condition. There is no automatic receipt GC.

## SQL schema

Provision the current schema explicitly with `build_metrics_sql_metadata()` or
`migrations/init_schema.sql`. Runtime does not create or migrate these tables.
There is no committed older Metrics schema or data format to migrate.

Definition identity is the full `(namespace, metric_name, revision)` tuple.
Observation identity is the full `(namespace, observation_id)` tuple. Namespace
and observation ID retain their existing limits of 256 and 128 UTF-8 bytes.
MySQL uses `utf8mb4_bin`; PostgreSQL uses `C` for the namespace and observation
ID; SQLite uses its binary text comparison. No namespace or observation ID hash
is stored, calculated or used as a second identity.

Both unique keys index complete values. MySQL's ordinary observation indexes
use `(namespace(128), kind, active_occurred_at)` and
`(namespace(128), active_occurred_at)`. Queries still compare the complete
namespace, so a shared 128-character prefix does not combine partitions.
PostgreSQL and SQLite index the complete values.

`payload_digest` covers the complete canonical ObservationEnvelope before
model decoding, including namespace and observation ID. Active reads and prune
verify it against the stored payload. After pruning, it remains the content
comparison evidence for replay: the same identity and digest produce the
specific pruned conflict, while a different payload remains a normal conflict.
Changing a row's namespace or observation ID changes its storage identity; a
receipt at another key does not prove that the original key was pruned.
