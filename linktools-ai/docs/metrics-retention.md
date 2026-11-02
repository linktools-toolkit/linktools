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

## Existing-database maintenance switch

New installations use `build_metrics_sql_metadata()` or the matching
`migrations/init_schema.sql`. Runtime does not alter existing schemas. The
provisioning functions create missing tables; they are **not an upgrade** for an
existing observation table.

This change requires a coordinated **non-mixed-version maintenance window**:

1. Inventory and stop every old reader, writer and scheduled pruner. Prevent old
   processes, jobs and deployment revisions from restarting throughout the new
   protocol's lifetime. Extra columns and nullable payloads do not fence old
   binaries: the generic schema validator does not reject all extra columns or
   nullability differences, and old prune code physically deletes receipts.
2. Back up the database and record the schema, row count, allowed replay windows
   and exact writer/reader versions. While writes are stopped, fully decode all
   still-present raw envelopes, rejecting duplicate JSON members and malformed
   or unsupported records. Verify original namespace/ID, namespace/observation/
   payload digests, kind and UTC occurrence time against the existing columns.
   Resolve any corrupt rows before continuing; never turn them into receipts.
3. With authorized, dialect-specific DDL, add `namespace VARCHAR(256)` and
   `observation_id VARCHAR(128)` using the same binary text collation as canonical
   metadata, initially nullable for backfill; add nullable `pruned_at` and
   `active_occurred_at` using the canonical timestamp type. Permit SQL NULL in
   `payload_json`. No data-envelope version bump or schema manifest is required.
4. Backfill namespace/ID from each **validated** envelope, set `pruned_at` to
   NULL and `active_occurred_at = occurred_at`. Do not derive original identities
   from their hashes. Validate every backfilled row, count and invariant, then
   make namespace/ID NOT NULL. Preserve the observation-digest unique constraint
   and all audit columns/indexes.
5. Replace the two occurrence-time query indexes with
   `(namespace_digest, kind, active_occurred_at)` and
   `(namespace_digest, active_occurred_at)` using DDL compiled from canonical
   metadata for the target dialect. Generated portable index names use
   SQLAlchemy's deterministic shortening when they exceed that dialect's limit;
   MySQL index names remain table-local. SQLite requires an appropriate table rebuild to change nullability;
   preserve IDs, uniqueness and audit values when copying validated rows.
6. Run schema validation and a full backfill integrity check before starting only
   the matching new reader/writer/pruner versions. Verify active reads and query
   plans against the real dataset, then validate pruning and replay on controlled
   records. Record the exact deployed code/schema pair. Local unit tests do not
   establish a deployment's migration or query-plan performance.

No historical ID can be recovered from a payload already physically deleted by
an old pruner. The no-reactivation guarantee covers only retained valid receipts.
Earlier replay needs actual historical evidence or an explicit host acceptance/
rejection policy, not an invented backfill.

After any tombstone or NULL payload exists, do not restart an old binary as a
rollback. Stop all access again, reconcile the data and either restore compatible
backed-up data with an explicitly assessed replay window, or keep the new
protocol code. One initial process shutdown is not a permanent mixed-version
safety guarantee. No production migration, retention period or GC window is
chosen by this implementation.
