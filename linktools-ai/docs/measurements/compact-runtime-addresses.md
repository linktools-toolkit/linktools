# Compact state addresses without losing identity

The 64-byte budget applies at a field's owner, not by truncating an encoded
address. Existing `store_key` is already bounded independently. Common State
columns and dedicated runtime owners must be budgeted separately.

### Generic State columns and current limits

Exact record, relation and stream addresses share a 640-byte ceiling. That
budget includes framing and any compound owner parts, not a public business ID.

| Table | Columns | Public writer that can use the full width |
| --- | --- | --- |
| `ai_state_records` | `key`, `scope_key`, `parent_key` | `StoredRecord` accepts opaque bytes; `RecordQuery` compares the complete scope or parent. |
| `ai_state_facts` | `stream_key`, `owner_key` | `StoredFact` can refer to any such stream and record; owner-based deletion matches the original key exactly. |
| `ai_state_counters` | `stream_key` | `StateTransaction.next_sequence` and batch counter operations accept a raw stream address. |
| `ai_state_operations` | `operation_key` (301 bytes), `stream_key` (640 bytes) | The largest built-in operation receipt fits 301 bytes. Custom operation IDs share that limit; the independent opaque ordered stream still has a 640-byte contract. |

Shorter built-in addresses do not justify shrinking a column shared with
those callers. The SQL regressions in `test_state_stream_addresses.py` and
`test_state_store_keys.py` cover maximum-width raw identities, lookups and
ownership. The remaining dedicated columns must match their domain writers,
including direct SQL insertion and portable restore.

### Dedicated record columns

| Table | Key | Scope | Parent |
| --- | --- | --- | --- |
| `ai_task_graphs` | 224 bytes: six framed components, namespace and tenant at 64 bytes each and Graph ID at 64 | Removed; graph headers have no scope | Removed; graph headers have no parent |
| `ai_execution_records` | 360 bytes: up to 228 for Execution and 360 for its idempotency receipt | 300 bytes: full session or execution-idempotency resource relation | 239 bytes: full parent-execution relation |
| `ai_step_run_admissions` | 231 bytes: reserved run address with a bounded 64-byte run ID; other key shapes stay in generic State | 308 bytes: full conversation relation | 239 bytes: full parent-run relation |
| `ai_step_tool_operations` | 234 bytes: exact checkpoint address with a bounded 64-byte operation ID; other key shapes stay in generic State | 308 bytes: full agent-run relation | 309 bytes: full execution relation |

Persistence namespaces and tenant IDs are limited at ingress to 64 UTF-8 bytes
each. The same limits size the five native Step partition columns (146 bytes,
including tuple framing), as well as dedicated keys, scope and parent addresses.
This cutover changes the maximum accepted input, not the bytes of identities that
fit. Existing stores beyond the new bounds must not be opened with the new schema
until they have been rebuilt or migrated offline. Asset IDs and ObjectStore keys
are separate opaque paths; a persistence ID limit must not truncate them.
The generic `sort_key` still carries ordered data such as complete encoded
memory paths for prefix scans. It is not an identity column and is not included
in the raw-key ceiling.

For the dedicated owners, the full bounded key and relation address are
retained; the canonical reserved key determines which
widths apply. Generic record keys retain the full opaque relation budget. Separate
typed ownership and preserve opaque State records before tightening these
columns. The Task Graph owner now takes only its dedicated address shape;
other shapes stay in the generic table. For existing pre-release data using
a superseded Task Graph owner layout, rebuild before switching schemas.

| Field | What must survive | Where to change first |
| --- | --- | --- |
| `stream_key` | Partition, stream role, exact owner and position; generic capability keys are opaque bytes. Native run counter cursors project reversible encoded prefixes into lexicographic owner ranges. | Specify typed partition-local stream addresses for built-in owners. Carry distinct role and an already-bounded owner ID; leave arbitrary state keys alone. Port run counters, Trace and Task event readers/writers, sequence allocation and snapshot restoration together. |
| `owner_key` | Exact record owner for fact selection and deletion, portable snapshot verification and restore. An SQL row ID is not portable across backup/restore or necessarily across stores/tables. | Keep it until fact ownership can use a bounded, typed logical owner ID in every affected owner. Do not reuse a surrogate across stores or infer ownership from an index locator. |
| `operation_key` | Ordinary ledger actions and capability write receipts are distinct, exact identities. Some ordinary actions use the caller's idempotency key as the operation ID; capability keys include capability/revision/run/write identity. | Separate operation identity and caller request identity at their command entries first. Generated IDs may use existing bounded identities; an operation retry must still find the original decision under the original scope and key. Do not turn a long caller key into a new random operation on each retry. |

Built-in ordinary operation addresses are at most 163 bytes and capability
write receipt addresses are at most 301 bytes. The original caller key
remains exact. Execution, execution request, logical tool operation, Task Graph and native
agent run IDs now have a 64 UTF-8 byte ingress limit. Provider tool-call IDs
and independently named task/node IDs keep their own contracts. Native
Web/CLI/ACP turn request IDs are generated as 12 URL-safe ASCII characters with
72 random bits scoped by session. Buffer confirmations can outlive a producer
and must never acknowledge a successor's entry, so buffer IDs do not repeat
a producer-local sequence. Global durable identities, lease tokens and
digests are not truncated. Dropping the namespace/tenant/domain string merely because SQL
has a `store_key` is unsafe for public record and stream key generators:
their contract is globally scoped and backups can stage multiple owners
without the original SQL partition.

## Proposed cutover order

1. Bound each affected caller and generation boundary separately. Inventory
   model/provider identifiers, task graph IDs, injected/custom IDs and
   externally supplied idempotency keys. Do not globally shorten
   `validate_resource_id` or downgrade an exact provider key to a preview.
2. Introduce a partition-local encoding only inside storage for a typed
   built-in owner once one existing field is sufficient to recreate the
   exact identity. Keep scope and family checks at the same write/read
   boundary; preserve binary cursor ordering or declare a new cursor
   contract. Leave custom bytes intact; they cannot share a narrowed column
   until their public contract is intentionally changed.
3. For a pre-release schema cutover, rebuild affected disposable stores
   from the new owner format. Any retained environment needs one offline
   rewrite from the authoritative raw records before switching code:
   rewrite affected owner/stream/operation keys with their indexes,
   preserving sequence heads, idempotency decisions, and source references.
   Never mix old and new keys using dual writes or hash/truncate as a fallback.
   Old ordered cursors and portable snapshots need explicit migration or
   invalidation, not silent reinterpretation.
4. Tighten DDL only after every producer, including portable restore and
   generic StateStore paths for that column, proves the new bound. Record
   the resulting SQL/probe deltas; a new mapping table or lookup to save
   key width is not a default improvement.

Check current round-trip and rejection coverage in
`test_state_store_keys.py`, `test_sql_run_counter_owner.py`,
`test_live_history_snapshot_validation.py`, `test_native_backup_projection.py`
and ledger/execution recovery coverage. Include raw UTF-8 boundaries,
same-value keys from different tenants/families, exact capability receipt
replay, sorted counter paging and portable restore without row-ID reuse.
No database rewrite or SQL performance result is implied by this design.
