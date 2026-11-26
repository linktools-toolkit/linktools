# Compact state addresses without losing identity

The 64-byte budget applies at a field's owner, not by truncating an encoded
address. Existing `store_key` is already bounded independently. The three
remaining `VARBINARY(768)` fields contain original identity, not simply a
lookup projection. Narrowing their columns alone rejects valid data.

| Field | What must survive | Where to change first |
| --- | --- | --- |
| `stream_key` | Partition, stream role, exact owner and position; generic capability keys are opaque bytes. Native run counter cursors project reversible encoded prefixes into lexicographic owner ranges. | Specify typed partition-local stream addresses for built-in owners. Carry distinct role and an already-bounded owner ID; leave arbitrary state keys alone. Port run counters, Trace and Task event readers/writers, sequence allocation and snapshot restoration together. |
| `owner_key` | Exact record owner for fact selection and deletion, portable snapshot verification and restore. An SQL row ID is not portable across backup/restore or necessarily across stores/tables. | Keep it until fact ownership can use a bounded, typed logical owner ID in every affected owner. Do not reuse a surrogate across stores or infer ownership from an index locator. |
| `operation_key` | Ordinary ledger actions and capability write receipts are distinct, exact identities. Some ordinary actions use the caller's idempotency key as the operation ID; capability keys include capability/revision/run/write identity. | Separate operation identity and caller request identity at their command entries first. Generated IDs may use existing bounded identities; an operation retry must still find the original decision under the original scope and key. Do not turn a long caller key into a new random operation on each retry. |

At the current boundary, ordinary operation IDs still accept a 128-byte
caller-supplied idempotency key. The namespace and tenant are each also
allowed to reach 128 bytes; an ordinary Execution operation with all three
at their maxima occupies 402 bytes before any change to the codec. A 128-byte
valid graph or run ID alone already exceeds the target. Dropping the
namespace/tenant/domain string only because SQL has a `store_key` is unsafe
for the public key generators: their contract is globally scoped and
backups can stage multiple owners without the original SQL key.

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
