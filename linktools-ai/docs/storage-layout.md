# Runtime storage layout

Runtime stores ordinary text and structured state in the database. Size alone
is not a reason to put a prompt, result, capture, declaration or conversation in
ObjectStore. Real files and typed binary attachments use ObjectStore references.
Asset versions, transcript attachment references and frozen input captures keep
their own retention semantics; changing physical placement does not remove them.

## Body encoding

Native message bodies use `ai_step_messages.message` as raw or zlib-compressed
bytes. `compression` identifies the codec and `raw_size` bounds decompression.
The existing canonical message SHA-256 remains the content identity. There is
no second stored-body digest. Typed attachments retain their manifest in
`payload_json`; only the attachment bytes go to ObjectStore.

Trace HTTP request/response bodies use the same existing content archive and
preserve their exact original bytes, including non-JSON or already encoded
bodies. They are not reserialized as model messages. Metadata-only reads do not
hydrate these bodies.

Generic archived content uses `ai_state_records.payload_bytes` with the existing
content metadata in `payload_json`. Other records remain structured JSON. This
avoids compressing, Base64-encoding and then decoding the same body just to pass
through a JSON database column. Portable JSON snapshots still encode binary at
the transport boundary; that is not the database representation.

The binary SQL type is SQLite BLOB, PostgreSQL BYTEA and MySQL LONGBLOB. A plain
MySQL BLOB would impose an incompatible 65,535-byte body limit. Database packet,
driver and server limits still apply to an individual row. The library does not
silently move ordinary long text to ObjectStore when a database rejects a row.
Schema provisioning remains explicit. These definitions do not constitute a
production DBA approval or a deployed migration.

## Table decisions

| Table | Owner and retained representation |
| --- | --- |
| `ai_state_records` | Domain state, CAS/leases and query projections stay structured. Archive content alone can carry a binary body. |
| `ai_state_aliases` | Secondary identities only; no body to compress. |
| `ai_state_facts` | Ordered facts, native references and query projections stay JSON. Transcript bodies already have archive/native owners; no unused binary column is added. |
| `ai_state_sequences` | Monotonic counters only. |
| `ai_state_operations` | Replay/control ledger and references stay JSON. No separate compressed copy. |
| `ai_asset_heads` | Namespace revision only. |
| `ai_asset_entries` | Current Asset metadata and file reference, never inline file bytes. |
| `ai_asset_changes` | Immutable Asset metadata/history and file reference. |
| `ai_asset_batch_receipts` | Committed batch results reference the same immutable files. |
| `ai_objects` | Object headers, identity and size only. |
| `ai_object_chunks` | Already opaque binary file chunks; no automatic recompression of images, archives or other encoded files. |
| `ai_metric_definitions` | Definition metadata stays JSON. |
| `ai_metric_observations` | JSON measurements, facets and correlation remain available for SQL filtering and aggregation. |
| `ai_step_runs` | Run lineage and shared metadata remain structured. |
| `ai_step_events` | Event fields and native run references remain queryable; no second message body. |
| `ai_step_messages` | Deduplicated raw/compressed binary message bodies and typed attachment manifests. |
| `ai_step_sequences` | Immutable message-sequence links and counts, not copied bodies. |
| `ai_step_snapshots` | Bounded descriptor and sequence root; the descriptor is not the whole conversation. |
| `ai_native_content_edges` | Rebuildable dependency and collection queue metadata, not another durable semantic owner. |

Identity, sequence, state, lease, CAS, revision, ordering and time columns are not
compressed. Existing unbounded opaque string contracts are not silently narrowed
to fixed-width IDs. Snapshot pruning uses the `(store_digest, run_digest, pruned)`
index and an equality predicate compatible with MySQL 5.7 index selection.

## Captures and history

Input captures, invocations, graph templates and binding declarations belong to
TASK StateStore records or its existing content archive. Frozen captures remain
independently readable when their source execution was transient. Declared typed
binary slots retain file references, while application JSON is not interpreted
as an attachment merely because it contains similarly named keys.

Capture expiry uses state records and durable cleanup receipts. A receipt keeps
state keys separately from actual file references. Failure or cancellation
between deleting a capture and acknowledging cleanup does not lose the remaining
file cleanup work. Offline collection follows the same owner/content closure.

A Conversation turn is a thin publication receipt pointing at its canonical
Execution terminal data and message range. Timeline reads batch the needed
Execution fields without expanding execution bindings. Session controls admission
and sequencing; it is not a replacement for standalone or Task executions.

## Cost boundaries

Removing duplicate content and Base64 database bodies reduces those specific
writes and stored bytes. It does not prove fewer database operations in every
workload: timeline pages need an Execution projection read, native collection
adds dependency writes and bounded maintenance transactions, and even a small
Asset file now has one ObjectStore write. The derived dependency index itself
uses linear space. Compare matched workloads by reads, writes, transactions,
scanned rows, returned bytes and retained bytes before claiming an overall gain.

See [native boundary details](atomic-native-step-boundaries.md) for append,
recovery, collection and extension limits, and [public entry points](runtime-entrypoints.md)
for the Runtime handle and explicit-ID APIs.
