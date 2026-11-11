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
compressed. Snapshot pruning uses the `(store_digest, run_id, pruned)` index and
an equality predicate compatible with MySQL 5.7 index selection.

## Native Step identities and digest ownership

Native `run_id`, `conversation_id` and `parent_run_id` use the resource ID
contract: nonempty UTF-8 strings of at most 128 bytes, with no leading/trailing
whitespace or Unicode control characters (`Cc`). Optional lineage remains `None`
when absent. IDs are preserved exactly, without trimming, Unicode normalization,
truncation or digest substitution. The same limit applies to writes, lookups,
Runtime projections and backup restore. Generated Runtime run/conversation IDs
are 66 ASCII bytes; native SDK UUIDs also fit this contract.

The original IDs are indexed directly. Runs are unique by `(store_digest,
run_id)`; lineage indexes use `(store_digest, conversation_id)` and `(store_digest,
parent_run_id)`. Step tables and native Trace run references do not store another
run-identity hash. Rejecting edge whitespace keeps these IDs exact under MySQL
PAD SPACE comparisons too.

SQL declares the IDs as `VARCHAR(128)`, whose width counts characters rather than
UTF-8 bytes; application validation enforces the tighter byte bound on every
backend, including SQLite. With `utf8mb4`, a scope `CHAR(64)` plus an ID declares
768 key bytes, or 769 with the snapshot boolean. These fit InnoDB's 3072-byte
large-prefix budget, not its legacy 767-byte configuration. Existing Runtime
indexes already exceed 767 bytes, including `(store_digest, kind, sort_key(128))`.
Provision MySQL 5.7 with the applicable large-prefix row format/settings; this
contract adds no fallback for legacy index settings. The same bounded IDs stay
well below PostgreSQL's ordinary B-tree entry limit. DDL compilation is distinct
from validation on a deployed database server.

| Table | Retained digest or content reference | Contract |
| --- | --- | --- |
| `ai_step_runs` | `store_digest` | Composite namespace/tenant partition. The original run and lineage IDs are direct lookup keys. |
| `ai_step_events` | `store_digest`, `key_digest` | Partition and a unique composite of the original run ID with its replay key, or a distinct unkeyed append occurrence. |
| `ai_step_events` | `body_sha256`, `fields_sha256` | Managed lifecycle source integrity. Full recovery authenticates metadata too; metadata-free Trace projections authenticate their fields without reading the large metadata body. |
| `ai_step_snapshots` | `store_digest`, `key_digest` | Partition and retained composite replay identity. Snapshot pruning uses the original run ID. |
| `ai_step_snapshots` | `sequence_root` | The sole stored root of the immutable message sequence. The bounded descriptor stores its count and checkpoint control, not another root. |
| `ai_step_messages` | `store_digest`, `sha256` | Scoped anonymous content addressing and integrity of original uncompressed canonical message bytes. |
| `ai_step_sequences` | `store_digest`, `sha256`, `message_sha256`, `left_sha256`, `right_sha256`, `next_sha256` | Scoped immutable ordered sequence addressing and references to its content, subtrees and next forest. Repeated messages remain distinct occurrences. |
| `ai_native_content_edges` | None | Physical row IDs and publication/collection revisions form a rebuildable index, not another hashed content identity. |

ContentArchive uses the existing StateStore tables rather than another SQL
schema. Its `raw_digest` authenticates original content bytes across raw and
compressed representations; the anonymous content address binds that digest
to the codec. Sequence addresses bind content, count and ordered links. Archive
scope and record keys bind those addresses to their owning namespace, tenant
and content kind so recovery and collection cannot cross archive partitions.
Typed attachment references keep their own byte-integrity checks. These are
content, storage and source-proof contracts, not named behavior identities.

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

Large Execution input, result and prepared terminal bodies are compressed database
archive content, alongside large binding contracts. The control record retains
scoped content addresses; ordinary text and JSON are not ObjectStore files. Only
typed attachment slots contribute binary dependencies. The archived input also
owns its display/model view so a repeated prompt is not copied into control CAS.
Snapshots include and budget the referenced content before decoding it, rekey
physical archive scopes on restore, and verify the same typed attachment closure.
Offline collection keeps content while its Execution owner is retained.

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
