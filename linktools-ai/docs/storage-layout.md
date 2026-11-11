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

## State subject identities

Record and fact `subject_id` projections keep the original resource ID, using
its existing nonempty, exact UTF-8 contract and 128-byte limit. They do not add a
subject hash. Record lookup remains scoped by `scope_key`; fact lookup and
latest-per-subject grouping remain scoped by `stream_digest`. Subject identity
is immutable during record CAS, including changes between `None` and an ID.

The wire stores the original string. SQL encodes it once as UTF-8 in MySQL
`VARBINARY(128)`, PostgreSQL `BYTEA` or SQLite `BLOB`, preserving case and Unicode
code points without collation or normalization changes. Application validation
enforces the byte limit on every backend. The record subject index declares at
most 1152 bytes on MySQL and PostgreSQL. The fact subject index declares 392
MySQL key bytes including its sequence, or 200 PostgreSQL field bytes before
index tuple overhead, well below an ordinary B-tree entry limit.

This removes subject JSON serialization and hashing without another lookup or
encoded ordering key. SQL subject values are their actual UTF-8 length rather
than 64 ASCII hex characters: short IDs shrink, while a maximum-length ID can
use 128 bytes. It is not a claim that every ID or in-memory representation uses
less storage.

## State operation addresses

`StoredOperation.key` is a binary address of 1..1024 bytes. Public repositories
can share a bare or custom StateStore, so ordinary ledger keys carry their
namespace, tenant and domain even when standard stores also use a
`store_digest` partition. Built-in families reserve distinct first bytes:

| Family | Physical address | Maximum bytes |
| --- | --- | --- |
| Ordinary ledger | `01`, then namespace, tenant, domain and operation ID, each framed by its two-byte big-endian UTF-8 byte length | 661 |
| Capability receipt | `02` followed by the existing 32-byte composite receipt digest | 33 |
| Capability partition marker | `00` followed by ASCII `capability-partitions` | 22 |

Ordinary operation IDs use the existing idempotency-key contract: nonempty,
at most 256 UTF-8 bytes, no edge whitespace or Unicode control characters
(`Cc`). Namespace and tenant retain their existing 256- and 128-byte limits.
The longest built-in domain is `conversation` at 12 bytes. Four lengths plus
the family byte make the ordinary maximum `1 + 8 + 256 + 128 + 12 + 256 = 661`.
Case and Unicode code points are preserved without normalization or truncation.
Primitive callers may still supply any bounded binary key, including 32-byte
values; callers sharing built-in partitions must avoid reserved addresses.
Request digests and upstream idempotency ID generators are unchanged.
Capability receipts retain their composite SHA-256 of namespace, tenant,
capability ID, revision, run ID and write key, including currently unbounded
custom write keys. Their streams and the marker stream retain their existing
32-byte digests. Ordinary stream digests remain globally scoped, and repository
reads verify the key, stream and payload against the bound scope.

SQL stores operation and alias addresses directly: MySQL `VARBINARY(1024)`,
PostgreSQL `BYTEA`, SQLite `BLOB`. Each unique `(store_digest, address)` index
declares at most 1280 MySQL key bytes; PostgreSQL field payload is at most 1088
bytes before index tuple overhead. Binary comparison defines scan ordering on
all backends. Current snapshot wire encodes the key as canonical lowercase hex
and decodes the same bounded bytes. Native content ownership uses that binary
address to select the existing operation row ID. Snapshot aliases are rebuilt
from the manifest scope and their durable record identities; restore retains
the manifest namespace and tenant when moving state to another backend.

This removes ordinary physical-key JSON serialization and hashing, with no
additional SQL, mapping table, compatibility encoding or encoded sort key.
Scope is repeated to preserve public shared-store isolation. A generated
64-ASCII-byte operation ID with namespace `runtime`, tenant `tenant` and domain
`conversation` uses 98 bytes rather than the previous 64 SQL hex bytes or 32
in-memory digest bytes. No universal storage or measured runtime speedup is
implied.

## State record relations

`scope_key` and `parent_key` are nonempty binary addresses of at most 1024
bytes. Direct relation helpers encode a flat sequence of exact UTF-8 strings:
role (`scope` or `parent`), namespace, tenant, domain, record kind, relation,
then each logical identity component. Every component has a fixed two-byte
big-endian byte length. Scalars and nested JSON are not alternative encodings.
The role keeps relation addresses distinct from other address families. Global
scope remains in every key because public repositories can share one bare or
custom StateStore across namespaces, tenants and domains.

The complete encoded bound is checked at construction, queries and current
SQL/file readers. Namespace and tenant keep their 256- and 128-byte limits;
validated resource IDs use 128 bytes, principal IDs and local turn purposes
use 256. Some domain DTOs, including graph and budget scope IDs, do not impose
those individual bounds. Their actual restriction here is the complete encoded
1024-byte relation limit, not a new 128-byte ID limit. With a 384-byte
namespace/tenant pair, representative sizes at these component budgets are:

| Relation family | Remaining payload and all frame bytes | Encoded bytes |
| --- | --- | --- |
| Session principal owner | scope5 + conversation12 + session7 + owner5 + principal256 + 14 | 683 |
| Input capture principal owner | scope5 + task4 + input_capture13 + owner5 + principal256 + 14 | 681 |
| Graph binding principal owner | scope5 + task4 + task_graph_binding_capture26 + owner5 + principal256 + 14 | 694 |
| Turn purpose | scope5 + execution9 + turn4 + scope5 + purpose256 + 14 | 677 |
| Idempotency resource pair | scope5 + domain12 + idempotency11 + resource8 + resource-kind14 + resource-ID128 + 16 | 578 |
| Task definition parent with a 128-byte graph ID | parent6 + task4 + task_node_definition20 + graph5 + graph-ID128 + 14 | 561 |
| Tool execution parent | parent6 + checkpoint10 + tool_operation14 + execution9 + execution-ID128 + 14 | 565 |

Other direct resource relations use the same framing with their existing owner
components; custom primitive relations are bounded by the complete 1024-byte
limit. The existing memory-scope digest is a component, not recomputed as a
second relation hash. Raw reused record-parent digests and ContentArchive's
independent scope digest remain 32-byte values. Record keys, store identities,
fact streams, fact owners, alias targets and native Step scopes retain their
separate digest contracts in this change.

SQL binds raw bytes as MySQL `VARBINARY(1024)`, PostgreSQL `BYTEA` or SQLite
`BLOB`. `(scope_key, sort_key)` and `(parent_key, sort_key)` remain two-column
indexes, with a complete field payload of 1024 + 1152 = 2176 bytes. Allowing
40 bytes for headers, alignment and tuple overhead still fits PostgreSQL's
ordinary approximately 2704-byte B-tree limit and MySQL's 3072-byte limit.
`(scope_key, subject_id)` has at most 1152 field bytes. SQL still filters the
store partition; no new mapping, SQL round trip, index prefix or compatibility
encoding is introduced. Portable wire keeps canonical lowercase hex for these
bounded binary values, with the existing format version. Snapshot readers
recompute direct keys against the manifest scope rather than the destination.

## State record ordering

`StoredRecord.sort_key`, query prefixes and pagination boundaries are nonempty
binary values of at most 1152 bytes, including every prefix. Writers and readers
validate the same bound. Ordinary record order reuses the existing 32-byte
record key directly. Chronological order uses eight big-endian epoch-microsecond
bytes followed by that key, totaling 40 bytes. Explicit path, sequence and trace
orders retain their ASCII semantics as ASCII bytes. No hex or Base64 encoding
participates in SQL ordering.

State uses `VARBINARY(1152)` on MySQL, `BYTEA` on PostgreSQL and `BLOB` on SQLite.
Its indexes cover the full sort value, never a text or binary prefix. The
three-column sort index is `(store_digest, kind, sort_key)`. MySQL's declared
field budget is 256 + 128 + 1152 = 1536 bytes, below 3072 even allowing 32 bytes
for variable-length and record overhead. PostgreSQL's maximum actual field
payload is 64 + 128 + 1152 = 1344 bytes; a conservative 40-byte allowance for
field headers, alignment and the index tuple remains below its approximately
2704-byte B-tree entry limit. The wider two-column relation indexes have the
2176-byte field budget described above. These are size calculations, not
measurements on a deployed server.

Binary prefixes use a lexicographic lower bound and their least upper bound;
an all-`ff` prefix has no finite upper bound. Pagination uses `(sort, record
key)`, or strictly greater sort when the query omits an after-key. There is no
fabricated maximum record key. Record cursor transport contains a two-byte
sort length, raw sort, two-byte key length and raw key, then canonical unpadded
URL-safe Base64. Current 32-byte keys give at most 1188 framed bytes and 1584
encoded characters. Portable record JSON transports sort bytes as Base64 and
returns raw bytes before comparison. The current format version is unchanged.

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
indexes already exceed 767 bytes, including `(store_digest, kind, sort_key)`.
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
