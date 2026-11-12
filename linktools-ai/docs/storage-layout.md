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

Accepted execution and tool payload bytes use the existing `ai_step_messages`
content owner in a separate typed partition. Other archive families use
`ai_state_records.payload_bytes` with content metadata in `payload_json`; ordinary
control records remain structured JSON. This
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
| `ai_state_facts` | Ordered facts, native references and query projections stay JSON. Transcript bodies already have archive/native owners; no unused binary column is added. |
| `ai_state_counters` | Monotonic counters shared by execution, task, conversation and trace domains; not message sequences. |
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
| `ai_step_messages` | Deduplicated Native and accepted execution/tool payload bytes in disjoint typed partitions; attachment manifests retain file references. |
| `ai_step_run_admissions` | Domain membership and actual admitted projection differences, referencing canonical run registration. |
| `ai_step_tool_operations` | Authoritative tool effect state, fencing and stable operation identity; payloads reference fixed content owners. |
| `ai_trace_events` | Trace correlation and event fields or fixed Native event sources. |
| `ai_step_sequences` | Immutable message-sequence links and counts, not copied bodies. |
| `ai_step_snapshots` | Bounded descriptor and sequence root; the descriptor is not the whole conversation. |

Identity, sequence, state, lease, CAS, revision, ordering and time columns are not
compressed. Snapshot pruning uses the `(store_key, run_id, pruned)` index and
an equality predicate compatible with MySQL 5.7 index selection.

## State subject identities

Record and fact `subject_id` projections keep the original resource ID, using
its existing nonempty, exact UTF-8 contract and 128-byte limit. They do not add a
subject hash. Record lookup remains scoped by `scope_key`; fact lookup and
latest-per-subject grouping remain scoped by `stream_key`. Subject identity
is immutable during record CAS, including changes between `None` and an ID.

The wire stores the original string. SQL encodes it once as UTF-8 in MySQL
`VARBINARY(128)`, PostgreSQL `BYTEA` or SQLite `BLOB`, preserving case and Unicode
code points without collation or normalization changes. Application validation
enforces the byte limit on every backend. The record subject index declares at
most 1152 bytes on MySQL and PostgreSQL. The fact subject index declares 1416 field bytes
including its sequence on both MySQL and PostgreSQL, before index tuple overhead, well below an ordinary B-tree entry limit.

This removes subject JSON serialization and hashing without another lookup or
encoded ordering key. SQL subject values are their actual UTF-8 length rather
than 64 ASCII hex characters: short IDs shrink, while a maximum-length ID can
use 128 bytes. It is not a claim that every ID or in-memory representation uses
less storage.

## State record addresses

`StoredRecord.key` and fact `owner_key` are exact binary
addresses of 1..1152 bytes. Built-in addresses frame `record`, namespace,
tenant, domain, kind and each explicit owner component with a two-byte
big-endian UTF-8 byte length. Owners render their integer indices explicitly;
the common encoder rejects scalar, nested and non-string components. There
is no normalization, truncation, fallback hashing or alternate wire shape.
Archive system records use their reserved empty namespace and tenant binding.

Independent repository bindings can share one bare StateStore. The full
address preserves namespace, tenant, domain and family separation there.
Ordinary resource IDs retain resource128; task graph and node components are
bounded separately. Budget scopes permit 256 bytes, including built-in
`execution:` and `graph:` prefixes with resource128 suffixes. Evaluation
revisions and transcript indices use their owning int63 contracts. Capture
paths permit 512 ASCII bytes. Turn session/request IDs permit 393/385 UTF-8
bytes, preserving the generated request-backed session identity. At maximum
namespace256, tenant128 and scope256, a turn session key occupies 1074 bytes.

SQL uses MySQL `VARBINARY(1152)`, PostgreSQL `BYTEA` and SQLite `BLOB` for
record references. Default sort is the raw key; timestamp sorts prepend eight
bytes without imposing a separate identity bound on ordering data. Parent
references also permit 1152 bytes. Sort indexes follow the backend limits
and MySQL prefix policy described below.
The record unique index has at most 1920 MySQL declared key bytes with the
current store key. Native Step byte-integrity and anonymous-content checksums
retain their separate contracts, not hashed store or run locators.

Current wire record references are canonical lowercase hex decoded back to
bytes. The record cursor frames the bounded key with a two-byte length, followed
by the complete sort bytes, before canonical URL-safe Base64 encoding. Cache batch digests frame each
sorted, deduplicated key by its length, keeping variable component boundaries
and single-record/batch families distinct. Raw record, target, owner and
cursor addresses are omitted from object representations.

## State store partitions

Physical State stores accept one opaque `store_key` of 1..768 bytes. The three
built-in address producers first frame the complete validated tuple with fixed
two-byte big-endian UTF-8 component lengths, then apply SHA-256 once and return
32 raw bytes:

- Ordinary and configured capability routes: `(state_store, namespace, tenant, partition)`.
  Configured capability partitions retain `capability:` followed by the validated capability ID.
- Dynamic capability routes: `(capability-store, namespace, tenant, capability ID)`.
  They remain distinct from explicitly configured routes.
- Standalone SQL stores: `(standalone, partition)`.

The two short system tuples `(state-group-metadata)` and `(state-group-archive)`
remain raw framed bytes. Constructors, readers and partition opens accept the
already selected physical key without hashing it again. Aliases and reopened
handles therefore use the same bytes. Custom opaque keys retain their complete
1..768-byte API and are never normalized or implicitly hashed.

The input tuple bounds are unchanged: maximum configured capability framing is
670 bytes and dynamic capability framing is 664 bytes, before their single
compression to raw32. Compact built-in partitions leave index space for complete
Memory paths, including standalone stores later shared with Memory repositories.
Old valid custom32 keys keep their original bytes and capacity. Newly supported
custom33..768 keys remain accepted but do not gain the old compact-key capacity.
No mapping, extra SQL, fallback reader or schema version change is introduced.
Archive group UUIDs remain separate from the raw Archive scope (at most 403
bytes) and raw Native Step store scope (at most 402 bytes).

All five State table columns are MySQL `VARBINARY(768)`, PostgreSQL `BYTEA`
and SQLite `BLOB`, never hex text. Current SQL readers validate the same raw
bound. Store-qualified predicates and native owner selections reuse those
bytes without another query; native ownership edges remain existing row IDs.
The record cache's string transport uses an injective hex rendering of the
same raw address. Raw partition names are not included in mutation log context.

Store plus record or operation consumes at most 1920 index field bytes; store
plus stream and sequence 2056. These bounded identity
budgets fit MySQL's 3072-byte limit and ordinary PostgreSQL approximately
2704-byte B-tree tuples with overhead, not every PostgreSQL page configuration.
The MySQL store/kind/sort index declares 768 + 128 + 128 = 1024 field bytes
using its nonunique sort prefix. PostgreSQL indexes the complete sort bytes,
which have no application-level maximum, so that index has no fixed global
field budget. Built-in partitions contribute only 32 actual bytes. For a
2400-byte Memory path, store/kind/sort contributes 32 + 6 + 2400 = 2438 field
bytes before PostgreSQL tuple overhead; its physical limits still apply.
All indexes have at most three physical columns. Wide custom keys can exceed
the former compact partition, so this is not a claim of universal byte savings
or measured latency improvement.

## State operation addresses

`StoredOperation.key` is a binary address of 1..1152 bytes. Public repositories
can share a bare or custom StateStore, so ordinary ledger keys carry their
namespace, tenant and domain even when standard stores also use a
`store_key` partition. Built-in families reserve distinct first bytes:

| Family | Physical address | Maximum bytes |
| --- | --- | --- |
| Ordinary ledger | `01`, then namespace, tenant, domain and operation ID, each framed by its two-byte big-endian UTF-8 byte length | 661 |
| Capability receipt | `02`, then role, namespace, tenant, capability ID, revision, run ID and write key with u16 UTF-8 lengths | 1074 |
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
Capability receipts retain all original identity components under their owning
bounds, as detailed below. All streams remain globally scoped, and repository
reads verify the key, stream and payload against the bound scope.

SQL stores operation addresses directly: MySQL bounded `VARBINARY`,
PostgreSQL `BYTEA`, SQLite `BLOB`. Each unique `(store_key, address)` index
declares at most 1920 field bytes on MySQL and PostgreSQL before index tuple overhead. Binary comparison defines scan ordering on
all backends. Current snapshot wire encodes the key as canonical lowercase hex
and decodes the same bounded bytes. Native content ownership uses that binary
address to select the existing operation row ID. Tool recovery carries canonical operation identities, without a second alias index. Restore retains
the manifest namespace and tenant when moving state to another backend.

This removes ordinary physical-key JSON serialization and hashing, with no
additional SQL, mapping table, compatibility encoding or encoded sort key.
Scope is repeated to preserve public shared-store isolation. A generated
64-ASCII-byte operation ID with namespace `runtime`, tenant `tenant` and domain
`conversation` uses 98 bytes rather than the previous 64 SQL hex bytes or 32
in-memory digest bytes. No universal storage or measured runtime speedup is
implied.

## State stream and capability addresses

Fact and operation streams, pagination cursors and sequence counters use the
same exact binary `stream_key` of 1..1280 bytes. Built-in addresses encode the
flat tuple `(stream, namespace, tenant, domain, relation, *owner_parts)` with a
big-endian unsigned two-byte UTF-8 length for each component. Scope is complete
even when multiple public repositories share one bare StateStore. There is no
embedded record tuple, recursive JSON encoding, shadow mapping or hash fallback.
The tenant-wide session-list counter has an empty owner tuple. Pending transcript
positions are canonical decimal strings validated as nonnegative signed-64-bit
indices by their writer and current reader.

SQL stores `VARBINARY(1280)` / `BYTEA` / `BLOB` directly. The fact subject index is
`(stream_key, subject_id, sequence)` with at most 1280+128+8=1416 field bytes.
The operation status index is `(stream_key, state, sequence)`, at most
1280+256+8=1544 declared MySQL bytes. Existing store filters remain, and every
composite index still has at most three columns. Counter snapshots name the
same `stream_key`; canonical lowercase hex is only the portable wire encoding.
Current readers reject malformed or oversized keys without a format-version
bump or an obsolete hashed-format path. Stream fields are excluded from repr.

Capability receipts use family byte 2 followed by the exact tuple
`(capability-write, namespace, tenant, capability_id, revision, run_id, write_key)`.
The revision is canonical positive signed-64-bit decimal. Owning limits are
256 UTF-8 bytes for capability ID and write key and 128 bytes for run ID.
These retain all original parts and require at most 1074 bytes. Operation
addresses accept 1..1152 bytes; the receipt reuses its address as its stream.
Ordinary ledger addresses retain family byte 1. The fixed partition marker
retains `b"\\x00capability-partitions"` and uses that same fixed address as stream.
The receipt request SHA-256 stays because it detects conflicting replay content,
not identity. Buffer transport carries the raw hashable bytes. Cancellation,
unknown-commit readback and non-atomic partition marker semantics are unchanged.

## State record relations

`scope_key` is a nonempty binary address of at most 1024 bytes.
`parent_key` accepts at most 1152 bytes so it can also hold a complete record
reference. Generated parent relations retain their 1024-byte limit. Direct relation helpers encode a flat sequence of exact UTF-8 strings:
role (`scope` or `parent`), namespace, tenant, domain, record kind, relation,
then each logical identity component. Every component has a fixed two-byte
big-endian byte length. Scalars and nested JSON are not alternative encodings.
The role keeps relation addresses distinct from other address families. Global
scope remains in every key because public repositories can share one bare or
custom StateStore across namespaces, tenants and domains.

The complete encoded bound is checked at construction, queries and current
SQL/file readers. Namespace and tenant keep their 256- and 128-byte limits;
validated resource IDs use 128 bytes, principal IDs and local turn purposes
use 256. Task graph/node IDs are individually resource128; budget scopes
permit 256 bytes including their explicit prefixes. The complete encoded
relation still has its own 1024-byte limit. With a 384-byte
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
limit. Memory's query scope instead hashes the complete validated tuple once to
32 raw SHA-256 bytes: namespace, tenant, memory domain/kind, scope role,
persistent/transient discriminator, raw memory scope and optional execution ID.
Writers, list/capture queries and snapshot reconstruction use the same projection;
the DTO retains both raw owner fields. No inner owner hash, mapping or extra SQL
is introduced. The compact composite scope leaves more index space for full
paths: its 32 bytes do not exceed the previous 64-byte hex scope. This is a
physical query-scope tradeoff, not a second logical identity. It does not promise
the same capacity for every historical PostgreSQL row, because the separate
store/kind/sort index and PostgreSQL's real tuple limits still apply.
Reused record parents are the full binary record references described above.
ContentArchive and Native Step scopes remain raw tuples bounded to 403 and
402 bytes. State built-in partitions use the single raw32 projection described
above; custom State keys remain bounded to 768 bytes. These physical addresses
are distinct from body-integrity and anonymous-content checksums.

SQL binds raw bytes as MySQL `VARBINARY(1024)` for scopes and
`VARBINARY(1152)` for parents, PostgreSQL `BYTEA` or SQLite `BLOB`.
`(scope_key, sort_key)` and `(parent_key, sort_key)` remain two-column
indexes. MySQL indexes the complete relation and a nonunique 128-byte sort
prefix. PostgreSQL and SQLite index the complete sort value; PostgreSQL's
B-tree tuple size limit still applies to the actual indexed bytes. The generic
sort contract does not guarantee that arbitrarily long values fit that index.
`(scope_key, subject_id)` has at most 1152 field bytes. SQL still filters the
store partition; no mapping or SQL round trip is introduced. Portable wire
keeps canonical lowercase hex for bounded relation values, with the existing
format version. Snapshot readers recompute direct keys against the manifest
scope rather than the destination.

## State record ordering

`StoredRecord.sort_key`, query prefixes and pagination boundaries are nonempty
binary ordering data, not bounded resource identities. Ordinary record order
reuses the complete record key directly. Chronological order adds eight
big-endian epoch-microsecond bytes before that key. Explicit path, sequence
and trace orders retain their ASCII semantics as ASCII bytes. Memory paths
retain their 200-character ASCII segment rule with no aggregate path limit.
No hex or Base64 encoding participates in SQL ordering.

State uses `LONGBLOB` on MySQL, `BYTEA` on PostgreSQL and `BLOB` on SQLite.
MySQL's nonunique sort indexes use a 128-byte prefix without truncating stored
values or query bounds. Blob ordering remains subject to the server's
`max_sort_length`; deployments must account for that setting when ordering
long paths, as with the previous text column. The three-column sort index is
`(store_key, kind, sort_key)`, with at most 768 + 128 + 128 = 1024 declared
MySQL key bytes. Parent/sort and scope/sort indexes have at most 1280 and
1152 declared key bytes respectively. Record, operation and other identity
bounds and their full unique indexes are unchanged. PostgreSQL retains full
B-tree indexes, so its deployed page size and actual index tuple size constrain
accepted values. No unlimited PostgreSQL path support is implied by the
unbounded application-level sort contract or by DDL compilation.

Binary prefixes use a lexicographic lower bound and their least upper bound;
an all-`ff` prefix has no finite upper bound. Pagination uses `(sort, record
key)`, or strictly greater sort when the query omits an after-key. There is no
fabricated maximum record key. Record cursor transport contains a two-byte
key length, raw key and the remaining complete sort bytes, then canonical
unpadded URL-safe Base64. Only the already bounded identity uses fixed-width
length framing. Portable record JSON transports sort bytes as canonical
Base64 and returns raw bytes before comparison. The current format version
is unchanged, with no obsolete reader or fallback encoding.

## Native Step identities and digest ownership

Native `run_id`, `conversation_id` and `parent_run_id` use the resource ID
contract: nonempty UTF-8 strings of at most 128 bytes, with no leading/trailing
whitespace or Unicode control characters (`Cc`). Optional lineage remains `None`
when absent. IDs are preserved exactly, without trimming, Unicode normalization,
truncation or digest substitution. The same limit applies to writes, lookups,
Runtime projections and backup restore. Generated Runtime run/conversation IDs
are 66 ASCII bytes; native SDK UUIDs also fit this contract.

The original IDs are indexed directly. Runs are unique by `(store_key,
run_id)`; lineage indexes use `(store_key, conversation_id)` and `(store_key,
parent_run_id)`. Native `store_key` is the exact namespace/tenant tuple of at
most 402 raw bytes, not a digest. Events and snapshots retain the original
UTF-8 replay key as at most 512 binary bytes; `NULL` appends a distinct unkeyed
occurrence. Their unique key is `(store_key, run_id, idempotency_key)`.
Rejecting edge whitespace keeps run IDs exact under MySQL PAD SPACE comparisons.

SQL declares run IDs as `VARCHAR(128)`, whose width counts characters rather
than UTF-8 bytes; application validation enforces the tighter byte bound on
every backend, including SQLite. With `utf8mb4`, the raw 402-byte store plus
an ID declares 914 MySQL key bytes, or 915 with the snapshot boolean. Adding
a full replay key gives 1426 bytes. These fit InnoDB's 3072-byte large-prefix
budget, not its legacy 767-byte configuration. Provision MySQL 5.7 with the
applicable large-prefix row format/settings; this contract adds no fallback
for legacy settings. PostgreSQL's actual tuple limits still apply. DDL
compilation is distinct from validation on a deployed database server.

| Table | Address, checksum or content reference | Contract |
| --- | --- | --- |
| `ai_step_runs` | `store_key`, original run and lineage IDs | Direct scoped lookup keys, without a second run-identity hash. |
| `ai_step_events` | `store_key`, `run_id`, `idempotency_key` | Direct scoped replay identity; NULL replay keys append distinct occurrences. |
| `ai_step_events` | `body_sha256`, `fields_sha256` | Managed lifecycle source integrity. Full recovery authenticates metadata too; metadata-free Trace projections authenticate their fields without reading the large metadata body. |
| `ai_step_snapshots` | `store_key`, `run_id`, `idempotency_key` | Direct scoped replay identity. Snapshot pruning uses the original run ID. |
| `ai_step_snapshots` | `sequence_root` | The sole stored root of the immutable message sequence. The bounded descriptor stores its count and checkpoint control, not another root. |
| `ai_step_messages` | `store_key`, `sha256` | Raw scope plus anonymous-content addressing generated from original uncompressed canonical message bytes. Reads do not recompute that address digest. |
| `ai_step_sequences` | `store_key`, `sha256`, `message_sha256`, `left_sha256`, `right_sha256`, `next_sha256` | Raw scope plus immutable ordered sequence addressing and references to its content, subtrees and next forest. Repeated messages remain distinct occurrences. |

ContentArchive uses the existing StateStore tables rather than another SQL
schema. Its `raw_digest` identifies content across raw and compressed representations;
reads do not recompute it from the selected bytes. Its content ID is the exact codec plus `:` plus
that 64-character lowercase checksum, split at the final 65 ASCII characters.
Codecs retain exact UTF-8 (including colons and slashes), bounded to 64 bytes
and excluding NUL so JSON links are readable as PostgreSQL text;
`model-message:<checksum>` is 78 bytes, versus 64 for the former derived hash.
This removes one redundant hash, not storage bytes. Sequence addresses still
hash content, count and ordered links to authenticate immutable structure.

Archive scopes are the fixed-u16 tuple `("archive_scope", namespace, tenant_id)`:
at most 403 raw bytes for the existing 256/128-byte owner bounds. SQL stores
those bytes directly. Roots, references and object prefixes use one canonical,
unpadded URL-safe Base64 token (at most 538 characters). Record addresses encode
the actual namespace and tenant once, followed by archive kind and content or
sequence identity; they do not embed the transport token as a second owner.
The separately owned system `archive_scope` metadata keeps its empty binding.
There is no old digest reader or migration.

Archive group IDs retain exact UTF-8 with an explicit 1..128-byte limit. Object
keys remain `v1/archive/<group>/<scope-token>/<raw-object-checksum>`, never the
logical codec-bearing content ID. Their maximum is 743 UTF-8 bytes (647 for a
32-character UUID group), within the unchanged 1024-byte ObjectStore limit.
Restore changes the storage group only and retains the manifest namespace and
tenant. Roots and object reference/stat representations hide reversible paths.
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
archive content, alongside shared binding contracts. The control record retains
scoped content addresses; ordinary text and JSON are not ObjectStore files. Only
typed attachment slots contribute binary dependencies. The archived input also
owns its display/model view so a repeated prompt is not copied into control CAS.
Snapshots include and budget the referenced content before decoding it, rekey
physical archive scopes on restore, and verify the same typed attachment closure.
Offline collection keeps content while its Execution owner is retained.

Execution bindings retain only their fixed complete-contract reference, not a
second ID/revision/capability summary. Explicit metadata/control readers select
current owners and their required fixed configuration using bounded queries.
Full recovery still verifies and restores the admitted contract. Binding and
policy content shared within one batch is decoded and validated once per unique
owner; actual per-execution policy overrides are validated for each execution.

## Configuration and large-record ownership

- Execution Agent and Task bindings retain the complete admitted contract for
  declaration validation, cold recovery and portable snapshots. Named ID/revision
  alone does not reconstruct output schemas, resolved Agent declarations or
  captured child bindings. Identical bindings share one scoped archive content record across executions,
  including configurations below the former payload limit. Each execution keeps
  a fixed content reference without a second summary. A different complete contract
  with the same declared ID/revision remains a different immutable content value.
  Read projections select only the content their public contract requires.
- Task graph headers contain no node-definition body. Immutable node definitions
  and mutable node states are separate records. The prepared submission retains
  the initial graph until admission or cancellation, then deletes its payload;
  it is not a permanent second graph copy.
- Task graph binding captures deduplicate declarations within a graph. Nonempty
  task/expander declaration sets share content-addressed archive bytes across
  graphs regardless of size; empty sets remain inline without a content record. These are the admitted restore contract,
  while input captures and invocation captures preserve frozen input/provenance
  independently of a source execution's lifetime. They cannot be replaced by
  looking up the caller's current mutable configuration.
- Execution input, result and prepared terminal output use existing immutable
  content references at every size. The persisted codec omits repository
  instructions already owned by input context and replaces an identical
  terminal-handoff outcome with a reference to the final result. Step message
  bodies and sequence references have their own canonical owner.
- Tool operations retain a capability reference and binding/argument digests,
  not a full Tool/Skill/MCP configuration copy. Their argument/result/error
  payloads, approval requests and external-call results are effect/recovery
  facts. Memory and artifact content are user data. No safe deletion follows
  merely from these fields being large or sharing the records table.

For content-only publication, archive preparation performs an existence/body
integrity read; publication uses the existing byte-exact immutable insert/upsert
without another preceding SELECT. Reads repeated within the same SQL transaction
already reuse its record read set. The existence read also preserves the stored
compression or object representation, so constructing another encoding and
skipping verification is not an equivalent optimization. Preparations outside
publication cannot be treated as current-transaction existence proof. Sharing
small configurations adds an existence/integrity read and immutable publication
when a new execution or graph first references content. An unchanged binding
reference on hot Execution CAS is retained without republishing its body. Cold
full-record recovery still reads the content, with existing bounded batch reads
for multiple records. There is no new size threshold or claim of fewer SQL calls.
An extremely small or single-use Task binding can occupy more total bytes after
adding its reference and archive metadata; repeated substantial configurations
and hot owner rows are the intended space benefit. A regression workload counts
all record JSON and binary bytes for twenty repeated sub-64KiB Agent/Task bindings,
including the archive row, rather than measuring only the shortened owner JSON.
Its runtime results remain subject to CI; no production storage saving is claimed.
Accepted execution/tool payload content physically reuses `ai_step_messages`.
The `agent_run` and `tool_operation` logical interfaces route to their Step
owners, not generic record rows. Other archive families retain their existing
owner; moving rows alone is not evidence of a smaller whole database.

## Cost boundaries

Removing duplicate content and Base64 database bodies reduces those specific
writes and stored bytes. It does not prove fewer database operations in every
workload: timeline pages need an Execution projection read, offline collection
scans complete owner inventories under caller-enforced exclusion. There is no
online dependency-edge table or collection queue. Real Asset files retain
their ObjectStore cost. Compare matched workloads by reads, writes, transactions,
scanned rows, returned bytes and retained bytes before claiming an overall gain.

See [native boundary details](atomic-native-step-boundaries.md) for append,
recovery, collection and extension limits, and [public entry points](runtime-entrypoints.md)
for the Runtime handle and explicit-ID APIs.


## Requested storage simplification coverage

| Requirement | Implementation or retained owner | Verification boundary |
| --- | --- | --- |
| Trace events outside generic facts | `ai_trace_events`; native body references and bounded identity lookup | Existing Trace/source/recovery regressions; real-backend plans still require validation |
| Remove duplicate complete run metadata | Canonical `ai_step_runs` or non-SQL run registration; domain membership/CAS only | Routing, publication and portable snapshot coverage; no automatic shared-owner deletion |
| Domain-specific names | Tool calls use canonical operation IDs, shared `ai_state_counters`; `ai_step_runs` unchanged | Metadata/DDL/readers/writers and schema tests updated together |
| Execution Agent/Task configuration duplication | All sharing-enabled bindings use the existing complete-contract content archive | Repeated small/large configurations, changed same-revision contracts, cold recovery and CAS regressions |
| Other configuration duplication | Nonempty Task/Expander declaration sets share existing archive bytes; graph definitions already separate from node state | Small/large declaration load and collection regressions; empty sets create no archive |
| Hot state query cost | Narrow control/membership/result projections and bounded owner-content queries; no copied binding summary | Existing authorization, cache and query-budget regressions |
| Input/result/message bodies | All-size fixed payload owners; exact Native part/event reuse, preserving receipt format and real file references | Byte-exact restoration and owner-lifetime closure; non-equivalent accepted facts are retained |
| No historical baggage | Current writer/reader/schema evolve together; no legacy migration or new config registry | Current-shape round trips and explicit malformed-data rejection |
| Space and I/O trade-off | Reuse removes repeated contract bodies and shrinks hot rows; initial publication and cold recovery still cost reads | Matched SQLite synthetic measurements below; formal regression gates and MySQL/PostgreSQL latency remain separate |


## Matched SQLite measurement (2026-11-12)

This is a synthetic workload, not a production database or a complete inventory
of every records kind. Fresh databases each contain 20 Agent executions, 20 Task
executions, and 20 graph binding captures using repeated approximately 1 KiB
configuration. Each execution receives two hot CAS updates, two warm views, two
cold views after reopening, and one cold full read. Each capture receives one
cold load. No providers, tools, or formal test suite were run.

The four columns below are baseline `c99c687d`, sharing `a9fa3e38` (measured
identical local tree at `20418a62`), and capture transaction optimization
`d2be6c58`, followed by final measured code head
`63defe409822c727a6807e73fc43c03d7f04eb1c` (tree
`a98b781388b4fb5d20f1ddb46cf1c034eb624d78`). The final code includes execution
metadata/result projections and the first-writer archive-scope race fix. Later
evidence-only documentation changes do not change that measured code tree.

The standalone diagnostic [helper](../../tests/ai/_measure_shared_records.py) selects each source
with PYTHONPATH, checks its exact clean Git revision, and records imported module
paths. Script SHA-256:
`087f9bfb1044b458cea5f496cf76bc607d9698b2ed165ffe1a7f56ef4737ffce`.
Input SHA-256:
`db06aa6d773aad468007229c3e347f67fdf5ebb7100b41153a5e5aebd99b54a7`.
The committed [compact evidence](measurements/shared-records-sqlite-summary.json)
contains each of the four source SHA/tree pairs, input, dependencies, per-phase SQL counters,
all-column and per-kind records bytes, and final physical sizes. Source/import
paths are checkout-relative. Full intermediate table snapshots and elapsed
measurements are omitted; original raw-result hashes are retained for provenance.
The annotated helper was rerun against the first three clean source revisions: inputs,
SQL counters and every storage snapshot match the preceding standalone run.
The fourth run uses the identical helper and input on the final code head.

From a checkout with the recorded dependencies installed, invoke the helper once
per clean source tree with a fresh output/database path:

```bash
python tests/ai/_measure_shared_records.py \
  --source-root <clean-source-checkout> --expected-sha <source-sha> \
  --database <new-database.sqlite> --output <new-result.json> --label <label>
```

This is a manually invoked diagnostic, not a task runner or collected test.
It rejects mismatched or dirty source revisions and existing output files.
All four use the same interpreter/dependencies,
including SQLAlchemy 2.0.51, aiosqlite 0.22.1, and SQLite 3.53.1.

### Retained space

“All column bytes” sums `length(CAST(column AS BLOB))` for every SQL column,
including keys, timestamps and binary payloads, with NULL counted as zero. Numeric
values therefore use their text rendering, not SQLite's on-disk integer encoding.
This is a reproducible value-size metric, separate from physical `dbstat` pages.

| Record kind | Rows: baseline / sharing / capture / final | All column bytes: baseline / sharing / capture / final |
| --- | --- | --- |
| `execution` | 40 / 40 / 40 / 40 | 150012 / 104733 / 104733 / 109473 |
| `task_graph_binding_capture` | 20 / 20 / 20 / 20 | 45140 / 25360 / 25360 / 25360 |
| `archive_content` | 0 / 3 / 3 / 3 | 0 / 6216 / 6216 / 6216 |
| `archive_scope` | 1 / 1 / 1 / 1 | 281 / 281 / 281 / 281 |

| Total metric | Baseline | Sharing | Capture optimized | Final code |
| --- | ---: | ---: | ---: | ---: |
| Records all column bytes | 195433 | 136590 | 136590 | 141330 |
| Records JSON bytes | 170901 | 106376 | 106376 | 111116 |
| Records binary bytes | 0 | 4020 | 4020 | 4020 |
| Records table + indexes pages | 77 | 67 | 67 | 67 |
| Records table + indexes page bytes | 315392 | 274432 | 274432 | 274432 |
| Records table + indexes payload bytes | 223728 | 166682 | 166682 | 171422 |
| Closed database file bytes | 610304 | 569344 | 569344 | 569344 |

Pages are 4096 bytes. Closed WAL/SHM sizes are zero; no VACUUM was requested.
Other business tables have no rows in this workload, so all-business-table value
bytes equal the records total. Shared archive rows are included: records values
fall 27.7% at the final code head, records table/index allocation 13.0%, and
the full SQLite file 6.7%. The final metadata summary adds 4740 bytes to the
40 execution rows compared with the capture-only measurement.
These results do not establish savings for tiny single-use contracts, other
record mixtures, pre-existing databases, or other SQL backends.

### SQL operations and transaction boundaries

Entries below are SELECT / DML / commit events. Counts use SQLAlchemy events;
DML includes immutable INSERT/upserts even when the body already exists. They
are not network round trips, fsync counts, scanned bytes, or latency. Measurement
inspection itself uses a separate read-only SQLite connection and is excluded.

| Phase | Baseline | Sharing | Capture optimized | Final code |
| --- | --- | --- | --- | --- |
| initialize | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 | 0 / 0 / 1 |
| agent_create | 22 / 21 / 20 | 45 / 41 / 23 | 45 / 41 / 23 | 46 / 41 / 23 |
| task_create | 20 / 20 / 20 | 40 / 40 / 20 | 40 / 40 / 20 | 40 / 40 / 20 |
| capture_create | 20 / 20 / 40 | 100 / 40 / 140 | 40 / 40 / 40 | 40 / 40 / 40 |
| agent_hot_cas | 40 / 40 / 40 | 40 / 40 / 40 | 40 / 40 / 40 | 40 / 40 / 40 |
| agent_warm_view | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 |
| task_hot_cas | 40 / 40 / 40 | 40 / 40 / 40 | 40 / 40 / 40 | 40 / 40 / 40 |
| task_warm_view | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 |
| reopen | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| agent_cold_view | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 |
| task_cold_view | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 |
| agent_cold_full_read | 20 / 0 / 20 | 41 / 0 / 41 | 41 / 0 / 41 | 41 / 0 / 41 |
| task_cold_full_read | 20 / 0 / 20 | 40 / 0 / 40 | 40 / 0 / 40 | 40 / 0 / 40 |
| capture_cold_read | 20 / 0 / 20 | 60 / 0 / 60 | 40 / 0 / 20 | 40 / 0 / 20 |
| All phases | 362 / 141 / 381 | 566 / 201 / 565 | 486 / 201 / 425 | 487 / 201 / 425 |

Initialization separately executes 88 statements (30 PRAGMA, 58 CREATE), with
one commit and one rollback in each run. Reopening executes two PRAGMAs and one
rollback. All-phase statement totals, including BEGIN/DDL/PRAGMA, are
973 / 1421 / 1201 / 1202. The final first-writer identity check adds one
startup SELECT; hot CAS and warm/cold view counts are unchanged.

Capture publication now retains a read-only existing-owner replay, then publishes
new immutable content and its owner within the existing same-owner mutation
scope. Owner and contract recovery share one read scope; an already opened
archive is reused without another scope lookup. This reduces capture creation
from 100 SELECT/140 commits to 40/40, and cold loading from 60/60 to 40/20.
Atomic rollback and read-scope regression coverage accompany the change.
Compared with the inline baseline, the remaining content-integrity read and
immutable publication still add work: all-phase SELECT rises from 362 to 487
and DML from 141 to 201. This is a measured space/read trade-off, not an overall
SQL reduction. The final code head is measured here; formal CI and complete runtime workloads
remain separate verification requirements.

### Fixed content reads

Fixed content addresses select Runtime binding, policy, payload, capture and
Native/history message bytes. Reading or restoring those bytes does not recompute
their address hash. Valid changed bytes at the same address can therefore be
accepted, including through snapshot restoration. Declared sizes, bounded
decompression, codecs, schemas, types, source existence and namespace/tenant/scope
checks still apply. Business identity and idempotency checks, lifecycle/source
associations, and independent Artifact/ObjectRef integrity contracts remain in
force. Writers still generate content addresses from the original bytes and
reject conflicting publication at an existing address.
