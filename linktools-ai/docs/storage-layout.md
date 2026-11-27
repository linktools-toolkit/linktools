# Runtime storage layout

Runtime stores ordinary text and structured state in the database. Size alone
is not a reason to put a prompt, result, capture, declaration or conversation in
ObjectStore. Real files and typed binary attachments use ObjectStore references.
Asset versions, transcript attachment references and frozen input captures keep
their own retention semantics; changing physical placement does not remove them.

## Body encoding

Native message bodies use `ai_messages.message` as raw or zlib-compressed
bytes. `compression` identifies the codec and `raw_size` bounds decompression.
The existing canonical message SHA-256 remains the content identity. There is
no second stored-body digest. Typed attachments retain their manifest in
`payload_json`; only the attachment bytes go to ObjectStore.

Trace HTTP request/response bodies use the same existing content archive and
preserve their exact original bytes, including non-JSON or already encoded
bodies. They are not reserialized as model messages. Metadata-only reads do not
hydrate these bodies.

Accepted execution and tool payload bytes use the existing `ai_messages`
content owner in a separate typed partition. Other archive families use
`ai_state_records.payload_bytes` with content metadata in `payload_json`; ordinary
control payloads are stored as native JSON objects. Keeping body bytes separate
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
| `ai_sessions` | Ordinary conversation lifecycle, occupancy and committed history position. The host owns business access policy; agent selection is per turn. |
| `ai_session_turns` | Ordinary turn request, generation and terminal transcript. History reads the committed rows in sequence. |

Session turns reference the existing content-addressed `execution-binding` archive. Compiled Agent bindings are shared within the same archive scope, not copied into turn rows.
| `ai_state_records` | Generic domain control state, including TaskGraph headers, node definitions, and node state. Graph recovery uses a bounded candidate range in the existing scan index. Archive records can own binary bodies. |
| `ai_execution_records` | Strict execution lifecycle, exact request routes, coordination receipts, lease and content references. In split storage the Execution group also owns the strict session's occupied route. |
| `ai_state_facts` | Ordered fact payloads and native references; transcript bodies stay with their archive or native owner. |
| `ai_state_counters` | Monotonic counters shared by execution, task, conversation and trace domains; not message sequences. |
| `ai_state_operations` | Ordered replay/control receipts; no separate compressed copy. |
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
| `ai_messages` | Deduplicated Native and accepted execution/tool payload bytes in disjoint typed partitions; attachment manifests retain file references. |
| `ai_step_run_admissions` | Domain membership and actual admitted projection differences, referencing canonical run registration. |
| `ai_step_tool_operations` | Authoritative tool effect state, fencing and stable operation identity; payloads reference fixed content owners. |
| `ai_trace_events` | Trace correlation and event fields or fixed Native event sources. |
| `ai_message_sequences` | Immutable message-sequence links and counts, not copied bodies. |
| `ai_step_snapshots` | Bounded descriptor and sequence root; the descriptor is not the whole conversation. |

For the six Runtime payload columns that store native JSON objects, see
[SQL state payload storage](sql-payload-storage.md).

Identity, sequence, state, lease, CAS, revision, ordering and time columns are not
compressed. Snapshot pruning uses the `(store_locator, run_id, pruned)` index.
The indexed locator narrows the candidates; the query still checks the original
`store_key` for scope correctness.

## State keys, lookup projections and ordering

| Value | Current application and SQL contract |
| --- | --- |
| `store_key` | 1–128 opaque bytes; built-in State partitions derive 32 bytes from the complete framed scope. This is not the Native StepStore partition format. |
| Record identity, scope and parent keys | 1–768 bytes. Built-in record/relation keys use exact length-prefixed UTF-8 components, including namespace and tenant. |
| Fact and operation streams, counters and operation addresses | 1–768 bytes. Streams include the full logical scope; the ordinary operation and capability-receipt families encode their own distinct inputs. |
| `subject_id` | The exact resource ID as UTF-8, at most 128 bytes. No subject digest. |
| `sort_key` | The complete nonempty binary sort token; it is not bounded to the identity width. `sort_prefix` holds its first 128 bytes for indexed filtering and ordering; ties still use the full token. |

Namespace and tenant limits are 128 UTF-8 bytes each. Generic turn StateStore
fallback addresses permit at most 265/257 UTF-8 bytes for the encoded
session/request identities, respectively. The local turn scope is at most 128
UTF-8 bytes. Built-in Runtime SQL uses typed turn slots and admissions with
separate slot and route kinds, not generic turn record addresses.

SQL indexes use compact `key_locator`, `scope_locator`, `parent_locator`,
`stream_locator`, `owner_locator` and `operation_locator` projections according
to the owning query. These are SHA-256 lookup projections of exact raw keys,
not replacement durable identities. Indexed queries narrow candidates by the
locator and still compare the original key. Fact lookup by subject uses
`(stream_locator, subject_id, sequence)`; operation lookup by state uses
`(stream_locator, state, sequence)`. Consult the
[canonical schema](../migrations/init_schema.sql) for each typed owner's
distinct indexes; they do not all share the same access paths.

Record cursor ordering is by the complete `(sort_key, key)`. Timestamp sorts
prepend eight bytes to the record key. Binary prefix scans use a lower bound and
the least upper bound, if one exists. Cursor or portable-snapshot transport can
encode binary keys; the backend decodes those representations before comparing
raw bytes.

Capability receipts retain the exact namespace, tenant, capability ID, revision,
run ID and write key; a request digest detects conflicting replay content and is
not the receipt identity. Named behavior remains identified by its explicit
kind, ID and revision.

## Native Step identities and digest ownership

Native run and lineage IDs keep their original resource IDs (at most 128
UTF-8 bytes). The exact `store_key` encodes the framed namespace and tenant
tuple and is at most 274 bytes. SQL queries use the indexed `store_locator`
alongside the exact scope; they do not hash the run ID.

Events and snapshots use the unique key
`(store_locator, run_id, idempotency_key)`. The SQL replay key is non-null:
a supplied key preserves its exact UTF-8 bytes (up to 128, without NUL);
an append with no replay key stores a distinct reserved occurrence key, which
decodes back to an absent public key. Snapshot pruning retains a keyed replay
tombstone and changes neither the run nor replay identity. See the
[backup boundary](storage-caching.md#native-stepstore-backup-boundary) for what
portable restore preserves.

| Owner | Content digest or reference |
| --- | --- |
| `ai_step_events` | `body_sha256` authenticates the full managed lifecycle source; `fields_sha256` authenticates its metadata-free projection. |
| `ai_step_snapshots` | `sequence_root` points at the immutable sequence; the bounded descriptor does not repeat the message bodies. |
| `ai_messages` | The canonical uncompressed body digest identifies immutable bytes within the owning typed partition. |
| `ai_message_sequences` | The structural digest identifies the ordered tree and its content/sequence links; repeated messages remain distinct occurrences. |

ContentArchive uses an explicit codec and raw-content digest for content
identity. Its logical references retain namespace and tenant; a portable
restore can relocate the physical group while preserving those logical owners.
Ordinary archived bytes are in SQL; typed attachments retain ObjectStore
references and their byte-integrity checks. These are content-address and
integrity contracts, not named behavior identities.

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
Accepted execution/tool payload content physically reuses `ai_messages`.
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


## Historical matched SQLite measurement (2026-11-12)

These results describe the pinned historical revisions below, not a benchmark
of the current schema.

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
