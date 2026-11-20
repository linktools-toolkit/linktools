# SQL state payload storage

These Runtime StateStore tables store canonical JSON text directly in their
`payload_json` columns: `ai_state_records`, `ai_execution_records`,
`ai_task_graphs`, `ai_step_tool_operations`, `ai_step_run_admissions`,
`ai_state_facts` and `ai_state_operations`. The SQL type is `LONGTEXT` on MySQL
and `TEXT` on PostgreSQL and SQLite. These columns are not native JSON objects.

## Encoding and reading

The StateStore serializes each domain mapping once to canonical UTF-8 JSON
text, binds that text as a SQL string, and decodes it once on read. The
database stores `{"value":1.0}` directly, not a quoted and escaped JSON string
containing that text. This preserves the numeric spelling required by typed
readers and hashes, including `1.0` versus `1` and signed zero, without the
outer serialization. The StateStore rejects invalid JSON and non-mapping
roots on read.

True structured JSON columns, such as the native Step attachment manifest and
Trace event metadata, retain their own representations. Raw message bytes are
not converted to JSON text.

## SQL link extraction

Canonical execution lifecycle and execution-domain idempotency rows share
`ai_execution_records`, so they can still be inserted or updated in one batch.
Canonical tool rows remain in `ai_step_tool_operations`. Opaque or unrelated
records remain in `ai_state_records`, without built-in content lookup columns.
Typed readers use the owning table directly; generic kind queries retain both
physical sources.

Execution content reads select fixed same-row references for admitted policy,
binding, user input, result template and output, including the indexed native
tail alternatives for input and output. For these execution slots the SHA-256
is stored once in its nullable SQL column, not repeated in the JSON. The
remaining JSON descriptor retains scope, range and projection semantics; the
StateStore reconstructs the complete logical record before typed decoding,
archive validation and export. Failure details, deferred recovery and prepared
handoff references remain in their complete JSON form and load in the same
read scope when needed. Tool-operation read projections retain their own
existing columns on the tool owner.

Fixed slots let an owner, its immutable bodies and eligible native suffixes
share one SQL read without database-side JSON extraction on the normal read
path. Full scope and integrity checks still apply; other native message ranges
use the exact range reader.

Archive traversal and native run/event reference queries extract fields from the
stored text only when evaluating those links. SQLite and MySQL accept the
canonical text directly as JSON input; PostgreSQL casts to JSON inside the
extraction expression. The normal typed payload read receives the original
text, not a database-reserialized object.

These expressions resolve links and references, not replacement payloads. Full
StateStore payloads still pass through the typed read and inner-text decoder;
they are not reconstructed from server-parsed objects. Compare-and-swap
revisions, leases, fences and transaction boundaries continue to use the
existing state columns. Snapshots retain their logical record/fact/operation
representation and restore through the same StateStore writer.

## Provisioning

Canonical metadata and `migrations/init_schema.sql` describe the required new
schema. Provision shared schemas explicitly through the deployment process.
Runtime startup does not ALTER existing tables; the local SQLite state backend
may initialize its own schema.

Pre-release physical layouts and exported snapshots that do not satisfy the
current v1 wire contract are not supported. Use fresh storage for this
contract; there is no compatibility reader, startup conversion or migration
obligation for development data.

A snapshot exported by a runtime using the current v1 logical contract can
restore into a fresh local root. Export from a quiescent source and verify
the restored state before switching over. Restore packs the reference
columns from logical records; these SQL details are not part of the portable
wire representation.

For built-in SQL Runtime turn admissions, offline restore preserves the execution
and request identities, fencing generation and stored operation proof, but expires
any imported active grant. The old grant cannot renew or retire, and its prior
renewal receipt no longer confirms a live lease. A completed release remains
confirmable; restore does not synthesize a release. Normal takeover can continue
the same execution with the next fencing generation.

A future migration of a populated, released v1 store would require an
explicit cutover coordinated with its deployment. Preserve exact encoded keys,
logical payloads, revision and lease state, and timestamps; preserve opaque
custom rows in their owning store. Do not mix writers that use different
physical owners. The v1 schema does not provide an in-place migrator.

The schema validator expects text for these fields and requires `LONGTEXT`
on MySQL so payloads larger than 64 KiB continue to fit. Do not provision a
narrower `TEXT` or `VARCHAR` column there. Existing databases that used JSON
strings need fresh storage or an explicit deployment migration. To retain
those rows, unwrap exactly one JSON string layer while copying them; altering
the SQL column type alone would keep the old quotes and escapes. Verify
the copied mappings and their canonical text before switching writers. Runtime
does not migrate or silently read both physical shapes.

## Upgrading a released schema

Keep released logical identities, canonical content references, and the decoding
rules needed to read retained data. Derived SQL projections may be rebuilt from
the original references. A change to storage ownership or encoding must include
an explicit, restartable migration and verification against data written by the
previous supported release. If old and new application instances share storage
during a rollout, keep the expansion compatible with both; switch write
authority at a single cutover. Do not claim lossless rollback after the new
owner accepts writes without a verified way to preserve those writes.

## Acceptance

Use newly created data with the same SDK version for write, read, shutdown,
reopen and recovery. Check dataset/case publication, evaluation execution,
scoring, report/export, duplicate requests, cancellation and unknown-commit
readback. Include integral floats, fractional values, signed zero, integers,
booleans, null, nested JSON and payloads larger than 64 KiB. Check all three
StateStore primitive families, archive traversal, batch CAS and snapshot
restore. SQLite and compilation tests do not replace execution through the
actual MySQL or PostgreSQL server and driver used by a deployment.

Independent Asset and Observe stores have their own SQL payload contracts;
this document describes only the Runtime StateStore tables.
