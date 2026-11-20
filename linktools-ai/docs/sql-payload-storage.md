# SQL state payload storage

These Runtime StateStore tables use a `payload_json` column whose **SQL
type is JSON but whose value is a JSON string**: `ai_state_records`,
`ai_execution_records`, `ai_task_graphs`, `ai_step_tool_operations`,
`ai_step_run_admissions`, `ai_state_facts` and `ai_state_operations`.

Each string contains the canonical JSON text of the domain payload. The SQLAlchemy
metadata and the [MySQL schema](../migrations/init_schema.sql) declare these
columns as JSON, not TEXT. SQLite's underlying storage does not change the
declared schema or the encoding contract.

## Encoding and reading

There are two JSON layers:

1. The StateStore writer serializes the domain mapping with
   `canonical_json_bytes(...).decode("utf-8")`.
2. SQLAlchemy's JSON binding encodes that Python string as a JSON string value
   in the column. For example, the domain payload `{"value": 1.0}` has inner
   canonical text `{"value":1.0}` and outer JSON representation
   `"{\"value\":1.0}"`.

On a normal typed SQLAlchemy read, the JSON result processing removes the outer
layer and returns the inner text as a Python `str`. The StateStore then decodes
that text and requires a mapping. A raw SQL read may still expose the outer JSON
representation; it is not the same as the typed reader's input.

Do not write a native JSON object directly into these columns. The reader rejects
a decoded mapping in place of the required string, as well as invalid inner JSON
or an inner value that is not a mapping. Column type validation alone does not
validate this row-level encoding.

Native database JSON processing of a payload object may change the lexical
representation of numbers. For example, turning `1.0` into `1` loses the
distinction required by typed domain readers and generic JSON payload hashes.
Keeping the payload as text **inside** the JSON string prevents the outer JSON
layer from interpreting its numbers. Integers, floats, signed zero, booleans and
null retain their existing semantics. Domain validation, semantic hashes and
idempotency keys are unchanged.

The outer SQL JSON string is the current storage representation, not what makes
the inner text canonical. A text column could preserve the same canonical bytes
without the outer encoding. It would still need an inner JSON decode when reading
the domain record, and a schema change would need to update the typed bindings
and the archive and native-reference queries below. This does not apply to real
JSON objects or arrays elsewhere that are queried as structured data.

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

Archive traversal and native run/event reference queries unwrap the outer JSON
string before extracting fields from the inner payload. The SQLAlchemy
expressions first select the JSON root as a string, then access fields as JSON:

- SQLite uses JSON extraction to unwrap the root and extract the inner fields.
- MySQL uses JSON extraction/unquoting to unwrap the root and read string fields.
- PostgreSQL extracts the root as text and explicitly casts that text to JSON
  before accessing inner fields.

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

The schema validator requires the JSON type family and JSON compatibility for
these columns. `TEXT`/`LONGTEXT` columns do not satisfy it, and PostgreSQL `JSONB`
is not interchangeable with the declared `JSON`. A correctly declared JSON
column containing native payload objects can pass schema validation but still
fail payload reads. Verify both schema and newly written data through the
supported StateStore path.

There is no compatibility reader or automatic migration for superseded
development payload shapes. This contract does not reconstruct numeric
distinctions already lost in old data. Use the current schema and writer for
new data rather than restoring the old text-column or native-object contract.

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
