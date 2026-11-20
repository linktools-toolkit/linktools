# SQL state payload storage

The SQL StateStore stores `ai_state_records.payload_json`,
`ai_state_facts.payload_json`, and `ai_state_operations.payload_json` in **JSON
columns whose values are JSON strings**. Each string contains the canonical JSON
text of the domain payload. Canonical SQLAlchemy metadata declares `JSON` for
MySQL, PostgreSQL and SQLite; the MySQL `migrations/init_schema.sql` also declares
`JSON`, not `TEXT` or `LONGTEXT`. SQLite's underlying storage does not change
this declared schema or the encoding contract.

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

## SQL link extraction

Canonical execution lifecycle and execution-domain idempotency rows share
`ai_execution_records`, so they can still be inserted or updated in one batch.
Canonical tool rows remain in `ai_step_tool_operations`. Opaque or unrelated
records remain in `ai_state_records`, without built-in content lookup columns.
Typed readers use the owning table directly; generic kind queries retain both
physical sources.

Execution and tool-history content captures use fixed same-row lookup
projections. Nullable SHA-256 columns are derived from the canonical record
in the same insert or compare-and-swap as its payload, without a second write
or another source of truth. This lets owner rows, archived content and eligible
native message suffixes share one SQL read without extracting those references
from JSON on the server. Canonical decoding and scope/integrity checks still
apply; other native message ranges use the existing range reader.

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

Before opening a populated store with this schema, move canonical execution
and execution-domain receipt rows into their new owner and derive the content
lookup columns as part of the explicit cutover. Preserve the exact encoded
store and record keys, payloads, revision/lease state and timestamps. Leave
opaque custom rows and unrelated domains in the original table. Do not mix
writers that use different physical owners. For a local SQLite store, quiesce writers, export a portable
snapshot with the old runtime, and restore it into a fresh root with the new
runtime. Restore derives the columns from the canonical records; the snapshot
wire format is unchanged. Verify the restored store before switching over.

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
