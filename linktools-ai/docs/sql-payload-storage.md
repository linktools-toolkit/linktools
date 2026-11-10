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
