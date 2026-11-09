# SQL state payload storage

The SQL StateStore stores `ai_state_records.payload_json`,
`ai_state_facts.payload_json`, and `ai_state_operations.payload_json` as canonical
JSON **text**. MySQL uses `LONGTEXT CHARACTER SET utf8mb4 COLLATE utf8mb4_bin`;
SQLite and PostgreSQL use their text types.

Native database JSON representations may change the lexical representation of
numbers. For example, turning `1.0` into `1` loses the distinction required by
typed domain readers and by generic JSON payload hashes. Casting the resulting
JSON back to text cannot restore information already lost. The StateStore
therefore serializes the original domain value before SQL binding and decodes
the returned text itself. Integers, floats, signed zero, booleans and null retain
their existing semantics. Domain validation, semantic hashes and idempotency
keys are unchanged.

Archive traversal extracts string links from the text inside SQL. These
expressions locate rows only; complete payloads are read as text and never
round-trip through a server JSON conversion. Compare-and-swap revisions, leases,
fences and transaction boundaries continue to use the existing state columns.
Snapshots retain their logical record/fact/operation representation and restore
through the same StateStore writer.

## Provisioning

Canonical metadata and `migrations/init_schema.sql` describe the required new
schema. Provision shared schemas explicitly through the deployment process.
Runtime startup does not ALTER existing tables. A previous native JSON column
is not the required text schema and fails schema validation. This change does
not reconstruct numeric distinctions lost in previous development data.

MySQL deployment checks must verify **LONGTEXT**, not a 64 KiB TEXT column.
The generic startup validator checks the text type family, not every backend
capacity detail. Verify the deployed column definitions separately.

## Acceptance

Use newly created data with the same SDK version for write, read, shutdown,
reopen and recovery. Check dataset/case publication, evaluation execution,
scoring, report/export, duplicate requests, cancellation and unknown-commit
readback. Include integral floats, fractional values, signed zero, integers,
booleans, null, nested JSON and payloads larger than 64 KiB. Check all three
StateStore primitive families, archive traversal, batch CAS and snapshot
restore. SQLite and compilation tests do not replace execution through the
actual MySQL server and driver used by a deployment.

Independent Asset and Observe stores have their own SQL payload contracts;
this document describes only the Runtime StateStore tables.
