# Explicit SQLite schema validation versions

`provision_database(context)` can reuse a successful complete-schema validation
when `context` is a `SqlStorageContext` holding the dedicated local SQLite writer
permission. The permission is acquired through the existing local-file ownership
API; owning an engine or observing an `application_id` alone is not permission.

The complete schema's manually maintained revision is stored in that owned
file's `PRAGMA main.user_version`. Equal versions skip table creation and schema
reflection. A different version runs complete creation and validation on the same
connection, then records the current revision in the same mutation. Failure does
not record success. Existing mutation cancellation, rollback and SQLite retry
behavior remains in force; there is no new lock or migration framework.

This optimization deliberately does **not** detect external schema edits while
the version remains unchanged. Use explicit `validate_sql` for diagnostics.
Schema changes must increment the complete-schema revision in
`migrate/_database.py`; the revision is separate from the persisted data version.

Bare engines, shared contexts, unknown ownership and partial-schema provisioning
never read or overwrite the caller's `user_version`. They do not acquire the
owned fast path. Ordinary Runtime initialization still performs no mandatory
schema reflection. MySQL and PostgreSQL provisioning creates requested metadata
without a default post-validation pass; explicit `validate_sql` remains available.

No table, column or index is added by this feature. The existing
`application_id` ownership profile is unchanged.
