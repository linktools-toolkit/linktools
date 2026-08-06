# linktools-ai

`linktools-ai` is the V8 Harness integration layer. Its public surface is the
`Runtime` Protocol and the typed Domain/Port contracts under
`linktools.ai.domain` and `linktools.ai.ports`.

The Baseline profiles are:

- `production-service`: Temporal and explicitly constructed domain stores over
  the generic async Storage kernel.
- `local-coding`: Local File/SQLite storage and a trusted local executor.

`production-sandboxed` is deliberately blocked until the installed official
Harness wheel contains and validates the Modal capability. It is not emulated
by a Linktools wrapper.

The generic Storage kernel exposes async `build_storage()` and
`build_sqlite_storage()` builders. Construction is lazy; callers explicitly
invoke `initialize_storage()` for DDL. Storage remains domain-independent and
does not own migrations, repositories or runtime state.

See `.docs/linktools-ai-harness-final-spec-v8.md` and the checked-in
traceability matrix for the release and verification contract.
