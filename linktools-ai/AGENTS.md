# AGENTS.md (linktools-ai)

Package instructions for `linktools-ai`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

`Required Rules` contain only long-lived constraints. Current package layout, class names, construction paths, and other implementation details belong in `Guidance` or the code itself.

## Required Rules

### Architecture

- Use public package APIs across boundaries. LinkTools package/module exported symbols are defined by static `__all__`; symbols not exported there are not public across owner boundaries. Do not import another package's private modules/members or use reflection to bypass an interface.
- Runtime dependencies must remain acyclic; annotation-only back-references belong under `TYPE_CHECKING`.
- Keep lower-level infrastructure independent from higher-level composition and SDK semantics. Do not introduce duplicate abstractions that compete for the same ownership.
- Keep vendor-specific behavior out of vendor-neutral core abstractions.
- Architecture and release gates encode long-lived invariants only. Do not freeze current package names, module depth, class names, or layout as policy.
- Build/release tooling must not become a second owner of Runtime semantic truth.

### Durable contracts and identity

- Runtime startup must not implicitly create or migrate database schemas; schema provisioning is an explicit deployment/migration operation. A local SQLite state backend is the explicit exception and may initialize its own local schema when that state store is created or opened.
- Durable wire formats and semantic identities are explicit LinkTools contracts. Honor published or explicitly committed compatibility obligations. Without such an obligation, remove obsolete pre-release readers, aliases, defaults and migrations while updating current writers, readers and verification together. Do not prebuild compatibility paths for hypothetical versions.
- Define each semantic or idempotency digest from one explicit minimal projection owned by the contract. Include only inputs needed to distinguish execution meaning, accepted request behavior or safe reuse/recovery. Non-semantic additions and default fields must not change that identity; whole-object reflection, incidental wire payloads and dependency serialization must not define it.
- Separate semantic identity from byte integrity and storage addressing. Physical locators, credentials, transport tuning and pure display/diagnostic data do not enter semantic identity. Complete stored bytes still require complete integrity checks. Preserve logical scope, effect policy, model-visible instructions/schema and provenance when the specific contract needs them.
- Use stable protocol discriminators, not Python class/module names or dependency/build versions. Normalize only equivalences established by the owning contract; preserve ordered inputs and effective business parameters. Identity-affecting data must remain stable after its contract is frozen, and identity comparisons must use the same projection as the digest.
- Every current writer output must be accepted by its matching reader. Define omitted optional fields in that wire contract, not through changing runtime defaults. Reject corrupt or unsupported durable data with typed errors; never guess, silently repair or reinterpret unknown execution semantics. Verify original stored bytes before adapting decoded values.

### Persistence and concurrency

- Runtime startup must not implicitly create or migrate database schemas; schema provisioning is an explicit deployment/migration operation. A local SQLite state backend is the explicit exception and may initialize its own local schema when that state store is created or opened.
- A semantic fact must have one durable owner. Any persisted duplicate used as an index, projection, or cache must be explicitly derived and must not become an independent source of truth or define conflicting recovery semantics.
- Caller cancellation does not determine durable truth. Resolve commit/readback state before reporting an unknown outcome.
- Filesystem coordination uses `filelock`. Database concurrency must avoid pessimistic locking.

### SQL schema policy

- Business tables use the `ai_` prefix, an `id BIGINT AUTO_INCREMENT` surrogate primary key, business comments, and `updated_at` immediately before `created_at`.
- Audit columns use the fixed definitions below and every table keeps `ix_updated_at` and `ix_created_at`.
- Unique/index names use `uk_<ordered_columns>` / `ix_<ordered_columns>`; MySQL index names do not include table names and composite keys contain at most three physical columns.
- SHA-256 values use `CHAR(64)` with `utf8mb4_bin`. Wide unique columns do not use prefix unique keys; ordinary wide indexes use a 128-character prefix.
- Redundant indexes, low-selectivity status-only indexes, foreign keys, and `FLOAT` columns are prohibited. `JSON`/`LONGBLOB` are used only where genuinely required.
- DDL and canonical metadata must describe the same schema; there is no schema manifest table.

```sql
updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    COMMENT 'Creation timestamp'
```

### Durable external effects

- Durable recovery must not infer external-effect success from process lifetime or caller outcome. External effects that can be retried or recovered must have explicit ownership and idempotency/replay-safety semantics.
- Define lifetime and recovery semantics before persisting external provider IDs, URLs, handles, tokens, or similar references.

## Guidance

### Package responsibilities

These describe the current architecture and may evolve; they are not rules by themselves.

| Package | Current responsibility |
| --- | --- |
| `core` | Pure values, IDs, JSON, paging, principals, canonical hashing |
| `storage` | Generic storage, overlays, revisions, locks, SQL primitives |
| `asset` | Raw Asset keys, metadata, `AssetStore`, backends |
| `spec` | Agent/Skill/MCP declarations and codecs |
| `model` | Model routes, credentials, registries, materialization |
| `observe` | Metrics contracts, facade, stores, codecs, and query semantics |
| `capability` | Capability composition, loaders, Skill/MCP/workspace projection |
| `task` | Task graph, DAG, lease, launcher contracts |
| `agent` | Agent compilation, definitions, output contracts, execution binding |
| `runtime` | Composition root, execution, persistence contracts, service APIs |
| `workspace` | Workspace paths, policy, configuration, sandbox contracts |
| `migrate` | Explicit database schema provisioning |

Repository-level checks under `scripts/check/ai` are release tooling, not another runtime architecture layer.

### Verification

```bash
python manage.py install --editable
python manage.py check linktools-ai
```

Run the project gate after changing architecture boundaries, public exports, persistence contracts, or schema definitions. Current pre-release scope has no obligation to read superseded development data. Use the current wire contract as the single baseline; retain normal defaults of current codecs and current execution recovery semantics. Published-version fixtures are required only after a real compatibility commitment exists.

For the current OpenAI binding, base_url, api_key and transport timeout/retry settings are connection concerns, not model semantic identity. Task execution timeout/retry policies and model-visible tool descriptions/schema have different responsibilities and remain semantic where used.

Measure test setup, execution and full check wall time in a matched environment before and after test simplification. Record removed-test responsibility mappings and retain the existing manage.py entry point and CI matrix. Do not infer speedup from fewer files or parametrization alone.
