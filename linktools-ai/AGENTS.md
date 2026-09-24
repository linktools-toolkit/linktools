# AGENTS.md (linktools-ai)

Package instructions for `linktools-ai`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

`Required Rules` contain only long-lived constraints. Current package layout, class names, construction paths, and other implementation details belong in `Guidance` or the code itself.

## Required Rules

### Architecture

- Use public package APIs across boundaries. LinkTools package/module exported symbols are defined by static `__all__`; symbols not exported there are not public across owner boundaries. Do not import another package's private modules/members or use reflection to bypass an interface.
- Runtime dependencies must remain acyclic; annotation-only back-references belong under `TYPE_CHECKING`.
- Keep lower-level infrastructure independent from higher-level composition and SDK semantics. Do not introduce duplicate abstractions that compete for the same ownership.
- Keep vendor-specific behavior out of vendor-neutral core abstractions.
- Keep authoring adapters separate from durable codecs. Fill defaults before
  canonical decoding; explicit values always win, and persistence stores
  resolved execution meaning.
- Capture a CapabilityGroup declaration snapshot once and reuse it for host
  inspection and Runtime composition. Bind cross-source reads to the captured
  revision or an immutable reference.
- Resource immutability belongs to AssetStore/AssetVersionRef. Skill and MCP contracts may carry Asset version references plus their own execution semantics, but must not define a separate frozen lifecycle or duplicate Asset ownership.
- AssetKey.id is an opaque logical identity. Only the owning path adapter or capability contract may interpret it hierarchically; Workspace paths and host filesystem paths must not become Asset identity.
- Workspace tool/input paths are canonical Workspace-relative POSIX strings. Workspace.root plus host and sandbox guest paths are physical deployment details and must not become logical path or durable semantic identity.
- Keep visibility, execution authorization, and OS isolation as separate
  boundaries. A successful close for a restricted child process requires
  proof that its owned process tree is quiescent.
- Architecture and release gates encode long-lived invariants only. Do not encode current package names, module depth, class names, or layout as policy.
- Build/release tooling must not become a second owner of Runtime semantic truth.

### Durable contracts and identity

- Runtime startup must not implicitly create or migrate database schemas; schema provisioning is an explicit deployment/migration operation. A local SQLite state backend is the explicit exception and may initialize its own local schema when that state store is created or opened.
- Durable wire formats and named behavior identities are explicit LinkTools contracts. Honor published or explicitly committed compatibility obligations. Without such an obligation, remove obsolete pre-release readers, aliases, defaults and migrations while updating current writers, readers and verification together. Do not prebuild compatibility paths for hypothetical versions.
- Named behavior identity is the explicit `(kind, id, revision)` reference. Do not hash it or maintain a second identity representation. Agent, Tool, Skill, MCP, generic Capability, Task, TaskExpander, and named Metric definitions keep full contracts for restore and validation; any behavior change under the same named contract requires an explicit revision bump.
- Digests are reserved for anonymous/composite values, byte integrity, and request/idempotency contracts. Define each digest from one explicit minimal projection owned by that contract. Non-contract additions and default fields must not change it; whole-object reflection, incidental wire payloads and dependency serialization must not define it.
- Separate named identity from contract validation, byte integrity, and storage addressing. Physical locators, credentials, transport tuning and pure display/diagnostic data do not enter named identity. Complete stored bytes still require complete integrity checks. Preserve logical scope, effect policy, model-visible instructions/schema and provenance when the specific contract needs them.
- Use stable protocol discriminators, not Python class/module names or dependency/build versions. Normalize only equivalences established by the owning contract; preserve ordered inputs and effective business parameters. Named identity comparisons use the explicit reference directly; digest comparisons use the owning digest projection.
- Every current writer output must be accepted by its matching reader. Define omitted optional fields in that wire contract, not through changing runtime defaults. Reject corrupt or unsupported durable data with typed errors; never guess, silently repair or reinterpret unknown execution semantics. Verify original stored bytes before adapting decoded values.

### Persistence and concurrency

- A semantic fact must have one durable owner. Any persisted duplicate used as an index, projection, or cache must be explicitly derived and must not become an independent source of truth or define conflicting recovery semantics.
- Asset history belongs to AssetStore. Durable Runtime bindings may persist Asset version references and resource/content digests, but must not copy Asset resource bytes into Runtime ObjectStore merely to pin execution dependencies.
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

### Verification scope

- Simplifying or consolidating tests must preserve every independent accepted contract and regression obligation. Remove duplicate examples, not distinct backend, concurrency, recovery, external-effect, or failure semantics.
- Named behavior changes require paired verification: unchanged revisions preserve identity, behavior changes require a revision bump, same-revision contract drift is rejected where current definitions are required, and current durable writers round-trip through current readers or current golden fixtures.

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

For the current OpenAI binding, route_id, base_url, api_key and transport timeout/retry settings are connection concerns, not model semantic identity. Task execution timeout/retry policies and model-visible tool descriptions/schema have different responsibilities and remain semantic where used.

Measure test setup, execution and full check wall time in a matched environment before and after test simplification. Record removed-test responsibility mappings and retain the existing manage.py entry point and CI matrix. Do not infer speedup from fewer files or parametrization alone.
