# AGENTS.md (linktools-ai)

## 1. Layout and ownership

Each package owns one concern:

| Package | Responsibility |
|---|---|
| `core` | Pure values, IDs, JSON, paging, principals, and canonical hashing |
| `storage` | Generic overlays, caches, revisions, files, locks, SQL dialects, ObjectStore, and SQL validation primitives |
| `asset` | Raw Asset keys, metadata, logical bindings, `AssetStore`, `AssetRepository`, and Asset backends |
| `spec` | Agent, Skill, and MCP declaration DTOs and codecs |
| `model` | Model routes, connections, credentials, registries, and materializers |
| `observe` | Vendor-neutral run context, middleware, traces, and snapshots |
| `capability` | Capability resolution, grants, bindings, and Pydantic AI materialization contracts |
| `task` | Generic TaskGraph, DAG, lease, and local launcher contracts |
| `agent` | Agent compilation, output schemas, execution binding, and the Pydantic AI runner |
| `runtime` | Persistence contracts and the execution, session, task, evaluation, approval, event, and artifact APIs |
| `workspace` | Workspace discovery, local tools, persistence selection, and the library composition root |
| `adapter` | Runtime, step, provider, identity, NATS, and history adapters; never Asset storage |
| `temporal` | Durable workflows, activities, gateway, worker, and launcher |
| `migrate` | Explicit database schema provisioning |
| `src/linktools/commands/ai` | Thin CLI composition over public `linktools.ai` APIs |

Repository-level architecture, import, dependency, code-style, logging, comments, and evidence gates are inherited from the root `AGENTS.md`. Release tooling lives under root `scripts/check/ai`; it is not part of the `linktools-ai` package architecture.

Normal library modules live directly under `linktools/ai/<package>/`. Only Temporal may use `workflow/` and `activity/` subpackages. A cross-package public boundary may live directly under `linktools/ai/` only when listed in the package policy's `public_modules`.

`AssetStore` owns raw files, `AssetRepository` owns logical Asset resolution, `spec` owns declaration codecs, `capability` owns provider resolution, `agent` owns compilation and executable binding, and `runtime` owns service behavior. Keep those boundaries visible in names and imports.

## 2. Dependency boundaries

- `core` and `storage` stay independent of Asset, Spec, Model, Agent, Runtime, Temporal, and SDK semantics.
- Imports inside `linktools.ai` use relative paths.
- Consumers import public package exports, never another package's `_`-prefixed module or member.
- Runtime module dependencies must remain acyclic. Legitimate annotation-only back-references belong under `TYPE_CHECKING`.
- `adapter` implements lower-level ports. It must not become a composition root or own Asset loading.
- SQL dialect detection, SQLite pragmas, vendor statements, integrity classification, and coordination scope belong in `storage`.
- `observe` stays vendor-neutral. Do not add vendor-specific telemetry dependencies or adapters to the core package surface.

## 3. Composition and lifecycle

Keep common construction paths short. Callers should pass `RuntimeState` and domain inputs; they should not assemble registries, table collections, manifests, or internal storage bundles.

The SQL lifecycle follows this rule:

```python
await provision_database(engine)
context = create_sql_storage_context(engine)
await context.initialize(metadata=owned_metadata)
domain = await open_sql_runtime(context, persist=frozenset({StorageDomain.CONVERSATION}))
assets = build_sql_asset_backend(context)
```

Schema owners expose public builders for migration and evidence generation. Runtime constructors validate their own metadata; normal callers do not assemble schema registries.

Keep classification identities distinct:

- Persistence `namespace` partitions Runtime records inside a backend.
- `tenant_id` is the authorization and resource ownership boundary within a persistence namespace.
- `memory_scope` selects a memory collection within a tenant; persistence stores only its derived `memory_scope_digest`.
- Asset `namespace` partitions raw Asset storage independently from Runtime persistence; Asset `kind` selects the logical representation.
- Task and tool lease records use `owner`; schema registries and storage overlays also use `owner` because each declaring type makes the role explicit.
- `StorageDomain` selects durable business domains; Blob, Media, StepStore, Idempotency, OperationLedger, Approval, ExternalResult, ToolState, and Repository remain implementation details of those domains.
- Prefer short object-local fields when the declaring type supplies the domain, including `Principal.kind`, `AssetKey.kind`, `OperationLedgerRecord.kind`, and `TaskLease.owner`. Add a qualifier when the same type or flattened boundary contains another plausible meaning, as with `resource_kind`, `lineage_kind`, `asset_kind`, and `memory_scope_digest`, or when it preserves an authorization identity domain, as with `owner_principal_id`.
- Free functions have no declaring-object context, so their names retain the domain they validate, such as `validate_persistence_namespace()` and `validate_lease_owner()`.
- Observation and Runtime trace records are distinct contracts: use `RecordedTraceItem` for recorder facts and `ExecutionTraceItem` for Runtime query projections.

Workspace composition defaults the Runtime tenant to `default`, but accepts an explicitly validated independent `tenant_id`. Namespace remains the workspace identity; lower-level Runtime persistence remains multi-tenant.

All lifecycle objects use `initialize()`; do not add parallel lifecycle aliases. `StorageOverlay.initialize()` initializes each distinct backend once. SQL initialization validates owned metadata only. Runtime initialization never creates or alters SQL tables; only `migrate.provision_database()` performs explicit provisioning.

Use `build_*` for pure composition in new APIs. `open_*`, `prepare_*`, and `initialize()` may perform documented I/O. Keep cleanup paired with opening through async context managers where ownership spans a scope.

### 3.1 Persistent-state evolution

Persistent state is long-lived. **Routine field-level evolution must never make an otherwise valid filesystem or SQL Runtime store globally unusable.** A minor DTO/dataclass field change must not force users to rebuild the whole database, discard unrelated records, or make unrelated Runtime domains fail to initialize.

Apply these rules to both filesystem and SQL persistence:

- Ordinary internal records use tolerant readers. Adding an optional/defaulted field, reordering fields, or adding a non-identity/non-state-machine field must keep previously persisted records readable. Unknown additive fields from a newer writer must not invalidate unrelated older readers when the record contract is explicitly tolerant.
- Keep rapidly evolving, non-query/non-index fields inside the existing payload/metadata envelope when that preserves the contract cleanly. Do not add a physical database column merely because an internal dataclass gained a field.
- Fields that participate in resource identity, authorization ownership, durable digests, idempotency keys, state-machine invariants, or exact execution/recovery binding remain fail-closed. If such a field changes meaning or representation, introduce an explicit contract version/revision and a deterministic decoder or migration path; never silently reinterpret old data.
- Exact durable contracts and tolerant internal records must stay separate. Do not weaken exact binding, recovery, operation, checksum, or state-transition validation merely to obtain compatibility.
- A malformed or incompatible individual business record should fail the operation reading that record with the appropriate integrity/compatibility error. It must not prevent unrelated domains or unrelated valid records from being opened unless the shared storage schema itself is genuinely incompatible.
- Initialization validates backend/schema capabilities and owned metadata; it must not eagerly deserialize every persisted business record as a prerequisite for opening the Runtime.
- Compatibility logic belongs in the canonical codec/version boundary. Do not scatter per-caller compatibility branches or ad-hoc migration fallbacks through services and repositories.

When choosing between strictness and field-evolution tolerance, preserve strictness for **semantic identity/invariants** and preserve tolerance for **incidental representation fields**.

## 4. Asset rules

`AssetBackend` and the public storage protocols are the extension boundary for custom loading. Compose backends with `StorageOverlay`, wrap the overlay in `AssetStore`, call `initialize()`, then create an `AssetRepository` from a frozen `AssetTypeRegistry` snapshot.

The default workspace loader reads `<workspace>/.linktools` through a read-only `DirectoryAssetBackend`. `PrefixAssetPathAdapter` maps Agent, Skill, and MCP kinds to their nested physical directories; only the workspace-owned repository bootstraps `agent/default`. Do not hard-code that policy into generic Asset or Storage code.

Register custom logical representations with `AssetTypeBinding` and `AssetVariantBinding`. Codecs validate bytes at the repository boundary. Backends store bytes and metadata only; they do not import Spec, Capability, or Agent types.

Layer precedence is primary first, followed by declared fallback layers. Tombstones hide lower values; reset entries reveal them. Writer routing, cache validation, revisions, and owner-aware reads stay in `StorageOverlay`.

## 5. SQL rules

- Runtime, Step, ObjectStore, and Asset builders register only tables they own into an optional shared `MetaData`.
- `migrations/init_schema.sql` is the manually maintained DBA reference; there is no schema registry or manifest table.
- SQLite requires WAL and process-scoped coordination. MySQL and PostgreSQL use shared-database coordination.
- Filesystem coordination uses `filelock`; do not add `fcntl`, `msvcrt`, or platform-specific advisory-lock code.
- Keep schema creation out of runtime startup. Tests and deployment tooling call `provision_database()` explicitly.
- Public SQL backends receive an `AsyncEngine`; composition creates one `SqlStorageContext` and shares its session factory internally. Each operation creates its own `AsyncSession`.
- Add logs at database preparation, schema validation, transaction retry, lease transition, and backend initialization boundaries.

The DBA reference must also satisfy these schema constraints:

1. All business tables use the `ai_` prefix.
2. There is no schema manifest table.
3. Every table and column has a business `COMMENT`.
4. Placeholder comments are prohibited.
5. Every table uses an `id BIGINT AUTO_INCREMENT` surrogate primary key.
6. `updated_at` is immediately before `created_at`.
7. Audit fields use the fixed MySQL definitions `updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp'` and `created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP COMMENT 'Creation timestamp'`; these built-in comments remain in English.
8. Every table retains `ix_updated_at` and `ix_created_at`.
9. Unique-key names use `uk_<ordered_columns>`.
10. Index names use `ix_<ordered_columns>`.
11. MySQL index names do not contain table names.
12. Composite unique and index keys contain at most three physical columns.
13. SHA-256 values use `CHAR(64)` with `utf8mb4_bin`.
14. Wide columns carrying uniqueness never use prefix unique keys.
15. Ordinary non-unique wide indexes use a 128-character prefix.
16. Redundant indexes fully covered by unique or primary keys are prohibited.
17. Low-selectivity status single-column indexes are prohibited.
18. Foreign keys are prohibited.
19. `FLOAT` columns are prohibited.
20. `JSON` and `LONGBLOB` are limited to the real fields in this schema.
21. The DDL and canonical metadata describe the same schema.

The fixed English audit columns apply to every Runtime, Asset, and ObjectStore
table, including `ai_state_records`, `ai_state_aliases`, `ai_state_facts`,
`ai_state_sequences`, `ai_state_operations`, `ai_asset_heads`,
`ai_asset_entries`, `ai_asset_changes`, `ai_objects`, and
`ai_object_chunks`:

```sql
updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    COMMENT 'Creation timestamp'
```

## 6. Package-specific Python constraints

Repository-wide Python style, public annotation, logging, module-structure, reflection, and comment-minimization rules come from the root `AGENTS.md`. `linktools-ai` adds only these stricter constraints:

- Do not use `from __future__ import annotations`.
- Public signatures do not use `Any`, `object`, untyped mappings, or unbounded `Callable` unless the value is genuinely untyped.
- `__init__.py` files contain static imports and `__all__` only.
- Do not add ad-hoc compatibility shims, dynamic imports, or service-level migration fallbacks. Required persistence compatibility belongs in the canonical codec/version boundary described in §3.1.
- All new or corrected comments must be in English.
- Keep lines readable by wrapping long signatures, expressions, and log calls instead of allowing dense overlong lines.

### 6.1 Specification conformance

- For specification-driven work, extract a requirement matrix before editing. Keep module ownership, public contracts, logs, tests, and evidence aligned with it.
- Perform a fresh cold-start review against the complete specification after implementation. Repeat the review and verification loop until no Critical or Important gap remains.
- Treat compatibility and migration as out of scope unless the specification explicitly requires a legacy decoder, first-read adapter, or rollback materialization path. This does not override the baseline tolerant-reader requirements in §3.1.

## 7. Temporal and external effects

Workflow code must remain deterministic. Network access, processes, filesystem mutation, model calls, and other external effects belong in Activities or explicitly local adapters. Do not import non-deterministic composition code into Temporal workflow modules.

## 8. Verification

Install the development environment once, then run the project gate from the repository root:

```bash
python manage.py install --editable
python manage.py check linktools-ai
```

The specification, package policy, contract map, traceability files, schema reference, and evidence matrices under root `scripts/check/ai/matrix` are release inputs. Update them only from deterministic current-source evidence. Run `python manage.py check linktools-ai` after changing imports, package ownership, public exports, module paths, or release evidence.