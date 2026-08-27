# AGENTS.md (linktools-ai)

## 1. Layout and ownership

Each package owns one concern:

| Package | Responsibility |
|---|---|
| `core` | Pure values, IDs, JSON, paging, principals, and canonical hashing |
| `storage` | Generic overlays, caches, revisions, files, locks, SQL dialects, ObjectStore, and SQL validation primitives |
| `asset` | Raw Asset keys, metadata, `AssetStore`, and Asset backends; no declaration interpretation |
| `spec` | Agent, Skill, and MCP declaration DTOs and codecs |
| `model` | Model routes, connections, credentials, registries, and materializers |
| `observe` | Vendor-neutral run context, middleware, traces, and snapshots |
| `capability` | `CapabilityGroup`, loaders, candidate semantics, Skill/MCP materialization, and workspace tool projection |
| `task` | Generic TaskGraph, DAG, lease, and local launcher contracts |
| `agent` | Agent compilation, immutable definitions, output contracts, and execution binding |
| `runtime` | Composition root, Pydantic AI execution infrastructure, persistence contracts, and service APIs |
| `workspace` | Workspace discovery, identity, paths, policy, configuration, and sandbox contracts |
| `adapter` | External provider, identity, and transport adapters; never Asset storage or Runtime composition |
| `temporal` | Durable workflows, activities, gateway, worker, and launcher |
| `migrate` | Explicit database schema provisioning |
| `src/linktools/commands/ai` | Thin CLI composition over public `linktools.ai` APIs |

Repository-level architecture, import, dependency, and evidence gates live under root `scripts/check/ai`; they are release tooling, not part of the `linktools-ai` package architecture.

Normal library modules live directly under `linktools/ai/<package>/`. Only Temporal may use `workflow/` and `activity/` subpackages. A cross-package public boundary may live directly under `linktools/ai/` only when listed in the package policy's `public_modules`.

`AssetStore` owns raw files, `CapabilityGroup` owns registration and declaration discovery, `spec` owns declaration codecs, `agent` owns compilation and executable binding, `runtime` owns composition and execution/service behavior, and `workspace` owns workspace identity and policy. Keep those boundaries visible in names and imports.

## 2. Dependency boundaries

- `core` and `storage` stay independent of Asset, Spec, Model, Agent, Runtime, Temporal, and SDK semantics.
- Imports inside `linktools.ai` use relative paths.
- Consumers import public package exports, never another package's `_`-prefixed module or member.
- Do not use reflection to bypass a public interface. Add a public method or protocol operation and implement it on every backend.
- Runtime module dependencies must remain acyclic. Legitimate annotation-only back-references belong under `TYPE_CHECKING`.
- `adapter` implements lower-level ports. It must not become a composition root or own Asset loading.
- SQL dialect detection, SQLite pragmas, vendor statements, integrity classification, and coordination scope belong in `storage`.
- `observe` stays vendor-neutral. Do not add vendor-specific telemetry dependencies or adapters to the core package surface.

## 3. Composition and lifecycle

Keep common construction paths short. Callers should pass `RuntimeState` and domain inputs; they should not assemble registries, table collections, manifests, or internal storage bundles.

SQL-backed Runtime composition follows this rule:

```python
await provision_database(engine)
state = RuntimeState.sql(engine)

async with Runtime.open(
    workspace,
    models=models,
    state=state,
) as runtime:
    ...
```

Schema owners expose public builders for migration and evidence generation. Runtime constructors validate their own metadata; normal callers do not assemble schema registries.

Keep classification identities distinct:

- Persistence `namespace` partitions Runtime records inside a backend.
- `tenant_id` is the authorization and resource ownership boundary within a persistence namespace.
- `memory_scope` selects a memory collection within a tenant; persistence stores only its derived `memory_scope_digest`.
- Asset `kind` partitions raw Asset keys independently from Runtime persistence.
- Task and tool lease records use `owner`; schema registries and storage overlays also use `owner` because each declaring type makes the role explicit.
- `RuntimeDomain` selects Runtime persistence domains; lower-level Step, ObjectStore, Idempotency, OperationLedger, Approval, external result, and tool-state records remain implementation details of those domains.
- Prefer short object-local fields when the declaring type supplies the domain, including `Principal.kind`, `AssetKey.kind`, `OperationLedgerRecord.kind`, and `TaskLease.owner`. Add a qualifier when the same type or flattened boundary contains another plausible meaning, as with `resource_kind`, `lineage_kind`, `asset_kind`, and `memory_scope_digest`, or when it preserves an authorization identity domain, as with `owner_principal_id`.
- Free functions have no declaring-object context, so their names retain the domain they validate, such as `validate_persistence_namespace()` and `validate_lease_owner()`.
- Observation and Runtime trace records are distinct contracts: use `RecordedTraceItem` for recorder facts and `ExecutionTraceItem` for Runtime query projections.

Workspace composition defaults the Runtime tenant to `default`, but accepts an explicitly validated independent `tenant_id`. Namespace remains the workspace identity; lower-level Runtime persistence remains multi-tenant.

All lifecycle objects use `initialize()`; do not add parallel lifecycle aliases. `StorageOverlay.initialize()` initializes each distinct backend once. SQL initialization validates owned metadata only. Runtime initialization never creates or alters SQL tables; only `migrate.provision_database()` performs explicit provisioning.

Use `build_*` for pure composition in new APIs. `open_*`, `prepare_*`, and `initialize()` may perform documented I/O. Keep cleanup paired with opening through async context managers where ownership spans a scope.

## 4. Asset and capability rules

`AssetBackend` and the public storage protocols are the extension boundary for raw Asset loading. Compose backends with `StorageOverlay`, wrap the overlay in `AssetStore`, and call `initialize()`. Asset backends store bytes and metadata only; they do not import Spec, Capability, or Agent types.

The default Runtime workspace source reads `<workspace>/.linktools` through a read-only `DirectoryAssetBackend`. `PrefixAssetPathAdapter` maps `agent`, `skill`, and `mcp` kinds to their nested physical directories. Runtime wraps that store with `CapabilityGroup.from_store("workspace", store)`, freezes one immutable candidate snapshot, and synthesizes `AgentSpec("default")` only when no `default` Agent declaration is present.

For downstream declaration formats or custom kinds such as `worker` or `audit`, implement `CapabilityLoader` and register it with `CapabilityGroup.loader()`. The loader converts frozen Asset metadata and bytes into normal `CapabilityContribution` values. Do not reintroduce a parallel Repository, Provider, Registry, or logical Asset-binding layer.

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

## 6. Python and logging style

- Python 3.10 or newer; do not use `from __future__ import annotations`.
- Every Python file starts with the standard shebang and UTF-8 header.
- Public functions and methods have parameter and return annotations.
- Quote annotations containing `|` or `[...]`. Keep annotation-only imports under `TYPE_CHECKING`.
- Public signatures do not use `Any`, `object`, untyped mappings, or unbounded `Callable` unless the value is genuinely untyped.
- `__init__.py` files contain static imports and `__all__` only.
- Obtain loggers through `from linktools.core import environ` and `environ.get_logger(...)` with a relative logger name.
- Log state transitions and decisions, not raw secrets or large payloads. Guard only expensive debug formatting with `if environ.debug`.
- Do not add compatibility shims, dynamic imports, migration fallbacks, or reflection-based adapters.
- Comments explain constraints that naming and structure cannot express. All new or corrected comments must be in English.
  Remove comments made stale by the current edit.
- Keep lines readable by wrapping long signatures, expressions, and log calls instead of allowing dense overlong lines.

## 6.1 Specification conformance

- For specification-driven work, extract a requirement matrix before editing.
  Keep module ownership, public contracts, logs, tests, and evidence aligned with it.
- Perform a fresh cold-start review against the complete specification after implementation.
  Repeat the review and verification loop until no Critical or Important gap remains.
- Legacy migration remains out of scope unless explicitly required, but every newly introduced persisted, replayed, cross-process, or public contract must be forward-compatible by design.

## 6.2 Compatibility design rules

Apply these during design, not after implementation:

- Persisted or replayed data is a versioned contract. Use an explicit codec/version discriminator; never overload ordinary text with magic prefixes.
- Integrity checks validate stored bytes and structure. Never require a dependency upgrade to re-encode old data byte-for-byte identically.
- Stable hashes and idempotency identities must not change because a dependency adds optional/default fields.
- If codec, version, or provenance changes semantics, include that discriminator in stable hashes and idempotency identities; never hash only the opaque payload.
- Caller cancellation does not determine durable truth. Keep shielded commits owned until operation/readback settles; use `STORAGE_COMMIT_UNKNOWN` only when final readback cannot resolve the outcome.
- Per-request model observations belong at the model-response boundary. Persist them before aggregation and derive run totals from the same request facts; never reconstruct single-call data from cumulative usage.
- Internal transport encoders, durable wire types, and implementation protocols stay private unless they are intentional extension points.
- A public Protocol change is an API change. Keep runtime-only bridges private instead of leaking internal policy parameters downstream.
- Serializable provider references are not automatically durable. Define lifetime/recovery semantics before persisting remote file IDs, URLs, handles, or tokens.
- Add frozen old-version/golden fixtures for durable codecs and test every public ingress path; same-version encode/decode tests are insufficient.

## 7. Temporal and external effects

Workflow code must remain deterministic. Network access, processes, filesystem mutation, model calls, and other external effects belong in Activities or explicitly local adapters. Do not import non-deterministic composition code into Temporal workflow modules.

## 8. Verification

Install the development environment once, then run the project gate from the repository root:

```bash
python manage.py install --editable
python manage.py check linktools-ai
```

The specification, package policy, contract map, traceability files, schema reference, and evidence matrices under root `scripts/check/ai/matrix` are release inputs. Update them only from deterministic current-source evidence. Run `python manage.py check linktools-ai` after changing imports, package ownership, public exports, module paths, or release evidence.
