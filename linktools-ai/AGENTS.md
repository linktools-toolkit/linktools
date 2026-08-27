# AGENTS.md (linktools-ai)

Package instructions for `linktools-ai`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

## Required Rules

### Architecture boundaries

- Imports inside `linktools.ai` use relative paths. Cross-package consumers use public package exports, never another package's `_`-prefixed module/member or reflection to bypass a public interface.
- Runtime dependencies remain acyclic. Annotation-only back-references belong under `TYPE_CHECKING`.
- `core` and `storage` stay independent of Asset, Spec, Model, Agent, Runtime, Temporal, and SDK semantics.
- `adapter` implements lower-level ports and must not own Runtime composition or Asset loading. `observe` remains vendor-neutral. SQL dialect/pragmas/vendor statements/integrity classification/coordination belong in `storage`.
- Normal modules live directly under `linktools/ai/<package>/`. Only Temporal may use `workflow/` and `activity/` subpackages; cross-package public modules at `linktools/ai/` must be declared in the package policy.

### Composition and lifecycle

- Keep common construction paths short: callers provide `RuntimeState` and domain inputs, not internal registries, schema collections, manifests, or storage bundles.
- Lifecycle objects use `initialize()`; do not add parallel lifecycle aliases. Runtime initialization never creates or alters SQL tables; schema provisioning belongs only to `migrate.provision_database()`.
- Use `build_*` for pure composition. `open_*`, `prepare_*`, and `initialize()` may perform documented I/O; pair owned cleanup with async context managers.
- Keep persistence `namespace`, authorization `tenant_id`, memory scope, Asset `kind`, and lease `owner` as distinct identity domains. Persist only the derived `memory_scope_digest` for memory scope.
- Observation facts and Runtime trace projections are distinct contracts: `RecordedTraceItem` and `ExecutionTraceItem` are not interchangeable.

### Asset and capability

- Asset backends store raw bytes/metadata only and do not interpret Spec, Capability, or Agent declarations. Compose storage through `StorageOverlay` and `AssetStore`.
- Default workspace assets come from `<workspace>/.linktools` through the read-only directory backend, then through the normal `CapabilityGroup` path. Synthesize `AgentSpec("default")` only when no default Agent declaration exists.
- Custom declaration formats/kinds use `CapabilityLoader` and produce normal `CapabilityContribution` values. Do not add a parallel Repository, Provider, Registry, or logical Asset-binding layer.
- Overlay precedence remains primary then fallback; tombstones hide lower values and reset entries reveal them. Writer routing, revisions, caching, and owner-aware reads stay in `StorageOverlay`.

### Persistence and SQL

- Runtime, Step, ObjectStore, and Asset builders register only tables they own. Public SQL backends accept an `AsyncEngine`; composition shares one `SqlStorageContext`, while each operation owns its `AsyncSession`.
- SQLite uses WAL and process-scoped coordination. MySQL/PostgreSQL use shared-database coordination. Filesystem coordination uses `filelock`; do not add platform-specific advisory locking.
- `migrations/init_schema.sql` is the DBA reference. Business tables use `ai_`, an `id BIGINT AUTO_INCREMENT` surrogate primary key, business comments, and fixed `updated_at`/`created_at` audit columns with `ix_updated_at`/`ix_created_at`.
- Index rules remain: `uk_<ordered_columns>` / `ix_<ordered_columns>`, no table name in MySQL index names, at most three physical columns in composite keys, SHA-256 as `CHAR(64)` with `utf8mb4_bin`, no prefix unique key on wide columns, 128-character prefix for ordinary wide indexes, and no redundant or low-selectivity status-only indexes.
- Foreign keys and `FLOAT` are prohibited. `JSON`/`LONGBLOB` stay limited to fields that genuinely require them. DDL and canonical metadata must describe the same schema; there is no schema manifest table.

The fixed audit definitions are:

```sql
updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    ON UPDATE CURRENT_TIMESTAMP COMMENT 'Update timestamp',
created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
    COMMENT 'Creation timestamp'
```

### Durable compatibility

- Persisted/replayed data is versioned. Use explicit codec/version discriminators; do not encode versions through magic text prefixes.
- Stable hashes/idempotency identities must not change because dependencies add optional/default fields. Include codec/version/provenance when those values change semantics.
- Caller cancellation does not determine durable truth. Resolve shielded commit/readback before reporting `STORAGE_COMMIT_UNKNOWN`.
- Capture per-request model observations at the model-response boundary before aggregation; derive totals from those same facts.
- Internal wire types, transport encoders, and runtime-only Protocols stay private unless intentionally designed as extension points. A public Protocol change is an API change.
- Define lifetime/recovery semantics before persisting remote provider references. Keep frozen old-version/golden fixtures and exercise every public ingress path for durable codecs.

### Temporal

- Workflow code remains deterministic. Network access, processes, filesystem mutation, model calls, and other external effects belong in Activities or explicitly local adapters.
- Do not import non-deterministic composition code into Temporal workflow modules.

### AI-specific code rules

- Python 3.10 or newer; do not use `from __future__ import annotations`.
- `__init__.py` files contain static imports and `__all__` only.
- Do not add compatibility shims, dynamic imports, migration fallbacks, or reflection-based adapters.
- New or corrected comments are in English.

### Specification work and verification

- For specification-driven work, establish the requirement matrix before editing and keep implementation, public contracts, tests, logs, and evidence aligned with it. Re-review the complete specification after implementation until no Critical or Important gap remains.
- Legacy migration is out of scope unless explicitly required; newly introduced public, persisted, replayed, or cross-process contracts must be forward-compatible.
- Run `python manage.py check linktools-ai` after changes to imports, ownership, public exports, module paths, persistence contracts, or release evidence. Files under `scripts/check/ai/matrix` are release inputs and must reflect deterministic current-source evidence.

## Guidance

### Package responsibilities

| Package | Responsibility |
| --- | --- |
| `core` | Pure values, IDs, JSON, paging, principals, canonical hashing |
| `storage` | Generic storage, overlays, revisions, locks, SQL primitives |
| `asset` | Raw Asset keys, metadata, `AssetStore`, backends |
| `spec` | Agent/Skill/MCP declarations and codecs |
| `model` | Model routes, credentials, registries, materialization |
| `observe` | Vendor-neutral run context, middleware, traces, snapshots |
| `capability` | Capability composition, loaders, Skill/MCP/workspace projection |
| `task` | Task graph, DAG, lease, launcher contracts |
| `agent` | Agent compilation, definitions, output contracts, execution binding |
| `runtime` | Composition root, execution, persistence contracts, service APIs |
| `workspace` | Workspace identity, paths, policy, configuration, sandbox contracts |
| `adapter` | External provider, identity, transport adapters |
| `temporal` | Durable workflows, activities, gateway, worker, launcher |
| `migrate` | Explicit database schema provisioning |
| `src/linktools/commands/ai` | Thin CLI composition over public `linktools.ai` APIs |

### Typical SQL-backed Runtime

```python
await provision_database(engine)
state = RuntimeState.sql(engine)

async with Runtime.open(workspace, models=models, state=state) as runtime:
    ...
```

Repository-level checks under `scripts/check/ai` are release tooling, not an additional runtime architecture layer.
