# AGENTS.md (linktools-ai)

Package instructions for `linktools-ai`. Repository-wide rules in [../AGENTS.md](../AGENTS.md) also apply.

`Required Rules` contain only long-lived constraints. Current package layout, class names, construction paths, and other implementation details belong in `Guidance` or the code itself.

## Required Rules

### Architecture

- Use public package APIs across boundaries. Do not import another package's private modules/members or use reflection to bypass an interface.
- Runtime dependencies must remain acyclic; annotation-only back-references belong under `TYPE_CHECKING`.
- Keep lower-level infrastructure independent from higher-level composition and SDK semantics. Do not introduce duplicate abstractions that compete for the same ownership.
- Keep vendor-specific behavior out of vendor-neutral core abstractions.

### Persistence and concurrency

- Runtime startup must not implicitly create or migrate database schemas; schema provisioning is an explicit deployment/migration operation.
- Persisted or replayed data is a versioned contract. Compatibility must be based on explicit semantics, not accidental byte-for-byte output of dependencies.
- Persistence protocols must remain evolvable and backward-compatible. Additive or non-semantic changes must not invalidate previously persisted data or make the storage system unreadable; incompatible changes require an explicit version boundary and a defined compatibility or migration path.
- Stable hashes and idempotency identities must remain stable when non-semantic optional/default fields change; include codec/version/provenance when they change semantics.
- Durable contracts must stay minimal. Persist stable references for dependencies intentionally resolved at use time; persist dependency semantics only when exact replay of an already-established durable fact requires them. Do not copy, embed, or recursively snapshot referenced configuration for convenience or speculative future recovery.
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

- Temporal workflow code must remain deterministic. Network access, processes, filesystem mutation, model calls, and other external effects belong in Activities or explicitly local adapters.
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
| `observe` | Vendor-neutral run context, middleware, traces, snapshots |
| `capability` | Capability composition, loaders, Skill/MCP/workspace projection |
| `task` | Task graph, DAG, lease, launcher contracts |
| `agent` | Agent compilation, definitions, output contracts, execution binding |
| `runtime` | Composition root, execution, persistence contracts, service APIs |
| `workspace` | Workspace identity, paths, policy, configuration, sandbox contracts |
| `adapter` | External provider, identity, transport adapters |
| `temporal` | Durable workflows, activities, gateway, worker, launcher |
| `migrate` | Explicit database schema provisioning |

Repository-level checks under `scripts/check/ai` are release tooling, not another runtime architecture layer.

### Verification

```bash
python manage.py install --editable
python manage.py check linktools-ai
```

Run the project gate after changing architecture boundaries, public exports, persistence contracts, schema definitions, or release evidence. Durable-codec verification currently uses frozen old-version/golden fixtures and should cover public ingress paths, not only same-version encode/decode.
