# CLAUDE.md (linktools-ai)

This file provides guidance to Claude Code (claude.ai/code) when working with
the `linktools-ai` sub-package. It complements the repo-root `CLAUDE.md`
(which covers the core framework and the shared Python code style).

## What this sub-package is

A pydantic-ai-based agent / session / execution runtime. It is a Python
library consumed by downstream applications, plus an optional CLI
(`linktools.ai.cli`) and TUI (`linktools.ai.cli.tui`). It registers itself
with the core framework via the `ai` capability (entry point
`linktools.capabilities.ai:__cap_ai__`).

## Layout

Source lives under `src/linktools/ai/`:

| Path | Responsibility |
|------|----------------|
| `agent/` | Agent assembly (`build_runtime`), tool invocation, sandbox, MCP, prompt/context policies, sub-agents, skills, extensions |
| `execution/` | Run lifecycle (PENDING → RUNNING → terminal), sessions/turns, snapshots, trace events, approvals, live events |
| `spec/` | Declarative agent/skill/extension specs over `SpecStore` (versioned, content-addressed document storage with change-log history) |
| `storage/` | The generic persistence kernel — see "Storage" below |
| `tasks/` | Task swarm (multi-agent plans), `TaskStore` |
| `artifact/` | Artifact store |
| `evaluation/` | Eval harness (`EvalExecutor`, `Evaluator`) |
| `governance/` | Authorization policy, security pipeline, tool-policy rules |
| `observability/` | Metrics, tracing, structured events |
| `model/` | Model registry, pricing |
| `cli/` | CLI surface: `console/` (run, continue, doctor, init), `tui/` (Textual TUI), `client.py` (RuntimeClient) |
| `runtime.py` | `build_runtime(...)` + the `Runtime` facade (`run` / `resume` / `cancel` / `inspect` / `aclose`) |

Tests live at **repo-root `tests/ai/`** (not `linktools-ai/tests/`).

## Storage kernel

This is the most architecturally significant subsystem. Everything in
`storage/` is generic over `KeyT` / `ValueT` / `InfoT`; domain stores (spec,
execution, task, tool, artifact) compose it.

### `StorageComposition` (`composition.py`)

The unified base for every domain store. Wires:
- a **primary reader** + optional ordered **layers** (primary-first fallback),
- a **writer** (`StorageWriter` / `BatchStorageWriter`),
- an optional **content cache** (`ContentCache`).

Owns: parallel per-layer metadata refresh, owner-aware entry merge, effective
revision, owner-directed `get` / `get_many` (metadata-miss rule), preload via
`contains_many`, and write post-processing (`_after_put` / `_after_delete` /
`_after_reset` / `_after_batch` → clear preloaded markers + notify the
`RevisionSource`).

### Metadata protocol (`revision.py`)

- `StorageMetadataBackend.load_metadata(after_revision | None)` — one call
  returns the head revision + either a REPLACE (full entry set) or a PATCH
  (diff since the caller's held revision). Replaces the old
  current_revision → list_changes → current_revision round trip.
- `LayerMetadataView` — single-flight refresh per backend: N concurrent
  readers trigger at most one backend load; a cancelled caller never publishes
  a half state.
- `RevisionSource` — injectable revision cache. A view consults
  `head_revision()` before paying for `load_metadata`; when it matches the
  held state, the load is skipped entirely. A downstream redis/file source
  shares one revision signal across machines. The default
  `_BackendHeadRevisionSource` probes `head_revision()` live (no cache).

### Protocols (`multi.py`, `versioning.py`, `cache.py`)

- `StorageReader` (`get`, `list_info`), `BatchStorageReader` (`get_many`),
  `StorageWriter` (`put`, `delete`, `reset`), `BatchStorageWriter`
  (`apply_batch`) — all `@runtime_checkable`; `batch_get` fans single `get`
  calls under a bounded semaphore when a reader lacks `get_many`.
- `VersionedStorage` (`list_versions`, `get_at_revision`) — point-in-time
  history; backends that retain a change log implement it, others omit it.
- `ContentCache` (`get`, `put`, `contains_many`) — L1 memory + L2 filesystem,
  best-effort.

### SQL backends (`storage/sqlalchemy/`)

- **No pessimistic locks.** All write concurrency is optimistic CAS:
  `UPDATE ... WHERE <all previously-read columns still match> ...` +
  `rowcount != 1 → StorageConflictError`. Monotonic columns (`fence`,
  `event_sequence`, `snapshot_revision`) on every CAS WHERE prevent ABA.
- `SqlAlchemyDialect` Protocol — the ONLY vendor seam. Provides
  `insert_ignore_conflict`, `insert_ignore_conflict_many`, `upsert`,
  `upsert_many` (multi-row, per-row conflict values via `excluded` / `inserted`),
  `upsert_increment` (self-seeding counter), `classify_integrity_error`.
  Three built-ins: `SqliteDialect`, `MySQLDialect`, `PostgreSQLDialect`.
- `apply_batch(puts, deletes)` — atomic incremental batch write (one
  transaction, one shared revision, mixed puts + deletes, other rows
  untouched). `StorageComposition.apply_batch` delegates to a
  `BatchStorageWriter` writer, or falls back to per-op `put` / `delete`.
- The core wheel carries **no** environment-specific DB driver. Install via
  the `sqlalchemy` or `sqlite` extras; a MySQL/PG deployment brings its own
  async driver.

## Dependencies

`requirements.yml` (the wheel METADATA source). Core deps:
`pydantic-ai-slim[mcp,openai]`, `linktools`, `jsonschema`. Optional extras:
`sqlalchemy`, `sqlite` (+ `aiosqlite`), `tui` (`textural`). Dev deps include
`pytest-asyncio`, `sqlalchemy[asyncio]`, `aiosqlite`, `lupa`. The wheel must
stay free of env-specific drivers/clients (a gate test enforces this).

## Python code style

Follows the repo-root `CLAUDE.md` rules exactly: Python ≥3.10, no
`from __future__ import annotations`, uniform file headers
(`#!/usr/bin/env python3` + `# -*- coding: utf-8 -*-`), public API annotated,
quote annotations containing `|` / `[...]`, annotation-only imports under
`TYPE_CHECKING`. The storage-layer Protocols are the canonical example of the
`@runtime_checkable` + `isinstance` capability-check pattern.

## Tests

```bash
# from the repo root, after installing dev deps
python -m pytest tests/ai/ -q
```

Key suites: `tests/ai/storage/` (composition, revision, cache, capabilities,
dialects), `tests/ai/spec/` (SQL spec backend, version history, apply_batch),
`tests/ai/persistence/` (execution/task/tool SQL backends — CAS claim/renew/
complete fencing), `tests/ai/architecture/` (invariants + boundary gates).
