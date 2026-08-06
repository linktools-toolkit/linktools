# AGENTS.md (linktools-ai)


## Layout

Source lives under `src/linktools/ai/`:

| Package | Responsibility |
|---|---|
| `foundation/` | IDs, digests, canonical JSON, clock, errors |
| `domain/` | Pure Agent, Execution, Session, Task, Trace, Evaluation and value objects |
| `ports/` | Runtime, Repository, Storage and external Protocols |
| `application/` | One-action actions and cross-entity services |
| `agent/` | Pydantic AI-independent runtime bindings, contracts and local executor |
| `session/`, `tasks/`, `trace/`, `schema/` | Stable public APIs for their respective product contracts |
| `local/` | Local-only Project, Skill, Private Agent and Index support |
| `storage/` | Generic composition, cache, revision, file, coordination and async SQL kernel |
| `outbound/` | External-system adapters; no business facts |
| `orchestration/temporal/` | Deterministic Workflow kernel and one-operation Activities |
| `inbound/` | API, CLI, and ACP protocol mapping |
| `build/` | Upstream Manifest, Bundle, signing, and architecture checks |
| `entrypoints/` | One Composition Root per published artifact |

The old `agent_runtime`, `evaluation_runtime`, `execution` and duplicate
storage-set paths are removed. Do not recreate compatibility imports.

## Storage boundary

Storage is domain-independent. `build_storage()` and
`build_sqlite_storage()` construct async SQL engines without I/O; callers must
invoke `initialize_storage()` explicitly. Domain stores are downstream of this
kernel and never imported back into it. There is no sync SQL API, storage test
package or storage-owned migration runner.

## Import and code rules

- Python >=3.10; every Python file starts with the two standard header lines.
- Do not use `from __future__ import annotations`.
- Imports within `linktools.ai` use relative paths.
- Public methods and functions have parameter and return annotations. Quote
  annotations containing `|` or `[...]`.
- Domain has no I/O, global state, SDK, Repository, or current-time access.
- Ports do not import concrete adapters. Use Cases do not import SQLAlchemy,
  Temporal, Modal, Logfire, or Pydantic AI.
- Workflow code is deterministic and has no database, network, Secret,
  object-store, or telemetry I/O. External effects are Activities.
- Obtain loggers through `from linktools.core import environ` and
  `environ.get_logger(...)`. Do not use `logging.getLogger`.
- `__init__.py` files only re-export stable API; they do not register or scan.
- Do not add `Manager`, `Helper`, `Utils`, generic service locators, private
  cross-package access, or reflection to bypass an interface.

## Verification

Run focused checks with:

```bash
PYTHONPATH=linktools-ai/src:linktools/src .venv/bin/python -m compileall -q linktools-ai/src/linktools/ai
PYTHONPATH=linktools-ai/src:linktools/src .venv/bin/python -m pytest tests/ai/ -q
```

The traceability matrix is `linktools-ai/linktools-ai-traceability-matrix.json`.
The machine-readable DAG policy is
`linktools-ai/linktools-ai-package-dependency-policy.json`.
