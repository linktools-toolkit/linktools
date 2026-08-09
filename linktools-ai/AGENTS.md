# AGENTS.md (linktools-ai)

## Layout

The package is deliberately small and has one owner per concern:

| Package | Responsibility |
|---|---|
| `core` | Pure values, errors, IDs, JSON, paging and principals |
| `storage` | Generic layers, cache, revision, versioning, files, locks and SQL setup |
| `asset` | Raw Asset file keys, metadata, `AssetStore` and file backends |
| `spec` | Agent/Prompt/Feature DTOs, output registry and codecs |
| `model` | Model registry, configuration and resolver |
| `observe` | Run context, middleware, trace and snapshots |
| `capability` | MCP, Skill, Tool, Sandbox, Retrieval and Extension contracts |
| `task` | Task, Job, Swarm and DAG contracts |
| `agent` | Pydantic AI boundary, dependencies, context and runner |
| `runtime` | Runtime persistence contracts, services and the seven runtime APIs |
| `workspace` | Workspace identity, discovery and local coding tools |
| `adapter` | Non-Asset external adapters and runtime/task ports |
| `temporal` | Durable workflows, activities, gateway, worker and launcher |
| `scripts/build` | Bundle compilation, architecture, import, dependency and data gates |
| `app` | Runtime/workspace composition, HTTP, CLI, ACP and the only composition root |

Normal library modules are directly under `linktools/ai/<package>/`. A
cross-package public boundary module may live directly under `linktools/ai/`
only when listed in `public_modules`. Build-time gates live under `scripts/build/`. Only Temporal
may use `workflow/` and `activity/` subpackages. `AssetStore` stores raw files;
the spec and capability packages own their DTO serialization.

## Boundaries and style

- Python >=3.10; every Python file starts with the standard two-line header.
- Do not use `from __future__ import annotations`.
- Imports inside `linktools.ai` use relative paths.
- Public APIs are fully annotated. Quote annotations containing `|` or `[...]`.
- Public signatures do not use `Any`, `object`, untyped mappings or unbounded
  `Callable`.
- `core` and `storage` remain independent of Asset, Spec, Agent, Runtime,
  Temporal and SDK semantics.
- `__init__.py` files only contain static exports and `__all__`.
- Obtain loggers through `from linktools.core import environ` and
  `environ.get_logger(...)`; log important state transitions.
- Do not add compatibility shims, dynamic imports, reflection or private
  cross-package access.
- Workflow code is deterministic. External effects belong to Activities.
- File and SQL initialization is explicit; builders do not perform I/O.
- Module naming is namespace-scoped, not globally unique. A semantic leaf may
  repeat under parallel package namespaces, such as
  `linktools.ai.aaa.bbb` and `linktools.ai.ccc.bbb`, but it must not shadow the
  same leaf in an ancestor namespace, such as `linktools.ai.aaa.bbb` and
  `linktools.ai.bbb`. The same rule applies to packages; a leading `_` is
  visibility-only and ignored when comparing names.

## Verification

```bash
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m compileall -q linktools-ai/src/linktools/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m pytest -q tests/ai
PYTHONPATH=linktools-ai:linktools-ai/src:linktools/src python3 -m ruff check linktools-ai/scripts/build linktools-ai/src/linktools/ai linktools-ai/src/linktools/commands/ai tests/ai
```

The specification, package policy, contract map, traceability and evidence
manifests under `scripts/build/matrix` are release inputs. Update them only
with deterministic evidence from the current source and tests.
