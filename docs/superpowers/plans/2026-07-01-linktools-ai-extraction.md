# linktools-ai Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract the domain-agnostic Agent/Session/Registry/Capability runtime currently living in `sec-smartops-svc`'s `engine/agent/` and `engine/capabilities/` into a new, independently-versioned `linktools-ai` sub-package inside the `linktools` monorepo, so it can be reused as the base for a future A-share quant trading agent.

**Architecture:** New sub-package `linktools-ai` (package `linktools.ai`) holds two layers: `linktools.ai.support` (generic hooks/utils/config/workspace helpers, migrated from `engine/infra/*`), and `linktools.ai.core` + `linktools.ai.capabilities` (the Agent/Session/Registry runtime and Skill/Subagent/MCP capability loading). Everything that depends on MySQL/Redis/the `EngineEnvironment` singleton stays in `sec-smartops-svc`; the boundary is expressed as `Protocol` types (`AgentEnvironment`, `CapabilityRepositoryProtocol`, `CapabilityCacheProtocol`, `TranscriptStore`/`HistoryStore`/`ArtifactStore` — the last three already exist as `Protocol` in `engine/agent/stores.py`) that `sec-smartops-svc`'s concrete classes satisfy structurally (duck typing, no inheritance required).

**Tech Stack:** Python 3.11, `pydantic-ai`, `pytest`, `pytest-asyncio`, existing `linktools` monorepo packaging (`manage.py`, `tool.linktools` pyproject metadata).

## Global Constraints

- `linktools-ai` depends on the `linktools` core package (Config/CLI conventions) but ships **no CLI commands** — it is import-only for now.
- `linktools-ai` must not import `sqlalchemy`, MySQL, Redis, or Kafka — those stay in `sec-smartops-svc`. Redis-shaped dependencies are expressed as a `Protocol`, never a concrete import.
- Every module moved must keep its existing public API (class/function names, signatures) unchanged, so `sec-smartops-svc`'s existing tests continue to pass unmodified after the import-path switch — this is a move-and-decouple refactor, not a rewrite.
- No new abstractions beyond what's needed to remove the `EngineEnvironment`/MySQL/Redis coupling — do not generalize further "for the future".

---

### Task 1: Scaffold the `linktools-ai` sub-package

**Files:**
- Create: `linktools-ai/src/linktools/ai/__init__.py`
- Create: `linktools-ai/README.md`
- Modify (generated): `linktools-ai/pyproject.toml`, `linktools-ai/requirements.yml`, `linktools-ai/capability.jinja2`, `linktools-ai/MANIFEST.in` (via `manage.py init`)
- Create: `linktools-ai/src/linktools/capabilities/ai.py` (hand-adapted from `linktools-common/src/linktools/capabilities/common.py`)
- Create: `linktools-ai/.version`
- Create: `linktools-ai/tests/__init__.py`

**Interfaces:**
- Produces: the `linktools-ai` package root importable as `linktools.ai` after editable install.

- [ ] **Step 1: Create the directory skeleton**

```bash
cd /workspace/projects/linktools
mkdir -p linktools-ai/src/linktools/ai linktools-ai/tests
touch linktools-ai/src/linktools/ai/__init__.py linktools-ai/tests/__init__.py
echo "v0.0.1" > linktools-ai/.version
```

- [ ] **Step 2: Generate packaging metadata from the monorepo templates**

```bash
python manage.py init linktools-ai
```

Expected: creates `linktools-ai/pyproject.toml`, `linktools-ai/requirements.yml`, `linktools-ai/capability.jinja2`, `linktools-ai/MANIFEST.in` (mirrors `linktools-common`'s layout — verify with `diff <(ls linktools-common) <(ls linktools-ai)`, the same four files plus `src/`/`tests/`/`.version`/`README.md` should be present).

- [ ] **Step 3: Hand-write the capability registration file**

Copy `linktools-common/src/linktools/capabilities/common.py` to `linktools-ai/src/linktools/capabilities/ai.py`, then edit the copied file so it reads:

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib

from linktools.core import Capability as _Capability


class Capability(_Capability):

    def __init__(self):
        super().__init__()
        self._root_path = pathlib.Path(os.path.dirname(os.path.dirname(__file__)))

    @property
    def name(self) -> str:
        return "linktools-ai"

    @property
    def version(self) -> str:
        return "0.0.1"

    @property
    def develop(self) -> bool:
        return True

    @property
    def release(self) -> bool:
        return False

    @property
    def root_path(self) -> pathlib.Path:
        return self._root_path


__cap_ai__ = Capability()
```

- [ ] **Step 4: Add a minimal README**

```markdown
# linktools-ai

Domain-agnostic Agent/Session/Registry/Capability runtime, extracted from
sec-smartops-svc. Library only — no CLI commands. See
`docs/superpowers/specs/2026-07-01-linktools-ai-extraction-design.md` for
the extraction boundary and rationale.
```

- [ ] **Step 5: Editable-install and verify the package imports**

```bash
python manage.py install --editable linktools-ai
python -c "import linktools.ai; print('ok')"
```

Expected: `ok` printed, no import errors.

- [ ] **Step 6: Commit**

```bash
git add linktools-ai
git commit -m "chore(ai): scaffold linktools-ai sub-package"
```

---

### Task 2: Migrate generic infra support (hooks / utils / config / workspace)

**Files:**
- Create: `linktools-ai/src/linktools/ai/support/__init__.py`
- Create: `linktools-ai/src/linktools/ai/support/hooks.py` (from `engine/infra/hooks.py`)
- Create: `linktools-ai/src/linktools/ai/support/utils.py` (from `engine/infra/utils.py`, unchanged)
- Create: `linktools-ai/src/linktools/ai/support/config.py` (from `engine/infra/config.py`, unchanged)
- Create: `linktools-ai/src/linktools/ai/support/workspace.py` (from `engine/infra/workspace.py`)
- Create: `linktools-ai/tests/support/test_hooks.py`
- Test: `linktools-ai/tests/support/test_hooks.py`

**Interfaces:**
- Produces: `linktools.ai.support.hooks.HookRegistry`, `linktools.ai.support.hooks.HookEvent` (generic subset only — see step 2), `linktools.ai.support.utils.{stable_json, json_ready, resolve_ref, truthy, model_type, deep_merge_dicts, call_id, safe_filename}`, `linktools.ai.support.config.{load_yaml_file, load_yaml_text, load_markdown_file, load_markdown_text, as_str_dict}`, `linktools.ai.support.workspace.{WorkspaceStore, LocalWorkspaceStore, TraceStore, SlotMode}`.
- Consumes: nothing from earlier tasks (leaf modules).

- [ ] **Step 1: Copy utils.py and config.py verbatim (no environ coupling to remove)**

```bash
cd /workspace/projects/sec-smartops-svc
cp engine/infra/utils.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/support/utils.py
cp engine/infra/config.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/support/config.py
```

- [ ] **Step 2: Copy hooks.py and strip the `environ` coupling + secops-specific event names**

Read `engine/infra/hooks.py` first. Copy it to `linktools-ai/src/linktools/ai/support/hooks.py`, then apply these edits to the copy:

```python
# Replace this:
from ..environ import environ
logger = environ.get_logger(__name__)

# With this:
import logging
logger = logging.getLogger(__name__)
```

Trim the `HookEvent` enum to the domain-agnostic values actually referenced by `engine/agent/*` and `engine/capabilities/*` (verified by `grep -rn "HookEvent\." engine/agent engine/capabilities`):

```python
class HookEvent(str, Enum):
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    LLM_CALL_START = "llm_call_start"
    MCP_CALL_START = "mcp_call_start"
    SUBAGENT_START = "subagent_start"
    SUBAGENT_END = "subagent_end"
    POST_LLM_CALL = "post_llm_call"
    POST_MCP_CALL = "post_mcp_call"
```

Keep `HookRegistry` (the class body below `HookEvent`) unchanged.

- [ ] **Step 3: Copy workspace.py and strip the `environ` coupling**

Copy `engine/infra/workspace.py` to `linktools-ai/src/linktools/ai/support/workspace.py`, then replace the same `from ..environ import environ` / `environ.get_logger(__name__)` pattern as step 2.

- [ ] **Step 4: Write a test that pins the trimmed HookEvent contract**

```python
# linktools-ai/tests/support/test_hooks.py
import asyncio

from linktools.ai.support.hooks import HookEvent, HookRegistry


def test_hook_registry_dispatches_registered_handler():
    registry = HookRegistry()
    seen = []

    def handler(payload):
        seen.append(payload)

    registry.on(HookEvent.AGENT_START, handler)
    registry.fire(HookEvent.AGENT_START, {"agent": "worker"})

    assert seen == [{"agent": "worker"}]


def test_hook_registry_swallows_handler_exceptions():
    registry = HookRegistry()

    def bad_handler(payload):
        raise RuntimeError("boom")

    registry.on(HookEvent.AGENT_END, bad_handler)
    registry.fire(HookEvent.AGENT_END, {})  # must not raise
```

Adjust the exact `HookRegistry.on`/`.fire` method names to match what `engine/infra/hooks.py` actually exposes (read the file before writing this test — the test must call the registry's real registration/dispatch method names, not invented ones).

- [ ] **Step 5: Run the test to verify it fails, then passes after the copy**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/support/test_hooks.py -v
```

Expected: PASS (the module already exists from step 2-3; this run is the verification, not red-green, since we're moving working code).

- [ ] **Step 6: Commit**

```bash
git add linktools-ai/src/linktools/ai/support linktools-ai/tests/support
git commit -m "feat(ai): migrate generic infra support (hooks/utils/config/workspace)"
```

---

### Task 3: Define Capability Protocols + in-memory reference implementations

**Files:**
- Create: `linktools-ai/src/linktools/ai/capabilities/protocols.py`
- Create: `linktools-ai/src/linktools/ai/capabilities/memory.py`
- Test: `linktools-ai/tests/capabilities/test_memory_repository.py`

**Interfaces:**
- Consumes: nothing (leaf module within `capabilities/`).
- Produces: `CapabilityRepositoryProtocol`, `CapabilityCacheProtocol` (both `typing.Protocol`), `InMemoryCapabilityRepository`, `InMemoryCapabilityCache` — these are what `CapabilityStore` (Task 4) type-hints against and what `sec-smartops-svc`'s MySQL/Redis classes will structurally satisfy (Task 9).

- [ ] **Step 1: Write `protocols.py`**

Method surface derived from actual call sites in `engine/capabilities/store.py` (verified by reading the file):

```python
# linktools-ai/src/linktools/ai/capabilities/protocols.py
from __future__ import annotations

from datetime import datetime
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CapabilityRepositoryProtocol(Protocol):
    async def upsert_file(
        self, *, kind: str, file_path: str, content: str, checksum: str, updated_by: str,
    ) -> tuple[int, int, bool]: ...

    async def tombstone_file(
        self, kind: str, file_path: str, *, checksum: str, updated_by: str,
    ) -> bool: ...

    async def apply_batch(
        self, *, kind: str, capability_id: str, primary_rel_path: str, primary_content: str,
        supplementary_files: list[dict[str, str | None]], deleted_rel_paths: list[str],
        expected_revision: str, updated_by: str,
    ) -> tuple[str, list[dict[str, Any]], bool]: ...

    async def list_capabilities_active(self, kind: str | None = None) -> list[dict[str, Any]]: ...

    async def capability_exists(self, kind: str, capability_id: str) -> bool: ...

    async def get_file(self, kind: str, file_path: str) -> dict[str, Any] | None: ...

    async def get_file_at_version(self, kind: str, file_path: str, version: int) -> dict[str, Any] | None: ...

    async def get_primary_files(self, kind: str, primary_rel: str) -> list[dict[str, Any]]: ...

    async def list_files(self, kind: str, capability_id: str) -> list[dict[str, Any]]: ...

    async def list_file_states(self, kind: str, capability_id: str) -> list[dict[str, Any]]: ...

    async def list_files_since(self, since: datetime | None) -> list[dict[str, Any]]: ...

    async def delete_files_all(self, kind: str, capability_id: str) -> int: ...

    async def restore_builtin_files(self, kind: str, capability_id: str) -> int: ...

    async def move_files(self, kind: str, old_capability_id: str, new_capability_id: str) -> int: ...

    async def was_capability_renamed(self, kind: str, old_capability_id: str, new_capability_id: str) -> bool: ...

    async def delete_file(self, kind: str, file_path: str) -> bool: ...

    async def rename_file(self, kind: str, old_file_path: str, new_file_path: str) -> tuple[int, int] | None: ...


@runtime_checkable
class _CacheConfigProtocol(Protocol):
    enabled: bool


@runtime_checkable
class CapabilityCacheProtocol(Protocol):
    config: _CacheConfigProtocol

    async def get(self, key: str) -> str | None: ...
    async def incr(self, key: str) -> int: ...
    async def delete(self, key: str) -> None: ...
    async def setex(self, key: str, ttl: int, value: str) -> None: ...
    async def try_acquire(self, key: str, value: str, ttl: int) -> bool: ...
    async def release_if_owner(self, key: str, value: str) -> bool: ...
```

- [ ] **Step 2: Write `memory.py` — in-memory reference implementations**

```python
# linktools-ai/src/linktools/ai/capabilities/memory.py
from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InMemoryCapabilityRepository:
    """Reference CapabilityRepositoryProtocol implementation backed by a plain dict.

    Not for production use (no persistence, no concurrency control) — intended for
    linktools-ai's own tests and as a starting point for lightweight consumers
    (e.g. a quant-agent backtest harness) that don't need MySQL.
    """

    rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    _next_id: int = 1

    async def upsert_file(self, *, kind, file_path, content, checksum, updated_by):
        key = (kind, file_path)
        existing = self.rows.get(key)
        if existing is not None and existing.get("checksum") == checksum:
            return existing["id"], existing["version"], False
        version = (existing["version"] + 1) if existing else 1
        file_id = existing["id"] if existing else self._next_id
        if existing is None:
            self._next_id += 1
        self.rows[key] = {
            "id": file_id, "kind": kind, "file_path": file_path, "content": content,
            "checksum": checksum, "version": version, "status": "active",
            "updated_by": updated_by, "updated_at": datetime.now(),
        }
        return file_id, version, True

    async def tombstone_file(self, kind, file_path, *, checksum, updated_by):
        key = (kind, file_path)
        existing = self.rows.get(key)
        if existing is None or existing.get("status") == "deleted":
            return False
        existing["status"] = "deleted"
        existing["checksum"] = checksum
        existing["updated_by"] = updated_by
        existing["version"] += 1
        return True

    async def apply_batch(self, *, kind, capability_id, primary_rel_path, primary_content,
                           supplementary_files, deleted_rel_paths, expected_revision, updated_by):
        raise NotImplementedError("apply_batch is exercised via integration tests with a real repository")

    async def list_capabilities_active(self, kind=None):
        return [row for row in self.rows.values() if row["status"] == "active" and (kind is None or row["kind"] == kind)]

    async def capability_exists(self, kind, capability_id):
        prefix = f"{capability_id}/"
        return any(k[0] == kind and k[1].startswith(prefix) for k in self.rows)

    async def get_file(self, kind, file_path):
        row = self.rows.get((kind, file_path))
        return dict(row) if row and row["status"] == "active" else None

    async def get_file_at_version(self, kind, file_path, version):
        row = self.rows.get((kind, file_path))
        return dict(row) if row and row["version"] == version else None

    async def get_primary_files(self, kind, primary_rel):
        return [row for (k, fp), row in self.rows.items() if k == kind and fp.endswith(f"/{primary_rel}")]

    async def list_files(self, kind, capability_id):
        prefix = f"{capability_id}/"
        return [row for (k, fp), row in self.rows.items() if k == kind and fp.startswith(prefix)]

    async def list_file_states(self, kind, capability_id):
        return await self.list_files(kind, capability_id)

    async def list_files_since(self, since):
        if since is None:
            return list(self.rows.values())
        return [row for row in self.rows.values() if row["updated_at"] >= since]

    async def delete_files_all(self, kind, capability_id):
        prefix = f"{capability_id}/"
        keys = [k for k in self.rows if k[0] == kind and k[1].startswith(prefix)]
        for k in keys:
            del self.rows[k]
        return len(keys)

    async def restore_builtin_files(self, kind, capability_id):
        return 0

    async def move_files(self, kind, old_capability_id, new_capability_id):
        old_prefix, new_prefix = f"{old_capability_id}/", f"{new_capability_id}/"
        keys = [k for k in self.rows if k[0] == kind and k[1].startswith(old_prefix)]
        for k in keys:
            row = self.rows.pop(k)
            new_fp = new_prefix + k[1][len(old_prefix):]
            row["file_path"] = new_fp
            self.rows[(kind, new_fp)] = row
        return len(keys)

    async def was_capability_renamed(self, kind, old_capability_id, new_capability_id):
        return False

    async def delete_file(self, kind, file_path):
        return self.rows.pop((kind, file_path), None) is not None

    async def rename_file(self, kind, old_file_path, new_file_path):
        row = self.rows.pop((kind, old_file_path), None)
        if row is None:
            return None
        row["file_path"] = new_file_path
        row["version"] += 1
        self.rows[(kind, new_file_path)] = row
        return row["id"], row["version"]


@dataclass
class _CacheConfig:
    enabled: bool = True


@dataclass
class InMemoryCapabilityCache:
    """Reference CapabilityCacheProtocol implementation — a plain dict, no TTL enforcement."""

    config: _CacheConfig = field(default_factory=_CacheConfig)
    _store: dict[str, str] = field(default_factory=dict)
    _locks: dict[str, str] = field(default_factory=dict)

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def incr(self, key: str) -> int:
        value = int(self._store.get(key, "0")) + 1
        self._store[key] = str(value)
        return value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        del ttl
        self._store[key] = value

    async def try_acquire(self, key: str, value: str, ttl: int) -> bool:
        del ttl
        if key in self._locks:
            return False
        self._locks[key] = value
        return True

    async def release_if_owner(self, key: str, value: str) -> bool:
        if self._locks.get(key) == value:
            del self._locks[key]
            return True
        return False
```

- [ ] **Step 3: Write the failing test**

```python
# linktools-ai/tests/capabilities/test_memory_repository.py
import pytest

from linktools.ai.capabilities.memory import InMemoryCapabilityCache, InMemoryCapabilityRepository
from linktools.ai.capabilities.protocols import CapabilityCacheProtocol, CapabilityRepositoryProtocol


@pytest.mark.asyncio
async def test_repository_upsert_then_get_roundtrips():
    repo = InMemoryCapabilityRepository()
    file_id, version, changed = await repo.upsert_file(
        kind="skill", file_path="demo/SKILL.md", content="hello", checksum="c1", updated_by="test",
    )
    assert changed is True
    assert version == 1

    row = await repo.get_file("skill", "demo/SKILL.md")
    assert row is not None
    assert row["content"] == "hello"
    assert row["id"] == file_id


@pytest.mark.asyncio
async def test_repository_upsert_same_checksum_is_noop():
    repo = InMemoryCapabilityRepository()
    await repo.upsert_file(kind="skill", file_path="demo/SKILL.md", content="hello", checksum="c1", updated_by="t")
    _, version, changed = await repo.upsert_file(kind="skill", file_path="demo/SKILL.md", content="hello", checksum="c1", updated_by="t")
    assert changed is False
    assert version == 1


@pytest.mark.asyncio
async def test_cache_try_acquire_is_exclusive():
    cache = InMemoryCapabilityCache()
    assert await cache.try_acquire("lock:a", "owner-1", ttl=30) is True
    assert await cache.try_acquire("lock:a", "owner-2", ttl=30) is False
    assert await cache.release_if_owner("lock:a", "owner-1") is True
    assert await cache.try_acquire("lock:a", "owner-2", ttl=30) is True


def test_implementations_satisfy_protocols():
    assert isinstance(InMemoryCapabilityRepository(), CapabilityRepositoryProtocol)
    assert isinstance(InMemoryCapabilityCache(), CapabilityCacheProtocol)
```

- [ ] **Step 4: Run to verify it fails, then implement (already written above), then verify it passes**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/capabilities/test_memory_repository.py -v
```

Expected: PASS. (If `pytest-asyncio` is missing from `linktools-ai/requirements.yml`, add `pytest-asyncio` under `dev-dependencies` and re-run.)

- [ ] **Step 5: Commit**

```bash
git add linktools-ai/src/linktools/ai/capabilities linktools-ai/tests/capabilities
git commit -m "feat(ai): capability repository/cache Protocols + in-memory reference impls"
```

---

### Task 4: Migrate `engine/capabilities/*` (minus `repository.py`)

**Files:**
- Create: `linktools-ai/src/linktools/ai/capabilities/store.py` (from `engine/capabilities/store.py`)
- Create: `linktools-ai/src/linktools/ai/capabilities/skill.py` (from `engine/capabilities/skill.py`)
- Create: `linktools-ai/src/linktools/ai/capabilities/subagent.py` (from `engine/capabilities/subagent.py`)
- Create: `linktools-ai/src/linktools/ai/capabilities/mcp.py` (from `engine/capabilities/mcp.py`)
- Create: `linktools-ai/src/linktools/ai/capabilities/run.py` (from `engine/capabilities/run.py`)
- Test: `linktools-ai/tests/capabilities/test_store_sync.py` (adapted from `sec-smartops-svc`'s `tests/test_capability_store_sync.py`)

**Interfaces:**
- Consumes: `CapabilityRepositoryProtocol`, `CapabilityCacheProtocol`, `InMemoryCapabilityRepository`, `InMemoryCapabilityCache` (Task 3); `HookEvent`, `HookRegistry` from `linktools.ai.support.hooks` (Task 2).
- Produces: `linktools.ai.capabilities.store.CapabilityStore`, `CapabilityConflictError`, `revision_token`; `linktools.ai.capabilities.{skill,subagent,mcp,run}` unchanged public API.

- [ ] **Step 1: Copy the four leaf modules unchanged except their `HookEvent` import**

```bash
cd /workspace/projects/sec-smartops-svc
for f in skill subagent mcp run; do
  cp engine/capabilities/$f.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/capabilities/$f.py
done
```

In each copied file, replace:

```python
from ..infra.hooks import HookEvent
```

with:

```python
from ..support.hooks import HookEvent
```

(`run.py` has no `HookEvent` import — verify with `grep -n HookEvent` on the copy before editing; leave it untouched if absent.)

- [ ] **Step 2: Copy `store.py` and decouple it from `EngineEnvironment`/MySQL/Redis concrete types**

```bash
cp engine/capabilities/store.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/capabilities/store.py
```

Apply these edits to the copy:

```python
# Replace:
from ..environ import environ
from .repository import CapabilityRepository, _with_file_parts
from ..infra.utils import json_ready

if TYPE_CHECKING:
    from ..infra.redis import RedisClient

logger = environ.get_logger("capabilities.store")

# With:
import logging
from .protocols import CapabilityCacheProtocol, CapabilityRepositoryProtocol
from ..support.utils import json_ready

logger = logging.getLogger("linktools.ai.capabilities.store")


def _with_file_parts(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    parts = _split_db_path(str(row.get("file_path") or ""))
    if parts is not None:
        row["capability_id"], row["rel_path"] = parts
    return row
```

(`_split_db_path` already exists earlier in this same file — the inlined `_with_file_parts` above reuses it, removing the only import that pointed at `repository.py`.)

Then change the constructor's type hints from concrete classes to the Protocols:

```python
# Replace:
def __init__(self, repo: "CapabilityRepository", redis: "RedisClient", workspace_root: "Path | None" = None) -> None:

# With:
def __init__(self, repo: "CapabilityRepositoryProtocol", redis: "CapabilityCacheProtocol", workspace_root: "Path | None" = None) -> None:
```

Remove the now-unused `if TYPE_CHECKING:` block if nothing else in the file references it (verify with `grep -n TYPE_CHECKING` on the copy).

- [ ] **Step 3: Port the existing sync test as the verification for this task**

Read `sec-smartops-svc`'s `tests/test_capability_store_sync.py` in full, then create `linktools-ai/tests/capabilities/test_store_sync.py` with the same test bodies, but:
- import `CapabilityStore` from `linktools.ai.capabilities.store` instead of `engine.capabilities.store`
- replace the test file's hand-rolled `_FakeRepo`/`_FakeRedis` classes with `InMemoryCapabilityRepository`/`InMemoryCapabilityCache` from `linktools.ai.capabilities.memory` wherever their behavior matches (keep any test-specific fake that exercises an edge case `InMemoryCapabilityRepository` doesn't cover, e.g. injected DB failures — those stay as local fakes in the test file)

- [ ] **Step 4: Run the ported test suite**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/capabilities/ -v
```

Expected: PASS. Fix any behavioral drift introduced by the `_with_file_parts` inlining or Protocol type-hint change before moving on (these are typing-only changes and must not alter runtime behavior).

- [ ] **Step 5: Commit**

```bash
git add linktools-ai/src/linktools/ai/capabilities linktools-ai/tests/capabilities
git commit -m "feat(ai): migrate capability loading (skill/subagent/mcp/run/store)"
```

---

### Task 5: Define the `AgentEnvironment` Protocol

**Files:**
- Create: `linktools-ai/src/linktools/ai/core/environment.py`
- Test: `linktools-ai/tests/core/test_environment.py`

**Interfaces:**
- Consumes: `HookRegistry` from `linktools.ai.support.hooks` (Task 2).
- Produces: `AgentEnvironment` (`Protocol`) with `hooks` and `get_logger` — this is what `BaseAgent.__init__` (Task 7) type-hints its `environ` parameter as, and what `sec-smartops-svc`'s `EngineEnvironment` will structurally satisfy (Task 9).

- [ ] **Step 1: Read `engine/agent/agent.py`'s actual use of `self.environ`**

```bash
cd /workspace/projects/sec-smartops-svc
grep -n "self\.environ\." engine/agent/*.py
```

Confirm the only members accessed are `.hooks` (everywhere) and `.get_logger(name)` (indirectly, via other modules that also take `environ`). `build_model(self.environ, request.model_type)` in `model_runtime.py` additionally reads `.env`, `.engine_policy()`, and `.debug` — re-run this grep against `engine/agent/model_runtime.py` specifically and fold any additional members found into the Protocol below before writing it (do not guess; the grep output is the source of truth).

- [ ] **Step 2: Write the Protocol**

```python
# linktools-ai/src/linktools/ai/core/environment.py
from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from ..support.hooks import HookRegistry


@runtime_checkable
class AgentEnvironment(Protocol):
    """Minimal contract the Agent runtime needs from its host environment.

    Concrete environments (e.g. sec-smartops-svc's EngineEnvironment) satisfy this
    structurally — no inheritance required.
    """

    hooks: HookRegistry | None

    def get_logger(self, name: str) -> logging.Logger: ...

    def engine_policy(self) -> dict[str, object]: ...
```

(If step 1's grep turns up model-provider fields such as `.env`/`.debug`, add them here as additional Protocol members with the exact same names before proceeding — the Protocol must match what `model_runtime.py` actually dereferences, checked in Task 7.)

- [ ] **Step 3: Write a minimal reference implementation + test**

```python
# linktools-ai/tests/core/test_environment.py
import logging

from linktools.ai.core.environment import AgentEnvironment
from linktools.ai.support.hooks import HookRegistry


class _MinimalEnv:
    def __init__(self) -> None:
        self.hooks = HookRegistry()

    def get_logger(self, name: str) -> logging.Logger:
        return logging.getLogger(name)

    def engine_policy(self) -> dict[str, object]:
        return {}


def test_minimal_env_satisfies_protocol():
    assert isinstance(_MinimalEnv(), AgentEnvironment)
```

- [ ] **Step 4: Run to verify it passes**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/core/test_environment.py -v
```

- [ ] **Step 5: Commit**

```bash
git add linktools-ai/src/linktools/ai/core/environment.py linktools-ai/tests/core/test_environment.py
git commit -m "feat(ai): define AgentEnvironment Protocol"
```

---

### Task 6: Migrate `engine/agent/stores.py` (Session store Protocols + file-backed defaults)

**Files:**
- Create: `linktools-ai/src/linktools/ai/core/stores.py` (from `engine/agent/stores.py`, unchanged body)
- Test: `linktools-ai/tests/core/test_stores.py` (adapted from `sec-smartops-svc` session-store tests)

**Interfaces:**
- Consumes: `model_runtime` and `session_window` types (Task 7 migrates those — see note below on import order).
- Produces: `TranscriptStore`, `HistoryStore`, `ArtifactStore` (`Protocol`), `InMemorySessionStatusStore`, `FileHistoryStore`, `DbHistoryStore` (Protocol-driven, no DB dependency despite the name — verified in the design doc), `LocalArtifactStore`, `ReadOnlyArtifactStore`, `ArchiveArtifactStore`.

- [ ] **Step 1: Confirm `stores.py` has zero direct `environ`/DB imports**

```bash
cd /workspace/projects/sec-smartops-svc
grep -n "^from\|^import" engine/agent/stores.py
```

Expected: only stdlib, `pydantic_ai`, and relative imports of `.model_runtime` and `.session_window` — no `..environ`, no `sqlalchemy`. If this grep turns up anything else, stop and re-check the design doc's boundary assumption before continuing (this file was inspected during planning and found clean; a mismatch means the codebase changed since).

- [ ] **Step 2: Copy the file as-is (this task has no import fixes to make)**

```bash
cp engine/agent/stores.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/stores.py
```

Note: this file imports `.model_runtime` and `.session_window`, which don't exist in `linktools-ai/core/` yet until Task 7. That's fine — Task 7 lands in the same PR sequence before this module is exercised end-to-end; this task's test (step 3) only imports the pieces of `stores.py` that don't require those two modules to be import-clean (`TranscriptStore`, `HistoryStore`, `ArtifactStore`, `InMemorySessionStatusStore`). If Python's import system complains because the module-level imports at the top of `stores.py` pull in `.model_runtime`/`.session_window` immediately, do Task 7's copy step first, then return here — do not reorder the plan, just copy both files before running this task's test if the import graph forces it.

- [ ] **Step 3: Port a stores-only test**

Read `sec-smartops-svc`'s `tests/test_agent_session.py` and `tests/test_agent_artifact.py` for existing coverage of `FileHistoryStore`/`LocalArtifactStore`/`InMemorySessionStatusStore`. Create `linktools-ai/tests/core/test_stores.py` with the subset of those tests that exercise `stores.py` directly (not the full `Session`/`Agent` stack), updating imports from `engine.agent.stores` to `linktools.ai.core.stores`.

- [ ] **Step 4: Run to verify it passes**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/core/test_stores.py -v
```

- [ ] **Step 5: Commit**

```bash
git add linktools-ai/src/linktools/ai/core/stores.py linktools-ai/tests/core/test_stores.py
git commit -m "feat(ai): migrate session store Protocols and file-backed defaults"
```

---

### Task 7: Migrate the remaining `engine/agent/*` core runtime modules

**Files:**
- Create: `linktools-ai/src/linktools/ai/core/{registry,session,session_coordination,session_window,model_runtime,mcp_client,prompt,builtin_tools,artifact,skill_view,runtime,agent}.py`
- Test: `linktools-ai/tests/core/test_agent_runtime_kernel.py` (ported from `sec-smartops-svc`'s `tests/test_agent_runtime_kernel.py`)

**Interfaces:**
- Consumes: `AgentEnvironment` (Task 5), `TranscriptStore`/`HistoryStore`/`ArtifactStore`/`InMemorySessionStatusStore` (Task 6), `HookEvent`/`HookRegistry` (Task 2), `SkillCapability`/`SubagentCapability`/`HookedMCPCapability`/`RuntimeRunCapability` (Task 4).
- Produces: `AgentSpec`, `MCPServerSpec`, `SkillSpec`, `SubagentSpec`, `SpecSource`, `AgentRegistry`, `SkillRegistry`, `SubagentRegistry`, `MCPRegistry` (from `registry.py`); `Session`, `FileSession`, `DbSession`, `RunContext`, `SessionTranscript`, `SessionTranscriptHead`, `FileSessionSpec`, `SessionTurn` (from `session.py`); `AgentExecutionContext`, `AgentKernel` (from `runtime.py`); `BaseAgent`, `LlmAgent`, `RuntimeAgent`, `SubAgent` (from `agent.py`).

- [ ] **Step 1: Copy the six modules with no `environ` coupling, unchanged**

```bash
cd /workspace/projects/sec-smartops-svc
for f in session session_coordination session_window mcp_client prompt builtin_tools artifact skill_view runtime; do
  cp engine/agent/$f.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/$f.py
done
```

For each copied file, fix any `..infra.hooks` / `..capabilities.X` import to point at the new locations (only `builtin_tools.py` imports `HookEvent`, and `agent.py`/`skill_view.py` import from `..capabilities.*` — verify per-file with `grep -n "^from \.\.\|^from \." <file>` before editing, only touch lines that actually need it):

```python
# builtin_tools.py
from ..infra.hooks import HookEvent
# becomes
from ..support.hooks import HookEvent
```

- [ ] **Step 2: Copy and fix `registry.py`**

```bash
cp engine/agent/registry.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/registry.py
```

Apply:

```python
# Replace:
from ..environ import environ
from ..infra.config import (
    load_yaml_file as _load_yaml_file,
    load_yaml_text as _load_yaml_text,
    load_markdown_file as _load_markdown_file,
    load_markdown_text as _load_markdown_text,
    as_str_dict as _as_str_dict,
)

if TYPE_CHECKING:
    from ..capabilities.store import CapabilityStore

logger = environ.get_logger("agent.registry")

# With:
import logging
from ..support.config import (
    load_yaml_file as _load_yaml_file,
    load_yaml_text as _load_yaml_text,
    load_markdown_file as _load_markdown_file,
    load_markdown_text as _load_markdown_text,
    as_str_dict as _as_str_dict,
)

if TYPE_CHECKING:
    from ..capabilities.store import CapabilityStore

logger = logging.getLogger("linktools.ai.core.registry")
```

- [ ] **Step 3: Copy and fix `model_runtime.py`**

```bash
cp engine/agent/model_runtime.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/model_runtime.py
```

Apply:

```python
# Replace:
from ..environ import EngineEnvironment, environ
from ..infra.config import load_yaml_file
from ..infra.utils import resolve_ref as _resolve_ref, safe_filename

logger = environ.get_logger("agent.model_runtime")

# With:
import logging
from .environment import AgentEnvironment
from ..support.config import load_yaml_file
from ..support.utils import resolve_ref as _resolve_ref, safe_filename

logger = logging.getLogger("linktools.ai.core.model_runtime")
```

Then replace every remaining `EngineEnvironment` type annotation in the file with `AgentEnvironment`:

```bash
sed -i 's/\bEngineEnvironment\b/AgentEnvironment/g' /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/model_runtime.py
```

Re-run `grep -n "AgentEnvironment\." linktools-ai/src/linktools/ai/core/model_runtime.py` and cross-check every attribute accessed there (`.env`, `.debug`, `.engine_policy()`, etc.) is present on the `AgentEnvironment` Protocol from Task 5 — extend the Protocol if this turns up a gap (this is the check flagged in Task 5 step 2).

- [ ] **Step 4: Copy and fix `agent.py`**

```bash
cp engine/agent/agent.py /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/agent.py
```

Apply:

```python
# Replace:
from ..environ import EngineEnvironment
from ..infra.hooks import HookEvent
from ..infra.utils import model_type as _model_type, call_id as _call_id
from ..capabilities.run import RuntimeRunCapability
from ..capabilities.mcp import HookedMCPCapability
from ..capabilities.skill import SkillCapability
from ..capabilities.subagent import SubagentCapability

# With:
from .environment import AgentEnvironment
from ..support.hooks import HookEvent
from ..support.utils import model_type as _model_type, call_id as _call_id
from ..capabilities.run import RuntimeRunCapability
from ..capabilities.mcp import HookedMCPCapability
from ..capabilities.skill import SkillCapability
from ..capabilities.subagent import SubagentCapability
```

```bash
sed -i 's/\bEngineEnvironment\b/AgentEnvironment/g' /workspace/projects/linktools/linktools-ai/src/linktools/ai/core/agent.py
```

- [ ] **Step 5: Create `linktools-ai/src/linktools/ai/core/__init__.py`**

```python
"""Domain-agnostic Agent/Session/Registry runtime."""
```

- [ ] **Step 6: Port the runtime-kernel test**

Read `sec-smartops-svc`'s `tests/test_agent_runtime_kernel.py` in full. Create `linktools-ai/tests/core/test_agent_runtime_kernel.py` with the same test bodies, updating every `from engine.agent.X import Y` to `from linktools.ai.core.X import Y`, and replacing the import of `AgentRuntime` from `engine.secops.runtime` (that class stays in `sec-smartops-svc`, it's the secops-specific bootstrap) with a local minimal stand-in built from `AgentKernel` + `_Registry` (already defined in the existing test file) — read the existing test's `_Registry`/`AgentRuntime` usage first to see exactly what surface needs stubbing.

- [ ] **Step 7: Run the full linktools-ai test suite**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/ -v
```

Expected: PASS. Fix import errors iteratively — every failure here is either a missed `environ`/`infra` reference (fix per steps 1-4's pattern) or a missing Protocol member (extend `AgentEnvironment` per step 3's note).

- [ ] **Step 8: Commit**

```bash
git add linktools-ai/src/linktools/ai/core linktools-ai/tests/core
git commit -m "feat(ai): migrate Agent/Registry/Runtime core"
```

---

### Task 8: Package-level verification for `linktools-ai`

**Files:**
- Modify: `linktools-ai/requirements.yml` (add `pydantic-ai`, `pytest-asyncio` if not already present from earlier tasks)

**Interfaces:**
- Consumes: everything from Tasks 1-7.
- Produces: a green, independently buildable `linktools-ai` wheel.

- [ ] **Step 1: Full test run**

```bash
cd /workspace/projects/linktools
python -m pytest linktools-ai/tests/ -v
```

Expected: all PASS.

- [ ] **Step 2: Build the wheel to catch packaging errors early**

```bash
python manage.py build linktools-ai
```

Expected: wheel produced under `linktools-ai/dist/` with no errors. If `tool.linktools.dependencies` in `requirements.yml` is missing `pydantic-ai`, add it and rebuild.

- [ ] **Step 3: Commit**

```bash
git add linktools-ai/requirements.yml
git commit -m "chore(ai): finalize linktools-ai packaging dependencies"
```

---

### Task 9: sec-smartops-svc — adapt concrete classes to the new Protocols

**Files:**
- Modify: `/workspace/projects/sec-smartops-svc/requirements.txt`
- Modify: `/workspace/projects/sec-smartops-svc/engine/environ.py` (add `get_logger`/`hooks`/`engine_policy` already present — verify it satisfies `AgentEnvironment` as-is)
- Modify: `/workspace/projects/sec-smartops-svc/engine/capabilities/repository.py` (no code change — it already implements every method the Protocol requires; this step only removes the now-dead `_with_file_parts` export if nothing else imports it)
- Test: `/workspace/projects/sec-smartops-svc/tests/test_capability_store_sync.py` (run unchanged against the adapter)

**Interfaces:**
- Consumes: `linktools.ai.capabilities.protocols.{CapabilityRepositoryProtocol, CapabilityCacheProtocol}`, `linktools.ai.core.environment.AgentEnvironment` (published by the `linktools-ai` package built in Task 8).

- [ ] **Step 1: Add the dependency**

```bash
cd /workspace/projects/sec-smartops-svc
echo "-e /workspace/projects/linktools/linktools-ai" >> requirements.txt
pip install -r requirements.txt
python -c "from linktools.ai.core.agent import BaseAgent; print('ok')"
```

Expected: `ok`.

- [ ] **Step 2: Verify `EngineEnvironment` already satisfies `AgentEnvironment`**

```bash
python -c "
from engine.environ import EngineEnvironment
from linktools.ai.core.environment import AgentEnvironment
env = EngineEnvironment.from_process()
assert isinstance(env, AgentEnvironment), 'EngineEnvironment does not satisfy AgentEnvironment'
print('ok')
"
```

Expected: `ok`. If this assertion fails, the failure message names the missing attribute — add it to `engine/environ.py`'s `EngineEnvironment` (it almost certainly already has `hooks`, `get_logger`, `engine_policy` per the class body read during planning; a failure here means Task 5/7 found an extra Protocol member that `EngineEnvironment` doesn't expose under that exact name — reconcile the naming instead of adding a shim).

- [ ] **Step 3: Verify `CapabilityRepository` and `RedisClient` already satisfy the capability Protocols**

```bash
python -c "
from engine.capabilities.repository import CapabilityRepository
from engine.infra.redis import RedisClient
from linktools.ai.capabilities.protocols import CapabilityRepositoryProtocol, CapabilityCacheProtocol
print(issubclass(CapabilityRepository, CapabilityRepositoryProtocol))
print(issubclass(RedisClient, CapabilityCacheProtocol))
"
```

Expected: both print `True` (structural typing via `@runtime_checkable`). If `False`, `Protocol.__instancecheck__` will report which method is missing when you additionally run `isinstance(repo_instance, CapabilityRepositoryProtocol)` on a real instance — reconcile by renaming/adding the missing method on the `sec-smartops-svc` side (the Protocol was derived from this exact class's call sites in Task 3, so a mismatch means a method was renamed since planning — re-check `engine/capabilities/repository.py`).

- [ ] **Step 4: Run the existing capability store test against the real classes**

```bash
python -m pytest tests/test_capability_store_sync.py -v
```

Expected: PASS unchanged (this test doesn't import `engine.capabilities.store` yet — that import switch happens in Task 10).

- [ ] **Step 5: Commit**

```bash
git add requirements.txt
git commit -m "chore(agent): depend on linktools-ai, verify Protocol conformance"
```

---

### Task 10: sec-smartops-svc — cut over imports and delete the migrated directories

**Files:**
- Delete: `engine/agent/` (all files except none — the whole directory is superseded)
- Delete: `engine/capabilities/store.py`, `engine/capabilities/skill.py`, `engine/capabilities/subagent.py`, `engine/capabilities/mcp.py`, `engine/capabilities/run.py` (keep `repository.py`, it stays as the MySQL adapter)
- Delete: `engine/infra/hooks.py`, `engine/infra/utils.py`, `engine/infra/config.py`, `engine/infra/workspace.py` (superseded by `linktools.ai.support.*`) — **only** after confirming nothing outside `engine/agent`/`engine/capabilities` imports them (see step 1)
- Modify: every file across `engine/` and `tests/` importing from `engine.agent.*` or `engine.capabilities.{store,skill,subagent,mcp,run}`

**Interfaces:**
- Consumes: `linktools-ai` (Task 9).

- [ ] **Step 1: Find every external consumer of the infra modules slated for deletion**

```bash
cd /workspace/projects/sec-smartops-svc
grep -rln "infra\.hooks\|infra\.utils\|infra\.config\|infra\.workspace" engine tests --include="*.py" | grep -v "^engine/agent/\|^engine/capabilities/"
```

For every file listed, keep the corresponding `engine/infra/*.py` file (don't delete it) and instead only delete the ones with zero remaining consumers outside `engine/agent`/`engine/capabilities`. This is a discovery step — do not delete anything yet.

- [ ] **Step 2: Rewrite imports repo-wide**

```bash
grep -rl "from engine\.agent\." --include="*.py" engine tests | xargs sed -i \
  -e 's/from engine\.agent\./from linktools.ai.core./g'
grep -rl "from \.\.agent\." --include="*.py" engine | xargs sed -i \
  -e 's/from \.\.agent\./from linktools.ai.core./g'
grep -rl "from \.agent\." --include="*.py" engine | xargs sed -i \
  -e 's/from \.agent\./from linktools.ai.core./g'
grep -rl "from engine\.capabilities\.\(store\|skill\|subagent\|mcp\|run\)\b" --include="*.py" engine tests | xargs sed -i \
  -e 's/from engine\.capabilities\.\(store\|skill\|subagent\|mcp\|run\)/from linktools.ai.capabilities.\1/g'
```

Run `grep -rn "engine\.agent\|engine\.capabilities\.\(store\|skill\|subagent\|mcp\|run\)\b" engine tests --include="*.py"` afterward — expect zero remaining hits.

- [ ] **Step 3: Update `EngineEnvironment.get_worker_registry`/`get_stage_registry`/etc. and any other factory that constructs `CapabilityStore` with the MySQL repo/Redis client**

```bash
grep -n "CapabilityStore(" engine/**/*.py
```

For each call site found, confirm it passes `CapabilityRepository`/`RedisClient` instances positionally (unaffected by the Protocol rename — Python doesn't check types at call time) — no code change needed there, only the `import` line at the top of that file needs the same `linktools.ai.capabilities.store` rewrite from step 2.

- [ ] **Step 4: Delete the superseded directories/files**

```bash
git rm -r engine/agent
git rm engine/capabilities/store.py engine/capabilities/skill.py engine/capabilities/subagent.py engine/capabilities/mcp.py engine/capabilities/run.py
# Only for infra files confirmed to have zero remaining consumers in step 1:
git rm engine/infra/hooks.py engine/infra/utils.py engine/infra/config.py engine/infra/workspace.py
```

- [ ] **Step 5: Run the full existing test suite**

```bash
python -m pytest tests/ -v
```

Expected: all PASS — this is the regression baseline referenced in the design doc. Fix any remaining import or Protocol-mismatch failures before proceeding; do not skip or weaken a failing test to force green.

- [ ] **Step 6: Commit**

```bash
git add -A
git commit -m "refactor(agent): cut over to linktools-ai, remove migrated engine/agent and engine/capabilities modules"
```

---

## Self-Review Notes

- **Spec coverage:** every section of the design doc (`support`, `capabilities` Protocols, `core` runtime, `AgentEnvironment`, sec-smartops-svc adapter, deletion + import cutover, testing) maps to Task 1-10 above.
- **Ambiguity flagged inline:** Task 5 Step 1 and Task 7 Step 3 both instruct re-verifying the exact `AgentEnvironment` attribute surface against the live codebase before finalizing the Protocol, since `model_runtime.py`'s full dependency on `EngineEnvironment` (beyond `.hooks`/`.get_logger`) wasn't exhaustively enumerated during planning — this is a deliberate "verify against the grep, not the plan's guess" checkpoint rather than a placeholder.
- **infra/hooks.py `HookEvent` trim:** Task 2 Step 2 narrows the enum to the 8 values agent/capabilities code references; `sec-smartops-svc` keeps its own broader pipeline-specific `HookEvent` (or extends the trimmed one) for `POST_DISPATCH`/`POST_DECISION`/etc. — those stay defined in `sec-smartops-svc` since Task 10 only deletes `engine/infra/hooks.py` if step 1's grep shows zero remaining consumers; if secops pipeline code also uses `HookEvent`, that file survives instead, importing the trimmed `HookEvent` from `linktools.ai.support.hooks` and extending it locally.
