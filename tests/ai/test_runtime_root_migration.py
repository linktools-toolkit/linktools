#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A local SQLite root never silently replaces recognizable filesystem state."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.runs import AgentRunRecord

from .test_step_projection import _execution


NAMESPACE, TENANT = "legacy-reader", "tenant"


def _contents(root: Path) -> dict[str, bytes]:
    return {path.relative_to(root).as_posix(): path.read_bytes() for path in root.rglob("*") if path.is_file() and path.name not in {"state.lock", "publication.lock"}}


async def _write_retained_run(storage: PersistenceStorage, run_id: str) -> AgentRunRecord:
    record = AgentRunRecord(run_id)
    async with Persistence.open(NAMESPACE, storage=storage, tenant_id=TENANT) as persistence:
        execution = replace(_execution(), revision=1, producer_generation=1, producer_claim_id="owner-claim", restart_checkpoints=False)
        repository = persistence.execution.executions
        await repository.create_execution(execution)
        await persistence.run_store.read_store(PersistenceDomain.TRACE).register_agent_run(
            record, execution_id="execution",
        )
    return record


async def _retained_run(persistence: Persistence, run_id: str) -> AgentRunRecord | None:
    return await persistence.run_store.read_store(PersistenceDomain.TRACE).get_agent_run(agent_run_id=run_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("manifest", ("group", "members", "direct_member"))
async def test_from_root_rejects_recognizable_filesystem_state_without_touching_it(tmp_path: Path, manifest: str) -> None:
    root = tmp_path / "retained"
    layouts = {
        "group": ".state-groups/scope/manifest.json",
        "members": "checkpoint/namespace/tenant/manifest.json",
        "direct_member": "manifest.json",
    }
    path = root / layouts[manifest]
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"format": "linktools-ai-state-group" if manifest == "group" else "linktools-ai-state"}))
    before = _contents(root)
    with pytest.raises(AIError) as rejected:
        PersistenceStorage.from_root(root)
    assert rejected.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert "filesystem" in str(rejected.value).lower()
    assert _contents(root) == before
    assert not (root / "runtime.db").exists()


@pytest.mark.asyncio
async def test_from_root_does_not_treat_objects_cache_or_unrelated_manifests_as_state(tmp_path: Path) -> None:
    examples = {
        "objects/execution/n/t/manifest.json": {"format": "linktools-ai-state"},
        "cache/.state-groups/scope/manifest.json": {"format": "linktools-ai-state-group"},
        ".state-groups/unrelated/manifest.json": {"format": "application-settings"},
        "execution/n/t/manifest.json": {"format": "application-data"},
        "manifest.json": {"format": ["linktools-ai-state"]},
        ".state-groups/non-object/manifest.json": ["not a state manifest"],
    }
    for name, value in examples.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value), encoding="utf-8")
    malformed = tmp_path / "task/n/t/manifest.json"
    malformed.parent.mkdir(parents=True)
    malformed.write_text("ordinary non-JSON content", encoding="utf-8")
    before = _contents(tmp_path)
    await _write_retained_run(PersistenceStorage.from_root(tmp_path), "sqlite-run")
    assert (tmp_path / "runtime.db").is_file()
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())


@pytest.mark.asyncio
async def test_existing_sqlite_root_keeps_its_explicit_source_with_filesystem_state_present(tmp_path: Path) -> None:
    root = tmp_path / "mixed"
    path = root / ".state-groups/scope/manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"format": "linktools-ai-state-group"}))
    before = _contents(root)
    sqlite_run = await _write_retained_run(PersistenceStorage.sqlite(root / "runtime.db"), "sqlite-run")
    async with Persistence.open(NAMESPACE, storage=PersistenceStorage.from_root(root),
                                tenant_id=TENANT, read_only=True) as persistence:
        assert await _retained_run(persistence, "sqlite-run") == sqlite_run
        assert await _retained_run(persistence, "run") is None
    assert all((root / name).read_bytes() == value for name, value in before.items())
