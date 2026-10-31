#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Regression tests for workspace PersistenceStorage composition."""
from ._runtime_test_helpers import model_features
from linktools.ai.model import create_model

from ._runtime_test_helpers import workspace_features

import pytest
from linktools.ai.persistence import Persistence
from linktools.ai.asset import (
    AssetStore,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import AssetCapability
from linktools.ai.runtime import RuntimeFeatures

from linktools.ai.runtime import Runtime
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.core import ExecutionContext
from linktools.ai.capability import MemoryCapability
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import Workspace
from pydantic_ai_harness.memory import Memory


@pytest.mark.asyncio
async def test_runtime_memory_store_implements_harness_path_storage() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="memory-regression", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(PersistenceDomain.MEMORY),
            namespace="memory-regression",
            tenant_id="tenant",
            execution_id="execution",
            memory_scope="workspace",
        )
        created = await store.write(
            "memory/MEMORY.md",
            "remember commit-writer",
            expected_version=None,
        )

        assert created.version is not None
        assert created.existed is False
        assert await store.list_paths("memory/", limit=10) == ["memory/MEMORY.md"]
        stored = await store.read("memory/MEMORY.md", max_chars=100)
        assert stored is not None
        assert stored.content == "remember commit-writer"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_capability_is_harness_owned_over_runtime_storage() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="memory-capability", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(PersistenceDomain.MEMORY),
            namespace="memory-capability",
            tenant_id="tenant",
            execution_id="execution",
            memory_scope="workspace",
        )
        capability = MemoryCapability().bind(
            (
                "delete_memory",
                "read_memory",
                "search_memory",
                "write_memory",
            ),
            store=store,
        )

        assert isinstance(capability, Memory)
        assert capability.id == "linktools.ai.memory"
        assert capability.inject_memory is False
        assert capability.store is not store
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_missing_delete_uses_harness_mutation_contract() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="memory-missing-delete", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(PersistenceDomain.MEMORY),
            namespace="memory-missing-delete",
            tenant_id="tenant",
            execution_id="execution",
            memory_scope="workspace",
        )

        deleted = await store.delete("memory/notes.md", expected_version=None)

        assert deleted.version is None
        assert deleted.existed is False
        assert deleted.replayed is False
        created = await store.write(
            "memory/notes.md",
            "created after the missing read",
            expected_version=None,
        )
        assert created.version is not None
        assert created.existed is False
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_store_remains_writable_across_missing_delete_receipts() -> None:
    state = Persistence(PersistenceStorage.in_memory())
    await state.initialize(namespace="memory-sequence-gap", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(PersistenceDomain.MEMORY),
            namespace="memory-sequence-gap",
            tenant_id="tenant",
            execution_id="execution",
            memory_scope="workspace",
        )

        first_delete = await store.delete("memory/notes.md", expected_version=None)
        second_delete = await store.delete("memory/notes.md", expected_version=None)
        assert first_delete.existed is False
        assert second_delete.existed is False

        created = await store.write(
            "memory/notes.md",
            "created",
            expected_version=None,
        )
        created_record = await store._record("memory/notes.md")
        assert created_record is not None
        assert created_record.revision == 1
        assert "storage_revision" not in created_record.metadata
        current = await store.read("memory/notes.md", max_chars=100)
        assert current is not None
        assert current.version == created.version
        assert current.content == "created"

        updated = await store.write(
            "memory/notes.md",
            "updated",
            expected_version=current.version,
        )
        assert updated.existed is True
        updated_record = await store._record("memory/notes.md")
        assert updated_record is not None
        assert updated_record.revision == 2
        assert "storage_revision" not in updated_record.metadata
        deleted = await store.delete(
            "memory/notes.md",
            expected_version=updated.version,
        )
        assert deleted.existed is True
        assert await store.read("memory/notes.md", max_chars=100) is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_workspace_store_loads_kind_scoped_declarations(tmp_path) -> None:
    assets_root = tmp_path / ".linktools"
    agent_path = assets_root / "agents" / "default" / "AGENT.md"
    skill_path = assets_root / "skills" / "review" / "SKILL.md"
    mcp_path = assets_root / "mcp" / "local" / "mcp.json"
    agent_path.parent.mkdir(parents=True)
    skill_path.parent.mkdir(parents=True)
    mcp_path.parent.mkdir(parents=True)
    agent_path.write_text(
        "---\nmodel: gpt-test\n---\n",
        encoding="utf-8",
    )
    skill_path.write_text(
        "---\nname: review\ndescription: Review changes.\n---\n\nReview changes.\n",
        encoding="utf-8",
    )
    mcp_path.write_text('{"command":"echo"}', encoding="utf-8")

    source = DirectoryAssetBackend(
        str(assets_root),
        path_adapter=PrefixAssetPathAdapter(
            {"agent": "agents", "skill": "skills", "mcp": "mcp"}
        ),
        kinds=("agent", "skill", "mcp"),
    )
    store = AssetStore(StorageOverlay(source))
    await store.initialize()

    asset_features = RuntimeFeatures("workspace")
    asset_features.capability(AssetCapability(store), scope="runtime")
    snapshot = await asset_features.capture()

    assert [(item.kind, item.id) for item in snapshot.contributions] == [
        ("agent", "default"),
        ("mcp", "local"),
        ("skill", "review"),
        ("capability", "runtime.assets"),
    ]


def _workspace_runtime_storage(workspace: Workspace) -> PersistenceStorage:
    return PersistenceStorage.from_root(workspace.storage_root / "runtime")


@pytest.mark.asyncio
async def test_workspace_session_survives_cold_restart(tmp_path) -> None:
    workspace = Workspace.load(tmp_path)
    models = create_model(api='openai', model="gpt-test")
    from linktools.ai.capability import ExecutionCapability

    features = workspace_features("workspace", workspace=workspace)
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    async with Persistence.open('default', storage=_workspace_runtime_storage(workspace)) as persistence_225, Runtime.attach(persistence=persistence_225, features=model_features(models, features)) as runtime:
        assert runtime.tenant_id == "default"
        assert runtime.default_principal.tenant_id == "default"
        created = await runtime.agents.get("default").create_session("remember")
        agent_created = await runtime.agents.get("default").create_session("remember-agent")
        assert (
            await runtime.sessions.history(
                created.session_id,
                principal=runtime.default_principal,
            )
        ).items == ()

    async with Persistence.open('default', storage=_workspace_runtime_storage(workspace), tenant_id='tenant-a') as persistence_241, Runtime.attach(persistence=persistence_241, context=ExecutionContext(None), features=model_features(models, features)) as runtime:
        assert runtime.tenant_id == "tenant-a"
        assert runtime.default_principal.tenant_id == "tenant-a"
        await runtime.agents.get("default").create_session("custom-tenant")

    async with Persistence.open('default', storage=_workspace_runtime_storage(workspace)) as persistence_251, Runtime.attach(persistence=persistence_251, features=model_features(models, features)) as runtime:
        loaded = await runtime.sessions.get(
            created.session_id,
            principal=runtime.default_principal,
        )
        agent_loaded = await runtime.sessions.get(
            agent_created.session_id,
            principal=runtime.default_principal,
        )
        history = await runtime.sessions.history(
            created.session_id,
            principal=runtime.default_principal,
        )

    assert loaded.session_id == created.session_id
    assert agent_loaded.session_id == agent_created.session_id
    assert history.items == ()
