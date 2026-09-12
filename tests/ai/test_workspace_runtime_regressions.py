#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for workspace RuntimeState composition."""

import pytest
from linktools.ai.asset import (
    AssetStore,
    DirectoryAssetBackend,
    PrefixAssetPathAdapter,
)
from linktools.ai.capability import CapabilityGroup
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import Runtime, RuntimeDomain, RuntimeState
from linktools.ai.runtime._context import RuntimeContext
from linktools.ai.runtime._harness_memory import build_harness_memory
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.spec import AgentSpec, AgentSpecCodec, MCPServerSpec, MCPServerSpecCodec
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import Workspace
from pydantic_ai_harness.memory import Memory


@pytest.mark.asyncio
async def test_runtime_memory_store_implements_harness_path_storage() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-regression", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(RuntimeDomain.MEMORY),
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
async def test_memory_capability_is_harness_owned_over_runtime_state() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-capability", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(RuntimeDomain.MEMORY),
            namespace="memory-capability",
            tenant_id="tenant",
            execution_id="execution",
            memory_scope="workspace",
        )
        capability = build_harness_memory(
            store,
            selected_tool_names=(
                "delete_memory",
                "read_memory",
                "search_memory",
                "write_memory",
            ),
            capability_id="linktools-memory",
        )

        assert isinstance(capability, Memory)
        assert capability.id == "linktools-memory"
        assert capability.inject_memory is False
        assert capability.store is not store
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_missing_delete_uses_harness_mutation_contract() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-missing-delete", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(RuntimeDomain.MEMORY),
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
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-sequence-gap", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state.object_store(RuntimeDomain.MEMORY),
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
        assert "storage_version" not in created_record.metadata
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
        assert "storage_version" not in updated_record.metadata
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
    agent_path = assets_root / "agents" / "default"
    skill_path = assets_root / "skills" / "review" / "SKILL.md"
    mcp_path = assets_root / "mcp" / "local"
    agent_path.parent.mkdir(parents=True)
    skill_path.parent.mkdir(parents=True)
    mcp_path.parent.mkdir(parents=True)
    agent_path.write_bytes(AgentSpecCodec().encode(AgentSpec("default", model="gpt-test")))
    skill_path.write_text(
        "---\nname: review\ndescription: Review changes.\n---\n\nReview changes.\n",
        encoding="utf-8",
    )
    mcp_path.write_bytes(MCPServerSpecCodec().encode(MCPServerSpec("local", "echo")))

    source = DirectoryAssetBackend(
        str(assets_root),
        path_adapter=PrefixAssetPathAdapter(
            {"agent": "agents", "skill": "skills", "mcp": "mcp"}
        ),
        kinds=("agent", "skill", "mcp"),
    )
    store = AssetStore(StorageOverlay(source))
    await store.initialize()

    frozen = await CapabilityGroup.from_store("workspace", store).freeze()

    assert [(item.kind, item.id) for item in frozen] == [
        ("agent", "default"),
        ("mcp", "local"),
        ("skill", "review"),
    ]


@pytest.mark.asyncio
async def test_workspace_session_survives_cold_restart(tmp_path) -> None:
    workspace = Workspace.load(tmp_path, workspace_id="workspace")
    models = ModelRegistry.openai(model="gpt-test")
    async with Runtime.open(workspace, models=models) as runtime:
        assert runtime.tenant_id == "default"
        assert runtime.default_principal.tenant_id == "default"
        created = await runtime.agent("default").create_session("remember")
        agent_created = await runtime.agent("default").create_session("remember-agent")
        assert (
            await runtime.session.history(
                created.session_id,
                principal=runtime.default_principal,
            )
        ).items == ()

    async with Runtime.open(
        workspace,
        context=RuntimeContext(None, tenant_id="tenant-a"),
        models=models,
    ) as runtime:
        assert runtime.tenant_id == "tenant-a"
        assert runtime.default_principal.tenant_id == "tenant-a"
        await runtime.agent("default").create_session("custom-tenant")

    async with Runtime.open(workspace, models=models) as runtime:
        loaded = await runtime.session.get(
            created.session_id,
            principal=runtime.default_principal,
        )
        agent_loaded = await runtime.session.get(
            agent_created.session_id,
            principal=runtime.default_principal,
        )
        history = await runtime.session.history(
            created.session_id,
            principal=runtime.default_principal,
        )

    assert loaded.session_id == created.session_id
    assert agent_loaded.session_id == agent_created.session_id
    assert history.items == ()
