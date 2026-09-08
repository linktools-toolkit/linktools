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
from linktools.ai.runtime import Runtime, RuntimeContext, RuntimeDomain, RuntimeState
from linktools.ai.runtime._capabilities import _SelectedMemory
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.spec import AgentSpec, AgentSpecCodec, MCPServerSpec, MCPServerSpecCodec
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import Workspace
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage


@pytest.mark.asyncio
async def test_runtime_memory_store_uses_flat_logical_files() -> None:
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
        await store.write(
            "MEMORY",
            "remember commit-writer",
            expected_version=None,
        )
        result = await store.search("commit-writer", limit=10)
        assert [match.file for match in result.matches] == ["MEMORY.md"]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_capability_preserves_append_and_replace_semantics() -> None:
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
        capability = _SelectedMemory(
            store,
            selected_tool_names=("write_memory",),
            id="memory",
        )

        def context(call_id: str) -> RunContext[None]:
            return RunContext(
                deps=None,
                model=TestModel(),
                usage=RunUsage(),
                run_id="run",
                tool_call_id=call_id,
            )

        assert (
            await capability._write_memory(context("create"), "first")
        )["status"] == "created"
        assert (
            await capability._write_memory(context("append"), "second")
        )["status"] == "appended"
        assert (
            await capability._write_memory(
                context("replace"),
                "updated",
                old_text="first\nsecond\n",
            )
        )["status"] == "updated"

        result = await store.read("MEMORY.md", max_chars=100)
        assert result is not None
        assert result.content == "updated"
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_memory_missing_delete_commits_a_not_found_receipt() -> None:
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

        deleted = await store.delete("notes", expected_version=None)

        assert deleted.status == "not_found"
        assert deleted.version is None
        created = await store.write(
            "notes",
            "created after the missing read",
            expected_version=None,
        )
        assert created.status == "created"
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
    workspace = Workspace.load(tmp_path)
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
