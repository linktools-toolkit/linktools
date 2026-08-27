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
from linktools.ai.runtime._memory import RuntimeMemoryStore
from linktools.ai.spec import AgentSpec, AgentSpecCodec, MCPServerSpec, MCPServerSpecCodec
from linktools.ai.storage import StorageOverlay
from linktools.ai.workspace import Workspace


@pytest.mark.asyncio
async def test_runtime_memory_store_accepts_harness_scoped_paths() -> None:
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
            "workspace/memory/MEMORY.md",
            "remember commit-writer",
            expected_version=None,
        )
        assert await store.list_paths("workspace/memory/", limit=10) == [
            "workspace/memory/MEMORY.md"
        ]
        result = await store.search(
            "workspace/memory/",
            "commit-writer",
            limit=10,
            max_files=10,
            max_chars=1_000,
            max_file_chars=1_000,
        )
        assert [match.path for match in result.matches] == [
            "workspace/memory/MEMORY.md"
        ]
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
        tenant_id="tenant-a",
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
