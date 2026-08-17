#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for workspace RuntimeState composition."""

import pytest

from linktools.ai.adapter import RuntimeMemoryStore
from linktools.ai.capability import SkillCapabilityProvider
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.spec import AgentCapabilityRef, AgentSpec, AgentSpecCodec, builtin_asset_bindings
from linktools.ai.workspace import CapabilitySource, Workspace, open_workspace_runtime
from linktools.ai.workspace._factory import _build_asset_repository, _merge_default_capabilities


class _MCPSourceProvider:
    provider = "mcp"


def _mcp_source() -> CapabilitySource:
    bindings = {binding.kind: binding for binding in builtin_asset_bindings()}
    return CapabilitySource(bindings["mcp"], _MCPSourceProvider())


def test_default_capability_merge_preserves_explicit_values_and_order() -> None:
    explicit = (
        AgentCapabilityRef(
            "custom",
            "prod",
            revision=7,
            required=False,
            config={"mode": "strict"},
        ),
        AgentCapabilityRef("custom", "prod", required=True),
        AgentCapabilityRef("other", "explicit", required=False),
    )
    discovered = (
        AgentCapabilityRef("custom", "prod", required=True),
        AgentCapabilityRef("custom", "auto-only", required=True),
    )

    merged = _merge_default_capabilities(explicit, discovered)

    assert merged[:2] == (
        AgentCapabilityRef(
            "custom",
            "prod",
            revision=7,
            required=True,
            config={"mode": "strict"},
        ),
        explicit[1],
    )
    assert merged[2] == explicit[2]
    assert merged[3] == discovered[1]


@pytest.mark.asyncio
async def test_runtime_memory_store_accepts_harness_scoped_paths() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="memory-regression", tenant_id="tenant")
    try:
        store = RuntimeMemoryStore(
            state.memory,
            object_store=state._object_store(RuntimeDomain.MEMORY),
            namespace="memory-regression",
            tenant_id="tenant",
            execution_id="execution",
            memory_scope="workspace",
        )
        await store.write("workspace/memory/MEMORY.md", "remember commit-writer", expected_version=None)
        assert await store.list_paths("workspace/memory/", limit=10) == ["workspace/memory/MEMORY.md"]
        result = await store.search(
            "workspace/memory/",
            "commit-writer",
            limit=10,
            max_files=10,
            max_chars=1_000,
            max_file_chars=1_000,
        )
        assert [match.path for match in result.matches] == ["workspace/memory/MEMORY.md"]
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_workspace_assets_use_kind_scoped_paths(tmp_path) -> None:
    assets_root = tmp_path / ".linktools"
    agent_path = assets_root / "agents" / "default"
    skill_path = assets_root / "skills" / "review" / "SKILL.md"
    mcp_path = assets_root / "mcp" / "local"
    agent_path.parent.mkdir(parents=True)
    skill_path.parent.mkdir(parents=True)
    mcp_path.parent.mkdir(parents=True)
    agent_path.write_bytes(
        AgentSpecCodec().encode(
            AgentSpec("default", 1, "gpt-test", (), "assistant_text", 1)
        )
    )
    skill_path.write_text(
        "---\nname: review\ndescription: Review changes.\n---\n\nReview changes.\n",
        encoding="utf-8",
    )
    mcp_path.write_text(
        '{"id":"local","revision":1,"command":"echo"}',
        encoding="utf-8",
    )

    assets = await _build_asset_repository(
        Workspace.load(tmp_path),
        asset=None,
        sources=(
            CapabilitySource(
                {binding.kind: binding for binding in builtin_asset_bindings()}["skill"],
                SkillCapabilityProvider(),
            ),
            _mcp_source(),
        ),
    )

    agents = await assets.list(kind="agent")
    skills = await assets.list(kind="skill")
    mcp_servers = await assets.list(kind="mcp")
    assert [item.ref.id for item in agents.items] == ["default"]
    assert [item.ref.id for item in skills.items] == ["review"]
    assert [item.ref.id for item in mcp_servers.items] == ["local"]


@pytest.mark.asyncio
async def test_workspace_session_survives_cold_restart(tmp_path) -> None:
    workspace = Workspace.load(tmp_path)
    models = ModelRegistry.openai(model="gpt-test")
    async with open_workspace_runtime(workspace, models=models) as runtime:
        assert runtime.tenant_id == "default"
        assert runtime.default_principal.tenant_id == "default"
        created = await runtime.create_session("remember")
        agent_created = await runtime.agent("default").create_session("remember-agent")
        assert (
            await runtime.session.history(
                created.session_id,
                principal=runtime.default_principal,
            )
        ).items == ()

    async with open_workspace_runtime(workspace, tenant_id="tenant-a", models=models) as runtime:
        assert runtime.tenant_id == "tenant-a"
        assert runtime.default_principal.tenant_id == "tenant-a"
        await runtime.create_session("custom-tenant")

    async with open_workspace_runtime(workspace, models=models) as runtime:
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

    assert loaded is not None
    assert loaded.session_id == created.session_id
    assert agent_loaded is not None
    assert agent_loaded.session_id == agent_created.session_id
    assert history.items == ()
