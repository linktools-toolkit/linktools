#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for workspace skill bootstrap and memory paths."""

from pathlib import Path

import pytest
from linktools.ai.adapter import RuntimeMemoryStore, build_in_memory_runtime
from linktools.ai.spec import AgentSpec
from linktools.ai.workspace import RuntimePersistenceConfig, Workspace, open_workspace_runtime


@pytest.mark.asyncio
async def test_workspace_default_agent_declares_local_skills(tmp_path: Path) -> None:
    mcp = tmp_path / ".linktools" / "mcp" / "local"
    mcp.parent.mkdir(parents=True)
    mcp.write_text('{"args":[],"command":"echo","id":"local","revision":1}', encoding="utf-8")

    workspace = Workspace.load(tmp_path)
    async with open_workspace_runtime(
        workspace,
        config=RuntimePersistenceConfig.in_memory(namespace=workspace.workspace_id),
        model="test-model",
    ) as runtime:
        definition = await runtime.compile_agent()

    assert isinstance(definition.spec, AgentSpec)
    assert tuple((item.provider, item.id) for item in definition.spec.capabilities) == (("mcp", "local"),)
    assert any(item.provider == "mcp" for item in definition.effective_capabilities)


@pytest.mark.asyncio
async def test_runtime_memory_store_accepts_harness_scoped_paths() -> None:
    runtime = build_in_memory_runtime(namespace="memory-regression")
    await runtime.initialize()
    try:
        store = RuntimeMemoryStore(runtime.persistence, tenant_id="tenant", namespace="workspace")
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
        await runtime.close()
