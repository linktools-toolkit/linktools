#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for workspace RuntimeState composition."""

import pytest

from linktools.ai.adapter import RuntimeMemoryStore
from linktools.ai.model import ModelRegistry
from linktools.ai.runtime import RuntimeDomain, RuntimeState
from linktools.ai.workspace import Workspace, open_workspace_runtime, trusted_workspace_principal


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
async def test_workspace_session_survives_cold_restart(tmp_path) -> None:
    workspace = Workspace.load(tmp_path)
    principal = trusted_workspace_principal(workspace.workspace_id)
    models = ModelRegistry.openai(model="gpt-test")
    async with open_workspace_runtime(workspace, models=models) as runtime:
        created = await runtime.create_session("remember", principal=principal)

    async with open_workspace_runtime(workspace, models=models) as runtime:
        loaded = await runtime.session.get(created.session_id, principal=principal)

    assert loaded is not None
    assert loaded.session_id == created.session_id
