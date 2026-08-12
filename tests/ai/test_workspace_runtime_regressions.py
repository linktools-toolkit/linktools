#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression tests for workspace skill bootstrap and memory paths."""

from datetime import datetime, timezone
from pathlib import Path

import pytest
from linktools.ai.adapter import RuntimeMemoryStore, build_in_memory_runtime
from linktools.ai.core import (
    AuthorizationAction,
    ExecutionLineageKind,
    ExecutionStatus,
    Principal,
    ResourceKind,
    ResourceRef,
    SessionStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import CreateSessionRequest, ExecutionRecord, LocalExecutionBackend, SessionRecord
from linktools.ai.spec import AgentSpec
from linktools.ai import RuntimeStorage
from linktools.ai.workspace import Workspace, open_workspace_runtime, trusted_workspace_principal
from pydantic_ai_harness.step_persistence import InMemoryStepStore


class _MCPRuntime:
    fingerprint = "a" * 64

    async def connect(self, server_id: str) -> object:
        del server_id
        raise AssertionError("MCP process startup is not part of bootstrap")

    async def toolsets(self, servers: object, *, principal: object, execution: object) -> tuple[object, ...]:
        del servers, principal, execution
        raise AssertionError("MCP toolset materialization is not part of bootstrap")


@pytest.mark.asyncio
async def test_workspace_default_agent_declares_configured_mcp_servers(tmp_path: Path) -> None:
    mcp = tmp_path / ".linktools" / "assets" / "mcp" / "local"
    mcp.parent.mkdir(parents=True)
    mcp.write_text('{"args":[],"command":"echo","id":"local","revision":1}', encoding="utf-8")

    workspace = Workspace.load(tmp_path)
    async with open_workspace_runtime(
        workspace,
        storage=RuntimeStorage.memory(),
        model="test-model",
        mcp_runtime=_MCPRuntime(),
    ) as runtime:
        definition = await runtime.compile_agent()
        principal = trusted_workspace_principal(workspace.workspace_id)
        session = await runtime.session.create(
            definition.digest,
            CreateSessionRequest(principal, "session", "create-session"),
        )
        with pytest.raises(AIError) as foreign_error:
            await runtime.session.create(
                definition.digest,
                CreateSessionRequest(Principal("principal", "foreign-tenant"), "foreign-session", "create-foreign-session"),
            )

    assert isinstance(definition.spec, AgentSpec)
    assert session.session_id == "session"
    assert foreign_error.value.code is ErrorCode.AUTHORIZATION_DENIED
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


@pytest.mark.asyncio
async def test_memory_namespace_is_independent_from_tenant_and_path() -> None:
    runtime = build_in_memory_runtime(namespace="runtime-namespace")
    await runtime.initialize()
    try:
        first = RuntimeMemoryStore(runtime.persistence, tenant_id="tenant", namespace="memory-a")
        second = RuntimeMemoryStore(runtime.persistence, tenant_id="tenant", namespace="memory-b")
        await first.write("notes/shared.md", "first", expected_version=None)
        await second.write("notes/shared.md", "second", expected_version=None)

        assert (await first.read("notes/shared.md", max_chars=100)).content == "first"
        assert (await second.read("notes/shared.md", max_chars=100)).content == "second"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_workspace_tenant_policy_rejects_a_self_consistent_foreign_tenant() -> None:
    policy = TenantAuthorizationPolicy("workspace-tenant")
    foreign = Principal("principal", "foreign-tenant")
    resource = ResourceRef(ResourceKind.EXECUTION, "execution", "foreign-tenant")

    with pytest.raises(AIError) as error:
        await policy.authorize(foreign, AuthorizationAction.EXECUTION_RUN, resource)

    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_local_reconcile_uses_workspace_tenant_not_persistence_namespace(tmp_path: Path) -> None:
    runtime = build_in_memory_runtime(namespace="runtime-namespace")
    await runtime.initialize()
    now = datetime.now(timezone.utc)
    try:
        for tenant_id in ("workspace-tenant", "foreign-tenant"):
            await runtime.persistence.conversation.create(
                SessionRecord("session", tenant_id, "principal", "binding", SessionStatus.OPEN, 0, 0, None, {}, now, now, None)
            )
            await runtime.persistence.execution.create(
                ExecutionRecord(
                    execution_id="execution",
                    tenant_id=tenant_id,
                    session_id="session",
                    binding_digest="binding",
                    parent_execution_id=None,
                    root_execution_id="execution",
                    source_execution_id=None,
                    base_execution_id=None,
                    lineage_kind=ExecutionLineageKind.RUN,
                    status=ExecutionStatus.STARTED,
                    revision=0,
                    event_sequence=0,
                    agent_run_sequence=1,
                    result_ref=None,
                    result_digest=None,
                    error_code=None,
                    safe_error_details={},
                    created_at=now,
                    updated_at=now,
                )
            )
        backend = LocalExecutionBackend(
            runtime.persistence,
            InMemoryStepStore(),
            object(),
            {},
            tenant_id="workspace-tenant",
            execution_root=tmp_path,
        )

        await backend.reconcile()

        workspace_execution = await runtime.persistence.execution.get("execution", tenant_id="workspace-tenant")
        foreign_execution = await runtime.persistence.execution.get("execution", tenant_id="foreign-tenant")
        assert workspace_execution.status is ExecutionStatus.STARTED
        assert foreign_execution.status is ExecutionStatus.STARTED
    finally:
        await runtime.close()
