#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One host declaration supplies new execution memory without rewriting captures."""

import inspect
from pathlib import Path

import pytest
from pydantic_ai.capabilities import CombinedCapability, WrapperCapability

from linktools.ai.capability import ExecutionCapability, MemoryCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime import Agent, Runtime, RuntimeFeatures, Session
from linktools.ai.runtime.agent_execution import AgentExecutionPort
from linktools.commands.ai._common import _open_local_runtime
from linktools.ai.workspace import Workspace

from ._runtime_test_helpers import model_features
from .test_explicit_runtime_features import _FeatureModels


def _features(scope: str | None) -> RuntimeFeatures:
    features = model_features(_FeatureModels())
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True))
    if scope is not None:
        features.capability(MemoryCapability(scope=scope))
    return features


@pytest.mark.parametrize("value", [None, "", " leading", "trailing ", "a\x00b", 1])
def test_memory_capability_validates_scope(value: object) -> None:
    with pytest.raises(AIError) as raised:
        MemoryCapability(scope=value)  # type: ignore[arg-type]
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


def test_memory_declaration_requires_runtime_and_invocations_have_no_scope() -> None:
    capability = MemoryCapability()
    assert capability.scope == "default"
    with pytest.raises(AIError) as raised:
        capability.for_agent(object())
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    for owner, methods in ((Agent, ("start", "run", "plan")),
                           (Session, ("start", "run", "plan")),
                           (AgentExecutionPort, ("start",))):
        for method in methods:
            assert "memory_scope" not in inspect.signature(getattr(owner, method)).parameters


@pytest.mark.asyncio
@pytest.mark.parametrize("scope", [None, "default", "workspace-memory"])
@pytest.mark.parametrize("owner", ["agent", "session", "lazy-session"])
@pytest.mark.parametrize("method", ["start", "run", "plan"])
async def test_memory_scope_is_injected_across_agent_and_session_entries(scope: str | None, owner: str, method: str) -> None:
    async with Runtime.open("memory-entries", storage=PersistenceStorage.in_memory(),
                            features=_features(scope)) as runtime:
        agent = runtime.agents.get()
        if owner == "agent":
            handle = agent
        else:
            session = agent.session("session")
            if owner == "session":
                handle = session
            else:
                await agent.create_session("session")
                handle = await runtime.sessions.get("session")
        result = await getattr(handle, method)("hello", idempotency_key="memory-entry")
        if method == "start":
            result = await result.wait()
        assert result.result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(
            result.result.execution_id, tenant_id=runtime.tenant_id,
        )
        assert record.memory_scope == scope


@pytest.mark.asyncio
@pytest.mark.parametrize("shape", ["direct", "combined", "wrapper"])
@pytest.mark.parametrize("duplicate", [False, True])
async def test_memory_declaration_rejects_agent_scope_and_duplicates(shape: str, duplicate: bool) -> None:
    def wrap(capability):
        if shape == "combined":
            return CombinedCapability([capability])
        return WrapperCapability(capability) if shape == "wrapper" else capability

    features = model_features(_FeatureModels())
    first = wrap(MemoryCapability())
    features.capability(first, id=first.id or "memory-one", scope="runtime" if duplicate else "agent")
    if duplicate:
        second = wrap(MemoryCapability(scope="other"))
        features.capability(second, id=second.id or "memory-two")
    with pytest.raises(AIError) as raised:
        async with Runtime.open("memory-declarations", storage=PersistenceStorage.in_memory(), features=features):
            pytest.fail("invalid declaration was accepted")
    assert raised.value.code is (ErrorCode.CAPABILITY_CONFLICT if duplicate else ErrorCode.CAPABILITY_RESOLUTION_INVALID)


@pytest.mark.asyncio
async def test_retry_retains_admitted_memory_after_host_configuration_changes(tmp_path: Path) -> None:
    storage = PersistenceStorage.filesystem(tmp_path)
    async with Runtime.open("memory-retry", storage=storage, features=_features("original")) as runtime:
        source = await runtime.agents.get().start("first", idempotency_key="first")
        assert (await source.wait()).result.status is ExecutionStatus.SUCCEEDED
        execution_id = source.execution_id
    async with Runtime.open("memory-retry", storage=storage, features=_features("changed")) as runtime:
        source = await runtime.executions.get(execution_id)
        retry = await source.retry("again", idempotency_key="again")
        assert (await retry.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(retry.execution_id, tenant_id=runtime.tenant_id)
        assert record.memory_scope == "original"
        fresh = await runtime.agents.get().start("new", idempotency_key="new")
        assert (await fresh.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(fresh.execution_id, tenant_id=runtime.tenant_id)
        assert record.memory_scope == "changed"


@pytest.mark.asyncio
async def test_local_cli_composition_registers_default_memory(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    async with _open_local_runtime(workspace, model=_FeatureModels()) as runtime:
        assert runtime.agents.get().execution_port.memory_scope == "default"


@pytest.mark.asyncio
async def test_memory_configuration_is_captured_once_by_runtime() -> None:
    capability = MemoryCapability(scope="admitted-default")
    features = _features(None)
    features.capability(capability)
    async with Runtime.open("memory-capture", storage=PersistenceStorage.in_memory(), features=features) as runtime:
        capability._scope = "mutated-authoring-object"
        execution = await runtime.agents.get().start("hello")
        assert (await execution.wait()).result.status is ExecutionStatus.SUCCEEDED
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record.memory_scope == "admitted-default"
