#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded History metadata reads and recorder-local invalidations."""

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    HmacCursorSigner, Principal, ResourceKind, ResourceRef,
    TenantAuthorizationPolicy, agent_conversation_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._history_projection import StepExecutionHistoryReader
from linktools.ai.runtime._history_service import DefaultExecutionHistoryService
from linktools.ai.runtime._runtime_history import RuntimeHistory
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.service_api import UsageReadCutoff
from linktools.ai.runtime.state import RuntimeDomain, RuntimeRetentionMode
from linktools.ai.runtime.state._step_contracts import AgentRunRecord
from linktools.ai.runtime.state._steps import (
    InMemoryStepArchive, RuntimeAgentRunStore, StagingAgentRunStore,
)

from .test_model_interaction_lifecycle_paging import (
    _Executions, _HistoryStore, _interaction, _reader,
    _record, _running_interaction, _terminal,
)
from .test_live_history_readback_integration import _Models
from ._runtime_test_helpers import _UsageFunctionModel


@pytest.mark.asyncio
async def test_model_metadata_suffix_and_active_reread_keep_identity_and_hide_content() -> None:
    reader, executions, store = _reader()
    store.model_interaction_history_available = True

    async def no_tree(*args: object, **kwargs: object) -> object:
        raise AssertionError("known-execution reads must not discover descendants")

    executions.list_children = no_tree
    boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert boundary.cutoffs == (UsageReadCutoff("root", 1, 2),)
    assert boundary.durable_cutoffs == (UsageReadCutoff("root", 1, 0),)
    assert boundary.local_staging_available
    first = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=2, limit=1,
    )
    assert [(item.model_request_seq, item.status, item.usage) for item in first] == [(1, "RUNNING", None)]
    assert all(not item.content_included and item.request == {} and item.response is None for item in first)
    run_id = next(iter(store.staged))
    store.staged[run_id][1] = _terminal(store.staged[run_id][1])
    store.staged[run_id][3] = _running_interaction("root", 1, 3, datetime.now(timezone.utc))
    active = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=1,
    )
    assert [(item.model_request_seq, item.status) for item in active] == [(1, "CANCELLED")]
    suffix = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=1, through_model_request_seq=2,
    )
    assert [item.model_request_seq for item in suffix] == [2]


@pytest.mark.asyncio
async def test_model_metadata_archive_handoff_and_future_staged_rows_respect_fixed_range() -> None:
    reader, _, store = _reader()
    store.model_interaction_history_available = True
    run_id = next(iter(store.staged))
    store.handoff_during_snapshot[run_id] = [_interaction(run_id, 1)]
    values = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=1,
    )
    assert [item.status for item in values] == ["CANCELLED"]
    store.interactions[run_id].append(_interaction(run_id, 2))
    store.staged[run_id].clear()
    store.staged[run_id][3] = _running_interaction("root", 1, 3, datetime.now(timezone.utc))
    values = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=2,
    )
    assert [item.model_request_seq for item in values] == [1, 2]
    boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert boundary.cutoffs == (UsageReadCutoff("root", 1, 3),)
    assert boundary.durable_cutoffs == (UsageReadCutoff("root", 1, 2),)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("missing_identity", "conflicting_terminal", "wrong_owner"))
async def test_model_metadata_rejects_inconsistent_source_facts(failure: str) -> None:
    reader, _, store = _reader()
    run_id = next(iter(store.staged))
    if failure == "missing_identity":
        del store.staged[run_id][1]
    elif failure == "conflicting_terminal":
        store.staged[run_id][1] = _terminal(store.staged[run_id][1])
        store.interactions[run_id] = [_interaction(run_id, 1)]
    else:
        store.staged[run_id][1] = replace(store.staged[run_id][1], agent_run_id="other")
    with pytest.raises(AIError) as raised:
        await reader.read_model_interaction_metadata(
            "root", tenant_id="tenant", agent_run_seq=1,
            after_model_request_seq=0, through_model_request_seq=1,
        )
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_model_metadata_remote_history_does_not_claim_local_activity() -> None:
    original, executions, local = _reader()
    del original
    archive = _HistoryStore()
    archive.runs.update(local.runs)
    run_id = next(iter(local.runs))
    archive.interactions[run_id] = [_interaction(run_id, 1)]
    local.runs.clear()
    local.staged.clear()
    local.interactions[run_id] = archive.interactions[run_id]
    local.model_interaction_history_available = True
    reader = StepExecutionHistoryReader(
        namespace="history", executions=executions, store=archive,
        cursor_signer=HmacCursorSigner("metadata", b"metadata"), staging_store=local,
    )
    boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert not boundary.local_staging_available
    assert boundary.durable_history_available
    assert boundary.cutoffs == boundary.durable_cutoffs == (UsageReadCutoff("root", 1, 1),)
    archive.runs.clear()
    with pytest.raises(AIError) as corrupt:
        await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    archive.interactions.clear()
    missing = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert not missing.local_staging_available
    assert not missing.durable_history_available
    assert missing.cutoffs == ()


@pytest.mark.asyncio
async def test_model_metadata_disabled_archive_does_not_certify_empty_remote_history() -> None:
    execution = _record("root")
    execution.agent_run_seq = 0
    reader = StepExecutionHistoryReader(
        namespace="history", executions=_Executions(execution), store=_HistoryStore(),
        cursor_signer=HmacCursorSigner("metadata", b"metadata"),
        durable_history_available=False,
    )
    boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert boundary.cutoffs == ()
    assert not boundary.local_staging_available and not boundary.durable_history_available
    assert await reader.subscribe_model_interactions("root", tenant_id="tenant") is None


def _run_store() -> RuntimeAgentRunStore:
    return RuntimeAgentRunStore(
        StagingAgentRunStore(),
        conversation_archive=InMemoryStepArchive(RuntimeDomain.CONVERSATION),
        execution_archive=None, recovery_archive=None,
        conversation_retention=RuntimeRetentionMode.VOLATILE,
        execution_retention=RuntimeRetentionMode.VOLATILE,
        recovery_retention=RuntimeRetentionMode.VOLATILE,
    )


@pytest.mark.asyncio
async def test_model_metadata_subscription_preserves_capture_races_and_isolates_executions() -> None:
    store = _run_store()
    await store.initialize()
    conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root")
    subscription = store.subscribe_model_interactions(conversation_id)
    second = store.subscribe_model_interactions(conversation_id)
    unrelated = store.subscribe_model_interactions("another-conversation")
    generation = subscription.generation
    running = _running_interaction("root", 1, 1, datetime.now(timezone.utc))
    try:
        await store.register_agent_run(AgentRunRecord(
            running.agent_run_id, agent_conversation_id=conversation_id,
            metadata={"agent_run_seq": "1"},
        ))
        store.stage_model_interaction(running)
        store.stage_model_interaction(_terminal(running))
        latest = await asyncio.wait_for(subscription.wait(generation), 1)
        assert latest > generation
        assert second.generation == latest
        assert unrelated.generation == 0
        pending = asyncio.create_task(subscription.wait(latest))
        await asyncio.sleep(0)
        assert not pending.done()
        await subscription.close()
        with pytest.raises(AIError) as closed:
            await pending
        assert not closed.value.retryable
        store.stage_model_interaction(_running_interaction("root", 1, 2, datetime.now(timezone.utc)))
        assert subscription.generation == latest
        assert second.generation > latest
    finally:
        await subscription.close()
        await second.close()
        await unrelated.close()
        await store.preflight_close()
        with pytest.raises(AIError) as closed:
            store.subscribe_model_interactions(conversation_id)
        assert not closed.value.retryable
        await store.close()


@pytest.mark.asyncio
async def test_model_metadata_staging_only_source_is_not_reported_as_durable() -> None:
    store = _run_store()
    await store.initialize()
    execution = _record("root")
    execution.agent_run_seq = 0
    reader = StepExecutionHistoryReader(
        namespace="history", executions=_Executions(execution),
        store=store.read_store(RuntimeDomain.EXECUTION), staging_store=store,
        cursor_signer=HmacCursorSigner("metadata", b"metadata"),
    )
    try:
        before = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
        assert not before.local_staging_available and not before.durable_history_available
        running = _running_interaction("root", 1, 1, datetime.now(timezone.utc))
        execution.agent_run_seq = 1
        await store.register_agent_run(AgentRunRecord(
            running.agent_run_id,
            agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root"),
            metadata={"agent_run_seq": "1"},
        ))
        store.stage_model_interaction(running)
        boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
        assert boundary.local_staging_available and not boundary.durable_history_available
        assert boundary.durable_cutoffs == (UsageReadCutoff("root", 1, 0),)
        values = await reader.read_model_interaction_metadata(
            "root", tenant_id="tenant", agent_run_seq=1,
            after_model_request_seq=0, through_model_request_seq=1,
        )
        assert [item.status for item in values] == ["RUNNING"]
    finally:
        await store.preflight_close()
        await store.close()


class _AuthorizedExecutions(_Executions):
    tenant_id = "tenant"

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        if tenant_id == self.tenant_id and execution_id == "root":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None


@pytest.mark.asyncio
async def test_model_metadata_public_history_authorizes_every_port() -> None:
    reader, executions, store = _reader()
    store.model_interaction_history_available = True
    service = DefaultExecutionHistoryService(
        _AuthorizedExecutions(executions.root), TenantAuthorizationPolicy("tenant"), reader,
    )
    history = RuntimeHistory(service, tenant_id="tenant")
    owner = Principal("caller", "tenant", "service")
    boundary = await history.capture_model_interaction_cutoffs("root", principal=owner)
    assert boundary.cutoffs == (UsageReadCutoff("root", 1, 2),)
    values = await history.read_model_interaction_metadata(
        "root", principal=owner, agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=2, limit=1,
    )
    assert len(values) == 1 and not values[0].content_included
    other = Principal("caller", "other", "service")
    operations = (
        lambda: history.capture_model_interaction_cutoffs("root", principal=other),
        lambda: history.read_model_interaction_metadata(
            "root", principal=other, agent_run_seq=1,
            after_model_request_seq=0, through_model_request_seq=2,
        ),
        lambda: history.subscribe_model_interactions("root", principal=other),
    )
    for operation in operations:
        with pytest.raises(AIError) as denied:
            await operation()
        assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
async def test_model_metadata_live_and_archived_backend_contract(backend: str, tmp_path: Path) -> None:
    entered = asyncio.Event()
    release = asyncio.Event()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("private response")])

    def storage() -> RuntimeStorage:
        if backend == "memory":
            return RuntimeStorage.in_memory()
        if backend == "filesystem":
            return RuntimeStorage.filesystem(tmp_path / "state")
        return RuntimeStorage.sqlite(tmp_path / "state.db")

    group = CapabilityGroup[None]("metadata")
    group.agent("default", model="default", allow_tools=())
    async with Runtime.open(
        "metadata-live", models=_Models(_UsageFunctionModel(model)),
        storage=storage(), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("private prompt")
        principal = runtime.default_principal
        await asyncio.wait_for(entered.wait(), 5)
        subscription = await runtime.history.subscribe_model_interactions(execution.execution_id, principal=principal)
        assert subscription is not None
        try:
            generation = subscription.generation
            live = await runtime.history.capture_model_interaction_cutoffs(execution.execution_id, principal=principal)
            assert live.local_staging_available
            assert live.cutoffs == (UsageReadCutoff(execution.execution_id, 1, 1),)
            values = await runtime.history.read_model_interaction_metadata(
                execution.execution_id, principal=principal, agent_run_seq=1,
                after_model_request_seq=0, through_model_request_seq=1,
            )
            assert [(item.status, item.usage, item.content_included) for item in values] == [("RUNNING", None, False)]
            assert values[0].request == {} and values[0].response is None
            release.set()
            await execution.wait(timeout_seconds=5)
            assert await asyncio.wait_for(subscription.wait(generation), 5) > generation
            final = await runtime.history.capture_model_interaction_cutoffs(execution.execution_id, principal=principal)
            assert final.durable_history_available
            assert final.cutoffs == final.durable_cutoffs == live.cutoffs
            terminal = await runtime.history.read_model_interaction_metadata(
                execution.execution_id, principal=principal, agent_run_seq=1,
                after_model_request_seq=0, through_model_request_seq=1,
            )
            assert terminal[0].status == "SUCCEEDED"
            assert terminal[0].request == {} and terminal[0].response is None
        finally:
            release.set()
            await subscription.close()
    if backend != "memory":
        async with RuntimeHistory.open("metadata-live", storage=storage()) as history:
            boundary = await history.capture_model_interaction_cutoffs(execution.execution_id, principal=principal)
            assert not boundary.local_staging_available and boundary.durable_history_available
            assert boundary.durable_cutoffs == final.durable_cutoffs
            assert await history.subscribe_model_interactions(execution.execution_id, principal=principal) is None
            restored = await history.read_model_interaction_metadata(
                execution.execution_id, principal=principal, agent_run_seq=1,
                after_model_request_seq=0, through_model_request_seq=1,
            )
            assert restored == terminal
