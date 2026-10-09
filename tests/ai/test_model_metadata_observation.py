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
from linktools.ai.runtime._watch_cursor import decode_execution_watch_cursor
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.service_api import UsageReadCutoff
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._step_contracts import AgentRunRecord

from .test_model_interaction_lifecycle_paging import (
    _Executions, _HistoryStore, _interaction, _reader,
    _record, _running_interaction, _running_record, _terminal,
)
from .test_live_history_readback_integration import _Models
from ._runtime_test_helpers import _UsageFunctionModel, _wait_for_committed


@pytest.mark.asyncio
async def test_model_metadata_suffix_and_active_reread_keep_identity_and_hide_content() -> None:
    reader, executions, store = _reader()

    async def no_tree(*args: object, **kwargs: object) -> object:
        raise AssertionError("known-execution reads must not discover descendants")

    executions.list_children = no_tree
    boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert boundary.cutoffs == (UsageReadCutoff("root", 1, 2),)
    assert boundary.durable_cutoffs == boundary.cutoffs
    assert boundary.durable_history_available and not boundary.local_staging_available
    first = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=2, limit=1,
    )
    assert [(item.model_request_seq, item.status, item.usage) for item in first] == [(1, "RUNNING", None)]
    assert all(not item.content_included and item.request == {} and item.response is None for item in first)
    run_id = next(iter(store.interactions))
    store.interactions[run_id][0] = _terminal(store.interactions[run_id][0])
    store.interactions[run_id].append(_running_record("root", 1, 3, datetime.now(timezone.utc)))
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
async def test_model_metadata_completion_and_later_admission_respect_fixed_range() -> None:
    reader, _, store = _reader()
    run_id = next(iter(store.interactions))
    store.before_reads[run_id] = [_interaction(run_id, 1)]
    values = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=1,
    )
    assert [item.status for item in values] == ["CANCELLED"]
    store.interactions[run_id][1] = _interaction(run_id, 2)
    store.interactions[run_id].append(_running_record("root", 1, 3, datetime.now(timezone.utc)))
    values = await reader.read_model_interaction_metadata(
        "root", tenant_id="tenant", agent_run_seq=1,
        after_model_request_seq=0, through_model_request_seq=2,
    )
    assert [item.model_request_seq for item in values] == [1, 2]
    boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
    assert boundary.cutoffs == (UsageReadCutoff("root", 1, 3),)
    assert boundary.durable_cutoffs == boundary.cutoffs


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("missing_identity", "duplicate_identity", "wrong_owner"))
async def test_model_metadata_rejects_inconsistent_source_facts(failure: str) -> None:
    reader, _, store = _reader()
    run_id = next(iter(store.interactions))
    if failure == "missing_identity":
        del store.interactions[run_id][0]
    elif failure == "duplicate_identity":
        store.interactions[run_id] = [_interaction(run_id, 1), _interaction(run_id, 1)]
    else:
        store.interactions[run_id][0] = replace(store.interactions[run_id][0], agent_run_id="other")
    with pytest.raises(AIError) as raised:
        await reader.read_model_interaction_metadata(
            "root", tenant_id="tenant", agent_run_seq=1,
            after_model_request_seq=0, through_model_request_seq=2,
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
    local.interactions[run_id] = archive.interactions[run_id]
    reader = StepExecutionHistoryReader(
        namespace="history", executions=executions, store=archive,
        cursor_signer=HmacCursorSigner("metadata", b"metadata"), notifications=local,
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


@pytest.mark.asyncio
async def test_model_metadata_subscription_preserves_capture_races_and_isolates_executions() -> None:
    from linktools.ai.runtime.state._contracts import ExecutionHistoryHeadRecord, ExecutionHistoryState

    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="history", tenant_id="tenant")
    repository = storage.execution.executions
    await repository.state_store.mutate(lambda transaction: repository.insert_history_head_in_transaction(
        transaction, ExecutionHistoryHeadRecord("root", ExecutionHistoryState.OPEN, 0, None),
    ))
    store = storage.run_store
    conversation_id = agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root")
    subscription = store.subscribe_model_interactions(conversation_id)
    second = store.subscribe_model_interactions(conversation_id)
    unrelated = store.subscribe_model_interactions("another-conversation")
    generation = subscription.generation
    running = replace(_running_interaction("root", 1, 1, datetime.now(timezone.utc)),
                      request_context=None, request_envelope_digest=None)
    try:
        await store.register_agent_run(AgentRunRecord(
            running.agent_run_id, agent_conversation_id=conversation_id,
            metadata={"agent_run_seq": "1"},
        ))
        assert await asyncio.wait_for(subscription.wait(generation), 1) > generation
        generation = subscription.generation
        store.stage_model_interaction(running)
        assert subscription.generation == generation
        await store.flush_execution_projection(running.agent_run_id, execution_id="root")
        latest = await asyncio.wait_for(subscription.wait(generation), 1)
        assert latest > generation and second.generation == latest
        store.stage_model_interaction(_terminal(running))
        assert subscription.generation == latest
        await store.flush_execution_projection(running.agent_run_id, execution_id="root")
        latest = await asyncio.wait_for(subscription.wait(latest), 1)
        assert second.generation == latest and unrelated.generation == 0
        pending = asyncio.create_task(subscription.wait(latest))
        await asyncio.sleep(0)
        assert not pending.done()
        await subscription.close()
        with pytest.raises(AIError) as closed:
            await pending
        assert not closed.value.retryable
        store.stage_model_interaction(replace(running, model_request_seq=2, step_index=2))
        assert subscription.generation == second.generation == latest
        await store.flush_execution_projection(running.agent_run_id, execution_id="root")
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
        await storage.close()


@pytest.mark.asyncio
async def test_model_metadata_excludes_uncommitted_staging() -> None:
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="history", tenant_id="tenant")
    execution = _record("root")
    reader = StepExecutionHistoryReader(
        namespace="history", executions=_Executions(execution),
        store=storage.run_store.read_store(RuntimeDomain.EXECUTION), notifications=storage.run_store,
        cursor_signer=HmacCursorSigner("metadata", b"metadata"),
    )
    try:
        running = _running_interaction("root", 1, 1, datetime.now(timezone.utc))
        await storage.run_store.register_agent_run(AgentRunRecord(
            running.agent_run_id,
            agent_conversation_id=agent_conversation_id(namespace="history", tenant_id="tenant", execution_id="root"),
            metadata={"agent_run_seq": "1"},
        ))
        storage.run_store.stage_model_interaction(running)
        boundary = await reader.capture_model_interaction_cutoffs("root", tenant_id="tenant")
        assert not boundary.local_staging_available and not boundary.durable_history_available
        assert boundary.cutoffs == boundary.durable_cutoffs == ()
    finally:
        await storage.close()


class _AuthorizedExecutions(_Executions):
    tenant_id = "tenant"

    async def get_header(self, execution_id: str, *, tenant_id: str) -> ResourceRef | None:
        if tenant_id == self.tenant_id and execution_id == "root":
            return ResourceRef(ResourceKind.EXECUTION, execution_id, tenant_id)
        return None


@pytest.mark.asyncio
async def test_model_metadata_public_history_authorizes_every_port() -> None:
    reader, executions, store = _reader()
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
            live = await _wait_for_committed(
                lambda: runtime.history.capture_model_interaction_cutoffs(execution.execution_id, principal=principal),
                lambda boundary: boundary.cutoffs == (UsageReadCutoff(execution.execution_id, 1, 1),),
            )
            assert live.durable_history_available
            assert live.cutoffs == live.durable_cutoffs
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


@pytest.mark.asyncio
async def test_live_request_progress_precedes_scheduled_canonical_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.core import ExecutionEventType
    from linktools.ai.runtime.state._step_archive import StateStepArchive

    provider_entered = asyncio.Event()
    release_provider = asyncio.Event()
    publication_started = asyncio.Event()
    release_publication = asyncio.Event()
    prepare_interactions = StateStepArchive.prepare_interactions

    async def block_publication(self, run, interactions, payload, **kwargs):
        if self.runtime_domain is RuntimeDomain.EXECUTION and any(
            item.request_context is not None for item in interactions
        ):
            publication_started.set()
            await release_publication.wait()
        return await prepare_interactions(self, run, interactions, payload, **kwargs)

    monkeypatch.setattr(StateStepArchive, "prepare_interactions", block_publication)

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        del messages, info
        provider_entered.set()
        await release_provider.wait()
        return ModelResponse(parts=[TextPart("published answer")])

    group = CapabilityGroup("scheduled-history")
    group.agent("default", model="default", allow_tools=())
    storage_path = tmp_path / "scheduled"
    async with Runtime.open(
        "scheduled-history", models=_Models(_UsageFunctionModel(model)),
        storage=RuntimeStorage.filesystem(storage_path), capabilities=(group,),
    ) as runtime:
        execution = await runtime.agents.get("default").start("scheduled prompt")
        principal = runtime.default_principal
        stream = execution.watch()
        try:
            # A blocked history write must not keep the model from entering.
            await asyncio.wait_for(provider_entered.wait(), 5)
            previous_cursor = None
            while True:
                observed = await asyncio.wait_for(stream.__anext__(), 5)
                if observed.event.event_type == ExecutionEventType.MODEL_REQUEST_STARTED:
                    break
                previous_cursor = observed.cursor
            progress = observed.event
            assert progress.durable_seq is None
            positions = [
                {} if cursor is None else decode_execution_watch_cursor(
                    "scheduled-history", principal.tenant_id, execution.execution_id,
                    cursor, include_content=False,
                )
                for cursor in (previous_cursor, observed.cursor)
            ]
            assert positions[0] == positions[1]
            assert progress.payload["execution_id"] == execution.execution_id
            assert progress.payload["agent_run_seq"] == progress.payload["model_request_seq"] == 1
            assert "scheduled prompt" not in str(progress.payload)
            await asyncio.wait_for(publication_started.wait(), 5)
            async with RuntimeHistory.open(
                "scheduled-history", storage=RuntimeStorage.filesystem(storage_path),
            ) as independent:
                admitted = await independent.model_interactions(
                    execution.execution_id, principal=principal, include_content=True,
                )
                assert [(item.model_request_seq, item.status, item.request, item.response)
                        for item in admitted.items] in ([], [(1, "RUNNING", {}, None)])
                release_publication.set()
                committed = await _wait_for_committed(
                    lambda: independent.model_interactions(
                        execution.execution_id, principal=principal, include_content=True,
                    ), lambda page: bool(page.items),
                )
                assert not release_provider.is_set()
                assert [(item.execution_id, item.agent_run_seq, item.model_request_seq, item.status)
                        for item in committed.items] == [(execution.execution_id, 1, 1, "RUNNING")]
                assert committed.items[0].request == {}
                assert committed.items[0].response is None
        finally:
            release_publication.set()
            release_provider.set()
            await stream.aclose()
        assert (await execution.wait(timeout_seconds=5)).result.status.value == "SUCCEEDED"
        terminal = await execution.model_interactions(include_content=True)
        assert terminal.items[0].status == "SUCCEEDED"
        assert terminal.items[0].response is not None
        replayed = [event async for event in execution.watch(cursor=observed.cursor)
                    if event.event.event_type == ExecutionEventType.MODEL_REQUEST_STARTED]
        assert len(replayed) == 1
        assert replayed[0].event.durable_seq is not None
        assert replayed[0].event.payload == progress.payload
