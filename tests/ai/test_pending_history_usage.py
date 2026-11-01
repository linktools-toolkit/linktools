#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independent history distinguishes invisible usage from measured zero."""

import asyncio
from contextlib import asynccontextmanager
from copy import copy
from dataclasses import replace
from typing import TYPE_CHECKING

import pytest
from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import (
    CommitPolicy,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    TenantAuthorizationPolicy,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import ModelInteractionReadBoundary, UsageReadCutoff
from linktools.ai.history import History
from linktools.ai.history._projection import StepExecutionHistoryReader
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.usage import RequestUsage

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_model_interaction_lifecycle_paging import (
    _Executions,
    _HistoryStore,
    _interaction,
    _record,
)
from .test_state_turn_composition import _BorrowedCache, _HostCoordination

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from linktools.ai.core import AuthorizationAction, ResourceRef


@asynccontextmanager
async def _opened(
    path: "Path", cache: _BorrowedCache, *, namespace: str = "pending-usage", read_only: bool = False,
) -> "AsyncIterator[Persistence]":
    coordination = _HostCoordination(cache)
    storage = PersistenceStorage.sqlite(
        path, cache=InMemoryCacheBackend(max_bytes=2**20), coordination=coordination,
    )
    async with Persistence.open(namespace, storage=storage, tenant_id="tenant", read_only=read_only) as state:
        coordination.bind(state)
        yield state


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_trace", (False, True))
async def test_independent_default_usage_is_unknown_until_source_publication(
    tmp_path: "Path", retain_trace: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")], usage=RequestUsage(input_tokens=11, output_tokens=7))

    cache = _BorrowedCache()
    features = RuntimeFeatures("pending-usage")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal(buffer=cache)), scope="runtime")
    if retain_trace:
        features.capability(TraceCapability(commit=CommitPolicy.terminal(buffer=cache)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    path = tmp_path / "state.db"
    async with _opened(path, cache) as writer, Runtime.attach(
        persistence=writer, features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None,
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="request")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            async with _opened(path, cache, read_only=True) as reader, History.open(persistence=reader) as history:
                repository = reader.execution.executions
                assert await repository.get(execution.execution_id, tenant_id="tenant") is None
                source_before = await repository.state_store.read(lambda tx: tx.scan_records())
                observed = await history.inspect_execution(execution.execution_id, principal=runtime.default_principal)
                assert observed.status is ExecutionStatus.STARTED
                assert observed.usage is not None
                assert observed.usage.unrecorded_executions == 1
                assert observed.usage.logical_requests == observed.usage.unknown_usage_requests == 0
                assert observed.usage.input_tokens == observed.usage.output_tokens == 0
                assert (await history.inspect_execution(execution.execution_id, principal=runtime.default_principal,
                                                        include_usage=False)).usage is None
                local = await runtime.history.usage(execution.execution_id, principal=runtime.default_principal)
                if retain_trace:
                    assert local.logical_requests == 0 and local.unrecorded_executions == 1
                else:
                    assert local.logical_requests == local.running_requests == 1
                    assert local.unrecorded_executions == 0
                for principal in (Principal("other", "tenant"), Principal("other", "other")):
                    with pytest.raises(AIError) as denied:
                        await history.inspect_execution(execution.execution_id, principal=principal)
                    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                with pytest.raises(AIError) as absent:
                    await history.inspect_execution("missing", principal=runtime.default_principal)
                assert absent.value.code is ErrorCode.AUTHORIZATION_DENIED
                assert await repository.get(execution.execution_id, tenant_id="tenant") is None
                assert await repository.state_store.read(lambda tx: tx.scan_records()) == source_before
                async with _opened(path, cache, namespace="unrelated", read_only=True) as foreign, History.open(
                    persistence=foreign,
                ) as other_history:
                    with pytest.raises(AIError) as denied:
                        await other_history.inspect_execution(execution.execution_id, principal=runtime.default_principal)
                    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                async with _opened(tmp_path / "unrelated.db", cache) as foreign, History.open(
                    persistence=foreign,
                ) as other_history:
                    with pytest.raises(AIError) as denied:
                        await other_history.inspect_execution(execution.execution_id, principal=runtime.default_principal)
                    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
                release.set()
                assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                published = await history.inspect_execution(execution.execution_id, principal=runtime.default_principal)
                assert published.status is ExecutionStatus.SUCCEEDED
                assert published.usage is not None
                if retain_trace:
                    assert published.usage.unrecorded_executions == 0
                    assert published.usage.logical_requests == published.usage.succeeded_requests == 1
                    assert published.usage.input_tokens == 11 and published.usage.output_tokens == 7
                    assert published.usage == await runtime.history.usage(execution.execution_id, principal=runtime.default_principal)
                else:
                    assert published.usage.unrecorded_executions == 1
                    assert published.usage.logical_requests == 0
                assert published.usage == await history.usage(execution.execution_id, principal=runtime.default_principal)
                async with _opened(path, cache, read_only=True) as fresh, History.open(persistence=fresh) as fresh_history:
                    assert published.usage == await fresh_history.usage(execution.execution_id, principal=runtime.default_principal)
                assert calls == 1
        finally:
            release.set()


@pytest.mark.asyncio
async def test_partial_usage_keeps_measured_requests_and_marks_missing_run() -> None:
    from linktools.ai.core import agent_conversation_id, agent_run_id
    from linktools.ai.persistence.runs import AgentRunRecord

    record = _record("root")
    record.binding_kind = "agent"
    record.agent_run_seq = 2
    store = _HistoryStore()
    run_id = agent_run_id(namespace="history", tenant_id="tenant", execution_id="root", agent_run_seq=1)
    store.runs[run_id] = AgentRunRecord(run_id, agent_conversation_id=agent_conversation_id(
        namespace="history", tenant_id="tenant", execution_id="root"), metadata={"agent_run_seq": "1"})
    store.interactions[run_id] = [replace(_interaction(run_id, 1), usage=None)]
    reader = StepExecutionHistoryReader(namespace="history", executions=_Executions(record), store=store,
        cursor_signer=HmacCursorSigner("history", b"history"))
    summary = await reader.usage("root", tenant_id="tenant")
    assert summary.logical_requests == summary.unknown_usage_requests == 1
    assert summary.unrecorded_executions == 1
    for cutoff in (UsageReadCutoff("other", 1, 1), UsageReadCutoff("root", 2, 1), UsageReadCutoff("root", 1, 2)):
        with pytest.raises(AIError) as invalid:
            await reader.usage("root", tenant_id="tenant", cutoffs=(cutoff,))
        assert invalid.value.code is ErrorCode.CURSOR_INVALID


@pytest.mark.asyncio
async def test_known_empty_usage_is_distinct_from_unavailable_history() -> None:
    record = _record("root")
    record.binding_kind = "agent"
    record.agent_run_seq = 0
    record.status = ExecutionStatus.STARTED
    reader = StepExecutionHistoryReader(namespace="history", executions=_Executions(record), store=_HistoryStore(),
        cursor_signer=HmacCursorSigner("history", b"history"))
    summary = await reader.usage("root", tenant_id="tenant")
    assert summary.logical_requests == summary.unrecorded_executions == summary.unknown_usage_requests == 0


@pytest.mark.asyncio
async def test_pending_authorization_cannot_select_another_source_owner() -> None:
    record = _record("root")
    record.principal_id = "source-owner"
    pending = copy(record)
    pending.principal_id = "pending-owner"

    async def read_pending(execution_id: str, *, tenant_id: str):
        assert execution_id == "root" and tenant_id == "tenant"
        return pending

    reader = StepExecutionHistoryReader(namespace="history", executions=_Executions(record), store=_HistoryStore(),
        cursor_signer=HmacCursorSigner("history", b"history"))
    reader.bind_pending_execution_reader(read_pending)
    with pytest.raises(AIError) as denied:
        await reader.usage("root", tenant_id="tenant")
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED


@asynccontextmanager
async def _paused_history(path: "Path", *, retain_trace: bool, terminal: bool = False):
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")], usage=RequestUsage(input_tokens=11, output_tokens=7))

    cache = _BorrowedCache()
    features = RuntimeFeatures("pending-usage")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal(buffer=cache)), scope="runtime")
    if retain_trace:
        features.capability(TraceCapability(commit=CommitPolicy.terminal(buffer=cache)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with _opened(path, cache) as writer, Runtime.attach(
        persistence=writer, features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None,
    ) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="request")
        try:
            await asyncio.wait_for(entered.wait(), 5)
            if terminal:
                release.set()
                assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            class Authorization(TenantAuthorizationPolicy):
                async def authorize(
                    self, principal: Principal, action: "AuthorizationAction", resource: "ResourceRef",
                ) -> None:
                    await super().authorize(principal, action, resource)
                    if principal.principal_id != runtime.default_principal.principal_id:
                        raise AIError(ErrorCode.AUTHORIZATION_DENIED)

            async with _opened(path, cache, read_only=True) as reader, History.open(
                persistence=reader, authorization=Authorization("tenant"),
            ) as history:
                yield runtime, execution, history, reader, release
        finally:
            release.set()
            await execution.wait(timeout_seconds=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_trace", (False, True))
@pytest.mark.parametrize("terminal", (False, True))
@pytest.mark.parametrize("operation", ("capture", "updates", "metadata"))
async def test_source_pending_metadata_reads_keep_authority_outside_snapshot(
    tmp_path: "Path", retain_trace: bool, terminal: bool, operation: str,
) -> None:
    async with _paused_history(tmp_path / "state.db", retain_trace=retain_trace, terminal=terminal) as (
        runtime, execution, history, reader, _release,
    ):
        identity, principal = execution.execution_id, runtime.default_principal
        before = await reader.execution.executions.state_store.read(lambda tx: tx.scan_records())
        through = int(terminal and retain_trace)
        if operation == "capture":
            boundary = await history.capture_model_interaction_cutoffs(identity, principal=principal)
        elif operation == "updates":
            boundary, items = await history.read_model_interaction_updates(identity, principal=principal)
            assert len(items) == through
        else:
            items = await history.read_model_interaction_metadata(
                identity, principal=principal, agent_run_seq=1,
                after_model_request_seq=0, through_model_request_seq=through,
            )
            assert len(items) == through
        if operation != "metadata":
            assert boundary.cutoffs == ((UsageReadCutoff(identity, 1, 1),) if through else ())
            assert boundary.durable_history_available is bool(through)
            assert not boundary.local_staging_available
        assert await reader.execution.executions.state_store.read(lambda tx: tx.scan_records()) == before
        for denied in (Principal("other", "tenant"), Principal("other", "other")):
            with pytest.raises(AIError) as error:
                await history.capture_model_interaction_cutoffs(identity, principal=denied)
            assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
@pytest.mark.parametrize("retain_trace", (False, True))
async def test_usage_observes_terminal_publication_without_inspection_refresh(
    tmp_path: "Path", retain_trace: bool,
) -> None:
    async with _paused_history(tmp_path / "state.db", retain_trace=retain_trace) as (
        runtime, execution, history, reader, release,
    ):
        identity, principal = execution.execution_id, runtime.default_principal
        observed = await history.inspect_execution(identity, principal=principal)
        assert observed.usage is not None and observed.usage.unrecorded_executions == 1
        release.set()
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        summary = await history.usage(identity, principal=principal)
        assert summary.unrecorded_executions == int(not retain_trace)
        assert summary.logical_requests == summary.succeeded_requests == int(retain_trace)
        assert summary.input_tokens == (11 if retain_trace else 0)
        assert summary.output_tokens == (7 if retain_trace else 0)
        assert summary == await history.usage(identity, principal=principal)
        for denied in (Principal("other", "tenant"), Principal("other", "other")):
            with pytest.raises(AIError) as error:
                await history.usage(identity, principal=denied)
            assert error.value.code is ErrorCode.AUTHORIZATION_DENIED


@pytest.mark.asyncio
async def test_inherited_usage_preserves_custom_public_capture() -> None:
    record = _record("root")
    record.binding_kind = "agent"
    record.agent_run_seq = 1
    boundary = ModelInteractionReadBoundary((UsageReadCutoff("root", 1, 0),), (), False, True)

    class Reader(StepExecutionHistoryReader):
        async def capture_model_interaction_cutoffs(
            self, execution_id: str, *, tenant_id: str,
        ) -> ModelInteractionReadBoundary:
            assert execution_id == "root" and tenant_id == "tenant"
            assert active_state_scope() is None
            return boundary

    reader = Reader(namespace="history", executions=_Executions(record), store=_HistoryStore(),
        cursor_signer=HmacCursorSigner("history", b"history"))
    summary = await reader.usage("root", tenant_id="tenant")
    assert summary.cutoffs == boundary.cutoffs
    assert summary.logical_requests == summary.unrecorded_executions == 0
