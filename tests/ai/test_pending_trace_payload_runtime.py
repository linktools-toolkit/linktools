#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""New pending execution heads publish their admitted Trace at one source boundary."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, PersistenceDomain, agent_run_id
from linktools.ai.persistence.contracts import ExecutionHistoryState
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, SqlStorageContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_terminal_payload_runtime import _PayloadBuffer


@pytest.mark.asyncio
@pytest.mark.parametrize(("outcome", "sessionless", "execution_buffered", "trace_buffered"), (
    ("success", False, True, True), ("failure", False, True, True),
    ("cancelled", False, True, True), ("success", True, True, True),
    ("reject_before", False, True, True), ("reject_response", False, True, True),
    ("cancel_reject", False, True, True), ("source_unknown", False, True, True),
    ("success", False, True, False), ("success", False, False, True),
))
async def test_pending_execution_and_trace_payloads_share_the_terminal_source_boundary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str, sessionless: bool,
    execution_buffered: bool, trace_buffered: bool,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    execution_buffer = _PayloadBuffer()
    trace_bytes, accepted, acknowledged = [], [], []
    calls = 0
    prohibit_buffer_read = False

    class TraceBuffer(InMemoryCacheBackend):
        async def append(self, scope, id, key, content, *, wait):
            assert not wait and isinstance(content, bytes)
            if outcome == "reject_before" or outcome in {"reject_response", "cancel_reject"} and calls:
                return False
            result = await super().append(scope, id, key, content, wait=wait)
            if result:
                trace_bytes.append(content)
                accepted.append((scope, id))
            return result

        async def read_batch(self, scope, *, limit):
            assert not prohibit_buffer_read, "source readback must not depend on the transport"
            return await super().read_batch(scope, limit=limit)

        async def ack(self, scope, ids):
            acknowledged.extend((scope, id) for id in ids)
            await super().ack(scope, ids)

    trace_buffer = TraceBuffer(max_bytes=16 * 1024 * 1024)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        if outcome == "failure":
            raise ValueError("provider failed")
        if outcome in {"cancelled", "cancel_reject"}:
            raise asyncio.CancelledError
        return ModelResponse(parts=[TextPart("retained answer")])

    features = RuntimeFeatures("pending-trace-payload")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=execution_buffer if execution_buffered else None)), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.terminal(buffer=trace_buffer if trace_buffered else None)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("pending-trace-payload", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            target = runtime.agents.get() if sessionless else await runtime.agents.get().create_session("session")
            backend = runtime._execution_service.runtime_backend()
            if outcome == "source_unknown":
                from linktools.ai.errors import AIError, ErrorCode

                original_commit = backend._runtime_commands.commit_pending_turn
                original_read = execution_buffer.read_batch

                async def uncertain_commit(*args, **kwargs):
                    nonlocal prohibit_buffer_read
                    await original_commit(*args, **kwargs)
                    prohibit_buffer_read = True
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

                async def forbidden_read(*args, **kwargs):
                    assert not prohibit_buffer_read, "source readback must use its original sealed batch"
                    return await original_read(*args, **kwargs)

                monkeypatch.setattr(backend._runtime_commands, "commit_pending_turn", uncertain_commit)
                monkeypatch.setattr(execution_buffer, "read_batch", forbidden_read)
            transactions = []
            mutation = SqlStorageContext.run_mutation

            async def observe_mutation(context, callback, **kwargs):
                result = await mutation(context, callback, **kwargs)
                transactions.append(kwargs.get("domain"))
                return result

            monkeypatch.setattr(SqlStorageContext, "run_mutation", observe_mutation)
            execution = await target.start("buffered pending question", idempotency_key="request")
            run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                execution_id=execution.execution_id, agent_run_seq=1)
            trace = runtime._run_store.read_store(PersistenceDomain.TRACE)
            try:
                if outcome != "reject_before":
                    await asyncio.wait_for(entered.wait(), 5)
                    if trace_buffered:
                        assert trace_bytes and any(b"buffered pending question" in content for content in trace_bytes)
                    assert bool(execution_buffer.captures) is execution_buffered
                    assert await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
                    assert await trace.get_agent_run(agent_run_id=run_id) is None
                    assert transactions == []
            finally:
                release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            expected = {"success": ExecutionStatus.SUCCEEDED, "failure": ExecutionStatus.FAILED,
                        "cancelled": ExecutionStatus.CANCELLED, "reject_before": ExecutionStatus.FAILED,
                        "reject_response": ExecutionStatus.FAILED, "cancel_reject": ExecutionStatus.CANCELLED,
                        "source_unknown": ExecutionStatus.SUCCEEDED}[outcome]
            assert result.status is expected
            assert calls == (0 if outcome == "reject_before" else 1) and len(transactions) == 1
            source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert source.status is expected
            head = await persistence.execution.executions.get_history_head(execution.execution_id, tenant_id=runtime.tenant_id)
            assert head.state is ExecutionHistoryState.SEALED and head.seal_digest is not None
            assert (await trace.get_agent_run(agent_run_id=run_id) is None) == (outcome == "reject_before")
        assert set(acknowledged) == set(accepted)
        assert set(execution_buffer.acknowledged) == set(execution_buffer.accepted)


@pytest.mark.asyncio
async def test_unbuffered_terminal_trace_keeps_its_source_admission(
    tmp_path: Path,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("unbuffered-terminal-trace")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("unbuffered-terminal-trace", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            execution = await runtime.agents.get().start("question")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert source.status is ExecutionStatus.STARTED
            finally:
                release.set()
            assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
