#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Source-admitted executions still publish their produced bytes through the buffer."""
import asyncio
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus, PersistenceDomain, agent_run_id
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionHistoryState
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend, StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_execution_terminal_payload_runtime import _PayloadBuffer


@pytest.mark.asyncio
@pytest.mark.parametrize(("promotion", "split"), ((False, False), (True, False), (True, True)))
async def test_source_execution_terminal_uses_buffer(tmp_path: Path, promotion: bool, split: bool) -> None:
    entered, release = asyncio.Event(), asyncio.Event()
    buffer = _PayloadBuffer()
    calls = 0
    answer = "answer" * 13000

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart(answer)])

    features = RuntimeFeatures("source-payload-buffer")
    features.capability(ExecutionCapability(retain_history=promotion,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    if split:
        storage = PersistenceStorage.from_plan(replace(storage.plan,
            conversation=StorageRoute.sqlite(tmp_path / "conversation.db")),
            coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("source-payload-buffer", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            backend = runtime._execution_service.runtime_backend()
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question", idempotency_key="request")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                if promotion:
                    await backend.promote_pending_execution(execution.execution_id)
                source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert source is not None
                assert buffer.captures[-1].audit_events
                # Source publication must select backend bytes, not this RAM view.
                backend._pending_audit_events.pop(execution.execution_id, None)
            finally:
                release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"text": answer}
            assert calls == 1
            terminal = [capture for capture in buffer.captures if capture.terminal is not None][-1]
            assert terminal.terminal.result.output.kind == "inline"
            assert terminal.terminal.result.output.decode() == {"text": answer}
            events = await persistence.execution.events.list(execution.execution_id,
                tenant_id=runtime.tenant_id, after_event_seq=0, limit=100)
            source_audits = [(event.event_type, event.payload) for event in events.items]
            for audit in terminal.audit_events:
                assert (audit.event_type, audit.payload) in source_audits
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("accepted_prefix", (0, 1))
@pytest.mark.parametrize("split", (False, True))
async def test_source_execution_known_rejection_settles_failed_without_rejected_bytes(
    tmp_path: Path, accepted_prefix: int, split: bool,
) -> None:
    class RejectingBuffer(_PayloadBuffer):
        async def append(self, scope, id, key, content, *, wait):
            if len(self.accepted) >= accepted_prefix:
                return False
            return await super().append(scope, id, key, content, wait=wait)

    buffer = RejectingBuffer()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("must not publish rejected success")])

    features = RuntimeFeatures("source-payload-rejection")
    features.capability(ExecutionCapability(retain_history=split,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    if split:
        storage = PersistenceStorage.from_plan(replace(storage.plan,
            conversation=StorageRoute.sqlite(tmp_path / "conversation.db")))
    async with Persistence.open("source-payload-rejection", storage=storage) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            execution = await runtime.agents.get().start("question", idempotency_key="request")
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.FAILED
            assert result.output is None
            source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert source.status is ExecutionStatus.FAILED
            assert (await execution.wait(timeout_seconds=10)).result == result
            assert calls <= 1
        assert len(buffer.accepted) == accepted_prefix
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("reject", (False, True))
async def test_agent_task_child_source_result_handoff_uses_buffer(tmp_path: Path, reject: bool) -> None:
    from linktools.ai.core import TaskStatus
    from linktools.ai.inputs import AgentTaskInput
    from linktools.ai.task import Task, TaskGraph, TaskNode
    from linktools.ai.task.engine import TaskEngine

    class TaskBuffer(_PayloadBuffer):
        async def append(self, scope, id, key, content, *, wait):
            from linktools.ai.runtime._execution_buffer import _ExecutionPayloadCapture
            if reject and _ExecutionPayloadCapture.decode(content).terminal is not None:
                return False
            return await super().append(scope, id, key, content, wait=wait)

    buffer = TaskBuffer()
    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("task result")])

    features = RuntimeFeatures("source-task-buffer")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("source-task-buffer",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime, \
            TaskEngine.open(persistence=persistence) as engine:
            task = Task.from_agent("buffer.agent", runtime.agents.get())
            graph = await engine.with_definitions(task).start(
                TaskGraph("buffer", (TaskNode("agent", task=task, input=AgentTaskInput("work")),)),
                idempotency_key="start", principal=runtime.default_principal)
            try:
                await asyncio.wait_for(entered.wait(), 5)
                node, = (await graph.state()).node_states
                assert node.execution_id is not None
                source = await persistence.execution.executions.get(node.execution_id, tenant_id=runtime.tenant_id)
                assert source is not None and source.status is ExecutionStatus.STARTED
            finally:
                release.set()
            result = (await graph.wait(timeout_seconds=10)).result
            assert result.status is (TaskStatus.FAILED if reject else TaskStatus.SUCCEEDED)
            node, = (await graph.state()).node_states
            assert node.status is result.status
            source = await persistence.execution.executions.get(node.execution_id, tenant_id=runtime.tenant_id)
            assert source.status is (ExecutionStatus.FAILED if reject else ExecutionStatus.SUCCEEDED)
            if reject:
                assert source.result.output is None
            else:
                assert source.result.output.decode() == {"text": "task result"}
                terminal = [capture for capture in buffer.captures if capture.terminal is not None][-1]
                assert terminal.execution.execution_id == node.execution_id
                assert terminal.terminal.result.output.decode() == {"text": "task result"}
            assert calls == 1
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("failure", "cancelled"))
async def test_source_control_outcome_survives_its_own_terminal_rejection(tmp_path: Path, outcome: str) -> None:
    from linktools.ai.runtime._execution_buffer import _ExecutionPayloadCapture

    class RejectTerminal(_PayloadBuffer):
        async def append(self, scope, id, key, content, *, wait):
            if _ExecutionPayloadCapture.decode(content).terminal is not None:
                return False
            return await super().append(scope, id, key, content, wait=wait)

    buffer = RejectTerminal()
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if outcome == "failure":
            raise ValueError("provider failed")
        raise asyncio.CancelledError

    features = RuntimeFeatures("source-control-rejection")
    features.capability(ExecutionCapability(retain_history=False,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("source-control-rejection",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            execution = await runtime.agents.get().start("question", idempotency_key="request")
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is (ExecutionStatus.FAILED if outcome == "failure" else ExecutionStatus.CANCELLED)
            assert result.output is None
            source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert source.status is result.status
            assert calls == 1
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("interference", ("source_unknown", "remote_cancel"))
async def test_source_terminal_confirmation_and_remote_cancel_preserve_buffer_truth(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, interference: str,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.execution import CancelExecutionRequest
    from linktools.ai.storage import StoredPayload
    from .test_execution_payload_materialization import _Objects

    buffer = _PayloadBuffer()
    entered, release = asyncio.Event(), asyncio.Event()
    answer = "gated-result" * 10000
    digest = StoredPayload.inline_json({"text": answer}).digest
    calls = source_calls = 0

    class Objects(_Objects):
        async def put(self, key, chunks, *, expected_size, expected_digest):
            if expected_digest == digest:
                assert any(c.terminal is not None and c.terminal.result.output.digest == digest
                    for c in buffer.captures)
                entered.set()
                await release.wait()
            return await super().put(key, chunks, expected_size=expected_size, expected_digest=expected_digest)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart(answer)])

    features = RuntimeFeatures("source-terminal-race")
    features.capability(ExecutionCapability(retain_history=False,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("source-terminal-race",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db", object_store=Objects())) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime, \
            Runtime.attach(persistence=persistence,
                features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as remote:
            backend = runtime._execution_service.runtime_backend()
            command = backend._runtime_commands.commit_terminal_checkpoint

            async def source(*args, **kwargs):
                nonlocal source_calls
                source_calls += 1
                value = await command(*args, **kwargs)
                if interference == "source_unknown":
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return value

            monkeypatch.setattr(backend._runtime_commands, "commit_terminal_checkpoint", source)
            execution = await runtime.agents.get().start("question", idempotency_key="request")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                if interference == "remote_cancel":
                    await remote.executions.cancel(execution.execution_id,
                        CancelExecutionRequest(remote.default_principal, "cancel"))
            finally:
                release.set()
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is (ExecutionStatus.CANCELLED if interference == "remote_cancel" else ExecutionStatus.SUCCEEDED)
            assert calls == 1
            if interference == "source_unknown":
                assert source_calls == 1
                assert result.output == {"text": answer}
            else:
                assert result.output is None
            assert (await execution.wait(timeout_seconds=10)).result == result
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("trace", (False, True))
async def test_pending_candidate_can_promote_for_cancel_during_context_prepare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trace: bool,
) -> None:
    from linktools.ai.execution import CancelExecutionRequest

    buffer = _PayloadBuffer()
    entered, release = asyncio.Event(), asyncio.Event()
    promoted = asyncio.Event()
    calls = preparations = 0
    accepted, acknowledged = [], []

    class TraceBuffer(InMemoryCacheBackend):
        async def append(self, scope, id, key, content, *, wait):
            assert not wait and isinstance(content, bytes)
            result = await super().append(scope, id, key, content, wait=wait)
            if result:
                accepted.append((scope, id))
            return result

        async def ack(self, scope, ids):
            acknowledged.extend((scope, id) for id in ids)
            await super().ack(scope, ids)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("candidate")])

    features = RuntimeFeatures("promotion-cancel-buffer")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    if trace:
        features.capability(TraceCapability(commit=CommitPolicy.terminal(
            buffer=TraceBuffer(max_bytes=16 * 1024 * 1024))), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("promotion-cancel-buffer",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db",
            coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime, \
            Runtime.attach(persistence=persistence,
                features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as remote:
            backend = runtime._execution_service.runtime_backend()
            prepare = backend._context_archive.prepare
            coordinator = remote._execution_service.runtime_backend()._pending_turns.coordination
            cancel = coordinator.cancel

            async def observe_cancel(*args, **kwargs):
                value = await cancel(*args, **kwargs)
                promoted.set()
                return value

            monkeypatch.setattr(coordinator, "cancel", observe_cancel)

            async def gated_prepare(*args, **kwargs):
                nonlocal preparations
                preparations += 1
                if preparations == 1:
                    entered.set()
                    await release.wait()
                return await prepare(*args, **kwargs)

            monkeypatch.setattr(backend._context_archive, "prepare", gated_prepare)
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question", idempotency_key="request")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                assert buffer.captures[-1].terminal is not None
                assert await persistence.execution.executions.get(
                    execution.execution_id, tenant_id=runtime.tenant_id) is None
                if trace:
                    assert accepted and not acknowledged
                cancelling = asyncio.create_task(remote.executions.cancel(execution.execution_id,
                    CancelExecutionRequest(remote.default_principal, "cancel")))
                await asyncio.wait_for(promoted.wait(), 5)
            finally:
                release.set()
            await asyncio.wait_for(cancelling, 10)
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.CANCELLED
            assert result.output is None
            assert calls == 1
            if trace:
                head = await persistence.execution.executions.get_history_head(
                    execution.execution_id, tenant_id=runtime.tenant_id)
                assert head.state is ExecutionHistoryState.SEALED
                archive = runtime._run_store.read_store(PersistenceDomain.TRACE)
                run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                    execution_id=execution.execution_id, agent_run_seq=1)
                interactions = await archive.list_model_interactions(agent_run_id=run_id)
                assert "candidate" in repr(await archive.resolve_model_interactions(interactions))
        assert set(buffer.acknowledged) == set(buffer.accepted)
        assert set(acknowledged) == set(accepted)


@pytest.mark.asyncio
async def test_execution_buffer_does_not_recapture_rejected_trace_checkpoint(tmp_path: Path) -> None:
    from linktools.ai.capability import CheckpointCapability, TraceCapability

    calls = 0

    class TraceBuffer(InMemoryCacheBackend):
        async def append(self, scope, id, key, content, *, wait):
            if calls:
                return False
            return await super().append(scope, id, key, content, wait=wait)

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("rejected trace response")])

    buffer = _PayloadBuffer()
    features = RuntimeFeatures("source-trace-rejection")
    features.capability(ExecutionCapability(retain_history=False,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.capability(TraceCapability(commit=CommitPolicy.batch(
        buffer=TraceBuffer(max_bytes=4 * 1024 * 1024), batch_size=128, flush_interval=60)), scope="runtime")
    features.capability(CheckpointCapability(), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("source-trace-rejection",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            execution = await runtime.agents.get().start("question", idempotency_key="request")
            result = (await execution.wait(timeout_seconds=10)).result
            assert result.status is ExecutionStatus.FAILED
            assert result.output is None
            assert calls == 1
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
async def test_source_audit_watermark_requires_verified_prefix_and_keeps_new_tail() -> None:
    from dataclasses import replace
    from linktools.ai.core import ExecutionEventType
    from linktools.ai.errors import AIError
    from linktools.ai.persistence.contracts import ExecutionEventAppend
    from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer
    from .test_execution_payload_buffer import _capture

    backend = _PayloadBuffer()
    owner = _ExecutionPayloadBuffer(backend)
    first = replace(_capture(), audit_events=(ExecutionEventAppend(ExecutionEventType.MODEL_REQUEST_STARTED,
        {"frozen": "first"}),))
    execution_id = first.execution.execution_id
    await owner.append(first)
    with pytest.raises(AIError):
        owner.confirm_audits(execution_id, start=0, count=1)
    selected, ids = await owner.read(execution_id)
    owner.confirm_audits(execution_id, start=0, count=1)
    second = replace(selected, audit_events=(*selected.audit_events,
        ExecutionEventAppend(ExecutionEventType.MODEL_REQUEST_STARTED, {"frozen": "second"})))
    await owner.append(second)
    selected, ids = await owner.read(execution_id)
    assert selected.audit_events[owner.audit_position(execution_id):] == second.audit_events[1:]
    with pytest.raises(AIError):
        owner.confirm_audits(execution_id, start=0, count=1)
    with pytest.raises(AIError):
        owner.confirm_audits(execution_id, start=1, count=2)
    owner.seal(execution_id, ids)
    owner.reject_source(execution_id, ids)
    await owner.append(second)
    _, final_ids = await owner.read(execution_id, seal=True)
    owner.confirm_audits(execution_id, start=1, count=1)
    owner.confirm(execution_id, final_ids)
    await owner.release(execution_id)
    assert owner.settled(execution_id)
    await owner.close()
    assert set(backend.acknowledged) == set(backend.accepted)


@pytest.mark.asyncio
@pytest.mark.parametrize("reopen", (False, True))
async def test_deferred_source_boundary_releases_buffer_and_resumes_new_producer(
    tmp_path: Path, reopen: bool,
) -> None:
    from contextlib import AsyncExitStack
    from pydantic_ai.messages import ToolCallPart
    from linktools.ai.core import ApprovalDecision, Principal
    from linktools.ai.execution import ApprovalDecisionRequest
    from linktools.ai.workspace import ToolPermissionPolicy, Workspace, WorkspacePolicy
    from ._runtime_test_helpers import _wait_for_committed, workspace_features

    buffer = _PayloadBuffer()
    (tmp_path / "data.txt").write_text("approved body")
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "data.txt"}, "read-1")])
        assert calls == 2
        assert "approved body" in str(messages[-1])
        return ModelResponse(parts=[TextPart("resumed")])

    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default="ask")))
    features = workspace_features("source-deferred-buffer", workspace=workspace)
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.agent("reader", model="default", allow_tools=("read_file",), tool_retries=0, output_retries=0)
    features = model_features(_Models(_UsageFunctionModel(model)), features)
    async with Persistence.open("source-deferred-buffer",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with AsyncExitStack() as stack:
            context = Runtime.attach(persistence=persistence, features=features, metrics=None)
            runtime = await stack.enter_async_context(context)
            session = await runtime.agents.get("reader").create_session("session")
            execution = await session.start("read", idempotency_key="request")
            tenant = runtime.tenant_id
            await _wait_for_committed(
                lambda: persistence.execution.executions.get(execution.execution_id, tenant_id=tenant),
                lambda record: record is not None and record.status is ExecutionStatus.WAITING_DEFERRED)
            approvals = await persistence.recovery.approvals.list_pending(execution.execution_id, tenant_id=tenant)
            assert len(approvals) == 1 and calls == 1
            if reopen:
                await stack.aclose()
                assert set(buffer.acknowledged) == set(buffer.accepted)
                runtime = await stack.enter_async_context(Runtime.attach(persistence=persistence, features=features, metrics=None))
            await runtime.approvals.decide(execution.execution_id, ApprovalDecisionRequest(
                Principal("approver", tenant, "service"), approvals[0].approval_id, "approve", ApprovalDecision.APPROVE))
            result = (await runtime.executions.wait(execution.execution_id,
                principal=runtime.default_principal, timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {"text": "resumed"}
            assert calls == 2
        assert set(buffer.acknowledged) == set(buffer.accepted)


@pytest.mark.asyncio
async def test_zero_prefix_rejection_still_requires_source_control_confirmation_before_close() -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer
    from .test_execution_payload_buffer import _capture

    owner = _ExecutionPayloadBuffer(InMemoryCacheBackend(max_bytes=0))
    capture = _capture()
    execution_id = capture.execution.execution_id
    with pytest.raises(AIError) as refusal:
        await owner.append(capture)
    assert await owner.settle_rejection(execution_id, refusal.value)
    assert await owner.source_prefix(execution_id) == (None, ())
    owner.seal(execution_id, ())
    with pytest.raises(AIError) as unresolved:
        await owner.close()
    assert unresolved.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    owner.confirm(execution_id, ())
    await owner.release(execution_id)
    await owner.close()
    assert owner._scopes[execution_id].failure is None
    assert owner._scopes[execution_id].rejection_prefix is None


@pytest.mark.asyncio
async def test_execution_buffer_retirement_uses_source_generation_and_keeps_old_cleanup_owned() -> None:
    from linktools.ai.errors import AIError, ErrorCode
    from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer
    from .test_execution_payload_buffer import _Buffer, _capture

    backend = _Buffer(max_bytes=1024 * 1024)
    backend.fail_ack = True
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    await owner.append(capture)
    owner.bind_producer(execution_id, 7)
    _, ids = await owner.read(execution_id)
    owner.confirm(execution_id, ids)
    owner.retire(execution_id, 7)
    for invalid in (6, 7):
        with pytest.raises(AIError):
            owner.bind_producer(execution_id, invalid)
    owner.bind_producer(execution_id, 11)
    with pytest.raises(AIError):
        owner.require_producer(execution_id, 7)
    owner.require_producer(execution_id, 11)
    await owner.append(capture)
    _, ids = await owner.read(execution_id, seal=True)
    owner.confirm(execution_id, ids)
    await owner.release(execution_id)
    with pytest.raises(AIError):
        owner.bind_producer(execution_id, 12)
    with pytest.raises(AIError) as cleanup:
        await owner.close()
    assert cleanup.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    backend.fail_ack = False
    await owner.close()
    assert owner._retired_scopes == []


@pytest.mark.asyncio
async def test_zero_prefix_source_failure_retains_control_until_same_execution_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.core import agent_run_id
    from linktools.ai.errors import AIError, ErrorCode

    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart("not called")])

    features = RuntimeFeatures("zero-prefix-source-failure")
    features.capability(ExecutionCapability(retain_history=False,
        commit=CommitPolicy.terminal(buffer=InMemoryCacheBackend(max_bytes=0))), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    async with Persistence.open("zero-prefix-source-failure",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=None) as runtime:
            backend = runtime._execution_service.runtime_backend()
            command = backend._runtime_commands.commit_terminal_checkpoint

            async def unavailable(*args, **kwargs):
                raise AIError(ErrorCode.STORAGE_UNAVAILABLE)

            monkeypatch.setattr(backend._runtime_commands, "commit_terminal_checkpoint", unavailable)
            execution = await runtime.agents.get().start("question", idempotency_key="request")
            with pytest.raises(AIError):
                await execution.wait(timeout_seconds=10)
            current = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert current.status is ExecutionStatus.STARTED
            with pytest.raises(AIError) as unresolved:
                await backend._execution_buffer.close()
            assert unresolved.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
            monkeypatch.setattr(backend._runtime_commands, "commit_terminal_checkpoint", command)
            run_id = agent_run_id(namespace=runtime.namespace, tenant_id=runtime.tenant_id,
                execution_id=execution.execution_id, agent_run_seq=current.agent_run_seq)
            generation = runtime._run_store.execution_producer_generation(run_id, execution_id=execution.execution_id)
            confirmed = await backend._commit_failure(current, AIError(ErrorCode.STORAGE_UNAVAILABLE),
                agent_run_id=run_id, producer_generation=generation)
            assert confirmed.status is ExecutionStatus.FAILED
            assert confirmed.execution_id == execution.execution_id
            await backend._execution_buffer.close()
            await backend.release_runtime_execution(execution.execution_id, tenant_id=runtime.tenant_id)
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.FAILED
            assert calls == 0


@pytest.mark.asyncio
async def test_unknown_tool_effect_retires_only_source_confirmed_execution_audits(tmp_path: Path) -> None:
    from linktools.ai.capability import AgentContext, CheckpointCapability
    from linktools.ai.errors import AIError, ErrorCode
    from .test_task_mixed_node_reliability import _TaskTestModels

    buffer = _PayloadBuffer()
    effects = []

    async def write(context: AgentContext[None]) -> str:
        effects.append("applied")
        raise RuntimeError("provider outcome unknown")

    features = RuntimeFeatures("source-effect-unknown")
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal(buffer=buffer)), scope="runtime")
    features.capability(CheckpointCapability(), scope="runtime")
    features.tool(write, effect_policy="non_replay_safe")
    features.agent("default", allow_tools=("write",))
    async with Persistence.open("source-effect-unknown",
        storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as persistence:
        async with Runtime.attach(persistence=persistence,
            features=model_features(_TaskTestModels(), features), metrics=None) as runtime:
            execution = await runtime.agents.get().start("write", idempotency_key="request")
            with pytest.raises(AIError) as unknown:
                await execution.wait(timeout_seconds=10)
            assert unknown.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            source = await persistence.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert source.status is ExecutionStatus.RECOVERY_REQUIRED
            captured_audits = buffer.captures[-1].audit_events
            assert captured_audits
            page = await persistence.execution.events.list(execution.execution_id,
                tenant_id=runtime.tenant_id, after_event_seq=0, limit=100)
            actual = [(event.event_type, event.payload) for event in page.items]
            expected = [(event.event_type, event.payload) for event in captured_audits]
            assert actual[-len(expected) - 1:-1] == expected
            assert all(actual.count(event) == 1 for event in expected)
            assert effects == ["applied"]
            with pytest.raises(AIError) as blocked:
                await runtime.executions.recover(execution.execution_id,
                    principal=runtime.default_principal, idempotency_key="unsafe-retry")
            assert blocked.value.code is ErrorCode.TOOL_EFFECT_UNKNOWN
            assert effects == ["applied"]
        assert set(buffer.acknowledged) == set(buffer.accepted)
