#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Explicit optional features and core durability regression tests."""

from ._runtime_test_helpers import workspace_features
import asyncio
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, TextPart, UserPromptPart
from pydantic_ai.models import CompletedStreamedResponse, Model, ModelRequestParameters, StreamedResponse
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.settings import ModelSettings

from linktools.ai.capability import AgentContext, RuntimeFeatures
from linktools.ai.core import ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.capability import ConversationCapability, CheckpointCapability, TraceCapability
from linktools.ai.runtime import Runtime, RuntimeStorage, RuntimeStoragePlan, RuntimeStorageRoute
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.storage._database import SqlStorageContext


class _FeatureFunctionModel(FunctionModel):
    @asynccontextmanager
    async def request_stream(self, messages: list[ModelMessage], model_settings: ModelSettings | None,
                             model_request_parameters: ModelRequestParameters,
                             run_context: object | None = None) -> AsyncIterator[StreamedResponse]:
        response = await self.request(messages, model_settings, model_request_parameters)
        yield CompletedStreamedResponse(response, model_request_parameters=model_request_parameters, replay_events=True)


class _FeatureModels:
    route_id = "default"
    provider = "test"
    model_identity = "test:explicit-features"
    contract: dict[str, JsonValue] = {"provider": "test", "model": "explicit-features"}

    def __init__(self, model: Model | None = None) -> None:
        self.model = model or TestModel(custom_output_text="done")

    def capture(self):
        return self

    def resolve(self, route_id: str):
        assert route_id == self.route_id
        return self

    def restore(self, payload: Mapping[str, JsonValue], *, route_id: str | None = None):
        assert route_id in (None, self.route_id) and dict(payload) == self.contract
        return self

    def materialize(self) -> Model:
        return self.model


def _definitions() -> RuntimeFeatures:
    features = RuntimeFeatures("agents")
    features.agent("default", model="default", allow_tools=())
    return features


def _features(*capabilities) -> RuntimeFeatures:
    features = _definitions()
    for capability in capabilities:
        features.capability(capability, scope="runtime")
    return features


@pytest.mark.asyncio
@pytest.mark.parametrize("split_storage", (False, True))
async def test_minimal_runtime_rejects_conversation_but_runs_agent(tmp_path: Path, split_storage: bool):
    storage = (RuntimeStorage(RuntimeStoragePlan(
        conversation=RuntimeStorageRoute.sqlite(tmp_path / "conversation.db"),
        execution=RuntimeStorageRoute.sqlite(tmp_path / "execution.db"),
    )) if split_storage else RuntimeStorage.in_memory())
    async with Runtime.open("minimal", storage=storage, models=_FeatureModels(), features=_definitions()) as runtime:
        with pytest.raises(AIError) as missing:
            await runtime.agents.get().create_session("session")
        assert missing.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        execution = await runtime.agents.get().start("question")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None
        assert not record.conversation and not record.restart_checkpoints and not record.durable_trace
        with pytest.raises(AIError) as missing:
            await execution.fork("fork")
        assert missing.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING


@pytest.mark.asyncio
@pytest.mark.parametrize("cache_kind", ("none", "unsupported", "full"))
@pytest.mark.parametrize("evidence", ("none", "checkpoint", "trace"))
async def test_conversation_pending_requires_complete_cache(tmp_path, monkeypatch, cache_kind, evidence):
    entered, release = asyncio.Event(), asyncio.Event()
    async def model(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])
    features = _features(ConversationCapability())
    if evidence != "none":
        features.capability(CheckpointCapability() if evidence == "checkpoint" else TraceCapability(), scope="runtime")
    cache = InMemoryCacheBackend(max_bytes=2**20) if cache_kind == "full" else object() if cache_kind == "unsupported" else None
    storage = RuntimeStorage.sqlite(tmp_path / "state.db", cache=cache)
    async with Runtime.open("pending", storage=storage, models=_FeatureModels(_FeatureFunctionModel(model)), features=features) as runtime:
        session = await runtime.agents.get().create_session("session")
        writes = []
        original = SqlStorageContext.run_mutation
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            writes.append(kwargs.get("domain"))
            return result
        monkeypatch.setattr(SqlStorageContext, "run_mutation", count)
        execution = await session.start("question", idempotency_key="turn")
        await asyncio.wait_for(entered.wait(), 5)
        pending = cache_kind == "full" and evidence == "none"
        try:
            record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert (record is None) == pending
            if pending:
                assert writes == []
        finally:
            release.set()
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert len(writes) == 1 if pending else len(writes) > 1
        total = len(writes)
        replay = await session.start("question", idempotency_key="turn")
        assert replay.execution_id == execution.execution_id and len(writes) == total


@pytest.mark.asyncio
async def test_no_optional_features_preserves_required_deferred_frontier(tmp_path):
    from linktools.ai.core import ApprovalStatus
    from linktools.ai.runtime.state._contracts import RecoveryCheckpointState
    from linktools.ai.workspace import Workspace, WorkspacePolicy, ToolPermissionPolicy
    workspace = Workspace.load(tmp_path, policy=WorkspacePolicy(tool_permissions=ToolPermissionPolicy(default="ask")))
    features = workspace_features("deferred", workspace=workspace)
    features.agent("default", model="default", allow_tools=("read_file",))
    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    async with Runtime.open("deferred", storage=storage, models=_FeatureModels(TestModel(call_tools=["read_file"])), features=features) as runtime:
        execution = await runtime.agents.get().start("read a file")
        async def wait_for_deferred():
            while True:
                record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
                if record and record.status is ExecutionStatus.WAITING_DEFERRED:
                    return record
                await asyncio.sleep(.01)

        record = await asyncio.wait_for(wait_for_deferred(), timeout=5)
        assert not record.conversation and not record.restart_checkpoints and not record.durable_trace
        checkpoint = await storage.recovery.checkpoints.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert checkpoint is not None and checkpoint.state is RecoveryCheckpointState.WAITING
        assert checkpoint.pending_tools is not None
        approvals = await storage.recovery.approvals.list_pending(execution.execution_id, tenant_id=runtime.tenant_id)
        assert len(approvals) == 1 and approvals[0].status is ApprovalStatus.PENDING


@pytest.mark.asyncio
@pytest.mark.parametrize("feature_type", (ConversationCapability, CheckpointCapability, TraceCapability))
async def test_native_type_requirement_survives_binding(feature_type):
    from pydantic_ai.capabilities import AbstractCapability, CapabilityOrdering
    observed = []
    class RequiresFeature(AbstractCapability):
        id = "test.requires-runtime-feature"
        def get_ordering(self):
            return CapabilityOrdering(requires=[feature_type], wraps=[feature_type])
        async def before_run(self, ctx):
            observed.append(ctx.deps.execution_id)
    template = feature_type()
    async with Runtime.open("ordering", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), features=_features(template, RequiresFeature())) as runtime:
        execution = await runtime.agents.get().start("question")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        assert observed == [execution.execution_id]
    assert not hasattr(template, "capture")


@pytest.mark.asyncio
@pytest.mark.parametrize('effect_policy', ('none', 'replay_safe'))
async def test_reconstructed_test_actual_effect_promotes_before_invocation_but_read_only_tool_stays_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, effect_policy: str,
) -> None:
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    calls = []
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=2**20))
    owner = {}
    async def lookup(context: AgentContext[None]) -> str:
        record = await state.execution.executions.get(context.execution_id, tenant_id=context.principal.tenant_id)
        calls.append(record)
        return 'value'
    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            assert await state.execution.executions.get(owner['execution'], tenant_id=owner['tenant']) is None
            return ModelResponse(parts=[ToolCallPart('lookup', {}, tool_call_id='lookup-call')])
        return ModelResponse(parts=[TextPart('answer')])
    group = RuntimeFeatures('tool-pending')
    group.tool(lookup, effect_policy=effect_policy)
    group.agent('default', model='default', allow_tools=('lookup',))
    group.capability(ConversationCapability(), scope='runtime')
    async with Runtime.open('pending-tool', storage=state, models=_FeatureModels(_FeatureFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        original = SqlStorageContext.run_mutation
        commits = []
        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get('domain'))
            return result
        monkeypatch.setattr(SqlStorageContext, 'run_mutation', count)
        execution = await session.start('question')
        owner.update(execution=execution.execution_id, tenant=runtime.tenant_id)
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 1
        if effect_policy == 'none':
            assert calls == [None]
            assert len(commits) == 1
        else:
            assert calls[0] is not None and calls[0].status is ExecutionStatus.STARTED
            assert len(commits) > 1



@pytest.mark.asyncio
async def test_reconstructed_test_pending_cancel_promotes_and_quiesces_before_cancelled(tmp_path: Path) -> None:
    entered = asyncio.Event()
    stopped = asyncio.Event()
    async def model(messages, info):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()
    group = RuntimeFeatures('cancel-pending')
    group.agent('default', model='default', allow_tools=())
    group.capability(ConversationCapability(), scope='runtime')
    state = RuntimeStorage.sqlite(tmp_path / 'state.db', cache=InMemoryCacheBackend(max_bytes=2**20))
    async with Runtime.open('pending-cancel', storage=state, models=_FeatureModels(_FeatureFunctionModel(model)), features=group) as runtime:
        session = await runtime.agents.get().create_session('session')
        execution = await session.start('question')
        await asyncio.wait_for(entered.wait(), 5)
        assert await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
        await execution.cancel()
        assert stopped.is_set()
        result = (await execution.wait(timeout_seconds=5)).result
        assert result.status is ExecutionStatus.CANCELLED
        record = await state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.status is ExecutionStatus.CANCELLED



@pytest.mark.asyncio
async def test_cached_lost_ack_reconciles_without_replay(tmp_path, monkeypatch):
    storage = RuntimeStorage.sqlite(tmp_path / "state.db", cache=InMemoryCacheBackend(max_bytes=2**20))
    calls = []
    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart("done")])
    async with Runtime.open("lost-ack", storage=storage, models=_FeatureModels(_FeatureFunctionModel(model)), features=_features(ConversationCapability())) as runtime:
        session = await runtime.agents.get().create_session("session")
        original = SqlStorageContext.run_mutation
        writes = []
        async def lose_ack(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            writes.append(result)
            if len(writes) == 1:
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            return result
        monkeypatch.setattr(SqlStorageContext, "run_mutation", lose_ack)
        execution = await session.start("question", idempotency_key="turn")
        assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED
        replay = await session.start("question", idempotency_key="turn")
        assert replay.execution_id == execution.execution_id
        assert len(calls) == len(writes) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("source_conversation", (False, True))
async def test_fork_uses_stored_conversation_promise_and_actual_history(tmp_path, source_conversation):
    database = tmp_path / "state.db"
    messages_seen = []
    async def model(messages, info):
        messages_seen.append(messages)
        return ModelResponse(parts=[TextPart("done")])
    features = _features(ConversationCapability() if source_conversation else TraceCapability())
    async with Runtime.open("fork-promise", storage=RuntimeStorage.sqlite(database), models=_FeatureModels(_FeatureFunctionModel(model)), features=features) as runtime:
        source = await runtime.agents.get().start("original input")
        await source.wait(timeout_seconds=5)
        execution_id = source.execution_id
    async with Runtime.open("fork-promise", storage=RuntimeStorage.sqlite(database), models=_FeatureModels(_FeatureFunctionModel(model)), features=_features(ConversationCapability())) as runtime:
        source = await runtime.executions.get(execution_id)
        if not source_conversation:
            with pytest.raises(AIError) as missing:
                await source.fork("new input")
            assert missing.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING
        else:
            fork = await source.fork("new input")
            await fork.wait(timeout_seconds=5)
            prompts = [part.content for message in messages_seen[-1] for part in message.parts if isinstance(part, UserPromptPart)]
            assert prompts == ["original input", "new input"]


@pytest.mark.asyncio
@pytest.mark.parametrize("current_conversation", (False, True))
@pytest.mark.parametrize("stored_guarantee", (False, True))
async def test_handoff_uses_stored_guarantee(monkeypatch, current_conversation, stored_guarantee):
    from linktools.ai.runtime.state._retention import RuntimeRetentionController
    observed = []
    async def release(self, execution_id, *, tenant_id, terminal_commit=False):
        observed.append(terminal_commit)
        return True
    monkeypatch.setattr(RuntimeRetentionController, "release_execution_handoff", release)
    features = _features(ConversationCapability()) if current_conversation else _definitions()
    async with Runtime.open("handoff", storage=RuntimeStorage.in_memory(), models=_FeatureModels(), features=features) as runtime:
        assert await runtime._execution_service._release_terminal("terminal", tenant_id=runtime.tenant_id, terminal_commit=stored_guarantee)
        assert observed == [stored_guarantee]


@pytest.mark.asyncio
async def test_capture_does_not_reinterpret_later_feature_registration():
    features = _features(ConversationCapability())
    captured = await features.capture()
    features.capability(TraceCapability(), scope="runtime")
    storage = RuntimeStorage.in_memory()
    async with Runtime.open("capture", storage=storage, models=_FeatureModels(), features=captured) as runtime:
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question")
        await execution.wait(timeout_seconds=5)
        record = await storage.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.conversation and not record.durable_trace
