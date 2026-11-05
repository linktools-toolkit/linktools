#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Effect-free failed turns publish their actual outcome without source promotion."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import InMemoryCacheBackend
from linktools.ai.storage._database import SqlStorageContext

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_runtime_turn_lifecycle import _ObservedCache
from .test_turn_scope_runtime_contract import _ProjectionAuthority


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
@pytest.mark.parametrize("sessionless", (False, True))
async def test_pending_unsuccessful_turn_has_one_source_commit_and_no_model_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool, sessionless: bool,
) -> None:
    calls = []

    async def model(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            if cancelled:
                raise asyncio.CancelledError
            raise AIError(ErrorCode.INTERNAL_ERROR)
        return ModelResponse(parts=[TextPart("next answer")])

    features = RuntimeFeatures("pending-outcome")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("pending-outcome", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        agent = runtime.agents.get()
        target = agent if sessionless else await agent.create_session("session")
        commits = []
        original = SqlStorageContext.run_mutation

        async def count(context, callback, **kwargs):
            result = await original(context, callback, **kwargs)
            commits.append(kwargs.get("domain"))
            return result

        monkeypatch.setattr(SqlStorageContext, "run_mutation", count)
        execution = await target.start("question", idempotency_key="first")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is (ExecutionStatus.CANCELLED if cancelled else ExecutionStatus.FAILED)
        assert len(commits) == 1
        record = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
        assert record is not None and record.status is result.status
        if not sessionless:
            session = await runtime._state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert session.active_execution_id is None
            assert session.continuation is None
        if sessionless:
            with pytest.raises(AIError) as replay:
                await target.start("question", idempotency_key="first")
            assert replay.value.code is (ErrorCode.EXECUTION_CANCELLED if cancelled else ErrorCode.INTERNAL_ERROR)
        else:
            replay = await target.start("question", idempotency_key="first")
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait(timeout_seconds=10)).result.status is result.status
        assert len(calls) == 1
        following = await target.start("next", idempotency_key="second")
        assert (await following.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert len(calls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("terminal", "promotion"))
@pytest.mark.parametrize("interference", ("cancel", "lease"))
async def test_pending_source_dispatch_rechecks_ownership_after_context_materialization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, phase: str, interference: str,
) -> None:
    model_entered = asyncio.Event()
    model_release = asyncio.Event()
    prepare_entered = asyncio.Event()
    prepare_release = asyncio.Event()
    cancel_observed = asyncio.Event()
    if phase == "terminal":
        model_release.set()

    async def model(messages, info):
        model_entered.set()
        await model_release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("pending-ownership")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    cache = _ObservedCache()
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=cache)
    persistence = Persistence(storage)
    await persistence.initialize(namespace="pending-ownership", tenant_id="default")
    source_group = persistence.execution.executions.state_store.storage_group
    try:
        context = Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features))
        runtime = await context.__aenter__()
        try:
            backend = runtime._execution_service.runtime_backend()
            original = backend._context_archive.prepare
            original_cancel = backend._pending_cancel

            async def observe_cancel(*args, **kwargs):
                cancel_observed.set()
                return await original_cancel(*args, **kwargs)

            monkeypatch.setattr(backend, "_pending_cancel", observe_cancel)

            async def gated(*args, **kwargs):
                prepare_entered.set()
                await prepare_release.wait()
                return await original(*args, **kwargs)

            monkeypatch.setattr(backend._context_archive, "prepare", gated)
            session = await runtime.agents.get().create_session("session")
            execution = await session.start("question")
            await asyncio.wait_for(model_entered.wait(), 5)
            promotion = asyncio.create_task(backend.promote_pending_execution(execution.execution_id)) if phase == "promotion" else None
            await asyncio.wait_for(prepare_entered.wait(), 5)
            pending = backend._pending_turns.get(execution.execution_id, tenant_id=runtime.tenant_id)
            coordination = backend._pending_turns.coordination
            if interference == "cancel":
                assert await coordination.cancel(execution.execution_id, tenant_id=runtime.tenant_id)
                await asyncio.wait_for(cancel_observed.wait(), 5)
            else:
                owner = backend._turn_owners[execution.execution_id]
                assert await cache.release_turn(pending.coordination.lease)

                async def wait_for_owner_loss():
                    while not owner.stopping:
                        await asyncio.sleep(0)

                await asyncio.wait_for(wait_for_owner_loss(), 5)
            prepare_release.set()
            if promotion is not None:
                with pytest.raises(asyncio.CancelledError if interference == "cancel" else AIError):
                    await promotion
                assert await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
            model_release.set()
            if interference == "cancel":
                result = (await execution.wait(timeout_seconds=10)).result
                assert result.status is ExecutionStatus.CANCELLED
            else:
                with pytest.raises(AIError) as failed:
                    await execution.wait(timeout_seconds=10)
                assert failed.value.code is ErrorCode.STORAGE_CONFLICT
                assert await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id) is None
        finally:
            prepare_release.set()
            model_release.set()
            await context.__aexit__(None, None, None)
            assert not backend._turn_owners
            if interference == "lease":
                assert await persistence.execution.executions.get(
                    execution.execution_id, tenant_id=runtime.tenant_id,
                ) is None
                assert await persistence.execution.executions.get(
                    execution.execution_id, tenant_id=runtime.tenant_id,
                ) is None
    finally:
        try:
            assert not persistence._runtime_owners
            await persistence.close()
        finally:
            await source_group.close()
            await source_group.context.close()
            assert source_group.context.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("lost", (False, True))
async def test_pending_tools_share_local_owner_without_per_call_renewal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, lost: bool,
) -> None:
    from contextvars import ContextVar
    from pydantic_ai.messages import ToolCallPart, ToolReturnPart
    from linktools.ai.capability import AgentContext

    from linktools.ai.storage import TurnCoordination
    tool_boundary = ContextVar("tool_boundary", default=False)
    cache = _ProjectionAuthority()
    renew = cache.renew_turn
    effects = []

    async def tracked_renew(*args: object, **kwargs: object) -> TurnCoordination | None:
        assert not tool_boundary.get(), "tool admission must use its current local lease"
        return await renew(*args, **kwargs)

    monkeypatch.setattr(cache, "renew_turn", tracked_renew)

    async def work(_context: AgentContext[None]) -> str:
        effects.append(1)
        return "done"

    async def model(messages, info):
        if not any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            if lost:
                cache.scopes[0].lost.set()
            return ModelResponse(parts=[ToolCallPart("work", {}, tool_call_id=f"call-{i}") for i in range(3)])
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("pending-owner-local")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.tool(work, effect_policy="replay_safe")
    features.agent("default", model="default", allow_tools=("work",))
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=cache)
    async with Persistence.open("pending-owner-local", storage=storage) as persistence, Runtime.attach(
        persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features),
    ) as runtime:
        backend = runtime._execution_service.runtime_backend()
        original = backend.require_tool_authority

        async def require_tool_authority(execution_id: str, agent_run_id: str) -> None:
            token = tool_boundary.set(True)
            try:
                await original(execution_id, agent_run_id)
            finally:
                tool_boundary.reset(token)

        monkeypatch.setattr(backend, "require_tool_authority", require_tool_authority)
        session = await runtime.agents.get().create_session("session")
        execution = await session.start("question", idempotency_key="once")
        if lost:
            with pytest.raises(AIError) as raised:
                await execution.wait(timeout_seconds=10)
            assert raised.value.code is ErrorCode.STORAGE_CONFLICT
            assert effects == []
            return
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert len(effects) == 3
