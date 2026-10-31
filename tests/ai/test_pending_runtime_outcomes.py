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
    if phase == "terminal":
        model_release.set()

    async def model(messages, info):
        model_entered.set()
        await model_release.wait()
        return ModelResponse(parts=[TextPart("answer")])

    features = RuntimeFeatures("pending-ownership")
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    storage = PersistenceStorage.sqlite(tmp_path / "state.db",
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open("pending-ownership", storage=storage) as persistence:
        context = Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features))
        runtime = await context.__aenter__()
        try:
            backend = runtime._execution_service.runtime_backend()
            original = backend._context_archive.prepare

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
            else:
                assert await coordination.release(pending.coordination.lease)
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
            if interference == "lease":
                with pytest.raises(AIError) as failed_close:
                    await context.__aexit__(None, None, None)
                assert failed_close.value.code is ErrorCode.STORAGE_CONFLICT
            else:
                await context.__aexit__(None, None, None)
