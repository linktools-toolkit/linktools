#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stopped turns retry only previously requested, fenced claim releases."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy, ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.storage import Lease

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_revoked_buffer_cleanup import _MissingPayloadBuffer


class _ReleaseFaultBuffer(_MissingPayloadBuffer):
    def __init__(self) -> None:
        super().__init__()
        self.fail_release = False
        self.failed_leases: list[Lease] = []

    async def release_turn(self, lease: Lease) -> bool:
        if self.fail_release:
            self.failed_leases.append(lease)
            raise OSError("temporary claim release outage")
        return await super().release_turn(lease)


async def _join_turn_tasks(backend, execution_id: str) -> None:
    async with asyncio.timeout(10):
        worker = backend._tasks.get(execution_id)
        if worker is not None:
            await asyncio.gather(worker, return_exceptions=True)
        # The worker's done callback installs execution-scoped cleanup tasks.
        await asyncio.sleep(0)
        while True:
            tasks = tuple(task for task in backend._execution_task_map().get(execution_id, ())
                          if not task.done())
            if not tasks:
                break
            await asyncio.gather(*tasks, return_exceptions=True)
        renewal = backend._pending_renewals.get(execution_id)
        assert renewal is None or renewal.done()
        assert not backend.worker_installed(execution_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("lose_buffer,lose_release,standalone", (
    pytest.param(True, False, False, id="buffer-only"),
    pytest.param(False, True, False, id="release-only"),
    pytest.param(True, True, False, id="buffer-and-release"),
    pytest.param(True, True, True, id="standalone-same-request"),
))
async def test_next_turn_recovers_after_payload_or_release_failure(
    tmp_path: Path, lose_buffer: bool, lose_release: bool, standalone: bool,
) -> None:
    buffer = _ReleaseFaultBuffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    features = RuntimeFeatures("pending-fault-isolation")
    features.capability(ExecutionCapability(retain_history=True, commit=policy))
    features.capability(TraceCapability(commit=policy))
    features.agent("default", model="default")
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            if lose_buffer:
                buffer.lose("execution-payload-v1")
            buffer.fail_release = lose_release
        return ModelResponse(parts=[TextPart(f"answer {calls}")])

    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=buffer)
    async with Persistence.open("pending-fault-isolation", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features, metrics=None,
                                  auto_recover=False) as runtime:
            session = (runtime.agents.get() if standalone else
                       await runtime.agents.get().create_session("session"))
            first = await session.start("first", idempotency_key="first")
            if lose_buffer:
                with pytest.raises(AIError) as failed:
                    await first.wait(timeout_seconds=10)
                assert failed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                assert (await first.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            backend = runtime._execution_service.runtime_backend()
            await _join_turn_tasks(backend, first.execution_id)
            source = await state.execution.executions.get(first.execution_id, tenant_id=state.tenant_id)
            assert (source is None) is lose_buffer
            buffer.fail_release = False
            # The transport is healthy again, while a failed release still
            # leaves the original claim alive until admission retries it.
            if lose_release:
                assert buffer.failed_leases
                owner = await backend._pending_turns.coordination.read_owner(
                    first.execution_id, tenant_id=state.tenant_id,
                )
                assert owner is not None and owner.execution_id == first.execution_id
            else:
                assert not buffer.failed_leases
            second = await session.start(
                "first" if standalone else "second", idempotency_key="first" if standalone else "second",
            )
            assert (await second.wait(timeout_seconds=10)).result.output == {"text": "answer 2"}
            assert calls == 2 and first.execution_id != second.execution_id
            if lose_buffer:
                with pytest.raises(AIError) as repeated:
                    await first.wait(timeout_seconds=10)
                assert repeated.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                assert (await first.wait(timeout_seconds=10)).result.output == {"text": "answer 1"}


@pytest.mark.asyncio
async def test_unknown_source_retains_claim_while_other_session_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence._runtime_commands import StateCommands

    buffer = _ReleaseFaultBuffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    features = RuntimeFeatures("unknown-claim-isolation")
    features.capability(ExecutionCapability(retain_history=True, commit=policy))
    features.capability(TraceCapability(commit=policy))
    features.agent("default", model="default")
    calls = 0

    async def model(messages, info):
        nonlocal calls
        calls += 1
        return ModelResponse(parts=[TextPart(f"answer {calls}")])

    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=buffer)
    original = StateCommands.commit_pending_turn
    captured = []

    async def unknown(commands, *args, **kwargs):
        captured.append((commands, args, kwargs))
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

    async with Persistence.open("unknown-claim-isolation", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features, metrics=None,
                                  auto_recover=False) as runtime:
            session = await runtime.agents.get().create_session("unknown")
            with monkeypatch.context() as patch:
                patch.setattr(StateCommands, "commit_pending_turn", unknown)
                first = await session.start("first", idempotency_key="first")
                with pytest.raises(AIError) as failed:
                    await first.wait(timeout_seconds=10)
                assert failed.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            backend = runtime._execution_service.runtime_backend()
            pending = backend._pending_turns.get(first.execution_id, tenant_id=state.tenant_id)
            assert pending is not None and pending.uncertain
            owner = await backend._pending_turns.coordination.read_session_owner("unknown", tenant_id=state.tenant_id)
            assert owner is not None and owner.execution_id == first.execution_id
            with pytest.raises(AIError) as blocked:
                await session.start("second", idempotency_key="second")
            assert blocked.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert calls == 1 and buffer.pending
            other = await runtime.agents.get().create_session("healthy")
            independent = await other.start("independent", idempotency_key="independent")
            assert (await independent.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
            still_owned = await backend._pending_turns.coordination.read_session_owner("unknown", tenant_id=state.tenant_id)
            assert still_owned is not None and still_owned.execution_id == first.execution_id
            assert pending.uncertain and calls == 2
            # Deliver the original transaction outcome, then let the existing
            # fixed-plan readback settle it without rerunning the model.
            assert len(captured) == 1
            commands, args, kwargs = captured[0]
            await original(commands, *args, **kwargs)
            assert (await first.wait(timeout_seconds=10)).result.output == {"text": "answer 1"}
            assert calls == 2


@pytest.mark.asyncio
async def test_release_retry_preserves_promoted_execution_and_confirmed_effect(tmp_path: Path) -> None:
    from pydantic_ai.messages import ToolCallPart
    from linktools.ai.capability import AgentContext

    buffer = _ReleaseFaultBuffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    entered, finish = asyncio.Event(), asyncio.Event()
    effects = []
    calls = 0

    async def write(context: AgentContext[None]) -> str:
        effects.append("applied")
        entered.set()
        await finish.wait()
        return "applied"

    async def model(messages, info):
        nonlocal calls
        calls += 1
        if calls == 1:
            buffer.fail_release = True
            return ModelResponse(parts=[ToolCallPart("write", {}, tool_call_id="write-call")])
        return ModelResponse(parts=[TextPart("confirmed")])

    features = RuntimeFeatures("promoted-release-retry")
    features.capability(ExecutionCapability(retain_history=True, commit=policy))
    features.capability(TraceCapability(commit=policy))
    features.tool(write, effect_policy="non_replay_safe")
    features.agent("default", model="default", allow_tools=("write",))
    model_features(_Models(_UsageFunctionModel(model)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=buffer)
    async with Persistence.open("promoted-release-retry", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features, metrics=None,
                                  auto_recover=False) as runtime:
            session = await runtime.agents.get().create_session("session")
            first = await session.start("write", idempotency_key="first")
            try:
                await asyncio.wait_for(entered.wait(), 10)
                assert effects == ["applied"] and buffer.failed_leases
                source = await state.execution.executions.get(first.execution_id, tenant_id=state.tenant_id)
                assert source is not None and source.status is ExecutionStatus.STARTED
                buffer.fail_release = False
                # Promotion already authorized the old ephemeral release, but
                # the source session still excludes another turn during effects.
                with pytest.raises(AIError) as busy:
                    await session.start("second", idempotency_key="second")
                assert busy.value.code is ErrorCode.SESSION_BUSY
                assert not finish.is_set() and effects == ["applied"] and calls == 1
            finally:
                buffer.fail_release = False
                finish.set()
            assert (await first.wait(timeout_seconds=10)).result.output == {"text": "confirmed"}
            replay = await session.start("write", idempotency_key="first")
            assert replay.execution_id == first.execution_id and calls == 2
            assert (await replay.wait(timeout_seconds=10)).result.output == {"text": "confirmed"}
            second = await session.start("second", idempotency_key="second")
            assert (await second.wait(timeout_seconds=10)).result.output == {"text": "confirmed"}
            assert effects == ["applied"] and calls == 3
