#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unsubmitted failed turns release only their own staging, not borrowed storage."""

import asyncio
from collections.abc import Sequence
from pathlib import Path

from pydantic_ai.models.function import AgentInfo
from pydantic_ai.messages import ModelMessage

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability, TraceCapability
from linktools.ai.core import CommitPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._pending_turn import _PendingExecutionTurn

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_revoked_buffer_cleanup import _MissingPayloadBuffer


class _ScopedAckFailureBuffer(_MissingPayloadBuffer):
    def __init__(self) -> None:
        super().__init__()
        self.failed_scopes: set[str] = set()

    async def ack(self, scope: str, ids: Sequence[str]) -> None:
        if scope in self.failed_scopes:
            raise OSError("temporary old-turn cleanup failure")
        await super().ack(scope, ids)


@pytest.mark.asyncio
@pytest.mark.parametrize("payload_format", ("execution-payload-v1", "trace-projection-v1"))
@pytest.mark.parametrize("ack_failure", (False, True))
async def test_failed_unsubmitted_turn_preserves_error_and_releases_exact_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, payload_format: str, ack_failure: bool,
) -> None:
    buffer = _ScopedAckFailureBuffer()
    foreign = ("other-owner", "entry")
    assert await buffer.append(*foreign, "key", b"{}", wait=False)
    policy = CommitPolicy.terminal(buffer=buffer)
    storage = PersistenceStorage.sqlite(tmp_path / "state.db", coordination=buffer)
    calls = 0
    failed_entries: set[tuple[str, str]] = set()

    async def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        nonlocal calls
        calls += 1
        if calls == 1:
            failed_entries.update(buffer.pending.keys() - {foreign})
            if ack_failure:
                buffer.failed_scopes.update(scope for scope, _ in failed_entries)
            buffer.lose(payload_format)
            return ModelResponse(parts=[TextPart("unconfirmed failed output")])
        return ModelResponse(parts=[TextPart("independent next output")])

    features = RuntimeFeatures("failed-pending")
    features.capability(ExecutionCapability(retain_history=True, commit=policy))
    features.capability(TraceCapability(commit=policy))
    features.agent("default", model="default")
    model_features(_Models(_UsageFunctionModel(model)), features)

    async with Persistence.open("failed-pending", storage=storage) as state:
        async with Runtime.attach(persistence=state, features=features, metrics=None,
                                  auto_recover=False) as runtime:
            backend = runtime._execution_service.runtime_backend()
            cleanup_finished = asyncio.Event()
            original_cleanup = backend._cleanup_failed_pending_turn

            async def cleanup(turn: _PendingExecutionTurn, *, forget: bool = False) -> None:
                try:
                    await original_cleanup(turn, forget=forget)
                finally:
                    if not forget:
                        cleanup_finished.set()

            monkeypatch.setattr(backend, "_cleanup_failed_pending_turn", cleanup)
            session = await runtime.agents.get().create_session("session")
            other = await runtime.agents.get().create_session("other-session")
            failed = await session.start("first turn", idempotency_key="first")
            with pytest.raises(AIError) as caught:
                await failed.wait(timeout_seconds=10)
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert await state.execution.executions.get(failed.execution_id, tenant_id=state.tenant_id) is None
            assert await state.execution.executions.get(failed.execution_id, tenant_id=state.tenant_id) is None
            assert failed_entries
            assert buffer.close_calls == 0
            await asyncio.wait_for(cleanup_finished.wait(), 10)
            if ack_failure:
                assert failed_entries & buffer.pending.keys()
            else:
                assert set(buffer.pending) == {foreign}
                assert not backend._run_store.has_active_execution_producer(failed.execution_id)
            # Automatic cleanup must not turn the original failed wait into success.
            with pytest.raises(AIError) as repeated:
                await failed.wait(timeout_seconds=10)
            assert repeated.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

            # Failure of only the old scopes must not block either Session.
            next_turn = await session.start("next request", idempotency_key="next")
            next_result = await next_turn.wait(timeout_seconds=10)
            assert next_result.result.output == {"text": "independent next output"}
            other_turn = await other.start("other request", idempotency_key="other")
            other_result = await other_turn.wait(timeout_seconds=10)
            assert other_result.result.output == {"text": "independent next output"}
            assert len({failed.execution_id, next_turn.execution_id, other_turn.execution_id}) == 3
            assert calls == 3
            if ack_failure:
                with pytest.raises(AIError):
                    await runtime.close()
                assert failed_entries & buffer.pending.keys()
                assert foreign in buffer.pending and buffer.close_calls == 0
                buffer.failed_scopes.clear()
                # Closing has stopped admission; retry the exact same Runtime close.
            await runtime.close()

            assert set(buffer.pending) == {foreign}
            assert buffer.close_calls == 0
            assert await state.execution.executions.get(failed.execution_id, tenant_id=state.tenant_id) is None
            assert await state.execution.executions.get(failed.execution_id, tenant_id=state.tenant_id) is None
