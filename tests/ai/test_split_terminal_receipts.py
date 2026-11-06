#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split handoff completion uses confirmed execution receipts."""

from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.capability import ExecutionCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence
from linktools.ai.persistence.contracts import ExecutionRecoveryState, RecoveryHandoffPhase
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.runtime._local import LocalExecutionBackend

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from ._session_tool_test_helpers import _split_sqlite_storage


@pytest.mark.asyncio
@pytest.mark.parametrize("committed", (False, True), ids=("unconfirmed", "lost-ack"))
@pytest.mark.parametrize("drop_notification", (False, True), ids=("notify", "no-notify"))
async def test_split_handoff_unknown_requires_source_confirmation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, committed: bool, drop_notification: bool,
) -> None:
    calls: list[str] = []
    publications: list[str] = []
    completion_writes: list[str] = []
    database = tmp_path / "terminal.db"

    async def model(messages, info):
        calls.append("model")
        return ModelResponse(parts=[TextPart("durable answer")])

    features = RuntimeFeatures("application")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.agent("default", model="default", allow_tools=())
    models = _Models(_UsageFunctionModel(model))
    original_publish = LocalExecutionBackend._publish_terminal_event

    def publish(self, execution_id, **kwargs):
        publications.append(execution_id)
        if not drop_notification:
            return original_publish(self, execution_id, **kwargs)

    async with Persistence.open("split-receipt", storage=_split_sqlite_storage(database)) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(models, features)) as runtime:
            repository = runtime._state.execution.executions
            original_swap = repository.compare_and_swap

            async def swap(execution_id, **kwargs):
                following = kwargs["next_record"]
                if (following.recovery is not None
                        and following.recovery.handoff_phase is RecoveryHandoffPhase.COMPLETED):
                    completion_writes.append(execution_id)
                    if committed:
                        await original_swap(execution_id, **kwargs)
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return await original_swap(execution_id, **kwargs)

            with monkeypatch.context() as fault:
                fault.setattr(repository, "compare_and_swap", swap)
                fault.setattr(LocalExecutionBackend, "_publish_terminal_event", publish)
                session = await runtime.agents.get().create_session("session")
                execution = await session.start("question", idempotency_key="question")
                assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
                await runtime.close()
                stored = await repository.get(execution.execution_id, tenant_id=runtime.tenant_id)
                assert stored is not None
                assert stored.recovery_state is (
                    ExecutionRecoveryState.COMPLETED if committed else ExecutionRecoveryState.HANDOFF
                )
                assert publications == [execution.execution_id]
                if not committed:
                    assert not stored.retention_closed
                assert completion_writes == [execution.execution_id]
                assert calls == ["model"]

    async with Persistence.open("split-receipt", storage=_split_sqlite_storage(database)) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(models, features)) as runtime:
            stored = await runtime._state.execution.executions.get(execution.execution_id, tenant_id=runtime.tenant_id)
            assert stored is not None and stored.recovery_state is ExecutionRecoveryState.COMPLETED
            session = runtime.agents.get().session("session")
            replay = await session.start("question", idempotency_key="question")
            assert replay.execution_id == execution.execution_id
            assert (await replay.wait(timeout_seconds=10)).result.output == {"text": "durable answer"}
            retained = await runtime._state.conversation.sessions.get("session", tenant_id=runtime.tenant_id)
            assert retained is not None and retained.active_execution_id is None
            assert calls == ["model"]
