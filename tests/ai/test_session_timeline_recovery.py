#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session timeline recovery handoff regressions."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, UserPromptPart

from linktools.ai.core import (
    ExecutionEventType,
    ExecutionStatus,
    SessionStatus,
    StopReason,
    UsageMetrics,
)
from linktools.ai.runtime._local import LocalExecutionBackend
from linktools.ai.runtime.state import RuntimeDomain, RuntimeStorage
from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    RecoveryConversationIntent,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    SessionRecord,
)
from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint, AgentRunRecord
from linktools.ai.storage import StoredPayload


def _session() -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id="session",
        owner_principal_id="owner",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        agent_id="agent",
    )


@pytest.mark.asyncio
async def test_recovery_handoff_commits_timeline_with_session_continuation() -> None:
    state = RuntimeStorage.in_memory()
    await state.initialize(namespace="session-timeline-recovery", tenant_id="tenant")
    try:
        session = await state.conversation.sessions.create(_session())
        assert session.history_id is not None
        await state.conversation.sessions.admit_execution(
            "session",
            tenant_id="tenant",
            execution_id="execution",
            expected=None,
        )

        archive = state.run_store.read_store(RuntimeDomain.CONVERSATION)
        now = datetime.now(timezone.utc)
        run = AgentRunRecord(
            agent_run_id="run",
            agent_conversation_id="conversation",
            parent_agent_run_id=None,
            agent_name="agent",
            metadata={"history_id": session.history_id},
            started_at=now,
        )
        checkpoint = AgentRunCheckpoint(
            agent_run_id="run",
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content="prompt")]),
                ModelResponse(parts=[TextPart(content="answer")]),
            ],
            agent_conversation_id="conversation",
            parent_agent_run_id=None,
            agent_name="agent",
            timestamp=now,
            state="complete",
            transcript_message_count_before=0,
        )

        class Lifecycle:
            async def materialize_from_recovery(self, **kwargs: object) -> None:
                assert kwargs == {
                    "target": RuntimeDomain.CONVERSATION,
                    "agent_run_id": "run",
                }
                await archive.materialize_checkpoint(run, checkpoint)

        backend = object.__new__(LocalExecutionBackend)
        backend._tenant_id = "tenant"
        backend._conversation = state.conversation
        backend._conversation_durable = True
        backend._step_reads = {RuntimeDomain.CONVERSATION: archive}
        backend._steps = state.run_store
        backend._step_lifecycle = Lifecycle()

        intent = RecoveryConversationIntent(
            "session",
            None,
            ConversationCursor("run", history_id=session.history_id),
        )
        handoff = RecoveryTerminalHandoff(
            RecoveryTerminalOutcome(
                terminal_status=ExecutionStatus.SUCCEEDED,
                error_code=None,
                safe_error_details={},
                stop_reason=StopReason.END_TURN,
                output=StoredPayload.inline_json({"text": "answer"}),
                object_source_domain=None,
                usage=UsageMetrics(),
                terminal_event_type=ExecutionEventType.EXECUTION_SUCCEEDED,
                terminal_event_payload={"agent_run_id": "run"},
                result_created_at=now,
            ),
            "run",
            intent,
        )
        checkpoint = SimpleNamespace(
            execution_id="execution",
            tenant_id="tenant",
        )

        await backend._resolve_handoff_conversation(checkpoint, handoff)

        resolved = await state.conversation.sessions.get("session", tenant_id="tenant")
        assert resolved is not None
        assert resolved.continuation == ConversationCursor(
            "run",
            history_id=session.history_id,
            message_count=2,
        )
        commits = await state.conversation.sessions.list_timeline_commits(
            "session",
            tenant_id="tenant",
            start_sequence=1,
            end_sequence=2,
        )
        assert len(commits) == 1
        assert commits[0].execution_id == "execution"
        assert commits[0].start_message_index == 0
        assert commits[0].end_message_index == 2
    finally:
        await state.close()
