#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session timeline recovery handoff regressions."""

from linktools.ai.persistence import Persistence
from dataclasses import replace
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
from linktools.ai.persistence import PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.contracts import (
    ConversationCursor,
    RecoveryConversationIntent,
    RecoveryTerminalHandoff,
    RecoveryTerminalOutcome,
    RecoveryCheckpoint,
    RecoveryCheckpointState,
    RecoveryHandoffPhase,
    SessionRecord,
)
from linktools.ai.persistence._step_contracts import (
    AgentRunCheckpoint,
    AgentRunRecord,
)
from linktools.ai.storage import StoredPayload
from linktools.ai.persistence._runtime_commands import StateCommands
from tests.ai.test_session_timeline import _execution


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
    state = PersistenceStorage.in_memory()
    state = Persistence(state)
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

        archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
        now = datetime.now(timezone.utc)
        run = AgentRunRecord(
            agent_run_id="run",
            agent_conversation_id="conversation",
            parent_agent_run_id=None,
            agent_id="agent",
            metadata={"history_id": session.history_id},
            started_at=now,
        )
        agent_run_checkpoint = AgentRunCheckpoint(
            agent_run_id="run",
            step_index=1,
            messages=[
                ModelRequest(parts=[UserPromptPart(content="prompt")]),
                ModelResponse(parts=[TextPart(content="answer")]),
            ],
            agent_conversation_id="conversation",
            parent_agent_run_id=None,
            agent_id="agent",
            timestamp=now,
            state="complete",
            transcript_message_count_before=0,
        )

        class Lifecycle:
            async def materialize_from_recovery(self, **kwargs: object) -> None:
                assert kwargs == {
                    "target": PersistenceDomain.CONVERSATION,
                    "agent_run_id": "run",
                }
                if await archive.latest_checkpoint(agent_run_id="run") is None:
                    await archive.materialize_checkpoint(run, agent_run_checkpoint)

        backend = object.__new__(LocalExecutionBackend)
        backend._tenant_id = "tenant"
        backend._conversation = state.conversation
        backend._conversation_durable = True
        backend._run_stores = {
            PersistenceDomain.CONVERSATION: archive,
            PersistenceDomain.EXECUTION: object(),
            PersistenceDomain.RECOVERY: object(),
        }
        backend._run_store = state.run_store
        backend._agent_run_lifecycle = Lifecycle()
        terminal = replace(
            _execution("execution", status=ExecutionStatus.SUCCEEDED, prompt="prompt"),
            conversation=True,
        )
        observed = replace(terminal, status=ExecutionStatus.FINALIZING, result=None)

        async def load_execution(execution_id: str, *, tenant_id: str):
            assert execution_id == "execution" and tenant_id == "tenant"
            return observed

        async def get_result(execution_id: str, *, tenant_id: str):
            assert execution_id == "execution" and tenant_id == "tenant"
            return observed.result

        backend.load_execution = load_execution
        backend._execution = SimpleNamespace(
            executions=SimpleNamespace(get_result=get_result)
        )
        backend._runtime_commands = StateCommands(
            state.execution.executions,
            namespace="session-timeline-recovery",
            events=state.execution.events,
            conversation=state.conversation.sessions,
            conversation_run_store=archive,
            background_tasks=set(),
        )

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
        recovery_checkpoint = RecoveryCheckpoint(
            execution_id="execution",
            agent_run_id="run",
            state=RecoveryCheckpointState.HANDOFF,
            revision=0,
            created_at=now,
            updated_at=now,
            handoff_phase=RecoveryHandoffPhase.PREPARED,
            terminal_handoff=handoff,
        )

        await backend._resolve_handoff_conversation(recovery_checkpoint, handoff)

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
            start_turn_seq=1,
            end_turn_seq=2,
        )
        assert commits == ()
        assert resolved.active_execution_id == "execution"

        observed = terminal
        await backend._resolve_handoff_conversation(recovery_checkpoint, handoff)
        commits = await state.conversation.sessions.list_timeline_commits(
            "session",
            tenant_id="tenant",
            start_turn_seq=1,
            end_turn_seq=2,
        )
        assert len(commits) == 1
        assert commits[0].execution_id == "execution"
        assert commits[0].start_message_index == 0
        assert commits[0].end_message_index == 2
    finally:
        await state.close()
