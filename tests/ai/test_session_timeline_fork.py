#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Session timeline fork regressions."""

from datetime import datetime, timezone

import pytest

from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.spec import AgentSpec
from linktools.ai.core import (
    ExecutionLineageKind,
    ExecutionStatus,
    HmacCursorSigner,
    Principal,
    SessionStatus,
    TenantAuthorizationPolicy,
)
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.service_api import ForkSessionRequest
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._contracts import (
    ExecutionRecord,
    SessionRecord,
    StoredUserInput,
)
from linktools.ai.storage import StoredPayload


def _binding() -> AgentBindingSnapshot:
    return AgentBindingSnapshot(
        agent_spec=AgentSpec("agent", model="model"),
        base_model={"version": 1, "id": "model"},
        selected=(),
        subagents=(),
        output_mode="text",
        output_schema={"type": "object"},
    )


def _session(session_id: str) -> SessionRecord:
    now = datetime.now(timezone.utc)
    return SessionRecord(
        session_id=session_id,
        tenant_id="tenant",
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


def _execution(
    execution_id: str,
    *,
    status: ExecutionStatus,
    prompt: str,
) -> ExecutionRecord:
    now = datetime.now(timezone.utc)
    return ExecutionRecord(
        execution_id=execution_id,
        tenant_id="tenant",
        session_id="source",
        parent_execution_id=None,
        root_execution_id=execution_id,
        source_execution_id=None,
        base_execution_id=None,
        lineage_kind=ExecutionLineageKind.SESSION_RESUME,
        status=status,
        revision=1,
        event_sequence=0,
        agent_run_sequence=0,
        error_code="MODEL_UNAVAILABLE" if status is ExecutionStatus.FAILED else None,
        safe_error_details={},
        created_at=now,
        updated_at=now,
        mode="run",
        planning=False,
        thinking=False,
        binding=_binding(),
        principal_id="owner",
        principal_kind="user",
        stored_user_input=StoredUserInput(
            "text",
            StoredPayload.inline_text(prompt),
            {
                "version": 1,
                "prompt": {"kind": "text", "text": prompt},
                "files": [],
            },
        ),
    )


class _Executions:
    def __init__(self, values: dict[str, ExecutionRecord]) -> None:
        self.values = values

    async def get(self, execution_id: str, *, tenant_id: str):
        assert tenant_id == "tenant"
        return self.values.get(execution_id)

    async def get_many(self, execution_ids, *, tenant_id: str):
        assert tenant_id == "tenant"
        return {execution_id: self.values[execution_id] for execution_id in execution_ids}


class _ExecutionService:
    pass


@pytest.mark.asyncio
async def test_fork_freezes_terminal_turns_and_excludes_active_turn() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="session-timeline-fork", tenant_id="tenant")
    try:
        await state.conversation.sessions.create(_session("source"))
        await state.conversation.sessions.admit_execution(
            "source",
            tenant_id="tenant",
            execution_id="stable",
            expected=None,
        )
        await state.conversation.sessions.release_execution(
            "source",
            tenant_id="tenant",
            execution_id="stable",
        )
        await state.conversation.sessions.admit_execution(
            "source",
            tenant_id="tenant",
            execution_id="active",
            expected=None,
        )

        executions = _Executions(
            {
                "stable": _execution(
                    "stable",
                    status=ExecutionStatus.FAILED,
                    prompt="stable prompt",
                ),
                "active": _execution(
                    "active",
                    status=ExecutionStatus.STARTED,
                    prompt="active prompt",
                ),
            }
        )
        service = DefaultSessionService(
            state.conversation,
            executions,  # type: ignore[arg-type]
            TenantAuthorizationPolicy(),
            _ExecutionService(),  # type: ignore[arg-type]
            HmacCursorSigner("session", b"session-timeline-key"),
            history_reader=object(),  # type: ignore[arg-type]
            transcript_store=object(),  # type: ignore[arg-type]
        )
        principal = Principal("owner", "tenant")

        await service.fork(
            "agent",
            "source",
            ForkSessionRequest(principal, "child", "fork-1"),
        )
        child = await state.conversation.sessions.get("child", tenant_id="tenant")
        assert child is not None
        assert child.timeline_parent_session_id == "source"
        assert child.timeline_parent_turn_sequence == 1

        page = await service.timeline("child", principal=principal)
        assert [turn.execution_id for turn in page.items] == ["stable"]
        assert page.items[0].user_input == {
            "version": 1,
            "prompt": {"kind": "text", "text": "stable prompt"},
            "files": [],
        }
        assert page.next_cursor is None
    finally:
        await state.close()
