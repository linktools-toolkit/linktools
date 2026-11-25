#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Trace authorizes session and original turn before reading evidence."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.core import AuthorizationAction, Principal, TenantAuthorizationPolicy
from linktools.ai.capability import AgentContext
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.persistence.runs import StepEvent

from .test_official_runtime_recovery import _runtime


class _TraceAuthorization(TenantAuthorizationPolicy):
    async def authorize(self, principal, action, resource):
        if action is AuthorizationAction.EXECUTION_READ:
            if principal.principal_id == "denied":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if principal.principal_id == "another":
                resource = replace(resource, owner_principal_id=None)
        await super().authorize(principal, action, resource)


class _TraceSource:
    """Isolate the typed read seam while the production canonical writer is separate."""

    has_turn_trace = True

    def __init__(self, turn_id):
        self.turn_id = turn_id
        self.calls = []
        self.events = []

    def record(self, step):
        time = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=step)
        self.events.append((step, StepEvent(
            self.turn_id, "MODEL_REQUEST_STARTED", step, timestamp=time,
            agent_conversation_id="chat", metadata={"linktools.ai.model_request_seq": str(step)},
        )))

    async def list_turn_trace_events(self, *, turn_id, high_water, after_timestamp, after_sequence, limit):
        assert turn_id == self.turn_id
        self.calls.append(turn_id)
        head = len(self.events) if high_water is None else high_water
        rows = [
            (seq, event) for seq, event in self.events if seq <= head and
            (after_timestamp is None or (event.timestamp, seq) > (after_timestamp, after_sequence))
        ]
        return head, rows[:limit]


@pytest.mark.asyncio
async def test_canonical_trace_is_authorized_and_cursor_keeps_captured_head(tmp_path: Path) -> None:
    policy = _TraceAuthorization()

    async def work(_context: AgentContext[None]) -> str:
        return "not called"

    async def model(_messages, _info):
        return ModelResponse(parts=[TextPart("answer")])

    async with _runtime(tmp_path / "canonical-trace-reader.db", model, work, authorization=policy) as (
        runtime, persistence, _steps,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("chat", principal=principal)
        await runtime.sessions.create_turn_session("other", principal=principal)
        turn_id = await runtime.sessions.start_turn(
            "chat", "question", agent_id="default", principal=principal, idempotency_key="start",
        )
        source = _TraceSource(turn_id)
        source.record(1)
        source.record(2)
        history = History(
            runtime.history, tenant_id=persistence.tenant_id, namespace=persistence.namespace,
            authorization=policy, session_turn_reader=persistence.session_turn_reader,
            session_trace_source=source,
        )
        try:
            assert history.has_turn_trace
            denied = Principal("denied", tenant_id=principal.tenant_id)
            for session, actor in (("chat", denied), ("other", principal)):
                with pytest.raises(AIError) as error:
                    await history.turn_trace(session, turn_id, principal=actor, limit=1)
                assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert source.calls == []

            first = await history.turn_trace("chat", turn_id, principal=principal, limit=1)
            one, = first.items
            assert (one.turn_id, one.step_event_seq, one.payload["model_request_seq"]) == (turn_id, 1, 1)
            assert first.next_cursor is not None

            another_user = Principal("another", tenant_id=principal.tenant_id)
            with pytest.raises(AIError) as invalid:
                await history.turn_trace("chat", turn_id, principal=another_user, cursor=first.next_cursor, limit=1)
            assert invalid.value.code is ErrorCode.CURSOR_INVALID
            assert source.calls == [turn_id]

            source.record(3)
            second = await history.turn_trace("chat", turn_id, principal=principal, cursor=first.next_cursor, limit=1)
            two, = second.items
            assert (two.turn_id, two.step_event_seq) == (turn_id, 2)
            assert second.next_cursor is None

            fresh = await history.turn_trace("chat", turn_id, principal=principal)
            assert [item.step_event_seq for item in fresh.items] == [1, 2, 3]
        finally:
            await history.close()
