#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Runtime session auto-create races."""

import pytest
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._session import RuntimeSessionService
from linktools.ai.execution import SessionView
from linktools.ai.core import Principal, PrincipalKind


class _RaceSessionService:
    def __init__(self, *, authorize_after_conflict: bool) -> None:
        self._authorize_after_conflict = authorize_after_conflict
        self.get_calls = 0
        self.create_calls = 0

    async def read_authorized(self, session_id: str, principal: object, action: object) -> SessionView:
        del principal, action
        self.get_calls += 1
        if self.get_calls == 1 or not self._authorize_after_conflict:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return SessionView(
            session_id,
            "agent",
            SessionStatus.OPEN,
        )

    async def create(self, agent_id: str, request: object) -> SessionView:
        del agent_id, request
        self.create_calls += 1
        raise AIError(ErrorCode.STORAGE_CONFLICT)


@pytest.mark.asyncio
async def test_session_auto_create_race_reuses_authorized_session() -> None:
    service = _RaceSessionService(authorize_after_conflict=True)

    await RuntimeSessionService._prepare_resume_session(
        service,
        "agent",
        "session",
        Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
    )

    assert service.get_calls == 2
    assert service.create_calls == 1


@pytest.mark.asyncio
async def test_session_auto_create_conflict_does_not_reveal_foreign_session() -> None:
    service = _RaceSessionService(authorize_after_conflict=False)

    with pytest.raises(AIError) as error:
        await RuntimeSessionService._prepare_resume_session(
            service,
            "agent",
            "session",
            Principal("workspace", "tenant", PrincipalKind.LOCAL_TRUSTED.value),
        )

    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert service.get_calls == 2
    assert service.create_calls == 1
