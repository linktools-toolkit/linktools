#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Regression coverage for Runtime session auto-create races."""

from types import SimpleNamespace

import pytest

from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, SessionView
from linktools.ai.workspace import trusted_workspace_principal


class _RaceSessionService:
    def __init__(self, *, authorize_after_conflict: bool) -> None:
        self._authorize_after_conflict = authorize_after_conflict
        self.get_calls = 0
        self.create_calls = 0

    async def get(self, session_id: str, *, principal: object) -> SessionView:
        del principal
        self.get_calls += 1
        if self.get_calls == 1 or not self._authorize_after_conflict:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return SessionView(
            session_id,
            "a" * 64,
            SessionStatus.OPEN,
        )

    async def create(self, binding_digest: str, request: object) -> SessionView:
        del binding_digest, request
        self.create_calls += 1
        raise AIError(ErrorCode.STORAGE_CONFLICT)


@pytest.mark.asyncio
async def test_session_auto_create_race_reuses_authorized_session() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _RaceSessionService(authorize_after_conflict=True)
    definition = SimpleNamespace(
        digest="a" * 64,
        spec=SimpleNamespace(id="agent"),
    )

    await runtime._ensure_session(
        definition,
        "session",
        trusted_workspace_principal("tenant"),
    )

    assert runtime.session.get_calls == 2
    assert runtime.session.create_calls == 1


@pytest.mark.asyncio
async def test_session_auto_create_conflict_does_not_reveal_foreign_session() -> None:
    runtime = object.__new__(Runtime)
    runtime.session = _RaceSessionService(authorize_after_conflict=False)
    definition = SimpleNamespace(
        digest="a" * 64,
        spec=SimpleNamespace(id="agent"),
    )

    with pytest.raises(AIError) as error:
        await runtime._ensure_session(
            definition,
            "session",
            trusted_workspace_principal("tenant"),
        )

    assert error.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert runtime.session.get_calls == 2
    assert runtime.session.create_calls == 1
