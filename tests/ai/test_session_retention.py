#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient Session retention regressions for forked timelines."""

from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from linktools.ai.core import SessionStatus
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import ConversationCursor
from linktools.ai.runtime.state._retention import RuntimeRetentionController


@dataclass(frozen=True)
class _Session:
    session_id: str
    status: SessionStatus
    continuation: ConversationCursor | None
    timeline_parent_session_id: str | None = None


class _Sessions:
    def __init__(self, records: tuple[_Session, ...]) -> None:
        self._records = records

    async def list(self, *, tenant_id: str, owner_principal_id=None):
        assert tenant_id == "tenant"
        assert owner_principal_id is None
        return self._records


class _Steps:
    def __init__(self) -> None:
        self.released: list[tuple[RuntimeDomain, str]] = []

    async def release_archive(self, domain: RuntimeDomain, run_id: str) -> None:
        self.released.append((domain, run_id))


class _Objects:
    def __init__(self) -> None:
        self.released: list[tuple[RuntimeDomain, str]] = []

    async def release_object_scope(
        self, domain: RuntimeDomain, *, owner_scope: str
    ) -> None:
        self.released.append((domain, owner_scope))


def _controller(records: tuple[_Session, ...]) -> tuple[RuntimeRetentionController, _Steps, _Objects]:
    steps = _Steps()
    objects = _Objects()
    controller = object.__new__(RuntimeRetentionController)
    controller._conversation = SimpleNamespace(sessions=_Sessions(records))
    controller._steps = steps
    controller._objects = objects
    controller._transient_domains = frozenset({RuntimeDomain.CONVERSATION})
    return controller, steps, objects


@pytest.mark.asyncio
async def test_closed_session_releases_unreferenced_conversation() -> None:
    continuation = ConversationCursor("parent-run")
    controller, steps, objects = _controller(
        (_Session("parent", SessionStatus.CLOSED, continuation),)
    )

    await controller.release_session(
        "parent", tenant_id="tenant", continuation=continuation
    )

    assert steps.released == [(RuntimeDomain.CONVERSATION, "parent-run")]
    assert objects.released == [
        (RuntimeDomain.CONVERSATION, "session:parent")
    ]


@pytest.mark.asyncio
async def test_open_fork_keeps_closed_parent_conversation() -> None:
    parent = _Session(
        "parent",
        SessionStatus.CLOSED,
        ConversationCursor("parent-run"),
    )
    child = _Session(
        "child",
        SessionStatus.OPEN,
        ConversationCursor("child-run"),
        timeline_parent_session_id="parent",
    )
    controller, steps, objects = _controller((parent, child))

    await controller.release_session(
        "parent", tenant_id="tenant", continuation=parent.continuation
    )

    assert steps.released == []
    assert objects.released == []


@pytest.mark.asyncio
async def test_last_closed_fork_releases_closed_ancestor_chain() -> None:
    parent = _Session(
        "parent",
        SessionStatus.CLOSED,
        ConversationCursor("parent-run"),
    )
    child = _Session(
        "child",
        SessionStatus.CLOSED,
        ConversationCursor("child-run"),
        timeline_parent_session_id="parent",
    )
    controller, steps, objects = _controller((parent, child))

    await controller.release_session(
        "child", tenant_id="tenant", continuation=child.continuation
    )

    assert steps.released == [
        (RuntimeDomain.CONVERSATION, "child-run"),
        (RuntimeDomain.CONVERSATION, "parent-run"),
    ]
    assert objects.released == [
        (RuntimeDomain.CONVERSATION, "session:child"),
        (RuntimeDomain.CONVERSATION, "session:parent"),
    ]
