#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskGraph entry points enforce tenant-scoped authorization before data access."""

import asyncio
from collections.abc import Awaitable, Callable
from types import SimpleNamespace

import pytest

from linktools.ai.core import AuthorizationAction, Principal, ResourceKind, ResourceRef, TaskStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.task import (
    CancelGraphRequest,
    DefaultTaskGraphService,
    RecoverGraphRequest,
    TaskEffectResolution,
    TaskEffectResolutionRequest,
    TaskGraphView,
    TaskInputSupplyRequest,
)

_Call = Callable[[DefaultTaskGraphService, Principal], Awaitable[object]]
_GRAPH_ID = "graph"
_PRINCIPAL = Principal("operator", "tenant")
_HEADER = ResourceRef(ResourceKind.TASK_GRAPH, _GRAPH_ID, "tenant", "original-owner")
_ENTRY_POINTS = [
    pytest.param(lambda s, p: s.recover(_GRAPH_ID, RecoverGraphRequest(p, "recover")),
                 AuthorizationAction.TASK_RUN, id="recover"),
    pytest.param(lambda s, p: s.recovery_nodes(_GRAPH_ID, principal=p),
                 AuthorizationAction.TASK_RUN, id="recovery_nodes"),
    pytest.param(lambda s, p: s.resume(_GRAPH_ID, "node", TaskInputSupplyRequest(p, "wait", None, "resume")),
                 AuthorizationAction.TASK_RUN, id="resume"),
    pytest.param(lambda s, p: s.resolve_effect(_GRAPH_ID, "node", TaskEffectResolutionRequest(
        p, 1, TaskEffectResolution("unknown"), "resolve")),
                 AuthorizationAction.TASK_RUN, id="resolve_effect"),
    pytest.param(lambda s, p: s.inspect(_GRAPH_ID, principal=p),
                 AuthorizationAction.TASK_READ, id="inspect"),
    pytest.param(lambda s, p: s.state(_GRAPH_ID, principal=p),
                 AuthorizationAction.TASK_READ, id="state"),
    pytest.param(lambda s, p: s.result_header(_GRAPH_ID, principal=p),
                 AuthorizationAction.TASK_READ, id="result_header"),
    pytest.param(lambda s, p: s.result_node_states(_GRAPH_ID, ("node",), principal=p),
                 AuthorizationAction.TASK_READ, id="result_node_states"),
    pytest.param(lambda s, p: s.list_events(_GRAPH_ID, principal=p),
                 AuthorizationAction.TASK_READ, id="list_events"),
    pytest.param(lambda s, p: s.stream_events(_GRAPH_ID, principal=p).__anext__(),
                 AuthorizationAction.TASK_READ, id="stream_events"),
    pytest.param(lambda s, p: s.wait(_GRAPH_ID, principal=p),
                 AuthorizationAction.TASK_READ, id="wait"),
    pytest.param(lambda s, p: s.cancel(_GRAPH_ID, CancelGraphRequest(p, "cancel")),
                 AuthorizationAction.TASK_CANCEL, id="cancel"),
    pytest.param(lambda s, p: s.cancel_node(_GRAPH_ID, "node", "execution", CancelGraphRequest(p, "cancel-node")),
                 AuthorizationAction.TASK_CANCEL, id="cancel_node"),
    pytest.param(lambda s, p: s.settle_execution_cancellation(
        _GRAPH_ID, "node", "execution", CancelGraphRequest(p, "settle"), cancel_confirmed=True),
                 AuthorizationAction.TASK_CANCEL, id="settle_execution_cancellation"),
]


class _Headers:
    def __init__(self, header: ResourceRef | None, events: list[object]) -> None:
        self.header = header
        self.events = events

    async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None:
        self.events.append(("header", graph_id, tenant_id))
        if self.header is not None and self.header.tenant_id == tenant_id:
            return self.header
        return None


class _Authorization:
    def __init__(self, events: list[object], error: AIError | None) -> None:
        self.events = events
        self.error = error

    async def authorize(
        self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
    ) -> None:
        self.events.append(("authorize", principal, action, resource))
        if self.error is not None:
            raise self.error


@pytest.mark.asyncio
@pytest.mark.parametrize("call,action", _ENTRY_POINTS)
@pytest.mark.parametrize("scope", ["available", "missing", "other-tenant"])
async def test_entry_points_deny_before_reading_or_mutating_graph(
    call: _Call, action: AuthorizationAction, scope: str,
) -> None:
    events: list[object] = []
    principal = Principal("operator", "other-tenant") if scope == "other-tenant" else _PRINCIPAL
    denial = AIError(ErrorCode.AUTHORIZATION_DENIED, safe_details={"reason": "policy"})
    service = DefaultTaskGraphService(
        SimpleNamespace(tasks=_Headers(None if scope == "missing" else _HEADER, events)),
        _Authorization(events, denial),
    )

    with pytest.raises(AIError) as caught:
        await call(service, principal)

    assert caught.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert events[0] == ("header", _GRAPH_ID, principal.tenant_id)
    if scope == "available":
        assert caught.value is denial
        assert events[1:] == [("authorize", principal, action, _HEADER)]
    else:
        assert events[1:] == []
        assert caught.value is not denial


@pytest.mark.asyncio
async def test_graph_access_rechecks_authorization_before_returning_data() -> None:
    events: list[object] = []
    view = TaskGraphView(_GRAPH_ID, TaskStatus.PENDING, ())

    class Graphs(_Headers):
        async def get_graph(self, graph_id: str, *, tenant_id: str) -> TaskGraphView:
            events.append(("read", graph_id, tenant_id))
            return view

    authorization = _Authorization(events, None)
    service = DefaultTaskGraphService(SimpleNamespace(tasks=Graphs(_HEADER, events)), authorization)
    assert await service.inspect(_GRAPH_ID, principal=_PRINCIPAL) is view
    authorization.error = AIError(ErrorCode.AUTHORIZATION_DENIED)
    with pytest.raises(AIError) as caught:
        await service.inspect(_GRAPH_ID, principal=_PRINCIPAL)
    assert caught.value is authorization.error
    assert events == [
        ("header", _GRAPH_ID, "tenant"),
        ("authorize", _PRINCIPAL, AuthorizationAction.TASK_READ, _HEADER),
        ("read", _GRAPH_ID, "tenant"),
        ("header", _GRAPH_ID, "tenant"),
        ("authorize", _PRINCIPAL, AuthorizationAction.TASK_READ, _HEADER),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("call,code", [
    pytest.param(lambda s, p: s.list_events(_GRAPH_ID, principal=p, limit=0),
                 ErrorCode.PAGE_LIMIT_INVALID, id="list_events"),
    pytest.param(lambda s, p: s.stream_events(_GRAPH_ID, principal=p, after_event_seq=-1).__anext__(),
                 ErrorCode.REQUEST_FIELD_INVALID, id="stream_events"),
    pytest.param(lambda s, p: s.wait(_GRAPH_ID, principal=p, timeout_seconds=-1),
                 ErrorCode.REQUEST_FIELD_INVALID, id="wait"),
])
async def test_query_argument_validation_precedes_authorization(call: _Call, code: ErrorCode) -> None:
    events: list[object] = []
    service = DefaultTaskGraphService(
        SimpleNamespace(tasks=_Headers(_HEADER, events)), _Authorization(events, None),
    )
    with pytest.raises(AIError) as caught:
        await call(service, _PRINCIPAL)
    assert caught.value.code is code
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["header", "authorization"])
async def test_wait_timeout_includes_graph_authorization(stage: str) -> None:
    events: list[object] = []
    interrupted = asyncio.Event()

    async def block() -> None:
        try:
            await asyncio.Event().wait()
        finally:
            interrupted.set()

    class Headers(_Headers):
        async def get_header(self, graph_id: str, *, tenant_id: str) -> ResourceRef | None:
            if stage == "header":
                await block()
            return await super().get_header(graph_id, tenant_id=tenant_id)

    class Authorization(_Authorization):
        async def authorize(
            self, principal: Principal, action: AuthorizationAction, resource: ResourceRef,
        ) -> None:
            await block()

    service = DefaultTaskGraphService(
        SimpleNamespace(tasks=Headers(_HEADER, events)), Authorization(events, None),
    )
    with pytest.raises(AIError) as caught:
        await asyncio.wait_for(service.wait(_GRAPH_ID, principal=_PRINCIPAL, timeout_seconds=0.01), 1)
    assert caught.value.code is ErrorCode.TASK_WAIT_TIMEOUT
    assert interrupted.is_set()
