#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local Web turn ownership is separate from durable storage and content caches."""

import asyncio
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CreateSessionRequest
from linktools.ai.storage import InMemoryCacheBackend, StreamBackend, TurnCoordinationBackend
from linktools.ai.workspace import Workspace
from linktools.commands.ai import web
from linktools.commands.ai._common import _local_runtime_storage, _open_local_runtime

from ._runtime_test_helpers import _Models, _UsageFunctionModel


def test_web_owns_a_fresh_coordination_backend_per_server(monkeypatch: pytest.MonkeyPatch) -> None:
    backends = []

    def run_local(args, execute, *, coordination=None):
        backends.append(coordination)
        return 0

    monkeypatch.setattr(web, "run_local", run_local)
    args = web.command.create_parser().parse_args([])
    assert web.command.run(args) == 0
    assert web.command.run(args) == 0
    assert len(backends) == 2
    assert backends[0] is not backends[1]
    for backend in backends:
        assert isinstance(backend, TurnCoordinationBackend)
        assert isinstance(backend, InMemoryCacheBackend)
        assert not isinstance(backend, StreamBackend)
        assert backend.max_bytes == 0


def test_coordination_does_not_replace_local_durable_routes_or_content_cache(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    default = _local_runtime_storage(workspace)
    backend = InMemoryCacheBackend(max_bytes=0)
    local = _local_runtime_storage(workspace, coordination=backend)
    assert default.coordination is None
    assert local.coordination is backend
    for domain in default.plan.durable_domains:
        assert local.plan.route(domain) == default.plan.route(domain)
        assert local.plan.route(domain).cache is default.plan.route(domain).cache
        assert not local.plan.route(domain).exclusive_writer


@pytest.mark.asyncio
@pytest.mark.parametrize("coordinated", (False, True), ids=("durable-default", "local-owner"))
async def test_local_session_rejects_busy_turn_and_releases(
    tmp_path: Path, coordinated: bool, record_property,
) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("complete")])

    workspace = Workspace.initialize(tmp_path)
    backend = InMemoryCacheBackend(max_bytes=0) if coordinated else None
    async with _open_local_runtime(
        workspace, model=_Models(_UsageFunctionModel(respond)), coordination=backend,
    ) as local:
        runtime = local.runtime
        actor = runtime.default_principal
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        first = await runtime.sessions.start("session", "hello", principal=actor, idempotency_key="first")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            from sqlalchemy import event
            from sqlalchemy.engine import Engine

            selects = []

            def count_select(conn, cursor, statement, parameters, context, executemany):
                if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                    selects.append(1)

            event.listen(Engine, "after_cursor_execute", count_select)
            try:
                with pytest.raises(AIError) as caught:
                    await runtime.sessions.start("session", "other", principal=actor, idempotency_key="second")
            finally:
                event.remove(Engine, "after_cursor_execute", count_select)
            record_property("busy_attempt_selects", len(selects))
            assert caught.value.code is ErrorCode.SESSION_BUSY
            await runtime.sessions.create("default", CreateSessionRequest(actor, "other-session", "create-other"))
            independent = await runtime.sessions.start("other-session", "independent", principal=actor, idempotency_key="other")
            assert independent.execution_id != first.execution_id
        finally:
            release.set()
        assert (await first.wait(timeout_seconds=10)).result.output == {"text": "complete"}
        assert (await independent.wait(timeout_seconds=10)).result.output == {"text": "complete"}
        next_run = await runtime.sessions.start("session", "next", principal=actor, idempotency_key="third")
        assert (await next_run.wait(timeout_seconds=10)).result.output == {"text": "complete"}
