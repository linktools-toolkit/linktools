#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The local CLI coordinates live turns through the shared durable source."""

import asyncio
import os
import sys
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from sqlalchemy import event
from sqlalchemy.engine import Engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import CreateSessionRequest
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _local_runtime_storage, _open_local_runtime

from ._runtime_test_helpers import _Models, _UsageFunctionModel


def test_local_uses_shared_sqlite_without_extra_turn_backend(tmp_path: Path) -> None:
    workspace = Workspace.initialize(tmp_path)
    local = _local_runtime_storage(workspace)
    durable_only = PersistenceStorage.from_root(workspace.storage_root / "runtime")
    assert local.coordination is None
    for domain in durable_only.plan.durable_domains:
        assert local.plan.route(domain) == durable_only.plan.route(domain)
        assert local.plan.route(domain).cache is durable_only.plan.route(domain).cache
        assert not local.plan.route(domain).exclusive_writer


async def _hold_turn(root: Path) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    async def respond(messages, info):
        entered.set()
        await release.wait()
        return ModelResponse(parts=[TextPart("complete")])

    async with _open_local_runtime(
        Workspace.initialize(root), model=_Models(_UsageFunctionModel(respond)),
    ) as local:
        runtime = local.runtime
        actor = runtime.default_principal
        await runtime.sessions.create("default", CreateSessionRequest(actor, "session", "create"))
        turn = await runtime.sessions.start("session", "first", principal=actor, idempotency_key="first")
        try:
            await asyncio.wait_for(entered.wait(), 10)
            print("turn-held", flush=True)
            await asyncio.to_thread(sys.stdin.readline)
        finally:
            release.set()
        assert (await turn.wait(timeout_seconds=10)).result.output == {"text": "complete"}


@pytest.mark.asyncio
async def test_local_session_rejects_cross_process_busy_and_releases(
    tmp_path: Path, record_property,
) -> None:
    model_calls = 0

    async def respond(messages, info):
        nonlocal model_calls
        model_calls += 1
        return ModelResponse(parts=[TextPart("complete")])

    owner = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "tests.ai.test_web_local_coordination", str(tmp_path),
        stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
        env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
    )
    try:
        assert owner.stdout is not None
        ready = await asyncio.wait_for(owner.stdout.readline(), 25)
        if ready != b"turn-held\n":
            _, stderr = await asyncio.wait_for(owner.communicate(), 15)
            pytest.fail(f"owner did not reach the held turn: {ready!r}\n{stderr.decode()}")
        async with _open_local_runtime(
            Workspace.discover(tmp_path), model=_Models(_UsageFunctionModel(respond)),
        ) as local:
            runtime = local.runtime
            actor = runtime.default_principal
            statements = []

            def count_statement(conn, cursor, statement, parameters, context, executemany):
                statements.append(statement.split(maxsplit=1)[0].upper())

            event.listen(Engine, "after_cursor_execute", count_statement)
            try:
                with pytest.raises(AIError) as caught:
                    await runtime.sessions.start("session", "other", principal=actor, idempotency_key="second")
            finally:
                event.remove(Engine, "after_cursor_execute", count_statement)
            record_property("busy_attempt_selects", sum(verb in {"SELECT", "WITH"} for verb in statements))
            assert caught.value.code is ErrorCode.SESSION_BUSY
            assert model_calls == 0

            await runtime.sessions.create("default", CreateSessionRequest(actor, "other-session", "create-other"))
            independent = await runtime.sessions.start("other-session", "independent", principal=actor, idempotency_key="other")
            assert (await independent.wait(timeout_seconds=10)).result.output == {"text": "complete"}

            _, stderr = await asyncio.wait_for(owner.communicate(b"release\n"), 20)
            assert owner.returncode == 0, stderr.decode()
            # The earlier rejected request is stable; a new request may enter.
            with pytest.raises(AIError) as replay:
                await runtime.sessions.start("session", "other", principal=actor, idempotency_key="second")
            assert replay.value.code is ErrorCode.SESSION_BUSY
            following = await runtime.sessions.start("session", "other", principal=actor, idempotency_key="after-release")
            assert (await following.wait(timeout_seconds=10)).result.output == {"text": "complete"}
            assert model_calls == 2
    finally:
        if owner.returncode is None:
            owner.kill()
            await owner.wait()


if __name__ == "__main__":
    asyncio.run(_hold_turn(Path(sys.argv[1])))
