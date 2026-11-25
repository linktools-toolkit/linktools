#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical local process notices trigger readback without a second stream or polling."""

import asyncio
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart
from sqlalchemy import event
from sqlalchemy.engine import Engine

pytest.importorskip("starlette")

from linktools.ai.web import create_app
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _open_local_runtime, local_turn_notices

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_web_console import _turn_text



async def _produce(project: Path, mode: str) -> None:
    if mode == "lost":
        from linktools.commands.ai import _session_notices

        publisher = _session_notices.session_notice_publisher

        @asynccontextmanager
        async def without_terminal(root: Path, **kwargs):
            async with publisher(root, **kwargs) as offer:
                def intercept(notice, discover=True):
                    if notice.get("phase") != "terminal":
                        offer(notice, discover=discover)
                yield intercept

        _session_notices.session_notice_publisher = without_terminal

    async def respond(messages, _info):
        if any(isinstance(part, ToolReturnPart) for part in messages[-1].parts):
            return ModelResponse(parts=[TextPart("complete")])
        await asyncio.to_thread(sys.stdin.readline)
        return ModelResponse(parts=[ToolCallPart(
            "write_memory", {"file": "notice.txt", "content": "tool-private-content"}, tool_call_id="call",
        )])

    async with _open_local_runtime(Workspace.discover(project), model=_Models(_UsageFunctionModel(respond))) as local:
        async with local_turn_notices(local):
            prompt = "user prompt from other process"
            if mode == "acp":
                import acp

                from linktools.ai.acp import ACPAgent

                agent = ACPAgent(local.runtime, principal=local.runtime.default_principal)
                await agent.initialize(acp.PROTOCOL_VERSION)
                result = await agent.prompt("session", [SimpleNamespace(text=prompt)])
                assert result.stop_reason == "end_turn"
            else:
                from linktools.ai.runtime.turn_clients import TurnObserver

                principal = local.runtime.default_principal
                with TurnObserver(local.runtime, "session", principal, local_workspace=True) as observer:
                    turn_id = await local.runtime.sessions.start_turn(
                        "session", prompt, agent="default", principal=principal, idempotency_key="produce",
                    )
                    async with asyncio.timeout(15):
                        async for result in observer.events(turn_id):
                            if result["type"] == "terminal":
                                break
                assert result["output"] == {"text": "complete"}, result
            if mode == "lost":
                print("committed", flush=True)
                await asyncio.Event().wait()


@pytest.mark.skipif(os.name != "posix", reason="private local process transport requires POSIX")
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["delivered", "lost", "acp"])
async def test_other_process_turns_use_committed_notices_and_targeted_readback_without_idle_sql(tmp_path: Path, mode: str) -> None:
    if mode == "acp":
        pytest.importorskip("acp")

    def forbidden_model(_messages, _info):
        pytest.fail("a passive observer must not execute the turn")

    workspace = Workspace.initialize(tmp_path)
    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(forbidden_model))) as local:
        runtime = local.runtime
        await runtime.sessions.create_turn_session("session", principal=runtime.default_principal, idempotency_key="create")
        async with local_turn_notices(local, receive=True) as feed:
            assert feed is not None
            app = create_app(runtime=runtime, session_notices=feed, local_workspace=True)
            path = "/api/session/events"
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"session_id=session&view=chat",
                "headers": [], "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345),
            }
            disconnected = asyncio.Event()
            frames: asyncio.Queue[dict[str, object]] = asyncio.Queue()

            async def receive() -> dict[str, object]:
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, object]) -> None:
                body = message.get("body")
                if isinstance(body, bytes) and b"data: " in body:
                    assert b"tool-private-content" not in body
                    for line in body.splitlines():
                        if line.startswith(b"data: "):
                            frames.put_nowait(json.loads(line[6:]))

            async def next_frame() -> dict[str, object]:
                return await asyncio.wait_for(frames.get(), 12)

            web_stream = asyncio.create_task(app(scope, receive, send))
            producer = None
            try:
                assert (await next_frame())["session"]["active_turn_id"] is None
                producer = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "tests.ai.test_cli_turn_sync", str(tmp_path), mode,
                    stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    cwd=Path(__file__).resolve().parents[2], env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
                )
                notice = await next_frame()
                turn_id = notice["turn_id"]
                assert notice["type"] == "gap" and turn_id

                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as http:
                    active = (await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})).json()
                    assert active["turn_id"] == turn_id and not active["conversation_committed"]
                    assert active["user_input"]["prompt"] == "user prompt from other process"

                    selects: list[str] = []

                    def count(_connection, _cursor, statement, _params, _context, _executemany):
                        if statement.lstrip().split(maxsplit=1)[0].upper() in {"SELECT", "WITH"}:
                            selects.append(statement)

                    event.listen(Engine, "after_cursor_execute", count)
                    try:
                        await asyncio.sleep(1.2)
                    finally:
                        event.remove(Engine, "after_cursor_execute", count)
                    assert not selects
                    assert producer.stdin is not None
                    producer.stdin.write(b"release\n")
                    await producer.stdin.drain()
                    if mode == "lost":
                        assert producer.stdout is not None
                        assert await asyncio.wait_for(producer.stdout.readline(), 10) == b"committed\n"
                        producer.kill()
                        await producer.wait()
                        # One-shot IPC gives no producer EOF to the Web stream.
                        # Explicit reconnect, not a timer, reads the committed release.
                        with pytest.raises(TimeoutError):
                            await asyncio.wait_for(frames.get(), 0.2)
                        disconnected.set()
                        await asyncio.wait_for(web_stream, 5)
                        disconnected.clear()
                        web_stream = asyncio.create_task(app(scope, receive, send))
                    terminal = await next_frame()
                    if mode == "lost":
                        assert terminal["type"] == "snapshot" and terminal["session"]["active_turn_id"] is None
                    else:
                        assert terminal["type"] == "gap" and terminal["turn_id"] == turn_id
                    saved = (await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})).json()
                    assert saved["turn_id"] == turn_id
                    assert saved["conversation_committed"] and _turn_text(saved) == "complete"
                    process, = saved["process"]
                    assert process["status"] == "SUCCEEDED" and process["key"] == "tool:call"
                    assert type(process["elapsed_ms"]) is int and process["elapsed_ms"] >= 0
                    assert process["argument_preview"] == "notice.txt"
                if mode == "lost":
                    return
                _, stderr = await asyncio.wait_for(producer.communicate(), 10)
                assert producer.returncode == 0, stderr.decode()
            finally:
                if producer is not None and producer.returncode is None:
                    producer.kill()
                    await producer.wait()
                disconnected.set()
                await asyncio.wait_for(web_stream, 5)


if __name__ == "__main__":
    asyncio.run(_produce(Path(sys.argv[1]), sys.argv[2]))
