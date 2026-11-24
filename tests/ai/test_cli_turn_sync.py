#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local process notices drive a Web session without an execution watch."""

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

from linktools.ai.session import CreateSessionRequest
pytest.importorskip("starlette")

from linktools.ai.web import create_app
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _open_local_runtime, local_turn_notices

from ._runtime_test_helpers import _Models, _UsageFunctionModel



async def _produce(project: Path, mode: str) -> None:
    if mode == "lost":
        from linktools.commands.ai import _session_notices

        publisher = _session_notices.session_notice_publisher

        @asynccontextmanager
        async def without_terminal(root: Path, **kwargs):
            async with publisher(root, **kwargs) as offer:
                def intercept(notice, discover=True):
                    if notice["kind"] != "terminal":
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
                execution = await local.runtime.sessions.start(
                    "session", prompt, principal=local.runtime.default_principal, idempotency_key="produce",
                )
                result = (await execution.wait(timeout_seconds=15)).result
                assert result.output == {"text": "complete"}, result
            if mode == "lost":
                print("committed", flush=True)
                await asyncio.Event().wait()


@pytest.mark.skipif(os.name != "posix", reason="private local process transport requires POSIX")
@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["delivered", "lost", "acp"])
async def test_other_process_tools_use_the_web_session_stream_without_idle_sql(tmp_path: Path, mode: str) -> None:
    if mode == "acp":
        pytest.importorskip("acp")

    def forbidden_model(_messages, _info):
        pytest.fail("a passive observer must not execute the turn")

    workspace = Workspace.initialize(tmp_path)
    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(forbidden_model))) as local:
        runtime = local.runtime
        await runtime.sessions.create("default", CreateSessionRequest(runtime.default_principal, "session", "create"))
        async with local_turn_notices(local, receive=True) as feed:
            assert feed is not None
            app = create_app(runtime=runtime, session_notices=feed)
            path = "/api/session/events"
            scope = {
                "type": "http", "asgi": {"version": "3.0", "spec_version": "2.0"}, "method": "GET",
                "scheme": "http", "path": path, "raw_path": path.encode(), "query_string": b"session_id=session&view=chat",
                "headers": [], "server": ("127.0.0.1", 8765), "client": ("127.0.0.1", 12345),
            }
            disconnected = asyncio.Event()
            frames: asyncio.Queue[tuple[str, dict[str, object]]] = asyncio.Queue()

            async def receive() -> dict[str, object]:
                await disconnected.wait()
                return {"type": "http.disconnect"}

            async def send(message: dict[str, object]) -> None:
                body = message.get("body")
                if isinstance(body, bytes) and b"data: " in body:
                    prefix, payload = body.decode().split("data: ", 1)
                    assert "tool-private-content" not in payload
                    frames.put_nowait((prefix.strip(), json.loads(payload)))

            async def next_frame() -> tuple[str, dict[str, object]]:
                return await asyncio.wait_for(frames.get(), 12)

            web_stream = asyncio.create_task(app(scope, receive, send))
            producer = None
            try:
                assert (await next_frame())[1]["active_execution_id"] is None
                producer = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "tests.ai.test_cli_turn_sync", str(tmp_path), mode,
                    stdin=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    cwd=Path(__file__).resolve().parents[2], env={**os.environ, "PYDANTIC_AI_NO_BANNER": "1"},
                )
                kind, admitted = await next_frame()
                execution_id = admitted["active_execution_id"]
                assert not kind and execution_id and admitted["local_execution"] is False

                async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765") as http:
                    active = (await http.get("/api/session?session_id=session")).json()["timeline"]["items"][0]
                    assert active["execution_id"] == execution_id
                    assert active["user_input"]["prompt"]["text"] == "user prompt from other process"

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
                    process = []
                    for _ in range(2):
                        kind, payload = await next_frame()
                        assert kind == "event: session_process", payload
                        assert payload["execution_id"] == execution_id
                        process.append(payload["item"])
                    assert [item["status"] for item in process] == ["RUNNING", "SUCCEEDED"]
                    assert {item["key"] for item in process} == {"tool:call"}
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
                    kind, released = await next_frame()
                    assert not kind and released["active_execution_id"] is None
                    turns = (await http.get("/api/session?session_id=session")).json()["timeline"]["items"]
                    saved, = turns
                    assert saved["conversation_committed"] and saved["output"] == {"text": "complete"}
                    assert saved["process"] == [process[-1]]
                    assert type(process[-1]["elapsed_ms"]) is int and process[-1]["elapsed_ms"] >= 0
                    assert all(item["argument_preview"] == "notice.txt" for item in process)
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
