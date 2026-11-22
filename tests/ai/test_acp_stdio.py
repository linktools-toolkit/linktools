#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Verify Web and ACP share session identity through real ACP stdio."""

import asyncio
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _open_local_runtime

from ._runtime_test_helpers import _Models, _UsageFunctionModel

if TYPE_CHECKING:
    from acp.schema import AgentMessageChunk, UserMessageChunk


@pytest.mark.skipif(sys.platform != "linux", reason="local Linux ACP stdio integration")
@pytest.mark.asyncio
@pytest.mark.parametrize("create_in_acp", [False, True], ids=["web-created", "acp-created"])
async def test_acp_stdio_lists_and_loads_sessions_with_effective_web_workspace(tmp_path: Path, create_in_acp: bool) -> None:
    acp = pytest.importorskip("acp")
    pytest.importorskip("starlette")
    from linktools.ai.web import create_app

    root = tmp_path / "workspace"
    root.mkdir()
    nested = root / "nested"
    nested.mkdir()
    outside_link = root / "outside-link"
    outside_link.symlink_to(tmp_path, target_is_directory=True)
    workspace = Workspace.initialize(root)
    async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(
        lambda _messages, _info: ModelResponse(parts=[TextPart("Saved answer")]),
    ))) as local:
        app = create_app(runtime=local.runtime)
        headers = {"Origin": "http://127.0.0.1:8765", "Sec-Fetch-Site": "same-origin", "X-Linktools-Console": "1"}
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8765", headers=headers,
        ) as web:
            for name, cwd in (("web-default", None), ("web-root", "."), ("web-nested", "nested")):
                payload = {"session_id": name, "request_id": name}
                if cwd is not None:
                    payload["cwd"] = cwd
                response = await web.post("/api/sessions", json=payload)
                assert response.status_code == 201, response.text
            response = await web.post("/api/session/messages?session_id=web-default", json={
                "request_id": "first-turn", "prompt": "Saved question",
            })
            assert response.status_code == 202, response.text
            execution = await local.runtime.executions.get(
                response.json()["execution_id"], principal=local.runtime.default_principal,
            )
            await execution.wait(timeout_seconds=10)

    class Client:
        def __init__(self) -> None:
            self.updates: list[tuple[str, str, str]] = []

        async def session_update(self, session_id: str, update: "AgentMessageChunk | UserMessageChunk") -> None:
            self.updates.append((session_id, update.session_update, update.content.text))

    source = Path(__file__).resolve().parents[2]
    env = {**os.environ, "LINKTOOLS_PATH": str(tmp_path / "home"), "PYDANTIC_AI_NO_BANNER": "1",
           "PYTHONPATH": os.pathsep.join((str(source / "linktools-ai/src"), str(source / "linktools/src"),
                                        os.environ.get("PYTHONPATH", "")))}

    recorder = Client()
    async with asyncio.timeout(25):
        async with acp.spawn_agent_process(
            recorder, sys.executable, "-m", "linktools", "ai", "acp", "--project", str(root),
            "--model", "stdio-test", "--base-url", "http://127.0.0.1:1", "--api-key", "offline-test-key",
            env=env, cwd=tmp_path, use_unstable_protocol=True,
        ) as (client, _process):
            initialized = await client.initialize(acp.PROTOCOL_VERSION)
            assert initialized.agent_capabilities.load_session
            expected = {"web-default": str(root), "web-root": str(root), "web-nested": str(nested)}
            if create_in_acp:
                created = await client.new_session(cwd=str(nested), mcp_servers=[])
                expected[created.session_id] = str(nested)
                for outside in (tmp_path, outside_link):
                    with pytest.raises(acp.RequestError) as invalid_directory:
                        await client.new_session(cwd=str(outside), mcp_servers=[])
                    assert invalid_directory.value.code == -32602
            all_sessions = await client.list_sessions()
            selected = {cwd: (await client.list_sessions(cwd=str(cwd))).sessions for cwd in (root, nested)}
            assert ({item.session_id: item.cwd for item in all_sessions.sessions}, {
                str(cwd): {item.session_id: item.cwd for item in sessions} for cwd, sessions in selected.items()
            }) == (expected, {
                str(cwd): {session_id: path for session_id, path in expected.items() if path == str(cwd)}
                for cwd in (root, nested)
            })
            assert not (await client.list_sessions(cwd=str(tmp_path))).sessions
            with pytest.raises(acp.RequestError) as wrong_directory:
                await client.load_session(cwd=str(nested), session_id="web-default", mcp_servers=[])
            assert wrong_directory.value.code == -32602
            assert recorder.updates == []

            for sessions in selected.values():
                for item in sessions:
                    await client.load_session(cwd=item.cwd, session_id=item.session_id, mcp_servers=[])
            assert recorder.updates == [
                ("web-default", "user_message_chunk", "Saved question"),
                ("web-default", "agent_message_chunk", "Saved answer"),
            ]
