#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Default local SQL Web sessions exercise real Step-backed workspace tools."""

import asyncio

import pytest

pytest.importorskip("starlette")
pytest.importorskip("uvicorn")

from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart

from linktools.ai.runtime.turn_clients import TurnObserver
from linktools.ai.web import create_app
from linktools.ai.workspace import Workspace
from linktools.commands.ai._common import _open_local_runtime

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_web_console import _HEADERS, _turn_text, client


@pytest.mark.asyncio
async def test_default_local_web_memory_survives_runtime_and_session_reopen(tmp_path, caplog):
    workspace = Workspace.initialize(tmp_path)
    remembered = "stored by the first runtime"
    seen = []

    async def send(workspace, session_id, tool, args, answer):
        call_id = "call-" + tool

        async def model(messages, _info):
            result = next((part for part in messages[-1].parts
                           if isinstance(part, ToolReturnPart) and part.tool_call_id == call_id), None)
            if result is None:
                return ModelResponse(parts=[ToolCallPart(tool, args, tool_call_id=call_id)])
            assert result.outcome == "success"
            seen.append((tool, result.content))
            return ModelResponse(parts=[TextPart(answer)])

        async with _open_local_runtime(workspace, model=_Models(_UsageFunctionModel(model))) as local:
            runtime = local.runtime
            principal = runtime.default_principal
            async with client(create_app(runtime=runtime, local_workspace=True)) as http:
                created = await http.post("/api/sessions", headers=_HEADERS,
                    json={"session_id": session_id, "request_id": "create-" + session_id})
                assert created.status_code == 201, created.text

                with TurnObserver(runtime, session_id, principal, local_workspace=True) as observer:
                    accepted = await http.post("/api/session/messages", params={"session_id": session_id},
                        headers=_HEADERS, json={"prompt": "Use the memory tool", "request_id": "send-" + session_id})
                    assert accepted.status_code == 202 and set(accepted.json()) == {"turn_id"}, accepted.text
                    turn_id = accepted.json()["turn_id"]
                    async with asyncio.timeout(15):
                        async for frame in observer.events(turn_id):
                            if frame["type"] == "terminal":
                                if frame["state"] != "succeeded":
                                    source = next((record.getMessage() for record in reversed(caplog.records)
                                                   if record.name == "linktools.ai.runtime._regenerable_turn"
                                                   and record.msg.startswith("Canonical turn failed:")
                                                   and record.args and record.args[0] == turn_id), None)
                                    pytest.fail(f"Memory turn failed: state={frame['state']}, "
                                                f"code={frame.get('error_code')}, source={source}")
                                break
                        else:
                            pytest.fail("The turn did not publish a terminal frame")

                response = await http.get("/api/session", params={"session_id": session_id, "turn_id": turn_id})
                assert response.status_code == 200, response.text
                turn = response.json()
                assert turn["conversation_committed"] and _turn_text(turn) == answer
                process, = turn["process"]
                assert process["key"] == "tool:" + call_id and process["status"] == "SUCCEEDED"
                assert type(process["elapsed_ms"]) is int and process["elapsed_ms"] >= 0
                assert remembered not in response.text

    await send(workspace, "writer", "write_memory", {"file": "facts.md", "content": remembered}, "written")
    # A fresh local Runtime and separate session can know the value only from
    # persisted memory, not the previous turn's transcript or process state.
    await send(Workspace.discover(tmp_path), "reader", "read_memory", {"file": "facts.md"}, "read")
    assert [tool for tool, _ in seen] == ["write_memory", "read_memory"]
    assert remembered in str(seen[1][1])
