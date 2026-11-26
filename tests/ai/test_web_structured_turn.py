"""Structured native turn output remains visible across the Web replay boundary."""

import asyncio

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart

from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.model.messages import CANONICAL_OUTPUT_TOOL_NAME
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime
from linktools.ai.runtime.turn_clients import TurnObserver
from linktools.ai.web import create_app

from ._runtime_test_helpers import _Models, _UsageFunctionModel, _persistence_capabilities, model_features
from .test_web_console import _HEADERS, client


@pytest.mark.asyncio
async def test_web_structured_turn_replays_the_validated_output_item(tmp_path, caplog):
    class Policy(TenantAuthorizationPolicy):
        pass

    expected = {"count": 3}
    output_requests = []

    def respond(_messages, info):
        output_requests.append((len(info.output_tools), any(
            tool.name == CANONICAL_OUTPUT_TOOL_NAME for tool in info.output_tools
        )))
        output_tool, = info.output_tools
        assert output_tool.name == CANONICAL_OUTPUT_TOOL_NAME
        return ModelResponse(parts=[ToolCallPart(output_tool.name, expected, tool_call_id="output")])

    features = _persistence_capabilities()
    features.agent(
        "default", output_schema={
            "type": "object", "properties": {"count": {"type": "integer"}}, "required": ["count"],
        }, allow_tools=(), allow_skills=(), allow_subagents=(),
    )
    model_features(_Models(_UsageFunctionModel(respond)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "structured-web.db")

    async with Persistence.open("structured-web", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=Policy(),
    ) as runtime, client(create_app(runtime=runtime, local_workspace=True)) as http:
        created = await http.post("/api/sessions", headers=_HEADERS,
                                  json={"session_id": "session", "request_id": "create"})
        assert created.status_code == 201, created.text
        with TurnObserver(runtime, "session", runtime.default_principal, local_workspace=True) as observer:
            accepted = await http.post(
                "/api/session/messages?session_id=session", headers=_HEADERS,
                json={"request_id": "run", "agent_id": "default", "prompt": "count"},
            )
            assert accepted.status_code == 202, accepted.text
            turn_id = accepted.json()["turn_id"]
            async with asyncio.timeout(10):
                async for terminal in observer.events(turn_id):
                    if terminal["type"] == "terminal":
                        break
                else:
                    pytest.fail("Structured turn did not publish a terminal result")
        if terminal["state"] != "succeeded":
            source = next((record.getMessage() for record in reversed(caplog.records)
                           if record.name == "linktools.ai.runtime._regenerable_turn"
                           and record.msg.startswith("Canonical turn failed:")
                           and record.args and record.args[0] == turn_id), None)
            pytest.fail(f"Structured turn failed: state={terminal['state']}, code={terminal.get('error_code')}, "
                        f"output declarations (count, expected present)={output_requests}, source={source}")
        assert terminal["output"] == expected  # SDK contract remains unchanged.
        item, = terminal["items"]
        assert item["item_kind"] == "structured_output" and item["content"] == expected
        assert terminal["process"] == []

    async with Persistence.open("structured-web", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=Policy(),
    ) as runtime, client(create_app(runtime=runtime, local_workspace=True)) as http:
        replay = await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})
        assert replay.status_code == 200, replay.text
        assert replay.json()["items"] == terminal["items"]
        assert "output" not in replay.json() and CANONICAL_OUTPUT_TOOL_NAME not in replay.text
