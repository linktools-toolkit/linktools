"""Structured native turn output remains visible across the Web replay boundary."""

import asyncio

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import ModelResponse, ToolCallPart

from linktools.ai.core import TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
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

    class DifferentOutput(BaseModel):
        title: str

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

        duplicate = await http.post(
            "/api/session/messages?session_id=session", headers=_HEADERS,
            json={"request_id": "run", "agent_id": "default", "prompt": "count"},
        )
        assert duplicate.status_code == 202 and duplicate.json() == {"turn_id": turn_id}
        with pytest.raises(AIError) as conflict:
            await runtime.sessions.start_turn(
                "session", "count", agent="default", principal=runtime.default_principal,
                idempotency_key="run", output=DifferentOutput,
            )
        assert conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        assert output_requests == [(1, True)]

    async with Persistence.open("structured-web", storage=storage) as persistence, Runtime.open(
        persistence=persistence, features=features, authorization=Policy(),
    ) as runtime, client(create_app(runtime=runtime, local_workspace=True)) as http:
        replay = await http.get("/api/session", params={"session_id": "session", "turn_id": turn_id})
        assert replay.status_code == 200, replay.text
        assert replay.json()["items"] == terminal["items"]
        assert "output" not in replay.json() and CANONICAL_OUTPUT_TOOL_NAME not in replay.text


@pytest.mark.asyncio
@pytest.mark.parametrize("opted_in", (False, True))
async def test_native_structured_preview_is_live_selected_and_replaced_on_validation_retry(tmp_path, opted_in):
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel

    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.web._app import _native_turn_event_wire

    class Policy(TenantAuthorizationPolicy):
        pass

    class Answer(BaseModel):
        message: str
        internal: str

    requests = 0
    finish_first_attempt = asyncio.Event()

    async def respond(_messages, info):
        nonlocal requests
        requests += 1
        [tool] = info.output_tools
        if requests == 1:
            yield {0: DeltaToolCall(name=tool.name, json_args='{"message":"first', tool_call_id="wrong")}
            yield {0: DeltaToolCall(json_args=' draft')}
            if opted_in:
                # Do not complete the JSON before the selected preview is observed.
                await finish_first_attempt.wait()
            yield {0: DeltaToolCall(json_args='"}')}
        else:
            yield {0: DeltaToolCall(name=tool.name, json_args='{"internal":"no live leakage","message":"final',
                                    tool_call_id="valid")}
            yield {0: DeltaToolCall(json_args=' answer"}')}

    features = _persistence_capabilities()
    features.capability(ExecutionCapability(output_preview_field="message" if opted_in else None))
    features.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    model_features(_Models(FunctionModel(stream_function=respond)), features)
    storage = PersistenceStorage.sqlite(tmp_path / "structured-preview.db")
    expected = {"message": "final answer", "internal": "no live leakage"}

    async with Runtime.open("structured-preview", storage=storage, features=features, authorization=Policy()) as runtime:
        actor = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=actor)
        frames = []
        with TurnObserver(runtime, "session", actor, local_workspace=False) as observer:
            turn = await runtime.sessions.start("session", "answer", agent="default", principal=actor,
                                                idempotency_key="run", output=Answer)
            try:
                async with asyncio.timeout(10):
                    async for frame in observer.events(turn.key.turn_id, append_only=False):
                        frames.append(frame)
                        if frame.get("type") == "output_preview" and frame.get("text") == "first draft":
                            assert requests == 1
                            assert (await turn.result()).view["conversation_committed"] is False
                            finish_first_attempt.set()
            finally:
                finish_first_attempt.set()
        terminal = frames[-1]
        assert terminal["type"] == "terminal" and terminal["state"] == "succeeded"
        assert terminal["output"] == expected and requests == 2
        previews = [(i, frame) for i, frame in enumerate(frames) if frame["type"] == "output_preview"]
        if not opted_in:
            assert previews == []
        else:
            [(first, old), (second, new)] = [(i, value) for i, value in previews
                                            if value["text"] in {"first draft", "final answer"}]
            assert all(value["field"] == "message" and "no live leakage" not in str(value)
                       for _, value in previews)
            assert old["model_key"] != new["model_key"]
            assert any(value["type"] == "preview_reset" and (value["model_key"], value["attempt"])
                       == (old["model_key"], old["attempt"]) for value in frames[first + 1:second])
            assert _native_turn_event_wire({**new, "raw_args": "secret"}) == {
                key: new[key] for key in ("type", "turn_id", "generation", "model_key", "attempt",
                                          "field", "preview_seq", "text")
            }
        replay = [value async for value in turn.observe()]
        assert len(replay) == 1 and replay[0]["type"] == "terminal" and replay[0]["output"] == expected
