#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution stream consumers must honor the public string event contract."""

from collections.abc import AsyncIterator
from types import SimpleNamespace

import pytest

from linktools.ai.acp import ACPAgent, _acp_update
from linktools.ai.core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionLineageKind,
    Principal,
)
from linktools.ai.errors import ErrorCode
from linktools.ai.runtime import ExecutionStreamEvent, ExecutionTreeEvent
from linktools.cli import CommandError
from linktools.commands.ai.run import _emit_result


def _tree_event(
    event_type: str,
    payload: object,
    *,
    sequence: "int | None" = None,
) -> ExecutionTreeEvent:
    return ExecutionTreeEvent(
        "execution",
        "agent",
        ExecutionLineageKind.RUN,
        None,
        "execution",
        None,
        0,
        ExecutionStreamEvent(
            "execution",
            sequence,
            event_type,
            payload,  # type: ignore[arg-type]
        ),
    )


class _StreamingExecution:
    execution_id = "execution"

    def __init__(self, events: tuple[ExecutionTreeEvent, ...]) -> None:
        self._events = events

    def watch(
        self,
        *,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        assert include_content is True

        async def values() -> AsyncIterator[ExecutionTreeEvent]:
            for event in self._events:
                yield event

        return values()


class _StreamingAgent:
    def __init__(self, events: tuple[ExecutionTreeEvent, ...]) -> None:
        self._events = events

    async def start(
        self,
        prompt: str,
        *,
        session_id: str,
        memory_scope: str,
        planning: bool,
        thinking: bool,
    ) -> _StreamingExecution:
        del prompt, session_id, memory_scope, planning, thinking
        return _StreamingExecution(self._events)


class _StreamingRuntime:
    def __init__(self, events: tuple[ExecutionTreeEvent, ...]) -> None:
        self._events = events

    @property
    def agents(self) -> "_StreamingRuntime":
        return self

    def get(self) -> _StreamingAgent:
        return _StreamingAgent(self._events)


@pytest.mark.asyncio
async def test_cli_stream_consumes_string_event_types(
    capsys: pytest.CaptureFixture[str],
) -> None:
    events = (
        _tree_event(
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value,
            {"text": "hello"},
        ),
        _tree_event(
            ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value,
            {"text": "reason"},
        ),
        _tree_event(
            ExecutionEventType.TOOL_CALL_STARTED.value,
            {"tool_name": "read_file"},
            sequence=1,
        ),
        _tree_event(
            ExecutionEventType.TOOL_CALL_FINISHED.value,
            {"tool_name": "read_file", "status": "SUCCEEDED"},
            sequence=2,
        ),
        _tree_event(
            ExecutionEventType.EXECUTION_SUCCEEDED.value,
            {},
            sequence=3,
        ),
    )

    result = await _emit_result(
        _StreamingRuntime(events),  # type: ignore[arg-type]
        "prompt",
        "session",
        "memory",
        False,
        False,
        False,
    )

    captured = capsys.readouterr()
    assert result == 0
    assert captured.out == "hello\n"
    assert "[thinking] reason" in captured.err
    assert "[tool] read_file" in captured.err
    assert "[tool] finished read_file" in captured.err


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "payload", "expected_status"),
    (
        (
            ExecutionEventType.EXECUTION_FAILED.value,
            {
                "error_code": "MODEL_API_ERROR",
                "safe_error_details": {"provider": "test"},
            },
            "FAILED",
        ),
        (
            ExecutionEventType.EXECUTION_CANCELLED.value,
            {
                "error_code": "EXECUTION_CANCELLED",
                "safe_error_details": {},
            },
            "CANCELLED",
        ),
        (
            ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value,
            {
                "error_code": "TOOL_EFFECT_UNKNOWN",
                "safe_error_details": {"operation_id": "tool-operation"},
            },
            "RECOVERY_REQUIRED",
        ),
    ),
)
async def test_cli_stream_reads_string_terminal_failure(
    event_type: str,
    payload: dict[str, object],
    expected_status: str,
) -> None:
    runtime = _StreamingRuntime((_tree_event(event_type, payload, sequence=1),))

    with pytest.raises(CommandError) as raised:
        await _emit_result(
            runtime,  # type: ignore[arg-type]
            "prompt",
            "session",
            "memory",
            False,
            False,
            False,
        )

    assert f"status={expected_status}" in str(raised.value)
    assert str(payload["error_code"]) in str(raised.value)
    assert str(payload["safe_error_details"]) in str(raised.value)


class _ACPSchema:
    @staticmethod
    def TextContentBlock(**kwargs: object) -> dict[str, object]:
        return {"schema_type": "content", **kwargs}

    @staticmethod
    def AgentMessageChunk(**kwargs: object) -> dict[str, object]:
        return {"schema_type": "message", **kwargs}

    @staticmethod
    def AgentThoughtChunk(**kwargs: object) -> dict[str, object]:
        return {"schema_type": "thought", **kwargs}

    @staticmethod
    def ToolCallStart(**kwargs: object) -> dict[str, object]:
        return {"schema_type": "tool-start", **kwargs}

    @staticmethod
    def ToolCallProgress(**kwargs: object) -> dict[str, object]:
        return {"schema_type": "tool-progress", **kwargs}

    @staticmethod
    def PromptResponse(**kwargs: object) -> dict[str, object]:
        return kwargs


@pytest.mark.parametrize(
    ("event_type", "payload", "expected_kind"),
    (
        (
            ExecutionDeltaType.ASSISTANT_TEXT_DELTA.value,
            {"text": "hello"},
            "message",
        ),
        (
            ExecutionDeltaType.ASSISTANT_THINKING_DELTA.value,
            {"text": "reason"},
            "thought",
        ),
        (
            ExecutionEventType.TOOL_CALL_STARTED.value,
            {"call_id": "call", "tool_name": "read_file"},
            "tool-start",
        ),
        (
            ExecutionEventType.TOOL_CALL_FINISHED.value,
            {"call_id": "call", "status": "SUCCEEDED"},
            "tool-progress",
        ),
    ),
)
def test_acp_maps_string_stream_event_types(
    event_type: str,
    payload: dict[str, object],
    expected_kind: str,
) -> None:
    update = _acp_update(
        _ACPSchema,  # type: ignore[arg-type]
        event_type,
        payload,  # type: ignore[arg-type]
    )

    assert isinstance(update, dict)
    assert update["schema_type"] == expected_kind


def test_acp_ignores_unknown_additive_stream_event() -> None:
    assert _acp_update(
        _ACPSchema,  # type: ignore[arg-type]
        "FUTURE_EXECUTION_EVENT",
        {},
    ) is None


class _ACPExecution:
    execution_id = "execution"

    def __init__(self, event_type: str, payload: object = None) -> None:
        self._event_type = event_type
        self._payload = {} if payload is None else payload
        self.stream_finished = False

    def watch(
        self,
        *,
        include_content: bool = False,
    ) -> AsyncIterator[ExecutionTreeEvent]:
        assert include_content is True

        async def values() -> AsyncIterator[ExecutionTreeEvent]:
            yield _tree_event(self._event_type, self._payload, sequence=1)
            self.stream_finished = True

        return values()


class _ACPAgentRuntime:
    def __init__(self, execution: _ACPExecution) -> None:
        self._execution = execution
        self.sessions = SimpleNamespace(get=self._get_session)
        self.agents = SimpleNamespace(get=self._get_agent)

    async def _get_session(self, session_id: str, *, principal: Principal) -> object:
        del session_id, principal
        return SimpleNamespace(agent_id="agent")

    def _get_agent(self, agent_id: str) -> object:
        assert agent_id == "agent"
        execution = self._execution

        class BoundAgent:
            async def start(
                self,
                prompt: str,
                *,
                principal: Principal,
                session_id: str,
                memory_scope: str,
            ) -> _ACPExecution:
                del prompt, principal, session_id, memory_scope
                return execution

        return BoundAgent()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "expected_stop_reason"),
    (
        (ExecutionEventType.EXECUTION_SUCCEEDED.value, "end_turn"),
        (ExecutionEventType.EXECUTION_CANCELLED.value, "cancelled"),
    ),
)
async def test_acp_prompt_uses_string_terminal_stop_reason(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    expected_stop_reason: str,
) -> None:
    agent = ACPAgent(
        _ACPAgentRuntime(_ACPExecution(event_type)),  # type: ignore[arg-type]
        principal=Principal("principal", "tenant", "service"),
        memory_scope="memory",
    )
    agent._initialized = True
    monkeypatch.setattr(
        "linktools.ai.acp._require_acp",
        lambda: (SimpleNamespace(), _ACPSchema),
    )

    response = await agent.prompt(
        "session",
        [SimpleNamespace(text="prompt")],  # type: ignore[list-item]
    )

    assert response["stopReason"] == expected_stop_reason


class _ACPRequestError(Exception):
    def __init__(self, code: int, message: str, data: object = None) -> None:
        super().__init__(message)
        self.code = code
        self.data = data

    @classmethod
    def internal_error(cls, data: object = None) -> "_ACPRequestError":
        return cls(-32603, "Internal error", data)

    @classmethod
    def invalid_request(cls, data: object = None) -> "_ACPRequestError":
        return cls(-32600, "Invalid request", data)

    @classmethod
    def invalid_params(cls, data: object = None) -> "_ACPRequestError":
        return cls(-32602, "Invalid params", data)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("event_type", "stored_code", "expected_code"),
    (
        (ExecutionEventType.EXECUTION_FAILED.value, "MODEL_TIMEOUT", ErrorCode.MODEL_TIMEOUT),
        (ExecutionEventType.EXECUTION_FAILED.value, None, ErrorCode.EXECUTION_FAILED),
        (ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value, "secret-invalid-code", ErrorCode.STORAGE_RECOVERY_REQUIRED),
    ),
)
async def test_acp_surfaces_sanitized_execution_failures(
    monkeypatch: pytest.MonkeyPatch,
    event_type: str,
    stored_code: str | None,
    expected_code: ErrorCode,
) -> None:
    execution = _ACPExecution(event_type, {
        "error_code": stored_code,
        "exception_message": "secret-exception",
        "safe_error_details": {"unexpected": "secret-details"},
    })
    agent = ACPAgent(
        _ACPAgentRuntime(execution),  # type: ignore[arg-type]
        principal=Principal("principal", "tenant", "service"),
        memory_scope="memory",
    )
    agent._initialized = True
    monkeypatch.setattr(
        "linktools.ai.acp._require_acp",
        lambda: (SimpleNamespace(RequestError=_ACPRequestError), _ACPSchema),
    )
    with pytest.raises(_ACPRequestError) as error:
        await agent.prompt("session", [SimpleNamespace(text="prompt")])  # type: ignore[list-item]
    assert error.value.code == -32603
    assert isinstance(error.value.data, dict)
    assert error.value.data["code"] == expected_code.value
    assert error.value.data["operation_id"] == "execution"
    assert error.value.data["safe_details"] == {"execution_id": "execution"}
    assert "secret" not in str(error.value.data)
    assert execution.stream_finished


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("operation", "expected_code", "reason"),
    (
        ("initialize", -32600, "no_common_protocol_version"),
        ("authenticate", -32602, "unknown_auth_method"),
        ("new_session", -32600, "initialize_required"),
    ),
)
@pytest.mark.parametrize("sdk", (False, True))
async def test_acp_protocol_rejections_use_supported_error_factories(
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    expected_code: int,
    reason: str,
    sdk: bool,
) -> None:
    if sdk:
        acp = pytest.importorskip("acp")
        schema = pytest.importorskip("acp.schema")
    else:
        acp = SimpleNamespace(RequestError=_ACPRequestError, PROTOCOL_VERSION=1)
        schema = _ACPSchema
    agent = ACPAgent(
        SimpleNamespace(),  # type: ignore[arg-type]
        principal=Principal("principal", "tenant", "service"),
        memory_scope="memory",
    )
    monkeypatch.setattr("linktools.ai.acp._require_acp", lambda: (acp, schema))
    with pytest.raises(acp.RequestError) as error:
        if operation == "initialize":
            await agent.initialize(acp.PROTOCOL_VERSION + 1)
        elif operation == "authenticate":
            await agent.authenticate("unsupported")
        else:
            await agent.new_session("/workspace")
    assert error.value.code == expected_code
    assert error.value.data == {"reason": reason}


@pytest.mark.asyncio
@pytest.mark.parametrize("event_type", (
    ExecutionEventType.EXECUTION_SUCCEEDED.value,
    ExecutionEventType.EXECUTION_CANCELLED.value,
    ExecutionEventType.EXECUTION_FAILED.value,
    ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value,
))
async def test_acp_optional_sdk_serializes_terminal_prompt_outcomes(
    monkeypatch: pytest.MonkeyPatch, event_type: str,
) -> None:
    acp = pytest.importorskip("acp")
    schema = pytest.importorskip("acp.schema")
    monkeypatch.setattr("linktools.ai.acp._require_acp", lambda: (acp, schema))
    agent = ACPAgent(
        _ACPAgentRuntime(_ACPExecution(event_type)),  # type: ignore[arg-type]
        principal=Principal("principal", "tenant", "service"),
        memory_scope="memory",
    )
    initialized = await agent.initialize(acp.PROTOCOL_VERSION)
    assert initialized.protocol_version == acp.PROTOCOL_VERSION
    if event_type in {
        ExecutionEventType.EXECUTION_FAILED.value,
        ExecutionEventType.EXECUTION_RECOVERY_REQUIRED.value,
    }:
        with pytest.raises(acp.RequestError) as error:
            await agent.prompt("session", [SimpleNamespace(text="prompt")])  # type: ignore[list-item]
        assert error.value.to_error_obj()["code"] == -32603
    else:
        response = await agent.prompt("session", [SimpleNamespace(text="prompt")])  # type: ignore[list-item]
        assert response.model_dump(by_alias=True)["stopReason"] == (
            "cancelled" if event_type == ExecutionEventType.EXECUTION_CANCELLED.value else "end_turn"
        )
