#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution stream consumers must honor the public string event contract."""

from collections.abc import AsyncIterator

import pytest

from linktools.ai.acp import _acp_update
from linktools.ai.core import (
    ExecutionDeltaType,
    ExecutionEventType,
    ExecutionLineageKind,
)
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

    def watch(self) -> AsyncIterator[ExecutionTreeEvent]:
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

    def agent(self) -> _StreamingAgent:
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


class _ACPSchema:
    @staticmethod
    def TextContentBlock(**kwargs: object) -> dict[str, object]:
        return {"kind": "content", **kwargs}

    @staticmethod
    def AgentMessageChunk(**kwargs: object) -> dict[str, object]:
        return {"kind": "message", **kwargs}

    @staticmethod
    def AgentThoughtChunk(**kwargs: object) -> dict[str, object]:
        return {"kind": "thought", **kwargs}

    @staticmethod
    def ToolCallStart(**kwargs: object) -> dict[str, object]:
        return {"kind": "tool-start", **kwargs}

    @staticmethod
    def ToolCallProgress(**kwargs: object) -> dict[str, object]:
        return {"kind": "tool-progress", **kwargs}


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
    assert update["kind"] == expected_kind


def test_acp_ignores_unknown_additive_stream_event() -> None:
    assert _acp_update(
        _ACPSchema,  # type: ignore[arg-type]
        "FUTURE_EXECUTION_EVENT",
        {},
    ) is None
