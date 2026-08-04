#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from types import SimpleNamespace

import pytest

from linktools.ai.acp.history_mapper import AcpHistoryMapper
from linktools.ai.execution.domain import MessageCaptureState, RunStatus


def _view(messages):
    return SimpleNamespace(
        run_id="run-1",
        status=RunStatus.COMPLETED,
        capture_state=MessageCaptureState.COMPLETE,
        messages=tuple(messages),
    )


def test_history_mapper_preserves_roles_order_and_tool_id() -> None:
    updates = AcpHistoryMapper().preflight(
        "session-1",
        (
            _view(
                (
                    {"kind": "request", "parts": [{"type": "user_prompt", "content": "hello"}]},
                    {"kind": "response", "parts": [{"type": "text", "content": "hi"}, {"type": "tool_call", "call_id": "tool-1", "tool_name": "echo", "arguments": {"x": 1}}]},
                    {"kind": "request", "parts": [{"type": "tool_result", "call_id": "tool-1", "tool_name": "echo", "result": "ok", "status": "success"}]},
                )
            ),
        ),
    )

    assert [update.session_update for update in updates] == [
        "user_message_chunk",
        "agent_message_chunk",
        "tool_call",
        "tool_call_update",
    ]
    assert updates[2].tool_call_id == "tool-1"
    assert updates[3].tool_call_id == "tool-1"


def test_history_mapper_rejects_incomplete_history_before_mapping() -> None:
    partial = _view(())
    partial.capture_state = MessageCaptureState.PARTIAL
    with pytest.raises(Exception) as raised:
        AcpHistoryMapper().preflight("session-1", (partial,))
    assert raised.value.data["reason"] == "incomplete_history"


def test_history_mapper_rejects_unknown_part_without_partial_result() -> None:
    with pytest.raises(Exception) as raised:
        AcpHistoryMapper().preflight(
            "session-1",
            (_view(({"kind": "response", "parts": [{"type": "future_part"}]},)),),
        )
    assert raised.value.data["reason"] == "unsupported_history_part"


def test_history_mapper_maps_user_content_resources() -> None:
    updates = AcpHistoryMapper().preflight(
        "session-1",
        (
            _view(
                (
                    {
                        "kind": "request",
                        "parts": [
                            {
                                "type": "user_prompt",
                                "content": [
                                    {"kind": "text", "content": "hello"},
                                    {"kind": "image-url", "url": "https://example.test/a.png"},
                                ],
                            }
                        ],
                    },
                )
            ),
        ),
    )

    assert updates[0].content.type == "text"
    assert updates[1].content.type == "resource_link"
    assert updates[1].content.uri == "https://example.test/a.png"
