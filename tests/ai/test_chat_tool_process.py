#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import asdict

from pydantic_ai.messages import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

from linktools.ai.session import project_tool_process_event, project_turn_tool_process


def test_committed_chat_process_pairs_explicit_results_without_payloads() -> None:
    messages = (
        ModelResponse(parts=[
            ThinkingPart(content="private thinking", signature="opaque"),
            ToolCallPart(tool_name="search", args={"secret": "arguments"}, tool_call_id="a"),
            ToolCallPart(tool_name="search", args={}, tool_call_id="b"),
            ToolCallPart(tool_name="notify", args={}, tool_call_id="c"),
            ToolCallPart(tool_name="archive", args={}, tool_call_id="d"),
        ]),
        ModelRequest(parts=[
            ToolReturnPart(tool_name="archive", content="real interruption", tool_call_id="d", outcome="interrupted"),
            ToolReturnPart(tool_name="search", content={"secret": "result"}, tool_call_id="a"),
            ToolReturnPart(tool_name="search", content="internal diagnostic", tool_call_id="b", outcome="retried"),
            ToolReturnPart(
                tool_name="notify", content="synthetic interrupted", tool_call_id="c",
                outcome="interrupted", metadata={SYNTHESIZED_TOOL_RETURN_METADATA_KEY: True},
            ),
            ToolReturnPart(tool_name="other", content="unmatched", tool_call_id="unmatched"),
        ]),
    )
    assert [asdict(item) for item in project_turn_tool_process(messages)] == [
        {"kind": "tool", "key": "tool:a", "tool_name": "search", "status": "SUCCEEDED"},
        {"kind": "tool", "key": "tool:b", "tool_name": "search", "status": "FAILED"},
        {"kind": "tool", "key": "tool:c", "tool_name": "notify", "status": "UNCONFIRMED"},
        {"kind": "tool", "key": "tool:d", "tool_name": "archive", "status": "INTERRUPTED"},
    ]


def test_live_chat_process_uses_the_same_key_and_omits_diagnostics() -> None:
    started = project_tool_process_event(
        "TOOL_CALL_STARTED", {"call_id": "a", "tool_name": "search", "arguments": "private"},
    )
    finished = project_tool_process_event(
        "TOOL_CALL_FINISHED", {
            "call_id": "a", "tool_name": "search", "status": "FAILED", "result": "private",
            "error_code": "internal error",
        },
    )
    assert started is not None and asdict(started) == {
        "kind": "tool", "key": "tool:a", "tool_name": "search", "status": "RUNNING",
    }
    assert finished is not None and asdict(finished) == {
        "kind": "tool", "key": "tool:a", "tool_name": "search", "status": "FAILED",
    }
    assert project_tool_process_event("ASSISTANT_THINKING_DELTA", {"text": "private"}) is None
    assert project_tool_process_event(
        "TOOL_CALL_FINISHED", {"call_id": "a", "tool_name": "search"},
    ) is None
