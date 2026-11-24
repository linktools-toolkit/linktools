#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from pydantic_ai.messages import (
    SYNTHESIZED_TOOL_RETURN_METADATA_KEY,
    ModelRequest,
    ModelResponse,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
)

from linktools.ai.history._session_timeline import _project_turn_tool_process
from linktools.ai.errors import ErrorCode
from linktools.ai.session import ToolProcessItem, project_tool_process_event, tool_argument_preview


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
    assert _project_turn_tool_process(messages) == (
        ToolProcessItem("tool:a", "search", "SUCCEEDED", (1, 1)),
        ToolProcessItem("tool:b", "search", "FAILED", (1, 2), error_code=ErrorCode.TOOL_RETRY_REQUIRED),
        ToolProcessItem("tool:c", "notify", "UNCONFIRMED", (1, 3)),
        ToolProcessItem("tool:d", "archive", "INTERRUPTED", (1, 4)),
    )


def test_live_chat_process_uses_the_same_key_and_omits_diagnostics() -> None:
    started = project_tool_process_event(
        "TOOL_CALL_STARTED", {"call_id": "a", "tool_name": "search", "arguments": "private", "tool_position": [2, 0]},
    )
    finished = project_tool_process_event(
        "TOOL_CALL_FINISHED", {
            "call_id": "a", "tool_name": "search", "status": "FAILED", "result": "private",
            "error_code": "internal error", "tool_position": [2, 0],
        },
    )
    assert started == ToolProcessItem("tool:a", "search", "RUNNING", (2, 0))
    assert finished == ToolProcessItem("tool:a", "search", "FAILED", (2, 0))
    assert project_tool_process_event("ASSISTANT_THINKING_DELTA", {"text": "private"}) is None
    assert project_tool_process_event(
        "TOOL_CALL_FINISHED", {"call_id": "a", "tool_name": "search"},
    ) is None


@pytest.mark.parametrize(("name", "arguments", "expected"), [
    ("read_file", {"path": "docs/README.md"}, "docs/README.md"),
    ("search_files", {"path": "docs", "query": "secret"}, "docs"),
    ("read_file", {"path": ["README.md", "private"]}, None),
    ("read_file", {"path": {"token": "secret"}}, None),
    ("read_file", {"path": "../private"}, None),
    ("read_file", {"path": "https://site.example/token"}, None),
    ("read_file", {"path": "x" * 81}, None),
    ("run_command", {"path": "README.md", "command": "secret"}, None),
    ("write_file", {"path": "README.md", "content": "private body"}, "README.md"),
    ("load_skill", {"skill_id": "planning", "path": "private"}, "planning"),
    ("delegate_task", {"subagent_id": "reviewer", "task": "private prompt"}, "reviewer"),
    ("read_memory", {"file": "notes.md"}, "notes.md"),
    ("write_memory", {"file": "notes.md", "content": "private memory"}, "notes.md"),
    ("search_memory", {"query": "private memory"}, None),
])
def test_tool_title_preview_shows_only_simple_safe_fields(name: str, arguments: object, expected: str | None) -> None:
    assert tool_argument_preview(name, arguments) == expected


def test_terminal_tool_process_includes_known_reason_without_raw_diagnostic() -> None:
    item = project_tool_process_event("TOOL_CALL_FINISHED", {
        "call_id": "a", "tool_name": "read_file", "status": "FAILED", "tool_elapsed_ms": 0,
        "error_code": "STORAGE_NOT_FOUND", "error": "a raw internal diagnostic",
    })
    assert item == ToolProcessItem("tool:a", "read_file", "FAILED", elapsed_ms=0, error_code=ErrorCode.STORAGE_NOT_FOUND)
