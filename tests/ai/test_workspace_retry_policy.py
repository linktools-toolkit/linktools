#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace retry policies preserve rejection and external-effect boundaries."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from linktools.ai.capability import ToolCallRetry, workspace_capabilities
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import SandboxOperationRejected, Workspace


_FILE_REJECTIONS = frozenset({
    ErrorCode.REQUEST_FIELD_INVALID,
    ErrorCode.STORAGE_NOT_FOUND,
    ErrorCode.AUTHORIZATION_DENIED,
    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
})
_COMMAND_REJECTIONS = frozenset({
    ErrorCode.REQUEST_FIELD_INVALID,
    ErrorCode.AUTHORIZATION_DENIED,
    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
})
_COMMAND_ID_REJECTIONS = frozenset({
    ErrorCode.REQUEST_FIELD_INVALID,
    ErrorCode.STORAGE_NOT_FOUND,
    ErrorCode.AUTHORIZATION_DENIED,
})
_RETRY_MESSAGES = {
    ErrorCode.REQUEST_FIELD_INVALID: "invalid",
    ErrorCode.STORAGE_NOT_FOUND: "does not exist",
    ErrorCode.AUTHORIZATION_DENIED: "not allowed",
    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE: "too large",
    ErrorCode.STORAGE_CONFLICT: "current hash",
    ErrorCode.TOO_MANY_PENDING_OPERATIONS: "Too many workspace operations",
    ErrorCode.INTERNAL_ERROR: "",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (AIError, SandboxOperationRejected))
@pytest.mark.parametrize(
    ("name", "arguments", "rejected_codes", "effectful"),
    (
        ("attach_files", {"paths": ["note.txt"]}, _FILE_REJECTIONS, False),
        ("read_file", {"path": "note.txt"}, _FILE_REJECTIONS, False),
        ("write_file", {"path": "note.txt", "content": "new"}, _FILE_REJECTIONS, True),
        (
            "edit_file",
            {"path": "note.txt", "old_text": "old", "new_text": "new"},
            _FILE_REJECTIONS,
            True,
        ),
        ("list_directory", {"path": "."}, _FILE_REJECTIONS, False),
        ("search_files", {"pattern": "old"}, _FILE_REJECTIONS, False),
        ("find_files", {"pattern": "*.txt"}, _FILE_REJECTIONS, False),
        ("create_directory", {"path": "new"}, _FILE_REJECTIONS, True),
        ("file_info", {"path": "note.txt"}, _FILE_REJECTIONS, False),
        ("run_command", {"command": "pwd"}, _COMMAND_REJECTIONS, True),
        (
            "start_command",
            {"command": "pwd"},
            _COMMAND_REJECTIONS | {ErrorCode.TOO_MANY_PENDING_OPERATIONS},
            True,
        ),
        ("check_command", {"command_id": "command"}, _COMMAND_ID_REJECTIONS, False),
        ("stop_command", {"command_id": "command"}, _COMMAND_ID_REJECTIONS, True),
    ),
)
async def test_workspace_retry_requires_supported_code_and_known_effect(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    rejected_codes: frozenset[ErrorCode],
    effectful: bool,
    error_type: type[AIError],
) -> None:
    async def reject(*args: object, **kwargs: object) -> str:
        raise error

    method = "read_bytes" if name == "attach_files" else name
    session = SimpleNamespace(**{method: reject})
    capability = workspace_capabilities(
        Workspace.load(tmp_path),
        (name,),
        session=session,  # type: ignore[arg-type]
    )[0]
    tool = capability.get_toolset().tools[name]

    for code, message in _RETRY_MESSAGES.items():
        error = error_type(code)
        if code in rejected_codes and (
            not effectful or isinstance(error, SandboxOperationRejected)
        ):
            with pytest.raises(ToolCallRetry, match=message) as retry:
                await tool.function(**arguments)
            assert retry.value.__cause__ is error
        else:
            with pytest.raises(AIError) as failure:
                await tool.function(**arguments)
            assert failure.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (AIError, SandboxOperationRejected))
@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("write_file", {"path": "note.txt", "content": "new"}),
        ("edit_file", {"path": "note.txt", "old_text": "old", "new_text": "new"}),
    ),
)
async def test_workspace_conflict_retry_is_scoped_to_each_guarded_write(
    tmp_path: Path,
    name: str,
    arguments: dict[str, object],
    error_type: type[AIError],
) -> None:
    error = error_type(ErrorCode.STORAGE_CONFLICT)

    async def reject(*args: object, **kwargs: object) -> str:
        raise error

    session = SimpleNamespace(**{name: reject, "read_file": reject})
    capability = workspace_capabilities(
        Workspace.load(tmp_path),
        (name, "read_file"),
        session=session,  # type: ignore[arg-type]
    )[0]
    tools = capability.get_toolset().tools

    for expected_hash in ("", "current-hash", None):
        if expected_hash is not None and isinstance(error, SandboxOperationRejected):
            with pytest.raises(ToolCallRetry, match="current hash") as retry:
                await tools[name].function(**arguments, expected_hash=expected_hash)
            assert retry.value.__cause__ is error
        else:
            with pytest.raises(AIError) as failure:
                await tools[name].function(**arguments, expected_hash=expected_hash)
            assert failure.value is error

    with pytest.raises(AIError) as unguarded_read:
        await tools["read_file"].function(path="note.txt")
    assert unguarded_read.value is error
