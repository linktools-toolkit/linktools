#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace retry policies preserve rejection and external-effect boundaries."""

from pathlib import Path
from ._runtime_test_helpers import FileToolsStub

import pytest
from linktools.ai.capability import ToolCallRetry, filesystem_capabilities, ShellCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import WorkspaceOperationRejected, Workspace


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
    ErrorCode.REQUEST_FIELD_INVALID: "invalid|valid workspace directory",
    ErrorCode.STORAGE_NOT_FOUND: "does not exist|unknown",
    ErrorCode.AUTHORIZATION_DENIED: "not allowed",
    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE: "too large",
    ErrorCode.STORAGE_CONFLICT: "current hash",
    ErrorCode.TOO_MANY_PENDING_OPERATIONS: "Too many",
    ErrorCode.INTERNAL_ERROR: "",
}


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (AIError, WorkspaceOperationRejected))
@pytest.mark.parametrize(
    ("name", "arguments", "rejected_codes", "effectful"),
    (
        ("read_file", {"path": ["note.txt"], "as_file": True}, _FILE_REJECTIONS | {ErrorCode.STORAGE_CONFLICT}, False),
        ("read_file", {"path": "note.txt"}, _FILE_REJECTIONS | {ErrorCode.STORAGE_CONFLICT}, False),
        ("write_file", {"path": "note.txt", "content": "new"}, _FILE_REJECTIONS, True),
        (
            "write_file",
            {"path": "note.txt", "old_text": "old", "content": "new"},
            _FILE_REJECTIONS,
            True,
        ),
        ("list_files", {"path": "."}, _FILE_REJECTIONS, False),
        ("search_files", {"pattern": "old"}, _FILE_REJECTIONS, False),
        ("list_files", {"include_glob": "*.txt"}, _FILE_REJECTIONS, False),
        ("read_file", {"path": "note.txt", "full_metadata": True}, _FILE_REJECTIONS | {ErrorCode.STORAGE_CONFLICT}, False),
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

    method = {"read_file": "inspect_file",
              "search_files": "search_text"}.get(name, name)
    session = FileToolsStub()
    setattr(session, method, reject)
    if name.endswith("command"):
        capability = ShellCapability().bind((name,), backend=session)
    else:
        capability = filesystem_capabilities(Workspace.load(tmp_path), (name,), backend=session)[0]
    tool = capability.get_toolset().tools[name]

    for code, message in _RETRY_MESSAGES.items():
        error = error_type(code)
        if code in rejected_codes and (
            not effectful or isinstance(error, WorkspaceOperationRejected)
        ):
            with pytest.raises(ToolCallRetry, match=message) as retry:
                await tool.function(**arguments)
            assert retry.value.__cause__ is error
        else:
            with pytest.raises(AIError) as failure:
                await tool.function(**arguments)
            assert failure.value is error


@pytest.mark.asyncio
@pytest.mark.parametrize("error_type", (AIError, WorkspaceOperationRejected))
@pytest.mark.parametrize(
    ("name", "arguments"),
    (
        ("write_file", {"path": "note.txt", "content": "new"}),
        ("write_file", {"path": "note.txt", "old_text": "old", "content": "new"}),
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

    session = FileToolsStub()
    setattr(session, name, reject)
    session.inspect_file = reject
    capability = filesystem_capabilities(
        Workspace.load(tmp_path),
        (name, "read_file"),
        backend=session,  # type: ignore[arg-type]
    )[0]
    tools = capability.get_toolset().tools

    for expected_hash in ("", "current-hash", None):
        if expected_hash is not None and isinstance(error, WorkspaceOperationRejected):
            with pytest.raises(ToolCallRetry, match="current hash") as retry:
                await tools[name].function(**arguments, expected_hash=expected_hash)
            assert retry.value.__cause__ is error
        else:
            with pytest.raises(AIError) as failure:
                await tools[name].function(**arguments, expected_hash=expected_hash)
            assert failure.value is error

    with pytest.raises(ToolCallRetry) as changed_read:
        await tools["read_file"].function(path="note.txt")
    assert changed_read.value.__cause__ is error
