#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace root path adaptation for repository instructions."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ToolFailed
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import ToolDefinition

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._capabilities import (
    _WorkspaceToolGate,
    _repository_instruction_marker,
)
from linktools.ai.workspace import (
    RepositoryInstructionDocument,
    RepositoryInstructions,
    WorkspacePolicy,
)

_TOOL_CLASSES = (
    ("read_file", "filesystem.read"),
    ("write_file", "filesystem.write"),
    ("edit_file", "filesystem.write"),
    ("file_info", "filesystem.read"),
    ("create_directory", "filesystem.write"),
    ("list_directory", "filesystem.read"),
    ("search_files", "filesystem.read"),
    ("find_files", "filesystem.read"),
)


class _Resolver:
    def __init__(self, error: AIError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str | Path, frozenset[str]]] = []

    async def resolve(
        self,
        path: str | Path = ".",
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        self.calls.append((path, exclude_sources))
        if self.error is not None:
            raise self.error
        return RepositoryInstructions(())


def _gate(
    root: Path,
    resolver: _Resolver,
    *,
    trusted_tool_classes: tuple[tuple[str, str], ...],
    history: tuple[ModelRequest | ModelResponse, ...] = (),
    authority: frozenset[tuple[str, str]] = frozenset(),
) -> _WorkspaceToolGate:
    return _WorkspaceToolGate(
        execution_id="execution",
        workspace_root=root,
        repository_instruction_history=history,
        repository_instruction_marker_authority=authority,
        repository_instructions=RepositoryInstructions(()),
        instruction_resolver=resolver,
        policy=WorkspacePolicy(),
        trusted_tool_classes=trusted_tool_classes,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(("tool_name", "tool_class"), _TOOL_CLASSES)
async def test_empty_path_uses_root_for_instruction_lookup_without_rewriting_args(
    tmp_path: Path,
    tool_name: str,
    tool_class: str,
) -> None:
    resolver = _Resolver()
    gate = _gate(
        tmp_path,
        resolver,
        trusted_tool_classes=((tool_name, tool_class),),
    )
    args = {"path": ""}

    result = await gate.before_tool_execute(
        SimpleNamespace(tool_call_approved=False),  # type: ignore[arg-type]
        call=ToolCallPart(tool_name=tool_name, args=args, tool_call_id="call-1"),
        tool_def=ToolDefinition(name=tool_name),
        args=args,
    )

    assert resolver.calls == [(".", frozenset())]
    assert result is args
    assert args == {"path": ""}


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ("\x00", "../outside", "bad\\path", "bad|path"))
async def test_invalid_instruction_target_is_returned_to_model_without_resolver_call(
    tmp_path: Path,
    path: str,
) -> None:
    resolver = _Resolver()
    gate = _gate(
        tmp_path,
        resolver,
        trusted_tool_classes=(("list_directory", "filesystem.read"),),
    )
    args = {"path": path}

    with pytest.raises(ToolFailed) as raised:
        await gate.before_tool_execute(
            SimpleNamespace(tool_call_approved=False),  # type: ignore[arg-type]
            call=ToolCallPart(
                tool_name="list_directory",
                args=args,
                tool_call_id="call-1",
            ),
            tool_def=ToolDefinition(name="list_directory"),
            args=args,
        )

    assert raised.value.message == (
        "TOOL_EXECUTION_FAILED: workspace path is invalid or outside the workspace; "
        "use a path within the workspace root"
    )
    assert resolver.calls == []
    assert args == {"path": path}


@pytest.mark.asyncio
async def test_resolver_contract_error_for_valid_target_remains_fatal(tmp_path: Path) -> None:
    error = AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    resolver = _Resolver(error)
    gate = _gate(
        tmp_path,
        resolver,
        trusted_tool_classes=(("list_directory", "filesystem.read"),),
    )

    with pytest.raises(AIError) as raised:
        await gate.before_tool_execute(
            SimpleNamespace(tool_call_approved=False),  # type: ignore[arg-type]
            call=ToolCallPart(
                tool_name="list_directory",
                args={"path": "pkg"},
                tool_call_id="call-1",
            ),
            tool_def=ToolDefinition(name="list_directory"),
            args={"path": "pkg"},
        )

    assert raised.value is error
    assert resolver.calls == [("pkg", frozenset())]


def test_empty_path_root_marker_restores_as_valid_scope(tmp_path: Path) -> None:
    document = RepositoryInstructionDocument("agents:AGENTS.md", ".", "root")
    marker = _repository_instruction_marker(
        "execution",
        RepositoryInstructions((document,)),
    )
    call = ToolCallPart(
        tool_name="list_directory",
        args={"path": ""},
        tool_call_id="call-1",
    )
    history = (
        ModelResponse(parts=[call], run_id="run-1"),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="list_directory",
                    content=marker,
                    tool_call_id="call-1",
                    outcome="failed",
                )
            ],
            run_id="run-1",
        ),
    )

    gate = _gate(
        tmp_path,
        _Resolver(),
        trusted_tool_classes=(("list_directory", "filesystem.read"),),
        history=history,
        authority=frozenset({("run-1", "call-1")}),
    )

    rendered = gate.get_instructions()(SimpleNamespace())  # type: ignore[arg-type]
    assert "root" in rendered
