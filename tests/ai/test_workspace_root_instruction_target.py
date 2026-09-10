#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Workspace-root path adaptation before repository-instruction checks."""

from typing import Any

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)


class _Session:
    async def canonicalize_path(self, path: str) -> str:
        if path == "":
            return "."
        if "\x00" in path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if path == "../outside":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if "\\" in path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return path


class _RepositoryBoundary:
    def __init__(self, error: AIError | None = None) -> None:
        self.error = error
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def render(self) -> str:
        return ""

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        path_fields: tuple[str, ...],
    ) -> None:
        del tool_call_id, path_fields
        self.calls.append((tool_name, arguments))
        if self.error is not None:
            raise self.error


async def _list_directory(path: str = ".") -> str:
    return path


def _toolset(repository: _RepositoryBoundary) -> RuntimeToolBoundaryToolset:
    return RuntimeToolBoundaryToolset(
        (FunctionToolset([_list_directory]),),
        {
            "_list_directory": ManagedToolDescriptor(
                effect_owner="intrinsic",
                effect="none",
                tool_class="filesystem.read",
                workspace_path_fields=("path",),
            )
        },
        id="workspace",
        sandbox_session=_Session(),  # type: ignore[arg-type]
        repository_boundary=repository,
    )


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_empty_path_uses_workspace_root_before_instruction_lookup() -> None:
    repository = _RepositoryBoundary()
    toolset = _toolset(repository)
    context = _context()
    tools = await toolset.get_tools(context)

    assert (
        await toolset.call_tool(
            "_list_directory",
            {"path": ""},
            context,
            tools["_list_directory"],
        )
        == "."
    )
    assert repository.calls == [("_list_directory", {"path": "."})]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "expected_code"),
    (
        ("\x00", ErrorCode.REQUEST_FIELD_INVALID),
        ("../outside", ErrorCode.AUTHORIZATION_DENIED),
        ("bad\\path", ErrorCode.REQUEST_FIELD_INVALID),
    ),
)
async def test_invalid_workspace_target_fails_before_instruction_lookup(
    path: str,
    expected_code: ErrorCode,
) -> None:
    repository = _RepositoryBoundary()
    toolset = _toolset(repository)
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(AIError) as raised:
        await toolset.call_tool(
            "_list_directory",
            {"path": path},
            context,
            tools["_list_directory"],
        )

    assert raised.value.code is expected_code
    assert repository.calls == []


@pytest.mark.asyncio
async def test_instruction_boundary_error_for_valid_root_target_is_fatal() -> None:
    error = AIError(ErrorCode.OUTPUT_CONTRACT_INVALID)
    repository = _RepositoryBoundary(error)
    toolset = _toolset(repository)
    context = _context()
    tools = await toolset.get_tools(context)

    with pytest.raises(AIError) as raised:
        await toolset.call_tool(
            "_list_directory",
            {"path": "pkg"},
            context,
            tools["_list_directory"],
        )

    assert raised.value is error
    assert repository.calls == [("_list_directory", {"path": "pkg"})]
