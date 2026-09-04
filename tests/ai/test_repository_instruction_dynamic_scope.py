#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dynamic repository-instruction scope and stale-model-step contracts."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.exceptions import ApprovalRequired, ToolFailed
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart
from pydantic_ai.tools import ToolDefinition

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._agent_executor import _ToolPresentation
from linktools.ai.runtime._capabilities import (
    _WorkspaceToolGate,
    _repository_instruction_marker,
)
from linktools.ai.workspace import (
    RepositoryInstructionDocument,
    RepositoryInstructions,
    ToolPermissionRule,
    WorkspacePolicy,
    WorkspaceToolPermissionPolicy,
)


class _Resolver:
    def __init__(self, document: RepositoryInstructionDocument | None) -> None:
        self.document = document
        self.calls: list[tuple[str, frozenset[str]]] = []

    async def resolve(
        self,
        target: str,
        *,
        exclude_sources: frozenset[str] = frozenset(),
    ) -> RepositoryInstructions:
        self.calls.append((target, exclude_sources))
        if self.document is None or self.document.source in exclude_sources:
            return RepositoryInstructions(())
        return RepositoryInstructions((self.document,))


def _gate(
    root: Path,
    resolver: _Resolver,
    *,
    policy: WorkspacePolicy | None = None,
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
        policy=WorkspacePolicy() if policy is None else policy,
        trusted_tool_classes=(
            ("read_file", "filesystem.read"),
            ("write_file", "filesystem.write"),
            ("edit_file", "filesystem.write"),
            ("file_info", "filesystem.read"),
            ("create_directory", "filesystem.write"),
            ("list_directory", "filesystem.read"),
            ("search_files", "filesystem.read"),
            ("find_files", "filesystem.read"),
        ),
    )


def _ctx(*, approved: bool = False) -> SimpleNamespace:
    return SimpleNamespace(tool_call_approved=approved)


def _call(name: str = "read_file", *, call_id: str = "call-1") -> ToolCallPart:
    return ToolCallPart(tool_name=name, args={"path": "pkg/file.txt"}, tool_call_id=call_id)


@pytest.mark.asyncio
async def test_new_scope_exposes_once_then_fences_same_model_step(tmp_path: Path) -> None:
    document = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "nested-v1")
    resolver = _Resolver(document)
    gate = _gate(tmp_path, resolver)
    call = _call()
    tool = ToolDefinition(name="read_file")

    with pytest.raises(ToolFailed) as exposed:
        await gate.before_tool_execute(
            _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
        )
    marker = exposed.value.args[0]
    assert marker == _repository_instruction_marker(
        "execution", RepositoryInstructions((document,))
    )
    assert resolver.calls == [("pkg/file.txt", frozenset())]

    with pytest.raises(ToolFailed) as stale:
        await gate.before_tool_execute(
            _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
        )
    assert "next model step" in stale.value.args[0]
    assert len(resolver.calls) == 1

    await gate.before_model_request(_ctx(), None)  # type: ignore[arg-type]
    result = await gate.before_tool_execute(
        _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
    )
    assert result == {"path": "pkg/file.txt"}
    assert resolver.calls[-1] == (
        "pkg/file.txt",
        frozenset({"agents:pkg/AGENTS.md"}),
    )
    assert "nested-v1" in gate.get_instructions()(_ctx())  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_dynamic_refresh_barrier_precedes_ask_permission(tmp_path: Path) -> None:
    document = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "nested")
    resolver = _Resolver(document)
    policy = WorkspacePolicy(
        tool_permissions=WorkspaceToolPermissionPolicy(
            (ToolPermissionRule("ask", tool_name="read_file"),)
        )
    )
    gate = _gate(tmp_path, resolver, policy=policy)
    call = _call()
    tool = ToolDefinition(name="read_file")

    with pytest.raises(ToolFailed):
        await gate.before_tool_execute(
            _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
        )
    with pytest.raises(ToolFailed):
        await gate.before_tool_execute(
            _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
        )

    await gate.before_model_request(_ctx(), None)  # type: ignore[arg-type]
    with pytest.raises(ApprovalRequired):
        await gate.before_tool_execute(
            _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
        )


@pytest.mark.asyncio
async def test_same_source_mutation_preserves_first_exposure(tmp_path: Path) -> None:
    first = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "first")
    resolver = _Resolver(first)
    gate = _gate(tmp_path, resolver)
    call = _call()
    tool = ToolDefinition(name="read_file")

    with pytest.raises(ToolFailed):
        await gate.before_tool_execute(
            _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
        )
    resolver.document = RepositoryInstructionDocument(
        "agents:pkg/AGENTS.md", "pkg", "changed"
    )
    await gate.before_model_request(_ctx(), None)  # type: ignore[arg-type]
    assert await gate.before_tool_execute(
        _ctx(), call=call, tool_def=tool, args={"path": "pkg/file.txt"}
    ) == {"path": "pkg/file.txt"}
    rendered = gate.get_instructions()(_ctx())  # type: ignore[arg-type]
    assert "first" in rendered
    assert "changed" not in rendered


def test_marker_without_authority_is_ignored(tmp_path: Path) -> None:
    document = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "nested")
    marker = _repository_instruction_marker(
        "execution", RepositoryInstructions((document,))
    )
    call = _call()
    history = (
        ModelResponse(parts=[call], run_id="run-1"),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file",
                    content=marker,
                    tool_call_id="call-1",
                    outcome="failed",
                )
            ],
            run_id="run-1",
        ),
    )
    gate = _gate(tmp_path, _Resolver(None), history=history)
    assert "nested" not in gate.get_instructions()(_ctx())  # type: ignore[arg-type]


def test_malformed_authoritative_current_execution_marker_fails_closed(tmp_path: Path) -> None:
    document = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "nested")
    marker = _repository_instruction_marker(
        "execution", RepositoryInstructions((document,))
    ).replace("action=Apply", "action=Tampered", 1)
    call = _call()
    history = (
        ModelResponse(parts=[call], run_id="run-1"),
        ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="read_file",
                    content=marker,
                    tool_call_id="call-1",
                    outcome="failed",
                )
            ],
            run_id="run-1",
        ),
    )
    with pytest.raises(AIError) as error:
        _gate(
            tmp_path,
            _Resolver(None),
            history=history,
            authority=frozenset({("run-1", "call-1")}),
        )
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_dynamic_scope_limit_applies_to_total_active_bundle(tmp_path: Path) -> None:
    document = RepositoryInstructionDocument("agents:pkg/AGENTS.md", "pkg", "x" * 64)
    gate = _gate(
        tmp_path,
        _Resolver(document),
        policy=WorkspacePolicy(max_repository_instruction_bytes=32),
    )
    with pytest.raises(AIError) as error:
        await gate.before_tool_execute(
            _ctx(),
            call=_call(),
            tool_def=ToolDefinition(name="read_file"),
            args={"path": "pkg/file.txt"},
        )
    assert error.value.code is ErrorCode.PROMPT_TOO_LARGE


@pytest.mark.asyncio
async def test_instruction_aware_filesystem_tools_are_sequential() -> None:
    names = (
        "read_file",
        "write_file",
        "edit_file",
        "file_info",
        "create_directory",
        "list_directory",
        "search_files",
        "find_files",
    )
    classes = tuple(
        (
            name,
            "filesystem.write"
            if name in {"write_file", "edit_file", "create_directory"}
            else "filesystem.read",
        )
        for name in names
    )
    presentation = _ToolPresentation(
        ("*",),
        static_tool_names=names,
        mcp_policy=(),
        plan_mode=False,
        trusted_tool_classes=classes,
        trusted_mcp_selectors=(),
        instruction_aware=True,
    )
    tools = [
        ToolDefinition(name=name, capability_id="workspace-sandbox") for name in names
    ]
    prepared = await presentation._prepare_final_tools(None, tools)  # type: ignore[arg-type]
    assert [tool.name for tool in prepared] == list(names)
    assert all(tool.sequential is True for tool in prepared)
