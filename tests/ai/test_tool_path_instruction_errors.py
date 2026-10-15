#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool targets and instruction metadata have separate error contracts."""

from contextlib import nullcontext
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.workspace import LocalSandbox, ReadOnlySandboxPolicy, Workspace, WorkspaceInstructionResolver, WorkspacePolicy

from ._runtime_test_helpers import _Models, _UsageFunctionModel


@pytest.mark.asyncio
@pytest.mark.parametrize("target", ('report[1].txt', '["a.yaml", "b.yaml"]', 'report|draft.txt', 'report\nnext.txt'))
async def test_rule_target_is_not_instruction_header_metadata(tmp_path: Path, target: str) -> None:
    rules = RepositoryInstructions((RepositoryInstructionDocument("rule:root", ".", "root guidance"),))
    assert rules.for_path(target) == rules
    resolver = WorkspaceInstructionResolver(tmp_path, WorkspacePolicy(), rules)
    assert await resolver.resolve(target) == rules
    with pytest.raises(AIError) as caught:
        RepositoryInstructionDocument("rule:scoped", target, "invalid metadata")
    assert caught.value.code is ErrorCode.OUTPUT_CONTRACT_INVALID


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_target", ("missing.txt", "../outside.txt", ".env"))
async def test_runtime_path_retry_rechecks_allowed_target(tmp_path: Path, bad_target: str) -> None:
    (tmp_path / "allowed.txt").write_text("allowed content", encoding="utf-8")
    (tmp_path / ".env").write_text("must not be read", encoding="utf-8")
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": bad_target}, "bad")])
        if requests == 2:
            retries = [part for message in messages for part in message.parts if isinstance(part, RetryPromptPart)]
            assert len(retries) == 1 and retries[0].tool_call_id == "bad"
            assert "must not be read" not in str(messages)
            return ModelResponse(parts=[ToolCallPart("read_file", {"path": "allowed.txt"}, "corrected")])
        assert "allowed content" in str(messages)
        return ModelResponse(parts=[TextPart("finished")])

    group = CapabilityGroup("paths")
    group.agent("reader", model="default", allow_tools=("read_file",))
    async with Runtime.open(
        "paths", models=_Models(_UsageFunctionModel(model)), storage=RuntimeStorage.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=Workspace.load(tmp_path),
            sandbox=LocalSandbox(read_policy=ReadOnlySandboxPolicy(("allowed.txt", "missing.txt")))), group),
    ) as runtime:
        execution = await runtime.agents.get("reader").start("read")
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is ExecutionStatus.SUCCEEDED
        assert requests == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("malformed_document", "integrity", "infrastructure"))
async def test_runtime_instruction_failure_is_not_model_retry(tmp_path: Path, monkeypatch, failure: str) -> None:
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "file.txt").write_text("unread content", encoding="utf-8")
    expected = {
        "malformed_document": ErrorCode.OUTPUT_CONTRACT_INVALID,
        "integrity": ErrorCode.STORAGE_INTEGRITY_ERROR,
        "infrastructure": ErrorCode.STORAGE_UNAVAILABLE,
    }[failure]
    original = WorkspaceInstructionResolver.resolve

    async def resolve(self, path=".", **kwargs):
        if str(path) == "nested/file.txt" and failure != "malformed_document":
            raise AIError(expected)
        return await original(self, path, **kwargs)

    monkeypatch.setattr(WorkspaceInstructionResolver, "resolve", resolve)
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if failure == "malformed_document":
            (nested / "AGENTS.md").write_bytes(b"\xff")
        return ModelResponse(parts=[ToolCallPart("read_file", {"path": "nested/file.txt"}, "broken")])

    group = CapabilityGroup("paths")
    group.agent("reader", model="default", allow_tools=("read_file",))
    with nullcontext() if failure == "malformed_document" else pytest.raises(AIError, match=expected.value):
        async with Runtime.open(
            "paths", models=_Models(_UsageFunctionModel(model)), storage=RuntimeStorage.in_memory(),
            capabilities=(CapabilityGroup("workspace", workspace=Workspace.load(tmp_path)), group),
        ) as runtime:
            execution = await runtime.agents.get("reader").start("read")
            if failure == "malformed_document":
                result = (await execution.wait(timeout_seconds=10)).result
                assert result.status is ExecutionStatus.FAILED
                assert result.error_code == expected.value
                trace = (await execution.trace(tool_call_id="broken")).items
                errors = [item for item in trace if item.payload["kind"] == "TOOL_ERROR"]
                assert len(errors) == 1 and errors[0].payload["error_code"] == expected.value
            else:
                with pytest.raises(AIError) as caught:
                    await execution.wait(timeout_seconds=10)
                assert caught.value.code is expected
            assert requests == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name,args", (
    ("read_file", {"path": "data[1]/file.txt"}),
    ("file_info", {"path": "data[1]/file.txt"}),
    ("list_directory", {"path": "data[1]"}),
    ("list_files", {"path": "data[1]"}),
    ("find_files", {"path": "data[1]", "pattern": "*.txt"}),
    ("search_files", {"path": "data[1]", "pattern": "original"}),
    ("write_file", {"path": "data[1]/file.txt", "content": "changed"}),
    ("edit_file", {"path": "data[1]/file.txt", "old_text": "original", "new_text": "changed"}),
))
async def test_shared_rule_boundary_allows_literal_paths_for_workspace_tools(tmp_path: Path, name: str, args: dict) -> None:
    directory = tmp_path / "data[1]"
    directory.mkdir()
    target = directory / "file.txt"
    target.write_text("original", encoding="utf-8")
    requests = 0

    async def model(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            return ModelResponse(parts=[ToolCallPart(name, args, "literal")])
        assert not any(isinstance(part, RetryPromptPart) for part in messages[-1].parts)
        return ModelResponse(parts=[TextPart("finished")])

    group = CapabilityGroup("literal-paths")
    group.agent("reader", model="default", allow_tools=(name,))
    async with Runtime.open(
        "literal-paths", models=_Models(_UsageFunctionModel(model)), storage=RuntimeStorage.in_memory(),
        capabilities=(CapabilityGroup("workspace", workspace=Workspace.load(tmp_path)), group),
    ) as runtime:
        execution = await runtime.agents.get("reader").start("use the file")
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        assert requests == 2
        assert target.read_text(encoding="utf-8") == ("changed" if name in {"write_file", "edit_file"} else "original")
