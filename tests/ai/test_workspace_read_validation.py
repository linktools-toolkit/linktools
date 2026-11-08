#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Same-file-generation validation at the existing read owner boundary."""

import asyncio
from pathlib import Path
from typing import BinaryIO

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.workspace import (
    LocalWorkspaceBackend,
    WorkspaceOperationRejected,
    WorkspaceReadPolicy,
)


async def _read(backend: LocalWorkspaceBackend, operation: str):
    if operation == "inspect":
        return await backend.inspect_file("source.yaml")
    if operation == "text":
        return await backend.read_text("source.yaml", offset=1, limit=1)
    if operation == "binary":
        return await backend.read_bytes("source.yaml", max_bytes=1024)
    return await backend.search_text("second", path="source.yaml")


@pytest.mark.parametrize("operation", ["inspect", "text", "binary", "search"])
def test_validator_rejection_is_not_search_no_matches(tmp_path: Path, operation: str) -> None:
    (tmp_path / "source.yaml").write_text("first\nsecond\n", encoding="utf-8")

    def reject(path: str, resource: str | None, stream: BinaryIO) -> None:
        assert (path, resource) == ("source.yaml", None)
        assert stream.read(5) == b"first"
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "source_unverified"})

    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("**",), read_validator=reject))
        try:
            with pytest.raises(WorkspaceOperationRejected) as raised:
                await _read(backend, operation)
            assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
            assert raised.value.safe_details == {"reason": "source_unverified"}
            assert raised.value.retryable is False
        finally:
            await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize("operation", ["inspect", "text", "binary", "search"])
@pytest.mark.parametrize("mutation", ["replace", "same_inode"])
def test_state_captured_before_validator_and_checked_after_output(tmp_path: Path, operation: str, mutation: str) -> None:
    target = tmp_path / "source.yaml"
    target.write_text("first\nsecond\n", encoding="utf-8")

    def mutate(path: str, resource: str | None, stream: BinaryIO) -> None:
        assert stream.read() == b"first\nsecond\n"
        if mutation == "replace":
            replacement = tmp_path / "new.yaml"
            replacement.write_text("first\nSECRET\n", encoding="utf-8")
            replacement.replace(target)
        else:
            with target.open("r+b") as writer:
                writer.write(b"first\nSECRET\n")

    async def run() -> None:
        backend = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("**",), read_validator=mutate))
        try:
            if operation == "search":
                # Existing search partial-coverage semantics remain unchanged;
                # changed-file data is never included in its returned rows.
                result = await _read(backend, operation)
                assert "SECRET" not in result
                assert "source.yaml:2:" not in result
                assert "incomplete" in result
            else:
                with pytest.raises(AIError) as raised:
                    await _read(backend, operation)
                assert raised.value.code is ErrorCode.STORAGE_CONFLICT
        finally:
            await backend.close()

    asyncio.run(run())


@pytest.mark.parametrize("ending", ["\n", "\r\n", "\r", "\u0085", "\u2028", "\u2029"])
def test_validator_restores_position_and_preserves_unicode_page_semantics(tmp_path: Path, ending: str) -> None:
    text = ending.join(["first", "second 中文", "third"])
    (tmp_path / "source.yaml").write_bytes(text.encode())
    calls = []

    def validate(path: str, resource: str | None, stream: BinaryIO) -> None:
        calls.append((path, resource))
        assert stream.read(1024) == text.encode()

    async def run() -> None:
        baseline = LocalWorkspaceBackend(tmp_path)
        guarded = LocalWorkspaceBackend(tmp_path, read_policy=WorkspaceReadPolicy(("**",), read_validator=validate))
        try:
            for operation in ("inspect", "text", "binary", "search"):
                assert await _read(guarded, operation) == await _read(baseline, operation)
            first = await guarded.read_text("source.yaml", limit=1)
            if first.get("next_cursor"):
                assert await guarded.read_text("source.yaml", cursor=first["next_cursor"]) == await baseline.read_text("source.yaml", cursor=first["next_cursor"])
            assert len(calls) >= 5
        finally:
            await guarded.close()
            await baseline.close()

    asyncio.run(run())


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["whole", "partial", "search", "binary"])
@pytest.mark.parametrize("reject", [False, True])
async def test_real_runtime_filesystem_uses_validator_before_tool_result(tmp_path: Path, kind: str, reject: bool) -> None:
    import json
    from pydantic_ai.messages import ToolReturnPart
    from pydantic_ai.models.function import DeltaToolCall, FunctionModel
    from linktools.ai.capability import ExecutionCapability, FileSystem
    from linktools.ai.persistence import PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.workspace import Workspace
    from ._runtime_test_helpers import _Models

    root = tmp_path / "workspace"
    root.mkdir()
    field = "https://example.invalid/key?token=SYNTHETIC_SECRET" if reject else "company"
    value = json.dumps({field: "safe company", "count": 17}, indent=2)
    (root / "source.yaml").write_text(value, encoding="utf-8")
    validations = []
    calls = []

    def validate(path: str, resource: str | None, stream: BinaryIO) -> None:
        validations.append((path, resource))
        assert stream.read(4096).decode() == value
        if reject:
            raise WorkspaceOperationRejected(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "source_unverified"})

    def factory(path: Path, *, resources=()):
        return LocalWorkspaceBackend(path, resources=resources, read_policy=WorkspaceReadPolicy(("source.yaml",), read_validator=validate))

    async def model(messages, info):
        calls.append(messages)
        if len(calls) == 1:
            name = "read_file"
            args = {"path": "source.yaml"}
            if kind == "partial":
                args.update(offset=1, limit=1)
            elif kind == "binary":
                args["as_file"] = True
            elif kind == "search":
                name = "search_files"
                args["pattern"] = "safe company"
            yield {0: DeltaToolCall(name=name, json_args=json.dumps(args), tool_call_id="read")}
        else:
            returns = [p for m in messages for p in m.parts if isinstance(p, ToolReturnPart)]
            assert returns
            rendered = str([p.content for p in returns])
            if reject:
                assert "safe company" not in rendered
                assert "SYNTHETIC_SECRET" not in str(messages)
                assert "example.invalid" not in str(messages)
                expected = (
                    "The search_files arguments are invalid. Use a valid regular expression, "
                    "a valid workspace path, and a relative include_glob when provided."
                    if kind == "search" else
                    "The read_file arguments are invalid. Use a non-negative offset and a positive "
                    "limit; if the offset is past the file, retry from an earlier offset."
                )
                assert returns[-1].content == expected
            else:
                assert "source.yaml" in rendered
                if kind != "binary":
                    assert "safe company" in rendered
            yield "checked"

    features = RuntimeFeatures("read-validator", workspace_factory=factory)
    features.capability(ExecutionCapability(retain_history=True))
    features.capability(FileSystem(Workspace.load(root)))
    features.model("default", _Models(FunctionModel(stream_function=model)))
    features.agent("default", allow_tools=("read_file", "search_files"), allow_skills=(), allow_subagents=())
    async with Runtime.open("read-validator", features=features, storage=PersistenceStorage.from_root(tmp_path / "state")) as runtime:
        execution = await runtime.agents.get().run("Read the requested source.", timeout_seconds=10)
        assert str(execution.result.status) == "SUCCEEDED", execution.result
        assert len(calls) == 2
        assert validations and all(item == ("source.yaml", None) for item in validations)


def test_authorization_precedes_validator_and_page_budget_is_unchanged(tmp_path: Path) -> None:
    (tmp_path / "source.yaml").write_bytes(b"first\nsecond\nthird\n" * 100)
    (tmp_path / "private.yaml").write_text("not authorized")
    calls = []

    def validate(path: str, resource: str | None, stream: BinaryIO) -> None:
        calls.append(path)
        assert stream.read(5) == b"first"

    async def run() -> None:
        policy = WorkspaceReadPolicy(("source.yaml",), read_validator=validate)
        guarded = LocalWorkspaceBackend(tmp_path, read_policy=policy)
        baseline = LocalWorkspaceBackend(tmp_path)
        try:
            with pytest.raises(AIError) as raised:
                await guarded.read_text("private.yaml")
            assert raised.value.code is ErrorCode.AUTHORIZATION_DENIED
            assert calls == []
            result = await guarded.read_text("./source.yaml", max_read_bytes=17)
            assert result == await baseline.read_text("source.yaml", max_read_bytes=17)
            assert calls == ["source.yaml"]
            for backend in (guarded, baseline):
                with pytest.raises(AIError) as limited:
                    await backend.read_text("source.yaml", full_metadata=True, max_read_bytes=17)
                assert limited.value.code is ErrorCode.REQUEST_FIELD_INVALID
                assert limited.value.safe_details == {"reason": "scan_byte_limit"}
        finally:
            await guarded.close()
            await baseline.close()

    asyncio.run(run())
