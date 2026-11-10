#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hermetic public-boundary checks for invalid repository-instruction targets."""

import errno
from pathlib import Path
import pytest
from pydantic_ai.messages import ToolReturnPart
from pydantic_ai.models.function import FunctionModel, DeltaToolCall
from linktools.ai.capability import FileSystem, ExecutionCapability, TraceCapability
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.spec import RepositoryInstructions
from linktools.ai.workspace import (
    Workspace,
    WorkspaceInstructionResolver,
    WorkspacePolicy,
)


@pytest.mark.asyncio
async def test_long_argument_is_a_path_rejection(tmp_path: Path) -> None:
    resolver = WorkspaceInstructionResolver(
        tmp_path, WorkspacePolicy(), RepositoryInstructions(())
    )
    with pytest.raises(AIError) as failure:
        await resolver.resolve("a" * 300)
    assert failure.value.code is ErrorCode.STORAGE_PATH_INVALID
    assert failure.value.safe_details == {
        "reason": "repository_instruction_target_path_too_long"
    }


@pytest.mark.asyncio
async def test_real_io_failure_remains_storage_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original = Path.lstat

    def denied(path):
        if path.name == "AGENTS.md":
            raise PermissionError(errno.EACCES, "denied")
        return original(path)

    monkeypatch.setattr(Path, "lstat", denied)
    resolver = WorkspaceInstructionResolver(
        tmp_path, WorkspacePolicy(), RepositoryInstructions(())
    )
    with pytest.raises(AIError) as failure:
        await resolver.resolve("safe.txt")
    assert failure.value.code is ErrorCode.STORAGE_UNAVAILABLE


@pytest.mark.asyncio
@pytest.mark.parametrize("budget", [0, 1])
@pytest.mark.parametrize("path_kind", ["component", "nested"])
async def test_model_corrects_path_using_original_tool_retry_budget(
    tmp_path: Path, budget: int, path_kind: str
) -> None:
    (tmp_path / "safe.txt").write_text("retained safe text")
    requests = 0
    invalid = (
        "a" * 300
        if path_kind == "component"
        else "/".join(
            [
                "evidence/" + str(i) * 64 + "/synthetic-directory-" + str(i)
                for i in range(10)
            ]
        )
    )

    async def stream(messages, info):
        nonlocal requests
        requests += 1
        if requests == 1:
            import json

            yield {
                0: DeltaToolCall(
                    name="read_file",
                    json_args=json.dumps({"path": invalid}),
                    tool_call_id="invalid",
                )
            }
        elif requests == 2:
            assert any(
                isinstance(part, ToolReturnPart) and part.outcome == "retried"
                for msg in messages
                for part in msg.parts
            )
            yield {
                0: DeltaToolCall(
                    name="read_file",
                    json_args='{"path":"safe.txt"}',
                    tool_call_id="corrected",
                )
            }
        else:
            assert requests == 3 and "retained safe text" in str(messages[-1])
            yield "completed"

    class Binding:
        provider = "test"
        model_identity = "test:invalid-path"
        contract = {
            "provider": "test",
            "model_identity": model_identity,
            "input_mime_types": [],
            "settings": {},
        }

        def materialize(self) -> FunctionModel:
            return FunctionModel(stream_function=stream)

    features = RuntimeFeatures("invalid-path")
    features.capability(FileSystem(Workspace.load(tmp_path)), scope="runtime")
    features.capability(ExecutionCapability(retain_history=True), scope="runtime")
    features.capability(TraceCapability(), scope="runtime")
    features.model("default", Binding())
    features.agent(
        "reader", allow_tools=("read_file",), tool_retries=budget, output_retries=0
    )
    async with Persistence.open(
        "invalid-path", storage=PersistenceStorage.in_memory()
    ) as persistence:
        async with Runtime.attach(
            persistence=persistence, features=features
        ) as runtime:
            result = (
                await runtime.agents.get("reader").run(
                    "read safe text", timeout_seconds=10
                )
            ).result
            if budget:
                assert result.status.value == "SUCCEEDED" and requests == 3
            else:
                assert (
                    result.status.value == "FAILED"
                    and result.error_code == "MODEL_RESPONSE_INVALID"
                    and requests == 1
                )
