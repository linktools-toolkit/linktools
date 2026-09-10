#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution input and final tool-boundary regressions."""

from pathlib import Path

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.capability import WorkspaceAccess
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._input import ExecutionInputMaterializer
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.runtime.state._contracts import RuntimeStorageContract
from linktools.ai.storage import StoredPayload
from linktools.ai.workspace import SandboxResource, SandboxSession, Workspace


class _Session:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values
        self.reads: list[str] = []

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads.append(path)
        value = self.values[path]
        if max_bytes is not None and len(value) > max_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def close(self) -> None:
        return None


class _Sandbox:
    def __init__(self, session: _Session) -> None:
        self.session = session

    async def open(
        self,
        *,
        root: Path,
        resources: tuple[SandboxResource, ...] = (),
    ) -> SandboxSession:
        del root, resources
        return self.session  # type: ignore[return-value]


class _UnavailableAccess:
    async def canonicalize_path(self, path: str) -> str:
        del path
        raise AIError(ErrorCode.SANDBOX_UNAVAILABLE)


def _request(*, files: tuple[str, ...] = ()) -> ExecutionRequest:
    return ExecutionRequest(
        user_prompt="inspect",
        principal=Principal("user", "tenant", "local_trusted"),
        idempotency_key="file-input",
        memory_scope=None,
        mode="run",
        planning=False,
        thinking=False,
        files=files,
    )


@pytest.mark.asyncio
async def test_text_materialization_keeps_text_codec() -> None:
    access = WorkspaceAccess(_Sandbox(_Session({})), root=Path("."))
    materializer = ExecutionInputMaterializer(access, Workspace.load(".", workspace_id="workspace").policy)
    try:
        canonical = await materializer.materialize("plain text", ())
        stored = await materializer.store(canonical, tenant_id="tenant")
        assert canonical == "plain text"
        assert stored.codec == "text"
        assert stored.payload == StoredPayload.inline_text("plain text")
    finally:
        await materializer.close()


def test_invalid_user_content_is_rejected_at_request_boundary() -> None:
    with pytest.raises(AIError) as raised:
        ExecutionRequest(
            user_prompt=(123,),  # type: ignore[arg-type]
            principal=Principal("user", "tenant", "local_trusted"),
            idempotency_key="invalid-user-content",
            memory_scope=None,
            mode="run",
            planning=False,
            thinking=False,
        )

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_execution_freezes_materialized_input_once() -> None:
    session = _Session({"evidence.txt": b"evidence"})
    access = WorkspaceAccess(_Sandbox(session), root=Path("."))
    materializer = ExecutionInputMaterializer(access, Workspace.load(".", workspace_id="workspace").policy)
    service = object.__new__(DefaultExecutionService)
    service._input_materializer = materializer  # type: ignore[attr-defined]
    service._storage_contract_factory = (  # type: ignore[attr-defined]
        lambda _domains: RuntimeStorageContract(1, (), (), ())
    )

    try:
        canonical = await service._canonicalize_request(
            _request(files=("evidence.txt",))
        )
        prepared = await service._freeze_input(canonical, session_id=None)
        assert prepared.request.files == ()
        assert prepared.stored_user_input is not None
        assert isinstance(prepared.request.user_prompt, tuple)
        assert session.reads == ["evidence.txt"]

        replay = await materializer.restore(prepared.stored_user_input)
        assert replay == prepared.request.user_prompt
        assert session.reads == ["evidence.txt"]
    finally:
        await materializer.close()


async def _echo_path(path: str) -> str:
    return path


def _context() -> RunContext[None]:
    return RunContext(
        deps=None,
        model=TestModel(),
        usage=RunUsage(),
        run_id="run",
        tool_call_id="call",
    )


@pytest.mark.asyncio
async def test_final_tool_boundary_canonicalizes_workspace_arguments() -> None:
    session = _Session({})
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([_echo_path]),),
        {
            "_echo_path": ManagedToolDescriptor(
                effect_owner="intrinsic",
                effect="none",
                tool_class="filesystem.read",
                workspace_path_fields=("path",),
            )
        },
        id="workspace",
        sandbox_session=session,  # type: ignore[arg-type]
    )
    context = _context()
    tools = await boundary.get_tools(context)
    args = {"path": "file.txt"}

    result = await boundary.call_tool(
        "_echo_path",
        args,
        context,
        tools["_echo_path"],
    )

    assert result == "file.txt"
    assert args == {"path": "file.txt"}


@pytest.mark.asyncio
async def test_final_tool_boundary_does_not_freeze_transient_sandbox_failure() -> None:
    boundary = RuntimeToolBoundaryToolset(
        (FunctionToolset([_echo_path]),),
        {
            "_echo_path": ManagedToolDescriptor(
                effect_owner="intrinsic",
                effect="none",
                tool_class="filesystem.read",
                workspace_path_fields=("path",),
            )
        },
        id="workspace",
        sandbox_session=_UnavailableAccess(),  # type: ignore[arg-type]
    )
    context = _context()
    tools = await boundary.get_tools(context)

    with pytest.raises(AIError) as raised:
        await boundary.call_tool(
            "_echo_path",
            {"path": "a.txt"},
            context,
            tools["_echo_path"],
        )

    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
