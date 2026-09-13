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
from linktools.ai.capability import ToolCallRejected
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._input import ExecutionInputMaterializer
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    RuntimeToolBoundaryToolset,
)
from linktools.ai.storage import StoredPayload
from linktools.ai.workspace import (
    SandboxResource,
    SandboxSession,
    WorkspacePolicy,
)
from ._runtime_test_helpers import semantic_tool


class _Session:
    def __init__(self, values: dict[str, bytes]) -> None:
        self.values = values
        self.reads: list[str] = []

    async def canonicalize_path(self, path: str) -> str:
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads.append(path)
        try:
            value = self.values[path]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_NOT_FOUND) from error
        if max_bytes is not None and len(value) > max_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def close(self) -> None:
        return None


class _ErrorSession(_Session):
    def __init__(self, error: ErrorCode, *, canonicalize: bool) -> None:
        super().__init__({"evidence.txt": b"evidence"})
        self.error = error
        self.fail_canonicalize = canonicalize

    async def canonicalize_path(self, path: str) -> str:
        if self.fail_canonicalize:
            raise AIError(self.error)
        return path

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        if not self.fail_canonicalize:
            raise AIError(self.error)
        return await super().read_bytes(path, max_bytes=max_bytes)


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


class _DeniedAccess:
    async def canonicalize_path(self, path: str) -> str:
        del path
        raise AIError(ErrorCode.AUTHORIZATION_DENIED)


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


def _materializer(session: _Session) -> ExecutionInputMaterializer:
    return ExecutionInputMaterializer(
        WorkspaceAccess(_Sandbox(session), root=Path(".")),
        WorkspacePolicy(),
    )


@pytest.mark.asyncio
async def test_text_materialization_keeps_text_codec() -> None:
    materializer = _materializer(_Session({}))
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
    materializer = _materializer(session)
    service = object.__new__(DefaultExecutionService)
    service._input_materializer = materializer  # type: ignore[attr-defined]

    try:
        canonical = await service._canonicalize_request(
            _request(files=("evidence.txt",))
        )
        prepared = await service._freeze_input(canonical)
        assert prepared.request.files == ()
        assert prepared.stored_user_input is not None
        assert isinstance(prepared.request.user_prompt, tuple)
        assert session.reads == ["evidence.txt"]

        replay = await materializer.restore(prepared.stored_user_input)
        assert replay == prepared.request.user_prompt
        assert session.reads == ["evidence.txt"]
    finally:
        await materializer.close()


@pytest.mark.asyncio
async def test_execution_file_path_domain_error_becomes_request_error() -> None:
    materializer = _materializer(
        _ErrorSession(ErrorCode.AUTHORIZATION_DENIED, canonicalize=True)
    )
    try:
        with pytest.raises(AIError) as raised:
            await materializer.canonicalize_files(("../secret.txt",))
    finally:
        await materializer.close()

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details == {
        "field": "files",
        "reason": "path_not_allowed",
    }


@pytest.mark.asyncio
async def test_execution_missing_file_becomes_request_error() -> None:
    materializer = _materializer(_Session({}))
    try:
        with pytest.raises(AIError) as raised:
            await materializer.materialize("inspect", ("missing.txt",))
    finally:
        await materializer.close()

    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details == {
        "field": "files",
        "reason": "file_not_found",
    }


@pytest.mark.asyncio
async def test_execution_file_infrastructure_error_is_not_downgraded() -> None:
    materializer = _materializer(
        _ErrorSession(ErrorCode.STORAGE_UNAVAILABLE, canonicalize=False)
    )
    try:
        with pytest.raises(AIError) as raised:
            await materializer.materialize("inspect", ("evidence.txt",))
    finally:
        await materializer.close()

    assert raised.value.code is ErrorCode.STORAGE_UNAVAILABLE


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


def _workspace_boundary(sandbox_session: object) -> RuntimeToolBoundaryToolset:
    descriptor = ManagedToolDescriptor(
        effect_owner="none",
        effect="none",
        tool_class="filesystem.read",
        workspace_path_fields=("path",),
    )
    return RuntimeToolBoundaryToolset(
        (FunctionToolset([semantic_tool(_echo_path, descriptor)]),),
        {"_echo_path": descriptor},
        id="workspace",
        sandbox_session=sandbox_session,  # type: ignore[arg-type]
    )


@pytest.mark.asyncio
async def test_final_tool_boundary_canonicalizes_workspace_arguments() -> None:
    session = _Session({})
    boundary = _workspace_boundary(session)
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
async def test_final_tool_boundary_returns_model_retry_for_correctable_path_error() -> None:
    boundary = _workspace_boundary(_DeniedAccess())
    context = _context()
    tools = await boundary.get_tools(context)

    with pytest.raises(ToolCallRejected, match="not allowed"):
        await boundary.call_tool(
            "_echo_path",
            {"path": "../secret.txt"},
            context,
            tools["_echo_path"],
        )


@pytest.mark.asyncio
async def test_final_tool_boundary_does_not_freeze_transient_sandbox_failure() -> None:
    boundary = _workspace_boundary(_UnavailableAccess())
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
