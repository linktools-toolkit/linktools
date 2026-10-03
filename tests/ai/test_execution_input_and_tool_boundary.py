#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution input and final tool-boundary regressions."""

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic_ai.models.test import TestModel
from pydantic_ai.toolsets import FunctionToolset
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.core import HmacCursorSigner, Principal, PromptLimits, TenantAuthorizationPolicy
from linktools.ai.capability import ToolCallRetry
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest, RuntimeStorage
from linktools.ai.runtime._session import DefaultSessionService
from linktools.ai.runtime.service_api import (
    CreateSessionRequest,
    ExecutionHandle,
    ForkExecutionRequest,
    RetryExecutionRequest,
    ResumeSessionRequest,
)
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._input import (
    ExecutionInputMaterializer,
    stored_input_attachment_views,
    stored_user_input_view,
)
from linktools.ai.runtime._tool_boundary import (
    ManagedToolDescriptor,
    BoundaryToolset,
)
from linktools.ai.storage import StoredPayload
from linktools.ai.workspace import (
    SandboxResource,
    SandboxSession,
    WorkspaceAccess,
)
from ._runtime_test_helpers import tool_with_metadata


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
        PromptLimits(),
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
        assert stored.view is None
        assert stored_user_input_view(stored) == {
            "version": 1,
            "prompt": {"kind": "text", "text": "plain text"},
            "files": [],
            "attachments": [],
        }
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
        view = prepared.stored_user_input.view
        assert view is not None
        assert view["version"] == 1
        assert view["prompt"] == {"kind": "text", "text": "inspect"}
        assert view["files"] == [
            {
                "path": "evidence.txt",
                "media_type": "text/plain",
                "size": 8,
                "digest": (
                    "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
                ),
            }
        ]
        attachments = view["attachments"]
        assert isinstance(attachments, list)
        assert len(attachments) == 1
        assert attachments[0]["fact"] == "accepted"
        assert attachments[0]["source"] == "workspace"
        assert attachments[0]["media_type"] == "text/plain"
        assert attachments[0]["size"] == 8
        assert attachments[0]["digest"] == (
            "ee8250fb76e094b34b471f13a73dbbe51d1ae142e9df59d7c0d31ec20f0a0a8e"
        )
        assert attachments[0]["position"] == 0
        assert attachments[0]["call_id"] is None
        assert attachments[0]["input_identifier"] is None
        view_text = str(prepared.stored_user_input.view)
        assert "Workspace file path" not in view_text
        assert "ZXZpZGVuY2U=" not in view_text

        attachment_id = attachments[0]["attachment_id"]
        replay = await materializer.restore(prepared.stored_user_input)
        assert replay == prepared.request.user_prompt
        assert stored_input_attachment_views(prepared.stored_user_input)[0][
            "attachment_id"
        ] == attachment_id
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


def _workspace_boundary(sandbox_session: object) -> BoundaryToolset:
    descriptor = ManagedToolDescriptor(
        effect_owner="none",
        effect_policy="none",
        tool_class="filesystem.read",
        workspace_path_fields=("path",),
    )
    return BoundaryToolset(
        (FunctionToolset([tool_with_metadata(_echo_path, descriptor)]),),
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

    with pytest.raises(ToolCallRetry, match="not allowed"):
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


_REQUEST_TYPES = (ExecutionRequest, ResumeSessionRequest, RetryExecutionRequest, ForkExecutionRequest)


def _file_request(request_type, files):
    values = dict(
        user_prompt="inspect", principal=Principal("user", "tenant", "local_trusted"),
        idempotency_key="file-input", correlation={"trace": "original"}, files=files,
    )
    if request_type in (ExecutionRequest, ResumeSessionRequest):
        values.update(memory_scope="memory", mode="plan", planning=True, thinking=False)
    return request_type(**values)


@pytest.mark.parametrize("request_type", _REQUEST_TYPES)
@pytest.mark.parametrize(
    "files",
    (None, "file.txt", b"file.txt", bytearray(b"file.txt"), {"file.txt"},
     {"file.txt": 1}, iter(("file.txt",)), [""], [None], [1]),
)
def test_request_files_reject_invalid_sequences(request_type: type, files: object) -> None:
    with pytest.raises(AIError) as raised:
        _file_request(request_type, files)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.parametrize("request_type", _REQUEST_TYPES)
@pytest.mark.parametrize("files", ([], ["b.txt", "a.txt", "b.txt"], [" ", "../file.txt"]))
def test_request_file_normalization_preserves_input_meaning(request_type: type, files: list[str]) -> None:
    files = list(files)
    request = _file_request(request_type, files)
    assert request.files == tuple(files)
    assert request.correlation == {"trace": "original"}
    assert replace(request) == request
    assert repr(request).startswith(request_type.__name__ + "(")
    if request_type in (ExecutionRequest, ResumeSessionRequest):
        assert (request.memory_scope, request.mode, request.planning, request.thinking) == (
            "memory", "plan", True, False,
        )
    files.append("added-later.txt")
    assert "added-later.txt" not in request.files


def test_request_types_keep_distinct_identity_and_positional_fields() -> None:
    first_fields = {
        ExecutionRequest: ("user_prompt", "principal"),
        ResumeSessionRequest: ("principal", "user_prompt"),
        RetryExecutionRequest: ("user_prompt", "principal"),
        ForkExecutionRequest: ("user_prompt", "principal"),
    }
    requests = [_file_request(request_type, ()) for request_type in _REQUEST_TYPES]
    for request_type, expected in first_fields.items():
        assert tuple(field.name for field in fields(request_type))[:2] == expected
    for index, request in enumerate(requests):
        assert all(request != other for other in requests[index + 1:])


@pytest.mark.asyncio
@pytest.mark.parametrize("files", ("file.txt", {"file.txt"}, [""], [None]))
async def test_materializer_rejects_invalid_files_before_workspace_access(files: object) -> None:
    materializer = ExecutionInputMaterializer(None, PromptLimits())
    with pytest.raises(AIError) as raised:
        await materializer.canonicalize_files(files)
    assert raised.value.code is ErrorCode.REQUEST_FIELD_INVALID
    assert raised.value.safe_details == {}


@pytest.mark.asyncio
async def test_session_resume_preserves_explicit_execution_input_fields() -> None:
    storage = RuntimeStorage.in_memory()
    await storage.initialize(namespace="resume-input", tenant_id="tenant")
    execution = SimpleNamespace(
        start_for_session=AsyncMock(return_value=ExecutionHandle("execution")),
    )
    service = DefaultSessionService(
        storage.conversation, storage.execution.executions,
        TenantAuthorizationPolicy(), execution,
        HmacCursorSigner("session", b"session-key"), history_reader=object(),
    )
    request = _file_request(ResumeSessionRequest, ["b.txt", "a.txt", "b.txt"])
    try:
        await service.create("agent", CreateSessionRequest(request.principal, "session", "create"))
        handle = await service.resume("agent", "a" * 64, "session", request)
        captured = execution.start_for_session.await_args.args[3]
        assert handle == ExecutionHandle("execution")
        assert type(captured) is ExecutionRequest
        assert captured == _file_request(ExecutionRequest, request.files)
    finally:
        await storage.close()
