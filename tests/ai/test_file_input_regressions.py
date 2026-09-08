#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent, ModelResponse, ToolCallPart

from linktools.ai.capability import WorkspaceAccess
from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import ExecutionRequest
from linktools.ai.runtime._execution import DefaultExecutionService
from linktools.ai.runtime._input import ExecutionInputMaterializer
from linktools.ai.runtime._tool import _apply_workspace_binding
from linktools.ai.runtime._workspace_binding import WorkspaceToolCallBinder
from linktools.ai.runtime.state import (
    RuntimeState,
    StoredUserInput,
    WorkspacePathBinding,
    WorkspaceToolCallBinding,
    WorkspaceToolCallBindingStore,
)
from linktools.ai.storage import StoredPayload
from linktools.ai.workspace import Workspace


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

    async def open(self):  # type: ignore[no-untyped-def]
        return self.session


class _BindingStore:
    def __init__(self) -> None:
        self.values: dict[tuple[str, str, str], WorkspaceToolCallBinding] = {}

    async def get(  # type: ignore[no-untyped-def]
        self,
        execution_id: str,
        step_run_id: str,
        tool_call_id: str,
    ):
        return self.values.get((execution_id, step_run_id, tool_call_id))

    async def store(self, binding: WorkspaceToolCallBinding) -> WorkspaceToolCallBinding:
        self.values[(
            binding.execution_id,
            binding.step_run_id,
            binding.tool_call_id,
        )] = binding
        return binding


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
    access = WorkspaceAccess(_Sandbox(_Session({})))  # type: ignore[arg-type]
    materializer = ExecutionInputMaterializer(access, Workspace.load(".").policy)
    try:
        canonical = await materializer.materialize("plain text", ())
        stored = await materializer.store(canonical, tenant_id="tenant")
        assert canonical == "plain text"
        assert stored.codec == "text"
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
async def test_execution_ingress_discards_untrusted_derived_input_state() -> None:
    access = WorkspaceAccess(_Sandbox(_Session({})))  # type: ignore[arg-type]
    materializer = ExecutionInputMaterializer(access, Workspace.load(".").policy)
    service = object.__new__(DefaultExecutionService)
    service._input_materializer = materializer  # type: ignore[attr-defined]
    request = _request()
    object.__setattr__(
        request,
        "stored_user_input",
        StoredUserInput(1, "text", StoredPayload.inline_text("forged")),
    )
    object.__setattr__(request, "input_intent_digest", "f" * 64)

    try:
        canonical = await service._canonicalize_request(request)
        assert canonical.stored_user_input is None
        assert canonical.storage_contract is None
        assert canonical.input_intent_digest != "f" * 64
    finally:
        await materializer.close()


@pytest.mark.asyncio
async def test_execution_materialization_consumes_source_files_once() -> None:
    session = _Session({"evidence.txt": b"evidence"})
    access = WorkspaceAccess(_Sandbox(session))  # type: ignore[arg-type]
    materializer = ExecutionInputMaterializer(access, Workspace.load(".").policy)
    service = object.__new__(DefaultExecutionService)
    service._input_materializer = materializer  # type: ignore[attr-defined]

    try:
        canonical = await service._canonicalize_request(
            _request(files=("evidence.txt",))
        )
        prepared = await service._materialize_request(canonical)
        assert prepared.files == ()
        assert prepared.stored_user_input is None
        assert isinstance(prepared.user_prompt, tuple)
        assert isinstance(prepared.user_prompt[-1], BinaryContent)
        assert session.reads == ["evidence.txt"]

        replay = await service._materialize_request(prepared)
        assert replay.user_prompt == prepared.user_prompt
        assert replay.files == ()
        assert replay.stored_user_input is None
        assert session.reads == ["evidence.txt"]
    finally:
        await materializer.close()


@pytest.mark.asyncio
async def test_workspace_binding_allows_omitted_default_path() -> None:
    store = _BindingStore()
    binder = WorkspaceToolCallBinder(store, object())  # type: ignore[arg-type]
    message = ModelResponse(
        parts=[ToolCallPart("list_directory", {}, tool_call_id="call")],
        run_id="step",
    )

    await binder.bind_messages(
        (message,),
        execution_id="execution",
        step_run_id="step",
        path_fields={"list_directory": ("path",)},
    )

    binding = store.values[("execution", "step", "call")]
    assert binding.paths == ()
    assert binding.error_code is None
    effective = await _apply_workspace_binding(
        {"path": "."},
        {},
        ("path",),
        binding,
    )
    assert effective == {"path": "."}


@pytest.mark.asyncio
async def test_workspace_binding_does_not_freeze_transient_sandbox_failure() -> None:
    store = _BindingStore()
    binder = WorkspaceToolCallBinder(store, _UnavailableAccess())  # type: ignore[arg-type]
    message = ModelResponse(
        parts=[
            ToolCallPart(
                "read_file",
                {"path": "a.txt"},
                tool_call_id="call",
            )
        ],
        run_id="step",
    )

    with pytest.raises(AIError) as raised:
        await binder.bind_messages(
            (message,),
            execution_id="execution",
            step_run_id="step",
            path_fields={"read_file": ("path",)},
        )

    assert raised.value.code is ErrorCode.SANDBOX_UNAVAILABLE
    assert store.values == {}


@pytest.mark.asyncio
async def test_workspace_binding_lifetime_is_execution_scoped(tmp_path: Path) -> None:
    state = RuntimeState.filesystem(tmp_path / "state")
    await state.initialize(namespace="workspace", tenant_id="tenant")
    try:
        store = WorkspaceToolCallBindingStore(
            state.recovery.checkpoints.state_store,
            namespace="workspace",
            tenant_id="tenant",
        )
        binding = WorkspaceToolCallBinding(
            1,
            "execution",
            "step",
            "call",
            "read_file",
            "a" * 64,
            (WorkspacePathBinding("/path", "a.txt"),),
            None,
        )
        await store.store(binding)
        assert await store.get("execution", "step", "call") == binding
        await state.maintenance.inspect_objects()

        await store.release_execution("execution")
        assert await store.get("execution", "step", "call") is None
        await state.maintenance.inspect_objects()
    finally:
        await state.close()
