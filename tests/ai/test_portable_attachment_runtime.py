#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from collections.abc import AsyncIterator, Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    UserPromptPart,
)
from pydantic_ai.models.function import AgentInfo, DeltaToolCall, FunctionModel

from linktools.ai.capability import CapabilityGroup
from linktools.ai.core import (
    ExecutionStatus,
    JsonValue,
    Principal,
    ToolOperationStatus,
    step_run_id,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime, RuntimeState
from linktools.ai.runtime.state import (
    RuntimeDomain,
    managed_attachment_path,
    record_key_digest,
)
from linktools.ai.runtime.state._attachment_repository import AttachmentRepository
from linktools.ai.runtime.state._exposure_repository import ModelExposureRepository
from linktools.ai.workspace import DisabledSandbox, Workspace


async def _text_model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
    del messages, info
    return ModelResponse(parts=[TextPart("ok")])


async def _stream_text_model(
    messages: list[ModelMessage], info: AgentInfo
) -> AsyncIterator[str]:
    del messages, info
    yield "ok"


class _TextModelBinding:
    route_id = "default"
    provider = "test"
    model_identity = "test:test"
    fingerprint = "d" * 64
    semantic_payload: dict[str, JsonValue] = {
        "provider": "test",
        "model": "test",
    }

    def materialize(self) -> FunctionModel:
        return FunctionModel(
            function=_text_model,
            stream_function=_stream_text_model,
        )


class _TextModels:
    def snapshot(self) -> "_TextModels":
        return self

    def resolve(self, route_id: str) -> _TextModelBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _TextModelBinding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _TextModelBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _TextModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _TextModelBinding()


class _CaptureBinding(_TextModelBinding):
    def __init__(self, seen: list[list[ModelMessage]]) -> None:
        self._seen = seen

    def materialize(self) -> FunctionModel:
        seen = self._seen

        async def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            seen.append(list(messages))
            return ModelResponse(parts=[TextPart("ok")])

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str]:
            del info
            seen.append(list(messages))
            yield "ok"

        return FunctionModel(function=request, stream_function=stream)


class _CaptureModels(_TextModels):
    def __init__(self) -> None:
        self.seen: list[list[ModelMessage]] = []

    def resolve(self, route_id: str) -> _CaptureBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return _CaptureBinding(self.seen)

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _CaptureBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _TextModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return _CaptureBinding(self.seen)


class _ReadAttachmentBinding(_TextModelBinding):
    def __init__(self, path: str, body: bytes, seen: list[list[ModelMessage]]) -> None:
        self._path = path
        self._body = body
        self._seen = seen
        self.binary_seen = False

    def materialize(self) -> FunctionModel:
        path = self._path
        body = self._body
        seen = self._seen
        binding = self

        async def request(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            del info
            seen.append(list(messages))
            return ModelResponse(parts=[TextPart("ok")])

        async def stream(
            messages: list[ModelMessage], info: AgentInfo
        ) -> AsyncIterator[str | dict[int, DeltaToolCall]]:
            seen.append(list(messages))
            if _contains_binary(messages, body):
                binding.binary_seen = True
                yield "ok"
                return
            if "read_attachment" not in {tool.name for tool in info.function_tools}:
                raise AssertionError("read_attachment tool is not available")
            yield {
                0: DeltaToolCall(
                    name="read_attachment",
                    json_args=json.dumps(
                        {"path": path},
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    tool_call_id="read-attachment-1",
                )
            }

        return FunctionModel(function=request, stream_function=stream)


class _ReadAttachmentModels(_TextModels):
    def __init__(self, path: str, body: bytes) -> None:
        self.path = path
        self.body = body
        self.seen: list[list[ModelMessage]] = []
        self.bindings: list[_ReadAttachmentBinding] = []

    def _binding(self) -> _ReadAttachmentBinding:
        binding = _ReadAttachmentBinding(self.path, self.body, self.seen)
        self.bindings.append(binding)
        return binding

    @property
    def binary_seen(self) -> bool:
        return any(binding.binary_seen for binding in self.bindings)

    def resolve(self, route_id: str) -> _ReadAttachmentBinding:
        if route_id != "default":
            raise AssertionError(route_id)
        return self._binding()

    def restore(
        self,
        payload: Mapping[str, JsonValue],
        *,
        route_id: str | None = None,
    ) -> _ReadAttachmentBinding:
        if (
            route_id not in {None, "default"}
            or dict(payload) != _TextModelBinding.semantic_payload
        ):
            raise AIError(ErrorCode.MODEL_CONNECTION_NOT_FOUND)
        return self._binding()


def _contains_binary(messages: list[ModelMessage], body: bytes) -> bool:
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            if any(
                isinstance(item, BinaryContent) and item.data == body
                for item in part.content
            ):
                return True
    return False


def _reader_group() -> CapabilityGroup[object]:
    group: CapabilityGroup[object] = CapabilityGroup("attachment-reader-test")
    group.agent(
        "reader",
        allow_tools=("read_attachment",),
        allow_skills=(),
        allow_subagents=(),
    )
    return group


@pytest.mark.asyncio
async def test_managed_path_admission_replays_without_source_read(tmp_path: Path) -> None:
    source = tmp_path / "evidence.txt"
    source.write_bytes(b"immutable evidence")
    workspace = Workspace.load(tmp_path, workspace_id="portable-runtime")
    state = RuntimeState.in_memory()
    key = "managed-runtime-replay-key"

    async with Runtime.open(
        workspace,
        models=_TextModels(),  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        first = await runtime.agent("default").run(
            "inspect the attachment",
            attachments=("evidence.txt",),
            idempotency_key=key,
            timeout_seconds=10,
        )
        assert first.status is ExecutionStatus.SUCCEEDED

        execution = await state.execution.executions.get(
            first.execution_id,
            tenant_id="default",
        )
        assert execution is not None
        assert len(execution.attachment_manifest) == 1
        assert execution.input_digest is not None
        assert execution.path_origin is not None

        repository = AttachmentRepository(
            state.execution.executions.state_store,
            namespace=workspace.workspace_id,
            tenant_id="default",
        )
        owner = repository.prepare_key("execution.run", key)
        preparation = await repository.get_prepare(owner, tenant_id="default")
        assert preparation is not None
        assert preparation.status == "ADOPTED"
        assert preparation.input is None
        assert preparation.target is not None
        assert preparation.target.at.resource == "state:execution"
        assert preparation.target.at.key == repository._key(
            "execution", first.execution_id
        ).hex()

        source.unlink()
        replayed = await runtime.agent("default").run(
            "inspect the attachment",
            attachments=("evidence.txt",),
            idempotency_key=key,
            timeout_seconds=10,
        )
        assert replayed.execution_id == first.execution_id
        assert replayed.status is ExecutionStatus.SUCCEEDED


@pytest.mark.asyncio
async def test_direct_binary_is_exposed_only_at_model_boundary(tmp_path: Path) -> None:
    body = b"direct-image-body"
    workspace = Workspace.load(tmp_path, workspace_id="portable-binary")
    state = RuntimeState.in_memory()
    models = _CaptureModels()

    async with Runtime.open(
        workspace,
        models=models,  # type: ignore[arg-type]
        state=state,
    ) as runtime:
        result = await runtime.agent("default").run(
            (
                "inspect this image",
                BinaryContent(body, media_type="image/png", identifier="evidence"),
            ),
            idempotency_key="direct-binary-key",
            timeout_seconds=10,
        )

        assert result.status is ExecutionStatus.SUCCEEDED
        assert any(_contains_binary(messages, body) for messages in models.seen)

        execution = await state.execution.executions.get(
            result.execution_id,
            tenant_id="default",
        )
        assert execution is not None
        assert len(execution.attachment_manifest) == 1
        run_id = step_run_id(
            namespace=workspace.workspace_id,
            tenant_id="default",
            execution_id=result.execution_id,
            segment_sequence=execution.agent_run_sequence,
        )
        exposure = await ModelExposureRepository(
            state.recovery.checkpoints.state_store,
            namespace=workspace.workspace_id,
            tenant_id="default",
        ).get(
            execution_id=result.execution_id,
            step_run_id=run_id,
            run_step=1,
        )
        assert exposure is not None
        assert len(exposure.entries) == 1
        assert exposure.entries[0].entry == execution.attachment_manifest[0]


@pytest.mark.asyncio
async def test_read_attachment_commits_fact_then_exposes_body_next_model_step(
    tmp_path: Path,
) -> None:
    body = b"ordinary attachment body"
    (tmp_path / "evidence.txt").write_bytes(body)
    workspace = Workspace.load(tmp_path, workspace_id="portable-read")
    state = RuntimeState.in_memory()
    models = _ReadAttachmentModels("evidence.txt", body)

    async with Runtime.open(
        workspace,
        models=models,  # type: ignore[arg-type]
        state=state,
        capabilities=(_reader_group(),),
    ) as runtime:
        result = await runtime.agent("reader").run(
            "read the evidence",
            idempotency_key="read-attachment-runtime-key",
            timeout_seconds=10,
        )

        assert result.status is ExecutionStatus.SUCCEEDED
        assert models.binary_seen
        execution = await state.execution.executions.get(
            result.execution_id,
            tenant_id="default",
        )
        assert execution is not None
        run_id = step_run_id(
            namespace=workspace.workspace_id,
            tenant_id="default",
            execution_id=result.execution_id,
            segment_sequence=execution.agent_run_sequence,
        )
        operation = await cast(Any, state.recovery.tools).get_by_call(
            run_id,
            "read-attachment-1",
            tenant_id="default",
        )
        assert operation is not None
        assert operation.status is ToolOperationStatus.COMPLETED
        assert operation.attachment_result is not None
        assert operation.attachment_result.entry.media_type == "text/plain"

        source = await AttachmentRepository(
            state.execution.executions.state_store,
            namespace=workspace.workspace_id,
            tenant_id="default",
        ).get_source(
            result.execution_id,
            "evidence.txt",
            tenant_id="default",
        )
        assert source is not None
        assert source.entry == operation.attachment_result.entry

        exposure = await ModelExposureRepository(
            state.recovery.checkpoints.state_store,
            namespace=workspace.workspace_id,
            tenant_id="default",
        ).get(
            execution_id=result.execution_id,
            step_run_id=run_id,
            run_step=2,
        )
        assert exposure is not None
        assert exposure.entries[-1].entry == operation.attachment_result.entry


@pytest.mark.asyncio
async def test_virtual_read_attachment_does_not_open_disabled_sandbox(
    tmp_path: Path,
) -> None:
    body = b"virtual attachment body"
    key = "virtual-read-runtime-key"
    workspace = Workspace.load(
        tmp_path,
        workspace_id="portable-virtual-read",
        sandbox=DisabledSandbox(),
    )
    state = RuntimeState.in_memory()
    prepare_owner = record_key_digest(
        workspace.workspace_id,
        "default",
        RuntimeDomain.EXECUTION.value,
        "input_prepare",
        ["execution.run", key],
    ).hex()
    managed_path = managed_attachment_path("p", prepare_owner, 0)
    models = _ReadAttachmentModels(managed_path, body)
    principal = Principal("reader", "default", "local_trusted")

    async with Runtime.open(
        workspace,
        models=models,  # type: ignore[arg-type]
        state=state,
        capabilities=(_reader_group(),),
    ) as runtime:
        upload = await runtime.attachments.upload(
            body,
            media_type="application/octet-stream",
            name="evidence.bin",
            principal=principal,
            idempotency_key="virtual-upload-key",
        )
        result = await runtime.agent("reader").run(
            "read the managed evidence",
            attachments=(upload.path,),
            principal=principal,
            idempotency_key=key,
            timeout_seconds=10,
        )

        assert result.status is ExecutionStatus.SUCCEEDED
        assert models.binary_seen
