#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Recoverable input bytes are transient while their accepted references endure."""

import asyncio
import base64
import hashlib
import json
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from pydantic_ai.capabilities import CombinedCapability, WrapperCapability
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    ModelResponse,
    TextContent,
    TextPart,
    ToolReturn,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel
from pydantic_ai.exceptions import ModelHTTPError

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ReferencedInputsCapability, TraceCapability
from linktools.ai.core import ExecutionStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.model import ModelClientRequestRecorder, ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._harness import SummaryProviderModel
from linktools.ai.inputs import decode_task_prompt_draft, task_prompt_draft
from linktools.ai.model.messages import (
    binary_content_usage,
    decode_model_messages,
    encode_model_messages,
    project_transient_binary_content,
)
from linktools.ai.persistence._views import project_session_history_message
from linktools.ai.model._tool_return_codec import decode_tool_return_content, encode_tool_return_content
from linktools.ai.runtime._metric_capability import model_request_error
from linktools.ai.storage import StorageEntryRevision
from linktools.ai.spec import CapabilityRef
from ._runtime_test_helpers import model_features

_BODY = b"synthetic-file-body-never-durable"
_SOURCE = CapabilityRef("capability", "files", 1)


def _content(data: bytes = _BODY) -> ReferencedContent:
    return ReferencedContent(AssetVersionRef(
        AssetKey("document", "immutable-file"), "configured", StorageEntryRevision(1),
        hashlib.sha256(data).hexdigest(), len(data),
    ), media_type="application/pdf", source=_SOURCE)


def _context() -> AgentContext[None]:
    return AgentContext(None, service_principal("tenant", "admitted-user"),
                        "test", None, "execution", {})


class _Binding:
    provider = "test"
    model_identity = "test:references"
    contract = {"input_mime_types": ["application/pdf"]}


def test_descriptor_round_trips_without_bytes_through_prompt_history_and_views() -> None:
    content = _content()
    draft = task_prompt_draft(["inspect", content])
    restored_prompt = decode_task_prompt_draft(draft)
    assert restored_prompt[1] == content
    messages = [ModelRequest(parts=[UserPromptPart(restored_prompt)])]
    raw = encode_model_messages(messages)
    restored, = decode_model_messages(raw)
    assert restored.parts[0].content[1] == content
    view, = project_session_history_message(restored)
    assert view.content[1]["kind"] == "referenced-content"
    assert view.content[1]["ref"] == content.ref.to_payload()
    serialized = raw + json.dumps(draft).encode() + json.dumps(view.content).encode()
    assert _BODY not in serialized
    assert base64.b64encode(_BODY) not in serialized
    assert binary_content_usage(messages) == (1, len(_BODY))


def test_complete_response_retires_reference_but_pending_tail_remains_recoverable() -> None:
    consumed = ModelRequest(parts=[UserPromptPart([_content()])])
    pending = ModelRequest(parts=[UserPromptPart([_content(b"pending")])])
    messages = [consumed, ModelResponse(parts=[TextPart("understood")]), pending]
    projected = project_transient_binary_content(messages)
    assert isinstance(projected[0].parts[0].content[0], TextContent)
    assert _content().ref.to_payload()["digest"] in encode_model_messages(projected).decode()
    assert projected[2] is pending
    assert isinstance(consumed.parts[0].content[0], ReferencedContent)


def test_consumed_reference_summary_uses_the_current_descriptor() -> None:
    content = _content()
    content.ref = _content(b"x").ref
    messages = [ModelRequest(parts=[UserPromptPart([content])]), ModelResponse(parts=[TextPart("done")])]
    projected = project_transient_binary_content(messages)
    summary = projected[0].parts[0].content[0]
    assert json.loads(summary.content)["size"] == 1
    assert content.ref.digest in encode_model_messages(projected).decode()


@pytest.mark.asyncio
async def test_provider_gets_authorized_bytes_without_mutating_durable_messages() -> None:
    context = _context()
    calls = []

    async def authorized_load(ctx, content):
        assert ctx is context
        assert content.ref == _content().ref
        calls.append("authorized")
        return _BODY

    async def respond(messages, info):
        item = messages[0].parts[0].content[0]
        assert type(item) is BinaryContent and item.data == _BODY
        calls.append("provider")
        return ModelResponse(parts=[TextPart("done")])

    original = [ModelRequest(parts=[UserPromptPart([_content()])])]
    encoded = encode_model_messages(original)
    model = SummaryProviderModel(FunctionModel(respond), context=context,
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(authorized_load)},
                                 binding=_Binding())
    with ModelClientRequestRecorder(record_http_bodies=True) as recorder:
        await model.request(original, None, ModelRequestParameters())
        assert recorder.record_http_bodies is True
    assert calls == ["authorized", "provider"]
    assert encode_model_messages(original) == encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["denied", "changed", "cancelled"])
async def test_failed_load_never_dispatches_and_preserves_original_ref(failure: str) -> None:
    requests = []
    cleanup = []

    async def load(ctx, content):
        try:
            if failure == "denied":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if failure == "cancelled":
                raise asyncio.CancelledError
            return b"changed bytes"
        finally:
            cleanup.append(True)

    async def respond(messages, info):
        requests.append(messages)
        return ModelResponse(parts=[TextPart("unexpected")])

    original = [ModelRequest(parts=[UserPromptPart([_content()])])]
    model = SummaryProviderModel(FunctionModel(respond), context=_context(),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)}, binding=_Binding())
    with pytest.raises(asyncio.CancelledError if failure == "cancelled" else AIError):
        await model.request(original, None, ModelRequestParameters())
    assert not requests and cleanup == [True]
    assert isinstance(original[0].parts[0].content[0], ReferencedContent)


@pytest.mark.asyncio
async def test_reference_without_selected_source_does_not_become_plain_text() -> None:
    model = SummaryProviderModel(FunctionModel(lambda *_: ModelResponse(parts=[TextPart("bad")])))
    with pytest.raises(AIError) as failure:
        await model.request([ModelRequest(parts=[UserPromptPart([_content()])])], None,
                            ModelRequestParameters())
    assert failure.value.code is ErrorCode.CAPABILITY_REQUIRED_MISSING


@pytest.mark.parametrize("wrapped", (False, True))
def test_reference_source_requires_durable_agent_scope(wrapped: bool) -> None:
    async def load(ctx, content):
        return _BODY

    source = ReferencedInputsCapability(load)
    if wrapped:
        source = WrapperCapability(CombinedCapability([source]), id="files")
    group = RuntimeFeatures()
    with pytest.raises(AIError) as failure:
        group.capability(source, id="files")
    assert failure.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    group.capability(source, id="files", revision=1, scope="agent")


def test_ordinary_binary_keeps_existing_durable_bytes_contract() -> None:
    item = BinaryContent(_BODY, media_type="application/pdf")
    messages = [ModelRequest(parts=[UserPromptPart([item])])]
    raw = encode_model_messages(messages)
    assert base64.b64encode(_BODY) in raw
    assert decode_model_messages(raw)[0].parts[0].content[0].data == _BODY


def test_result_object_snapshot_is_plain_data_not_a_reference_input_carrier() -> None:
    @dataclass
    class Result:
        attachment: ReferencedContent

    payload = encode_tool_return_content(ToolReturn(return_value=Result(_content())))
    restored = decode_tool_return_content(payload)
    assert isinstance(restored, dict)
    attachment = restored["return_value"]["attachment"]
    assert isinstance(attachment, dict)
    assert not isinstance(attachment, ReferencedContent)
    serialized = json.dumps(payload).encode()
    assert _BODY not in serialized and base64.b64encode(_BODY) not in serialized
    assert b"authorized_load" not in serialized


def test_existing_safe_diagnostics_do_not_authorize_raw_error_message_or_details() -> None:
    reason = "referenced_content_lifecycle"
    diagnostics = ErrorDiagnostics.from_exception(ValueError("private"), content_omission_reason=reason)
    error = AIError(ErrorCode.MODEL_API_ERROR, "synthetic-private-body",
                    safe_details={"body": "synthetic-private-body", "status_code": 503},
                    diagnostics=diagnostics)
    projected = model_request_error(error, content_omission_reason=reason)
    assert projected is not error
    assert projected.diagnostics is diagnostics
    assert "synthetic-private-body" not in str(projected)
    assert "synthetic-private-body" not in json.dumps(projected.safe_details)
    assert projected.safe_details["status_code"] == 503


def test_tool_content_recovery_preserves_reference_instead_of_sdk_text_snapshot() -> None:
    value = ToolReturn(return_value={"ready": True}, content=["attachment", _content()], metadata={"ok": True})
    wire = encode_tool_return_content(value)
    restored = decode_tool_return_content(wire)
    assert type(restored) is ToolReturn
    assert restored.content == value.content
    assert isinstance(restored.content[1], ReferencedContent)
    assert restored.return_value == value.return_value and restored.metadata == value.metadata
    assert base64.b64encode(_BODY).decode() not in json.dumps(wire)
    assert decode_tool_return_content(encode_tool_return_content(_content())) == _content()
    for value in ({"nested": [_content()]}, [[_content()]]):
        with pytest.raises(AIError):
            encode_tool_return_content(ToolReturn(return_value=value))


@pytest.mark.asyncio
@pytest.mark.parametrize("changed", ["id", "revision"])
async def test_imported_unconsumed_reference_cannot_select_a_different_source(changed: str) -> None:
    calls = []

    async def load(ctx, content):
        calls.append(content)
        return _BODY

    source = CapabilityRef("capability", "other" if changed == "id" else "files",
                           2 if changed == "revision" else 1)
    imported = decode_model_messages(encode_model_messages([
        ModelRequest(parts=[UserPromptPart([_content()])]),
    ]))
    model = SummaryProviderModel(FunctionModel(lambda *_: ModelResponse(parts=[])),
                                 context=_context(), referenced_inputs={source: ReferencedInputsCapability(load)},
                                 binding=_Binding())
    with pytest.raises(AIError) as failure:
        await model.request(list(imported), None, ModelRequestParameters())
    assert failure.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    assert not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ["success", "wrapped", "entry", "iteration", "exit"])
async def test_runtime_retains_refs_and_safe_failure_structure_never_original_bytes(
    tmp_path: Path, boundary: str,
) -> None:
    calls = []

    async def load(ctx, content):
        calls.append((ctx.principal, content))
        return _BODY

    def fail():
        raise ModelHTTPError(503, model_name="synthetic", body=base64.b64encode(_BODY).decode())

    async def stream(messages, info):
        assert any(isinstance(item, BinaryContent) and item.data == _BODY
                   for message in messages if isinstance(message, ModelRequest)
                   for part in message.parts if isinstance(part, UserPromptPart)
                   and not isinstance(part.content, str) for item in part.content)
        if boundary == "entry":
            fail()
        yield "analysis result"
        if boundary == "iteration":
            fail()

    class _Model(FunctionModel):
        @asynccontextmanager
        async def request_stream(self, *args, **kwargs):
            async with super().request_stream(*args, **kwargs) as response:
                yield response
            if boundary == "exit":
                fail()

    class Binding(_Binding):
        contract = {"provider": "test", "model": "references", "input_mime_types": ["application/pdf"]}

        def materialize(self):
            return _Model(stream_function=stream)

    group = RuntimeFeatures("references")
    source = ReferencedInputsCapability(load)
    if boundary == "wrapped":
        source = WrapperCapability(CombinedCapability([source]), id="files")
    group.capability(source, id="files", revision=1, scope="agent")
    group.capability(TraceCapability(record_http_bodies=True), scope="runtime")
    group.agent("default", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('references', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_304, Runtime.open(persistence=persistence_304, features=model_features(Binding(), group)) as runtime:
        execution = await runtime.agents.get().start(["inspect", _content()])
        result = (await execution.wait(timeout_seconds=10)).result
        assert result.status is (ExecutionStatus.SUCCEEDED if boundary in {"success", "wrapped"}
                                 else ExecutionStatus.FAILED)
        assert calls and calls[0][0] == runtime.default_principal
        interactions = (await execution.model_interactions(include_content=True)).items
        assert interactions
        if boundary not in {"success", "wrapped"}:
            assert result.error_code == ErrorCode.MODEL_UNAVAILABLE.value
            assert result.error_diagnostics.content_omission_reason == "referenced_content_lifecycle"
            assert "ModelHTTPError" in result.error_diagnostics.traceback
            assert "fail" in result.error_diagnostics.traceback
    stored = b"\n".join(path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
    assert _BODY not in stored
    assert base64.b64encode(_BODY) not in stored


@pytest.mark.parametrize("extra", [False, True])
@pytest.mark.parametrize("container", ["mapping", "list"])
def test_nested_tool_return_cannot_hide_reference_inputs(extra: bool, container: str) -> None:
    inner = ToolReturn(return_value="summary", content=[_content()]) if extra else ToolReturn(return_value=[_content()])
    nested = {"inner": inner} if container == "mapping" else [inner]
    with pytest.raises(AIError) as failure:
        encode_tool_return_content(nested)
    assert failure.value.code is ErrorCode.REQUEST_FIELD_INVALID
    wire = encode_tool_return_content(inner)
    node = wire["value"]
    wire["value"] = ({"type": "mapping", "items": {"inner": node}} if container == "mapping"
                     else {"type": "sequence", "items": [node]})
    with pytest.raises(AIError) as failure:
        decode_tool_return_content(wire)
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
