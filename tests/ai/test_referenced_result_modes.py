#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from linktools.ai.task import Task
from linktools.ai.task_engine import TaskEngine

from linktools.ai.persistence import Persistence
"""Reference representation and exact selected-source routing contracts."""

import asyncio
import base64
import hashlib
import json
from types import SimpleNamespace

import pytest
from pydantic_ai.capabilities import CombinedCapability
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, ToolReturn, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.capability import AgentContext, ReferencedInputsCapability
from linktools.ai.core import ExecutionStatus, service_principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime import Runtime, RuntimeFeatures
from linktools.ai.persistence import PersistenceStorage
from linktools.ai.runtime._harness import SummaryProviderModel
from linktools.ai.inputs import decode_task_prompt_draft, task_prompt_draft
from linktools.ai.model.messages import decode_model_messages, encode_model_messages, project_transient_binary_content
from linktools.ai.model._tool_return_codec import decode_tool_return_content, encode_tool_return_content
from linktools.ai.model._user_content_codec import decode_user_content_item, encode_user_content_item
from linktools.ai.persistence._views import project_session_history_message
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

_BODY = '{"synthetic": "正文"}'.encode()
_SOURCE = CapabilityRef("capability", "text-inputs", 3)
_OTHER = CapabilityRef("capability", "other-inputs", 1)


def _reference(*, body: bytes = _BODY, mime: str = "application/json", representation: str = "text",
               source: CapabilityRef = _SOURCE, source_parameters=None) -> ReferencedContent:
    return ReferencedContent(
        AssetVersionRef(AssetKey("document", "synthetic-text"), "configured", StorageEntryRevision(1),
                        hashlib.sha256(body).hexdigest(), len(body)),
        media_type=mime, source=source, representation=representation, source_parameters=source_parameters,
    )


def _context() -> AgentContext[None]:
    return AgentContext(None, service_principal("tenant", "admitted-owner"), "references", None, "run", {})


def _binding(profile: list[str]) -> object:
    return SimpleNamespace(provider="test", model_identity="test:references", contract={"input_mime_types": profile})


def test_representation_survives_all_reference_codecs_and_consumed_summary() -> None:
    reference = _reference()
    prompt = decode_task_prompt_draft(task_prompt_draft([reference]))
    messages = [ModelRequest(parts=[UserPromptPart(prompt)])]
    raw = encode_model_messages(messages)
    restored = decode_model_messages(raw)
    assert restored[0].parts[0].content[0] == reference
    content = decode_tool_return_content(encode_tool_return_content(ToolReturn(return_value="summary", content=[reference])))
    assert content.content == [reference]
    view, = project_session_history_message(restored[0])
    assert view.content[0]["representation"] == "text"
    consumed = project_transient_binary_content([*restored, ModelResponse(parts=[TextPart("done")])])
    summary = consumed[0].parts[0].content[0]
    assert json.loads(summary.content)["representation"] == "text"
    assert decode_user_content_item(encode_user_content_item(summary)["metadata"]["referenced_content"]) == reference
    serialized = raw + json.dumps(view.content).encode()
    assert _BODY not in serialized and base64.b64encode(_BODY) not in serialized
    # Omission is the explicit binary default, not MIME-based inference.
    binary = _reference(representation="binary", mime="text/plain")
    payload = encode_user_content_item(binary)
    assert set(payload) == {"kind", "ref", "source", "media_type"}
    assert decode_user_content_item(payload).representation == "binary"
    invalid = {**encode_user_content_item(reference), "representation": "guess"}
    with pytest.raises(ValueError):
        decode_user_content_item(invalid)
    with pytest.raises(ValueError):
        _reference(representation="guess")


@pytest.mark.asyncio
@pytest.mark.parametrize("profile", ([], ["image/png"]))
@pytest.mark.parametrize("mime", ("text/plain", "application/json", "text/csv"))
async def test_explicit_text_reference_needs_no_attachment_profile(profile: list[str], mime: str) -> None:
    context, loaded = _context(), []
    reference = _reference(mime=mime)
    original = [ModelRequest(parts=[UserPromptPart([reference])])]
    encoded = encode_model_messages(original)

    async def load(ctx, content):
        assert ctx is context and content == reference
        loaded.append(content)
        return _BODY

    async def respond(messages, info):
        assert messages[0].parts[0].content == [_BODY.decode("utf-8")]
        assert type(messages[0].parts[0].content[0]) is str
        return ModelResponse(parts=[TextPart("done")])

    model = SummaryProviderModel(FunctionModel(respond), context=context, binding=_binding(profile),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    await model.request(original, None, ModelRequestParameters())
    assert loaded == [reference] and encode_model_messages(original) == encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("allowed", (False, True))
async def test_default_binary_text_document_keeps_attachment_admission(allowed: bool) -> None:
    calls = []
    reference = _reference(mime="text/plain", representation="binary")

    async def load(ctx, content):
        calls.append("load")
        return _BODY

    async def respond(messages, info):
        item = messages[0].parts[0].content[0]
        assert type(item) is BinaryContent and item.data == _BODY
        calls.append("provider")
        return ModelResponse(parts=[TextPart("done")])

    model = SummaryProviderModel(FunctionModel(respond), context=_context(),
                                 binding=_binding(["text/plain"] if allowed else []),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    if allowed:
        await model.request([ModelRequest(parts=[UserPromptPart([reference])])], None, ModelRequestParameters())
        assert calls == ["load", "provider"]
    else:
        with pytest.raises(AIError) as failure:
            await model.request([ModelRequest(parts=[UserPromptPart([reference])])], None, ModelRequestParameters())
        assert failure.value.code is ErrorCode.REQUEST_FIELD_INVALID and not calls


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("utf8", "changed", "denied", "cancelled"))
async def test_text_load_failure_never_dispatches_or_rewrites_the_reference(failure: str) -> None:
    body = b"\xffinvalid-utf8" if failure == "utf8" else _BODY
    reference = _reference(body=body)
    original = [ModelRequest(parts=[UserPromptPart([reference])])]
    encoded = encode_model_messages(original)
    calls = []

    async def load(ctx, content):
        calls.append("load")
        if failure == "denied":
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if failure == "cancelled":
            raise asyncio.CancelledError()
        return b"changed" if failure == "changed" else body

    async def respond(messages, info):
        calls.append("provider")
        return ModelResponse(parts=[])

    model = SummaryProviderModel(FunctionModel(respond), context=_context(), binding=_binding([]),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    with pytest.raises(asyncio.CancelledError if failure == "cancelled" else AIError) as raised:
        await model.request(original, None, ModelRequestParameters())
    if failure != "cancelled":
        expected = {"utf8": ErrorCode.REQUEST_FIELD_INVALID, "changed": ErrorCode.ASSET_CONTENT_MISMATCH,
                    "denied": ErrorCode.AUTHORIZATION_DENIED}[failure]
        assert raised.value.code is expected
    assert calls == ["load"] and encode_model_messages(original) == encoded


@pytest.mark.asyncio
@pytest.mark.parametrize("missing", (False, True))
async def test_each_reference_uses_its_exact_selected_source_without_fallback(missing: bool) -> None:
    context, calls = _context(), []
    first, second = _reference(), _reference(source=_OTHER, mime="text/csv")

    async def load_first(ctx, content):
        assert ctx is context and content.source == _SOURCE
        calls.append("first")
        return _BODY

    async def load_second(ctx, content):
        assert ctx is context and content.source == _OTHER
        calls.append("second")
        return _BODY

    async def respond(messages, info):
        assert messages[0].parts[0].content == [_BODY.decode(), _BODY.decode()]
        calls.append("provider")
        return ModelResponse(parts=[TextPart("done")])

    sources = {_SOURCE: ReferencedInputsCapability(load_first)}
    if not missing:
        sources[_OTHER] = ReferencedInputsCapability(load_second)
    model = SummaryProviderModel(FunctionModel(respond), context=context, binding=_binding([]),
                                 referenced_inputs=sources)
    messages = [ModelRequest(parts=[UserPromptPart([first, second])])]
    if missing:
        with pytest.raises(AIError) as failure:
            await model.request(messages, None, ModelRequestParameters())
        assert failure.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
        assert not calls  # Validate all sources before loading any one of them.
    else:
        await model.request(messages, None, ModelRequestParameters())
        assert calls == ["first", "second", "provider"]


@pytest.mark.asyncio
async def test_multiple_loaders_in_one_contribution_are_ambiguous() -> None:
    calls = []

    async def load(ctx, content):
        calls.append("load")
        return _BODY

    async def respond(messages, info):
        calls.append("provider")
        return ModelResponse(parts=[TextPart("unexpected")])

    class Binding:
        provider = "test"
        model_identity = "test:references"
        contract = {"input_mime_types": []}

        def materialize(self):
            return FunctionModel(respond)

    features = RuntimeFeatures("ambiguous-references")
    features.model("default", Binding())
    features.capability(CombinedCapability([ReferencedInputsCapability(load), ReferencedInputsCapability(load)]),
                        id="ambiguous", scope="agent")
    features.agent("reader", allow_tools=(), allow_skills=(), allow_subagents=())
    async with Persistence.open('ambiguous-references', storage=PersistenceStorage.in_memory()) as persistence_230, Runtime.attach(persistence=persistence_230, features=features) as runtime:
        result = (await runtime.agents.get("reader").run("inspect", timeout_seconds=10)).result
        assert result.status is ExecutionStatus.FAILED
        assert result.error_code == ErrorCode.CAPABILITY_CONFLICT.value
    assert not calls


def test_text_reference_budget_counts_original_bytes_not_summary_length() -> None:
    from linktools.ai.core import PromptLimits
    from linktools.ai.runtime._compaction import _validate_pending_binary_content

    reference = _reference(body=b"a" * 4096)
    messages = [ModelRequest(parts=[UserPromptPart([reference])])]
    with pytest.raises(AIError) as failure:
        _validate_pending_binary_content(messages, PromptLimits(max_binary_input_bytes=4095))
    assert failure.value.code is ErrorCode.REQUEST_FIELD_INVALID
    _validate_pending_binary_content(messages, PromptLimits(max_binary_input_bytes=4096))


@pytest.mark.asyncio
async def test_two_sources_are_selected_and_restored_after_runtime_reopen(tmp_path) -> None:
    from linktools.ai.inputs import AgentTaskInput, CaptureInputRequest
    from linktools.ai.task import TaskGraph, TaskNode
    from .test_tool_result_processing import _stored_bytes
    from ._runtime_test_helpers import _UsageFunctionModel

    first, second = _reference(), _reference(source=_OTHER, mime="text/csv")
    principal = service_principal("default", "admitted-reference-owner")
    calls = []

    async def respond(messages, info):
        values = [item for message in messages for part in message.parts
                  if isinstance(part, UserPromptPart) and not isinstance(part.content, str)
                  for item in part.content]
        assert values == [_BODY.decode(), _BODY.decode()]
        return ModelResponse(parts=[TextPart("done")])

    def features(host: str, revision: int = _OTHER.revision) -> RuntimeFeatures:
        async def load_first(context, content):
            assert context.principal == principal and content.source == _SOURCE
            calls.append((host, "first"))
            return _BODY

        async def load_second(context, content):
            assert context.principal == principal and content.source == _OTHER
            calls.append((host, "second"))
            return _BODY

        result = RuntimeFeatures("two-source-reopen")
        result.model("default", SimpleNamespace(provider="test", model_identity="test:two-sources",
                     contract={"input_mime_types": []}, materialize=lambda: _UsageFunctionModel(respond)))
        result.capability(ReferencedInputsCapability(load_first), id=_SOURCE.id,
                          revision=_SOURCE.revision, scope="agent")
        result.capability(ReferencedInputsCapability(load_second), id=_OTHER.id,
                          revision=revision, scope="agent")
        result.agent("reader", allow_tools=(), allow_skills=(), allow_subagents=())
        return result

    async with Persistence.open('two-source-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_288, Runtime.attach(persistence=persistence_288, features=features('original')) as runtime, TaskEngine.open(persistence=persistence_288, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        execution = await runtime.agents.get("reader").start([first, second], principal=principal)
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
        capture = await runtime_engine.inputs.capture_input(
            execution.execution_id, CaptureInputRequest(principal, "two-source-capture", "clean"),
        )
    async with Persistence.open('two-source-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_295, Runtime.attach(persistence=persistence_295, features=features('reopened')) as runtime, TaskEngine.open(persistence=persistence_295, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        task_capture = await runtime_engine.inputs.read_agent(capture, principal=principal)
        task_capture.require_importable()
        task = Task.from_agent("captured-sources", runtime.agents.restore(task_capture.binding))
        graph = await runtime_engine.with_definitions(task).start(TaskGraph("captured-source-graph", (
            TaskNode("read", task=task, input=AgentTaskInput([first, second])),
        )), principal=principal, idempotency_key="captured-source-graph")
        await graph.wait(timeout_seconds=15)
        execution = await graph.execution("read")
        assert (await execution.wait(timeout_seconds=10)).result.status is ExecutionStatus.SUCCEEDED
    assert calls == [("original", "first"), ("original", "second"), ("reopened", "first"), ("reopened", "second")]
    async with Persistence.open('two-source-reopen', storage=PersistenceStorage.filesystem(tmp_path)) as persistence_305, Runtime.attach(persistence=persistence_305, features=features('changed', revision=2)) as runtime, TaskEngine.open(persistence=persistence_305, context=runtime.context, authorization=runtime._execution_service._authorization) as runtime_engine:
        with pytest.raises(AIError) as unavailable:
            Task.from_agent("unavailable-sources", runtime.agents.restore((await runtime_engine.inputs.read_agent(capture, principal=principal)).binding))
        assert unavailable.value.code is ErrorCode.AGENT_BINDING_UNAVAILABLE
    stored = _stored_bytes(tmp_path)
    assert _BODY not in stored and base64.b64encode(_BODY) not in stored


def test_source_parameters_are_immutable_durable_and_absent_from_default_summary_and_repr() -> None:
    long_member = "synthetic-member/" * 60
    parameters = {"parent": {"version": "immutable"}, "member": long_member, "query": ["term", {"page": 2}]}
    reference = _reference(source_parameters=parameters)
    parameters["parent"]["version"] = "changed"
    borrowed = reference.source_parameters["query"]
    borrowed[1]["page"] = 99
    assert reference.source_parameters["parent"] == {"version": "immutable"}
    assert reference.source_parameters["query"] == ["term", {"page": 2}]
    with pytest.raises(TypeError):
        reference.source_parameters["member"] = "changed"
    assert long_member not in reference.content and long_member not in repr(reference)
    payload = encode_user_content_item(reference)
    restored = decode_user_content_item(payload)
    assert restored == reference and restored.ref == reference.ref
    assert payload["source_parameters"]["member"] == long_member
    messages = [ModelRequest(parts=[UserPromptPart([reference])]), ModelResponse(parts=[TextPart("done")])]
    summary = project_transient_binary_content(messages)[0].parts[0].content[0]
    assert long_member not in summary.content
    assert decode_user_content_item(encode_user_content_item(summary)["metadata"]["referenced_content"]) == reference
    assert decode_task_prompt_draft(task_prompt_draft([reference])) == (reference,)
    content = decode_tool_return_content(encode_tool_return_content(ToolReturn(return_value="summary", content=[reference])))
    assert content.content == [reference]
    for invalid in ({"invalid": b"bytes"}, {"invalid": float("nan")}):
        with pytest.raises((TypeError, ValueError)):
            _reference(source_parameters=invalid)
    with pytest.raises(ValueError):
        decode_user_content_item({**payload, "source_parameters": None})


@pytest.mark.asyncio
async def test_same_resource_with_different_source_parameters_is_not_collapsed() -> None:
    first = _reference(source_parameters={"selection": {"page": 1}})
    second = _reference(source_parameters={"selection": {"page": 2}})
    assert first.ref == second.ref and first != second
    assert encode_user_content_item(first) != encode_user_content_item(second)
    calls = []

    async def load(context, content):
        calls.append(content.source_parameters["selection"]["page"])
        return _BODY

    async def respond(messages, info):
        assert messages[0].parts[0].content == [_BODY.decode(), _BODY.decode()]
        return ModelResponse(parts=[TextPart("done")])

    model = SummaryProviderModel(FunctionModel(respond), context=_context(), binding=_binding([]),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    await model.request([ModelRequest(parts=[UserPromptPart([first, second])])], None, ModelRequestParameters())
    assert calls == [1, 2]


def test_local_text_budget_uses_byte_estimate_only_beyond_provider_usage_anchor() -> None:
    from pydantic_ai.usage import RequestUsage
    from linktools.ai.core import PromptLimits
    from linktools.ai.runtime._compaction import CompactionCapability

    capability = CompactionCapability(10000, limits=PromptLimits())
    reference = _reference(body=b"x" * 4096)
    request = ModelRequest(parts=[UserPromptPart([reference])])
    parameters = ModelRequestParameters()

    def target(messages):
        return capability._request_target(SimpleNamespace(
            model=SimpleNamespace(context_window=None), messages=messages,
            model_request_parameters=parameters, model_settings=None,
        ))

    growth = (reference.ref.size - len(reference.content)) // 4
    assert target([request]) == 10000 - growth
    anchor = ModelResponse(parts=[TextPart("done")], usage=RequestUsage(input_tokens=2000, output_tokens=10))
    assert target([request, anchor]) == 10000
    assert target([anchor, request]) == 10000 - growth
    binary = _reference(body=b"x" * 4096, representation="binary")
    assert target([ModelRequest(parts=[UserPromptPart([binary])])]) == 10000
    small = _reference(body=b"x")
    assert target([ModelRequest(parts=[UserPromptPart([small])])]) == 10000


@pytest.mark.asyncio
async def test_provider_token_count_receives_real_text_not_descriptor() -> None:
    from pydantic_ai.usage import RequestUsage

    class CountModel(FunctionModel):
        async def count_tokens(self, messages, model_settings, model_request_parameters):
            assert messages[0].parts[0].content == [_BODY.decode()]
            return RequestUsage(input_tokens=123)

    async def load(context, content):
        return _BODY

    model = SummaryProviderModel(CountModel(lambda *_: ModelResponse(parts=[])), context=_context(),
                                 binding=_binding([]), referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    result = await model.count_tokens([ModelRequest(parts=[UserPromptPart([_reference()])])],
                                      None, ModelRequestParameters())
    assert result.input_tokens == 123


@pytest.mark.asyncio
async def test_large_text_estimate_still_dispatches_once_without_placeholder_or_retry() -> None:
    from dataclasses import replace
    from pydantic_ai.models.test import TestModel
    from .test_model_context_compaction import _project

    reference = _reference()
    reference.ref = replace(reference.ref, size=1_000_000)
    original = [ModelRequest(parts=[UserPromptPart([reference])])]
    # Compaction estimates descriptor sizes without authorizing or loading bytes.
    projected = await asyncio.wait_for(
        _project(TestModel(profile={"context_window": 200}), messages=original), 5,
    )
    assert projected == original
