#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Authorized batches retain descriptors, producer identity and source ownership."""

import asyncio
from dataclasses import replace

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, ToolReturn, ToolReturnPart, UserPromptPart
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import ReferencedInputResult, ReferencedInputsCapability
from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent, ReferencedInputModel, ReferencedInputOrigin
from linktools.ai.model.messages import (
    bind_referenced_input_origin, decode_model_messages, decode_tool_return_content,
    encode_model_messages, encode_tool_return_content, native_checkpoint_messages,
)
from linktools.ai.runtime._harness import SummaryProviderModel
from linktools.ai.spec import CapabilityRef
from .test_referenced_content import _Binding, _BODY, _SOURCE, _content, _context


def _origin(**changes: object) -> ReferencedInputOrigin:
    context = _context()
    return replace(ReferencedInputOrigin(
        namespace=context.namespace, principal=context.principal,
        execution_id="producer-execution", agent_run_id="producer-run", agent_run_seq=1,
        model_request_seq=2, tool_call_id="producer-call", tool_name="read_original",
    ), **changes)


@pytest.mark.asyncio
async def test_batch_authorizes_once_per_read_preserves_duplicates_and_rechecks_revocation() -> None:
    requested = (_content(), _content(b"second"), _content())
    admitted = True
    reads = []

    async def scalar(context, content):
        pytest.fail("The batch reader owns this read")

    async def batch(context, contents):
        assert context is consumer
        reads.append(contents)
        if not admitted:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return [ReferencedInputResult(content, _BODY if content.ref == requested[0].ref else b"second")
                for content in contents]

    consumer = _context()
    reader = ReferencedInputsCapability(scalar, authorized_load_many=batch)
    assert await reader.load_many(consumer, requested) == (_BODY, b"second", _BODY)
    admitted = False
    with pytest.raises(AIError) as denied:
        await reader.load_many(consumer, requested)
    assert denied.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert reads == [requested, requested]


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["cardinality", "order", "mime", "ref", "bytes", "mutation"])
async def test_batch_rejects_partial_reordered_or_substituted_results(invalid: str) -> None:
    expected = (_content(), _content(b"different"))

    async def scalar(context, content):
        return _BODY

    async def batch(context, contents):
        results = [ReferencedInputResult(contents[0], _BODY), ReferencedInputResult(contents[1], b"different")]
        if invalid == "cardinality":
            return results[:1]
        if invalid == "order":
            return list(reversed(results))
        if invalid == "mime":
            results[0] = ReferencedInputResult(ReferencedContent(contents[0].ref, media_type="image/png", source=_SOURCE), _BODY)
        if invalid == "ref":
            results[0] = ReferencedInputResult(contents[1], _BODY)
        if invalid == "bytes":
            results[0] = ReferencedInputResult(contents[0], b"x")
        if invalid == "mutation":
            contents[0].ref = contents[1].ref
            results[0] = ReferencedInputResult(contents[0], b"different")
        return results

    with pytest.raises(AIError) as failed:
        await ReferencedInputsCapability(scalar, authorized_load_many=batch).load_many(_context(), expected)
    assert failed.value.code is ErrorCode.ASSET_CONTENT_MISMATCH
    assert expected[0] == _content()


@pytest.mark.asyncio
async def test_total_batch_budget_and_mixed_sources_fail_before_source_io() -> None:
    reads = []

    async def scalar(context, content):
        reads.append(content)
        return _BODY

    reader = ReferencedInputsCapability(scalar, limits=PromptLimits(max_binary_input_bytes=len(_BODY)))
    with pytest.raises(AIError) as oversized:
        await reader.load_many(_context(), (_content(), _content()))
    assert oversized.value.code is ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE
    other = ReferencedContent(_content().ref, media_type="application/pdf", source=CapabilityRef("capability", "other", 1))
    with pytest.raises(AIError) as mixed:
        await ReferencedInputsCapability(scalar).load_many(_context(), (_content(), other))
    assert mixed.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID
    assert not reads


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["success", "failure", "cancel"])
async def test_batch_download_owner_cleans_up_before_return_or_failure(tmp_path, failure: str) -> None:
    path = tmp_path / "owned-download"

    async def scalar(context, content):
        pytest.fail("No scalar fallback")

    async def batch(context, contents):
        try:
            path.write_bytes(_BODY)
            if failure == "failure":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if failure == "cancel":
                raise asyncio.CancelledError
            return [ReferencedInputResult(contents[0], path.read_bytes())]
        finally:
            path.unlink()

    operation = ReferencedInputsCapability(scalar, authorized_load_many=batch).load_many(_context(), (_content(),))
    if failure == "success":
        assert await operation == (_BODY,)
    else:
        with pytest.raises(asyncio.CancelledError if failure == "cancel" else AIError):
            await operation
    assert not path.exists()


def test_tool_producer_round_trips_in_result_history_and_native_checkpoint() -> None:
    origin = _origin()
    original = _content()
    value = bind_referenced_input_origin(ToolReturn(return_value=[original], content=[original]), origin)
    restored = decode_tool_return_content(encode_tool_return_content(value))
    assert restored.return_value[0].origin == origin == restored.content[0].origin
    messages = [ModelRequest(parts=[ToolReturnPart("read_original", restored.return_value, tool_call_id="producer-call")])]
    reopened = decode_model_messages(encode_model_messages(messages))
    assert reopened[0].parts[0].content[0].origin == origin
    assert native_checkpoint_messages(native_checkpoint_messages(messages), restore=True)[0].parts[0].content[0].origin == origin
    assert original.origin is None
    assert bind_referenced_input_origin(value, _origin(tool_call_id="forwarder")).return_value[0].origin == origin


@pytest.mark.asyncio
async def test_provider_batches_different_calls_in_same_admission_without_impersonating_producer() -> None:
    consumer = _context()
    initial = _content()
    origin = _origin()
    historical = _content(b"historical").with_origin(origin)
    second_origin = _origin(model_request_seq=3, tool_call_id="second-call", tool_name="other_tool")
    second = _content().with_origin(second_origin)
    other_origin = _origin(agent_run_id="other-run")
    other = _content().with_origin(other_origin)
    reads = []

    async def scalar(context, content):
        pytest.fail("No scalar fallback")

    async def batch(context, contents):
        assert context is consumer and context.execution_id != origin.execution_id
        reads.append(contents)
        # The host reauthorizes the consumer against the original producer's
        # trusted admission. Descriptor provenance alone is not a grant.
        assert all(value.origin in (None, origin, second_origin, other_origin) for value in contents)
        return [ReferencedInputResult(value, b"historical" if value.ref == historical.ref else _BODY)
                for value in contents]

    async def respond(messages, info):
        values = [item for message in messages for part in message.parts
                  for item in (part.content if isinstance(part.content, list) else [])]
        assert [item.data for item in values if isinstance(item, BinaryContent)] == [_BODY, b"historical", _BODY, _BODY]
        return ModelResponse(parts=[TextPart("done")])

    model = SummaryProviderModel(FunctionModel(respond), context=consumer, binding=_Binding(),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(scalar, authorized_load_many=batch)})
    messages = [ModelRequest(parts=[UserPromptPart([initial]), ToolReturnPart("read_original", [historical], tool_call_id=origin.tool_call_id),
                                   ToolReturnPart("other_tool", [second], tool_call_id=second_origin.tool_call_id),
                                   ToolReturnPart("read_original", [other], tool_call_id=other_origin.tool_call_id)])]
    encoded = encode_model_messages(messages)
    await model.request(messages, None, ModelRequestParameters())
    assert reads == [(initial,), (historical, second), (other,)]
    assert encode_model_messages(messages) == encoded


@pytest.mark.asyncio
async def test_unknown_tool_producer_and_unselected_source_never_enter_loader() -> None:
    async def load(context, content):
        pytest.fail("Unknown inputs cannot enter the loader")

    model = SummaryProviderModel(FunctionModel(lambda *_: ModelResponse(parts=[TextPart("unexpected")])),
                                 context=_context(), binding=_Binding(),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(load)})
    for reference, code in ((_content(), ErrorCode.AUTHORIZATION_DENIED),
                            (ReferencedContent(_content().ref, media_type="application/pdf",
                                               source=CapabilityRef("capability", "unknown", 1)), ErrorCode.CAPABILITY_RESOLUTION_INVALID)):
        with pytest.raises(AIError) as failed:
            await model.request([ModelRequest(parts=[ToolReturnPart("read", [reference], tool_call_id="unknown")])], None, ModelRequestParameters())
        assert failed.value.code is code


@pytest.mark.asyncio
async def test_native_model_optional_batch_loader_checks_cardinality_before_provider() -> None:
    async def load(content):
        pytest.fail("No scalar fallback")

    async def batch(contents):
        return ()

    model = ReferencedInputModel(FunctionModel(lambda *_: pytest.fail("No provider dispatch")),
                                 load=load, load_many=batch, binding=_Binding())
    with pytest.raises(AIError) as failed:
        await model.request([ModelRequest(parts=[UserPromptPart([_content()])])], None, ModelRequestParameters())
    assert failed.value.code is ErrorCode.ASSET_CONTENT_MISMATCH


@pytest.mark.asyncio
async def test_batch_only_reader_supports_scalar_without_duplicate_authorization() -> None:
    calls = []

    async def batch(context, contents):
        calls.append(contents)
        return [ReferencedInputResult(value, _BODY) for value in contents]

    reader = ReferencedInputsCapability(authorized_load_many=batch)
    assert await reader.load(_context(), _content()) == _BODY
    assert calls == [(_content(),)]


@pytest.mark.asyncio
async def test_grouping_by_producer_cannot_evade_the_complete_source_budget() -> None:
    async def batch(context, contents):
        pytest.fail("The whole source budget must be checked before any source IO")

    reader = ReferencedInputsCapability(authorized_load_many=batch,
                                        limits=PromptLimits(max_binary_input_bytes=len(_BODY)))
    model = SummaryProviderModel(FunctionModel(lambda *_: pytest.fail("No provider dispatch")),
                                 context=_context(), binding=_Binding(), referenced_inputs={_SOURCE: reader})
    messages = [ModelRequest(parts=[UserPromptPart([_content(), _content().with_origin(_origin())])])]
    with pytest.raises(AIError) as failed:
        await model.request(messages, None, ModelRequestParameters())
    assert failed.value.code is ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE


@pytest.mark.asyncio
async def test_claimed_producer_does_not_grant_cross_execution_access() -> None:
    content = _content().with_origin(_origin())
    consumers = []

    async def batch(context, contents):
        consumers.append(context.execution_id)
        if any(value.origin.execution_id != context.execution_id for value in contents):
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        return [ReferencedInputResult(value, _BODY) for value in contents]

    model = SummaryProviderModel(FunctionModel(lambda *_: pytest.fail("No provider dispatch")),
                                 context=_context(), binding=_Binding(),
                                 referenced_inputs={_SOURCE: ReferencedInputsCapability(authorized_load_many=batch)})
    with pytest.raises(AIError) as failed:
        await model.request([ModelRequest(parts=[UserPromptPart([content])])], None, ModelRequestParameters())
    assert failed.value.code is ErrorCode.AUTHORIZATION_DENIED
    assert consumers == ["execution"]


def test_binding_producer_preserves_inherited_content_and_custom_descriptor_type() -> None:
    class CustomContent(ReferencedContent):
        def __init__(self):
            content = _content()
            super().__init__(content.ref, media_type=content.media_type, source=content.source)

    reference = CustomContent()
    reference.content = "existing user-visible text"
    reference.metadata = {"display": "existing metadata"}
    bound = reference.with_origin(_origin())
    assert type(bound) is CustomContent and bound is not reference
    assert bound.content == reference.content and bound.metadata == reference.metadata
    assert bound.ref == reference.ref and bound.source == reference.source
    assert reference.origin is None and bound.origin == _origin()


@pytest.mark.asyncio
async def test_materialized_native_capability_uses_the_same_ordered_batch_boundary() -> None:
    from pydantic_ai import Agent
    from linktools.ai.capability import CapabilityContribution

    calls = []

    async def batch(context, contents):
        calls.append(contents)
        return [ReferencedInputResult(value, _BODY) for value in contents]

    def respond(messages, info):
        values = messages[0].parts[-1].content
        assert [item.data for item in values if isinstance(item, BinaryContent)] == [_BODY, _BODY]
        return ModelResponse(parts=[TextPart("done")])

    class InputModel(FunctionModel):
        def validate_inputs(self, value):
            assert isinstance(value, list)

    capability = CapabilityContribution.from_opaque(
        "capability", _SOURCE.id, ReferencedInputsCapability(authorized_load_many=batch),
        revision=_SOURCE.revision,
    ).materialize_capability()
    result = await Agent(InputModel(respond), capabilities=[capability]).run(
        [_content(), _content()], deps=_context(),
    )
    assert result.output == "done" and calls == [(_content(), _content())]
