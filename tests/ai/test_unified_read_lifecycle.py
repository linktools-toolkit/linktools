#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unified read results preserve attachment identity, lifetime and text coverage."""

import hashlib
from copy import deepcopy

import pytest
from pydantic_ai import Agent as PydanticAgent
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    TextPart,
    ToolCallPart,
    ToolReturn,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_ai.models.test import TestModel
from pydantic_ai.tools import RunContext
from pydantic_ai.usage import RunUsage

from linktools.ai.core import PromptLimits, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model._attachment import bind_tool_return_attachments, request_attachment_facts
from linktools.ai.runtime._compaction import CompactionCapability, CompactionPolicy
from linktools.ai.model.messages import binary_content_usage, decode_model_messages, encode_model_messages, project_transient_binary_content


def _text(**changes: object) -> dict[str, object]:
    return {
        "path": "note.txt", "input_index": 0, "content": "evidence\n",
        "offset": 0, "end_offset": 1, "byte_offset": 0, "end_byte_offset": 9,
        "line_offset": 0, "end_line_offset": 0, "lossy": False,
        "truncated": False, "eof": True, "next_cursor": None, "size": 9,
        **changes,
    }


def _result(
    *, duplicate: bool = False, legacy: bool = False, media_type: str = "image/png",
) -> ToolReturn:
    body = b"image"
    file = {
        "path": "evidence.png", ("media_type" if legacy else "mime_type"): media_type, "size": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
    }
    files = [dict(file)]
    if duplicate:
        files.append(dict(file))
    if not legacy:
        for index, item in enumerate(files):
            item["input_index"] = index + 1
    value = {"files": files}
    if not legacy:
        value["texts"] = [_text()]
    return ToolReturn(
        return_value=value,
        content=[BinaryContent(body, media_type=media_type) for _ in files],
    )


def _request(result: ToolReturn, call_id: str, tool: str = "read_file") -> ModelRequest:
    return ModelRequest(parts=[
        ToolReturnPart(tool, result.return_value, tool_call_id=call_id),
        UserPromptPart(content=result.content or []),
    ])


def _bind(result: ToolReturn, call_id: str, tool: str = "read_file") -> ToolReturn:
    bound = bind_tool_return_attachments(tool, call_id, result)
    assert isinstance(bound, ToolReturn)
    return bound


def test_unified_read_binds_only_files_with_distinct_stable_occurrences() -> None:
    original = _result(duplicate=True)
    raw = deepcopy(original)
    bound = _bind(original, "read-call")
    retried = _bind(original, "read-call")
    repeated = _bind(original, "read-again")
    facts = request_attachment_facts(
        [_request(bound, "read-call")], (), accepted_attachment_ids=set(),
    )

    assert original == raw
    ids = [item.identifier for item in bound.content if isinstance(item, BinaryContent)]
    assert bound.return_value == {
        **original.return_value,
        "files": [
            {**file, "attachment": identifier}
            for file, identifier in zip(original.return_value["files"], ids)
        ],
    }
    assert ids == [item.identifier for item in retried.content if isinstance(item, BinaryContent)]
    assert ids != [item.identifier for item in repeated.content if isinstance(item, BinaryContent)]
    accepted = [fact for fact in facts if fact["fact"] == "accepted"]
    included = [fact for fact in facts if fact["fact"] == "included_in_request"]
    assert len(accepted) == len(included) == 2
    assert {fact["source"] for fact in facts} == {"read_file"}
    assert {fact["call_id"] for fact in facts} == {"read-call"}
    assert [fact["position"] for fact in accepted] == [0, 1]
    assert len({fact["attachment_id"] for fact in accepted}) == 2
    assert [fact["attachment_id"] for fact in accepted] == [
        fact["attachment_id"] for fact in included
    ]


def test_legacy_attach_files_identity_and_read_strings_are_unchanged() -> None:
    result = _result(legacy=True)
    legacy = _bind(result, "old-call", "attach_files")
    assert legacy.return_value == result.return_value
    facts = request_attachment_facts(
        [_request(legacy, "old-call", "attach_files")], (),
        accepted_attachment_ids=set(),
    )
    digest = hashlib.sha256(b"image").hexdigest()
    expected = canonical_sha256({
        "contract": "attachment-occurrence-v1", "source": "attach_files",
        "position": 0, "content_key": digest, "call_id": "old-call",
    })
    assert {fact["attachment_id"] for fact in facts} == {expected}
    assert {fact["source"] for fact in facts} == {"attach_files"}
    assert {fact["position"] for fact in facts} == {0}
    current = _bind(_result(), "old-call")
    current_facts = request_attachment_facts(
        [_request(current, "old-call")], (), accepted_attachment_ids=set(),
    )
    assert current_facts[0]["attachment_id"] != expected
    assert bind_tool_return_attachments("read_file", "text-call", "old text") == "old text"
    assert request_attachment_facts(
        [ModelRequest(parts=[ToolReturnPart("read_file", "old text", "text-call")])],
        (), accepted_attachment_ids=set(),
    ) == ()


@pytest.mark.parametrize("tool", ["read_file", "attach_files"])
def test_stored_file_tool_media_type_keeps_identity_and_facts(tool: str) -> None:
    current = _result()
    legacy = deepcopy(current)
    file = legacy.return_value["files"][0]
    file["media_type"] = file.pop("mime_type")
    bound_current = _bind(current, "read-call", tool)
    bound_legacy = _bind(legacy, "read-call", tool)

    assert bound_current.content[0].identifier == bound_legacy.content[0].identifier
    assert "media_type" not in bound_current.return_value["files"][0]
    assert "mime_type" not in bound_legacy.return_value["files"][0]
    assert request_attachment_facts(
        [_request(bound_current, "read-call", tool)], (), accepted_attachment_ids=set(),
    ) == request_attachment_facts(
        [_request(bound_legacy, "read-call", tool)], (), accepted_attachment_ids=set(),
    )
    history = [_request(bound_legacy, "read-call", tool)]
    encoded = encode_model_messages(history)
    assert encode_model_messages(decode_model_messages(encoded)) == encoded

    conflicting = deepcopy(current)
    conflicting.return_value["files"][0]["media_type"] = "application/pdf"
    with pytest.raises(AIError) as error:
        _bind(conflicting, "read-call", tool)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("value", [
    {"files": [], "texts": [_text()]},
    {"error": {"code": "PERMISSION_DENIED"}},
])
def test_text_only_reads_and_tool_errors_do_not_create_attachment_facts(value: object) -> None:
    original = ToolReturn(return_value=value)
    bound = _bind(original, "text-call")
    assert bound == original
    assert request_attachment_facts(
        [_request(bound, "text-call")], (), accepted_attachment_ids=set(),
    ) == ()


@pytest.mark.parametrize("corruption", ["digest", "missing_body", "orphan_body", "input_order"])
def test_unified_attachment_binding_rejects_inconsistent_results(corruption: str) -> None:
    result = _result(duplicate=True)
    if corruption == "digest":
        result.return_value["files"][0]["sha256"] = "0" * 64
    elif corruption == "missing_body":
        result.content = []
    elif corruption == "orphan_body":
        result.return_value["files"] = []
    else:
        result.return_value["files"][1]["input_index"] = 1
    with pytest.raises(AIError) as caught:
        _bind(result, "bad-call")
    assert caught.value.code == ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("state", ["incomplete", "suspended", "interrupted"])
def test_unified_read_retry_and_recovery_preserve_pending_attachments(state: str) -> None:
    bound = _bind(_result(), "read-call")
    request = _request(bound, "read-call")
    response = ModelResponse(parts=[TextPart("unfinished")], state=state)
    raw = [request, response]
    recovered = decode_model_messages(encode_model_messages(raw))
    projected = project_transient_binary_content(recovered)
    accepted: set[str] = set()
    first = request_attachment_facts(projected, (), accepted_attachment_ids=accepted)
    retry = request_attachment_facts(projected, (), accepted_attachment_ids=accepted)

    assert binary_content_usage(projected) == (1, 5)
    assert [fact["fact"] for fact in first] == ["accepted", "included_in_request"]
    assert [fact["fact"] for fact in retry] == ["included_in_request"]
    assert first[-1]["attachment_id"] == retry[-1]["attachment_id"]
    assert encode_model_messages(projected) == encode_model_messages(raw)


@pytest.mark.parametrize("media_type", ["image/png", "text/plain"])
def test_unified_read_complete_response_consumes_files_but_preserves_text_and_history(
    media_type: str,
) -> None:
    request = _request(_bind(_result(media_type=media_type), "read-call"), "read-call")
    raw = [request, ModelResponse(parts=[TextPart("done")])]
    before = encode_model_messages(raw)
    projected = project_transient_binary_content(raw)
    assert binary_content_usage(projected) == (0, 0)
    assert binary_content_usage(raw) == (1, 5)
    assert encode_model_messages(raw) == before
    assert projected[0].parts[0].content == request.parts[0].content
    facts = request_attachment_facts(projected, (), accepted_attachment_ids=set())
    assert [fact["fact"] for fact in facts] == ["accepted"]


async def _compact(
    messages: list[ModelMessage], *, policy: str = "workspace_file_read_v1",
    target: int | None = None, limits: PromptLimits | None = None,
) -> list[ModelMessage]:
    model = TestModel()
    context = RunContext(deps=None, model=model, usage=RunUsage(), run_id="run")
    request = ModelRequestContext(
        model=model, messages=messages, model_settings=None,
        model_request_parameters=ModelRequestParameters(),
    )
    captured: list[ModelMessage] = []

    async def handler(current: ModelRequestContext) -> ModelResponse:
        captured.extend(current.messages)
        return ModelResponse(parts=[TextPart("done")])

    capability = CompactionCapability(
        target, limits=limits or PromptLimits(),
        policy=CompactionPolicy(context_dedupe_by_tool={"read_file": policy}),
    )
    await capability.wrap_model_request(context, request_context=request, handler=handler)
    return captured


def _reads(*results: object) -> list[ModelMessage]:
    history: list[ModelMessage] = []
    for index, result in enumerate(results):
        call_id = f"read-{index}"
        history.extend([
            ModelResponse(parts=[ToolCallPart("read_file", {"path": "note.txt"}, call_id)]),
            ModelRequest(parts=[ToolReturnPart("read_file", result, call_id)]),
        ])
    return history


@pytest.mark.asyncio
async def test_dedupe_rejects_unsupported_policy_version() -> None:
    with pytest.raises(AIError) as raised:
        await _compact(_reads(_result().return_value), policy="workspace_file_read_v2")
    assert raised.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_dedupe_preserves_unstructured_results() -> None:
    messages = _reads("first text", "second text")
    original = deepcopy(messages)
    assert await _compact(messages) == original
    assert messages == original


@pytest.mark.asyncio
async def test_unified_dedupe_only_clears_same_version_same_range_text() -> None:
    text = _text(hash="a" * 64, total_lines=1)
    first = _result().return_value
    first["texts"] = [text, _text(path="other.txt", input_index=2)]
    second = {"files": [], "texts": [text]}
    messages = _reads(first, second)
    original = deepcopy(messages)
    projected = await _compact(messages)
    earlier = projected[1].parts[0].content
    assert earlier["texts"][0]["content"] == "[superseded file read]"
    assert earlier["texts"][1] == first["texts"][1]
    assert earlier["files"] == first["files"]
    assert projected[-1].parts[0].content == second
    assert messages == original


@pytest.mark.parametrize("changes", [
    {"hash": None}, {"hash": "b" * 64}, {"byte_offset": 3},
    {"end_byte_offset": 4}, {"line_offset": 2}, {"lossy": True},
])
@pytest.mark.asyncio
async def test_unified_dedupe_preserves_unproven_versions_and_other_ranges(
    changes: dict[str, object],
) -> None:
    first = {"files": [], "texts": [_text(hash="a" * 64)]}
    second = {"files": [], "texts": [_text(hash="a" * 64) | changes]}
    messages = _reads(first, second)
    projected = await _compact(messages)
    assert projected == messages


@pytest.mark.asyncio
async def test_unified_basic_reads_are_not_deduplicated_from_state_identity() -> None:
    result = {"files": [], "texts": [_text(state="same-file-identity")]}
    messages = _reads(result, result)
    assert await _compact(messages) == messages


@pytest.mark.asyncio
async def test_unified_mixed_result_survives_whole_result_clearing() -> None:
    first = _result().return_value
    messages = _reads(first, *({"files": [], "texts": [_text()]} for _ in range(4)))
    original = deepcopy(messages)
    projected = await _compact(messages, target=1)
    assert projected[1].parts[0].content == first
    assert messages == original


@pytest.mark.asyncio
async def test_unified_media_read_keeps_latest_occurrence_through_compaction() -> None:
    first = _bind(_result(), "first")
    second = _bind(_result(), "second")
    messages = [
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "evidence.png"}, "first")]),
        _request(first, "first"),
        ModelResponse(parts=[ToolCallPart("read_file", {"path": "evidence.png"}, "second")]),
        _request(second, "second"),
    ]
    projected = await _compact(messages)
    facts = request_attachment_facts(projected, (), accepted_attachment_ids=set())
    included = [fact for fact in facts if fact["fact"] == "included_in_request"]
    assert len(included) == 1
    assert included[0]["call_id"] == "second"
    assert binary_content_usage(projected) == (1, 5)
    assert binary_content_usage(messages) == (2, 10)


@pytest.mark.asyncio
async def test_unified_parallel_reads_obey_combined_attachment_budget() -> None:
    messages = [_request(_bind(_result(), name), name) for name in ("one", "two")]
    with pytest.raises(AIError) as caught:
        await _compact(messages, limits=PromptLimits(max_binary_input_parts=1))
    assert caught.value.code == ErrorCode.REQUEST_FIELD_INVALID


@pytest.mark.asyncio
async def test_unified_read_sdk_loop_consumes_media_after_one_interaction() -> None:
    observed: list[tuple[int, int]] = []
    facts: list[dict[str, object]] = []
    accepted: set[str] = set()

    async def read_file(ctx: RunContext[None], path: str) -> ToolReturn:
        assert ctx.tool_call_id is not None
        result = (
            _result() if path == "evidence.png"
            else ToolReturn(return_value={"files": [], "texts": [_text()]})
        )
        return _bind(result, ctx.tool_call_id)

    def model(messages: list[ModelMessage], _info: AgentInfo) -> ModelResponse:
        observed.append(binary_content_usage(messages))
        facts.extend(request_attachment_facts(
            messages, (), accepted_attachment_ids=accepted,
        ))
        if len(observed) <= 2:
            path = "evidence.png" if len(observed) == 1 else "note.txt"
            return ModelResponse(parts=[ToolCallPart(
                "read_file", {"path": path}, f"read-{len(observed)}",
            )])
        return ModelResponse(parts=[TextPart("done")])

    agent = PydanticAgent(FunctionModel(model), tools=[read_file])
    result = await agent.run(
        "read the evidence",
        capabilities=[CompactionCapability(
            None, limits=PromptLimits(),
            policy=CompactionPolicy(context_dedupe_by_tool={
                "read_file": "workspace_file_read_v1",
            }),
        )],
    )
    assert result.output == "done"
    assert observed == [(0, 0), (1, 5), (0, 0)]
    assert [fact["fact"] for fact in facts] == ["accepted", "included_in_request"]
    assert {fact["source"] for fact in facts} == {"read_file"}
    assert binary_content_usage(result.all_messages()) == (1, 5)
