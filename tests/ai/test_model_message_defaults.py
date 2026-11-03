#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declared model-message defaults preserve explicit wire and durable identities."""

import copy
import json
from dataclasses import MISSING, Field, fields
from datetime import datetime, timedelta, timezone

import pytest
from pydantic_ai import RequestUsage
from pydantic_ai import messages as sdk

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import messages as codec

_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _fixtures():
    yield sdk.ModelRequest(parts=[], timestamp=_STAMP), (), sdk.ModelRequest
    yield sdk.ModelResponse(parts=[], timestamp=_STAMP), (), sdk.ModelResponse
    request_parts = [
        sdk.SystemPromptPart('system', timestamp=_STAMP),
        codec.SummaryPromptPart('summary', timestamp=_STAMP),
        sdk.UserPromptPart('prompt', timestamp=_STAMP),
        sdk.RetryPromptPart('retry', tool_call_id='call', timestamp=_STAMP),
        sdk.ToolReturnPart('tool', 'result', tool_call_id='call', timestamp=_STAMP),
        sdk.ToolAvailabilityDeltaPart(tools_added=['tool']),
        sdk.InstructionDeltaPart(id=None, content='content'),
    ]
    if getattr(sdk, 'RetryFeedbackPart', None) is not None:
        request_parts.append(sdk.RetryFeedbackPart('retry', retry=sdk.RetryInfo(cause='model_retry'), timestamp=_STAMP))
    if getattr(sdk, 'ModelSettingsDeltaPart', None) is not None:
        request_parts.append(sdk.ModelSettingsDeltaPart(settings={'thinking': 'medium'}, top_level=True))
    response_parts = [
        sdk.TextPart('text'), sdk.ThinkingPart('thinking'), sdk.CompactionPart(),
        sdk.FilePart(sdk.BinaryContent(b'body', media_type='image/png')),
        sdk.ToolCallPart('tool', tool_call_id='call'),
        sdk.NativeToolCallPart('native', tool_call_id='call'),
        sdk.NativeToolReturnPart('native', 'result', tool_call_id='call', timestamp=_STAMP),
        sdk.SpeechPart(speaker='assistant', transcript='speech'),
    ]
    for part in request_parts:
        yield sdk.ModelRequest(parts=[part], timestamp=_STAMP), ('parts', 0), type(part)
    for part in response_parts:
        yield sdk.ModelResponse(parts=[part], timestamp=_STAMP), ('parts', 0), type(part)
    for block in (sdk.InstructionPart('instruction'), sdk.InstructionDigest(content_hash='a' * 64)):
        yield sdk.ModelRequest(parts=[sdk.InstructionBaselinePart(instructions=[block])]), ('parts', 0, 'instructions', 0), type(block)
    if getattr(sdk, 'RetryInfo', None) is not None:
        yield sdk.ModelRequest(parts=[sdk.RetryFeedbackPart('retry', retry=sdk.RetryInfo(cause='model_retry'))]), ('parts', 0, 'retry'), sdk.RetryInfo
    yield sdk.ModelResponse(parts=[], usage=RequestUsage(), timestamp=_STAMP), ('usage',), RequestUsage
    attempt = sdk.ModelRequestAttempt(model_name='model', outcome='error', timestamp=_STAMP, duration=timedelta())
    yield sdk.ModelResponse(parts=[], timestamp=_STAMP, failed_attempts=[attempt]), ('failed_attempts', 0), sdk.ModelRequestAttempt


def _at(value, path):
    for key in path:
        value = value[key] if isinstance(value, (dict, list)) else getattr(value, key)
    return value


def _default_cases():
    for message, path, target in _fixtures():
        wire = json.loads(codec.encode_model_messages((message,)))[0]
        selected = _at(wire, path)
        for field in fields(target):
            name = field.name
            if name in {'kind', 'part_kind', 'tool_call_id'} or name not in selected:
                continue
            if field.init and (field.default is not MISSING or field.default_factory is not MISSING):
                yield pytest.param(message, path, field, name, id=f'{target.__name__}-{name}')


@pytest.mark.parametrize('message,path,field,wire_name', list(_default_cases()))
def test_declared_defaults_apply_only_to_omitted_wire_fields(
    message: sdk.ModelMessage, path: tuple[str | int, ...], field: Field, wire_name: str,
) -> None:
    wire = json.loads(codec.encode_model_messages((message,)))
    del _at(wire[0], path)[wire_name]
    original = copy.deepcopy(wire)
    before = datetime.now(timezone.utc)
    restored, = codec.decode_model_messages(canonical_json_bytes(wire))
    after = datetime.now(timezone.utc)
    actual = getattr(_at(restored, path), field.name)
    if field.default is not MISSING:
        assert actual == field.default
    elif isinstance(actual, datetime):
        assert before <= actual <= after
    else:
        assert actual == field.default_factory()
    assert wire == original
    assert codec.decode_model_messages(codec.encode_model_messages((restored,))) == (restored,)


@pytest.mark.parametrize('message', [
    sdk.ModelResponse(parts=[sdk.ToolCallPart('tool', tool_call_id='call')], timestamp=_STAMP),
    sdk.ModelResponse(parts=[sdk.NativeToolCallPart('tool', tool_call_id='call')], timestamp=_STAMP),
    sdk.ModelRequest(parts=[sdk.ToolReturnPart('tool', 'done', tool_call_id='call', timestamp=_STAMP)]),
    sdk.ModelResponse(parts=[sdk.NativeToolReturnPart('tool', 'done', tool_call_id='call', timestamp=_STAMP)], timestamp=_STAMP),
    sdk.ModelRequest(parts=[sdk.RetryPromptPart('retry', tool_call_id='call', timestamp=_STAMP)]),
])
def test_durable_tool_identity_is_not_generated_during_decode(message: sdk.ModelMessage) -> None:
    wire = json.loads(codec.encode_model_messages((message,)))
    del wire[0]['parts'][0]['tool_call_id']
    with pytest.raises(AIError) as caught:
        codec.decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize('field,value', [('usage', None), ('timestamp', None), ('state', 'unknown'), ('model_name', False)])
def test_explicit_invalid_message_fields_never_use_defaults(field: str, value: object) -> None:
    wire = json.loads(codec.encode_model_messages((sdk.ModelResponse(parts=[], timestamp=_STAMP),)))
    wire[0][field] = value
    with pytest.raises(AIError) as caught:
        codec.decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize('field', ['input_tokens', 'details', 'cost'])
def test_usage_provider_extensions_do_not_replace_omitted_standard_fields(field: str) -> None:
    wire = json.loads(codec.encode_model_messages((sdk.ModelResponse(parts=[], timestamp=_STAMP),)))
    del wire[0]['usage'][field]
    wire[0]['usage']['provider_extra'] = {field: -1}
    restored, = codec.decode_model_messages(canonical_json_bytes(wire))
    assert getattr(restored.usage, field) == getattr(RequestUsage(), field)
    assert restored.usage.__dict__['provider_extra'] == {field: -1}


def test_default_factories_are_independent_for_repeated_message_reads() -> None:
    wire = json.loads(codec.encode_model_messages((sdk.ModelResponse(parts=[], timestamp=_STAMP),)))
    del wire[0]['usage']
    first, = codec.decode_model_messages(canonical_json_bytes(wire))
    second, = codec.decode_model_messages(canonical_json_bytes(wire))
    first.usage.details['local'] = 1
    assert second.usage.details == {}


@pytest.mark.parametrize("path,key", [
    ((), "version"), ((), "kind"), ((), "parts"),
    (("parts", 0), "part_kind"), (("parts", 0), "content"),
])
def test_protocol_discriminators_and_nondefault_fields_remain_required(
    path: tuple[str | int, ...], key: str,
) -> None:
    wire = json.loads(codec.encode_model_messages((sdk.ModelResponse(parts=[sdk.TextPart("text")], timestamp=_STAMP),)))
    del _at(wire[0], path)[key]
    with pytest.raises(AIError):
        codec.decode_model_messages(canonical_json_bytes(wire))


@pytest.mark.parametrize("path", [(), ("parts", 0)])
def test_unknown_message_fields_are_not_silently_dropped(path: tuple[str | int, ...]) -> None:
    wire = json.loads(codec.encode_model_messages((sdk.ModelResponse(parts=[sdk.TextPart("text")], timestamp=_STAMP),)))
    _at(wire[0], path)["unknown"] = None
    with pytest.raises(AIError) as caught:
        codec.decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize('kind,field,bad', [
    ('request', 'state', 'invalid'),
    ('response', 'state', 'invalid'),
    ('response', 'finish_reason', 'invalid'),
    ('response', 'timestamp', None),
    ('response', 'failed_attempts', 'invalid'),
])
def test_early_message_validation_precedes_unknown_part_version(
    kind: str, field: str, bad: object,
) -> None:
    message = sdk.ModelRequest(parts=[]) if kind == 'request' else sdk.ModelResponse(parts=[], timestamp=_STAMP)
    wire = json.loads(codec.encode_model_messages((message,)))
    wire[0]['parts'] = [{'part_kind': 'future-part'}]
    with pytest.raises(AIError) as unknown:
        codec.decode_model_messages(canonical_json_bytes(wire))
    assert unknown.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    wire[0][field] = bad
    with pytest.raises(AIError) as malformed:
        codec.decode_model_messages(canonical_json_bytes(wire))
    assert malformed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize('part', [
    {'part_kind': 'tool-call', 'tool_name': 'tool', 'tool_call_id': 'call', 'tool_kind': 7},
    {'part_kind': 'tool-call', 'tool_name': False, 'tool_call_id': 'call'},
    {'part_kind': 'tool-call', 'tool_call_id': 'call'},
])
def test_tool_parts_reject_invalid_kind_types_or_required_fields(part: dict) -> None:
    wire = json.loads(codec.encode_model_messages((sdk.ModelResponse(parts=[], timestamp=_STAMP),)))
    wire[0]['parts'] = [part]
    with pytest.raises(AIError):
        codec.decode_model_messages(canonical_json_bytes(wire))


@pytest.mark.parametrize('name', ['model_id', 'canonical_system'])
def test_response_optional_sdk_identity_survives_durable_round_trip(name: str) -> None:
    if name not in {field.name for field in fields(sdk.ModelResponse)}:
        pytest.skip('installed SDK does not expose this response identity')
    response = sdk.ModelResponse(parts=[sdk.TextPart('answer')], timestamp=_STAMP, **{name: 'provider:model'})
    restored = codec.decode_model_messages(codec.encode_model_messages((response,)))
    assert restored == (response,)
    assert getattr(restored[0], name) == 'provider:model'
    original_default = codec.decode_model_messages(codec.encode_model_messages((sdk.ModelResponse(parts=[], timestamp=_STAMP),)))
    assert getattr(original_default[0], name) is None
