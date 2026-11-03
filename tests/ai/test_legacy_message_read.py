#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen filesystem message bytes are read, never rewritten or emitted anew."""
import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic_ai import RequestUsage, messages as sdk

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.model.messages import (
    RetiredReferencedContent, SummaryPromptPart, decode_model_message_payloads,
    decode_model_messages, encode_model_messages,
)
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)
_FIXTURE = Path(__file__).parent / 'fixtures' / 'persistence' / 'legacy_model_messages_v1.json'


def _messages():
    reference = ReferencedContent(
        AssetVersionRef(AssetKey('document', 'retained'), 'layer', StorageEntryRevision(1),
                        hashlib.sha256(b'retained').hexdigest(), 8),
        media_type='application/pdf', source=CapabilityRef('capability', 'files', 1),
    )
    identity = sdk.InstructionId(sdk.CapabilityInstructionSource('catalog'), name='details')
    return (
        sdk.ModelRequest(parts=[SummaryPromptPart('summary', timestamp=_STAMP),
            sdk.UserPromptPart('text', timestamp=_STAMP)], conversation_id='conversation', timestamp=_STAMP),
        sdk.ModelRequest(parts=[sdk.InstructionBaselinePart(instructions=[
            sdk.InstructionPart(id=identity, content='instruction', dynamic=True),
            sdk.InstructionDigest(id=identity, content_hash='a'*64, dynamic=True)]),
            sdk.InstructionDeltaPart(id=identity, content='new', base_hash='a'*64),
            sdk.UserPromptPart(['plain', sdk.BinaryContent(b'bytes', media_type='application/pdf'),
                sdk.ImageUrl('https://example.invalid/image.png'), reference,
                RetiredReferencedContent(reference, 'retained summary')], timestamp=_STAMP),
            sdk.ToolReturnPart('read', [reference, {'ordinary': 'value'}], tool_call_id='call', timestamp=_STAMP),
            sdk.RetryPromptPart('retry', tool_call_id='call', timestamp=_STAMP)], timestamp=_STAMP),
        sdk.ModelResponse(parts=[sdk.TextPart('answer'), sdk.ToolCallPart('read', {'a': 1}, tool_call_id='call'),
            sdk.FilePart(sdk.BinaryContent(b'file', media_type='application/pdf'))], timestamp=_STAMP,
            usage=RequestUsage(input_tokens=3, provider_extra=7),
            failed_attempts=[sdk.ModelRequestAttempt(model_name='model', provider_name='provider', outcome='error',
                error='retained error', timestamp=_STAMP, duration=timedelta(microseconds=1234567),
                usage=RequestUsage(input_tokens=2, provider_extra=8))]),
    )


def test_real_legacy_writer_bytes_restore_all_retained_facts_and_keep_raw_archive_identity():
    raw = _FIXTURE.read_bytes()
    provenance = json.loads(_FIXTURE.with_suffix('.provenance.json').read_text())
    assert hashlib.sha256(raw).hexdigest() == provenance['sha256']
    expected = _messages()
    restored = decode_model_messages(raw)
    assert encode_model_messages(restored) == encode_model_messages(expected)
    for original in json.loads(raw):
        leaf = canonical_json_bytes([original])
        combined, decoded = decode_model_message_payloads((leaf,))
        assert combined == leaf
        assert encode_model_messages(decoded) == encode_model_messages(decode_model_messages(leaf))
    current = json.loads(encode_model_messages(restored))
    assert all('agent_conversation_id' not in row and 'conversation_id' in row for row in current)
    assert _FIXTURE.read_bytes() == raw


def test_legacy_ordinary_content_keeps_its_additive_field_compatibility():
    rows = json.loads(_FIXTURE.read_bytes())
    for item in rows[1]['parts'][2]['content']['items'][:3]:
        item['future_display_hint'] = 'optional'
    raw = canonical_json_bytes(rows)
    restored = decode_model_messages(raw)
    assert encode_model_messages(restored) == encode_model_messages(_messages())
    for row in rows:
        leaf = canonical_json_bytes([row])
        combined, _ = decode_model_message_payloads((leaf,))
        assert combined == leaf


@pytest.mark.parametrize('damage', ['dual_identity', 'prompt_extra', 'prompt_bad', 'reference_extra', 'usage_collision',
                                   'attempt_negative', 'attempt_overflow', 'attempt_dual', 'attempt_pairs', 'attempts_mapping', 'unknown_field'])
def test_legacy_read_rejects_ambiguous_or_corrupt_values(damage):
    rows = json.loads(_FIXTURE.read_bytes())
    if damage == 'dual_identity': rows[0]['conversation_id'] = 'other'
    elif damage == 'prompt_extra': rows[0]['parts'][1]['content']['extra'] = 1
    elif damage == 'prompt_bad': rows[0]['parts'][1]['content']['value'] = 1
    elif damage == 'reference_extra': rows[1]['parts'][2]['content']['items'][3]['extra'] = 1
    elif damage == 'usage_collision': rows[2]['usage']['extensions']['input_tokens'] = 99
    elif damage == 'attempt_negative': rows[2]['failed_attempts'][0]['duration_us'] = -1
    elif damage == 'attempt_overflow': rows[2]['failed_attempts'][0]['duration_us'] = 10**100
    elif damage == 'attempt_pairs': rows[2]['failed_attempts'][0] = list(rows[2]['failed_attempts'][0].items())
    elif damage == 'attempts_mapping': rows[2]['failed_attempts'] = {}
    elif damage == 'attempt_dual': rows[2]['failed_attempts'][0]['duration'] = 'PT1S'
    elif damage == 'unknown_field': rows[0]['unknown'] = 'cannot discard'
    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(rows))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
