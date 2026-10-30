#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical payload aggregation preserves wire facts, not SDK projections."""

import json

import pytest

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import (
    combine_model_message_payloads,
    decode_model_message_payloads,
    decode_model_messages,
    encode_model_messages,
)


def test_combining_preserves_omitted_fields_without_materializing_defaults() -> None:
    message = {'version': 1, 'kind': 'request', 'parts': []}
    raw = canonical_json_bytes([message])
    combined = combine_model_message_payloads(iter([raw, raw]))
    assert combined == canonical_json_bytes([message, message])
    assert json.loads(combined) == [message, message]
    restored = decode_model_messages(combined)
    assert len(restored) == 2
    assert restored[0].state == restored[1].state
    assert encode_model_messages(restored) != combined


@pytest.mark.parametrize('metadata', [
    {'unicode': '中文🙂\n\"\\', 'nested': [[], [{'array': [1, None, True, False]}]]},
    {'numbers': [0, -1, 2**80, 1.0, -0.0, 1e-20, 1e20]},
])
def test_combining_canonical_json_preserves_exact_wire_bytes(metadata: dict) -> None:
    messages = [
        {'version': 1, 'kind': 'request', 'parts': [], 'metadata': metadata},
        {'version': 1, 'kind': 'request', 'parts': []},
    ]
    leaves = [canonical_json_bytes([message]) for message in messages]
    combined = combine_model_message_payloads(iter(leaves))
    assert combined == canonical_json_bytes(messages)
    assert combined == b'[' + b','.join(leaf[1:-1] for leaf in leaves) + b']'
    assert len(decode_model_messages(combined)) == 2


def test_combining_empty_iterable_returns_canonical_empty_array() -> None:
    assert combine_model_message_payloads(iter(())) == b'[]'
    assert decode_model_messages(combine_model_message_payloads(())) == ()


@pytest.mark.parametrize('raw', [
    b'[]', b'[{},{}]', b'{}', b'null', b'[ {} ]',
    b'[{"b":1,"a":2}]', b'[{"a":1,"a":1}]', b'[',
    b'[NaN]', b'[Infinity]', b'[-Infinity]', b'[1e999]', b'["\xff"]',
])
def test_invalid_or_nonsingleton_leaf_is_rejected(raw: bytes) -> None:
    with pytest.raises(AIError) as caught:
        combine_model_message_payloads([raw])
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_unknown_message_version_is_left_to_normal_semantic_decoder() -> None:
    raw = canonical_json_bytes([{'version': 999, 'kind': 'request', 'parts': []}])
    assert combine_model_message_payloads([raw]) == raw
    with pytest.raises(AIError) as caught:
        decode_model_messages(combine_model_message_payloads([raw]))
    assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


@pytest.mark.parametrize('message', [42, {'version': 1, 'kind': 'request', 'parts': [], 'state': 'invalid'}, {'version': 1, 'kind': 'request', 'parts': [], 'unknown': 1}])
def test_aggregation_does_not_replace_normal_semantic_validation(message: object) -> None:
    raw = canonical_json_bytes([message])
    assert combine_model_message_payloads([raw]) == raw
    with pytest.raises(AIError) as caught:
        decode_model_messages(combine_model_message_payloads([raw]))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize('metadata', [
    {'unicode': '中文🙂', 'nested': [[], [1, None, True]]},
    {'numbers': [0, -1, 2**80, 1.0, -0.0, 1e-20, 1e20]},
])
def test_reading_leaves_returns_original_bytes_and_normal_message_values(metadata: dict) -> None:
    message = {'version': 1, 'kind': 'request', 'parts': [],
               'timestamp': '2026-01-01T00:00:00+00:00', 'metadata': metadata}
    leaves = (canonical_json_bytes([message]),) * 2
    raw, restored = decode_model_message_payloads(iter(leaves))
    assert raw == canonical_json_bytes([message, message])
    assert restored == decode_model_messages(raw)
    assert encode_model_messages(restored) != raw


def test_reading_empty_leaf_sequence_returns_empty_values() -> None:
    assert decode_model_message_payloads(()) == (b'[]', ())


@pytest.mark.parametrize('raw, code', [
    (b'[]', ErrorCode.STORAGE_INTEGRITY_ERROR),
    (b'[{},{}]', ErrorCode.STORAGE_INTEGRITY_ERROR),
    (b'[ {} ]', ErrorCode.STORAGE_INTEGRITY_ERROR),
    (b'[{"a":1,"a":1}]', ErrorCode.STORAGE_INTEGRITY_ERROR),
    (canonical_json_bytes([{'version': 999, 'kind': 'request', 'parts': []}]), ErrorCode.STORAGE_VERSION_UNSUPPORTED),
    (canonical_json_bytes([{'version': 1, 'kind': 'request', 'parts': [], 'state': 'invalid'}]), ErrorCode.STORAGE_INTEGRITY_ERROR),
])
def test_reading_leaves_preserves_structural_and_semantic_rejections(raw: bytes, code: ErrorCode) -> None:
    with pytest.raises(AIError) as caught:
        decode_model_message_payloads([raw])
    assert caught.value.code is code
