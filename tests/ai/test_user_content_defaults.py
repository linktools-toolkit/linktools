#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""User-content constructor defaults across durable public entry points."""

import copy
import json
from datetime import datetime, timezone

import pytest
from pydantic_ai import messages as sdk

from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import decode_user_content_payload
from linktools.ai.model.messages import decode_model_messages, encode_model_messages
from linktools.ai.model._user_content_codec import encode_user_content_item


def _items():
    yield sdk.TextContent('text'), ['metadata']
    yield sdk.BinaryContent(b'body', media_type='text/plain'), ['identifier', 'vendor_metadata']
    for target, suffix in [(sdk.ImageUrl, 'png'), (sdk.AudioUrl, 'mp3'), (sdk.DocumentUrl, 'pdf'), (sdk.VideoUrl, 'mp4')]:
        yield target(f'https://example.com/file.{suffix}'), ['media_type', 'identifier', 'vendor_metadata', 'force_download']
    yield sdk.UploadedFile('file-id', 'openai'), ['media_type', 'identifier', 'vendor_metadata']
    yield sdk.CachePoint(), ['ttl']


def _read(item: dict, route: str) -> object:
    if route == 'input':
        return decode_user_content_payload({'items': [item]})[0]
    message = sdk.ModelRequest(parts=[sdk.UserPromptPart([], timestamp=datetime(2026, 1, 1, tzinfo=timezone.utc))])
    wire = json.loads(encode_model_messages((message,)))
    wire[0]['parts'][0]['content'] = {'type': 'items', 'items': [item]}
    return decode_model_messages(canonical_json_bytes(wire))[0].parts[0].content[0]


@pytest.mark.parametrize('route', ['input', 'message'])
@pytest.mark.parametrize('item,field', [pytest.param(item, field, id=f'{type(item).__name__}-{field}') for item, names in _items() for field in names])
def test_omitted_content_fields_use_constructor_defaults(item: object, field: str, route: str) -> None:
    wire = encode_user_content_item(item)
    del wire[field]
    original = copy.deepcopy(wire)
    restored = _read(wire, route)
    assert restored == item
    assert getattr(restored, field) == getattr(item, field)
    assert encode_user_content_item(restored) == encode_user_content_item(item)
    assert wire == original


@pytest.mark.parametrize('route', ['input', 'message'])
@pytest.mark.parametrize('wire', [
    {'kind': 'binary', 'data': '*', 'media_type': 'text/plain'},
    {'kind': 'binary', 'data': 'YQ==', 'media_type': ''},
    {'kind': 'binary', 'data': 'YQ==', 'media_type': 'text/plain', 'identifier': False},
    {'kind': 'binary', 'data': 'YQ==', 'media_type': 'text/plain', 'vendor_metadata': []},
    {'kind': 'image-url', 'url': 'https://example.com/a.png', 'force_download': None},
    {'kind': 'image-url', 'url': 'https://example.com/a.png', 'media_type': False},
    {'kind': 'uploaded-file', 'file_id': 'file', 'provider_name': 'unknown'},
    {'kind': 'cache-point', 'ttl': None},
    {'kind': 'text-content'},
    {'kind': 'binary', 'data': 'YQ=='},
    {'kind': 'image-url'},
    {'kind': 'uploaded-file', 'file_id': 'file'},
    {'ttl': '5m'},
])
def test_explicit_invalid_and_required_content_fields_are_not_defaulted(wire: dict, route: str) -> None:
    with pytest.raises(AIError) as caught:
        _read(wire, route)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_url_missing_media_type_retains_deferred_inference_failure() -> None:
    restored = _read({'kind': 'image-url', 'url': 'https://example.com/opaque'}, 'message')
    with pytest.raises(ValueError, match='infer media type'):
        _ = restored.media_type


def test_identifier_default_is_stable_but_does_not_restore_a_custom_identifier() -> None:
    wire = {'kind': 'binary', 'data': 'YQ==', 'media_type': 'text/plain'}
    first = _read(wire, 'input')
    second = _read(wire, 'message')
    assert first.identifier == second.identifier == sdk.BinaryContent(b'a', media_type='text/plain').identifier
    assert _read({**wire, 'identifier': 'custom'}, 'input').identifier == 'custom'


def test_unknown_fields_remain_tolerated_in_ordinary_content() -> None:
    for item, _ in _items():
        wire = encode_user_content_item(item)
        wire['future'] = 1
        assert _read(wire, 'input') == item


@pytest.mark.parametrize('route', ['input', 'message'])
def test_retired_reference_provenance_survives_only_when_explicit(route: str) -> None:
    from linktools.ai.model import ReferencedContent
    from linktools.ai.model._user_content_codec import RetiredReferencedContent
    from .test_referenced_attachment_facts import _reference

    reference = _reference(identifier='attachment')
    summary = RetiredReferencedContent(reference, 'retired')
    wire = encode_user_content_item(summary)
    restored = _read(wire, route)
    assert isinstance(restored, RetiredReferencedContent)
    assert restored.reference == reference
    assert isinstance(restored.reference, ReferencedContent)
    del wire['metadata']
    ordinary = _read(wire, route)
    assert type(ordinary) is sdk.TextContent
    assert ordinary.content == 'retired' and ordinary.metadata is None


@pytest.mark.parametrize('route', ['input', 'message'])
@pytest.mark.parametrize('mutation', ['ref', 'source', 'media_type', 'revision', 'parameters-null', 'source-extra', 'metadata-reference'])
def test_reference_identity_and_provenance_are_not_defaulted(route: str, mutation: str) -> None:
    from .test_referenced_attachment_facts import _reference

    wire = encode_user_content_item(_reference())
    if mutation in {'ref', 'source', 'media_type'}:
        del wire[mutation]
    elif mutation == 'revision':
        del wire['source']['revision']
    elif mutation == 'parameters-null':
        wire['source_parameters'] = None
    elif mutation == 'source-extra':
        wire['source']['extra'] = 1
    else:
        wire = {'kind': 'text-content', 'content': 'retired', 'metadata': {'referenced_content': {'kind': 'text', 'text': 'not a descriptor'}}}
    with pytest.raises(AIError) as caught:
        _read(wire, route)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_shared_default_detection_keeps_noninit_and_no_default_required() -> None:
    from dataclasses import dataclass, field
    from linktools.ai.model._wire_fields import decode_fields

    @dataclass
    class Model:
        required: str
        optional: list = field(default_factory=list)
        derived: str = field(init=False, default='derived')

    decoders = {name: lambda value: value for name in ['required', 'optional', 'derived']}
    assert decode_fields({'required': 'r', 'derived': 'derived'}, Model, decoders) == {'required': 'r', 'derived': 'derived'}
    for wire in [{'required': 'r'}, {'derived': 'derived'}]:
        with pytest.raises(ValueError, match='required field'):
            decode_fields(wire, Model, decoders)


@pytest.mark.parametrize('route', ['input', 'message'])
@pytest.mark.parametrize('item', [
    sdk.TextContent('text', metadata={'origin': 'application'}),
    sdk.BinaryContent(b'body', media_type='text/plain', identifier='explicit', vendor_metadata={'detail': 'high'}),
    sdk.ImageUrl('https://example.com/opaque', media_type='image/png', identifier='explicit', force_download='allow-local', vendor_metadata={'detail': 'high'}),
    sdk.UploadedFile('file-id', 'openai', media_type='text/plain', identifier='explicit', vendor_metadata={'detail': 'high'}),
    sdk.CachePoint(ttl='1h'),
])
def test_explicit_content_values_are_never_replaced_with_defaults(item: object, route: str) -> None:
    wire = encode_user_content_item(item)
    restored = _read(wire, route)
    assert encode_user_content_item(restored) == wire
