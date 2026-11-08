#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Responses image encoding needs a MIME data URI, not a document filename."""

import json

import httpx2
import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, UserPromptPart
from pydantic_ai.models import ModelRequestParameters

from linktools.ai.errors import AIError
from linktools.ai.model import create_model, validate_model_inputs
from .test_model_protocols import model_binding, response


@pytest.mark.asyncio
@pytest.mark.parametrize('mime', ['image/png', 'image/heic', 'application/pdf'])
async def test_responses_binary_mime_preserves_native_request(monkeypatch: pytest.MonkeyPatch, mime: str) -> None:
    requests = []
    def handler(request: httpx2.Request) -> httpx2.Response:
        requests.append(json.loads(request.content))
        return httpx2.Response(200, json=response('openai-responses'))
    binding, _ = model_binding(monkeypatch, 'openai_responses', 'gpt-5', handler,
                              input_types=('mime:' + mime,))
    content = BinaryContent(b'synthetic file bytes', media_type=mime)
    validate_model_inputs(binding, content)
    async with binding.materialize() as model:
        await model.request([ModelRequest(parts=[UserPromptPart(content=[content])])], None, ModelRequestParameters())
    wire = requests[0]['input'][0]['content'][0]
    if mime.startswith('image/'):
        assert wire['type'] == 'input_image'
        assert wire['image_url'] == content.data_uri
    else:
        assert wire['type'] == 'input_file'
        assert wire['file_data'] == content.data_uri
        assert wire['filename'] == 'filename.pdf'


@pytest.mark.parametrize('mime,declared,reason', [
    ('image/heic', False, 'model_input_not_supported'),
    ('application/x-private', True, 'input_encoding_not_supported'),
    ('audio/wav', True, 'input_encoding_not_supported'),
    ('video/mp4', True, 'input_encoding_not_supported'),
])
def test_responses_mime_permission_and_document_constraints_remain(mime: str, declared: bool, reason: str) -> None:
    binding = create_model(api='openai-responses', model='gpt-5',
                           input_types=('mime:' + mime,) if declared else ())
    with pytest.raises(AIError) as raised:
        validate_model_inputs(binding, BinaryContent(b'bytes', media_type=mime))
    assert raised.value.safe_details['reason'] == reason
