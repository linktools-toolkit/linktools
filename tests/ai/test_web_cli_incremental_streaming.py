#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The actual CLI streams text through ASGI before a provider completes."""

import asyncio
import json
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from .test_live_history_readback_integration import _Models
from .test_web_console import client, _HEADERS


@pytest.mark.parametrize("provider", ["function", "openai"])
def test_web_command_delivers_text_before_model_completion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, provider: str) -> None:
    import uvicorn
    import linktools.commands.ai.web as web_command

    release = asyncio.Event()
    entered = asyncio.Event()
    completed = asyncio.Event()
    first_text = asyncio.Event()
    frames = []

    async def model(messages, info):
        entered.set()
        yield 'first '
        await release.wait()
        yield 'last'
        completed.set()

    class Server:
        def __init__(self, config):
            self.config = config

        async def serve(self):
            async with client(self.config.app) as http:
                response = await http.post('/api/sessions', json={'session_id': 'session', 'request_id': 'create'}, headers=_HEADERS)
                assert response.status_code == 201, response.text
                response = await http.post('/api/session/messages?session_id=session', json={'prompt': 'hello', 'request_id': 'turn'}, headers=_HEADERS)
                assert response.status_code == 202, response.text
                identity = response.json()['execution_id']
                path = f'/api/executions/{identity}/events'
                requested = False
                disconnect = asyncio.Event()

                async def receive():
                    nonlocal requested
                    if not requested:
                        requested = True
                        return {'type': 'http.request', 'body': b'', 'more_body': False}
                    await disconnect.wait()
                    return {'type': 'http.disconnect'}

                async def send(message):
                    if message['type'] == 'http.response.start':
                        assert message['status'] == 200
                    if message['type'] == 'http.response.body' and message.get('body'):
                        text = message['body'].decode()
                        frames.append(text)
                        for line in text.splitlines():
                            if not line.startswith('data: '):
                                continue
                            payload = json.loads(line[6:])
                            event = payload.get('item', {}).get('event', {})
                            if event.get('event_type') == 'ASSISTANT_TEXT_DELTA':
                                assert payload['item']['depth'] == 0
                                assert event['payload']['text']
                                assert not release.is_set() or first_text.is_set()
                                first_text.set()

                scope = {'type': 'http', 'asgi': {'version': '3.0', 'spec_version': '2.0'}, 'method': 'GET', 'scheme': 'http', 'path': path, 'raw_path': path.encode(), 'query_string': b'', 'headers': [(b'host', b'127.0.0.1:8765')], 'server': ('127.0.0.1', 8765), 'client': ('127.0.0.1', 12345)}
                observing = asyncio.create_task(self.config.app(scope, receive, send))
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    await asyncio.wait_for(first_text.wait(), 5)
                    assert not completed.is_set()
                    assert not release.is_set()
                    release.set()
                    await asyncio.wait_for(observing, 10)
                    assert completed.is_set()
                    assert 'OBSERVATION_FAILED' not in ''.join(frames)
                    assert 'event: snapshot' in ''.join(frames)
                finally:
                    release.set()
                    disconnect.set()
                    if not observing.done():
                        observing.cancel()
                    await asyncio.gather(observing, return_exceptions=True)

    if provider == 'function':
        monkeypatch.setattr(web_command, '_local_runtime_models', lambda *_: _Models(FunctionModel(stream_function=model)))
    else:
        import httpx2
        from openai import DefaultAsyncHttpxClient
        from linktools.ai.model import _openai

        class Stream(httpx2.AsyncByteStream):
            async def __aiter__(self):
                async for text in model(None, None):
                    chunk = {'id': 'chatcmpl-gated', 'object': 'chat.completion.chunk', 'created': 1, 'model': 'gpt-4o', 'choices': [{'index': 0, 'delta': {'content': text}, 'finish_reason': None}]}
                    yield b'data: ' + json.dumps(chunk).encode() + b'\n\n'
                yield b'data: [DONE]\n\n'

        async def handle(request):
            assert json.loads(request.content)['stream'] is True
            return httpx2.Response(200, headers={'content-type': 'text/event-stream'}, stream=Stream())

        monkeypatch.setattr(_openai, 'DefaultAsyncHttpxClient', lambda **kwargs: DefaultAsyncHttpxClient(transport=httpx2.MockTransport(handle), **kwargs))
    monkeypatch.setattr(uvicorn, 'Server', Server)
    args = web_command.command.create_parser().parse_args(['--project', str(tmp_path), '--model', 'gpt-4o', '--base-url', 'https://stream.invalid/v1', '--api-key', 'test-only-placeholder'])
    assert web_command.command.run(args) == 0
