#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""ACP baseline content reaches Runtime without implicit resource access."""
from types import SimpleNamespace

import pytest

from linktools.ai.acp import ACPAgent
from linktools.ai.core import ExecutionStatus, Principal


@pytest.mark.asyncio
@pytest.mark.parametrize('dispatch', [False, True])
@pytest.mark.parametrize('linked', [False, True])
async def test_acp_baseline_content_preserves_context_and_routes_principal(dispatch: bool, linked: bool) -> None:
    acp = pytest.importorskip('acp')
    schema = pytest.importorskip('acp.schema')
    from acp.agent.router import build_agent_router
    principal = Principal('caller', 'tenant', 'service')
    calls = []
    async def get_session(session_id, *, principal):
        calls.append(('session', session_id, principal))
        return SimpleNamespace(agent_id='selected-agent')
    async def wait(**kwargs):
        return SimpleNamespace(result=SimpleNamespace(status=ExecutionStatus.SUCCEEDED), observation_error=None)
    async def start(prompt, **kwargs):
        calls.append(('start', prompt, kwargs))
        return SimpleNamespace(execution_id='execution', wait=wait)
    def get_agent(agent_id):
        calls.append(('agent', agent_id))
        return SimpleNamespace(start=start)
    agent = ACPAgent(SimpleNamespace(sessions=SimpleNamespace(get=get_session), agents=SimpleNamespace(get=get_agent)), principal=principal)
    response = await agent.initialize(acp.PROTOCOL_VERSION)
    assert not response.agent_capabilities.prompt_capabilities.embedded_context
    blocks = [{'type': 'text', 'text': 'before'}]
    if linked:
        blocks += [
            {'type': 'resource_link', 'name': 'minimal', 'uri': 'custom://unchanged/%2F?q=1'},
            {'type': 'resource_link', 'name': 'source', 'uri': 'file:///not-read.py', 'title': 'Friendly title',
             'description': 'Client context', 'mimeType': 'text/plain', 'size': 0},
        ]
    blocks += [{'type': 'text', 'text': 'after'}]
    request = schema.PromptRequest.model_validate({'sessionId': 'session', 'prompt': blocks})
    if dispatch:
        result = await build_agent_router(agent)('session/prompt', request.model_dump(by_alias=True), False)
    else:
        result = await agent.prompt(request.session_id, request.prompt)
    assert result.stop_reason == 'end_turn'
    assert calls[:2] == [('session', 'session', principal), ('agent', 'selected-agent')]
    assert calls[2][2] == {'principal': principal, 'session_id': 'session'}
    expected = ('before\nResource: minimal\nURI: custom://unchanged/%2F?q=1\n'
                '\nResource: source\nURI: file:///not-read.py\nTitle: Friendly title\n'
                'Description: Client context\nMIME type: text/plain\nSize: 0\nafter') if linked else 'beforeafter'
    assert calls[2][1] == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('content', [
    {'type': 'image', 'data': 'AA==', 'mimeType': 'image/png'},
    {'type': 'audio', 'data': 'AA==', 'mimeType': 'audio/wav'},
    {'type': 'resource', 'resource': {'uri': 'file:///not-read.txt', 'text': 'embedded'}},
])
async def test_acp_unadvertised_content_returns_protocol_error(content: dict[str, object]) -> None:
    acp = pytest.importorskip('acp')
    schema = pytest.importorskip('acp.schema')
    from acp.agent.router import build_agent_router
    agent = ACPAgent(SimpleNamespace(), principal=Principal('caller', 'tenant'))
    await agent.initialize(acp.PROTOCOL_VERSION)
    request = schema.PromptRequest.model_validate({'sessionId': 'session', 'prompt': [content]})
    with pytest.raises(acp.RequestError) as failed:
        await build_agent_router(agent)('session/prompt', request.model_dump(by_alias=True), False)
    assert failed.value.code == -32602
    assert failed.value.data == {'reason': 'unsupported_prompt_content'}
