#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local command composition preserves completed execution observations."""

from contextlib import asynccontextmanager
from pathlib import Path

import pytest

from linktools.ai.runtime import Runtime
from linktools.ai.web import create_app
from ._runtime_test_helpers import RuntimeUsageModels
from .test_web_console import client, _HEADERS


def test_web_command_preserves_terminal_history_after_handoff_and_reopen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import uvicorn
    import linktools.commands.ai.web as web_command

    opened = []
    identities = []
    original_open = Runtime.open

    @asynccontextmanager
    async def tracked_open(*args, **kwargs):
        async with original_open(*args, **kwargs) as runtime:
            opened.append(runtime)
            yield runtime

    class Server:
        def __init__(self, config):
            self.config = config

        async def serve(self):
            async with client(self.config.app) as http:
                if not identities:
                    response = await http.post('/api/sessions', json={
                        'session_id': 'session', 'request_id': 'create',
                    }, headers=_HEADERS)
                    assert response.status_code == 201, response.text
                    response = await http.post('/api/session/messages?session_id=session', json={
                        'prompt': 'hello', 'request_id': 'turn',
                    }, headers=_HEADERS)
                    assert response.status_code == 202, response.text
                    identity = response.json()['execution_id']
                    identities.append(identity)
                    stream = await http.get(f'/api/executions/{identity}/events')
                    assert stream.status_code == 200, stream.text
                    assert 'OBSERVATION_FAILED' not in stream.text, stream.text
                    result = await (await opened[-1].executions.get(identity)).wait()
                    assert result.result.output == {'text': 'done'}
                identity = identities[0]
                for _ in range(2):
                    response = await http.get(f'/api/executions/{identity}/models?include_content=true')
                    assert response.status_code == 200, response.text
                    assert response.json()['items']
                    response = await http.get(f'/api/executions/{identity}/history?include_content=true')
                    assert response.status_code == 200, response.text
                    assert response.json()['items']
                    response = await http.get(f'/api/executions/{identity}/result')
                    assert response.status_code == 200, response.text
                    assert response.json()['output'] == {'text': 'done'}

    monkeypatch.setattr(Runtime, 'open', tracked_open)
    monkeypatch.setattr(web_command, '_local_models', lambda *_, **__: RuntimeUsageModels())
    monkeypatch.setattr(uvicorn, 'Server', Server)
    for _ in range(2):
        args = web_command.command.create_parser().parse_args([
            '--project', str(tmp_path), '--model', 'fake',
        ])
        assert web_command.command.run(args) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize('session_bound', [False, True])
async def test_local_runtime_retains_terminal_model_history(
    tmp_path: Path, session_bound: bool,
) -> None:
    from linktools.ai.runtime import RuntimeHistory
    from linktools.commands.ai._common import (
        _load_workspace, _local_runtime_storage, _open_local_runtime,
    )

    workspace = _load_workspace(str(tmp_path))
    async with _open_local_runtime(workspace, models=RuntimeUsageModels()) as runtime:
        agent = runtime.agents.get('default')
        if session_bound:
            session = await agent.create_session('session')
            execution = await session.start('hello', idempotency_key='turn')
        else:
            execution = await agent.start('hello', idempotency_key='turn')
        waited = await execution.wait()
        assert waited.result.output == {'text': 'done'}
        identity = execution.execution_id
        models = await execution.model_interactions(include_content=True)
        assert models.items
    async with RuntimeHistory.open('default', storage=_local_runtime_storage(workspace)) as history:
        async with client(create_app(history=history)) as http:
            response = await http.get(f'/api/executions/{identity}/models?include_content=true')
            assert response.status_code == 200, response.text
            assert response.json()['items']
            response = await http.get(f'/api/executions/{identity}/history?include_content=true')
            assert response.status_code == 200, response.text
            assert response.json()['items']


def test_run_command_history_is_readable_after_command_close(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    import asyncio
    import json
    import linktools.commands.ai.run as run_command
    from linktools.ai.runtime import RuntimeHistory
    from linktools.commands.ai._common import _load_workspace, _local_runtime_storage

    monkeypatch.setattr(run_command, '_local_runtime_models', lambda *_: RuntimeUsageModels())
    args = run_command.command.create_parser().parse_args([
        '--project', str(tmp_path), '--model', 'fake', '--json', 'hello',
    ])
    assert run_command.command.run(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'SUCCEEDED'
    assert result['output'] == {'text': 'done'}

    async def inspect() -> None:
        workspace = _load_workspace(str(tmp_path))
        async with RuntimeHistory.open('default', storage=_local_runtime_storage(workspace)) as history:
            async with client(create_app(history=history)) as http:
                response = await http.get(f"/api/executions/{result['execution_id']}/models?include_content=true")
                assert response.status_code == 200, response.text
                assert response.json()['items']

    asyncio.run(inspect())
