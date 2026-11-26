#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local command composition preserves completed execution observations."""

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

import pytest

from linktools.ai.persistence import Persistence
from linktools.ai.runtime import Runtime
from linktools.ai.web import create_app
import linktools.commands.ai._common as common_command
from ._runtime_test_helpers import RuntimeUsageModels, _wait_for_committed
from .test_web_console import client, _HEADERS


def test_web_command_preserves_terminal_history_after_handoff_and_reopen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path,
) -> None:
    import asyncio
    import uvicorn
    import linktools.commands.ai.web as web_command
    from linktools.ai.runtime.turn_clients import TurnObserver

    opened = []
    identities = []
    durable_trace = []
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
                    runtime = opened[-1]
                    with TurnObserver(runtime, 'session', runtime.default_principal, local_workspace=True) as observer:
                        response = await http.post('/api/session/messages?session_id=session', json={
                            'prompt': 'hello', 'request_id': 'turn',
                        }, headers=_HEADERS)
                        assert response.status_code == 202, response.text
                        identity = response.json()['turn_id']
                        async with asyncio.timeout(10):
                            async for frame in observer.events(identity):
                                if frame['type'] == 'terminal':
                                    assert frame['state'] == 'succeeded', frame
                                    assert frame['output'] == {'text': 'done'}
                                    break
                            else:
                                pytest.fail('Missing terminal turn')
                    identities.append(identity)

                    async def trace_items():
                        response = await http.get(f'/api/session?session_id=session&turn_id={identity}&trace=true')
                        assert response.status_code == 200, response.text
                        return response.json()['items']

                    # Optional Trace publishes independently of the terminal
                    # response. Observe its commit before testing cold reads.
                    durable_trace[:] = await _wait_for_committed(trace_items, bool)
                    assert all(item['turn_id'] == identity for item in durable_trace)
                identity = identities[0]
                for _ in range(2):
                    response = await http.get(f'/api/session?session_id=session&turn_id={identity}&trace=true')
                    assert response.status_code == 200, response.text
                    trace = response.json()['items']
                    assert trace == durable_trace
                    response = await http.get(f'/api/session?session_id=session&turn_id={identity}')
                    assert response.status_code == 200, response.text
                    turn = response.json()
                    assert turn['turn_id'] == identity and turn['conversation_committed']
                    assert turn['user_input']['prompt'] == 'hello'
                    assert [(item['item_kind'], item['content']) for item in turn['items']] == [('assistant', 'done')]
                    response = await http.get('/api/session?session_id=session')
                    assert response.status_code == 200, response.text
                    assert response.json()['timeline']['items'] == [turn]

    monkeypatch.setattr(Runtime, 'open', tracked_open)
    monkeypatch.setattr(common_command._ModelOptions, 'create_model_binding', lambda _self: RuntimeUsageModels())
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
    from linktools.ai.history import History
    from linktools.commands.ai._common import (
        _load_workspace, _local_runtime_storage, _open_local_runtime,
    )

    workspace = _load_workspace(str(tmp_path))
    async with _open_local_runtime(workspace, model=RuntimeUsageModels()) as local:
        runtime = local.runtime
        agent = runtime.agents.get('default')
        if session_bound:
            session = await agent.create_session('session')
            turn = await runtime.sessions.start(session.session_id, 'hello', agent=agent,
                                                principal=runtime.default_principal, idempotency_key='turn')
            result = await turn.wait_final()
            assert result.view['output'] == {'text': 'done'}
            identity = turn.key.turn_id
            trace_before_close = await _wait_for_committed(
                lambda: runtime.history.turn_trace(session.session_id, identity, principal=runtime.default_principal),
                lambda page: bool(page.items),
            )
            assert all(item.turn_id == identity for item in trace_before_close.items)
            detail = await runtime.sessions.refresh(session.session_id, principal=runtime.default_principal)
            assert detail["timeline"]["items"][0]["turn_id"] == identity
            with pytest.raises(ValueError, match="requires max_age=0"):
                await runtime.sessions.refresh(session.session_id, principal=runtime.default_principal, max_age=1)
        else:
            execution = await agent.start('hello', idempotency_key='turn')
            waited = await execution.wait()
            assert waited.result.output == {'text': 'done'}
            identity = execution.execution_id
            models = await execution.model_interactions(include_content=True)
            assert models.items
    from linktools.ai.core import TenantAuthorizationPolicy

    async with (
        Persistence.open('default', storage=_local_runtime_storage(workspace), tenant_id="default", read_only=True) as persistence,
        History.open(persistence=persistence, authorization=TenantAuthorizationPolicy("default")) as history,
        client(create_app(history=history, local_workspace=True)) as http,
    ):
        if session_bound:
            params = {'session_id': 'session', 'turn_id': identity}
            turn_response = await http.get('/api/session', params=params)
            assert turn_response.status_code == 200, turn_response.text
            published = turn_response.json()
            assert published['conversation_committed']
            assert published['user_input']['prompt'] == 'hello'
            assert [item['content'] for item in published['items'] if item['item_kind'] == 'assistant'] == ['done']
            assert 'output' not in published
            trace = await http.get('/api/session', params={**params, 'trace': 'true'})
            assert trace.status_code == 200, trace.text
            assert trace.json()['items'] == [asdict(item) for item in trace_before_close.items]
        else:
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
    from linktools.ai.core import TenantAuthorizationPolicy
    from linktools.ai.history import History
    from linktools.commands.ai._common import _load_workspace, _local_runtime_storage

    monkeypatch.setattr(common_command._ModelOptions, 'create_model_binding', lambda _self: RuntimeUsageModels())
    args = run_command.command.create_parser().parse_args([
        '--project', str(tmp_path), '--model', 'fake', '--json', 'hello',
    ])
    assert run_command.command.run(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'SUCCEEDED'
    assert result['output'] == {'text': 'done'}

    async def inspect() -> None:
        workspace = _load_workspace(str(tmp_path))
        async with (
            Persistence.open('default', storage=_local_runtime_storage(workspace), tenant_id='default', read_only=True) as persistence,
            History.open(persistence=persistence, authorization=TenantAuthorizationPolicy('default')) as history,
        ):
            assert history.has_turn_sessions
            async with client(create_app(history=history, local_workspace=True)) as http:
                params = {"session_id": "default", "turn_id": result["turn_id"]}
                response = await http.get("/api/session", params=params)
                assert response.status_code == 200, response.text
                saved = response.json()
                assert saved["turn_id"] == result["turn_id"] and saved["conversation_committed"]
                assert saved["user_input"]["prompt"] == "hello"
                assert [(item["item_kind"], item["content"]) for item in saved["items"]] == [("assistant", "done")]
                trace = await http.get("/api/session", params={**params, "trace": "true"})
                assert trace.status_code == 200, trace.text
                items = trace.json()["items"]
                # This one-shot command does not wait for optional Trace.
                # Any already committed observation still belongs to this turn.
                assert all(item["turn_id"] == result["turn_id"] for item in items)

    asyncio.run(inspect())
