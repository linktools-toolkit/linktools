#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Unmodified upstream StepPersistence measurement, not a LinkTools contract."""
import importlib.metadata
import json
import os
import sqlite3
import time
from collections import Counter
from pathlib import Path

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart
from pydantic_ai.models.function import FunctionModel
from pydantic_ai_harness.step_persistence import SqliteStepStore, StepPersistence

pytestmark = pytest.mark.skipif(not os.getenv('UPSTREAM_STEP_OUTPUT'), reason='Explicit measurement only')


def _bytes(value: object) -> int:
    if isinstance(value, str):
        return len(value.encode())
    if isinstance(value, bytes):
        return len(value)
    if isinstance(value, (tuple, list)):
        return sum(map(_bytes, value))
    return 0


@pytest.mark.asyncio
@pytest.mark.parametrize('capture_frontier', [False, True])
@pytest.mark.parametrize('payload_bytes', [4, 81920])
async def test_two_turn_upstream_step(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
                                      capture_frontier: bool, payload_bytes: int) -> None:
    counts: Counter[str] = Counter()
    statements: list[dict[str, object]] = []
    active = False
    original_connect = sqlite3.connect

    class CountedConnection(sqlite3.Connection):
        dirty = False

        def execute(self, sql, parameters=(), /):
            before = self.in_transaction
            result = super().execute(sql, parameters)
            self.observe(sql, parameters, 1, before)
            return result

        def executemany(self, sql, parameters, /):
            parameters = list(parameters)
            before = self.in_transaction
            result = super().executemany(sql, parameters)
            self.observe(sql, parameters, len(parameters), before)
            return result

        def observe(self, sql, parameters, rows, before):
            if not active:
                return
            word = sql.split()[0].upper()
            counts['api.' + word] += 1
            counts['executions.' + word] += rows
            counts['bind_text_blob_bytes'] += _bytes(parameters)
            statements.append({'sql': sql, 'rows': rows, 'bytes': _bytes(parameters)})
            if word in {'INSERT', 'UPDATE', 'DELETE', 'REPLACE'}:
                if self.in_transaction:
                    self.dirty = True
                else:
                    counts['write_commits'] += rows
                    counts['autocommit_writes'] += rows
            elif word == 'COMMIT' and before:
                counts['explicit_commits'] += 1
                if self.dirty:
                    counts['write_commits'] += 1
                self.dirty = False

    def connect(*args, **kwargs):
        if active:
            counts['connections'] += 1
        return original_connect(*args, **kwargs, factory=CountedConnection)

    monkeypatch.setattr(sqlite3, 'connect', connect)
    store = SqliteStepStore(database=tmp_path / 'upstream.db')
    await store.list_runs()  # Explicit schema initialization is outside measurement.
    calls: Counter[str] = Counter()
    return_counts: list[int] = []
    tool_calls = 0

    def model(messages, info):
        question = next(part.content for message in reversed(messages)
                        for part in reversed(message.parts) if isinstance(part, UserPromptPart))
        calls[question] += 1
        return_counts.append(sum(isinstance(part, ToolReturnPart)
                                 for message in messages for part in message.parts))
        if calls[question] <= 3:
            return ModelResponse(parts=[ToolCallPart('work', {}, tool_call_id=f'{question}-{calls[question]}')])
        return ModelResponse(parts=[TextPart('answer')])

    agent = Agent(FunctionModel(model), name='upstream_measurement',
                  capabilities=[StepPersistence(store=store, capture_frontier=capture_frontier)])

    @agent.tool_plain
    def work() -> str:
        nonlocal tool_calls
        tool_calls += 1
        return 'done' if payload_bytes == 4 else 'x' * payload_bytes

    history = []
    turns = []
    elapsed = 0.0
    for turn in range(2):
        before = counts.copy()
        active = True
        started = time.perf_counter()
        result = await agent.run(f'question-{turn}', message_history=history, conversation_id='measurement')
        elapsed += time.perf_counter() - started
        active = False
        assert result.output == 'answer'
        history = result.all_messages()
        turns.append(dict(counts - before))
    assert tool_calls == 6
    assert sum(calls.values()) == 8
    assert return_counts == [0, 1, 2, 3, 3, 4, 5, 6]
    assert len(history) == 16
    runs = await store.list_runs()
    assert len(runs) == 2
    snapshots = [await store.latest_snapshot(run_id=run.run_id) for run in runs]
    assert [len(snapshot.messages) for snapshot in snapshots] == [8, 16]
    for run in runs:
        assert not await store.list_unresolved_tool_effects(run_id=run.run_id)
    with original_connect(tmp_path / 'upstream.db') as connection:
        table_rows = {name: connection.execute(f'SELECT count(*) FROM {name}').fetchone()[0]
                      for name in ['runs', 'events', 'snapshots', 'snapshot_messages', 'snapshot_message_refs', 'tool_effects']}
        stored_body_bytes = connection.execute('SELECT sum(length(CAST(message AS BLOB))) FROM snapshot_messages').fetchone()[0]
    result_record = {'capture_frontier': capture_frontier, 'payload_bytes': payload_bytes,
                     'versions': {name: importlib.metadata.version(name) for name in ['pydantic-ai-slim', 'pydantic-ai-harness']},
                     'counts': dict(counts), 'turns': turns, 'seconds': elapsed, 'table_rows': table_rows,
                     'stored_message_body_bytes': stored_body_bytes,
                     'workload': {'turns': 2, 'tools': tool_calls, 'model_calls': sum(calls.values()), 'messages': len(history)},
                     'statements': statements}
    output = Path(os.environ['UPSTREAM_STEP_OUTPUT'])
    with output.open('a') as stream:
        stream.write(json.dumps(result_record) + '\n')


@pytest.mark.asyncio
async def test_upstream_checkpoint_cold_continue_uses_saved_tool_result(tmp_path: Path) -> None:
    from pydantic_ai_harness.step_persistence import continue_run

    database = tmp_path / 'recovery.db'
    store = SqliteStepStore(database=database)
    effects = []

    def failing_model(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            raise RuntimeError('model interrupted after settled tool')
        return ModelResponse(parts=[ToolCallPart('effect', {}, tool_call_id='effect-1')])

    agent = Agent(FunctionModel(failing_model), name='initial',
                  capabilities=[StepPersistence(store=store, capture_frontier=True)])

    @agent.tool_plain
    def effect() -> str:
        effects.append('performed')
        return 'saved-result'

    with pytest.raises(RuntimeError, match='model interrupted'):
        await agent.run('question', conversation_id='recovery')
    original = (await store.list_runs())[0]
    cold = SqliteStepStore(database=database)
    restored = await continue_run(cold, run_id=original.run_id)
    assert [part.content for message in restored for part in message.parts if isinstance(part, ToolReturnPart)] == ['saved-result']
    assert not await cold.list_unresolved_tool_effects(run_id=original.run_id)

    def resumed_model(messages, info):
        assert any(isinstance(part, ToolReturnPart) and part.content == 'saved-result'
                   for message in messages for part in message.parts)
        return ModelResponse(parts=[TextPart('recovered')])

    resumed = Agent(FunctionModel(resumed_model), name='resumed',
                    capabilities=[StepPersistence(store=cold, capture_frontier=True)])
    result = await resumed.run(message_history=restored, conversation_id='recovery')
    assert result.output == 'recovered'
    assert effects == ['performed']
    assert len(await cold.list_runs()) == 2


@pytest.mark.asyncio
async def test_upstream_capability_runs_through_existing_runtime(tmp_path: Path) -> None:
    from linktools.ai.capability import AgentContext, ExecutionCapability
    from linktools.ai.persistence import Persistence, PersistenceStorage
    from linktools.ai.runtime import Runtime, RuntimeFeatures

    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    store = SqliteStepStore(database=tmp_path / 'native.db')
    features = RuntimeFeatures('upstream_passthrough')
    features.capability(ExecutionCapability(retain_history=True), scope='runtime')
    features.capability(StepPersistence(store=store, capture_frontier=True), scope='runtime')
    effects = []

    async def work(_context: AgentContext[None]) -> str:
        effects.append('done')
        return 'native-result'

    async def model(messages, info):
        if any(isinstance(part, ToolReturnPart) for message in messages for part in message.parts):
            return ModelResponse(parts=[TextPart('answer')])
        return ModelResponse(parts=[ToolCallPart('work', {}, tool_call_id='native-tool')])

    features.tool(work, effect_policy='replay_safe')
    features.agent('default', model='default', allow_tools=('work',))
    async with Persistence.open('upstream_passthrough', storage=PersistenceStorage.sqlite(tmp_path / 'runtime.db')) as persistence:
        async with Runtime.attach(persistence=persistence, features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get().create_session('session')
            execution = await session.start('question', idempotency_key='question')
            result = await execution.wait(timeout_seconds=10)
            assert result.result.output == {'text': 'answer'}
            replay = await session.start('question', idempotency_key='question')
            assert replay.execution_id == execution.execution_id
            assert effects == ['done']
    run = (await store.list_runs())[0]
    snapshot = await store.latest_snapshot(run_id=run.run_id)
    assert [part.content for message in snapshot.messages for part in message.parts if isinstance(part, ToolReturnPart)] == ['native-result']
