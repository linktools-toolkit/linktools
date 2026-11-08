#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native checkpoint envelopes preserve content without retyping business JSON."""

import json
import sqlite3
from collections.abc import AsyncIterator
from dataclasses import replace
from pathlib import Path

import pytest
import pytest_asyncio
from pydantic_ai.messages import (
    ModelMessagesTypeAdapter, ModelRequest, StructuredContent, TextContent, ToolReturnPart, UserPromptPart,
)
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, FileStepStore, InMemoryStepStore, RunRecord, SqliteStepStore, StepStore

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.model.messages import native_checkpoint_messages
from linktools.ai.model._user_content_codec import RetiredReferencedContent
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence._official_step_reader import OfficialStepReader
from linktools.ai.runtime._step_store import RuntimeStepStore

_GOLDEN = json.loads((Path(__file__).parent / 'fixtures/native_checkpoint_v1_golden.json').read_text())
_COLLISION = {'linktools-content': {'type': 'runtime-reference', 'value': {'ordinary': True}}}


@pytest_asyncio.fixture(params=['memory', 'file', 'sqlite', 'project-sql'])
async def store(request: pytest.FixtureRequest, tmp_path: Path) -> AsyncIterator[StepStore]:
    engine = connection = None
    if request.param == 'memory':
        value = InMemoryStepStore()
    elif request.param == 'file':
        value = FileStepStore(tmp_path / 'native')
    elif request.param == 'sqlite':
        connection = sqlite3.connect(tmp_path / 'native.db', check_same_thread=False)
        value = SqliteStepStore(connection=connection)
    else:
        from sqlalchemy.ext.asyncio import create_async_engine
        engine = create_async_engine('sqlite+aiosqlite:///' + str(tmp_path / 'project.db'))
        value = SqlStepStore(engine, namespace='content', tenant_id='tenant', max_cached_messages=0)
        await value.initialize(local=True)
    try:
        yield value
    finally:
        if engine is not None:
            await value.close()
            await engine.dispose()
        if connection is not None:
            connection.close()


def _wire(messages):
    return json.loads(ModelMessagesTypeAdapter.dump_json(messages))


def _item(case):
    if case in ('reference', 'retired'):
        reference = native_checkpoint_messages(ModelMessagesTypeAdapter.validate_python(_GOLDEN['reference']), restore=True)[0].parts[0].content[0]
        return reference if case == 'reference' else RetiredReferencedContent(reference, 'retired summary')
    if case == 'text':
        return TextContent('ordinary', metadata=_COLLISION)
    if case == 'structured':
        return StructuredContent({'state': [1, True]}, content='rendered', tag='application', metadata=_COLLISION)
    if case == 'valid-reference-json':
        return _GOLDEN['reference'][0]['parts'][0]['content'][0]
    if case == 'nested-json':
        return {'nested': [{'kind': 'text-content', 'content': 'ordinary', 'metadata': _COLLISION}]}
    if case == 'plain-json':
        return {'kind': 'text-content', 'content': 'ordinary', 'metadata': _COLLISION}
    if case == 'new-envelope-json':
        return {'kind': 'text-content', 'content': 'ordinary', 'metadata': {'linktools-content': {
            'type': 'runtime-reference', 'version': 2, 'value': {'kind': 'json', 'value': 42},
        }}}
    raise AssertionError(case)


@pytest.mark.asyncio
@pytest.mark.parametrize('case,position', [
    ('reference', 'user'), ('retired', 'tool'),
    ('text', 'user'), ('structured', 'user'), ('text', 'tool'), ('structured', 'tool-list'),
    ('plain-json', 'tool'), ('valid-reference-json', 'tool-list'),
    ('nested-json', 'tool'), ('new-envelope-json', 'tool'),
])
async def test_native_checkpoint_current_content_cold_read_and_resave(store: StepStore, case: str, position: str) -> None:
    item = _item(case)
    part = UserPromptPart([item]) if position == 'user' else ToolReturnPart('tool', [item] if position == 'tool-list' else item, 'call')
    messages = [ModelRequest(parts=[part])]
    writer = RuntimeStepStore(store, InMemoryStepStore())
    await writer.register_run(RunRecord(run_id='run'))
    await writer.save_snapshot(ContinuableSnapshot(run_id='run', step_index=1, idempotency_key='first', messages=messages))
    raw = await store.latest_snapshot(run_id='run')
    reader = OfficialStepReader(store)
    restored = await reader.latest_snapshot(run_id='run')
    assert _wire(restored.messages) == _wire(messages)
    found = restored.messages[0].parts[0].content
    found = found[0] if position in ('user', 'tool-list') else found
    assert type(found) is type(item)
    if case == 'structured':
        assert found.value == item.value and found.tag == item.tag
    assert _wire((await writer.latest_snapshot(run_id='run')).messages) == _wire(messages)
    await writer.save_snapshot(replace(restored, step_index=2, idempotency_key='second'))
    assert _wire((await store.latest_snapshot(run_id='run')).messages) == _wire(raw.messages)


@pytest.mark.asyncio
@pytest.mark.parametrize('name', ['reference', 'ordinary_new_type'])
async def test_native_checkpoint_legacy_writer_fixture_remains_readable(store: StepStore, name: str) -> None:
    messages = ModelMessagesTypeAdapter.validate_python(_GOLDEN[name])
    await store.register_run(RunRecord(run_id='old'))
    await store.save_snapshot(ContinuableSnapshot(run_id='old', step_index=1, messages=messages))
    restored = await OfficialStepReader(store).latest_snapshot(run_id='old')
    if name == 'reference':
        assert isinstance(restored.messages[0].parts[0].content[0], ReferencedContent)
    else:
        assert _wire(restored.messages) == _GOLDEN[name]
    assert _wire(native_checkpoint_messages(restored.messages)) == _GOLDEN[name]


@pytest.mark.asyncio
@pytest.mark.parametrize('damage', ['legacy', 'unknown-version', 'bool-version', 'native', 'kind', 'user-dict', 'user-list', 'user-int', 'user-null'])
async def test_native_checkpoint_cold_read_corruption_has_typed_error(store: StepStore, damage: str) -> None:
    marker = {'type': 'runtime-reference', 'version': 2, 'value': {'kind': 'native', 'value': {'kind': 'text-content', 'content': 'text', 'metadata': None}}}
    if damage == 'legacy':
        marker = {'type': 'runtime-reference', 'value': {'ordinary': True}}
    elif damage == 'unknown-version':
        marker['version'] = 99
    elif damage == 'bool-version':
        marker['version'] = True
    elif damage == 'native':
        marker['value']['value'] = {'kind': 'not-native'}
    elif damage == 'kind':
        marker['value']['kind'] = 'unknown'
    else:
        marker['value'] = {'kind': 'json', 'value': {'user-dict': {'ordinary': True}, 'user-list': [1], 'user-int': 1, 'user-null': None}[damage]}
    messages = [ModelRequest(parts=[UserPromptPart([TextContent('stored', metadata={'linktools-content': marker})])])]
    await store.register_run(RunRecord(run_id='bad'))
    await store.save_snapshot(ContinuableSnapshot(run_id='bad', step_index=1, messages=messages))
    with pytest.raises(AIError) as caught:
        await OfficialStepReader(store).latest_snapshot(run_id='bad')
    assert caught.value.code == ErrorCode.STORAGE_INTEGRITY_ERROR


def test_native_checkpoint_adapter_allows_ordinary_extra_native_fields() -> None:
    marker = {'type': 'runtime-reference', 'version': 2, 'value': {'kind': 'native', 'value': {
        'kind': 'text-content', 'content': 'text', 'metadata': None, 'future_ordinary_field': 1,
    }}}
    restored = native_checkpoint_messages([ModelRequest(parts=[UserPromptPart([TextContent('stored', metadata={'linktools-content': marker})])])], restore=True)
    assert restored[0].parts[0].content[0].content == 'text'
