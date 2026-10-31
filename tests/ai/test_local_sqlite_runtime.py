#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Managed SQLite roots retain one physical writer without changing shared SQL."""

from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize('other', ['root', 'sqlite', 'engine'])
async def test_local_root_rejects_other_writable_sources_but_allows_readers(tmp_path: Path, other: str) -> None:
    root = tmp_path / 'root'
    storage = PersistenceStorage.from_root(root, exclusive_writer=True)
    engine = create_async_engine(f'sqlite+aiosqlite:///{root / "runtime.db"}')
    first = Persistence(storage)
    await first.initialize(namespace='owner', tenant_id='tenant')
    try:
        assert first.exclusive_execution_writer
        selected = (storage if other == 'root' else PersistenceStorage.sqlite(root / 'runtime.db')
                    if other == 'sqlite' else PersistenceStorage.sql(engine, object_store=storage.object_store))
        second = Persistence(selected)
        with pytest.raises(AIError) as rejected:
            await second.initialize(namespace='another', tenant_id='other')
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        assert first.ready and first.exclusive_execution_writer
        reader = Persistence(selected)
        await reader.initialize(namespace='owner', tenant_id='tenant', read_only=True)
        assert not reader.exclusive_execution_writer
        await reader.close()
    finally:
        await first.close()
        await engine.dispose()
    reopened = Persistence(storage)
    await reopened.initialize(namespace='owner', tenant_id='tenant')
    assert reopened.exclusive_execution_writer
    await reopened.close()


async def test_existing_shared_from_root_preserves_its_profile_and_shared_writers(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    source = Persistence(PersistenceStorage.sqlite(root / 'runtime.db'))
    await source.initialize(namespace='old', tenant_id='tenant')
    try:
        existing = Persistence(PersistenceStorage.from_root(root))
        await existing.initialize(namespace='old', tenant_id='tenant')
        try:
            assert not source.exclusive_execution_writer
            assert not existing.exclusive_execution_writer
        finally:
            await existing.close()
    finally:
        await source.close()


async def test_exclusive_request_does_not_retag_an_existing_shared_root(tmp_path: Path) -> None:
    root = tmp_path / 'root'
    original = Persistence(PersistenceStorage.from_root(root))
    await original.initialize(namespace='old', tenant_id='tenant')
    await original.close()
    database = root / 'runtime.db'
    before = database.read_bytes()
    requested = Persistence(PersistenceStorage.from_root(root, exclusive_writer=True))
    with pytest.raises(AIError) as rejected:
        await requested.initialize(namespace='old', tenant_id='tenant')
    assert rejected.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert database.read_bytes() == before
    normal = Persistence(PersistenceStorage.from_root(root))
    await normal.initialize(namespace='old', tenant_id='tenant')
    assert not normal.exclusive_execution_writer
    await normal.close()


async def test_read_only_root_does_not_request_or_change_shared_writer_profile(tmp_path: Path) -> None:
    async with Persistence.open('reader', storage=PersistenceStorage.from_root(tmp_path)):
        pass
    database = tmp_path / 'runtime.db'
    before = database.read_bytes()
    async with Persistence.open('reader', read_only=True,
        storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True)) as reader:
        assert not reader.exclusive_execution_writer
    assert database.read_bytes() == before


@pytest.mark.parametrize('auto_recover', [False, True])
async def test_local_root_keeps_one_agent_producer_while_allowing_task_only_views(
    tmp_path: Path, auto_recover: bool,
) -> None:
    import asyncio
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.runtime import Runtime
    from ._runtime_test_helpers import _Models, model_features
    from .test_terminal_commit_shared_runtime import _group

    entered, release = asyncio.Event(), asyncio.Event()
    calls = 0

    async def answer(messages, info):
        nonlocal calls
        calls += 1
        entered.set()
        await release.wait()
        yield 'one producer'

    features = model_features(_Models(FunctionModel(stream_function=answer)), _group(False))
    async with Persistence.open('local-runtime', storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True)) as state:
        async with Runtime.attach(persistence=state, features=features) as first, Runtime.attach(persistence=state):
            execution = await first.agents.get().start('only one owner', idempotency_key='same')
            await asyncio.wait_for(entered.wait(), 5)
            second = Runtime.attach(persistence=state, features=features, auto_recover=auto_recover)
            try:
                with pytest.raises(AIError) as rejected:
                    await second.__aenter__()
                assert rejected.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
                assert calls == 1
            finally:
                release.set()
                await second.close()
            await asyncio.wait_for(execution.wait(), 10)
            assert calls == 1


@pytest.mark.parametrize('auto_recover', [False, True])
async def test_local_sqlite_process_exit_preserves_same_execution_recovery(tmp_path: Path, auto_recover: bool) -> None:
    import asyncio
    import json
    import sys
    import textwrap
    from pydantic_ai.models.function import FunctionModel
    from linktools.ai.core import ExecutionStatus
    from linktools.ai.runtime import Runtime
    from ._runtime_test_helpers import model_features
    from .test_live_history_readback_integration import _Models
    from .test_runtime_auto_recovery import _capabilities

    child = textwrap.dedent('''
        import asyncio, json, sys
        from pydantic_ai.models.function import FunctionModel
        from linktools.ai.persistence import Persistence, PersistenceStorage
        from linktools.ai.runtime import Runtime
        from tests.ai._runtime_test_helpers import model_features
        from tests.ai.test_live_history_readback_integration import _Models
        from tests.ai.test_runtime_auto_recovery import _capabilities

        async def model(messages, info):
            raise AssertionError('old process must not invoke the provider')
            yield ''

        async def main():
            storage = PersistenceStorage.from_root(sys.argv[1], exclusive_writer=True)
            async with Persistence.open('startup-recovery', storage=storage) as state:
                async with Runtime.attach(persistence=state, features=model_features(_Models(FunctionModel(stream_function=model)), _capabilities())) as runtime:
                    async def interrupted_launch(*args, **kwargs):
                        pass
                    runtime._execution_service.runtime_backend().launch = interrupted_launch
                    execution = await runtime.agents.get().start('preserve original admission', idempotency_key='same')
                    print('READY=' + json.dumps({'execution_id': execution.execution_id}), flush=True)
                    await asyncio.Event().wait()
        asyncio.run(main())
    ''')
    process = await asyncio.create_subprocess_exec(sys.executable, '-c', child, str(tmp_path),
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    try:
        assert process.stdout is not None
        while True:
            line = await asyncio.wait_for(process.stdout.readline(), 20)
            if not line:
                raise AssertionError((await process.communicate())[1].decode())
            if line.startswith(b'READY='):
                execution_id = json.loads(line.removeprefix(b'READY='))['execution_id']
                break
        contender = Persistence(PersistenceStorage.from_root(tmp_path, exclusive_writer=True))
        with pytest.raises(AIError) as rejected:
            await contender.initialize(namespace='startup-recovery', tenant_id='default')
        assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
        process.kill()
        await asyncio.wait_for(process.communicate(), 10)
    finally:
        if process.returncode is None:
            process.kill()
            await process.communicate()

    calls = []

    async def recovered(messages, info):
        calls.append(messages)
        yield 'recovered once'

    async with Persistence.open('startup-recovery', storage=PersistenceStorage.from_root(tmp_path, exclusive_writer=True)) as state:
        async with Runtime.attach(persistence=state,
            features=model_features(_Models(FunctionModel(stream_function=recovered)), _capabilities()),
            auto_recover=auto_recover) as runtime:
            if not auto_recover:
                assert not calls
                assert (await runtime.executions.inspect(execution_id, principal=runtime.default_principal)).status is ExecutionStatus.STARTED
                await runtime.executions.recover(execution_id, principal=runtime.default_principal, idempotency_key='recover')
            result = (await runtime.executions.wait(execution_id, principal=runtime.default_principal, timeout_seconds=10)).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {'text': 'recovered once'}
            replay = await runtime.agents.get().start('preserve original admission', idempotency_key='same')
            assert replay.execution_id == execution_id
            assert len(calls) == 1


async def test_local_sqlite_unknown_terminal_keeps_owner_until_same_commit_is_confirmed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import asyncio
    from pydantic_ai.messages import ModelResponse, TextPart
    from linktools.ai.capability import ExecutionCapability
    from linktools.ai.core import CommitPolicy, ExecutionStatus
    from linktools.ai.runtime import Runtime, RuntimeFeatures
    from linktools.ai.storage import InMemoryCacheBackend
    from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features

    calls, writes = [], []

    async def model(messages, info):
        calls.append(messages)
        return ModelResponse(parts=[TextPart('confirmed once')])

    features = RuntimeFeatures('local-unknown')
    features.capability(ExecutionCapability(retain_history=True, commit=CommitPolicy.terminal()), scope='runtime')
    features.agent('default', model='default', allow_tools=())
    storage = PersistenceStorage.from_root(tmp_path, exclusive_writer=True,
        coordination=InMemoryCacheBackend(max_bytes=4 * 1024 * 1024))
    async with Persistence.open('local-unknown', storage=storage) as state:
        async with Runtime.attach(persistence=state,
            features=model_features(_Models(_UsageFunctionModel(model)), features)) as runtime:
            session = await runtime.agents.get().create_session('session')
            group = state.execution.executions.state_store.storage_group
            original_mutate, original_read = type(group).mutate, type(group).read_many
            committed = asyncio.Event()
            unavailable = True

            async def lose_ack(owner, stores, callback):
                result = await original_mutate(owner, stores, callback)
                if owner is group:
                    writes.append(1)
                    if len(writes) == 1:
                        committed.set()
                        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
                return result

            async def fail_confirmation(owner, stores, callback):
                if owner is group and committed.is_set() and unavailable:
                    raise OSError('source confirmation temporarily unavailable')
                return await original_read(owner, stores, callback)

            monkeypatch.setattr(type(group), 'mutate', lose_ack)
            monkeypatch.setattr(type(group), 'read_many', fail_confirmation)
            execution = await session.start('question', idempotency_key='same')
            await asyncio.wait_for(committed.wait(), 5)
            with pytest.raises(AIError) as unknown:
                await execution.wait(timeout_seconds=5)
            assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            with pytest.raises(AIError) as unsettled:
                await runtime.close()
            assert unsettled.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            with pytest.raises(AIError) as borrowed:
                await state.close()
            assert borrowed.value.code is ErrorCode.RUNTIME_DEPENDENCY_NOT_READY
            assert state.ready and state.exclusive_execution_writer
            contender = Persistence(storage)
            with pytest.raises(AIError) as locked:
                await contender.initialize(namespace='local-unknown', tenant_id='default')
            assert locked.value.code is ErrorCode.STORAGE_CONFLICT
            assert len(calls) == len(writes) == 1

            unavailable = False
            await runtime.close()
            stored = await state.execution.executions.get(execution.execution_id, tenant_id='default')
            assert stored is not None and stored.status is ExecutionStatus.SUCCEEDED
            assert len(calls) == len(writes) == 1
    async with Persistence.open('local-unknown', storage=storage) as reopened:
        assert reopened.exclusive_execution_writer
        async with Runtime.attach(persistence=reopened,
            features=model_features(_Models(_UsageFunctionModel(model)), features)) as following:
            session = following.agents.get().session('session')
            replay = await session.start('question', idempotency_key='same')
            assert replay.execution_id == execution.execution_id
            result = (await replay.wait()).result
            assert result.status is ExecutionStatus.SUCCEEDED
            assert result.output == {'text': 'confirmed once'}
            assert len(calls) == len(writes) == 1
