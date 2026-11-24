#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Confirmed preparation inputs never replace authoritative commit validation."""
from dataclasses import replace
from pathlib import Path
import pytest
from pydantic_ai.messages import ModelResponse, TextPart
from linktools.ai.errors import AIError
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._archive import PreparedArchive
from linktools.ai.persistence._model_interaction import StagedModelInteraction, StagedContextProjection, StagedContextSpan
from linktools.ai.persistence._step_contracts import AgentRunRecord
from .test_incremental_projection_boundaries import _storage
from .test_checkpoint_metadata_snapshot_io import _trace_archive

@pytest.mark.asyncio
@pytest.mark.parametrize('external', ['FAILED', 'CANCELLED'])
async def test_retained_preparation_does_not_overwrite_independent_archive_writer(tmp_path: Path, monkeypatch, external):
    state = Persistence(_storage('sqlite', tmp_path))
    await state.initialize(namespace='prepared-projection', tenant_id='tenant')
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord('run')
        staged = StagedModelInteraction('run', 1, 1, 'agent', None, {}, None, None, None,
                                       'RUNNING', None, None, None, run.started_at, None)
        known, = await archive.prepare_interactions(run, (staged,), lambda _: b'{}', prepared=PreparedArchive())
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(known,), execution_id='execution')
        foreign = replace(known, status=external, duration_ns=1, finished_at=run.started_at,
                          error_code='MODEL_ERROR' if external == 'FAILED' else None)
        # This independent archive publication advances durable state after our snapshot.
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(foreign,), execution_id='execution')
        messages = (ModelResponse(parts=[TextPart('stale success')], timestamp=run.started_at),)
        successor = replace(staged, status='SUCCEEDED', duration_ns=1, finished_at=run.started_at,
                            response_context=StagedContextProjection((StagedContextSpan(0, 1),)))
        async def forbid_read(**kwargs):
            raise AssertionError('hot preparation must not query interactions')
        monkeypatch.setattr(archive, 'list_model_interactions', forbid_read)
        bundle = PreparedArchive()
        prepared, = await archive._prepare_interactions(run, (successor,), lambda _: b'{}',
            local_message_count=1, local_messages=messages, prepared=bundle,
            replay=True, confirmed_interactions=(known,))
        with pytest.raises(AIError):
            await archive.sync_projection(run, events=(), checkpoints=(), interactions=(prepared,),
                                          archive=bundle, execution_id='execution')
        key = archive._interaction_key('run', 1)
        current = await archive.state_store.read(lambda tx: tx.get_record(key))
        assert archive._decode_interaction_record(current) == foreign
    finally:
        await state.close()

@pytest.mark.asyncio
async def test_cold_or_other_owner_retained_context_uses_authoritative_replay(tmp_path, monkeypatch):
    state = Persistence(_storage('sqlite', tmp_path))
    await state.initialize(namespace='prepared-projection', tenant_id='tenant')
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord('run')
        staged = StagedModelInteraction('run', 1, 1, 'agent', None, {}, None, None, None,
                                       'RUNNING', None, None, None, run.started_at, None)
        known, = await archive.prepare_interactions(run, (staged,), lambda _: b'{}', prepared=PreparedArchive())
        calls = []
        async def read(**kwargs):
            calls.append(kwargs)
            return [known]
        monkeypatch.setattr(archive, 'list_model_interactions', read)
        for retained in ((), (replace(known, agent_run_id='other'),)):
            await archive._prepare_interactions(run, (staged,), lambda _: b'{}',
                prepared=PreparedArchive(), replay=True, confirmed_interactions=retained)
        assert len(calls) == 2
    finally:
        await state.close()

@pytest.mark.asyncio
async def test_acknowledgment_loss_never_retains_interaction_hint(tmp_path, monkeypatch):
    from linktools.ai.persistence import runs
    from .test_projection_receipt_ack_loss import test_projection_commit_ack_loss_reads_back_trace_without_delivering_receipt
    original_commit = runs.run_durable_commit
    original_finalize = runs.RuntimeAgentRunStore.finalize_execution_projection
    outcomes = []
    inspected = []
    async def commit(*args, **kwargs):
        result = await original_commit(*args, **kwargs)
        outcomes.append(result)
        return result
    async def finalize(self, *args, **kwargs):
        if outcomes and outcomes[-1].error is not None:
            inspected.append(kwargs.get('confirmed_interactions', ()))
        return await original_finalize(self, *args, **kwargs)
    monkeypatch.setattr(runs, 'run_durable_commit', commit)
    monkeypatch.setattr(runs.RuntimeAgentRunStore, 'finalize_execution_projection', finalize)
    await test_projection_commit_ack_loss_reads_back_trace_without_delivering_receipt(tmp_path, monkeypatch)
    assert inspected and all(not value for value in inspected)

@pytest.mark.asyncio
@pytest.mark.parametrize('retained', [False, True])
async def test_retained_context_still_verifies_stored_bytes(tmp_path, retained):
    from linktools.ai.persistence import NativeMessageRoot
    from linktools.ai.persistence._step_schema import build_step_sql_metadata
    from .test_history_projection_conformance import _message_content_records
    state = Persistence(_storage('sqlite', tmp_path))
    await state.initialize(namespace='prepared-projection', tenant_id='tenant')
    try:
        archive = await _trace_archive(state)
        run = AgentRunRecord('run')
        messages = (ModelResponse(parts=[TextPart('committed bytes')], timestamp=run.started_at),)
        staged = StagedModelInteraction('run', 1, 1, 'agent', None, {}, None, None,
            StagedContextProjection((StagedContextSpan(0, 1),)), 'SUCCEEDED', None, 1,
            None, run.started_at, run.started_at)
        bundle = PreparedArchive()
        known, = await archive.prepare_interactions(run, (staged,), lambda _: b'{}',
            local_message_count=1, local_messages=messages, prepared=bundle)
        await archive.sync_projection(run, events=(), checkpoints=(), interactions=(known,),
                                      archive=bundle, execution_id='execution')
        root = known.response_context.items[0].root
        contents = await archive.transcript_repository.content_archive()
        content, = await _message_content_records(contents, (root,))
        if isinstance(root, NativeMessageRoot):
            table = build_step_sql_metadata().tables['ai_step_messages']
            async with contents.store.context.sessions() as sql, sql.begin():
                changed = await sql.execute(table.update().where(
                    table.c.store_key == root.store_key, table.c.sha256 == content.raw_digest,
                ).values(message=b'corrupted', compression='zlib'))
                assert changed.rowcount == 1
        else:
            key = contents._key('archive_content', content.identity)
            async def corrupt(tx):
                record = await tx.get_record(key)
                return await tx.replace_record(replace(record, payload_bytes=b'corrupted',
                    storage_revision=record.storage_revision + 1), expected_storage_revision=record.storage_revision)
            assert await contents.store.mutate(corrupt)
        with pytest.raises(AIError):
            await archive._prepare_interactions(run, (staged,), lambda _: b'{}',
                local_message_count=1, local_messages=messages, prepared=PreparedArchive(), replay=True,
                confirmed_interactions=(known,) if retained else ())
    finally:
        await state.close()

@pytest.mark.asyncio
@pytest.mark.parametrize('same_producer', [True, False])
async def test_runtime_reuses_interactions_only_for_confirmed_producer(tmp_path, monkeypatch, same_producer):
    from linktools.ai.session import CreateSessionRequest
    from linktools.ai.persistence.runs import RuntimeAgentRunStore
    from linktools.ai.persistence._step_archive import StateStepArchive
    from linktools.ai.workspace import Workspace
    from linktools.commands.ai._common import _open_local_runtime
    from ._runtime_test_helpers import _Models, _UsageFunctionModel
    prepare = StateStepArchive._prepare_interactions
    finalize = RuntimeAgentRunStore.finalize_execution_projection
    hints = []
    async def observed(self, *args, **kwargs):
        if kwargs.get('replay'):
            hints.append(bool(kwargs.get('confirmed_interactions')))
        return await prepare(self, *args, **kwargs)
    async def finalized(self, flight, *args, **kwargs):
        result = await finalize(self, flight, *args, **kwargs)
        if not same_producer:
            self.trace_persistence.offsets[flight.agent_run_id].confirmed_interaction_producer = ('other-execution', 1)
        return result
    monkeypatch.setattr(StateStepArchive, '_prepare_interactions', observed)
    monkeypatch.setattr(RuntimeAgentRunStore, 'finalize_execution_projection', finalized)
    def respond(_messages, _info):
        return ModelResponse(parts=[TextPart('complete')])
    async with _open_local_runtime(Workspace.initialize(tmp_path), model=_Models(_UsageFunctionModel(respond))) as local:
        runtime = local.runtime
        actor = runtime.default_principal
        await runtime.sessions.create('default', CreateSessionRequest(actor, 'session', 'create'))
        run = await runtime.sessions.start('session', 'hello', principal=actor, idempotency_key='turn')
        assert (await run.wait(timeout_seconds=20)).result.output == {'text': 'complete'}
    assert hints
    assert any(hints) is same_producer

@pytest.mark.asyncio
@pytest.mark.parametrize('change', ['generation', 'execution'])
async def test_takeover_before_finalize_never_relabels_old_confirmed_interactions(tmp_path, monkeypatch, change):
    from linktools.ai.session import CreateSessionRequest
    from linktools.ai.persistence.runs import RuntimeAgentRunStore
    from linktools.ai.workspace import Workspace
    from linktools.commands.ai._common import _open_local_runtime
    from ._runtime_test_helpers import _Models, _UsageFunctionModel
    finalize = RuntimeAgentRunStore.finalize_execution_projection
    inspected = []
    async def finalize_after_takeover(self, flight, *args, **kwargs):
        if not kwargs.get('confirmed_interactions'):
            return await finalize(self, flight, *args, **kwargs)
        producer = self._execution_producers[flight.agent_run_id]
        self._execution_producers[flight.agent_run_id] = (
            (producer[0], producer[1] + 1) if change == 'generation' else ('replacement', producer[1])
        )
        try:
            result = await finalize(self, flight, *args, **kwargs)
            offset = self.trace_persistence.offsets[flight.agent_run_id]
            inspected.append((offset.confirmed_interactions, offset.confirmed_interaction_producer))
            return result
        finally:
            # Limit the injected race to FINALIZE; the real runtime keeps its lease.
            self._execution_producers[flight.agent_run_id] = producer
    monkeypatch.setattr(RuntimeAgentRunStore, 'finalize_execution_projection', finalize_after_takeover)
    def respond(_messages, _info):
        return ModelResponse(parts=[TextPart('complete')])
    async with _open_local_runtime(Workspace.initialize(tmp_path), model=_Models(_UsageFunctionModel(respond))) as local:
        runtime = local.runtime
        actor = runtime.default_principal
        await runtime.sessions.create('default', CreateSessionRequest(actor, 'session', 'create'))
        run = await runtime.sessions.start('session', 'hello', principal=actor, idempotency_key='turn')
        assert (await run.wait(timeout_seconds=20)).result.output == {'text': 'complete'}
    assert inspected and all(value == ((), None) for value in inspected)
