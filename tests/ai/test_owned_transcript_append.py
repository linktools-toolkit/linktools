#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned transcript capture appends bytes without rescanning SDK history."""

import hashlib
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart, UserPromptPart

from linktools.ai.persistence import MessageArchive, MessageSnapshot
from linktools.ai.persistence import _transcript_staging
from linktools.ai.persistence.runs import StagingAgentRunStore
from linktools.ai.runtime._agent_run_recorder import AgentRunRecorder


@pytest.mark.asyncio
async def test_owned_recorder_append_and_pending_boundaries_do_not_rescan_prefix(monkeypatch) -> None:
    hashed_bytes = 0
    encoded_bytes = 0
    encode = _transcript_staging.encode_model_messages

    class Digest:
        def __init__(self, data=b'', *, state=None):
            nonlocal hashed_bytes
            hashed_bytes += len(data)
            self.state = hashlib.sha256(data) if state is None else state

        def update(self, data):
            nonlocal hashed_bytes
            hashed_bytes += len(data)
            self.state.update(data)

        def copy(self):
            return Digest(state=self.state.copy())

        def hexdigest(self):
            return self.state.hexdigest()

    def encoded(messages):
        nonlocal encoded_bytes
        value = encode(messages)
        encoded_bytes += len(value)
        return value

    monkeypatch.setattr(_transcript_staging, 'hashlib', SimpleNamespace(sha256=Digest))
    monkeypatch.setattr(_transcript_staging, 'encode_model_messages', encoded)
    store = StagingAgentRunStore()
    await store.initialize()
    recorder = AgentRunRecorder(store, execution_id='execution', agent_run_id='run')
    source = ModelRequest(parts=[UserPromptPart('large-prefix-' * 4096)])
    recorder.append_transcript_message(source)
    baseline = recorder.transcript_messages()
    baseline_digest = baseline.encoded_digest()
    hashed_bytes = encoded_bytes = 0
    for index in range(8):
        recorder.append_transcript_message(ModelResponse(parts=[TextPart(str(index))]))
        recorder.stage_tool_result(ToolReturnPart('tool', str(index), f'call-{index}'))
        # Pending-only boundaries must retain the exact immutable message bytes.
        assert store.staged_transcript('run').messages == recorder.transcript_messages()
    assert encoded_bytes < 10_000
    assert hashed_bytes < 20_000
    assert baseline.encoded_digest() == baseline_digest
    assert len(baseline) == 1
    assert len(recorder.transcript_messages()) == 9


@pytest.mark.asyncio
async def test_owned_transcript_has_no_mutable_aliases_in_sources_or_readers() -> None:
    store = StagingAgentRunStore()
    await store.initialize()
    recorder = AgentRunRecorder(store, execution_id='execution', agent_run_id='run')
    source = ModelResponse(parts=[TextPart('original')])
    returned = recorder.append_transcript_message(source)
    source.parts[0].content = 'source mutation'
    returned.parts[0].content = 'return mutation'
    recorder.transcript_messages()[0].parts[0].content = 'reader mutation'
    store.staged_transcript('run').messages[0].parts[0].content = 'staging mutation'
    recorder.append_transcript_message(ModelResponse(parts=[TextPart('next')]))
    assert recorder.transcript_messages()[0].parts[0].content == 'original'
    assert store.staged_transcript('run').messages[0].parts[0].content == 'original'


def test_explicit_append_requires_frozen_prefix_and_freezes_mutable_delta() -> None:
    archive = MessageArchive()
    source = ModelRequest(parts=[UserPromptPart('before')])
    with pytest.raises(TypeError, match='immutable prefix'):
        archive.append([source], ())
    prefix = archive.snapshot([source])
    delta = ModelResponse(parts=[TextPart('delta')])
    appended = archive.append(prefix, [delta])
    delta.parts[0].content = 'changed'
    assert isinstance(appended, MessageSnapshot)
    assert appended[:len(prefix)].encoded_digest() == prefix.encoded_digest()
    assert appended[-1].parts[0].content == 'delta'


@pytest.mark.asyncio
async def test_owned_checkpoint_batch_reaches_native_publication_without_refreezing(tmp_path, monkeypatch) -> None:
    from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage, SqlStepStore
    from linktools.ai.persistence._archive import PreparedArchive
    from linktools.ai.persistence._step_archive import _ProjectionOffset
    from linktools.ai.persistence.runs import AgentRunCheckpoint, AgentRunRecord

    encoded_messages = 0
    encode = _transcript_staging.encode_model_messages

    def encoded(messages):
        nonlocal encoded_messages
        encoded_messages += len(messages)
        return encode(messages)

    monkeypatch.setattr(_transcript_staging, 'encode_model_messages', encoded)
    state = Persistence(PersistenceStorage.sqlite(tmp_path / 'owned.db'))
    await state.initialize(namespace='owned-append', tenant_id='tenant')
    try:
        native = SqlStepStore.for_state_store(state.execution.executions.state_store,
                                             namespace='owned-append', tenant_id='tenant')
        assert native is not None
        state.run_store.bind_step_store(native)
        owner = state.run_store.read_store(PersistenceDomain.TRACE)
        repository = owner.transcript_repository
        await repository.content_archive(create=True)
        staging = StagingAgentRunStore()
        await staging.initialize()
        recorder = AgentRunRecorder(staging, execution_id='execution', agent_run_id='run')
        await recorder.register_agent_run(AgentRunRecord('run'))
        baseline = None
        for index in range(4):
            recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart(str(index))]))
            before = encoded_messages
            await recorder.save_checkpoint(AgentRunCheckpoint(
                'run', index, recorder.transcript_messages(), transcript_message_count_before=index,
            ))
            batch = staging.capture_projection_local('run', _ProjectionOffset(checkpoints=index))
            candidate = batch.checkpoints[0]
            prepared = PreparedArchive()
            projection = await repository.prepare_model_projection(candidate.messages, prepared, baseline=baseline)

            async def publish(transaction):
                await repository.store_projection(transaction, 'run', projection)
                await repository.publish_native_archive(transaction, prepared, owner_written=True)

            await owner.state_store.mutate(publish)
            baseline = projection
            assert encoded_messages == before
        # The public store boundary still returns independent mutable SDK values.
        public = await staging.latest_checkpoint(agent_run_id='run')
        assert isinstance(public.messages, list)
        public.messages[0].parts[0].content = 'not durable'
        assert recorder.transcript_messages()[0].parts[0].content == '0'
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('use_callback', [False, True])
async def test_custom_checkpoint_store_and_callback_receive_sdk_lists(use_callback) -> None:
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from linktools.ai.persistence.runs import AgentRunCheckpoint

    received = []

    async def save(value, **kwargs):
        assert isinstance(value.messages, list)
        assert isinstance(value.context_messages, list)
        ModelMessagesTypeAdapter.dump_json(value.messages)
        ModelMessagesTypeAdapter.dump_json(value.context_messages)
        received.append(value)

    custom = SimpleNamespace(save_checkpoint=save, stage_transcript=lambda *args: None)
    recorder = AgentRunRecorder(custom, execution_id='execution', agent_run_id='run',
                                checkpoint_sink=save if use_callback else None)
    recorder.append_transcript_message(ModelRequest(parts=[UserPromptPart('raw')]))
    raw = recorder.transcript_messages()
    context = MessageArchive().snapshot([ModelRequest(parts=[UserPromptPart('context')])])
    await recorder.save_checkpoint(AgentRunCheckpoint('run', 1, raw, context_messages=context))
    assert len(received) == 1
    received[0].messages[0].parts[0].content = 'custom edit'
    received[0].context_messages[0].parts[0].content = 'custom context edit'
    assert raw[0].parts[0].content == 'raw'
    assert context[0].parts[0].content == 'context'


@pytest.mark.asyncio
async def test_custom_archive_batch_receives_materialized_checkpoint_lists() -> None:
    from pydantic_ai.messages import ModelMessagesTypeAdapter
    from linktools.ai.persistence._step_archive import _ProjectionOffset, _materialize_checkpoint, _sync_projection
    from linktools.ai.persistence.runs import AgentRunCheckpoint, AgentRunRecord

    received = []

    def accept(value):
        assert isinstance(value.messages, list)
        assert isinstance(value.context_messages, list)
        ModelMessagesTypeAdapter.dump_json(value.messages)
        ModelMessagesTypeAdapter.dump_json(value.context_messages)
        received.append(value)

    class CustomArchive:
        async def sync_projection(self, run, *, checkpoints, **kwargs):
            for checkpoint in checkpoints:
                accept(checkpoint)

        async def materialize_checkpoint(self, run, checkpoint, **kwargs):
            accept(checkpoint)

    staging = StagingAgentRunStore()
    await staging.initialize()
    run = AgentRunRecord('run')
    await staging.register_agent_run(run)
    frozen = MessageArchive().snapshot([ModelRequest(parts=[UserPromptPart('owned')])])
    await staging.save_checkpoint(AgentRunCheckpoint('run', 1, frozen, context_messages=frozen))
    batch = staging.capture_projection_local('run', _ProjectionOffset())
    assert isinstance(batch.checkpoints[0].messages, MessageSnapshot)
    custom = CustomArchive()
    await _sync_projection(custom, run, (), batch.checkpoints)
    await _materialize_checkpoint(custom, run, batch.checkpoints[0])
    assert len(received) == 2
    received[0].messages[0].parts[0].content = 'external edit'
    assert batch.checkpoints[0].messages[0].parts[0].content == 'owned'
