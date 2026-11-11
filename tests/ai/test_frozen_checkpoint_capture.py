#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen checkpoint ownership and delta hashing without trusting mutable history."""

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot

from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import MessageArchive, MessageSnapshot, NativeCheckpoint
from linktools.ai.runtime._step_store import RuntimeStepStore


def test_frozen_append_hashes_only_delta_and_preserves_canonical_digest(monkeypatch) -> None:
    archive = MessageArchive()
    prefix = archive.snapshot([ModelRequest(parts=[UserPromptPart('large prefix' * 1024)])])
    prefix.encoded_digest()
    consumed = 0
    original = MessageSnapshot._payloads

    def observed(snapshot):
        nonlocal consumed
        for payload in original(snapshot):
            consumed += len(payload)
            yield payload

    # Observe actual bytes traversed, rather than charging cached digest results.
    monkeypatch.setattr(MessageSnapshot, '_payloads', observed)
    for index in range(8):
        delta = archive.snapshot([ModelRequest(parts=[UserPromptPart(str(index))])])
        previous = prefix.encoded_digest()
        appended = archive.concatenate(prefix, delta)
        assert appended[:len(prefix)].encoded_digest() == previous
        appended.encoded_digest()
        prefix = appended
    assert consumed < 10_000
    monkeypatch.setattr(MessageSnapshot, '_payloads', original)
    encoded = encode_model_messages(list(prefix))
    assert prefix.encoded_digest() == (hashlib.sha256(encoded).hexdigest(), len(encoded))
    assert len(prefix._roots) <= len(prefix).bit_length()


def test_checkpoint_capture_detaches_mutation_and_keeps_retry_state() -> None:
    store = RuntimeStepStore(SimpleNamespace(), SimpleNamespace(), output_retries_used=lambda: 3)
    message = ModelRequest(parts=[UserPromptPart('before')])
    source = ContinuableSnapshot(run_id='run', step_index=1, messages=[message])
    captured = store.capture_snapshot(source)
    message.parts[0].content = 'after'
    decoded = NativeCheckpoint.decode(store._encode_snapshot(captured))
    assert isinstance(captured.messages, MessageSnapshot)
    assert decoded.output_retries_used == 3
    assert decoded.messages[0].parts[0].content == 'before'
    assert store.capture_snapshot(captured).messages is captured.messages
    captured.messages[0].parts[0].content = 'unowned edit'
    assert captured.messages[0].parts[0].content == 'before'
    fresh = NativeCheckpoint.decode(store._encode_snapshot(store.capture_snapshot(source)))
    assert fresh.messages[0].parts[0].content == 'after'


@pytest.mark.parametrize('messages', [[], [ModelRequest(parts=[UserPromptPart('input')])]])
def test_frozen_retry_envelope_round_trips(messages) -> None:
    source = ContinuableSnapshot(run_id='run', step_index=1, messages=MessageArchive().snapshot(messages))
    encoded = NativeCheckpoint.encode(source, output_retries_used=2)
    restored = NativeCheckpoint.decode(encoded)
    assert isinstance(restored.messages, MessageSnapshot)
    assert restored.messages == messages
    assert restored.output_retries_used == 2


@pytest.mark.asyncio
async def test_terminal_confirmation_compares_logical_reference_messages() -> None:
    from linktools.ai.capability.steps import AtomicStepPersistence
    from linktools.ai.model.messages import native_checkpoint_messages
    from .test_native_referenced_model import _messages

    commits = []

    async def commit(**values):
        commits.append(values)

    async def emit(event):
        pass

    runtime = RuntimeStepStore(SimpleNamespace(), SimpleNamespace(),
                               commit_boundary=commit, output_retries_used=lambda: 2)
    capability = AtomicStepPersistence(store=runtime, run_id='run')
    messages = _messages()
    ctx = SimpleNamespace(run_step=1, conversation_id=None, emit=emit)
    snapshot = capability._atomic_prepare_snapshot(ctx, messages, 1)
    await capability._atomic_commit(snapshot=snapshot)
    await capability._atomic_confirm_snapshot(ctx, snapshot)
    await capability.after_run(ctx, result=SimpleNamespace(all_messages=lambda: messages, output='done'))
    assert len(commits) == 2
    assert commits[1]['snapshot'] is None
    persisted = NativeCheckpoint.decode(commits[0]['snapshot'])
    assert persisted.output_retries_used == 2
    assert native_checkpoint_messages(persisted.messages, restore=True) == messages


@pytest.mark.asyncio
@pytest.mark.parametrize('runtime', (False, True))
async def test_frozen_snapshot_batches_do_not_revisit_committed_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, runtime: bool,
) -> None:
    from sqlalchemy.ext.asyncio import create_async_engine
    from linktools.ai.persistence import SqlStepStore

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'batches.db'}")
    native = SqlStepStore(engine, namespace='batches', tenant_id='tenant', max_cached_messages=2)
    await native.initialize(local=True)
    writer = (RuntimeStepStore(native, SimpleNamespace(), output_retries_used=lambda: 3)
              if runtime else native)
    archive = MessageArchive()
    messages = archive.snapshot([
        ModelRequest(parts=[UserPromptPart(f'history-{index}' * 1024)]) for index in range(7)
    ])
    visited = []
    original = MessageSnapshot._payloads

    def observed(snapshot):
        for payload in original(snapshot):
            visited.append(payload)
            yield payload

    try:
        await writer.save_snapshot(ContinuableSnapshot(run_id='run', step_index=0, messages=messages))
        monkeypatch.setattr(MessageSnapshot, '_payloads', observed)
        for step in range(1, 3):
            committed = set(original(messages))
            visited.clear()
            for index in range(2):
                messages = archive.append(messages, [ModelRequest(parts=[UserPromptPart(f'{step}-{index}')])])
            await writer.save_snapshot(ContinuableSnapshot(run_id='run', step_index=step, messages=messages))
            # The body budget observes actual traversal, including digest input,
            # rather than counting calls that might return cached digest values.
            assert not committed.intersection(visited)
        monkeypatch.setattr(MessageSnapshot, '_payloads', original)
        restored = await writer.latest_snapshot(run_id='run')
        assert restored.messages == list(messages)
        if runtime:
            assert restored.output_retries_used == 3
        changed = list(messages)
        changed[0].parts[0].content = 'changed historical content'
        await writer.save_snapshot(ContinuableSnapshot(run_id='run', step_index=3, messages=changed))
        assert (await writer.latest_snapshot(run_id='run')).messages == changed
        native.invalidate_message_content()
        assert not native._message_snapshots
    finally:
        await native.close()
        await engine.dispose()


def test_shared_prefix_proof_stops_at_changed_frozen_content() -> None:
    archive = MessageArchive()
    source = [ModelRequest(parts=[UserPromptPart(str(index))]) for index in range(7)]
    original = archive.snapshot(source)
    extended = archive.append(archive.append(original, source[:1]), source[1:2])
    assert extended.shared_prefix_length(original) == len(original)
    source[3].parts[0].content = 'changed'
    changed = archive.snapshot(source)
    assert changed.shared_prefix_length(original) == 3
    assert changed[:3].shared_prefix_length(original) == 3
    assert original.shared_prefix_length(archive.snapshot([])) == 0
