#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen checkpoint ownership and delta hashing without trusting mutable history."""

import hashlib
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
