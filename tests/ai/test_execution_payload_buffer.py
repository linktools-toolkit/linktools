#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Real execution bytes and exact-prefix cleanup never substitute source truth."""

import asyncio
from dataclasses import replace

import pytest

from linktools.ai.asset import AssetKey
from linktools.ai.core import ExecutionEventType, ExecutionStatus, PersistenceDomain
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.execution import RuntimePayloadRef, StoredUserInput
from linktools.ai.inputs import RepositoryInstructionPayload
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionEventAppend
from linktools.ai.runtime._execution_buffer import _ExecutionPayloadBuffer, _ExecutionPayloadCapture
from linktools.ai.spec import RepositoryInstructionDocument, RepositoryInstructions
from linktools.ai.storage import InMemoryCacheBackend, ObjectRef, StoredPayload

from .test_history_projection_conformance import _record
from .test_rule_context_storage import _archive, _assets, _rules


class _Buffer(InMemoryCacheBackend):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.admissions = []
        self.acks = []
        self.ack_started = asyncio.Event()
        self.ack_release = asyncio.Event()
        self.ack_release.set()
        self.fail_append = False
        self.fail_ack = False
        self.fail_read = False

    async def append(self, scope, id, key, content, *, wait):
        assert not wait
        result = await super().append(scope, id, key, content, wait=wait)
        self.admissions.append((scope, id, key, content))
        if self.fail_append:
            raise RuntimeError("append acknowledgment lost")
        return result

    async def read_batch(self, scope, *, limit):
        if self.fail_read:
            raise RuntimeError("buffer read unavailable")
        return await super().read_batch(scope, limit=limit)

    async def ack(self, scope, ids):
        self.acks.append((scope, tuple(ids)))
        self.ack_started.set()
        await self.ack_release.wait()
        if self.fail_ack:
            raise RuntimeError("ack transport unavailable")
        await super().ack(scope, ids)


def _capture(value="frozen body"):
    return _ExecutionPayloadCapture(replace(
        _record(ExecutionStatus.STARTED, 1),
        stored_user_input=StoredUserInput("text", StoredPayload.inline_text(value)),
    ))


@pytest.mark.asyncio
async def test_execution_buffer_reads_full_payload_and_confirms_only_captured_prefix() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    first = replace(_capture("body" * 20000), audit_events=(
        ExecutionEventAppend(ExecutionEventType.MODEL_REQUEST_STARTED, {"request": "frozen audit"}),
    ))
    await owner.append(first)
    observed, first_ids = await owner.read(first.execution.execution_id)
    assert observed == first
    assert b"bodybody" in backend.admissions[0][3]
    second = replace(first, execution=replace(first.execution, revision=2))
    await owner.append(second)
    owner.confirm(first.execution.execution_id, first_ids)
    await asyncio.wait_for(backend.ack_started.wait(), 1)
    latest, second_ids = await owner.read(first.execution.execution_id, seal=True)
    assert latest == second
    assert not set(first_ids).intersection(second_ids)
    with pytest.raises(AIError) as rejected:
        await owner.append(_capture("too late"))
    assert rejected.value.code is ErrorCode.STORAGE_CONFLICT
    owner.confirm(first.execution.execution_id, second_ids)
    await owner.close()
    assert {id for _, ids in backend.acks for id in ids} == set((*first_ids, *second_ids))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ("capacity", "append_unknown"))
async def test_failed_execution_buffer_admission_is_sticky_without_source_fallback(failure: str) -> None:
    backend = _Buffer(max_bytes=0 if failure == "capacity" else 1024 * 1024)
    backend.fail_append = failure == "append_unknown"
    backend.fail_read = failure == "append_unknown"
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    with pytest.raises((AIError, RuntimeError)):
        await owner.append(capture)
    assert owner.failed(capture.execution.execution_id)
    with pytest.raises((AIError, RuntimeError)):
        await owner.read(capture.execution.execution_id)
    assert len(backend.admissions) == 1
    backend.fail_read = False
    owner.discard(capture.execution.execution_id)
    await owner.close()
    scope, id, _, _ = backend.admissions[0]
    assert await backend.read_batch(scope, limit=2) == ()
    if failure == "append_unknown":
        assert (scope, (id,)) in backend.acks


@pytest.mark.asyncio
async def test_lost_execution_buffer_payload_cannot_be_recovered_from_a_ram_mirror() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    await owner.append(capture)
    scope, id, _, _ = backend.admissions[0]
    await backend.ack(scope, (id,))
    with pytest.raises(AIError) as lost:
        await owner.read(capture.execution.execution_id)
    assert lost.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert owner.failed(capture.execution.execution_id)
    owner.discard(capture.execution.execution_id)
    await owner.close()


@pytest.mark.asyncio
async def test_source_confirmation_does_not_wait_for_ack_or_lose_new_confirmed_tail() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    backend.ack_release.clear()
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    id = capture.execution.execution_id
    await owner.append(capture)
    _, first_ids = await owner.read(id)
    owner.confirm(id, first_ids)
    await asyncio.wait_for(backend.ack_started.wait(), 1)
    await owner.append(replace(capture, execution=replace(capture.execution, revision=2)))
    _, second_ids = await owner.read(id, seal=True)
    owner.confirm(id, second_ids)
    backend.ack_release.set()
    await owner.close()
    assert {entry for _, ids in backend.acks for entry in ids} == set((*first_ids, *second_ids))


@pytest.mark.asyncio
async def test_cleanup_failure_retries_only_ack_and_retains_canonical_confirmation() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    backend.fail_ack = True
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    await owner.append(capture)
    _, ids = await owner.read(capture.execution.execution_id, seal=True)
    owner.confirm(capture.execution.execution_id, ids)
    with pytest.raises(AIError) as cleanup:
        await owner.close()
    assert cleanup.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    backend.fail_ack = False
    owner.confirm(capture.execution.execution_id, ids)
    await owner.close()
    owner.confirm(capture.execution.execution_id, ids)
    assert len(backend.admissions) == 1


@pytest.mark.asyncio
async def test_unknown_append_is_confirmed_only_by_same_identity_and_content() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    backend.fail_append = True
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    await owner.append(capture)
    observed, ids = await owner.read(capture.execution.execution_id, seal=True)
    assert observed == capture
    assert ids == (backend.admissions[0][1],)
    owner.confirm(capture.execution.execution_id, ids)
    await owner.close()
    assert len(backend.admissions) == 1


@pytest.mark.asyncio
async def test_new_admission_transport_does_not_block_prior_source_confirmation() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Delayed(_Buffer):
        async def append(self, *args, **kwargs):
            if self.admissions:
                started.set()
                await release.wait()
            return await super().append(*args, **kwargs)

    backend = Delayed(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    await owner.append(capture)
    _, ids = await owner.read(capture.execution.execution_id)
    following = asyncio.create_task(owner.append(replace(capture, execution=replace(capture.execution, revision=2))))
    await asyncio.wait_for(started.wait(), 1)
    owner.confirm(capture.execution.execution_id, ids)
    assert not following.done()
    release.set()
    await following
    _, ids = await owner.read(capture.execution.execution_id, seal=True)
    owner.confirm(capture.execution.execution_id, ids)
    await owner.close()


@pytest.mark.asyncio
async def test_cancelled_caller_waits_for_owned_admission_outcome() -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    class Delayed(_Buffer):
        async def append(self, *args, **kwargs):
            result = await super().append(*args, **kwargs)
            started.set()
            await release.wait()
            return result

    backend = Delayed(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    pending = asyncio.create_task(owner.append(capture))
    await asyncio.wait_for(started.wait(), 1)
    pending.cancel()
    await asyncio.sleep(0)
    assert not pending.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await pending
    observed, ids = await owner.read(capture.execution.execution_id, seal=True)
    assert observed == capture
    owner.confirm(capture.execution.execution_id, ids)
    await owner.close()
    assert len(backend.admissions) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ("append", "read", "ack"))
async def test_buffer_transport_has_a_bounded_owner_deadline(monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
    from linktools.ai.runtime import _execution_buffer

    monkeypatch.setattr(_execution_buffer, "_BUFFER_TIMEOUT_SECONDS", 0.01)
    release = asyncio.Event()

    class Delayed(_Buffer):
        async def append(self, *args, **kwargs):
            if phase == "append":
                await release.wait()
            return await super().append(*args, **kwargs)

        async def read_batch(self, *args, **kwargs):
            if phase == "read":
                await release.wait()
            return await super().read_batch(*args, **kwargs)

        async def ack(self, *args, **kwargs):
            if phase == "ack":
                await release.wait()
            return await super().ack(*args, **kwargs)

    backend = Delayed(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    if phase == "append":
        with pytest.raises(AIError) as timed_out:
            await asyncio.wait_for(owner.append(capture), 1)
        assert timed_out.value.code is ErrorCode.STORAGE_UNAVAILABLE
    else:
        await owner.append(capture)
        if phase == "read":
            with pytest.raises(AIError) as timed_out:
                await asyncio.wait_for(owner.read(capture.execution.execution_id), 1)
            assert timed_out.value.code is ErrorCode.STORAGE_UNAVAILABLE
        else:
            _, ids = await owner.read(capture.execution.execution_id, seal=True)
            owner.confirm(capture.execution.execution_id, ids)
            with pytest.raises(AIError) as failure:
                await asyncio.wait_for(owner.close(), 1)
            assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    release.set()
    if phase != "ack":
        owner.discard(capture.execution.execution_id)
    await asyncio.wait_for(owner.close(), 1)


@pytest.mark.asyncio
async def test_execution_buffer_freezes_nested_events_before_owned_append_yields() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    payload = {"nested": {"value": "original"}}
    capture = replace(_capture(), audit_events=(ExecutionEventAppend(
        ExecutionEventType.MODEL_REQUEST_STARTED, payload,
    ),))
    task = asyncio.create_task(owner.append(capture))
    await asyncio.sleep(0)
    payload["nested"]["value"] = "mutated"
    await task
    frozen, ids = await owner.read(capture.execution.execution_id)
    assert frozen.audit_events[0].payload == {"nested": {"value": "original"}}
    owner.confirm(capture.execution.execution_id, ids)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ("input", "audit", "audit_truncation"))
async def test_execution_buffer_rejects_a_snapshot_that_rewrites_an_unconfirmed_prefix(change: str) -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    payload = {"nested": {"value": "original"}}
    first = replace(_capture(), audit_events=(ExecutionEventAppend(
        ExecutionEventType.MODEL_REQUEST_STARTED, payload,
    ),))
    await owner.append(first)
    if change == "input":
        second = replace(first, execution=_capture("different input").execution)
    elif change == "audit_truncation":
        second = replace(first, audit_events=())
    else:
        payload["nested"]["value"] = "rewritten"
        second = first
    await owner.append(second)
    with pytest.raises(AIError) as raised:
        await owner.read(first.execution.execution_id, seal=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    owner.discard(first.execution.execution_id)
    await owner.close()


async def _instruction_representations() -> tuple[RuntimePayloadRef, RuntimePayloadRef]:
    async with _assets() as assets, Persistence.open(
        "buffer-instructions", storage=PersistenceStorage.in_memory(),
    ) as state:
        await assets.put(AssetKey("rule", "first.md"), b"---\nscope: src\n---\noriginal Rule")
        await assets.put(AssetKey("rule", "second.md"), b"another Rule")
        accepted = RepositoryInstructions((*((await _rules(assets)).documents),
            RepositoryInstructionDocument("agents:AGENTS.md", ".", "Workspace instructions"),
            RepositoryInstructionDocument("rule:custom", ".", "custom instructions"),
        ))
        full = await _archive(state).prepare_repository_instruction_payload(
            accepted, source_domain=PersistenceDomain.EXECUTION,
        )
        archive = _archive(state, {"rules": assets})
        compact = await archive.prepare_repository_instruction_payload(
            accepted, source_domain=PersistenceDomain.EXECUTION,
        )
        assert full.payload.digest != compact.payload.digest
        assert await archive.load_repository_instruction_payload(full) == accepted
        assert await archive.load_repository_instruction_payload(compact) == accepted
        return full, compact


@pytest.mark.asyncio
@pytest.mark.parametrize("field, reverse", (
    ("repository_instructions", False), ("repository_instructions", True), ("input_context", False),
))
async def test_execution_buffer_preserves_payload_identity_across_storage_representations(
    field: str, reverse: bool,
) -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    if field == "repository_instructions":
        first_ref, second_ref = await _instruction_representations()
        if reverse:
            first_ref, second_ref = second_ref, first_ref
        assert RepositoryInstructionPayload.from_reference(first_ref).has_same_content(
            RepositoryInstructionPayload.from_reference(second_ref))
    else:
        payload = StoredPayload.inline_json({"body": "immutable context"})
        first_ref = RuntimePayloadRef(payload, None)
        reference = StoredPayload.object(ObjectRef("objects", "context", payload.digest, payload.size))
        second_ref = RuntimePayloadRef(reference, None)
    initial = _capture()
    first = replace(initial, execution=replace(initial.execution, **{field: first_ref}),
                    checkpoint_payload=b"opaque checkpoint bytes")
    await owner.append(first)
    second = replace(first, execution=replace(first.execution, **{field: second_ref}))
    await owner.append(second)
    captured, ids = await owner.read(first.execution.execution_id)
    assert captured == second
    assert captured.checkpoint_payload == b"opaque checkpoint bytes"
    owner.seal(first.execution.execution_id, ids)
    owner.confirm(first.execution.execution_id, ids)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("repository_instructions", "input_context"))
async def test_execution_buffer_rejects_rewritten_instruction_or_context_bytes(field: str) -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    if field == "repository_instructions":
        first_ref, _ = await _instruction_representations()
        changed = first_ref.payload.decode()
        changed["instructions"]["documents"][0]["content"] = "replacement"
        second_ref = replace(first_ref, payload=StoredPayload.inline_json(changed))
    else:
        first_ref = RuntimePayloadRef(StoredPayload.inline_json({"body": "original"}), None)
        second_ref = RuntimePayloadRef(StoredPayload.inline_json({"body": "replacement"}), None)
    initial = _capture()
    first = replace(initial, execution=replace(initial.execution, **{field: first_ref}))
    await owner.append(first)
    second = replace(first, execution=replace(first.execution, **{field: second_ref}))
    await owner.append(second)
    with pytest.raises(AIError) as changed:
        await owner.read(first.execution.execution_id)
    assert changed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    owner.discard(first.execution.execution_id)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("change", (
    "content", "content_digest", "asset", "asset_source_id", "source", "scope", "order",
    "metadata", "document_metadata", "semantic_digest", "workspace", "custom",
))
async def test_execution_buffer_rejects_changed_rule_projection_with_copied_semantic_digest(change: str) -> None:
    full, compact = await _instruction_representations()
    changed = (full if change == "content" else compact).payload.decode()
    documents = changed["instructions"]["documents"]
    rule = next(document for document in documents if document.get("asset") is not None)
    if change == "content":
        rule["content"] = "different Rule with copied semantic digest"
    elif change == "content_digest":
        rule["content_digest"] = "0" * 64
    elif change == "asset":
        rule["asset"]["revision"] += 1
    elif change in {"asset_source_id", "source", "scope"}:
        rule[change] = "different"
    elif change == "order":
        documents.reverse()
    elif change == "metadata":
        changed["instructions"]["extra"] = "different"
    elif change == "document_metadata":
        rule["extra"] = "different"
    elif change == "semantic_digest":
        changed["semantic_digest"] = "0" * 64
    else:
        source = "agents:AGENTS.md" if change == "workspace" else "rule:custom"
        next(document for document in documents if document["source"] == source)["content"] = "different"
    changed_ref = replace(compact, payload=StoredPayload.inline_json(changed))
    owner = _ExecutionPayloadBuffer(_Buffer(max_bytes=1024 * 1024))
    initial = _capture()
    first = replace(initial, execution=replace(initial.execution, repository_instructions=compact if change == "content" else full))
    await owner.append(first)
    await owner.append(replace(first, execution=replace(first.execution, repository_instructions=changed_ref)))
    with pytest.raises(AIError) as failure:
        await owner.read(first.execution.execution_id)
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    owner.discard(first.execution.execution_id)
    await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("metadata", "document_metadata", "asset_revision"))
@pytest.mark.parametrize("replacement", (True, 1.0), ids=("bool", "float"))
async def test_execution_buffer_rejects_rule_numeric_type_changes(
    field: str, replacement: bool | float,
) -> None:
    full, compact = await _instruction_representations()
    original, changed = full.payload.decode(), compact.payload.decode()
    for value, number in ((original, 1), (changed, replacement)):
        instructions = value["instructions"]
        rule = next(document for document in instructions["documents"] if document.get("asset") is not None)
        if field == "metadata":
            instructions["extra"] = number
        elif field == "document_metadata":
            rule["extra"] = number
        else:
            rule["asset"]["revision"] = number
    first_ref = replace(full, payload=StoredPayload.inline_json(original))
    second_ref = replace(compact, payload=StoredPayload.inline_json(changed))
    first_payload = RepositoryInstructionPayload.from_reference(first_ref)
    second_payload = RepositoryInstructionPayload.from_reference(second_ref)
    assert not first_payload.has_same_content(second_payload)
    assert not second_payload.has_same_content(first_payload)
    owner = _ExecutionPayloadBuffer(_Buffer(max_bytes=1024 * 1024))
    initial = _capture()
    first = replace(initial, execution=replace(initial.execution, repository_instructions=first_ref))
    await owner.append(first)
    await owner.append(replace(first, execution=replace(first.execution, repository_instructions=second_ref)))
    with pytest.raises(AIError) as failure:
        await owner.read(first.execution.execution_id)
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    owner.discard(first.execution.execution_id)
    await owner.close()


@pytest.mark.asyncio
async def test_execution_buffer_seals_only_the_current_admitted_prefix() -> None:
    owner = _ExecutionPayloadBuffer(_Buffer(max_bytes=1024 * 1024))
    captured = _capture()
    await owner.append(captured)
    _, first = await owner.read(captured.execution.execution_id)
    await owner.append(replace(captured, execution=replace(captured.execution, revision=2)))
    with pytest.raises(AIError):
        owner.seal(captured.execution.execution_id, first)
    _, current = await owner.read(captured.execution.execution_id)
    owner.seal(captured.execution.execution_id, current)
    with pytest.raises(AIError):
        owner.confirm(captured.execution.execution_id, (*current, "unknown-id"))
    owner.confirm(captured.execution.execution_id, current)
    await owner.close()


@pytest.mark.asyncio
async def test_dispatched_prefix_cannot_be_discarded_without_canonical_proof() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    await owner.append(capture)
    _, ids = await owner.read(capture.execution.execution_id)
    owner.seal(capture.execution.execution_id, ids)
    owner.discard(capture.execution.execution_id)
    with pytest.raises(AIError) as error:
        await owner.close()
    assert error.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert not backend.acks
    owner.confirm(capture.execution.execution_id, ids)
    await owner.close()


@pytest.mark.asyncio
async def test_execution_cleanup_keeps_unconfirmed_bytes_and_accepts_later_admission() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    await owner.append(capture)
    with pytest.raises(AIError) as retained:
        await owner.flush_cleanup(execution_id)
    assert retained.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    assert not backend.acks
    assert (await owner.read(execution_id))[0] == capture
    await owner.append(replace(capture, execution=replace(capture.execution, revision=2)))
    owner.discard(execution_id)
    await owner.flush_cleanup(execution_id)
    await owner.close()


@pytest.mark.asyncio
async def test_execution_cleanup_drains_retired_generation_without_waiting_for_peer_ack() -> None:
    peer_started, peer_release = asyncio.Event(), asyncio.Event()

    class SelectiveBuffer(_Buffer):
        blocked_scope = None

        async def ack(self, scope, ids):
            if scope == self.blocked_scope:
                peer_started.set()
                await peer_release.wait()
            await super().ack(scope, ids)

    backend = SelectiveBuffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    backend.fail_ack = True
    owner.bind_producer(execution_id, 0)
    await owner.append(capture)
    owner.retire(execution_id, 0, abandon=True)
    with pytest.raises(AIError):
        await owner.flush_cleanup(execution_id)
    owner.bind_producer(execution_id, 1)
    await owner.append(capture)
    owner.discard(execution_id)

    peer = replace(capture, execution=replace(capture.execution, execution_id="healthy-peer"))
    await owner.append(peer)
    backend.blocked_scope = backend.admissions[-1][0]
    owner.discard("healthy-peer")
    await asyncio.wait_for(peer_started.wait(), 1)
    backend.fail_ack = False
    try:
        await asyncio.wait_for(owner.flush_cleanup(execution_id), 1)
        assert not peer_release.is_set()
        assert not owner._retired_scopes
        for scope, _, _, _ in backend.admissions[:-1]:
            assert await backend.read_batch(scope, limit=10) == ()
        assert await backend.read_batch(backend.blocked_scope, limit=10)
        following = replace(capture, execution=replace(capture.execution, execution_id="next-turn"))
        await owner.append(following)
        owner.discard("next-turn")
    finally:
        peer_release.set()
        await owner.close()


@pytest.mark.asyncio
async def test_execution_cleanup_retry_preserves_caller_cancellation_and_owned_ack() -> None:
    ack_started, ack_release = asyncio.Event(), asyncio.Event()

    class RetryingBuffer(_Buffer):
        calls = 0
        cancelled = False

        async def ack(self, scope, ids):
            self.calls += 1
            if self.calls == 1:
                raise OSError("first ACK failed")
            ack_started.set()
            try:
                await ack_release.wait()
            except asyncio.CancelledError:
                self.cancelled = True
                raise
            await super().ack(scope, ids)

    backend = RetryingBuffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    await owner.append(capture)
    owner.discard(execution_id)
    first_cleanup = owner._scopes[execution_id].cleanup
    assert first_cleanup is not None
    await first_cleanup
    assert backend.calls == 1
    flushing = asyncio.create_task(owner.flush_cleanup(execution_id))
    try:
        await asyncio.wait_for(ack_started.wait(), 1)
        flushing.cancel("caller stopped waiting")
        with pytest.raises(asyncio.CancelledError) as cancelled:
            await flushing
        assert cancelled.value.args == ("caller stopped waiting",)
        assert not backend.cancelled
        assert await backend.read_batch(backend.admissions[0][0], limit=10)
        assert any(not task.done() for task in owner._tasks)
        retrying = asyncio.create_task(owner.flush_cleanup(execution_id))
        await asyncio.sleep(0)
        assert not retrying.done()
        ack_release.set()
        await retrying
        assert backend.calls == 2
        assert await backend.read_batch(backend.admissions[0][0], limit=10) == ()
        following = replace(capture, execution=replace(capture.execution, execution_id="next-turn"))
        await owner.append(following)
        owner.discard("next-turn")
    finally:
        ack_release.set()
        await asyncio.gather(flushing, return_exceptions=True)
        await owner.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("ack_outcome", ("blocked", "failed"))
async def test_finished_owner_retains_only_unacknowledged_transport(ack_outcome: str) -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    backend.ack_release.clear()
    backend.fail_ack = ack_outcome == "failed"
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    await owner.append(capture)
    _, ids = await owner.read(execution_id, seal=True)
    scope = owner._scopes[execution_id]
    owner.confirm(execution_id, ids)
    await asyncio.wait_for(backend.ack_started.wait(), 1)
    await owner.release(execution_id)
    owner.confirm(execution_id, ids)
    owner.release_owner(execution_id)
    assert not owner.has_scope(execution_id)
    assert scope.confirmed == set(ids)
    assert await backend.read_batch(scope.id, limit=10)
    if ack_outcome == "failed":
        backend.ack_release.set()
        with pytest.raises(AIError) as failure:
            await owner.flush_cleanup(execution_id)
        assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        assert scope.confirmed == set(ids)
        assert await backend.read_batch(scope.id, limit=10)
        backend.fail_ack = False
    backend.ack_release.set()
    await owner.flush_cleanup(execution_id)
    assert not scope.confirmed
    assert not owner._retired_scopes
    assert await backend.read_batch(scope.id, limit=10) == ()
    assert len(backend.admissions) == 1
    await owner.close()


@pytest.mark.asyncio
async def test_owner_release_preserves_unconfirmed_source_and_waiting_generation() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    owner.bind_producer(execution_id, 2)
    await owner.append(capture)
    _, ids = await owner.read(execution_id, seal=True)
    await owner.release(execution_id)
    owner.release_owner(execution_id)
    assert owner.has_scope(execution_id)
    assert not backend.acks
    assert await backend.read_batch(backend.admissions[0][0], limit=10)
    with pytest.raises(AIError) as unknown:
        await owner.flush_cleanup(execution_id)
    assert unknown.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
    owner.confirm(execution_id, ids)
    owner.retire(execution_id, 2)
    await owner.flush_cleanup(execution_id)
    owner.release_owner(execution_id)
    assert owner.has_scope(execution_id)
    assert owner.producer_generation(execution_id) == 2
    owner.confirm(execution_id, ids)
    with pytest.raises(AIError) as stale:
        owner.bind_producer(execution_id, 2)
    assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    owner.bind_producer(execution_id, 3)
    owner.require_producer(execution_id, 3)
    await owner.append(capture)
    owner.discard(execution_id)
    owner.release_owner(execution_id)
    await owner.flush_cleanup(execution_id)
    assert not owner._scopes
    assert not owner._retired_scopes
    await owner.close()


@pytest.mark.asyncio
async def test_old_ack_does_not_release_successor_buffer_owner() -> None:
    backend = _Buffer(max_bytes=1024 * 1024)
    backend.ack_release.clear()
    owner = _ExecutionPayloadBuffer(backend)
    capture = _capture()
    execution_id = capture.execution.execution_id
    owner.bind_producer(execution_id, 2)
    await owner.append(capture)
    _, old_ids = await owner.read(execution_id, seal=True)
    old_scope = owner._scopes[execution_id]
    owner.confirm(execution_id, old_ids)
    await asyncio.wait_for(backend.ack_started.wait(), 1)
    owner.retire(execution_id, 2)
    owner.bind_producer(execution_id, 3)
    await owner.append(capture)
    successor = owner._scopes[execution_id]
    backend.ack_release.set()
    assert old_scope.cleanup is not None
    await old_scope.cleanup
    assert not old_scope.confirmed
    assert owner._scopes[execution_id] is successor
    assert (await owner.read(execution_id))[0] == capture
    with pytest.raises(AIError) as wrong_prefix:
        owner.confirm(execution_id, old_ids)
    assert wrong_prefix.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    owner.discard(execution_id)
    owner.release_owner(execution_id)
    await owner.flush_cleanup(execution_id)
    assert not owner._scopes
    assert not owner._retired_scopes
    await owner.close()
