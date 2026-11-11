#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Durable pending parts have incremental cost and immutable read cuts."""

from collections import Counter
from dataclasses import replace
from pathlib import Path
import random

import pytest
import pytest_asyncio
from pydantic_ai.messages import ModelRequest, ModelResponse, TextPart, ToolReturnPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence import _history
from linktools.ai.persistence._history import TranscriptRepository
from linktools.ai.persistence._memory import InMemoryStateStore
from linktools.ai.persistence._memory_transaction import _MemoryTransaction
from linktools.ai.persistence.state import StateStore
from linktools.ai.storage import InMemoryObjectStore


@pytest_asyncio.fixture(params=("memory", "sqlite"))
async def observation_store(request: pytest.FixtureRequest, tmp_path: Path):
    if request.param == "memory":
        store = InMemoryStateStore()
        try:
            yield store
        finally:
            await store.close()
    else:
        async with Persistence.open(
            "observation", tenant_id="tenant",
            storage=PersistenceStorage.sqlite(tmp_path / "transcript.db"),
        ) as persistence:
            yield persistence.execution.executions.state_store


def _repository(store: StateStore) -> TranscriptRepository:
    return TranscriptRepository(
        store, object_store=None, namespace="observation", tenant_id="tenant",
        runtime_domain=PersistenceDomain.TRACE,
    )


@pytest.mark.asyncio
async def test_pending_parts_preserve_captured_boundary_and_final_tool_order(observation_store) -> None:
    store = observation_store
    await store.initialize()
    repository = _repository(store)
    await repository.create_head("run")
    first = ToolReturnPart("tool", {"large": "first"}, tool_call_id="first")
    second = ToolReturnPart("tool", None, tool_call_id="second")
    one = await repository.prepare_observation(
        "run", (), first_message_index=0, pending=ModelRequest(parts=[first]),
        pending_keys=("tool_result:first",),
    )
    await store.mutate(lambda tx: repository.commit_observation(tx, one))
    frozen_head = await repository.get_head("run")
    assert frozen_head is not None and frozen_head.pending is not None
    captured_parts, captured_keys = await store.read(
        lambda tx: repository.capture_pending_in_transaction(tx, frozen_head)
    )
    two = await repository.prepare_observation(
        "run", (), first_message_index=0, pending=ModelRequest(parts=[first, second]),
        pending_keys=("tool_result:first", "tool_result:second"),
    )
    await store.mutate(lambda tx: repository.commit_observation(tx, two))
    final = ModelRequest(parts=[second, first])
    completed = await repository.prepare_observation(
        "run", (final,), first_message_index=0, pending=None, pending_keys=(),
    )
    await store.mutate(lambda tx: repository.commit_observation(tx, completed))
    assert (await repository.load_messages("run")) == (final,)
    frozen = await repository.read_pending_message(frozen_head.pending, captured_parts)
    assert captured_keys == ("tool_result:first",)
    assert frozen.parts == [first]
    assert await repository.verify_observation(one)
    assert await repository.verify_observation(two)
    assert await repository.verify_observation(completed)
    await store.close()


@pytest.mark.asyncio
async def test_pending_observation_rejects_drift_and_stale_commit(observation_store) -> None:
    store = observation_store
    await store.initialize()
    repository = _repository(store)
    await repository.create_head("run")
    initial = ModelResponse(parts=[TextPart("original")])
    prepared = await repository.prepare_observation(
        "run", (), first_message_index=0, pending=initial, pending_keys=("part:0",),
    )
    assert not await repository.verify_observation(prepared)
    await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
    with pytest.raises(AIError) as stale:
        await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
    assert stale.value.code is ErrorCode.STORAGE_CONFLICT
    with pytest.raises(AIError) as drift:
        await repository.prepare_observation(
            "run", (), first_message_index=0,
            pending=ModelResponse(parts=[TextPart("changed")]), pending_keys=("part:0",),
        )
    assert drift.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    with pytest.raises(AIError):
        await repository.prepare_observation(
            "run", (ModelResponse(parts=[TextPart("changed")]),), first_message_index=0,
            pending=None, pending_keys=(),
        )
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", (
    pytest.param((3,), id="contract"),
    pytest.param((16, 32, 64), marks=pytest.mark.manual, id="growth"),
))
async def test_pending_part_encoding_grows_with_added_bodies(
    monkeypatch: pytest.MonkeyPatch, counts: tuple[int, ...],
) -> None:
    original = _history.encode_model_messages
    encoded_parts = 0

    def measure(messages):
        nonlocal encoded_parts
        encoded_parts += sum(len(message.parts) for message in messages)
        return original(messages)

    monkeypatch.setattr(_history, "encode_model_messages", measure)
    totals = []
    for count in counts:
        store = InMemoryStateStore()
        await store.initialize()
        repository = _repository(store)
        await repository.create_head("run")
        before = encoded_parts
        parts = []
        keys = []
        for index in range(count):
            before_part = encoded_parts
            parts.append(ToolReturnPart("tool", "body" * 128, tool_call_id=str(index)))
            keys.append(f"tool_result:{index}")
            pending = ModelRequest(parts=list(parts))
            prepared = await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=tuple(keys),
            )
            assert len(prepared.new_pending_parts) == 1
            await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
            unchanged = await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=tuple(keys),
            )
            assert unchanged.new_pending_parts == ()
            await store.mutate(lambda tx: repository.commit_observation(tx, unchanged))
            assert encoded_parts - before_part <= 2, "adding one body must not encode the confirmed prefix"
            warmed = encoded_parts
            await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=tuple(keys),
            )
            assert encoded_parts == warmed, "an unchanged confirmed observation must not encode bodies"
        before_final = encoded_parts
        message = replace(pending, parts=list(reversed(parts)))
        completed = await repository.prepare_observation(
            "run", (message,), first_message_index=0,
            pending=None, pending_keys=(),
        )
        await store.mutate(lambda tx: repository.commit_observation(tx, completed))
        assert encoded_parts - before_final <= count * 2
        totals.append(encoded_parts - before)
        assert await repository.load_messages("run") == (message,)
        await repository.validate_integrity()
        await store.close()
    for smaller, larger in zip(totals, totals[1:]):
        assert larger <= smaller * 2 + 4


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", (
    pytest.param((3,), id="contract"),
    pytest.param((8, 16, 32), marks=pytest.mark.manual, id="growth"),
))
async def test_pending_object_bytes_scale_with_new_content(
    monkeypatch: pytest.MonkeyPatch, counts: tuple[int, ...],
) -> None:
    original = InMemoryObjectStore.put
    submitted_bytes = 0
    submitted_keys = Counter()

    async def measure(self, key, chunks, *, expected_size, expected_digest):
        nonlocal submitted_bytes
        submitted_bytes += expected_size
        submitted_keys[key] += 1
        return await original(self, key, chunks, expected_size=expected_size, expected_digest=expected_digest)

    monkeypatch.setattr(InMemoryObjectStore, "put", measure)
    totals = []
    retained = []
    for count in counts:
        store = InMemoryStateStore()
        await store.initialize()
        objects = InMemoryObjectStore("runtime")
        repository = TranscriptRepository(
            store, object_store=objects, namespace="observation", tenant_id="tenant",
            runtime_domain=PersistenceDomain.TRACE,
        )
        await repository.create_head("run")
        generator = random.Random(17)
        before = submitted_bytes
        submitted_keys.clear()
        parts = []
        keys = []
        for index in range(count):
            before_part = submitted_bytes
            parts.append(ToolReturnPart("tool", generator.randbytes(12_000).hex(), tool_call_id=str(index)))
            keys.append(f"tool_result:{index}")
            pending = ModelRequest(parts=list(parts))
            prepared = await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=tuple(keys),
            )
            assert len(prepared.new_pending_parts) == 1
            await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
            assert submitted_bytes > before_part, "each random body must use external object storage"
            warmed = submitted_bytes
            unchanged = await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=tuple(keys),
            )
            assert unchanged.new_pending_parts == ()
            await store.mutate(lambda tx: repository.commit_observation(tx, unchanged))
            await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=tuple(keys),
            )
            assert submitted_bytes == warmed, "confirmed bodies must not be resubmitted"
        message = ModelRequest(parts=list(reversed(parts)))
        final = await repository.prepare_observation(
            "run", (message,),
            first_message_index=0, pending=None, pending_keys=(),
        )
        await store.mutate(lambda tx: repository.commit_observation(tx, final))
        totals.append(submitted_bytes - before)
        retained.append(sum(len(raw) for raw in objects._objects.values()))
        assert submitted_keys and max(submitted_keys.values()) == 1
        assert totals[-1] == retained[-1], "object writes must contain only distinct retained content"
        assert await repository.load_messages("run") == (message,)
        await repository.validate_integrity()
        await store.close()
    for values in (totals, retained):
        for smaller, larger in zip(values, values[1:]):
            assert larger <= smaller * 2.2


@pytest.mark.asyncio
async def test_new_repository_resumes_pending_observation_without_replaying_parts(observation_store) -> None:
    store = observation_store
    await store.initialize()
    first = _repository(store)
    await first.create_head("run")
    part = TextPart("durable")
    prepared = await first.prepare_observation(
        "run", (), first_message_index=0, pending=ModelResponse(parts=[part]),
        pending_keys=("part:0",),
    )
    await store.mutate(lambda tx: first.commit_observation(tx, prepared))
    resumed = _repository(store)
    next_part = TextPart("next")
    next_boundary = await resumed.prepare_observation(
        "run", (), first_message_index=0, pending=ModelResponse(parts=[part, next_part]),
        pending_keys=("part:0", "part:1"),
    )
    await store.mutate(lambda tx: resumed.commit_observation(tx, next_boundary))
    head = await resumed.get_head("run")
    assert head is not None
    message, keys = await resumed.load_pending(head)
    assert message is not None and message.parts == [part, next_part]
    assert keys == ("part:0", "part:1")
    await store.close()


@pytest.mark.asyncio
async def test_rolled_back_pending_observation_can_be_retried(observation_store) -> None:
    store = observation_store
    await store.initialize()
    repository = _repository(store)
    await repository.create_head("run")
    message = ModelResponse(parts=[TextPart("durable after retry")])
    first = await repository.prepare_observation(
        "run", (), first_message_index=0, pending=message, pending_keys=("part:0",),
    )

    failure = RuntimeError("abort transaction")

    async def rollback(transaction) -> None:
        await repository.commit_observation(transaction, first)
        raise failure

    with pytest.raises((RuntimeError, AIError)) as raised:
        await store.mutate(rollback)
    if isinstance(raised.value, AIError):
        assert raised.value.code is ErrorCode.INTERNAL_ERROR
        assert raised.value.__cause__ is failure
    else:
        assert raised.value is failure
    assert not await repository.verify_observation(first)
    retry = await repository.prepare_observation(
        "run", (), first_message_index=0, pending=message, pending_keys=("part:0",),
    )
    await store.mutate(lambda tx: repository.commit_observation(tx, retry))
    assert await repository.verify_observation(retry)
    head = await repository.get_head("run")
    assert head is not None and head.pending_part_count == 1
    await repository.validate_integrity()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("nested", (False, True))
@pytest.mark.parametrize("complete", (False, True))
async def test_rolled_back_pending_cache_uses_other_writers_committed_parts(
    observation_store, nested: bool, complete: bool,
) -> None:
    store = observation_store
    await store.initialize()
    first, second = _repository(store), _repository(store)
    await first.create_head("run")
    base = TextPart("confirmed prefix")
    shared = TextPart("same tail after a different middle part")
    shell = ModelResponse(parts=[])

    async def prepare(repository, parts):
        return await repository.prepare_observation(
            "run", (), first_message_index=0, pending=replace(shell, parts=parts),
            pending_keys=tuple(f"part:{index}" for index in range(len(parts))),
        )

    initial = await prepare(first, [base])
    await store.mutate(lambda tx: first.commit_observation(tx, initial))
    await prepare(first, [base])
    abandoned = await prepare(first, [base, TextPart("abandoned middle"), shared])

    failure = RuntimeError("abort outer transaction")

    async def rollback(transaction) -> None:
        await first.commit_observation(transaction, abandoned)
        if nested:
            extra = await prepare(first, [base, TextPart("abandoned middle"), shared, TextPart("abandoned extra")])
            await first.commit_observation(transaction, extra)
        raise failure

    with pytest.raises((RuntimeError, AIError)) as raised:
        await store.mutate(rollback)
    if isinstance(raised.value, AIError):
        assert raised.value.code is ErrorCode.INTERNAL_ERROR
        assert raised.value.__cause__ is failure
    else:
        assert raised.value is failure
    winning = [base, TextPart("committed middle"), shared]
    committed = await prepare(second, winning)
    await store.mutate(lambda tx: second.commit_observation(tx, committed))
    if nested:
        winning.append(TextPart("committed extra"))
        committed = await prepare(second, winning)
        await store.mutate(lambda tx: second.commit_observation(tx, committed))

    if complete:
        message = replace(shell, parts=winning)
        resumed = await first.prepare_observation(
            "run", (message,), first_message_index=0, pending=None, pending_keys=(),
        )
        await store.mutate(lambda tx: first.commit_observation(tx, resumed))
        assert await first.load_messages("run") == (message,)
        later = replace(shell, parts=[TextPart("new pending message")])
        for pending in (later, replace(later, parts=[*later.parts, TextPart("later suffix")])):
            prepared = await first.prepare_observation(
                "run", (), first_message_index=1, pending=pending,
                pending_keys=tuple(f"part:{index}" for index in range(len(pending.parts))),
            )
            await store.mutate(lambda tx: first.commit_observation(tx, prepared))
        expected = pending.parts
    else:
        winning.append(TextPart("next part"))
        resumed = await prepare(first, winning)
        await store.mutate(lambda tx: first.commit_observation(tx, resumed))
        expected = winning
    head = await first.get_head("run")
    pending, _keys = await first.load_pending(head)
    assert pending is not None and pending.parts == expected
    await first.validate_integrity()
    await store.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("counts", (
    pytest.param((3,), id="contract"),
    pytest.param((16, 32, 64), marks=pytest.mark.manual, id="growth"),
))
async def test_confirmed_pending_prefix_reads_only_added_parts(
    monkeypatch: pytest.MonkeyPatch,
    counts: tuple[int, ...],
) -> None:
    rows = 0
    decoded = 0
    original_read = _MemoryTransaction.list_facts
    original_decode = TranscriptRepository._decode_chunk_messages

    async def read(self, query):
        nonlocal rows
        values = await original_read(self, query)
        rows += sum(value.kind == "transcript_pending_part" for value in values)
        return values

    async def decode(self, chunk):
        nonlocal decoded
        decoded += 1
        return await original_decode(self, chunk)

    monkeypatch.setattr(_MemoryTransaction, "list_facts", read)
    monkeypatch.setattr(TranscriptRepository, "_decode_chunk_messages", decode)
    for count in counts:
        store = InMemoryStateStore()
        await store.initialize()
        repository = _repository(store)
        await repository.create_head("run")
        before = rows, decoded
        parts = []
        for index in range(count):
            before_part = rows, decoded
            parts.append(ToolReturnPart("tool", "body" * 128, tool_call_id=str(index)))
            pending = ModelRequest(parts=list(parts))
            keys = tuple(f"tool_result:{position}" for position in range(len(parts)))
            for _ in range(2):
                prepared = await repository.prepare_observation(
                    "run", (), first_message_index=0, pending=pending, pending_keys=keys,
                )
                await store.mutate(lambda tx: repository.commit_observation(tx, prepared))
            assert rows - before_part[0] <= 1, "adding one body must not reread the confirmed prefix"
            assert decoded - before_part[1] <= 1
            warmed = rows, decoded
            await repository.prepare_observation(
                "run", (), first_message_index=0, pending=pending, pending_keys=keys,
            )
            assert (rows, decoded) == warmed, "unchanged confirmed parts must not be read again"
        assert rows - before[0] <= count
        assert decoded - before[1] <= count
        message = replace(pending, parts=list(reversed(parts)))
        completed = await repository.prepare_observation(
            "run", (message,), first_message_index=0, pending=None, pending_keys=(),
        )
        await store.mutate(lambda tx: repository.commit_observation(tx, completed))
        assert (rows, decoded) == warmed
        assert await repository.load_messages("run") == (message,)
        await repository.validate_integrity()
        await store.close()
