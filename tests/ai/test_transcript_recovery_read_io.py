#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Full transcript recovery batches immutable reads without changing streaming."""

from pathlib import Path
from time import perf_counter

import pytest
from pydantic_ai.messages import ModelResponse, TextPart

from linktools.ai.persistence._transcript_staging import StagedTranscript

from .test_step_archive_read_boundaries import _archive


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
@pytest.mark.parametrize("count", (9, 65))
async def test_full_transcript_recovery_bounds_archive_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str, count: int,
) -> None:
    async with _archive(tmp_path, backend) as (state, archive, run):
        messages = tuple(ModelResponse(parts=[TextPart(f"message-{index}")]) for index in range(count))
        for index, message in enumerate(messages):
            observation = await archive.transcript_repository.prepare_observation(
                "run", (message,), first_message_index=index, pending=None, pending_keys=(),
            )
            await archive.sync_prepared_projection(
                run, events=(), checkpoints=(), observation=observation,
                execution_id="execution", producer_generation=1,
            )
        content = await archive.transcript_repository.content_archive()
        reads = 0
        original = content.store.read

        async def read(operation):
            nonlocal reads
            reads += 1
            return await original(operation)

        monkeypatch.setattr(content.store, "read", read)
        started = perf_counter()
        await state.run_store.register_agent_run(run, execution_id="execution")
        elapsed = perf_counter() - started
        assert state.run_store.staged_transcript("run") == StagedTranscript(messages)
        print(f"TRANSCRIPT_RECOVERY backend={backend} count={count} archive_reads={reads} seconds={elapsed:.6f}")
        assert reads <= (count + 63) // 64 + 1, "complete recovery must batch immutable archive reads"


async def _seed_messages(archive, run, messages) -> None:
    for index, message in enumerate(messages):
        observation = await archive.transcript_repository.prepare_observation(
            "run", (message,), first_message_index=index, pending=None, pending_keys=(),
        )
        await archive.sync_prepared_projection(
            run, events=(), checkpoints=(), observation=observation,
            execution_id="execution", producer_generation=1,
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_transcript_recovery_keeps_payloads_and_message_ranges(tmp_path: Path, backend: str) -> None:
    from linktools.ai.persistence.state import RecordQuery
    from linktools.ai.persistence._archive import ContentRecord

    async with _archive(tmp_path, backend) as (state, archive, run):
        messages = tuple(ModelResponse(parts=[TextPart(value)]) for value in ("first", "large" * 20_000, "last"))
        await _seed_messages(archive, run, messages)
        content = await archive.transcript_repository.content_archive()
        records = await content.store.read(lambda transaction: transaction.list_records(RecordQuery(kind="archive_content")))
        assert {ContentRecord.from_json(record.data).payload.kind for record in records} == {"inline", "object"}
        await state.run_store.register_agent_run(run, execution_id="execution")
        assert state.run_store.staged_transcript("run") == StagedTranscript(messages)
        assert tuple([value async for value in archive.iter_message_range(agent_run_id="run", start=1, end=2)]) == messages[1:2]
        assert tuple([value async for value in archive.iter_messages(agent_run_id="run")]) == messages


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_transcript_recovery_does_not_publish_incomplete_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool,
) -> None:
    import asyncio
    from linktools.ai.errors import AIError, ErrorCode

    async with _archive(tmp_path) as (state, archive, run):
        messages = tuple(ModelResponse(parts=[TextPart(str(index))]) for index in range(3))
        await _seed_messages(archive, run, messages)
        content = await archive.transcript_repository.content_archive()
        error = asyncio.CancelledError() if cancelled else AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        async def fail(operation):
            raise error

        with monkeypatch.context() as patch:
            patch.setattr(content.store, "read", fail)
            with pytest.raises(type(error)):
                await state.run_store.register_agent_run(run, execution_id="execution")
        assert state.run_store.staged_transcript("run") is None
        assert await state.run_store.get_agent_run(agent_run_id="run") is None
        await state.run_store.register_agent_run(run, execution_id="execution")
        assert state.run_store.staged_transcript("run") == StagedTranscript(messages)


@pytest.mark.asyncio
async def test_transcript_stream_does_not_read_unconsumed_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.errors import AIError, ErrorCode

    async with _archive(tmp_path) as (_state, archive, run):
        messages = tuple(ModelResponse(parts=[TextPart(str(index))]) for index in range(2))
        await _seed_messages(archive, run, messages)
        content = await archive.transcript_repository.content_archive()
        read = content.store.read
        reads = 0

        async def fail_suffix(operation):
            nonlocal reads
            reads += 1
            if reads > 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return await read(operation)

        monkeypatch.setattr(content.store, "read", fail_suffix)
        stream = archive.iter_messages(agent_run_id="run")
        assert await anext(stream) == messages[0]
        with pytest.raises(AIError) as raised:
            await anext(stream)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        await stream.aclose()


@pytest.mark.asyncio
async def test_transcript_recovery_keeps_custom_archive_stream(
    tmp_path: Path,
) -> None:
    from linktools.ai.persistence import PersistenceDomain
    from linktools.ai.persistence.runs import ModelInteractionStateStepArchive

    messages = (ModelResponse(parts=[TextPart("custom projection")]),)

    class CustomArchive(ModelInteractionStateStepArchive):
        async def iter_messages(self, *, agent_run_id: str):
            for message in messages:
                yield message

    async with _archive(tmp_path) as (state, archive, run):
        await _seed_messages(archive, run, (ModelResponse(parts=[TextPart("stored projection")]),))
        custom = CustomArchive(
            archive.state_store, object_store=None, namespace="archive-boundaries",
            tenant_id="tenant", runtime_domain=PersistenceDomain.TRACE,
        )
        state.run_store._archives[PersistenceDomain.TRACE] = custom
        await state.run_store.register_agent_run(run, execution_id="execution")
        assert state.run_store.staged_transcript("run") == StagedTranscript(messages)
