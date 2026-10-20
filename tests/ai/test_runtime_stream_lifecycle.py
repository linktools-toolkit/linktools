#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local preview independence and Runtime-owned remote observation lifetimes."""

import asyncio
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from pydantic_ai.models.function import FunctionModel

from linktools.ai.capability import ExecutionCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.storage import InMemoryStreamBackend, StreamBatch, StreamWriter
from ._runtime_test_helpers import _Models


class _Streams(InMemoryStreamBackend):
    def __init__(self, *, fail_open: bool = False) -> None:
        super().__init__(max_bytes=1024 * 1024)
        self.fail_open = fail_open
        self.reads = 0
        self.active_reads = 0
        self.entered = asyncio.Event()
        self.cleanup_entered = asyncio.Event()
        self.cleanup_release = asyncio.Event()
        self.cleanup_release.set()

    async def open_stream(self, key: bytes, producer: bytes) -> StreamWriter:
        if self.fail_open:
            raise OSError("preview transport unavailable")
        return await super().open_stream(key, producer)

    async def read_stream(self, key: bytes, after: int, *, limit: int, wait_seconds: float) -> StreamBatch:
        self.reads += 1
        self.active_reads += 1
        self.entered.set()
        try:
            return await super().read_stream(key, after, limit=limit, wait_seconds=wait_seconds)
        finally:
            self.cleanup_entered.set()
            await self.cleanup_release.wait()
            self.active_reads -= 1


@asynccontextmanager
async def _runtimes(tmp_path, cache, *, finish_provider: asyncio.Event | None = None):
    release = asyncio.Event()
    entered = asyncio.Event()

    async def model(messages, info):
        entered.set()
        yield "first "
        await release.wait()
        yield "last"
        if finish_provider is not None:
            await finish_provider.wait()

    group = RuntimeFeatures[None]("stream-lifecycle")
    group.agent("default", model="default", allow_tools=())
    group.capability(ExecutionCapability(coordination=cache), scope="runtime")
    models = _Models(FunctionModel(stream_function=model))
    path = tmp_path / "state.db"
    async with Runtime.open("stream-lifecycle", models=models, features=group,
                            storage=RuntimeStorage.sqlite(path)) as owner:
        storage = RuntimeStorage.sqlite(path)
        async with Runtime.open("stream-lifecycle", models=models, features=group, storage=storage) as reader:
            execution = await owner.agents.get().start("question")
            try:
                await asyncio.wait_for(entered.wait(), 5)
                yield owner, reader, storage, execution, release
            finally:
                cache.cleanup_release.set()
                release.set()
                if finish_provider is not None:
                    finish_provider.set()
                assert (await execution.wait(timeout_seconds=5)).result.output == {"text": "first last"}


async def _first_text(stream):
    async for item in stream:
        if item.event.event_type == "ASSISTANT_TEXT_DELTA":
            return item
    raise AssertionError("preview missing")


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ("paused", "pending", "cancel_close"))
async def test_runtime_close_drains_default_remote_watch(tmp_path: Path, mode: str) -> None:
    cache = _Streams()
    async with _runtimes(tmp_path, cache) as (_, reader, storage, execution, release):
        remote = await reader.executions.get(execution.execution_id)
        stream = remote.watch(include_content=True)
        pending = None
        try:
            assert (await asyncio.wait_for(_first_text(stream), 5)).event.payload["text"] == "first "
            if mode != "paused":
                cache.entered.clear()
                pending = asyncio.create_task(anext(stream))
                await asyncio.wait_for(cache.entered.wait(), 5)
                assert cache.active_reads == 1
            if mode == "cancel_close":
                cache.cleanup_release.clear()
                cache.cleanup_entered.clear()
                close = asyncio.create_task(reader.close())
                await asyncio.wait_for(cache.cleanup_entered.wait(), 5)
                assert reader._state.ready and not close.done()
                close.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await close
                close = asyncio.create_task(reader.close())
                await asyncio.sleep(0)
                assert reader._state.ready and not close.done()
                cache.cleanup_release.set()
                await asyncio.wait_for(close, 5)
            else:
                await asyncio.wait_for(reader.close(), 5)
            assert cache.active_reads == 0
            assert not reader._state.ready
            assert not reader._observation_sessions
            assert not release.is_set()
            if pending is not None:
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()
            await stream.aclose()
            writer = await cache.open_stream(b"host-owned", b"host")
            await writer.append(1, b"still available")
            assert (await cache.read_stream(b"host-owned", 0, limit=1, wait_seconds=0)).records == ((1, b"still available"),)
        finally:
            cache.cleanup_release.set()
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_open", (False, True))
async def test_local_watch_keeps_preview_without_shared_reads(tmp_path: Path, fail_open: bool) -> None:
    cache = _Streams(fail_open=fail_open)
    async with _runtimes(tmp_path, cache) as (_, _, _, execution, release):
        stream = execution.watch(include_content=True)
        try:
            first = await asyncio.wait_for(_first_text(stream), 5)
            assert first.event.payload["text"] == "first "
            assert cache.reads == 0 and not release.is_set()
            release.set()
            remaining = [item async for item in stream]
            assert any(item.event.event_type == "EXECUTION_SUCCEEDED" for item in remaining)
            assert cache.reads == 0
        finally:
            await stream.aclose()


@pytest.mark.asyncio
async def test_remote_preview_cursor_switches_to_local_with_explicit_reset(tmp_path: Path) -> None:
    cache = _Streams()
    async with _runtimes(tmp_path, cache) as (_, reader, _, execution, release):
        remote = await reader.executions.get(execution.execution_id)
        stream = remote.watch(include_content=True)
        try:
            first = await asyncio.wait_for(_first_text(stream), 5)
        finally:
            await stream.aclose()
        reads = cache.reads
        local = execution.watch(include_content=True, cursor=first.cursor)
        try:
            reset = await asyncio.wait_for(anext(local), 5)
            assert reset.event.event_type == "STREAM_RESET"
            assert reset.cursor != first.cursor
            preview = await asyncio.wait_for(_first_text(local), 5)
            assert preview.event.payload["text"] == "first "
            release.set()
            remaining = [item async for item in local]
            sequences = [item.event.durable_seq for item in remaining if item.event.durable_seq is not None]
            assert sequences == sorted(set(sequences))
            assert any(item.event.event_type == "EXECUTION_SUCCEEDED" for item in remaining)
            assert cache.reads == reads
        finally:
            await local.aclose()


@pytest.mark.asyncio
async def test_local_cursor_reconnects_remotely_through_authoritative_reset(tmp_path: Path) -> None:
    cache = _Streams()
    async with _runtimes(tmp_path, cache) as (_, reader, _, execution, release):
        local = execution.watch(include_content=True)
        try:
            first = await asyncio.wait_for(_first_text(local), 5)
        finally:
            await local.aclose()
        remote = await reader.executions.get(execution.execution_id)
        stream = remote.watch(include_content=True, cursor=first.cursor)
        try:
            reset = await asyncio.wait_for(anext(stream), 5)
            assert reset.event.event_type == "STREAM_RESET"
            release.set()
            remaining = [item async for item in stream]
            assert any(item.event.event_type == "EXECUTION_SUCCEEDED" for item in remaining)
        finally:
            await stream.aclose()


@pytest.mark.asyncio
async def test_cancelled_default_watch_read_releases_only_its_observer(tmp_path: Path) -> None:
    cache = _Streams()
    async with _runtimes(tmp_path, cache) as (_, reader, storage, execution, release):
        remote = await reader.executions.get(execution.execution_id)
        stream = remote.watch(include_content=True)
        pending = None
        try:
            await asyncio.wait_for(_first_text(stream), 5)
            cache.entered.clear()
            pending = asyncio.create_task(anext(stream))
            await asyncio.wait_for(cache.entered.wait(), 5)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
            assert cache.active_reads == 0 and reader._state.ready
            assert not reader._observation_sessions and not release.is_set()
        finally:
            if pending is not None:
                pending.cancel()
                await asyncio.gather(pending, return_exceptions=True)
            await stream.aclose()


@pytest.mark.asyncio
@pytest.mark.parametrize("close_early", (False, True))
async def test_remote_watch_before_first_admission_receives_later_preview(tmp_path: Path, monkeypatch, close_early: bool) -> None:
    from linktools.ai.runtime._local import LocalExecutionBackend

    admit = asyncio.Event()
    observed_unclaimed = asyncio.Event()
    release = asyncio.Event()
    run = LocalExecutionBackend._run

    async def delayed_run(self, *args, **kwargs):
        await admit.wait()
        return await run(self, *args, **kwargs)

    async def model(messages, info):
        yield "first "
        await release.wait()
        yield "last"

    monkeypatch.setattr(LocalExecutionBackend, "_run", delayed_run)
    cache = _Streams()
    group = RuntimeFeatures("early-preview")
    group.agent("default", model="default", allow_tools=())
    group.capability(ExecutionCapability(coordination=cache), scope="runtime")
    models = _Models(FunctionModel(stream_function=model))
    path = tmp_path / "early.db"
    async with Runtime.open("early-preview", models=models, features=group,
                            storage=RuntimeStorage.sqlite(path)) as owner:
        reader_storage = RuntimeStorage.sqlite(path)
        async with Runtime.open("early-preview", models=models, features=group,
                                storage=reader_storage) as reader:
            execution = await owner.agents.get().start("question")
            repository = reader._state.execution.executions
            read_head = repository.get_history_head

            async def observed_head(*args, **kwargs):
                head = await read_head(*args, **kwargs)
                if head is not None and head.producer_generation == 0 and head.producer_claim_id is None:
                    observed_unclaimed.set()
                return head

            monkeypatch.setattr(repository, "get_history_head", observed_head)
            remote = await reader.executions.get(execution.execution_id)
            stream = remote.watch(include_content=True)
            observed = []

            async def first_preview():
                async for item in stream:
                    observed.append(item.event)
                    if item.event.event_type == "ASSISTANT_TEXT_DELTA":
                        return item
                raise AssertionError("preview missing")

            first = asyncio.create_task(first_preview())
            try:
                await asyncio.wait_for(observed_unclaimed.wait(), 5)
                assert cache.reads == 0 and not first.done()
                if close_early:
                    await asyncio.wait_for(reader.close(), 5)
                    assert not reader._observation_sessions and not release.is_set()
                    assert not admit.is_set() and cache.active_reads == 0
                    return
                admit.set()
                item = await asyncio.wait_for(first, 5)
                assert item.event.payload["text"] == "first "
                assert any(event.event_type == "EXECUTION_STARTED" for event in observed)
                sequences = [event.durable_seq for event in observed if event.durable_seq is not None]
                assert sequences == sorted(set(sequences))
                current = await repository.get(execution.execution_id, tenant_id=reader.tenant_id)
                head = await read_head(execution.execution_id, tenant_id=reader.tenant_id)
                assert current.agent_run_seq == 1 and not current.durable_trace
                assert head.producer_generation > 1 and cache.reads > 0
                assert not release.is_set()
            finally:
                admit.set()
                release.set()
                first.cancel()
                await asyncio.gather(first, return_exceptions=True)
                await stream.aclose()
                assert (await execution.wait(timeout_seconds=5)).result.output == {"text": "first last"}


@pytest.mark.asyncio
@pytest.mark.parametrize("remote", (False, True))
async def test_preview_resume_keeps_received_prefix_until_terminal(tmp_path: Path, remote: bool) -> None:
    cache = _Streams()
    finish_provider = asyncio.Event()
    async with _runtimes(tmp_path, cache, finish_provider=finish_provider) as (_, reader, _, execution, release):
        observed = await reader.executions.get(execution.execution_id) if remote else execution
        first_watch = observed.watch(include_content=True)
        try:
            first = await asyncio.wait_for(_first_text(first_watch), 5)
            assert first.event.payload["text"] == "first "
        finally:
            await first_watch.aclose()
        resumed = observed.watch(include_content=True, cursor=first.cursor)
        received = asyncio.create_task(_first_text(resumed))
        try:
            await asyncio.sleep(0)
            assert not received.done()
            release.set()
            suffix = await asyncio.wait_for(received, 5)
            assert suffix.event.payload["text"] == "last"
            finish_provider.set()
            remaining = [item async for item in resumed]
            assert all(item.event.event_type != "STREAM_RESET" for item in remaining)
            assert any(item.event.event_type == "EXECUTION_SUCCEEDED" for item in remaining)
            assert first.event.payload["text"] + suffix.event.payload["text"] == "first last"
        finally:
            received.cancel()
            await asyncio.gather(received, return_exceptions=True)
            await resumed.aclose()
