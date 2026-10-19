#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Composition-local automatic scheduling leaves owner settlement independent."""

import asyncio

import pytest

from linktools.ai.storage import BatchWriteScope, InMemoryCacheBackend, WriteBufferEntry


class _Sink:
    def __init__(self, *, fail: bool = False) -> None:
        self.batches: list[tuple[WriteBufferEntry, ...]] = []
        self.written = asyncio.Event()
        self.fail = fail

    async def write(self, batch: tuple[WriteBufferEntry, ...]) -> None:
        self.batches.append(batch)
        self.written.set()
        if self.fail:
            raise ValueError("source unavailable")


@pytest.mark.asyncio
async def test_aggregate_threshold_dispatches_each_owner_without_merging_payloads() -> None:
    cache = InMemoryCacheBackend(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    sinks = [_Sink(), _Sink()]
    scopes = [factory(cache, sink.write, batch_size=4, flush_interval=60) for sink in sinks]
    try:
        for scope in scopes:
            await scope.submit("a", b"a")
        assert not any(sink.batches for sink in sinks)
        await scopes[0].submit("b", b"b")
        await scopes[1].submit("c", b"c")
        await asyncio.wait_for(asyncio.gather(*(sink.written.wait() for sink in sinks)), 1)
        assert [len(sink.batches[0]) for sink in sinks] == [2, 2]
        assert set(entry[1] for entry in sinks[0].batches[0]) == {"a", "b"}
        assert set(entry[1] for entry in sinks[1].batches[0]) == {"a", "c"}
    finally:
        await asyncio.gather(*(scope.close() for scope in scopes))


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", ["buffer", "size", "interval", "composition"])
async def test_unrelated_groups_do_not_contribute_to_threshold(difference: str) -> None:
    cache = InMemoryCacheBackend(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    other_factory = BatchWriteScope.scoped_factory() if difference == "composition" else factory
    sinks = [_Sink(), _Sink()]
    a = factory(cache, sinks[0].write, batch_size=2, flush_interval=60)
    b = other_factory(
        InMemoryCacheBackend(max_bytes=4096) if difference == "buffer" else cache,
        sinks[1].write, batch_size=3 if difference == "size" else 2,
        flush_interval=61 if difference == "interval" else 60,
    )
    try:
        await a.submit("a", b"a")
        await b.submit("b", b"b")
        await asyncio.sleep(0.01)
        assert not any(sink.batches for sink in sinks)
        assert (await a.flush()).completed
        assert not sinks[1].batches
    finally:
        await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_earliest_pending_deadline_wakes_later_owner() -> None:
    cache = InMemoryCacheBackend(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    sinks = [_Sink(), _Sink()]
    a, b = [factory(cache, sink.write, batch_size=128, flush_interval=0.1) for sink in sinks]
    try:
        await a.submit("a", b"a")
        await asyncio.sleep(0.06)
        await b.submit("b", b"b")
        await asyncio.wait_for(asyncio.gather(*(sink.written.wait() for sink in sinks)), 0.08)
    finally:
        await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_owner_close_does_not_flush_or_stop_sibling() -> None:
    cache = InMemoryCacheBackend(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    sinks = [_Sink(), _Sink()]
    a, b = [factory(cache, sink.write, batch_size=4, flush_interval=60) for sink in sinks]
    try:
        await a.submit("a", b"a")
        await b.submit("b", b"b")
        await a.close()
        assert sinks[0].batches and not sinks[1].batches
        for key in ("c", "d", "e"):
            await b.submit(key, key.encode())
        await asyncio.wait_for(sinks[1].written.wait(), 1)
        assert b.status().accepting
    finally:
        await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_failed_source_keeps_existing_owner_local_weak_settlement() -> None:
    cache = InMemoryCacheBackend(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    good, bad = _Sink(), _Sink(fail=True)
    a = factory(cache, good.write, batch_size=2, flush_interval=60)
    b = factory(cache, bad.write, batch_size=2, flush_interval=60)
    try:
        await a.submit("a", b"a")
        await b.submit("b", b"b")
        results = await asyncio.gather(a.flush(), b.flush())
        assert all(result.completed for result in results)
        assert a.status().confirmed == 1
        assert a.status().write_failures == 0
        assert b.status().unconfirmed == 1
        assert b.status().write_failures == 1
    finally:
        await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_inflight_prefix_does_not_retrigger_shared_threshold() -> None:
    cache = InMemoryCacheBackend(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow(batch: tuple[WriteBufferEntry, ...]) -> None:
        entered.set()
        await release.wait()

    sink = _Sink()
    a = factory(cache, slow, batch_size=2, flush_interval=60)
    b = factory(cache, sink.write, batch_size=2, flush_interval=60)
    try:
        await a.submit("a", b"a")
        await a.submit("b", b"b")
        await asyncio.wait_for(entered.wait(), 1)
        await b.submit("c", b"c")
        await asyncio.sleep(0.01)
        assert not sink.batches
        release.set()
        await a.flush()
        assert not sink.batches
    finally:
        release.set()
        await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_unconfirmed_append_does_not_contribute_to_shared_threshold() -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class DelayedReply(InMemoryCacheBackend):
        async def append(self, scope, id, key, content, *, wait):
            admitted = await super().append(scope, id, key, content, wait=wait)
            if key == "delayed":
                entered.set()
                await release.wait()
            return admitted

    cache = DelayedReply(max_bytes=4096)
    factory = BatchWriteScope.scoped_factory()
    sinks = [_Sink(), _Sink()]
    a, b = [factory(cache, sink.write, batch_size=2, flush_interval=60) for sink in sinks]
    pending = asyncio.create_task(a.submit("delayed", b"a"))
    try:
        await entered.wait()
        await b.submit("b", b"b")
        await asyncio.sleep(0.01)
        assert a.status().accepted == 0
        assert not any(sink.batches for sink in sinks)
        release.set()
        await pending
        await asyncio.wait_for(asyncio.gather(*(sink.written.wait() for sink in sinks)), 1)
    finally:
        release.set()
        await asyncio.gather(pending, return_exceptions=True)
        await asyncio.gather(a.close(), b.close())


@pytest.mark.asyncio
async def test_last_owner_close_releases_idle_schedule_and_its_timer() -> None:
    import gc
    import weakref

    factory = BatchWriteScope.scoped_factory()
    sink = _Sink()
    scope = factory(InMemoryCacheBackend(max_bytes=4096), sink.write, flush_interval=3600)
    await scope.submit("a", b"a")
    schedule = weakref.ref(scope._schedule)
    assert schedule() is not None
    await scope.close()
    gc.collect()
    assert schedule() is None
