#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""batch_get and BatchStorageReader detection.

Validates that a batch read uses get_many when the backend supports it,
otherwise fans single get calls under bounded concurrency."""

import asyncio

import pytest

from linktools.ai.storage.multi import BatchStorageReader, batch_get


class BatchBackend:
    def __init__(self, values):
        self.values = dict(values)
        self.get_many_calls = 0

    async def get_many(self, keys):
        self.get_many_calls += 1
        return {key: self.values[key] for key in keys if key in self.values}


class SingleBackend:
    def __init__(self, values):
        self.values = dict(values)
        self.get_calls = 0

    async def get(self, key):
        self.get_calls += 1
        return self.values.get(key)


@pytest.mark.asyncio
async def test_batch_get_uses_get_many_when_supported():
    backend = BatchBackend({"a": 1, "b": 2})
    result = await batch_get(backend, ("a", "b", "missing"))
    assert result == {"a": 1, "b": 2}
    assert backend.get_many_calls == 1


@pytest.mark.asyncio
async def test_batch_get_falls_back_to_bounded_concurrency():
    backend = SingleBackend({"a": 1, "b": 2})
    result = await batch_get(backend, ("a", "b", "missing"), concurrency=2)
    assert result == {"a": 1, "b": 2}
    assert backend.get_calls == 3  # one per key incl. the miss


@pytest.mark.asyncio
async def test_batch_get_empty_returns_empty():
    assert await batch_get(SingleBackend({}), ()) == {}


@pytest.mark.asyncio
async def test_batch_get_rejects_non_positive_concurrency():
    with pytest.raises(ValueError):
        await batch_get(SingleBackend({}), ("a",), concurrency=0)


def test_batch_protocol_detects_get_many():
    assert isinstance(BatchBackend({}), BatchStorageReader)
    assert not isinstance(SingleBackend({}), BatchStorageReader)


@pytest.mark.asyncio
async def test_batch_get_timeout_raises_on_slow_backend():
    # A backend.get that hangs past the timeout must raise TimeoutError instead
    # of holding a semaphore permit forever (which would deadlock the batch).

    class SlowBackend:
        async def get(self, key):
            await asyncio.sleep(10)
            return "never"

    with pytest.raises(asyncio.TimeoutError):
        await batch_get(SlowBackend(), ("a",), concurrency=2, timeout=0.05)


@pytest.mark.asyncio
async def test_batch_get_timeout_get_many_branch():
    # The timeout also bounds the get_many branch (single backend call).

    class SlowBatch:
        async def get_many(self, keys):
            await asyncio.sleep(10)
            return {}

    with pytest.raises(asyncio.TimeoutError):
        await batch_get(SlowBatch(), ("a",), timeout=0.05)


@pytest.mark.asyncio
async def test_batch_get_empty_keys_returns_empty():
    # Empty keys short-circuits to {} without touching the backend.
    class Spy:
        def __init__(self):
            self.called = False

        async def get(self, key):
            self.called = True
            return key

    spy = Spy()
    assert await batch_get(spy, ()) == {}
    assert spy.called is False
