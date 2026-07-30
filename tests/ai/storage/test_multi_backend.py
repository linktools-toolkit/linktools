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
