#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Commit preferences are immutable and restricted by each persistence owner."""

from dataclasses import FrozenInstanceError

import pytest

from linktools.ai.capability import ConversationCapability, ExecutionCapability, TraceCapability
from linktools.ai.core import DEFAULT_TRACE_BATCH_SIZE, DEFAULT_TRACE_FLUSH_INTERVAL
from linktools.ai.core import CommitPolicy


def test_commit_policy_defaults_and_immutable_shared_declarations() -> None:
    assert ExecutionCapability().commit == CommitPolicy.sync()
    assert TraceCapability().commit == CommitPolicy.batch(
        batch_size=DEFAULT_TRACE_BATCH_SIZE,
        flush_interval=DEFAULT_TRACE_FLUSH_INTERVAL,
    )
    policy = CommitPolicy.terminal()
    first = ExecutionCapability(commit=policy)
    second = ExecutionCapability(commit=policy)
    assert first.commit is second.commit is policy
    with pytest.raises(FrozenInstanceError):
        policy.mode = "sync"


@pytest.mark.parametrize("settings", [
    {"mode": "unknown"},
    {"mode": "sync", "batch_size": 1},
    {"mode": "terminal", "flush_interval": 1},
    {"mode": "batch"},
    {"mode": "batch", "batch_size": True, "flush_interval": 1},
    {"mode": "batch", "batch_size": 1, "flush_interval": float("nan")},
    {"mode": "batch", "batch_size": 1, "flush_interval": float("inf")},
])
def test_commit_policy_rejects_invalid_declarations(settings: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CommitPolicy(**settings)


def test_persistence_owners_reject_unsupported_policy_modes_and_removed_arguments() -> None:
    with pytest.raises(ValueError):
        ExecutionCapability(commit=CommitPolicy.batch(batch_size=2, flush_interval=1))
    with pytest.raises(ValueError):
        TraceCapability(commit=CommitPolicy.terminal())
    with pytest.raises(TypeError):
        ConversationCapability(commit=CommitPolicy.sync())
    with pytest.raises(TypeError):
        ConversationCapability(cache=None)
    with pytest.raises(TypeError):
        ConversationCapability(storage=None)
    with pytest.raises(TypeError):
        TraceCapability(batch_size=2)
    with pytest.raises(TypeError):
        TraceCapability(flush_interval=1)


def test_execution_commit_has_one_public_configuration_owner() -> None:
    import inspect
    from linktools.ai.runtime import Runtime
    assert "commit" not in inspect.signature(Runtime.open).parameters
    assert "commit" in inspect.signature(ExecutionCapability).parameters


def test_write_buffer_is_borrowed_and_separate_from_policy_value() -> None:
    from linktools.ai.storage import InMemoryCacheBackend

    first = InMemoryCacheBackend(max_bytes=1024)
    second = InMemoryCacheBackend(max_bytes=1024)
    policy = CommitPolicy.batch(batch_size=128, flush_interval=3, buffer=first)
    other = CommitPolicy.batch(batch_size=128, flush_interval=3, buffer=second)
    assert policy.buffer is first
    assert other.buffer is second
    assert policy == other
    assert hash(policy) == hash(other)
    assert "buffer" not in repr(policy)
    with pytest.raises(FrozenInstanceError):
        policy.buffer = second
    with pytest.raises(ValueError):
        CommitPolicy("sync", buffer=first)
    with pytest.raises(ValueError):
        CommitPolicy("terminal", buffer=first)


def test_write_policy_requires_a_complete_write_buffer() -> None:
    class ReadOnlyCache:
        async def get(self, key: str) -> bytes | None:
            return None

        async def put(self, key: str, content: bytes) -> None:
            pass

        async def delete(self, key: str) -> None:
            pass

    with pytest.raises(TypeError, match="append, read_batch and ack"):
        CommitPolicy.batch(batch_size=1, flush_interval=1, buffer=ReadOnlyCache())


def test_storage_cache_requires_hot_read_capability() -> None:
    from linktools.ai.storage import StorageRoute

    class WriteOnlyBuffer:
        async def append(self, *args: object, **kwargs: object) -> bool:
            return True

        async def read_batch(self, *args: object, **kwargs: object) -> tuple:
            return ()

        async def ack(self, *args: object, **kwargs: object) -> None:
            pass

    buffer = WriteOnlyBuffer()
    assert CommitPolicy.batch(batch_size=1, flush_interval=1, buffer=buffer).buffer is buffer
    with pytest.raises(TypeError, match="get, put and delete"):
        StorageRoute.memory(cache=buffer)


@pytest.mark.parametrize("first", ["core", "storage", "capability"])
def test_public_write_buffer_annotations_resolve_in_any_import_order(first: str) -> None:
    import subprocess
    import sys

    code = f"""
import importlib
from typing import get_type_hints
importlib.import_module('linktools.ai.{first}')
from linktools.ai.core import CommitPolicy, WriteBufferBackend, WriteBufferEntry
from linktools.ai.storage import WriteBufferBackend as StorageBuffer, WriteBufferEntry as StorageEntry
assert StorageBuffer is WriteBufferBackend
assert StorageEntry is WriteBufferEntry
assert get_type_hints(CommitPolicy)['buffer'] == WriteBufferBackend | None
assert get_type_hints(CommitPolicy.batch)['buffer'] == WriteBufferBackend | None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)


@pytest.mark.parametrize("member", ["append", "ack"])
def test_write_policy_and_scope_reject_noncallable_buffer_methods(member: str) -> None:
    from linktools.ai.storage import BatchWriteScope, InMemoryCacheBackend, WriteBufferEntry

    buffer = InMemoryCacheBackend(max_bytes=1024)
    setattr(buffer, member, 1)
    with pytest.raises(TypeError, match="append, read_batch and ack"):
        CommitPolicy.batch(batch_size=1, flush_interval=1, buffer=buffer)

    async def write_batch(entries: tuple[WriteBufferEntry, ...]) -> None:
        pass

    with pytest.raises(TypeError, match="append, read_batch and ack"):
        BatchWriteScope(buffer, write_batch)
