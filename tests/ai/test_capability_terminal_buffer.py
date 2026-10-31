#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Terminal capability buffers retain actual payloads until source publication."""

import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from linktools.ai.core import CommitPolicy, ImmutableJsonMapping
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.observe import Metrics

from .test_capability_atomic_writes import _Buffer, _opened

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite"))
@pytest.mark.parametrize("shared_buffer", (False, True))
async def test_terminal_payloads_join_source_only_at_completion(tmp_path: Path, monkeypatch, backend, shared_buffer) -> None:
    first = _Buffer()
    second = first if shared_buffer else _Buffer()
    policies = tuple(CommitPolicy.terminal(buffer=buffer) for buffer in (first, second))
    async with _opened(tmp_path, monkeypatch, backend=backend, policies=policies) as opened:
        audit, index = await opened.start()
        value = {"items": [1]}
        await audit.submit("audit", value)
        await index.submit("index", {"items": [2]})
        value["items"].append(999)
        assert opened.probe.writes == []
        scopes = (audit.writer.scope, index.writer.scope)
        for scope in set(scopes):
            entries = await scope.policy.buffer.read_batch(scope.id, limit=10)
            assert entries and all(isinstance(content, bytes) for _id, _key, content in entries)
            for _id, _key, content in entries:
                assert json.loads(content)["value"]["items"] in ([1], [2])
            # The transport, not this optional in-process view, supplies the write payload.
            for write in scope.pending:
                write.payload = ImmutableJsonMapping({"discarded": True})
        await opened.finish()
        assert opened.probe.writes[0].frame is opened.probe.writes[1].frame
        assert (await opened.probe.facts("audit"))[0].data["value"] == {"items": [1]}
        assert (await opened.probe.facts("index"))[0].data["value"] == {"items": [2]}
        for scope in set(scopes):
            if scope._cleanup_task is not None:
                await scope._cleanup_task
        assert await first.pending() == await second.pending() == 0
    assert not first.closed and not second.closed


async def test_terminal_full_buffer_rejects_without_blocking_or_source_fallback(tmp_path: Path, monkeypatch) -> None:
    buffer = _Buffer(max_size=1)
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, index = await opened.start()
        await audit.submit("accepted")
        with pytest.raises(AIError) as error:
            await asyncio.wait_for(index.submit("rejected"), 1)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert opened.probe.writes == [] and await buffer.pending() == 1
        # A caller can explicitly publish accepted writes, then retry a rejected
        # submission. Neither rejection nor close silently invents its payload.
        await audit.flush()
        assert len(await opened.probe.facts("audit")) == 1
        assert await opened.probe.facts("index") == ()
        await index.submit("rejected")
        await opened.finish()
        assert len(await opened.probe.facts("index")) == 1


async def test_shared_terminal_buffer_does_not_merge_runs(tmp_path: Path, monkeypatch) -> None:
    buffer = _Buffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        first, _ = await opened.start()
        second, _ = await opened.start()
        await first.submit("one")
        await second.submit("two")
        assert first.writer.scope is not second.writer.scope
        assert first.writer.scope.id != second.writer.scope.id
        await first.flush()
        assert await buffer.pending() == 1
        assert len(opened.probe.writes) == 1
        await opened.finish()
        assert len(opened.probe.writes) == 2


async def test_terminal_unknown_append_cannot_fall_back_to_local_payload(tmp_path: Path, monkeypatch) -> None:
    class Unavailable(_Buffer):
        async def append(self, *args, **kwargs):
            raise OSError("unknown buffer admission")

    buffer = Unavailable()
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, _ = await opened.start()
        with pytest.raises(OSError):
            await audit.submit("unknown")
        with pytest.raises(AIError) as error:
            await audit.flush()
        assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert opened.probe.writes == []
        scope = audit.writer.scope
        write, = scope.pending
        with pytest.raises(AIError) as closing:
            await opened.runtime._run_store.capability_writes.close()
        assert closing.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert scope.pending == [write] and buffer.acknowledged == []
        assert opened.probe.writes == []
        await _Buffer.append(buffer, scope.id, write.id, write.receipt_key.hex(), write.content, wait=False)
        await audit.flush()
        assert len(await opened.probe.facts("audit")) == 1


async def test_terminal_confirmation_does_not_wait_for_buffer_ack(tmp_path: Path, monkeypatch) -> None:
    entered, release = asyncio.Event(), asyncio.Event()

    class Delayed(_Buffer):
        async def ack(self, scope: str, ids: Sequence[str]) -> None:
            entered.set()
            await release.wait()
            await super().ack(scope, ids)

    buffer = Delayed()
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, _ = await opened.start()
        await audit.submit("accepted")
        try:
            await opened.finish()
            await asyncio.wait_for(entered.wait(), 1)
            assert len(await opened.probe.facts("audit")) == 1
            assert await buffer.pending() == 1
        finally:
            release.set()
            await audit.writer.scope._cleanup_task
        assert await buffer.pending() == 0


async def test_metrics_still_rejects_terminal_payload_policy() -> None:
    with pytest.raises(TypeError):
        Metrics.in_memory(commit=CommitPolicy.terminal(buffer=_Buffer()))


async def test_terminal_scopes_rollback_together_then_ack_only_confirmed_prefixes(tmp_path: Path, monkeypatch) -> None:
    first, second = _Buffer(), _Buffer()
    policies = tuple(CommitPolicy.terminal(buffer=buffer) for buffer in (first, second))
    async with _opened(tmp_path, monkeypatch, backend="sqlite", policies=policies) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        scope = audit.writer.scope
        manager = opened.runtime._run_store.capability_writes
        opened.probe.fail.add("index")
        prepared = await manager.prepare_terminal(scope.execution_id, scope.run_id, scope.group)
        try:
            with pytest.raises(AIError):
                await scope.group.mutate(prepared.stores, prepared.apply)
        finally:
            await prepared.abort()
        assert await opened.probe.facts("audit") == await opened.probe.facts("index") == ()
        assert await first.pending() == await second.pending() == 1
        assert first.acknowledged == second.acknowledged == []
        opened.probe.fail.clear()
        prepared = await manager.prepare_terminal(scope.execution_id, scope.run_id, scope.group)
        try:
            await scope.group.mutate(prepared.stores, prepared.apply)
            await prepared.settle()
        finally:
            await prepared.abort()
        for owner in (audit.writer.scope, index.writer.scope):
            await owner._cleanup_task
        assert await first.pending() == await second.pending() == 0
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        await opened.finish()


async def test_terminal_unknown_source_requires_receipt_confirmation_without_reexecution(tmp_path: Path, monkeypatch) -> None:
    buffer = _Buffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        group = audit.writer.scope.group
        mutate, read_many = group.mutate, group.read_many

        async def lose_ack(stores, callback):
            await mutate(stores, callback)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        async def unavailable(stores, callback):
            raise OSError("source unavailable")

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", lose_ack)
            patch.setattr(group, "read_many", unavailable)
            with pytest.raises(AIError) as error:
                await audit.flush()
            assert error.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert group.read_many == read_many
        assert len(opened.probe.writes) == 2
        assert await buffer.pending() == 2 and buffer.acknowledged == []
        await index.flush()
        assert len(opened.probe.writes) == 2 and await buffer.pending() == 0
        await opened.finish()


async def test_terminal_lost_payload_is_not_reconstructed_from_local_state(tmp_path: Path, monkeypatch) -> None:
    buffer = _Buffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, _ = await opened.start()
        await audit.submit("retained")
        scope = audit.writer.scope
        entry, = await buffer.read_batch(scope.id, limit=2)
        await buffer.ack(scope.id, (entry[0],))
        with pytest.raises(AIError) as error:
            await audit.flush()
        assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert opened.probe.writes == []
        # Restore precisely the original transport receipt, not a new execution.
        assert await buffer.append(scope.id, *entry, wait=False)
        await audit.flush()
        assert len(await opened.probe.facts("audit")) == 1
        await opened.finish()


async def test_terminal_ack_failure_cannot_reverse_success_or_repeat_source_writes(tmp_path: Path, monkeypatch) -> None:
    buffer = _Buffer()
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, _ = await opened.start()
        await audit.submit("retained")
        buffer.fail_ack = True
        try:
            await opened.finish()
            await audit.writer.scope._cleanup_task
            assert await buffer.pending() == 1
            assert len(opened.probe.writes) == 1
            with pytest.raises(OSError):
                await audit.flush()
            assert len(opened.probe.writes) == 1
        finally:
            buffer.fail_ack = False
            await audit.flush()
        assert await buffer.pending() == 0
        assert len(await opened.probe.facts("audit")) == 1


async def test_terminal_rejection_query_requires_original_error_and_live_owner(tmp_path: Path, monkeypatch) -> None:
    buffer = _Buffer(max_size=1)
    policy = CommitPolicy.terminal(buffer=buffer)
    async with _opened(tmp_path, monkeypatch, backend="memory", policies=(policy, policy)) as opened:
        audit, index = await opened.start()
        await audit.submit("accepted")
        with pytest.raises(AIError) as rejected:
            await index.submit("rejected")
        error = rejected.value
        scope = audit.writer.scope
        manager = opened.runtime._run_store.capability_writes
        assert manager.is_buffer_rejection(scope.run_id, execution_id=scope.execution_id, error=error)
        assert not manager.is_buffer_rejection("other", execution_id=scope.execution_id, error=error)
        assert not manager.is_buffer_rejection(scope.run_id, execution_id="other", error=error)
        for unrelated in (AIError(ErrorCode.STORAGE_CONFLICT), AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN), OSError()):
            assert not manager.is_buffer_rejection(scope.run_id, execution_id=scope.execution_id, error=unrelated)
        other, _ = await opened.start()
        assert not manager.is_buffer_rejection(other.writer.scope.run_id, execution_id=other.execution_id, error=error)
        # A definite rejection is not sticky: explicit publication and resubmission remain legal.
        await audit.flush()
        await index.submit("rejected")
        await opened.finish()
        if scope._cleanup_task is not None:
            await scope._cleanup_task
        assert not manager.is_buffer_rejection(scope.run_id, execution_id=scope.execution_id, error=error)
