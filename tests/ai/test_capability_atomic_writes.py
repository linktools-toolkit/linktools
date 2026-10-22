#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordinary Runtime capabilities share atomic publication without sharing runs."""

from __future__ import annotations
from ._runtime_test_helpers import model_features

import asyncio
from collections.abc import AsyncIterator, Hashable, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from copy import copy
from dataclasses import dataclass, field, replace
from hashlib import sha256
from pathlib import Path
from typing import Any, get_type_hints

import pytest
from pydantic_ai import AgentRunResult
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.tools import RunContext
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.capability import AgentContext, ExecutionCapability, TraceCapability
from linktools.ai.runtime import RuntimeFeatures
from linktools.ai.core import CommitPolicy, ExecutionStatus, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_runtime_database
from linktools.ai.observe import MetricRecorder, Metrics, Observation
from linktools.ai.observe._memory import InMemoryMetricStore
from linktools.ai.runtime import Runtime, RuntimeStorage
from linktools.ai.runtime.persistence import CapabilityPersistence, CapabilityWriter
from linktools.ai.runtime.state import (
    StateGroupTransaction, StateStorageGroup, StateStore, StateTransaction, StoredFact, StoredRecord,
)
from linktools.ai.runtime.state._filesystem import FilesystemStateStorageGroup
from linktools.ai.runtime.state._memory import InMemoryStateStorageGroup
from linktools.ai.runtime.state._sql import SqlStateStorageGroup, _SqlTransaction
from linktools.ai.storage import BatchWriteScopeFactory, InMemoryCacheBackend, StorageRoute

from ._runtime_test_helpers import _Models, _UsageFunctionModel
from .test_metrics_recording import _observation


@dataclass
class _Transaction:
    group: StateStorageGroup
    transaction: StateGroupTransaction
    stores: tuple[StateStore, ...]
    committed: bool = False


_current: ContextVar[_Transaction | None] = ContextVar("capability_test_transaction", default=None)


@dataclass
class _Write:
    capability: str
    payload: Mapping[str, JsonValue]
    frame: _Transaction
    transaction: StateTransaction
    sql_transaction: object | None


@dataclass
class _Probe:
    bound: dict[tuple[str, str], _Capability] = field(default_factory=dict)
    writes: list[_Write] = field(default_factory=list)
    stores: dict[str, StateStore] = field(default_factory=dict)
    frames: list[_Transaction] = field(default_factory=list)
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    fail: set[str] = field(default_factory=set)
    entered: asyncio.Event | None = None
    release: asyncio.Event | None = None

    def observe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        for backend in (InMemoryStateStorageGroup, FilesystemStateStorageGroup, SqlStateStorageGroup):
            original = backend.mutate

            async def mutate(group, stores, callback, *, original=original):
                frame = None

                async def observed(transaction):
                    nonlocal frame
                    if _current.get() is not None:
                        return await callback(transaction)
                    frame = _Transaction(group, transaction, tuple(stores))
                    self.frames.append(frame)
                    token = _current.set(frame)
                    try:
                        return await callback(transaction)
                    finally:
                        _current.reset(token)

                result = await original(group, stores, observed)
                if frame is not None:
                    frame.committed = True
                return result

            monkeypatch.setattr(backend, "mutate", mutate)

    async def binding(self, execution_id: str, capability: str) -> _Capability:
        async def wait() -> _Capability:
            while (execution_id, capability) not in self.bound:
                self.changed.clear()
                await self.changed.wait()
            return self.bound[execution_id, capability]
        return await asyncio.wait_for(wait(), 5)

    async def facts(self, capability: str) -> tuple[StoredFact, ...]:
        store = self.stores.get(capability)
        if store is None:
            return ()
        return await store.read(lambda tx: tx.scan_facts())


class _Buffer(InMemoryCacheBackend):
    def __init__(self, *, max_size: int | None = None) -> None:
        super().__init__(max_bytes=2**20, max_size=max_size)
        self.scopes: set[str] = set()
        self.acknowledged: list[tuple[str, tuple[str, ...]]] = []
        self.fail_ack = False
        self.acked = asyncio.Event()
        self.closed = False

    async def append(self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool) -> bool:
        result = await super().append(scope, id, key, content, wait=wait)
        if result:
            self.scopes.add(scope)
        return result

    async def ack(self, scope: str, ids: Sequence[str]) -> None:
        assert _current.get() is None, "buffer acknowledgement preceded transaction exit"
        if self.fail_ack:
            raise OSError("buffer acknowledgement unavailable")
        await super().ack(scope, ids)
        self.acknowledged.append((scope, tuple(ids)))
        self.acked.set()

    async def pending(self) -> int:
        counts = [len(await self.read_batch(scope, limit=1000)) for scope in self.scopes]
        return sum(counts)

    async def close(self) -> None:
        self.closed = True


class _Capability(AbstractCapability[AgentContext[None]]):
    def __init__(self, id: str, probe: _Probe, *, storage: StorageRoute | None, commit: CommitPolicy, terminal: bool = False) -> None:
        self.id = id
        self.probe = probe
        self.storage = storage
        self.commit = commit
        self.terminal = terminal
        self.writer: CapabilityWriter | None = None
        self.execution_id: str | None = None

    def bind_persistence(self, writer: CapabilityWriter) -> _Capability:
        bound = copy(self)
        bound.writer = writer
        return bound

    async def before_run(self, ctx: RunContext[AgentContext[None]]) -> None:
        assert self.writer is not None
        self.execution_id = ctx.deps.execution_id
        self.probe.bound[self.execution_id, self.id] = self
        self.probe.changed.set()

    async def after_run(self, ctx: RunContext[AgentContext[None]], *, result: AgentRunResult[Any]) -> AgentRunResult[Any]:
        if self.terminal:
            await self.submit("finished")
        return result

    async def submit(self, key: str, value: JsonValue = 1) -> None:
        assert self.writer is not None and self.execution_id is not None
        await self.writer.submit(key=key, payload={"run": self.execution_id, "key": key, "value": value})

    async def flush(self) -> None:
        assert self.writer is not None
        await self.writer.flush()

    async def persist_in_transaction(self, tx: StateTransaction, payload: Mapping[str, JsonValue]) -> None:
        frame = _current.get()
        assert frame is not None, "custom persistence must receive a real group transaction"
        store, = (store for store in frame.stores if frame.transaction.transaction(store) is tx)
        self.probe.stores[self.id] = store
        self.probe.writes.append(_Write(
            self.id, payload, frame, tx,
            tx._session.sync_session.get_transaction() if isinstance(tx, _SqlTransaction) else None,
        ))
        stream = sha256(str(payload["run"]).encode()).digest()
        owner = await tx.get_record(stream)
        if owner is None:
            await tx.insert_record(StoredRecord(
                stream, stream, None, self.id, str(payload["run"]), None, 0, None, 0, None, {},
            ))
        else:
            assert await tx.guard_record(stream, expected_storage_revision=owner.storage_revision) is not None
        sequence = await tx.next_sequence(stream)
        await tx.insert_fact(StoredFact(stream, sequence, stream, self.id, None, None, payload))
        if self.probe.entered is not None and self.probe.release is not None:
            self.probe.entered.set()
            await self.probe.release.wait()
        if self.id in self.probe.fail:
            raise AIError(ErrorCode.STORAGE_CONFLICT, "injected capability conflict")


class _Audit(_Capability):
    pass


class _Index(_Capability):
    pass


@dataclass
class _Opened:
    runtime: Runtime
    probe: _Probe
    templates: tuple[_Capability, _Capability]
    release: asyncio.Event
    ready: asyncio.Queue[None]
    executions: list[Any] = field(default_factory=list)

    async def start(self, *, session: bool = False) -> tuple[_Capability, _Capability]:
        agent = self.runtime.agents.get()
        target = await agent.create_session("session") if session else agent
        execution = await target.start("question")
        self.executions.append(execution)
        bound = tuple([await self.probe.binding(execution.execution_id, id) for id in ("audit", "index")])
        await asyncio.wait_for(self.ready.get(), 5)
        return bound

    async def finish(self) -> None:
        self.release.set()
        for execution in self.executions:
            assert (await execution.wait(timeout_seconds=5)).result.status is ExecutionStatus.SUCCEEDED


@asynccontextmanager
async def _opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, backend: str = "sqlite",
    policies: tuple[CommitPolicy, CommitPolicy], routes: tuple[StorageRoute | None, StorageRoute | None] = (None, None),
    terminal: bool = False, probe: _Probe | None = None,
    conversation_storage: StorageRoute | None = None, trace: bool = False,
    metrics: Metrics | None = None, declared_ids: bool = True, scope: str = "runtime",
) -> AsyncIterator[_Opened]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    probe = probe or _Probe()
    probe.observe(monkeypatch)
    engine = None
    if backend == "memory":
        storage = RuntimeStorage.in_memory()
    elif backend == "filesystem":
        storage = RuntimeStorage.filesystem(tmp_path / "state")
    elif backend == "sqlite":
        storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    else:
        engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'state.db'}")
        await provision_runtime_database(engine)
        storage = RuntimeStorage.sql(engine)
    if conversation_storage is not None:
        storage = RuntimeStorage.from_plan(replace(storage.plan, conversation=conversation_storage))
    release = asyncio.Event()
    ready: asyncio.Queue[None] = asyncio.Queue()

    async def model(messages, info):
        ready.put_nowait(None)
        await release.wait()
        return ModelResponse(parts=[TextPart("done")])

    templates = tuple(kind(id, probe, storage=route, commit=policy, terminal=terminal)
                      for kind, id, route, policy in zip((_Audit, _Index), ("audit", "index"), routes, policies))
    features = RuntimeFeatures("atomic-capabilities")
    features.agent("default", model="default", allow_tools=())
    features.capability(ExecutionCapability(retain_history=True,
        commit=CommitPolicy.terminal() if terminal else CommitPolicy.sync(),
        coordination=InMemoryCacheBackend(max_bytes=2**20),
    ), scope="runtime")
    if trace:
        features.capability(TraceCapability(commit=CommitPolicy.terminal()), scope="runtime")
    for id, capability in zip(("audit", "index"), templates):
        if not declared_ids:
            capability.id = None
        features.capability(capability, id=id, revision=1, scope=scope)
    try:
        async with Runtime.open("atomic-capabilities", storage=storage, features=model_features(_Models(_UsageFunctionModel(model)), features), metrics=metrics) as runtime:
            opened = _Opened(runtime, probe, templates, release, ready)
            try:
                yield opened
            finally:
                release.set()
                for execution in opened.executions:
                    await execution.wait(timeout_seconds=5)
    finally:
        release.set()
        if engine is not None:
            await engine.dispose()


def _batch(buffer: _Buffer, *, size: int = 100, interval: float = 300) -> CommitPolicy:
    return CommitPolicy.batch(buffer=buffer, batch_size=size, flush_interval=interval)


def test_public_persistence_types_resolve_without_runtime_import_cycles() -> None:
    for owner, names in ((CapabilityPersistence, ("bind_persistence", "persist_in_transaction")),
                         (CapabilityWriter, ("submit", "flush"))):
        for name in names:
            assert get_type_hints(getattr(owner, name))
    assert get_type_hints(_Capability.persist_in_transaction)["tx"] is StateTransaction


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
async def test_equal_policy_values_join_one_real_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    buffer = _Buffer()
    policies = (_batch(buffer), _batch(buffer))
    assert policies[0] == policies[1] and policies[0] is not policies[1]
    async with _opened(tmp_path, monkeypatch, backend=backend, policies=policies) as opened:
        audit, index = await opened.start()
        assert all(template.writer is None for template in opened.templates)
        assert audit is not opened.templates[0] and index is not opened.templates[1]
        await audit.submit("same-key", {"a": [1]})
        await index.submit("same-key", {"b": [2]})
        assert opened.probe.writes == []
        assert await buffer.pending() == 2
        await audit.flush()
        writes = opened.probe.writes
        assert {write.capability for write in writes} == {"audit", "index"}
        assert writes[0].frame is writes[1].frame and writes[0].frame.committed
        if backend in {"sqlite", "sql"}:
            assert writes[0].sql_transaction is not None
            assert writes[0].sql_transaction is writes[1].sql_transaction
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        assert await buffer.pending() == 0
        await opened.finish()
    assert not buffer.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql"))
async def test_participant_conflict_rolls_back_all_and_retains_explicit_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, backend=backend, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        opened.probe.fail.add("index")
        with pytest.raises(AIError) as failed:
            await audit.flush()
        assert failed.value.code is ErrorCode.STORAGE_CONFLICT
        assert await opened.probe.facts("audit") == await opened.probe.facts("index") == ()
        assert await buffer.pending() == 2 and buffer.acknowledged == []
        assert len(opened.probe.writes) == 2
        await asyncio.sleep(0)
        assert len(opened.probe.writes) == 2
        opened.probe.fail.clear()
        await index.flush()
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        assert await buffer.pending() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("difference", ("buffer", "size", "interval", "resource"))
async def test_distinct_transaction_rules_flush_independently(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: str) -> None:
    first, second = _Buffer(), _Buffer()
    policies = (_batch(first), _batch(second if difference == "buffer" else first,
                                   size=101 if difference == "size" else 100,
                                   interval=301 if difference == "interval" else 300))
    routes = (None, StorageRoute.sqlite(tmp_path / "other.db") if difference == "resource" else None)
    async with _opened(tmp_path, monkeypatch, policies=policies, routes=routes) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        await audit.flush()
        assert [write.capability for write in opened.probe.writes] == ["audit"]
        await index.flush()
        assert [write.capability for write in opened.probe.writes] == ["audit", "index"]
        assert opened.probe.writes[0].frame is not opened.probe.writes[1].frame


@pytest.mark.asyncio
async def test_sync_submission_confirms_without_waiting_for_other_capabilities(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _opened(tmp_path, monkeypatch, policies=(CommitPolicy.sync(), CommitPolicy.sync())) as opened:
        audit, index = await opened.start()
        await audit.submit("first")
        assert len(await opened.probe.facts("audit")) == 1
        assert await opened.probe.facts("index") == ()
        await index.submit("second")
        assert opened.probe.writes[0].frame is not opened.probe.writes[1].frame


@pytest.mark.asyncio
async def test_submission_keys_are_idempotent_and_payloads_are_frozen(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        value = {"items": [1]}
        await audit.submit("request", value)
        value["items"].append(2)
        await audit.submit("request", {"items": [1]})
        with pytest.raises(AIError) as pending_conflict:
            await audit.submit("request", {"items": [2]})
        assert pending_conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT
        await index.submit("request", {"items": [2]})
        await audit.flush()
        assert len(await opened.probe.facts("audit")) == 1
        assert (await opened.probe.facts("audit"))[0].data["value"] == {"items": [1]}
        before = len(opened.probe.writes)
        await audit.submit("request", {"items": [1]})
        await audit.flush()
        assert len(opened.probe.writes) == before
        with pytest.raises(AIError) as committed_conflict:
            await audit.submit("request", {"items": [3]})
            await audit.flush()
        assert committed_conflict.value.code is ErrorCode.IDEMPOTENCY_CONFLICT


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "filesystem", "sqlite", "sql", "separate-conversation", "pending-turn"))
async def test_terminal_custom_writes_join_the_existing_execution_transaction(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str) -> None:
    async with _opened(
        tmp_path, monkeypatch, backend="sqlite" if backend in {"separate-conversation", "pending-turn"} else backend,
        policies=(CommitPolicy.terminal(), CommitPolicy.terminal()), terminal=True,
        conversation_storage=StorageRoute.sqlite(tmp_path / "conversation.db") if backend == "separate-conversation" else None,
        trace=backend == "separate-conversation",
    ) as opened:
        await opened.start(session=backend == "pending-turn")
        repository = opened.runtime._state.execution.executions
        terminal_frames = []
        terminal_sql = []

        for name in ("commit_terminal_in_transaction", "insert_pending_turn_in_transaction", "commit_closed_terminal_in_transaction"):
            original = getattr(repository, name)

            async def commit(tx, *args, original=original, **kwargs):
                terminal_frames.append(_current.get())
                if isinstance(tx, _SqlTransaction):
                    terminal_sql.append(tx._session.sync_session.get_transaction())
                return await original(tx, *args, **kwargs)

            monkeypatch.setattr(repository, name, commit)
        assert opened.probe.writes == []
        await opened.finish()
        assert len(terminal_frames) == 1 and terminal_frames[0] is not None
        assert {write.capability for write in opened.probe.writes} == {"audit", "index"}
        assert all(write.frame is terminal_frames[0] for write in opened.probe.writes)
        if backend in {"sqlite", "sql"}:
            assert terminal_sql[0] is not None
            assert all(write.sql_transaction is terminal_sql[0] for write in opened.probe.writes)
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1


@pytest.mark.asyncio
async def test_lost_commit_reply_is_resolved_before_acknowledging_the_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        group = opened.runtime._state.execution.executions.state_store.storage_group
        original = group.mutate

        async def lost_reply(stores, callback):
            result = await original(stores, callback)
            if opened.probe.writes:
                assert await buffer.pending() == 2 and buffer.acknowledged == []
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "commit reply was lost")
            return result

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", lost_reply)
            await audit.flush()
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        assert len(opened.probe.writes) == 2 and await buffer.pending() == 0
        await index.flush()
        assert len(opened.probe.writes) == 2


@pytest.mark.asyncio
async def test_unresolved_commit_outcome_retains_all_entries_until_receipt_readback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        group = opened.runtime._state.execution.executions.state_store.storage_group
        original = group.mutate

        async def lost_reply(stores, callback):
            await original(stores, callback)
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "commit reply was lost")

        original_readback = group.read_many

        async def unavailable_readback(stores, callback):
            if any(store in opened.probe.stores.values() for store in stores):
                raise OSError("source readback unavailable")
            return await original_readback(stores, callback)

        with monkeypatch.context() as patch:
            patch.setattr(group, "mutate", lost_reply)
            patch.setattr(group, "read_many", unavailable_readback)
            with pytest.raises(AIError) as unknown:
                await audit.flush()
            assert unknown.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
        assert await buffer.pending() == 2 and buffer.acknowledged == []
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        await index.flush()
        assert await buffer.pending() == 0
        assert len(opened.probe.writes) == 2


@pytest.mark.asyncio
async def test_lost_buffer_ack_never_reexecutes_a_confirmed_domain_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        buffer.fail_ack = True
        try:
            await audit.flush()
        except (AIError, OSError):
            pass
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        assert await buffer.pending() == 2
        buffer.fail_ack = False
        await index.flush()
        assert await buffer.pending() == 0
        assert len(opened.probe.writes) == 2


@pytest.mark.asyncio
async def test_cancelled_flush_resolves_the_complete_group_before_returning(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        opened.probe.entered, opened.probe.release = asyncio.Event(), asyncio.Event()
        flushing = asyncio.create_task(audit.flush())
        await asyncio.wait_for(opened.probe.entered.wait(), 5)
        flushing.cancel()
        assert await buffer.pending() == 2 and buffer.acknowledged == []
        opened.probe.release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(flushing, 5)
        sizes = (len(await opened.probe.facts("audit")), len(await opened.probe.facts("index")))
        assert sizes in ((0, 0), (1, 1))
        assert await buffer.pending() == (2 if sizes == (0, 0) else 0)
        await index.flush()
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        assert await buffer.pending() == 0


@pytest.mark.asyncio
async def test_same_capabilities_keep_concurrent_run_buffers_separate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer, size=3), _batch(buffer, size=3))) as opened:
        first_audit, first_index = await opened.start()
        second_audit, second_index = await opened.start()
        assert first_audit is not second_audit and first_audit.writer is not second_audit.writer
        await first_audit.submit("same-key")
        await first_index.submit("same-key")
        await second_audit.submit("same-key")
        await second_index.submit("same-key")
        await asyncio.sleep(0)
        assert opened.probe.writes == []
        await first_audit.flush()
        assert {write.payload["run"] for write in opened.probe.writes} == {first_audit.execution_id}
        assert await buffer.pending() == 2
        await second_index.flush()
        assert {write.payload["run"] for write in opened.probe.writes} == {first_audit.execution_id, second_audit.execution_id}
        assert await buffer.pending() == 0


@pytest.mark.asyncio
async def test_shared_buffer_does_not_merge_runtime_owners(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    probe = _Probe()
    policies = (_batch(buffer, size=3), _batch(buffer, size=3))
    async with _opened(tmp_path, monkeypatch, policies=policies, probe=probe) as first:
        async with _opened(tmp_path, monkeypatch, policies=policies, probe=probe) as second:
            first_audit, first_index = await first.start()
            second_audit, second_index = await second.start()
            for capability in (first_audit, first_index, second_audit, second_index):
                await capability.submit("same-key")
            await asyncio.sleep(0)
            assert probe.writes == []
            await first_audit.flush()
            assert {write.payload["run"] for write in probe.writes} == {first_audit.execution_id}
            assert await buffer.pending() == 2
            await second_index.flush()
            assert await buffer.pending() == 0
        assert not buffer.closed
        await first_audit.submit("after-other-runtime-close")
        await first_audit.flush()
        assert any(write.payload["key"] == "after-other-runtime-close" for write in probe.writes)


@pytest.mark.asyncio
async def test_run_completion_drains_accepted_writes_without_closing_borrowed_buffer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    probe = _Probe()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer)), probe=probe) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        assert await buffer.pending() == 2
    assert len(probe.writes) == 2
    assert probe.writes[0].frame is probe.writes[1].frame
    assert probe.writes[0].frame.committed
    assert await buffer.pending() == 0 and not buffer.closed


@pytest.mark.asyncio
@pytest.mark.parametrize("trigger", ("threshold", "deadline"))
async def test_automatic_batch_trigger_commits_compatible_participants_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, trigger: str) -> None:
    buffer = _Buffer()
    policies = tuple(_batch(buffer, size=2 if trigger == "threshold" else 100,
                            interval=300 if trigger == "threshold" else 0.02) for _ in range(2))
    async with _opened(tmp_path, monkeypatch, policies=policies) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        await asyncio.wait_for(buffer.acked.wait(), 5)
        assert len(opened.probe.writes) == 2
        assert opened.probe.writes[0].frame is opened.probe.writes[1].frame
        assert await buffer.pending() == 0


@pytest.mark.asyncio
async def test_full_buffer_rejects_without_blocking_an_explicit_drain(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer(max_size=1)
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, index = await opened.start()
        await audit.submit("first")
        with pytest.raises(AIError) as full:
            await asyncio.wait_for(index.submit("second"), 2)
        assert full.value.code is ErrorCode.STORAGE_CONFLICT
        await asyncio.wait_for(audit.flush(), 2)
        assert [write.capability for write in opened.probe.writes] == ["audit"]
        await index.submit("second")
        await index.flush()
        assert [write.capability for write in opened.probe.writes] == ["audit", "index"]
        assert await buffer.pending() == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", (False, True))
async def test_interrupted_append_keeps_the_admitted_write_recoverable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, cancelled: bool) -> None:
    class InterruptedBuffer(_Buffer):
        interrupted = False

        async def append(self, scope: str, id: str, key: Hashable, content: bytes | None, *, wait: bool) -> bool:
            result = await super().append(scope, id, key, content, wait=wait)
            if not self.interrupted:
                self.interrupted = True
                if cancelled:
                    raise asyncio.CancelledError
                raise OSError("append reply lost")
            return result

    buffer = InterruptedBuffer()
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
        audit, _ = await opened.start()
        with pytest.raises(asyncio.CancelledError if cancelled else OSError):
            await audit.submit("accepted-before-interruption")
        assert await buffer.pending() == 1
        await audit.flush()
        assert len(await opened.probe.facts("audit")) == 1
        assert await buffer.pending() == 0
        await audit.submit("accepted-before-interruption")
        await audit.flush()
        assert len(opened.probe.writes) == 1


@pytest.mark.asyncio
async def test_registered_identity_is_bound_without_mutating_the_capability_template(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with _opened(tmp_path, monkeypatch, policies=(CommitPolicy.sync(), CommitPolicy.sync()), declared_ids=False) as opened:
        audit, index = await opened.start()
        assert all(template.id is None and template.writer is None for template in opened.templates)
        assert (audit.id, index.id) == ("audit", "index")
        await audit.submit("audit")
        await index.submit("index")
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1


@pytest.mark.asyncio
async def test_persistence_participation_requires_runtime_scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(AIError) as rejected:
        async with _opened(tmp_path, monkeypatch, policies=(CommitPolicy.sync(), CommitPolicy.sync()), scope="agent"):
            pytest.fail("unsupported persistence scope must be rejected before runtime use")
    assert rejected.value.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID


@pytest.mark.asyncio
async def test_metric_weak_failure_does_not_settle_custom_atomic_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    recorders: list[MetricRecorder] = []

    class FailedMetricStore(InMemoryMetricStore):
        async def put_observations(self, namespace: str, observations: Sequence[Observation]) -> None:
            raise OSError("metric source unavailable")

    class RecordingMetrics(Metrics):
        def recorder(self, *, batch_scopes: BatchWriteScopeFactory | None = None) -> MetricRecorder:
            recorder = super().recorder(batch_scopes=batch_scopes)
            recorders.append(recorder)
            return recorder

    buffer = _Buffer()
    metrics = RecordingMetrics.from_store(FailedMetricStore(), commit=_batch(buffer))
    async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer)), metrics=metrics) as opened:
        audit, index = await opened.start()
        await audit.submit("audit")
        await index.submit("index")
        assert await recorders[0].try_record(_observation("weak-metric"))
        assert (await opened.runtime.metrics.flush()).completed
        status = opened.runtime.metrics.status()
        assert status.pending == 0 and status.unconfirmed > 0 and status.write_failures > 0
        assert opened.probe.writes == [] and await buffer.pending() == 2
        await audit.flush()
        assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
        assert await buffer.pending() == 0


@pytest.mark.asyncio
async def test_close_failure_retains_writes_and_does_not_erase_the_execution_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    buffer = _Buffer()
    opened = None
    try:
        with pytest.raises(AIError) as retained_failure:
            async with _opened(tmp_path, monkeypatch, policies=(_batch(buffer), _batch(buffer))) as opened:
                audit, index = await opened.start()
                await audit.submit("audit")
                await index.submit("index")
                opened.probe.fail.add("index")
                try:
                    with pytest.raises(AIError) as failed:
                        await asyncio.wait_for(opened.runtime.close(), 5)
                    assert failed.value.code is ErrorCode.STORAGE_CONFLICT
                    assert await buffer.pending() == 2 and buffer.acknowledged == []
                    assert not buffer.closed
                    assert await opened.probe.facts("audit") == await opened.probe.facts("index") == ()
                finally:
                    opened.probe.fail.clear()
                    opened.release.set()
                await index.flush()
                assert len(await opened.probe.facts("audit")) == len(await opened.probe.facts("index")) == 1
                assert await buffer.pending() == 0 and not buffer.closed
                with pytest.raises(AIError) as execution_failure:
                    await asyncio.wait_for(opened.runtime.close(), 5)
                assert execution_failure.value.code is ErrorCode.STORAGE_CONFLICT
                opened.executions.clear()
        assert retained_failure.value.code is ErrorCode.STORAGE_CONFLICT
    finally:
        if opened is not None:
            await opened.runtime._state.close()
    assert not buffer.closed
