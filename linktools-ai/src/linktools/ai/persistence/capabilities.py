#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-local accepted prefixes for explicitly atomic capability participants."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from uuid import uuid4
from weakref import WeakValueDictionary

from linktools.ai.core import CommitPolicy, ImmutableJsonMapping, JsonValue, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from ._capability_contracts import CapabilityPersistence, CapabilityWriter
from .state import StateStore, StateTransaction, StateStorageGroup, StateGroupTransaction, StoredOperation
from .durability import CommitObservation, DurableCommitState, run_durable_commit, await_owned_task
from ._capability_marker import ensure_capability_partition_marker, has_capability_partition_marker

_Persist = Callable[[StateTransaction, Mapping[str, JsonValue]], Awaitable[None]]


@dataclass
class _Write:
    id: str
    store: StateStore
    persist: _Persist
    payload: Mapping[str, JsonValue]
    receipt_key: bytes
    digest: str
    content: bytes
    buffered: bool
    durable: bool
    queued: bool = False


class _PartitionMarker:
    def __init__(self, store: StateStore, tasks: set[asyncio.Task[object]]) -> None:
        self.store = store
        self.tasks = tasks
        self.confirmed = False
        self.lock = asyncio.Lock()

    async def confirm_before(self, group: StateStorageGroup) -> None:
        if self.confirmed or group is self.store.storage_group:
            return
        async with self.lock:
            if self.confirmed:
                return

            async def observe() -> CommitObservation[None]:
                found = await self.store.read(has_capability_partition_marker)
                return CommitObservation(DurableCommitState.COMMITTED if found else DurableCommitState.NOT_COMMITTED)

            outcome = await run_durable_commit(
                lambda: self.store.mutate(ensure_capability_partition_marker), observe,
                background_tasks=self.tasks,
            )
            if outcome.state is DurableCommitState.COMMITTED:
                self.confirmed = True
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return
            if outcome.state is DurableCommitState.NOT_COMMITTED:
                if outcome.error is not None:
                    raise outcome.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error


class _WriteScope:
    def __init__(
        self, *, run_id: str, execution_id: str, store: StateStore,
        policy: CommitPolicy, tasks: set[asyncio.Task[object]], marker: _PartitionMarker | None,
    ) -> None:
        self.run_id = run_id
        self.execution_id = execution_id
        self.group = store.storage_group
        self.policy = policy
        self.tasks = tasks
        self.marker = marker
        self.id = uuid4().hex
        self.lock = asyncio.Lock()
        self.pending: list[_Write] = []
        self.confirmed: set[str] = set()
        self.receipts: dict[bytes, str] = {}
        self.accepting = True
        self.timer: asyncio.TimerHandle | None = None
        self.failure: BaseException | None = None
        self._scheduled = False

    def stop_timer(self) -> None:
        if self.timer is not None:
            self.timer.cancel()
            self.timer = None

    async def submit(self, write: _Write) -> None:
        synchronous = self.policy.mode == "sync" or self.policy.mode == "batch" and self.policy.buffer is None
        cancelled = False
        async with self.lock:
            if not self.accepting:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "capability write scope is sealed")
            known = self.receipts.get(write.receipt_key)
            if known is not None:
                if known != write.digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return
            previous = next((item for item in self.pending if item.receipt_key == write.receipt_key and item.store is write.store), None)
            if previous is not None:
                if previous.digest != write.digest:
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            else:
                self.pending.append(write)
                if write.buffered:
                    assert self.policy.buffer is not None
                    try:
                        accepted, append_error, cancelled = await await_owned_task(
                            asyncio.create_task(self.policy.buffer.append(
                                self.id, write.id, write.receipt_key.hex(), write.content, wait=False,
                            )), self.tasks,
                        )
                        if append_error is not None:
                            raise append_error
                    except BaseException as error:
                        # The remote append may have succeeded. Retain the local
                        # submission, but never count it as accepted pressure.
                        self.failure = error
                        raise
                    if not accepted:
                        self.pending.remove(write)
                        raise AIError(ErrorCode.STORAGE_CONFLICT, "capability write buffer rejected append")
                    write.queued = True
            if self.policy.mode == "batch" and self.policy.buffer is not None and self.failure is None:
                assert self.policy.batch_size is not None and self.policy.flush_interval is not None
                if sum(item.queued for item in self.pending) >= self.policy.batch_size:
                    self.stop_timer()
                    self._schedule()
                elif self.timer is None:
                    self.timer = asyncio.get_running_loop().call_later(
                        self.policy.flush_interval, self._schedule,
                    )
        if synchronous:
            await self.flush()
        if cancelled:
            raise asyncio.CancelledError

    def _schedule(self) -> None:
        self.stop_timer()
        if self._scheduled or self.failure is not None or not self.accepting:
            return
        self._scheduled = True
        task = asyncio.create_task(self._automatic_flush())
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)

    async def _automatic_flush(self) -> None:
        try:
            await self.flush()
        except BaseException as error:
            self.failure = error
        finally:
            self._scheduled = False

    async def prepare(self) -> "PreparedCapabilityWrites":
        await self.lock.acquire()
        self.stop_timer()
        try:
            confirmed = tuple(write for write in self.pending if write.id in self.confirmed)
            if confirmed:
                await self._ack(confirmed)
            prefix = tuple(self.pending)
            if any(write.durable for write in prefix) and self.marker is not None:
                await self.marker.confirm_before(self.group)
            buffered = tuple(write for write in prefix if write.buffered)
            if buffered:
                assert self.policy.buffer is not None
                queued = await self.policy.buffer.read_batch(self.id, limit=len(buffered))
                expected = {write.id: (write.receipt_key.hex(), write.content) for write in buffered}
                returned = tuple(entry[0] for entry in queued)
                if (len(set(returned)) != len(returned)
                        or any(expected.get(identity) != (key, content) for identity, key, content in queued)
                        or returned != tuple(write.id for write in buffered if write.id in returned)
                        or any(write.queued and write.id not in returned for write in buffered)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "capability buffer prefix changed")
            return PreparedCapabilityWrites(self, prefix)
        except BaseException:
            self.lock.release()
            raise

    async def flush(self) -> None:
        prepared = await self.prepare()
        try:
            if not prepared.writes:
                self.failure = None
                return
            if all(write.id in self.confirmed for write in prepared.writes):
                await prepared.settle()
                self.failure = None
                return
            outcome = await run_durable_commit(
                lambda: self.group.mutate(prepared.stores, prepared.apply),
                prepared.observation,
                background_tasks=self.tasks,
            )
            if outcome.state is DurableCommitState.COMMITTED:
                await prepared.settle()
                self.failure = None
                if outcome.cancelled:
                    raise asyncio.CancelledError
                return
            if outcome.state is DurableCommitState.NOT_COMMITTED:
                if outcome.error is not None:
                    raise outcome.error
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            if outcome.state is DurableCommitState.PARTIAL_INTEGRITY_ERROR:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from outcome.error
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN) from outcome.error
        except BaseException as error:
            self.failure = error
            raise
        finally:
            await prepared.abort()

    async def _ack(self, writes: tuple[_Write, ...]) -> None:
        buffered = [write.id for write in writes if write.buffered]
        if buffered:
            assert self.policy.buffer is not None
            await self.policy.buffer.ack(self.id, buffered)
        ids = {write.id for write in writes}
        self.pending[:] = [write for write in self.pending if write.id not in ids]
        self.confirmed.difference_update(ids)


class PreparedCapabilityWrites:
    """One locked prefix that can join an already-owned terminal transaction."""

    def __init__(self, scope: _WriteScope | None, writes: tuple[_Write, ...] = ()) -> None:
        self.scope = scope
        self.writes = writes
        self.marker = scope.marker if scope is not None and any(write.durable for write in writes) and scope.marker is not None and not scope.marker.confirmed else None
        stores = tuple(write.store for write in writes)
        if self.marker is not None:
            stores += (self.marker.store,)
        self.stores = tuple(dict.fromkeys(stores))
        self._released = False
        self.on_settled: Callable[[], None] | None = None

    async def apply(self, group: StateGroupTransaction) -> None:
        if self.marker is not None:
            await ensure_capability_partition_marker(group.transaction(self.marker.store))
        for write in self.writes:
            transaction = group.transaction(write.store)
            receipt = await transaction.get_operation(write.receipt_key)
            if receipt is not None:
                self._validate_receipt(receipt, write)
                continue
            await write.persist(transaction, write.payload)
            await transaction.insert_operation(StoredOperation(
                key_digest=write.receipt_key, stream_digest=write.receipt_key,
                sequence=0, state="committed", compactable=False,
                data={"request_digest": write.digest},
            ))

    @classmethod
    def _validate_receipt(cls, receipt: StoredOperation, write: _Write) -> None:
        if receipt.state != "committed" or receipt.data.get("request_digest") != write.digest:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)

    async def readback(self) -> bool:
        if not self.writes:
            return True
        assert self.scope is not None
        async def read(group: StateGroupTransaction) -> bool:
            complete = True
            for write in self.writes:
                receipt = await group.transaction(write.store).get_operation(write.receipt_key)
                if receipt is None:
                    complete = False
                else:
                    self._validate_receipt(receipt, write)
            if complete and self.marker is not None and not await has_capability_partition_marker(group.transaction(self.marker.store)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "capability writes lack their durable partition marker")
            return complete
        return await self.scope.group.read_many(self.stores, read)

    async def observation(self) -> CommitObservation[None]:
        try:
            committed = await self.readback()
        except AIError as error:
            if error.code is ErrorCode.IDEMPOTENCY_CONFLICT:
                return CommitObservation(DurableCommitState.NOT_COMMITTED, error=error)
            raise
        return CommitObservation(DurableCommitState.COMMITTED if committed else DurableCommitState.NOT_COMMITTED)

    async def settle(self) -> None:
        if self.scope is None:
            return
        # Confirmation precedes ack. An ack failure keeps a confirmed prefix;
        # explicit retry settles transport without replaying the source mutation.
        if self.marker is not None:
            self.marker.confirmed = True
        self.scope.confirmed.update(write.id for write in self.writes)
        self.scope.receipts.update((write.receipt_key, write.digest) for write in self.writes)
        await self.scope._ack(self.writes)
        if self.on_settled is not None:
            self.on_settled()
        await self.abort()

    async def abort(self) -> None:
        if not self._released and self.scope is not None:
            self.scope.lock.release()
        self._released = True


class _BoundWriter(CapabilityWriter):
    def __init__(self, scope: _WriteScope, store: StateStore, identity: tuple[str, int, str], namespace: str, tenant_id: str, *, durable: bool) -> None:
        self.durable = durable
        self.scope = scope
        self.store = store
        self.identity = identity
        self.namespace = namespace
        self.tenant_id = tenant_id
        self.persist: _Persist | None = None

    async def submit(self, *, key: str, payload: Mapping[str, JsonValue]) -> None:
        if not isinstance(key, str) or not key:
            raise ValueError("capability write key must be nonempty")
        if self.persist is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        content = canonical_json_bytes(dict(payload))
        copied = ImmutableJsonMapping(json.loads(content))
        receipt_key = hashlib.sha256(canonical_json_bytes([
            "capability-write", self.namespace, self.tenant_id, *self.identity, key,
        ])).digest()
        await self.scope.submit(_Write(
            id=uuid4().hex, store=self.store, persist=self.persist, payload=copied,
            receipt_key=receipt_key, digest=hashlib.sha256(content).hexdigest(), content=content,
            durable=self.durable,
            buffered=self.scope.policy.mode == "batch" and self.scope.policy.buffer is not None,
        ))

    async def flush(self) -> None:
        await self.scope.flush()


class RuntimeCapabilityWrites:
    """One Runtime's optional writers; groups never span run error domains."""

    def __init__(
        self, stores: Mapping[str, StateStore], *, namespace: str, tenant_id: str,
        durable_capability_ids: frozenset[str], require_active_run: Callable[[str, str], None],
        has_active_execution: Callable[[str], bool], marker_store: StateStore | None = None,
    ) -> None:
        self.stores = dict(stores)
        self._durable_capability_ids = durable_capability_ids
        self._require_active_run = require_active_run
        self._has_active_execution = has_active_execution
        self.namespace = namespace
        self.tenant_id = tenant_id
        self._scopes: dict[tuple[str, int, int, CommitPolicy], _WriteScope] = {}
        self._tasks: set[asyncio.Task[object]] = set()
        self._marker = _PartitionMarker(marker_store, self._tasks) if marker_store is not None else None
        self._bound: WeakValueDictionary[int, CapabilityPersistence] = WeakValueDictionary()
        self._sealed: set[str] = set()
        self._closing = False

    def bind(
        self, capability_id: str, run_id: str, execution_id: str,
        capability: CapabilityPersistence, *, revision: int = 1,
    ) -> CapabilityPersistence:
        self._require_active_run(run_id, execution_id)
        if self._closing or execution_id in self._sealed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not isinstance(capability.commit, CommitPolicy):
            raise TypeError("capability commit must be a CommitPolicy")
        store = self.stores[capability_id]
        key = (run_id, id(store.storage_group), id(capability.commit.buffer), capability.commit)
        scope = self._scopes.get(key)
        if scope is None:
            scope = _WriteScope(run_id=run_id, execution_id=execution_id, store=store, policy=capability.commit, tasks=self._tasks, marker=self._marker)
            self._scopes[key] = scope
        writer = _BoundWriter(scope, store, (capability_id, revision, run_id), self.namespace, self.tenant_id,
                              durable=capability_id in self._durable_capability_ids)
        bound = capability.bind_persistence(writer)
        if bound is capability or id(bound) in self._bound:
            raise ValueError("bind_persistence must return an independent run-local capability")
        if not isinstance(bound, CapabilityPersistence) or not callable(bound.persist_in_transaction):
            raise TypeError("bound capability must implement CapabilityPersistence")
        if bound.commit != capability.commit or bound.commit.buffer is not capability.commit.buffer:
            raise ValueError("binding cannot change capability commit policy")
        writer.persist = bound.persist_in_transaction
        self._bound[id(bound)] = bound
        return bound

    async def prepare_terminal(
        self, execution_id: str, run_id: str | None, storage_group: StateStorageGroup,
    ) -> PreparedCapabilityWrites:
        if self._has_active_execution(execution_id):
            self._sealed.add(execution_id)
        chosen = None
        scopes = tuple(scope for scope in self._scopes.values() if scope.execution_id == execution_id)
        for scope in scopes:
            scope.accepting = False
            scope.stop_timer()
        for scope in scopes:
            if scope.run_id == run_id and scope.policy.mode == "terminal" and scope.group is storage_group:
                chosen = scope
            else:
                await scope.flush()
        if chosen is None:
            self._release_execution(execution_id)
            return PreparedCapabilityWrites(None)
        prepared = await chosen.prepare()
        prepared.on_settled = lambda: self._release_execution(execution_id)
        return prepared

    def _release_execution(self, execution_id: str) -> None:
        for key, scope in tuple(self._scopes.items()):
            if scope.execution_id == execution_id and not scope.pending:
                scope.stop_timer()
                del self._scopes[key]

    def release_run(self, run_id: str, execution_id: str) -> None:
        """Seal a released producer without retiring another run's writers."""
        for key, scope in tuple(self._scopes.items()):
            if scope.run_id == run_id and scope.execution_id == execution_id:
                scope.accepting = False
                scope.stop_timer()
                if not scope.pending:
                    del self._scopes[key]
        if not self._has_active_execution(execution_id):
            self._sealed.discard(execution_id)

    async def close(self) -> None:
        self._closing = True
        scopes = tuple(self._scopes.values())
        for scope in scopes:
            scope.accepting = False
            scope.stop_timer()
        for scope in scopes:
            await scope.flush()
        pending = tuple(task for task in self._tasks if not task.done() and task is not asyncio.current_task())
        if pending:
            await asyncio.gather(*pending)
        self._scopes.clear()
        self._sealed.clear()
        self._bound.clear()




__all__ = ['CapabilityPersistence', 'CapabilityWriter', 'PreparedCapabilityWrites', 'RuntimeCapabilityWrites']
