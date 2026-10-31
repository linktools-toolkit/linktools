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

from ..core import CommitPolicy, ImmutableJsonMapping, JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
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


class _BufferRejection(AIError):
    def __init__(self, scope: "_WriteScope") -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT, "capability write buffer rejected append")
        self.scope = scope


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
        self._cleanup_task: asyncio.Task[None] | None = None

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
                        raise _BufferRejection(self)
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
            prefix = tuple(write for write in self.pending if write.id not in self.confirmed)
            if any(write.durable for write in prefix) and self.marker is not None:
                await self.marker.confirm_before(self.group)
            buffered = tuple(write for write in prefix if write.buffered)
            if buffered:
                assert self.policy.buffer is not None
                retained = tuple(write for write in self.pending if write.buffered)
                queued = await self.policy.buffer.read_batch(self.id, limit=len(retained))
                expected = {write.id: (write.receipt_key.hex(), write.content) for write in retained}
                returned = tuple(entry[0] for entry in queued)
                if (len(set(returned)) != len(returned)
                        or any(expected.get(identity) != (key, content) for identity, key, content in queued)
                        or returned != tuple(write.id for write in retained if write.id in returned)
                        or any(write.queued and write.id not in returned for write in buffered)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "capability buffer prefix changed")
                if any(write.id not in returned for write in buffered):
                    raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN, "capability buffer admission is unresolved")
                contents = {identity: content for identity, _key, content in queued}
                for write in buffered:
                    content = contents[write.id]
                    assert content is not None
                    write.payload = ImmutableJsonMapping(json.loads(content))
            return PreparedCapabilityWrites(self, prefix)
        except BaseException:
            self.lock.release()
            raise

    async def flush(self, *, on_settled: Callable[[], None] | None = None) -> None:
        prepared = await self.prepare()
        prepared.on_settled = on_settled
        try:
            if not prepared.writes:
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

    async def _ack(self, writes: tuple[_Write, ...], *, lock_held: bool = False) -> None:
        buffered = [write.id for write in writes if write.buffered]
        if buffered:
            assert self.policy.buffer is not None
            await self.policy.buffer.ack(self.id, buffered)
        if lock_held:
            self._forget_confirmed(writes)
        else:
            async with self.lock:
                self._forget_confirmed(writes)

    def _forget_confirmed(self, writes: tuple[_Write, ...]) -> None:
        ids = {write.id for write in writes}
        self.pending[:] = [write for write in self.pending if write.id not in ids]
        self.confirmed.difference_update(ids)

    async def acknowledge_confirmed(self) -> None:
        async with self.lock:
            writes = tuple(write for write in self.pending if write.id in self.confirmed)
        if writes:
            await self._ack(writes)

    def schedule_acknowledgement(self, settled: Callable[[], None]) -> None:
        if self._cleanup_task is not None and not self._cleanup_task.done():
            return

        async def acknowledge() -> None:
            try:
                await self.acknowledge_confirmed()
            except BaseException as error:
                self.failure = error
            finally:
                settled()

        task = asyncio.create_task(acknowledge())
        self._cleanup_task = task
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


class PreparedCapabilityWrites:
    """Locked prefixes joining one already-owned transaction."""

    def __init__(self, scope: _WriteScope | None, writes: tuple[_Write, ...] = ()) -> None:
        self._set_prefixes(() if scope is None else ((scope, writes),))
        self._released = False
        self.on_settled: Callable[[], None] | None = None

    def _set_prefixes(self, prefixes: tuple[tuple[_WriteScope, tuple[_Write, ...]], ...]) -> None:
        self._prefixes = prefixes
        self.scope = prefixes[0][0] if prefixes else None
        self.writes = tuple(write for _scope, writes in prefixes for write in writes)
        self._markers = tuple(dict.fromkeys(
            scope.marker for scope, writes in prefixes
            if any(write.durable for write in writes) and scope.marker is not None and not scope.marker.confirmed
        ))
        stores = tuple(write.store for write in self.writes) + tuple(marker.store for marker in self._markers)
        self.stores = tuple(dict.fromkeys(stores))

    @classmethod
    def _combine(cls, prepared: tuple["PreparedCapabilityWrites", ...]) -> "PreparedCapabilityWrites":
        combined = cls(None)
        combined._set_prefixes(tuple(prefix for value in prepared for prefix in value._prefixes))
        if combined.scope is not None and any(
            scope.group is not combined.scope.group for scope, _writes in combined._prefixes
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT, "capability prefixes require one storage group")
        for value in prepared:
            value._released = True
        return combined

    async def apply(self, group: StateGroupTransaction) -> None:
        for marker in self._markers:
            await ensure_capability_partition_marker(group.transaction(marker.store))
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
            if complete:
                for marker in self._markers:
                    if not await has_capability_partition_marker(group.transaction(marker.store)):
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
        for marker in self._markers:
            marker.confirmed = True
        for scope, writes in self._prefixes:
            scope.confirmed.update(write.id for write in writes)
            scope.receipts.update((write.receipt_key, write.digest) for write in writes)
        if self.on_settled is not None:
            # Terminal publication must return its canonical result independently
            # of transport cleanup. The owner retains failed acknowledgements.
            await self.abort()
            for scope, _writes in self._prefixes:
                if any(write.buffered and write.id in scope.confirmed for write in scope.pending):
                    scope.schedule_acknowledgement(self.on_settled)
                else:
                    await scope.acknowledge_confirmed()
            self.on_settled()
        else:
            for scope, _writes in self._prefixes:
                confirmed = tuple(write for write in scope.pending if write.id in scope.confirmed)
                if confirmed:
                    await scope._ack(confirmed, lock_held=True)
            await self.abort()

    async def abort(self) -> None:
        if not self._released:
            for scope, _writes in reversed(self._prefixes):
                scope.lock.release()
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
            buffered=self.scope.policy.mode in {"batch", "terminal"} and self.scope.policy.buffer is not None,
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

    def is_buffer_rejection(
        self, agent_run_id: str, *, execution_id: str, error: BaseException,
    ) -> bool:
        """Recognize this live producer's definite non-admission, never source uncertainty."""
        return (
            isinstance(error, _BufferRejection)
            and error.scope.run_id == agent_run_id
            and error.scope.execution_id == execution_id
            and any(scope is error.scope for scope in self._scopes.values())
        )

    async def prepare_terminal(
        self, execution_id: str, run_id: str | None, storage_group: StateStorageGroup,
    ) -> PreparedCapabilityWrites:
        if self._has_active_execution(execution_id):
            self._sealed.add(execution_id)
        scopes = tuple(scope for scope in self._scopes.values() if scope.execution_id == execution_id)
        for scope in scopes:
            scope.accepting = False
            scope.stop_timer()
        chosen: list[_WriteScope] = []
        for scope in scopes:
            if scope.run_id == run_id and scope.policy.mode == "terminal" and scope.group is storage_group:
                chosen.append(scope)
            else:
                await scope.flush(on_settled=lambda: self._release_execution(execution_id))
        if not chosen:
            self._release_execution(execution_id)
            return PreparedCapabilityWrites(None)
        prefixes: list[PreparedCapabilityWrites] = []
        try:
            for scope in sorted(chosen, key=lambda value: value.id):
                prefixes.append(await scope.prepare())
            prepared = PreparedCapabilityWrites._combine(tuple(prefixes))
        except BaseException:
            for prefix in reversed(prefixes):
                await prefix.abort()
            raise
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
