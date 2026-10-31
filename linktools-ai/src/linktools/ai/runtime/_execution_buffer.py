#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-owned admission of frozen payloads before optional source writes."""

import asyncio
import hashlib
import json
from dataclasses import dataclass, field
from uuid import uuid4

from ..core import WriteBufferBackend, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..execution import ExecutionRecord, ExecutionStartReservation, ExecutionTerminalCommit
from ..persistence.codec import decode_domain, encode_domain
from ..persistence.contracts import ExecutionEventAppend
from ..persistence.durability import await_owned_task

_BUFFER_TIMEOUT_SECONDS = 5.0


@dataclass(frozen=True, slots=True)
class _ExecutionPayloadCapture:
    execution: ExecutionRecord
    reservation: ExecutionStartReservation | None = None
    terminal: ExecutionTerminalCommit | None = None
    audit_events: tuple[ExecutionEventAppend, ...] = ()
    checkpoint_payload: bytes | None = None

    def encode(self) -> bytes:
        return canonical_json_bytes({
            "format": "execution-payload-v1",
            "execution": encode_domain(self.execution),
            "reservation": None if self.reservation is None else encode_domain(self.reservation),
            "terminal": None if self.terminal is None else encode_domain(self.terminal),
            "audit_events": [encode_domain(event) for event in self.audit_events],
            "checkpoint_payload": None if self.checkpoint_payload is None else encode_domain(self.checkpoint_payload),
        })

    @classmethod
    def decode(cls, content: bytes) -> "_ExecutionPayloadCapture":
        try:
            value = json.loads(content)
            if not isinstance(value, dict) or value.get("format") != "execution-payload-v1":
                raise ValueError("invalid execution payload format")
            execution = decode_domain(value["execution"], ExecutionRecord)
            reservation = None if value["reservation"] is None else decode_domain(value["reservation"], ExecutionStartReservation)
            terminal = None if value["terminal"] is None else decode_domain(value["terminal"], ExecutionTerminalCommit)
            if not isinstance(value["audit_events"], list):
                raise ValueError("invalid execution audit events")
            audit_events = tuple(decode_domain(event, ExecutionEventAppend) for event in value["audit_events"])
            if (reservation is not None and reservation.execution.execution_id != execution.execution_id
                    or terminal is not None and terminal.execution != execution):
                raise ValueError("invalid execution payload identity")
            checkpoint_payload = (None if value["checkpoint_payload"] is None
                                  else decode_domain(value["checkpoint_payload"], bytes))
            return cls(execution, reservation, terminal, audit_events, checkpoint_payload)
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


@dataclass(slots=True)
class _Entry:
    id: str
    digest: str
    settled: bool = False
    admitted: bool = False


@dataclass(slots=True)
class _Scope:
    id: str = field(default_factory=lambda: "execution:" + uuid4().hex)
    entries: list[_Entry] = field(default_factory=list)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    failure: BaseException | None = None
    cleanup_failure: BaseException | None = None
    cleanup: asyncio.Task[None] | None = None
    sealed: bool = False
    released: bool = False
    uncertain: tuple[str, str, bytes] | None = None
    confirmed: set[str] = field(default_factory=set)
    producer_generation: int | None = None
    retired: bool = False
    audit_confirmed: int = 0
    audit_verified: int = 0
    rejection_prefix: tuple[_ExecutionPayloadCapture | None, tuple[str, ...]] | None = None


class _ExecutionBufferRejection(AIError):
    def __init__(self, scope: _Scope) -> None:
        super().__init__(ErrorCode.STORAGE_UNAVAILABLE, safe_details={"phase": "execution_buffer_admission"})
        self.scope = scope


class _ExecutionPayloadBuffer:
    """Borrow a transport; source owners alone confirm or discard its prefixes."""

    def __init__(self, backend: WriteBufferBackend) -> None:
        self._backend = backend
        self._scopes: dict[str, _Scope] = {}
        self._retired_scopes: list[_Scope] = []
        self._accepting = True
        self._tasks: set[asyncio.Task[object]] = set()

    def bind_producer(self, execution_id: str, producer_generation: int) -> None:
        scope = self._scopes.get(execution_id)
        if scope is None:
            self._scopes[execution_id] = _Scope(producer_generation=producer_generation)
            return
        if scope.producer_generation is None and not scope.released:
            scope.producer_generation = producer_generation
            return
        if scope.producer_generation == producer_generation and not scope.released:
            return
        if (not scope.retired or not scope.released or scope.producer_generation is None
                or producer_generation <= scope.producer_generation):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if scope.entries or scope.cleanup_failure is not None:
            self._retired_scopes.append(scope)
        self._scopes[execution_id] = _Scope(producer_generation=producer_generation)

    def producer_generation(self, execution_id: str) -> int:
        scope = self._scopes.get(execution_id)
        if scope is None or scope.producer_generation is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return scope.producer_generation

    def require_producer(self, execution_id: str, producer_generation: int) -> None:
        scope = self._scopes.get(execution_id)
        if scope is None or scope.released or scope.producer_generation != producer_generation:
            raise AIError(ErrorCode.STORAGE_CONFLICT)

    def retire(self, execution_id: str, producer_generation: int, *, abandon: bool = False) -> None:
        scope = self._scopes.get(execution_id)
        if scope is None or scope.producer_generation != producer_generation:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if abandon:
            if scope.sealed and any(not entry.settled for entry in scope.entries):
                raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
            self.discard(execution_id)
        elif any(not entry.settled for entry in scope.entries):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        scope.released = True
        scope.retired = True
        self._drop_settled_payload(scope)

    async def append(self, capture: _ExecutionPayloadCapture) -> None:
        if not self._accepting:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        execution_id, content = capture.execution.execution_id, capture.encode()
        task = asyncio.create_task(self._append(execution_id, content), name="execution-buffer-admission")
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        _, error, cancelled = await await_owned_task(task, self._tasks)
        if error is not None:
            raise error
        if cancelled:
            raise asyncio.CancelledError

    async def _append(self, execution_id: str, content: bytes) -> None:
        scope = self._scopes.get(execution_id)
        if scope is None:
            scope = self._scopes[execution_id] = _Scope()
        async with scope.lock:
            if scope.failure is not None:
                raise scope.failure
            if scope.sealed or scope.released:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            entry = _Entry(uuid4().hex, hashlib.sha256(content).hexdigest())
            scope.entries.append(entry)
            scope.uncertain = (entry.id, execution_id, content)
            try:
                async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
                    admitted = await self._backend.append(scope.id, *scope.uncertain, wait=False)
            except BaseException as error:
                try:
                    await self._read_entries(scope, execution_id)
                    admitted = True
                except BaseException:
                    scope.failure = AIError(ErrorCode.STORAGE_UNAVAILABLE,
                                            safe_details={"phase": "execution_buffer_admission"})
                    raise scope.failure from error
            if not admitted:
                scope.entries.remove(entry)
                scope.uncertain = None
                scope.failure = _ExecutionBufferRejection(scope)
                raise scope.failure
            entry.admitted = True
            scope.uncertain = None

    async def _read_entries(self, scope: _Scope, execution_id: str) -> dict[str, bytes]:
        expected = {entry.id: entry for entry in scope.entries}
        async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
            entries = await self._backend.read_batch(scope.id, limit=len(expected) + 1)
        found = {}
        for id, key, content in entries:
            entry = expected.get(id)
            if (entry is None or id in found or key != execution_id or not isinstance(content, bytes)
                    or hashlib.sha256(content).hexdigest() != entry.digest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            found[id] = content
        if tuple(id for id, _key, _content in entries if not expected[id].settled) != tuple(
            entry.id for entry in scope.entries if not entry.settled
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return found

    async def read(
        self, execution_id: str, *, seal: bool = False, rejection: BaseException | None = None,
    ) -> tuple[_ExecutionPayloadCapture, tuple[str, ...]]:
        scope = self._scopes.get(execution_id)
        if scope is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        async with scope.lock:
            if scope.failure is not None and not (
                rejection is scope.failure and self.is_rejection(execution_id, rejection)
            ):
                raise scope.failure
            try:
                pending = tuple(entry for entry in scope.entries if not entry.settled)
                if not pending:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                found = await self._read_entries(scope, execution_id)
                ids = tuple(entry.id for entry in pending)
                captures = tuple(_ExecutionPayloadCapture.decode(found[id]) for id in ids)
                for index, captured in enumerate(captures):
                    if captured.execution.execution_id != execution_id:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if index:
                        previous = captures[index - 1]
                        if (captured.execution.stored_user_input.digest != previous.execution.stored_user_input.digest
                                or captured.execution.binding_digest != previous.execution.binding_digest
                                or captured.execution.principal_id != previous.execution.principal_id
                                or captured.execution.principal_kind != previous.execution.principal_kind
                                or previous.reservation is not None and captured.reservation != previous.reservation
                                or previous.execution.repository_instructions is not None and (
                                    captured.execution.repository_instructions is None
                                    or captured.execution.repository_instructions.payload.digest
                                    != previous.execution.repository_instructions.payload.digest)
                                or previous.execution.input_context is not None and (
                                    captured.execution.input_context is None
                                    or captured.execution.input_context.payload.digest
                                    != previous.execution.input_context.payload.digest)
                                or captured.audit_events[:len(previous.audit_events)] != previous.audit_events):
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                scope.audit_verified = len(captures[-1].audit_events)
                scope.sealed = scope.sealed or seal
                return captures[-1], ids
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                scope.failure = (error if isinstance(error, AIError) and error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                                 else AIError(ErrorCode.STORAGE_UNAVAILABLE,
                                              safe_details={"phase": "execution_buffer_read"}))
                raise scope.failure from error

    def has_scope(self, execution_id: str) -> bool:
        return execution_id in self._scopes

    def settled(self, execution_id: str) -> bool:
        scope = self._scopes.get(execution_id)
        return scope is not None and scope.released and all(entry.settled for entry in scope.entries)

    def is_rejection(self, execution_id: str, error: BaseException) -> bool:
        scope = self._scopes.get(execution_id)
        return (scope is not None and isinstance(error, _ExecutionBufferRejection)
                and error.scope is scope and scope.failure is error)

    async def settle_rejection(self, execution_id: str, error: BaseException) -> bool:
        if not self.is_rejection(execution_id, error):
            return False
        scope = self._scopes[execution_id]
        if scope.rejection_prefix is None:
            scope.rejection_prefix = (await self.read(execution_id, rejection=error)
                                      if scope.entries else (None, ()))
        return True

    def rejection_settled(self, execution_id: str) -> bool:
        scope = self._scopes.get(execution_id)
        return scope is not None and scope.rejection_prefix is not None

    async def source_prefix(self, execution_id: str) -> tuple[_ExecutionPayloadCapture | None, tuple[str, ...]]:
        scope = self._scopes.get(execution_id)
        if scope is not None and scope.rejection_prefix is not None:
            return scope.rejection_prefix
        if (scope is not None and not scope.entries and scope.failure is None
                and not scope.audit_verified and not scope.sealed and not scope.released):
            return None, ()
        return await self.read(execution_id)

    def audit_position(self, execution_id: str) -> int:
        scope = self._scopes.get(execution_id)
        return 0 if scope is None else scope.audit_confirmed

    def confirm_audits(self, execution_id: str, *, start: int, count: int) -> None:
        scope = self._scopes.get(execution_id)
        if (scope is None or scope.audit_confirmed != start or count < 0
                or start + count > scope.audit_verified):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        scope.audit_confirmed += count

    def seal(self, execution_id: str, ids: tuple[str, ...]) -> None:
        """Fence the verified prefix immediately before its owned source dispatch."""
        scope = self._scopes.get(execution_id)
        if scope is None or (scope.failure is not None and scope.rejection_prefix is None) or scope.released:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        pending = tuple(entry for entry in scope.entries if not entry.settled)
        if tuple(entry.id for entry in pending) != ids or any(not entry.admitted for entry in pending):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        scope.sealed = True

    def reject_source(self, execution_id: str, ids: tuple[str, ...]) -> None:
        """Reopen only an exact source batch whose transaction did not commit."""
        scope = self._scopes.get(execution_id)
        if (scope is None or not scope.sealed or scope.released
                or tuple(entry.id for entry in scope.entries if not entry.settled) != ids):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        scope.sealed = False

    def confirm(self, execution_id: str, ids: tuple[str, ...]) -> None:
        """Mark the exact source-covered prefix without waiting for transport locks."""
        scope = self._scopes.get(execution_id)
        if scope is None:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        selected = tuple(entry for entry in scope.entries if entry.id in ids)
        if (len(set(ids)) != len(ids) or set(ids) - scope.confirmed - {entry.id for entry in selected}
                or any(not entry.admitted for entry in selected)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for entry in selected:
            entry.settled = True
        scope.confirmed.update(ids)
        if selected:
            self._schedule_cleanup(execution_id, scope)

    def discard(self, execution_id: str) -> None:
        """Explicitly abandon a stopped attempt, without claiming source durability."""
        scope = self._scopes.get(execution_id)
        if scope is None:
            return
        # A dispatched source batch needs canonical proof, not a worker's
        # exception category, before its only admitted bytes can be released.
        if scope.sealed and any(not entry.settled for entry in scope.entries):
            return
        for entry in scope.entries:
            entry.settled = True
        scope.released = True
        scope.retired = False
        self._schedule_cleanup(execution_id, scope)

    def failed(self, execution_id: str) -> bool:
        scope = self._scopes.get(execution_id)
        return scope is not None and scope.failure is not None

    def _drop_settled_payload(self, scope: _Scope) -> None:
        if not scope.released or scope.entries:
            return
        scope.rejection_prefix = None
        scope.failure = None
        scope.uncertain = None
        self._retired_scopes[:] = [value for value in self._retired_scopes if value is not scope]

    def _schedule_cleanup(self, execution_id: str, scope: _Scope) -> None:
        if scope.cleanup is not None and not scope.cleanup.done():
            return
        task = asyncio.create_task(self._cleanup(execution_id, scope), name="execution-buffer-cleanup")
        scope.cleanup = task
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _cleanup(self, execution_id: str, scope: _Scope) -> None:
        while True:
            async with scope.lock:
                ids = tuple(entry.id for entry in scope.entries if entry.settled)
            if not ids:
                self._drop_settled_payload(scope)
                return
            try:
                async with asyncio.timeout(_BUFFER_TIMEOUT_SECONDS):
                    await self._backend.ack(scope.id, ids)
            except BaseException as error:
                scope.cleanup_failure = error
                return
            async with scope.lock:
                scope.entries[:] = [entry for entry in scope.entries if entry.id not in ids]
                scope.cleanup_failure = None

    async def release(self, execution_id: str) -> None:
        scope = self._scopes.get(execution_id)
        if scope is None:
            return
        async with scope.lock:
            scope.released = True
            scope.retired = False
            self._drop_settled_payload(scope)

    async def close(self) -> None:
        self._accepting = False
        while self._tasks:
            await asyncio.gather(
                *(asyncio.shield(task) for task in tuple(self._tasks)), return_exceptions=True,
            )
        for execution_id, scope in tuple(self._scopes.items()):
            await self._cleanup(execution_id, scope)
        for scope in tuple(self._retired_scopes):
            await self._cleanup("retired", scope)
        unsettled = tuple(scope for scope in (*self._scopes.values(), *self._retired_scopes)
                          if scope.entries or scope.rejection_prefix is not None and not scope.released)
        if unsettled:
            failures = [scope.cleanup_failure or scope.failure for scope in unsettled]
            cause = next((error for error in failures if error is not None), None)
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED,
                          safe_details={"phase": "execution_buffer_close"}) from cause
