#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Atomic shared-budget admission on the execution StateStore."""

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import datetime, timezone
from uuid import uuid4

from ...core import BudgetUsage, RunBudget
from ...errors import AIError, ErrorCode
from ._budget_records import BudgetModelReservation, BudgetToolReservation
from ._durability import CommitObservation, DurableCommitState, run_durable_commit
from ._plan import RuntimeDomain
from ._repository_common import RepositoryBase, replace_checked
from ._store import RecordQuery, StateStore, StateTransaction, StoredRecord


class _UnresolvedLocalReservation(AIError):
    def __init__(self, scope_id: str, storage_version: int, *, inconsistent: bool = False) -> None:
        super().__init__(ErrorCode.STORAGE_CONFLICT)
        self.scope_id = scope_id
        self.storage_version = storage_version
        self.inconsistent = inconsistent


class BudgetRepositoryImpl(RepositoryBase):
    """Own reservations and their transactionally derived usage projection.

    Reservations are retained independently of individual execution cleanup:
    siblings and recovered descendants can still reference their shared scope.
    No effect is refunded on an uncertain or cancelled outcome.
    """

    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        super().__init__(store, namespace=namespace, tenant_id=tenant_id,
                         domain=RuntimeDomain.EXECUTION)
        self._owner_id = uuid4().hex
        self._active_requests: set[tuple[str, str, str]] = set()
        self._background_tasks: set[asyncio.Task[object]] = set()
        self._pending_commits: dict[asyncio.Task[object], int] = {}
        self._commits_drained = asyncio.Event()
        self._commits_drained.set()
        self._closing = False

    async def close(self) -> None:
        self._closing = True
        while self._pending_commits:
            await self._commits_drained.wait()
        self._active_requests.clear()

    def _require_open(self) -> None:
        if self._closing and asyncio.current_task() not in self._pending_commits:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                          safe_details={"reason": "budget_repository_closed"}, retryable=False)

    async def _mutate(
        self, callback: Callable[[StateTransaction], Awaitable[BudgetUsage]],
    ) -> BudgetUsage:
        return await self._retry(lambda: self._store.mutate(callback))

    async def _retry(
        self, operation: Callable[[], Awaitable[BudgetUsage]],
    ) -> BudgetUsage:
        for attempt in range(16):
            try:
                return await operation()
            except _UnresolvedLocalReservation as error:
                # A settled request leaves the live set after its commit. An older
                # transaction can still see its receipt, so validate outside it.
                scope_id = error.scope_id
                row, _ = await self._store.read(
                    lambda transaction: self._scope_record(transaction, scope_id)
                )
                if row.storage_version == error.storage_version:
                    if error.inconsistent:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                    self._reject(error.scope_id, "total_tokens", "unresolved_in_flight")
            except AIError as error:
                if error.code is not ErrorCode.STORAGE_CONFLICT or error.safe_details or attempt == 15:
                    raise
            await asyncio.sleep(min(0.001 * (2 ** attempt), 0.02))
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _commit(
        self,
        operation: Callable[[], Awaitable[BudgetUsage]],
        readback: Callable[[], Awaitable[BudgetUsage | None]],
        *,
        on_cancel: Callable[[], Awaitable[BudgetUsage]] | None = None,
    ) -> BudgetUsage:
        self._require_open()
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("budget mutations require an asyncio task")
        self._pending_commits[task] = self._pending_commits.get(task, 0) + 1
        self._commits_drained.clear()
        try:
            return await self._commit_owned(operation, readback, on_cancel=on_cancel)
        finally:
            remaining = self._pending_commits[task] - 1
            if remaining:
                self._pending_commits[task] = remaining
            else:
                del self._pending_commits[task]
                if not self._pending_commits:
                    self._commits_drained.set()

    async def _commit_owned(
        self,
        operation: Callable[[], Awaitable[BudgetUsage]],
        readback: Callable[[], Awaitable[BudgetUsage | None]],
        *,
        on_cancel: Callable[[], Awaitable[BudgetUsage]] | None,
    ) -> BudgetUsage:
        async def observe() -> CommitObservation[BudgetUsage]:
            value = await readback()
            return CommitObservation(
                DurableCommitState.NOT_COMMITTED if value is None
                else DurableCommitState.COMMITTED,
                value,
            )

        result = await run_durable_commit(
            operation, observe, background_tasks=self._background_tasks,
        )
        if result.cancelled:
            if result.committed and on_cancel is not None:
                await on_cancel()
            raise asyncio.CancelledError()
        if result.committed and result.value is not None:
            return result.value
        if result.error is not None:
            raise result.error
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)

    async def _scope_record(
        self, transaction: StateTransaction, scope_id: str,
    ) -> tuple[StoredRecord, BudgetUsage]:
        row = await transaction.get_record(self._key("budget_scope", scope_id))
        if row is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR,
                          safe_details={"reason": "missing_budget_scope", "scope_id": scope_id})
        value = await self._decode(row, BudgetUsage)
        if value.scope_id != scope_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return row, value

    async def _replace_usage(
        self, transaction: StateTransaction, row: StoredRecord, usage: BudgetUsage,
    ) -> None:
        candidate = self._stored("budget_scope", usage.scope_id, usage)
        await replace_checked(transaction, replace(candidate, storage_version=row.storage_version + 1),
                              row.storage_version)

    async def ensure(self, scope_id: str, limits: RunBudget) -> BudgetUsage:
        async def readback() -> BudgetUsage | None:
            row = await self._record(self._key("budget_scope", scope_id))
            if row is None:
                return None
            value = await self._decode(row, BudgetUsage)
            return value if value.limits == limits else None

        return await self._commit(
            lambda: self._mutate(
                lambda transaction: self.ensure_in_transaction(transaction, scope_id, limits)
            ),
            readback,
        )

    async def ensure_in_transaction(
        self, transaction: StateTransaction, scope_id: str, limits: RunBudget,
    ) -> BudgetUsage:
        row = await transaction.get_record(self._key("budget_scope", scope_id))
        if row is not None:
            value = await self._decode(row, BudgetUsage)
            if value.scope_id != scope_id:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if value.limits != limits:
                raise AIError(ErrorCode.STORAGE_CONFLICT,
                              safe_details={"reason": "budget_limits_changed"})
            return value
        initial = BudgetUsage(scope_id, limits)
        await transaction.insert_record(self._stored("budget_scope", scope_id, initial))
        return initial

    async def read(self, scope_id: str) -> BudgetUsage:
        self._require_open()
        return await self._store.read(
            lambda transaction: self.read_in_transaction(transaction, scope_id)
        )

    async def read_in_transaction(
        self, transaction: StateTransaction, scope_id: str,
    ) -> BudgetUsage:
        return (await self._scope_record(transaction, scope_id))[1]

    async def check(self, scope_id: str) -> BudgetUsage:
        """Gate uncounted graph work without reserving a model or tool call."""
        self._require_open()
        async def read(transaction: StateTransaction) -> BudgetUsage:
            row, usage = await self._scope_record(transaction, scope_id)
            await self._check_admission(transaction, row, usage, None)
            return usage
        return await self._retry(lambda: self._store.read(read))

    def _reject(self, scope_id: str, dimension: str, reason: str) -> None:
        raise AIError(ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED, safe_details={
            "scope_id": scope_id, "dimension": dimension, "reason": reason,
        })

    async def _check_admission(
        self, transaction: StateTransaction, scope_row: StoredRecord,
        usage: BudgetUsage, dimension: str | None,
    ) -> None:
        limits = usage.limits
        if limits.deadline_at is not None and datetime.now(timezone.utc) >= limits.deadline_at:
            self._reject(usage.scope_id, "deadline_at", "deadline_exceeded")
        if dimension is not None:
            limit = limits.model_requests if dimension == "model_requests" else limits.tool_calls
            count = usage.model_requests if dimension == "model_requests" else usage.tool_calls
            if limit is not None and count >= limit:
                self._reject(usage.scope_id, dimension, "limit_exceeded")
        if limits.total_tokens is None:
            return
        if usage.unknown_model_requests:
            self._reject(usage.scope_id, "total_tokens", "unknown_usage")
        if usage.in_flight_model_requests:
            rows = await transaction.list_records(RecordQuery(
                kind="budget_model", scope_digest=self._scope("budget_model", "budget", usage.scope_id),
                states=frozenset({"in_flight"}),
            ))
            if len(rows) != usage.in_flight_model_requests:
                raise _UnresolvedLocalReservation(
                    usage.scope_id, scope_row.storage_version, inconsistent=True,
                )
            for row in rows:
                reservation = await self._decode(row, BudgetModelReservation)
                if reservation.scope_id != usage.scope_id or reservation.status != "in_flight":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if reservation.owner_id != self._owner_id:
                    self._reject(usage.scope_id, "total_tokens", "unresolved_in_flight")
                if (
                    reservation.scope_id, reservation.request_id, reservation.admission_id
                ) not in self._active_requests:
                    raise _UnresolvedLocalReservation(usage.scope_id, scope_row.storage_version)
        if usage.total_tokens >= limits.total_tokens:
            self._reject(usage.scope_id, "total_tokens", "limit_exceeded")

    async def admit_model(self, scope_id: str, request_id: str) -> BudgetUsage:
        reservation = BudgetModelReservation(scope_id, request_id, self._owner_id, uuid4().hex)
        identity = [scope_id, request_id]
        active = (scope_id, request_id, reservation.admission_id)

        async def write(transaction: StateTransaction) -> BudgetUsage:
            row, usage = await self._scope_record(transaction, scope_id)
            if await transaction.get_record(self._key("budget_model", identity)) is not None:
                self._reject(scope_id, "model_requests", "duplicate_dispatch")
            await self._check_admission(transaction, row, usage, "model_requests")
            updated = replace(usage, model_requests=usage.model_requests + 1,
                              in_flight_model_requests=usage.in_flight_model_requests + 1)
            await transaction.insert_record(self._stored(
                "budget_model", identity, reservation,
                scope=self._scope("budget_model", "budget", scope_id), state="in_flight",
            ))
            await self._replace_usage(transaction, row, updated)
            return updated

        async def readback() -> BudgetUsage | None:
            async def read(transaction: StateTransaction) -> BudgetUsage | None:
                row = await transaction.get_record(self._key("budget_model", identity))
                if row is None:
                    return None
                value = await self._decode(row, BudgetModelReservation)
                if value.admission_id != reservation.admission_id:
                    return None
                return (await self._scope_record(transaction, scope_id))[1]
            return await self._store.read(read)

        self._active_requests.add(active)
        try:
            return await self._commit(
                lambda: self._mutate(write), readback,
                on_cancel=lambda: self.settle_model(scope_id, request_id, None),
            )
        except BaseException:
            self._active_requests.discard(active)
            raise

    async def settle_model(
        self, scope_id: str, request_id: str, total_tokens: int | None,
    ) -> BudgetUsage:
        if total_tokens is not None and (
            isinstance(total_tokens, bool) or not isinstance(total_tokens, int) or total_tokens < 0
        ):
            raise ValueError("total_tokens must be a non-negative integer or None")
        identity = [scope_id, request_id]
        status = "unknown" if total_tokens is None else "settled"

        async def write(transaction: StateTransaction) -> BudgetUsage:
            scope_row, usage = await self._scope_record(transaction, scope_id)
            row = await transaction.get_record(self._key("budget_model", identity))
            if row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            reservation = await self._decode(row, BudgetModelReservation)
            if (reservation.scope_id, reservation.request_id) != (scope_id, request_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if reservation.status != "in_flight":
                if reservation.status != status or reservation.total_tokens != total_tokens:
                    raise AIError(ErrorCode.STORAGE_CONFLICT,
                                  safe_details={"reason": "budget_settlement_changed"})
                return usage
            updated = replace(
                usage, total_tokens=usage.total_tokens + (0 if total_tokens is None else total_tokens),
                in_flight_model_requests=usage.in_flight_model_requests - 1,
                unknown_model_requests=usage.unknown_model_requests + int(total_tokens is None),
            )
            candidate = self._stored(
                "budget_model", identity, replace(reservation, status=status, total_tokens=total_tokens),
                scope=self._scope("budget_model", "budget", scope_id), state=status,
            )
            await replace_checked(transaction, replace(candidate, storage_version=row.storage_version + 1),
                                  row.storage_version)
            await self._replace_usage(transaction, scope_row, updated)
            return updated

        async def readback() -> BudgetUsage | None:
            async def read(transaction: StateTransaction) -> BudgetUsage | None:
                row = await transaction.get_record(self._key("budget_model", identity))
                if row is None:
                    return None
                value = await self._decode(row, BudgetModelReservation)
                if value.status != status or value.total_tokens != total_tokens:
                    return None
                return (await self._scope_record(transaction, scope_id))[1]
            return await self._store.read(read)

        try:
            return await self._commit(lambda: self._mutate(write), readback)
        finally:
            self._active_requests.difference_update(
                value for value in tuple(self._active_requests)
                if value[:2] == (scope_id, request_id)
            )

    async def admit_tool(self, scope_id: str, call_id: str) -> BudgetUsage:
        reservation = BudgetToolReservation(scope_id, call_id, uuid4().hex)
        identity = [scope_id, call_id]

        async def write(transaction: StateTransaction) -> BudgetUsage:
            row, usage = await self._scope_record(transaction, scope_id)
            if await transaction.get_record(self._key("budget_tool", identity)) is not None:
                self._reject(scope_id, "tool_calls", "duplicate_dispatch")
            await self._check_admission(transaction, row, usage, "tool_calls")
            updated = replace(usage, tool_calls=usage.tool_calls + 1)
            await transaction.insert_record(self._stored("budget_tool", identity, reservation))
            await self._replace_usage(transaction, row, updated)
            return updated

        async def readback() -> BudgetUsage | None:
            async def read(transaction: StateTransaction) -> BudgetUsage | None:
                row = await transaction.get_record(self._key("budget_tool", identity))
                if row is None:
                    return None
                value = await self._decode(row, BudgetToolReservation)
                if value.admission_id != reservation.admission_id:
                    return None
                return (await self._scope_record(transaction, scope_id))[1]
            return await self._store.read(read)

        return await self._commit(lambda: self._mutate(write), readback)


__all__ = ["BudgetRepositoryImpl"]
