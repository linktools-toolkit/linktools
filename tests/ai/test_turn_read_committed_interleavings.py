#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable release/read interleavings with no service configuration or credentials."""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, TypeVar

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable, Mapping, Sequence
    from datetime import datetime

    from linktools.ai.persistence.state import (
        StateStore,
        StateTransaction,
        StoredRecord,
    )
    from linktools.ai.storage import Lease, TurnCoordination
import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.turns import StateTurnCoordination
from linktools.ai.storage import TurnScopeIdentity

_Value = TypeVar("_Value")
_SCOPE = TurnScopeIdentity("turn-reader-fixture", "tenant", "scope")


class _StatementReads:
    def __init__(self, store: StateStore) -> None:
        self.store = store
        self.anchor_kind: str | None = None
        self.after_anchor: Callable[[], Awaitable[None]] | None = None
        self.after_batch: Callable[[], Awaitable[None]] | None = None
        self.corrupt: str | None = None
        self.after_error: Callable[[], Awaitable[None]] | None = None
        self.advances = 0

    async def read(
        self, callback: Callable[[StateTransaction], Awaitable[_Value]]
    ) -> _Value:
        class Reader:
            async def get_record(inner, key: bytes) -> StoredRecord | None:
                row = await self.store.read(lambda tx: tx.get_record(key))
                if (
                    row is not None
                    and row.kind == self.anchor_kind
                    and row.state == "ACTIVE"
                    and self.after_anchor is not None
                ):
                    advance, self.after_anchor = self.after_anchor, None
                    await advance()
                return row

            async def get_records(
                inner, keys: Sequence[bytes]
            ) -> Mapping[bytes, StoredRecord]:
                rows = await self.store.read(lambda tx: tx.get_records(keys))
                if self.after_batch is not None:
                    advance, self.after_batch = self.after_batch, None
                    await advance()
                if self.corrupt == "missing":
                    return {
                        key: row
                        for key, row in rows.items()
                        if row.kind != "turn_request"
                    }
                if self.corrupt == "identity":
                    return {
                        key: replace(row, data={**row.data, "request_digest": "wrong"})
                        if row.kind == "turn_request"
                        else row
                        for key, row in rows.items()
                    }
                return rows

            async def now(inner) -> datetime:
                return await self.store.read(lambda tx: tx.now())

        try:
            return await callback(Reader())
        except AIError as error:
            if (
                error.code is ErrorCode.STORAGE_INTEGRITY_ERROR
                and self.after_error is not None
            ):
                self.advances += 1
                await self.after_error()
            raise

    async def mutate(
        self, callback: Callable[[StateTransaction], Awaitable[_Value]]
    ) -> _Value:
        if self.after_anchor is not None or self.after_batch is not None:
            # A mixed graph must be rejected before any write is attempted.
            return await self.read(callback)
        return await self.store.mutate(callback)


async def _claim(writer: StateTurnCoordination) -> Lease:
    result = await writer.claim_turn(
        _SCOPE,
        session_key="session",
        request_key="request",
        request_digest="digest",
        execution_id="execution",
        owner_id="owner",
        lease_seconds=120,
    )
    assert result is not None and result.lease is not None
    return result.lease


@pytest.mark.asyncio
async def test_release_after_initial_admission_read_allows_new_claim() -> None:
    async with Persistence.open(
        "turn-reader-fixture", tenant_id="tenant", storage=PersistenceStorage.in_memory()
    ) as persistence:
        store = persistence.execution.executions.state_store
        writer = StateTurnCoordination(
            store, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        lease = await _claim(writer)
        sampled = _StatementReads(store)

        async def release() -> None:
            assert await writer.release_turn(lease)

        sampled.after_batch = release
        contender = StateTurnCoordination(
            sampled, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        replacement = await contender.claim_turn(
            _SCOPE,
            session_key="session",
            request_key="replacement-request",
            request_digest="replacement-digest",
            execution_id="replacement-execution",
            owner_id="replacement-owner",
            lease_seconds=120,
        )
        assert replacement is not None and replacement.lease is not None
        assert await writer.read_turn(_SCOPE, "execution") is None
        assert await writer.release_turn(replacement.lease)


async def _observe(reader: StateTurnCoordination, kind: str) -> TurnCoordination | None:
    if kind == "turn_session":
        return await reader.read_turn_by_session(_SCOPE, "session")
    if kind == "turn_request":
        return await reader.read_turn_by_request(_SCOPE, "request")
    return await reader.read_turn(_SCOPE, "execution")


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ["turn_session", "turn_request", "turn_execution"])
async def test_release_between_anchor_and_peers_is_observed_as_idle(kind: str) -> None:
    async with Persistence.open(
        "turn-reader-fixture", tenant_id="tenant", storage=PersistenceStorage.in_memory()
    ) as persistence:
        store = persistence.execution.executions.state_store
        writer = StateTurnCoordination(
            store, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        lease = await _claim(writer)
        sampled = _StatementReads(store)
        sampled.anchor_kind = kind

        async def release() -> None:
            assert await writer.release_turn(lease)

        sampled.after_anchor = release
        reader = StateTurnCoordination(
            sampled, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        assert await _observe(reader, kind) is None
        assert await writer.read_turn_by_session(_SCOPE, "session") is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "kind,corrupt",
    [
        ("turn_session", "missing"),
        ("turn_execution", "missing"),
        ("turn_execution", "identity"),
    ],
)
async def test_unchanged_anchor_keeps_real_integrity_errors(
    kind: str, corrupt: str
) -> None:
    async with Persistence.open(
        "turn-reader-fixture", tenant_id="tenant", storage=PersistenceStorage.in_memory()
    ) as persistence:
        store = persistence.execution.executions.state_store
        writer = StateTurnCoordination(
            store, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        lease = await _claim(writer)
        sampled = _StatementReads(store)
        sampled.corrupt = corrupt
        reader = StateTurnCoordination(
            sampled, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        with pytest.raises(AIError) as failure:
            await _observe(reader, kind)
        assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert await writer.read_turn_by_session(_SCOPE, "session") is not None
        assert await writer.release_turn(lease)


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["renew", "cancel", "release"])
async def test_release_before_mutation_validation_preserves_released_authority(
    operation: str,
) -> None:
    async with Persistence.open(
        "turn-reader-fixture", tenant_id="tenant", storage=PersistenceStorage.in_memory()
    ) as persistence:
        store = persistence.execution.executions.state_store
        writer = StateTurnCoordination(
            store, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        lease = await _claim(writer)
        sampled = _StatementReads(store)
        sampled.anchor_kind = "turn_execution"

        async def release() -> None:
            assert await writer.release_turn(lease)

        sampled.after_anchor = release
        contender = StateTurnCoordination(
            sampled, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        if operation == "renew":
            assert await contender.renew_turn(lease, lease_seconds=120) is None
        elif operation == "cancel":
            assert await contender.cancel_turn(_SCOPE, "execution") is False
        else:
            assert await contender.release_turn(lease) is False
        assert await writer.read_turn_by_session(_SCOPE, "session") is None
        replacement = await writer.claim_turn(
            _SCOPE,
            session_key="session",
            request_key="replacement-request",
            request_digest="replacement-digest",
            execution_id="replacement-execution",
            owner_id="replacement-owner",
            lease_seconds=120,
        )
        assert replacement is not None and replacement.lease is not None
        assert await writer.release_turn(replacement.lease)


@pytest.mark.asyncio
async def test_continuously_advancing_anchor_exhausts_bounded_conflict_retry() -> None:
    async with Persistence.open(
        "turn-reader-fixture", tenant_id="tenant", storage=PersistenceStorage.in_memory()
    ) as persistence:
        store = persistence.execution.executions.state_store
        writer = StateTurnCoordination(
            store, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        lease = await _claim(writer)
        sampled = _StatementReads(store)
        sampled.corrupt = "missing"

        async def advance() -> None:
            assert await writer.renew_turn(lease, lease_seconds=120) is not None

        sampled.after_error = advance
        reader = StateTurnCoordination(
            sampled, namespace=persistence.namespace, tenant_id=persistence.tenant_id
        )
        with pytest.raises(AIError) as failure:
            await reader.read_turn_by_session(_SCOPE, "session")
        assert failure.value.code is ErrorCode.STORAGE_CONFLICT
        assert 1 <= sampled.advances <= 3
        assert await writer.release_turn(lease)
