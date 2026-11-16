#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in MySQL cancellation and unknown-commit readback in non-UTC sessions."""

import asyncio
from contextlib import asynccontextmanager
from dataclasses import replace
from datetime import timedelta, timezone
from typing import AsyncIterator

import pytest
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncSessionTransaction

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import RecordLeaseTransition, StoredRecord
from linktools.ai.storage import SqlTransactionDisposition, SqlTransactionPhase
from tests.ai.persistence.test_mysql_timezone import (
    _engine, _physical_record, _process_timezone, _timezone_server, _utc_now,
    pytestmark as _timezone_marks,
)
from tests.ai.test_state_batch_delete_backends import _record

pytestmark = _timezone_marks


@asynccontextmanager
async def _leased_mysql(
    server: tuple[str, list[str]], record_id: int,
) -> AsyncIterator[tuple[SqlStateStore, SqlStateStore, StoredRecord]]:
    _, command = server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    writer_engine = _engine(url, "Asia/Shanghai")
    reader_engine = _engine(url, "+00:00")
    writer = SqlStateStore(writer_engine)
    reader = SqlStateStore(reader_engine)
    try:
        await _utc_now(writer_engine, "Asia/Shanghai")
        await _utc_now(reader_engine, "+00:00")
        await provision_database(writer_engine)
        await writer.initialize()
        await reader.initialize()
        record = _record(record_id)
        await writer.mutate(lambda tx: tx.insert_record(record))
        acquired = await writer.mutate(lambda tx: tx.transition_record_lease(
            record.key, expected_storage_revision=record.storage_revision,
            transition=RecordLeaseTransition("acquire", "owner", 0, 600),
        ))
        assert acquired is not None
        assert await reader.read(lambda tx: tx.get_record(record.key)) == acquired
        yield writer, reader, acquired
    finally:
        await reader.close()
        await writer.close()
        await reader_engine.dispose()
        await writer_engine.dispose()


async def test_mysql_non_utc_cancelled_guard_cannot_commit_caught_cancellation(
    _timezone_server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _process_timezone("Asia/Shanghai"):
        async with _leased_mysql(_timezone_server, 940) as (writer, reader, record):
            entered, release = asyncio.Event(), asyncio.Event()
            following = _record(941)
            clock = writer.context.dialect.database_current_time
            reads = 0

            async def delayed_clock(session):
                nonlocal reads
                sampled = await clock(session)
                reads += 1
                if reads == 2:
                    entered.set()
                    await release.wait()
                return sampled

            async def callback(tx):
                try:
                    await tx.transition_record_lease(
                        record.key, expected_storage_revision=record.storage_revision,
                        transition=RecordLeaseTransition("guard", "owner", record.lease_fence),
                    )
                except asyncio.CancelledError:
                    await tx.insert_record(following)

            before = await _physical_record(reader.context.engine, reader.store_key, record.key)
            with monkeypatch.context() as patch:
                patch.setattr(writer.context.dialect, "database_current_time", delayed_clock)
                worker = asyncio.create_task(writer.mutate(callback))
                try:
                    await asyncio.wait_for(entered.wait(), 5)
                    worker.cancel()
                    await asyncio.sleep(0)
                    release.set()
                    with pytest.raises(asyncio.CancelledError):
                        await asyncio.wait_for(worker, 5)
                finally:
                    release.set()
                    if not worker.done():
                        worker.cancel()
                    await asyncio.gather(worker, return_exceptions=True)
            assert await reader.read(lambda tx: tx.get_record(record.key)) == record
            assert await reader.read(lambda tx: tx.get_record(following.key)) is None
            assert await _physical_record(reader.context.engine, reader.store_key, record.key) == before


async def test_mysql_non_utc_lost_commit_response_reads_original_identity_without_replay(
    _timezone_server: tuple[str, list[str]], monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _process_timezone("Asia/Shanghai"):
        async with _leased_mysql(_timezone_server, 942) as (writer, reader, record):
            failure = OSError("lost commit response")
            exit_transaction = AsyncSessionTransaction.__aexit__
            classify = writer.context.dialect.classify_transaction_error
            attempted = 0
            injected = False

            async def commit_then_lose_response(transaction, kind, error, traceback):
                nonlocal injected
                result = await exit_transaction(transaction, kind, error, traceback)
                if kind is None and not injected:
                    injected = True
                    raise failure
                return result

            def disposition(error, *, phase, connection_invalidated):
                if error is failure and phase is SqlTransactionPhase.COMMIT:
                    return SqlTransactionDisposition.COMMIT_UNKNOWN
                return classify(error, phase=phase, connection_invalidated=connection_invalidated)

            async def callback(tx):
                nonlocal attempted
                attempted += 1
                return await tx.transition_record_lease(
                    record.key, expected_storage_revision=record.storage_revision,
                    transition=RecordLeaseTransition("guard", "owner", record.lease_fence),
                )

            before = await _physical_record(reader.context.engine, reader.store_key, record.key)
            commit_lower = await _utc_now(reader.context.engine, "+00:00")
            with monkeypatch.context() as patch:
                patch.setattr(AsyncSessionTransaction, "__aexit__", commit_then_lose_response)
                patch.setattr(writer.context.dialect, "classify_transaction_error", disposition)
                with pytest.raises(AIError) as caught:
                    await writer.mutate(callback)
            assert caught.value.code is ErrorCode.STORAGE_COMMIT_UNKNOWN
            assert injected and attempted == 1
            assert await reader.read(lambda tx: tx.get_record(record.key)) == replace(
                record, storage_revision=record.storage_revision + 1,
            )
            commit_upper = await _utc_now(reader.context.engine, "+00:00")
            physical = await _physical_record(reader.context.engine, reader.store_key, record.key)
            assert physical[0] == before[0]
            assert physical[2] == before[2]
            # DATETIME audit columns have whole-second precision.
            assert (commit_lower - timedelta(seconds=1)
                    <= physical[1].replace(tzinfo=timezone.utc)
                    <= commit_upper + timedelta(seconds=1))
