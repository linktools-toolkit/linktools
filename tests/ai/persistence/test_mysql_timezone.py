#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in real MySQL clock, lease and native-checkpoint round trips."""

import os
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from typing import Iterator

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence.runs import NativeCheckpoint, OfficialStepReader
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import RecordLeaseTransition
from tests.ai.test_state_batch_delete_backends import _record


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("LINKTOOLS_TEST_MYSQL_TIMEZONE") != "1",
                       reason="opt in to the disposable local MySQL timezone regression"),
    pytest.mark.skipif(not hasattr(time, "tzset"), reason="requires process TZ switching"),
]


@contextmanager
def _process_timezone(zone: str) -> Iterator[None]:
    previous = os.environ.get("TZ")
    os.environ["TZ"] = zone
    time.tzset()
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = previous
        time.tzset()


def _engine(url: URL, zone: str) -> AsyncEngine:
    # Session-local and applied on every connection, including reconnects.
    # +08:00 is modern Asia/Shanghai without requiring server timezone tables.
    assert zone in {"Asia/Shanghai", "+08:00", "+00:00"}
    return create_async_engine(url, pool_size=1, max_overflow=0,
                               connect_args={"init_command": f"SET time_zone = '{zone}'"})


async def _utc_now(engine: AsyncEngine, expected_zone: str) -> datetime:
    async with engine.connect() as connection:
        zone, utc, local = (await connection.execute(text(
            "SELECT @@session.time_zone, UTC_TIMESTAMP(6), CURRENT_TIMESTAMP(6)"
        ))).one()
    assert zone == expected_zone
    expected_offset = timedelta(hours=8) if zone in {"Asia/Shanghai", "+08:00"} else timedelta()
    assert local - utc == expected_offset
    return utc.replace(tzinfo=timezone.utc)


async def _physical_record(engine: AsyncEngine, store_key: bytes, key: bytes) -> tuple[datetime, datetime, datetime]:
    async with engine.connect() as connection:
        return tuple((await connection.execute(text(
            "SELECT created_at, updated_at, lease_expires_at FROM ai_state_records "
            "WHERE store_key=:store_key AND `key`=:key"
        ), {"store_key": store_key, "key": key})).one())


@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
@pytest.mark.parametrize("write_process,write_zone,read_process,read_zone", (
    ("Asia/Shanghai", "+08:00", "UTC", "+00:00"),
    ("UTC", "+00:00", "Asia/Shanghai", "+08:00"),
    ("Asia/Shanghai", "Asia/Shanghai", "UTC", "+00:00"),
    ("UTC", "+00:00", "Asia/Shanghai", "Asia/Shanghai"),
), ids=("offset-shanghai-to-utc", "offset-utc-to-shanghai",
        "named-shanghai-to-utc", "named-utc-to-shanghai"))
async def test_mysql_real_clock_lease_checkpoint_reopen_across_timezones(
    _server: tuple[str, list[str]], write_process: str, write_zone: str,
    read_process: str, read_zone: str,
) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    # The existing harness owns a disposable database, not an application DSN.
    writer = _engine(url, write_zone)
    reader = _engine(url, read_zone)
    named = "Asia/Shanghai" in (write_zone, read_zone)
    identity = "timezone-" + ("named-" if named else "offset-") + write_process.replace("/", "-")
    record = _record((810 if write_process == "Asia/Shanghai" else 811) + (10 if named else 0))
    state = SqlStateStore(writer)
    steps = SqlStepStore(writer, namespace=identity, tenant_id="clock-test")
    reopened = None
    reopened_steps = None
    try:
        try:
            await _utc_now(writer, write_zone)
            await _utc_now(reader, read_zone)
        except DBAPIError as error:
            if named and error.orig.args and error.orig.args[0] == 1298:
                pytest.skip("MySQL Asia/Shanghai named timezone is not installed; offset cases are separate")
            raise
        with _process_timezone(write_process):
            assert datetime.now().astimezone().utcoffset() == (
                timedelta(hours=8) if write_process == "Asia/Shanghai" else timedelta())
            await provision_database(writer)
            await state.initialize()
            before = await _utc_now(writer, write_zone)
            await state.mutate(lambda transaction: transaction.insert_record(record))
            acquired = await state.mutate(lambda transaction: transaction.transition_record_lease(
                record.key, expected_storage_revision=0,
                transition=RecordLeaseTransition("acquire", "writer", 0, 600),
            ))
            after = await _utc_now(writer, write_zone)
            assert acquired is not None and acquired.lease_fence == 1
            assert acquired.lease_expires_at is not None
            assert before + timedelta(seconds=600) <= acquired.lease_expires_at <= after + timedelta(seconds=600)
            clock = await state.read(lambda transaction: transaction.now())
            assert before <= clock <= await _utc_now(writer, write_zone)
            physical = await _physical_record(writer, state.store_key, record.key)
            # Existing audit DATETIME columns have whole-second precision.
            assert before - timedelta(seconds=1) <= physical[0].replace(tzinfo=timezone.utc) <= after + timedelta(seconds=1)
            assert before - timedelta(seconds=1) <= physical[1].replace(tzinfo=timezone.utc) <= after + timedelta(seconds=1)
            assert physical[0] <= physical[1]
            assert physical[2].replace(tzinfo=timezone.utc) == acquired.lease_expires_at
            snapshot = ContinuableSnapshot(run_id=identity, step_index=1,
                idempotency_key="checkpoint-1", messages=[ModelRequest(parts=[UserPromptPart("clock probe")])])
            await steps.register_run(RunRecord(run_id=identity))
            await steps.save_snapshot(NativeCheckpoint.encode(snapshot, output_retries_used=1))
            committed_snapshot = await OfficialStepReader(steps).latest_snapshot(run_id=identity)
            assert committed_snapshot is not None
            await steps.close()
            await state.close()

        with _process_timezone(read_process):
            await _utc_now(reader, read_zone)
            reopened = SqlStateStore(reader)
            reopened_steps = SqlStepStore(reader, namespace=identity, tenant_id="clock-test")
            await reopened.initialize()
            assert await reopened.read_record(record.key) == acquired
            assert await _physical_record(reader, reopened.store_key, record.key) == physical
            restored = await OfficialStepReader(reopened_steps).latest_snapshot(run_id=identity)
            assert restored is not None
            assert restored == committed_snapshot
            assert restored.messages == snapshot.messages
            assert restored.output_retries_used == 1
            # A different session timezone must not make a live lease available.
            assert await reopened.mutate(lambda transaction: transaction.transition_record_lease(
                record.key, expected_storage_revision=acquired.storage_revision,
                transition=RecordLeaseTransition("acquire", "contender", 1, 600),
            )) is None
            assert await reopened.mutate(lambda transaction: transaction.transition_record_lease(
                record.key, expected_storage_revision=acquired.storage_revision,
                transition=RecordLeaseTransition("renew", "writer", 0, 1200),
            )) is None
            renewed_before = await _utc_now(reader, read_zone)
            renewed = await reopened.mutate(lambda transaction: transaction.transition_record_lease(
                record.key, expected_storage_revision=acquired.storage_revision,
                transition=RecordLeaseTransition("renew", "writer", 1, 1200),
            ))
            renewed_after = await _utc_now(reader, read_zone)
            assert renewed is not None and renewed.lease_fence == 1
            assert renewed.lease_expires_at is not None
            assert renewed_before + timedelta(seconds=1200) <= renewed.lease_expires_at <= renewed_after + timedelta(seconds=1200)
            renewed_physical = await _physical_record(reader, reopened.store_key, record.key)
            assert renewed_physical[0] == physical[0]
            assert renewed_before - timedelta(seconds=1) <= renewed_physical[1].replace(tzinfo=timezone.utc) <= renewed_after + timedelta(seconds=1)
            assert renewed_physical[2].replace(tzinfo=timezone.utc) == renewed.lease_expires_at
    finally:
        if reopened_steps is not None:
            await reopened_steps.close()
        if reopened is not None:
            await reopened.close()
        await steps.close()
        await state.close()
        await reader.dispose()
        await writer.dispose()
