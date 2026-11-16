#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Opt-in real MySQL clock, lease and native-checkpoint round trips."""

import os
import shutil
import subprocess
import time
from contextlib import contextmanager
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator
from zoneinfo import TZPATH

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord
from sqlalchemy import text
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence import SqlStepStore
from linktools.ai.persistence.runs import NativeCheckpoint, OfficialStepReader
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import RecordLeaseTransition
from tests.ai.test_state_batch_delete_backends import _record
from tests.ai._metrics_server_test_helpers import _server_database, _server_process


pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(os.environ.get("LINKTOOLS_TEST_MYSQL_TIMEZONE") != "1",
                       reason="opt in to the disposable local MySQL timezone regression"),
    pytest.mark.skipif(not hasattr(time, "tzset"), reason="requires process TZ switching"),
]


@pytest.fixture(scope="module")
def _timezone_server() -> Iterator[tuple[str, list[str]]]:
    # A separate process: never populate timezone tables on the shared fixture.
    if os.environ.get("LINKTOOLS_TEST_MYSQL_TIMEZONE") != "1":
        pytest.skip("opt in to the disposable local MySQL timezone regression")
    loader = shutil.which("mysql_tzinfo_to_sql")
    zonefile = next((Path(root) / "Asia" / "Shanghai" for root in TZPATH
                     if (Path(root) / "Asia" / "Shanghai").is_file()), None)
    if loader is None or zonefile is None:
        pytest.skip("named timezone regression requires mysql_tzinfo_to_sql and system Asia/Shanghai tzdata")
    sql = subprocess.run([loader, str(zonefile), "Asia/Shanghai"], check=True,
                         capture_output=True, text=True, timeout=10).stdout
    assert sql.strip(), "mysql_tzinfo_to_sql returned no Asia/Shanghai definitions"
    with contextmanager(_server_process)("mysql") as server:
        _, command = server
        # command targets mysql on this newly initialized disposable server.
        subprocess.run(command, input=sql, check=True,
                       capture_output=True, text=True, timeout=10)
        with _server_database(server) as database:
            yield database


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


@pytest.mark.parametrize("write_process,write_zone,read_process,read_zone", (
    ("Asia/Shanghai", "+08:00", "UTC", "+00:00"),
    ("UTC", "+00:00", "Asia/Shanghai", "+08:00"),
    ("Asia/Shanghai", "Asia/Shanghai", "UTC", "+00:00"),
    ("UTC", "+00:00", "Asia/Shanghai", "Asia/Shanghai"),
), ids=("offset-shanghai-to-utc", "offset-utc-to-shanghai",
        "named-shanghai-to-utc", "named-utc-to-shanghai"))
async def test_mysql_real_clock_lease_checkpoint_reopen_across_timezones(
    _timezone_server: tuple[str, list[str]], write_process: str, write_zone: str,
    read_process: str, read_zone: str,
) -> None:
    _, command = _timezone_server
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
        # A supported fixture must exercise named sessions, not silently skip 1298.
        await _utc_now(writer, write_zone)
        await _utc_now(reader, read_zone)
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

            first_counter = f"{identity}-first".encode()
            second_counter = f"{identity}-second".encode()
            assert await state.mutate(lambda tx: tx.reserve_sequence(first_counter, 2)) == 2
            async with writer.connect() as connection:
                first_created = await connection.scalar(text(
                    "SELECT created_at FROM ai_state_counters WHERE store_key=:store AND stream_key=:key"
                ), {"store": state.store_key, "key": first_counter})
            assert await state.mutate(lambda tx: tx.reserve_sequence(first_counter, 1)) == 3
            single_done = await _utc_now(writer, write_zone)
            async with writer.connect() as connection:
                single_created, single_updated = (await connection.execute(text(
                    "SELECT created_at, updated_at FROM ai_state_counters WHERE store_key=:store AND stream_key=:key"
                ), {"store": state.store_key, "key": first_counter})).one()
            assert single_created == first_created
            assert before - timedelta(seconds=1) <= single_updated.replace(tzinfo=timezone.utc) <= single_done + timedelta(seconds=1)
            batch = {first_counter: 3, second_counter: 4}
            assert await state.mutate(lambda tx: tx.reserve_sequences(batch)) == {
                first_counter: 6, second_counter: 4,
            }
            counters_done = await _utc_now(writer, write_zone)
            async with writer.connect() as connection:
                counter_rows = (await connection.execute(text(
                    "SELECT stream_key, value, created_at, updated_at FROM ai_state_counters "
                    "WHERE store_key=:store AND stream_key IN (:first, :second)"
                ), {"store": state.store_key, "first": first_counter, "second": second_counter})).all()
            assert {bytes(key): value for key, value, _, _ in counter_rows} == {
                first_counter: 6, second_counter: 4,
            }
            for key, _, created, updated in counter_rows:
                if bytes(key) == first_counter:
                    assert created == first_created
                assert before - timedelta(seconds=1) <= created.replace(tzinfo=timezone.utc)
                assert created <= updated
                assert updated.replace(tzinfo=timezone.utc) <= counters_done + timedelta(seconds=1)
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
            # Seed an already expired lease without sleeping or altering server time.
            expired = replace(_record(910 + (10 if named else 0)
                                      + (0 if write_process == "Asia/Shanghai" else 1)),
                              lease_owner="expired", lease_fence=7,
                              lease_expires_at=renewed_before - timedelta(seconds=60))
            await reopened.mutate(lambda transaction: transaction.insert_record(expired))
            for operation, seconds in (("renew", 60), ("release", None)):
                assert await reopened.mutate(lambda transaction: transaction.transition_record_lease(
                    expired.key, expected_storage_revision=expired.storage_revision,
                    transition=RecordLeaseTransition(operation, "expired", 7, seconds),
                )) is None
            taken = await reopened.mutate(lambda transaction: transaction.transition_record_lease(
                expired.key, expected_storage_revision=expired.storage_revision,
                transition=RecordLeaseTransition("acquire", "successor", 7, 60),
            ))
            assert taken is not None and taken.lease_fence == 8
            assert taken.lease_owner == "successor"
    finally:
        if reopened_steps is not None:
            await reopened_steps.close()
        if reopened is not None:
            await reopened.close()
        await steps.close()
        await state.close()
        await reader.dispose()
        await writer.dispose()
