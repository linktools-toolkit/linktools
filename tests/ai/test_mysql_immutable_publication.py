#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL immutable comparison must reject conflicts in non-STRICT mode."""

import hashlib

import pytest
from sqlalchemy import event
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence._sql import SqlStateStore

from .test_immutable_record_publication import _assert_exact_publication, _assert_rejected_batch


@pytest.mark.asyncio
@pytest.mark.parametrize("_server", ("mysql",), indirect=True)
async def test_mysql_nonstrict_immutable_rows_preserve_bytes_and_poison_mixed_batches(_server) -> None:
    _, command = _server
    socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
    url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                     query={"unix_socket": socket, "charset": "utf8mb4"})
    engine = create_async_engine(url, hide_parameters=True)

    def nonstrict(connection, record) -> None:
        cursor = connection.cursor()
        cursor.execute("SET SESSION sql_mode = ''")
        cursor.close()

    event.listen(engine.sync_engine, "connect", nonstrict)
    await provision_database(engine)
    try:
        for difference in ("exact", "float", "integer", "boolean", "trailing-space", "case"):
            store = SqlStateStore(engine, store_key=hashlib.sha256(difference.encode()).digest())
            await store.initialize()
            try:
                if difference == "exact":
                    await _assert_exact_publication(store)
                else:
                    await _assert_rejected_batch(store, difference=difference)
            finally:
                await store.close()
    finally:
        await engine.dispose()
