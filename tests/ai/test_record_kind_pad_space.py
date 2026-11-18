#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MySQL kind comparisons must not hide records in dedicated owners."""

from dataclasses import replace

import pytest
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.migrate import provision_database
from linktools.ai.persistence._sql import SqlStateStore
from linktools.ai.persistence.state import RecordQuery, record_key

from ._metrics_server_test_helpers import _mysql_server, _server_database
from .test_state_batch_delete_backends import _record


@pytest.mark.asyncio
async def test_mysql_padded_kind_preserves_both_owners_and_raw_cursor(
    _mysql_server: object,
) -> None:
    with _server_database(_mysql_server) as (_, command):
        socket = next(value.split("=", 1)[1] for value in command if value.startswith("--socket="))
        url = URL.create("mysql+asyncmy", username="root", database=command[-1],
                         query={"unix_socket": socket, "charset": "utf8mb4"})
        engine = create_async_engine(url)
        store = SqlStateStore(engine)
        try:
            await provision_database(engine)
            await store.initialize()
            canonical = replace(
                _record(1),
                key=record_key("kind-padding", "tenant", "task", "task_graph", ("graph",)),
                kind="task_graph", state="PENDING",
            )
            opaque = replace(_record(2), kind="task_graph", state="PENDING")
            unrelated = replace(_record(3), kind="task_graph_extra", state="PENDING")
            await store.mutate(lambda transaction: transaction.insert_records((canonical, opaque, unrelated)))

            query = RecordQuery(kind="task_graph ", states=frozenset({"PENDING"}),
                                scope_key=canonical.scope_key, limit=1)
            first = await store.read(lambda transaction: transaction.list_records(query))
            assert first == (canonical,)
            query = replace(query, after_sort_key=first[0].sort_key, after_key=first[0].key)
            second = await store.read(lambda transaction: transaction.list_records(query))
            assert second == (opaque,)
            query = replace(query, after_sort_key=second[0].sort_key, after_key=second[0].key)
            assert await store.read(lambda transaction: transaction.list_records(query)) == ()

            # Only owner discovery may broaden. A tab is not SQL padding and
            # must not be stripped from the actual kind comparison.
            distinct = RecordQuery(kind="task_graph \t", limit=1)
            assert await store.read(lambda transaction: transaction.list_records(distinct)) == ()
        finally:
            await store.close()
            await engine.dispose()
