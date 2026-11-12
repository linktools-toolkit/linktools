#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Accepted payload content has a Step owner independent from shared bindings."""

from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._archive import ContentArchive, PreparedArchive
from linktools.ai.persistence._record_owner import record_owner_table


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_execution_payload_content_has_one_step_owner(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("payload-owner", tenant_id="tenant", storage=storage) as state:
        group = state.execution.executions.state_store.storage_group
        scope, store = await group.open_archive(create=True)
        archive = ContentArchive(store, scope, namespace="payload-owner", tenant_id="tenant", object_store=None)
        raw = b"accepted input" * 10000
        prepared = PreparedArchive()
        await archive.prepare_content((("execution-payload", raw), ("execution-binding", raw)), prepared)
        await store.mutate(lambda transaction: archive.publish(transaction, prepared))
        keys = archive.content_record_keys((("execution-payload", raw), ("execution-binding", raw)))
        assert record_owner_table(keys[0]) == "ai_step_payload_contents"
        assert record_owner_table(keys[1]) == "ai_state_records"
        records = await store.read_records(keys)
        assert len(records) == 2
        identity = records[keys[0]].sort_key.decode()
        assert await archive.read_contents((identity,), codec="execution-payload") == (raw,)
        # The specialized content-frontier SQL must use the same owner as get/batch/scan.
        values = await store.read(lambda transaction: transaction.read_archive_frontiers(
            scope_key=records[keys[0]].scope_key, requests=(), content_keys=keys)) if backend == "sqlite" else None
        if values is not None:
            assert set(values) == set(keys)
        if backend == "sqlite":
            async with store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_payload_contents")) == 1
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='archive_content'")) == 1
        else:
            assert len(store._step_payload_contents) == 1
            assert sum(record.kind == "archive_content" for record in store._records.values()) == 1
        await store.mutate(lambda transaction: transaction.delete_records(keys))
        assert await store.read_records(keys) == {}
