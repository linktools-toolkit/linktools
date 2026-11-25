#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A split first-Stop conflict must be confirmed outside MySQL's old read view."""

from contextlib import AsyncExitStack
from dataclasses import replace

import pytest
from sqlalchemy import select
from sqlalchemy.engine import URL
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.migrate import provision_database
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistencePlan, PersistenceStorage
from linktools.ai.persistence._session_turn_sql import SessionTurnKey
from linktools.ai.persistence._split_step_source import SqlSplitStepSource
from linktools.ai.persistence._step_schema import build_step_domain_sql_metadata
from linktools.ai.runtime._canonical_split_step import _CanonicalSplitStep
from linktools.ai.storage import StorageRoute

from ._metrics_server_test_helpers import _server_database
from .test_session_turn_sql import accept


@pytest.mark.asyncio
@pytest.mark.parametrize("winner", ("none", "exact", "unrelated"))
async def test_mysql_split_stop_reads_back_duplicate_in_fresh_transaction(_mysql_server, monkeypatch, winner):
    key = SessionTurnKey("chat", "turn")
    result = {"error_code": ErrorCode.EXECUTION_CANCELLED.value}
    async with AsyncExitStack() as stack:
        def engine(database):
            _, command = stack.enter_context(_server_database(database))
            socket = next(option.split("=", 1)[1] for option in command if option.startswith("--socket="))
            url = URL.create("mysql+asyncmy", username="root", database=command[-1], query={"unix_socket": socket})
            connection = create_async_engine(url, isolation_level="REPEATABLE READ", hide_parameters=True)
            stack.push_async_callback(connection.dispose)
            return connection

        conversation, execution = engine(_mysql_server), engine(_mysql_server)
        for database in (conversation, execution):
            await provision_database(database)
        source_route = StorageRoute.sql(execution)
        state = await stack.enter_async_context(Persistence.open(
            "mysql-split-stop", storage=PersistenceStorage(PersistencePlan(
                conversation=StorageRoute.sql(conversation), execution=source_route, checkpoint=source_route,
            )),
        ))
        turns = state.session_turns
        assert turns is not None
        native = await state.create_step_store()
        stack.push_async_callback(native.close)
        source_store = state._stores[PersistenceDomain.EXECUTION]
        # The physical schema is provisioned above; bind its native descriptors
        # for this fixture without enabling the runtime's MySQL split route.
        build_step_domain_sql_metadata((PersistenceDomain.CHECKPOINT,), metadata=source_store.storage_group.metadata)
        source_store.storage_group.bind_native_partition(native.store_key)
        owner = _CanonicalSplitStep(turns, source_store, native)
        await turns.create(key.session_id, config={})
        assert await accept(turns, key.turn_id)
        assert await turns.request_cancel(key)
        target = await turns.split_target_lease(key, generation=1, cancelling=True, expired=False)
        assert target is not None
        original_claim = SqlSplitStepSource.cancel_unregistered
        original_read = owner._read_pin
        reads = []
        expected = None

        async def claim(source, reader, **kwargs):
            nonlocal expected
            expected = kwargs["pin"]
            if winner != "none":
                query = select(source._runs.c.id).where(
                    *source._where(source._runs, turn_id=key.turn_id, session_id=key.session_id),
                )
                assert await reader.scalar(query) is None  # Establish transaction A's RR snapshot.
                async with async_sessionmaker(execution).begin() as writer:
                    second = dict(kwargs)
                    if winner == "unrelated":
                        second["pin"] = replace(expected, receipt_id="unrelated")
                    await original_claim(source, writer, **second)
                assert await reader.scalar(query) is None  # B committed; A still cannot see it.
            return await original_claim(source, reader, **kwargs)

        async def read(identity):
            reads.append(identity)
            return await original_read(identity)

        monkeypatch.setattr(SqlSplitStepSource, "cancel_unregistered", claim)
        monkeypatch.setattr(owner, "_read_pin", read)
        if winner == "unrelated":
            with pytest.raises(AIError) as failure:
                await owner._claim_unregistered_stop(key, target=target, result=result)
            assert failure.value.code is ErrorCode.STORAGE_RECOVERY_REQUIRED
        else:
            proof = await owner._claim_unregistered_stop(key, target=target, result=result)
            assert proof.pin == expected and proof.phase == "sealed"
        # The only receipt read follows a known unique conflict and A's rollback.
        assert reads == ([] if winner == "none" else [owner._identity(key, 1)])
        assert (await turns.get_turn(key))["state"] == "cancelling"
