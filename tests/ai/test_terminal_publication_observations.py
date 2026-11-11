#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Terminal metadata observations preserve owner CAS and immutable publication."""

from contextvars import ContextVar
from pathlib import Path

import pytest
from sqlalchemy import select, update

from linktools.ai.capability import TraceCapability
from pydantic_ai_harness.step_persistence import StepPersistence
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import Persistence
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._sql import _SqlGroupTransaction
from linktools.ai.persistence._step_message_content import NativeMessageContent
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.state import active_state_scope

from ._runtime_test_helpers import model_features
from ._session_tool_test_helpers import _application, _relevant_kinds, _ToolModels
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("fault", ("execution", "session", "archive_collision"))
async def test_terminal_observations_keep_physical_guards_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    application = _application([])
    application.capability(TraceCapability(), scope="runtime")
    terminal = ContextVar("terminal_publication_observation", default=False)
    original_commit = StateCommands.commit_terminal_checkpoint
    original_get = _SqlGroupTransaction.get_records
    original_insert = NativeMessageContent._insert_rows
    injected = False
    rejected = False

    async with Persistence.open("terminal-observations", storage=_storage("sqlite", tmp_path)) as persistence:
        native = await persistence.create_step_store(max_cached_messages=0)
        try:
            application.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
            async with Runtime.attach(persistence=persistence, features=model_features(_ToolModels(), application)) as runtime:
                async def snapshot():
                    async def read(tx):
                        return (
                            tuple(sorted(await tx.scan_records(), key=lambda value: value.key)),
                            tuple(sorted(await tx.scan_facts(), key=lambda value: (value.stream_digest, value.sequence))),
                            dict(await tx.scan_sequences()),
                        )
                    stores = dict.fromkeys(runtime._state._stores.values())
                    for store in tuple(stores):
                        opened = await store.storage_group.open_archive()
                        if opened is not None:
                            stores[opened[1]] = None
                    state = tuple([await store.read(read) for store in stores])
                    async with native.storage_context.engine.connect() as connection:
                        content = tuple([
                            tuple((await connection.execute(select(table).order_by(table.c.id))).all())
                            for table in build_step_sql_metadata().sorted_tables
                        ])
                    return state, content

                async def observe(self, requests):
                    nonlocal injected
                    result = await original_get(self, requests)
                    scope = active_state_scope()
                    if fault == "archive_collision" or not terminal.get() or injected or scope is None or not scope.writable:
                        return result
                    records = [(store, record) for store, values in result.items() for record in values.values()]
                    if not any(record.kind == "execution" for _store, record in records):
                        return result
                    injected = True
                    store, record = next((store, record) for store, record in records if record.kind == fault)
                    tx = self.transaction(store)
                    table = store.storage_group.metadata.tables["ai_state_records"]
                    # Execution includes recovery control under the same physical CAS.
                    # A competing physical revision must defeat the later CAS even
                    # when the transaction has already captured the previous row.
                    await tx._execute(update(table).where(
                        table.c.store_digest == store.store_digest.hex(),
                        table.c.key == record.key,
                    ).values(storage_revision=record.storage_revision + 1))
                    return result

                async def insert(self, session, table, rows):
                    nonlocal injected
                    if (fault == "archive_collision" and terminal.get() and not injected
                            and table.name == "ai_step_messages" and rows):
                        candidate = rows[0]
                        injected = True
                        # A cold publisher must compare the immutable winner before
                        # its owner references can commit in this same transaction.
                        changed = await session.execute(update(table).where(
                            table.c.store_digest == candidate["store_digest"],
                            table.c.sha256 == candidate["sha256"],
                        ).values(message=b'{"corrupt":true}'))
                        if changed.rowcount == 0:
                            await session.execute(table.insert().values(dict(candidate, message=b'{"corrupt":true}')))
                        else:
                            assert changed.rowcount == 1
                    return await original_insert(self, session, table, rows)

                async def commit(self, value, **kwargs):
                    nonlocal rejected
                    if rejected or value.execution.status is not ExecutionStatus.SUCCEEDED:
                        return await original_commit(self, value, **kwargs)
                    before = await snapshot()
                    token = terminal.set(True)
                    try:
                        with pytest.raises(AIError) as raised:
                            await original_commit(self, value, **kwargs)
                    finally:
                        terminal.reset(token)
                    assert injected
                    assert raised.value.code in {ErrorCode.STORAGE_CONFLICT, ErrorCode.STORAGE_INTEGRITY_ERROR}
                    assert await snapshot() == before
                    rejected = True
                    return await original_commit(self, value, **kwargs)

                monkeypatch.setattr(_SqlGroupTransaction, "get_records", observe)
                monkeypatch.setattr(NativeMessageContent, "_insert_rows", insert)
                monkeypatch.setattr(StateCommands, "commit_terminal_checkpoint", commit)
                await runtime.agents.get("default").create_session("session")
                session = runtime.agents.get("default").session("session")
                execution = await session.start("inspect", idempotency_key="turn")
                result = (await execution.wait(timeout_seconds=10)).result
                assert injected and rejected
                assert result.status is ExecutionStatus.SUCCEEDED
                assert _relevant_kinds((await session.history()).items) == ["user", "tool_call", "tool_result", "assistant"]
        finally:
            await native.close()
