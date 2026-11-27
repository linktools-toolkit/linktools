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
from linktools.ai.persistence._sql import _SqlTransaction
from linktools.ai.persistence._step_message_content import NativeMessageContent
from linktools.ai.persistence._step_schema import build_step_sql_metadata

from ._runtime_test_helpers import model_features
from ._session_tool_test_helpers import _application, _ToolModels
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("fault", ("execution", "archive_collision"))
async def test_terminal_observations_keep_physical_guards_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    application = _application([])
    application.capability(TraceCapability(), scope="runtime")
    terminal = ContextVar("terminal_publication_observation", default=False)
    original_commit = StateCommands.commit_terminal_checkpoint
    original_guard = _SqlTransaction.guard_record
    original_insert = NativeMessageContent._insert_rows
    injected = False
    rejected = False

    async with Persistence.open("terminal-observations", storage=_storage("sqlite", tmp_path)) as persistence:
        native = await persistence.create_step_store(max_cached_messages=0)
        try:
            application.capability(StepPersistence(store=native, capture_frontier=True), scope="runtime")
            async with Runtime.open(persistence=persistence, features=model_features(_ToolModels(), application)) as runtime:
                async def snapshot():
                    async def read(tx):
                        return (
                            tuple(sorted(await tx.scan_records(), key=lambda value: value.key)),
                            tuple(sorted(await tx.scan_facts(), key=lambda value: (value.stream_key, value.sequence))),
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

                async def guard(self, key, *, expected_storage_revision):
                    nonlocal injected
                    if fault == "execution" and terminal.get() and not injected:
                        record = await self.get_record(key)
                        if record is not None and record.kind == "execution":
                            from linktools.ai.persistence._record_owner import record_owner_table
                            store = persistence.execution.executions.state_store
                            table = store.storage_group.metadata.tables[record_owner_table(key)]
                            changed = await self._execute(update(table).where(
                                table.c.store_key == store.store_key, table.c.key == key,
                            ).values(storage_revision=expected_storage_revision + 1))
                            assert changed.rowcount == 1
                            injected = True
                    return await original_guard(self, key, expected_storage_revision=expected_storage_revision)

                async def insert(self, session, table, rows):
                    nonlocal injected
                    if (fault == "archive_collision" and terminal.get() and not injected
                            and table.name == "ai_messages" and rows):
                        candidate = rows[0]
                        injected = True
                        # A cold publisher must compare the immutable winner before
                        # its owner references can commit in this same transaction.
                        changed = await session.execute(update(table).where(
                            table.c.store_key == candidate["store_key"],
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

                monkeypatch.setattr(_SqlTransaction, "guard_record", guard)
                monkeypatch.setattr(NativeMessageContent, "_insert_rows", insert)
                monkeypatch.setattr(StateCommands, "commit_terminal_checkpoint", commit)
                execution = await runtime.agents.get("default").start("inspect", idempotency_key="turn")
                result = (await execution.wait(timeout_seconds=10)).result
                assert injected and rejected
                assert result.status is ExecutionStatus.SUCCEEDED
                assert result.output is not None
        finally:
            await native.close()


async def test_native_terminal_stale_generation_preserves_occupancy_and_transcript(tmp_path: Path) -> None:
    from sqlalchemy import func

    from linktools.ai.persistence.turns import SessionTurnKey

    from .test_session_tool_process_recovery import _expire
    from .test_session_turn_sql import _TRANSCRIPT, accept

    key = SessionTurnKey("chat", "turn")
    async with Persistence.open("terminal-observations", storage=_storage("sqlite", tmp_path)) as persistence:
        turns = persistence.session_turns
        await turns.create("chat", config={})
        assert await accept(turns, key.turn_id)
        await _expire(turns, key, generation=1)
        assert await turns.take_over(key, generation=1, lease_seconds=60)

        async def snapshot():
            messages = turns._turns.metadata.tables["ai_messages"]
            async with turns._context.engine.connect() as connection:
                count = (await connection.execute(select(func.count()).select_from(messages))).scalar_one()
            return await turns.get_session("chat"), await turns.get_turn(key), count

        before = await snapshot()
        assert before[0]["active_turn_id"] == b"turn" and before[1]["generation"] == 2
        assert not await turns.finish(key, generation=1, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        assert await snapshot() == before
        assert await turns.finish(key, generation=2, turn_no=1, state="succeeded", transcript=_TRANSCRIPT)
        after = await turns.get_session("chat")
        assert after["active_turn_id"] is None and after["last_turn_no"] == 1
        published = await turns.get_history_turn(key, lineage=())
        assert [message.text for message in published["messages"]] == ["two"]
