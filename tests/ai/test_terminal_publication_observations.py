#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from linktools.ai.persistence import Persistence
"""Terminal metadata observations preserve owner CAS and immutable publication."""
from ._runtime_test_helpers import model_features

from contextvars import ContextVar
from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import update

from linktools.ai.capability import CheckpointCapability, TraceCapability
from linktools.ai.core import ExecutionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import Runtime
from linktools.ai.persistence._runtime_commands import StateCommands
from linktools.ai.persistence._sql import _SqlGroupTransaction
from linktools.ai.persistence.state import active_state_scope

from ._session_tool_test_helpers import _application, _relevant_kinds, _ToolModels
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


@pytest.mark.parametrize("fault", ("execution", "session", "recovery_checkpoint", "archive_collision"))
async def test_terminal_observations_keep_physical_guards_and_rollback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str,
) -> None:
    application = _application([])
    application.capability(CheckpointCapability(), scope="runtime")
    application.capability(TraceCapability(), scope="runtime")
    terminal = ContextVar("terminal_publication_observation", default=False)
    original_commit = StateCommands.commit_terminal_checkpoint
    original_get = _SqlGroupTransaction.get_records
    injected = False
    rejected = False

    async with Persistence.open('terminal-observations', storage=_storage('sqlite', tmp_path)) as persistence_40, Runtime.open(persistence=persistence_40, features=model_features(_ToolModels(), application)) as runtime:
        async def snapshot():
            async def read(tx):
                return (
                    tuple(sorted(await tx.scan_records(), key=lambda value: value.key_digest)),
                    tuple(sorted(await tx.scan_facts(), key=lambda value: (value.stream_digest, value.sequence))),
                    dict(await tx.scan_sequences()),
                )
            return tuple([await store.read(read) for store in dict.fromkeys(runtime._state._stores.values())])

        async def observe(self, requests):
            nonlocal injected
            result = await original_get(self, requests)
            scope = active_state_scope()
            if not terminal.get() or injected or scope is None or not scope.writable:
                return result
            records = [(store, record) for store, values in result.items() for record in values.values()]
            if not any(record.kind == "execution" for _store, record in records):
                return result
            injected = True
            if fault == "archive_collision":
                store, record = next((store, record) for store, record in records if record.kind == "archive_content")
                missing = next(key for key in requests[store] if key not in result[store])
                await self.transaction(store).insert_record(replace(
                    record, key_digest=missing, kind="archive_sequence", data={"corrupt": True},
                ))
            else:
                store, record = next((store, record) for store, record in records if record.kind == fault)
                tx = self.transaction(store)
                table = store.storage_group.metadata.tables["ai_state_records"]
                # A competing physical revision must defeat the later CAS even
                # when the transaction has already captured the previous row.
                await tx._execute(update(table).where(
                    table.c.store_digest == store.store_digest.hex(),
                    table.c.key_digest == record.key_digest.hex(),
                ).values(storage_revision=record.storage_revision + 1))
            return result

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
        monkeypatch.setattr(StateCommands, "commit_terminal_checkpoint", commit)
        await runtime.agents.get("default").create_session("session")
        session = runtime.agents.get("default").session("session")
        execution = await session.start("inspect", idempotency_key="turn")
        result = (await execution.wait(timeout_seconds=10)).result
        assert injected and rejected
        assert result.status is ExecutionStatus.SUCCEEDED
        assert _relevant_kinds((await session.history()).items) == ["user", "tool_call", "tool_result", "assistant"]
