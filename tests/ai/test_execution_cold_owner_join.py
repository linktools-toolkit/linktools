#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Uncached full reads capture actual owners and fixed content in one snapshot."""

from pathlib import Path

import pytest
from sqlalchemy import event, text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.contracts import ExecutionRecord

from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("none", "source", "owner"))
async def test_uncached_full_execution_join_retries_cold_archive_and_checks_source(
    tmp_path: Path, damage: str,
) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "cold.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "fixed output", "fixed output")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        if damage == "source":
            async with repository.state_store.storage_group.context.engine.begin() as connection:
                await connection.execute(text("DELETE FROM ai_step_sequences"))
        if damage == "owner":
            await repository.state_store.mutate(
                lambda transaction: transaction.delete_record(repository.execution_record_key("execution")))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        repository = state.execution.executions
        group = repository.state_store.storage_group
        assert not repository.state_store.record_cache_enabled
        assert group.opened_archive is None
        captures = 0

        async def capture(_group: object) -> ExecutionRecord | None:
            nonlocal captures
            captures += 1
            return await repository.get("execution", tenant_id="tenant")

        if damage == "source":
            with pytest.raises(AIError) as raised:
                await repository.read_content(capture)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            assert captures == 2
        elif damage == "owner":
            assert await repository.read_content(capture) is None
            assert captures == 1
            assert group.opened_archive is None
        else:
            loaded = await repository.read_content(capture)
            assert loaded.stored_user_input == current.stored_user_input
            assert loaded.result == commit.result
            assert captures == 2
            statements: list[str] = []

            def observe(connection: object, cursor: object, statement: str, parameters: object,
                        context: object, executemany: bool) -> None:
                if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                    statements.append(statement)

            engine = group.context.engine.sync_engine
            event.listen(engine, "before_cursor_execute", observe)
            try:
                assert await repository.get_fresh("execution", tenant_id="tenant") == loaded
            finally:
                event.remove(engine, "before_cursor_execute", observe)
            # One current owner/content JOIN plus the authenticated Native range.
            assert len(statements) == 2
            assert sum("ai_step_sequences" in statement for statement in statements) == 1
