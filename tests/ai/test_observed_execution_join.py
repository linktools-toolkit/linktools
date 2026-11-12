#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Observed source reads batch immutable contents without weakening write boundaries."""
from pathlib import Path

import pytest
from sqlalchemy import event, text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.asyncio
async def test_observed_execution_join_keeps_fresh_native_sources(tmp_path: Path) -> None:
    async with Persistence.open("native-result", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "observed.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "fixed", "fixed")
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        group = repository.state_store.storage_group
        statements: list[str] = []

        def capture(connection: object, cursor: object, statement: str, parameters: object,
                    context: object, executemany: bool) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(statement)

        event.listen(group.context.engine.sync_engine, "before_cursor_execute", capture)
        try:
            observed = await repository.get_observed("execution", tenant_id="tenant")
            assert observed.value.result == commit.result
            assert len(statements) == 1 and "fixed_native_nodes" in statements[0]
            statements.clear()
            written = await repository.state_store.mutate(lambda transaction:
                repository.get_observed_in_transaction(transaction, "execution", tenant_id="tenant"))
            assert written == observed
            assert not any("fixed_native_nodes" in statement for statement in statements)
            assert sum("ai_step_sequences" in statement for statement in statements) == 1
        finally:
            event.remove(group.context.engine.sync_engine, "before_cursor_execute", capture)
        async with group.context.engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_sequences"))
        with pytest.raises(AIError) as missing:
            await repository.get_observed("execution", tenant_id="tenant")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        with pytest.raises(AIError) as missing_write:
            await repository.state_store.mutate(lambda transaction:
                repository.get_observed_in_transaction(transaction, "execution", tenant_id="tenant"))
        assert missing_write.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
