#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""History captures admission, head and counters in the same bounded SQL read."""

from pathlib import Path

import pytest
from sqlalchemy import text
from sqlalchemy.dialects import mysql, postgresql, sqlite

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence._sql import _SqlTransaction

from .test_native_text_result_runtime import _prepare, _publish


@pytest.mark.asyncio
@pytest.mark.parametrize("damage", ("none", "missing_counter", "orphan_counter", "wrong_raw_key", "negative"))
async def test_history_capture_joins_counters_and_rejects_corrupt_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, damage: str,
) -> None:
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "capture.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "output", "output")
        await _publish(commands, commit, current, checkpoint)
        archive = state.run_store.read_store(PersistenceDomain.TRACE)
        store = archive.state_store
        event_key = archive._stream("run", "event")
        await store.mutate(lambda tx: tx.reserve_sequence(event_key, 1))
        if damage == "orphan_counter":
            await store.mutate(lambda tx: tx.delete_records((archive._agent_run_key("run"), archive._history.head_key("run"))))
        elif damage in {"missing_counter", "wrong_raw_key", "negative"}:
            async with store.storage_group.context.engine.begin() as connection:
                if damage == "missing_counter":
                    await connection.execute(text("DELETE FROM ai_state_counters WHERE store_key=:store"), {"store": store.store_key})
                elif damage == "wrong_raw_key":
                    await connection.execute(text("UPDATE ai_state_counters SET stream_key=:other WHERE store_key=:store AND stream_key=:key"),
                        {"other": archive._stream("other", "event"), "store": store.store_key, "key": event_key})
                else:
                    await connection.execute(text("UPDATE ai_state_counters SET value=-1 WHERE store_key=:store AND stream_key=:key"),
                        {"store": store.store_key, "key": event_key})
        statements: list[str] = []
        original = _SqlTransaction._execute

        async def execute(self: _SqlTransaction, statement: object) -> object:
            sql = str(statement)
            if "joined_sequence_row" in sql:
                for dialect in (sqlite.dialect(), postgresql.dialect(), mysql.dialect()):
                    compiled = statement.compile(dialect=dialect, compile_kwargs={"render_postcompile": True})
                    assert len(compiled.params) < 999
                    assert "COLLATE utf8mb4_bin)" not in str(compiled)
            if sql.lstrip().upper().startswith(("SELECT", "WITH")):
                statements.append(sql)
            return await original(self, statement)

        monkeypatch.setattr(_SqlTransaction, "_execute", execute)
        if damage in {"orphan_counter", "wrong_raw_key", "negative"}:
            with pytest.raises(AIError) as caught:
                await archive.capture_history(("run",))
            assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        else:
            captured = await archive.capture_history(("run",))
            assert captured["run"].run.agent_run_id == "run"
            assert captured["run"].message_count == 1
            if damage == "missing_counter":
                assert captured["run"].event_count == 0
        assert len(statements) == 1
        assert "joined_sequence_row" in statements[0]
        statements.clear()
        assert await archive.capture_history(()) == {}
        assert statements == []
