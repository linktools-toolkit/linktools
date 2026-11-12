#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Keyed tool reads capture ordinary fixed payloads with their current owner."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart
from sqlalchemy import event, text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._step_payload_owner import payload_store_key
from linktools.ai.steps import native_store_key

from .test_native_text_result_runtime import _prepare, _publish
from .test_native_tool_result_runtime import _settle


@pytest.mark.asyncio
@pytest.mark.parametrize("native", (False, True))
async def test_keyed_tool_payload_join_preserves_sources(tmp_path: Path, native: bool) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "joined.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        operation, payload = await _settle(state, "receipt")
        if native:
            checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[
                ToolReturnPart("tool", "receipt", tool_call_id="call")]), *checkpoint.messages])
            await _publish(commands, commit, current, checkpoint)
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        repository = state.checkpoint.tools
        archive_open = repository.state_store.storage_group.opened_archive is not None
        engine = repository.state_store.storage_group.context.engine.sync_engine
        reads: list[str] = []

        def capture(_connection: object, _cursor: object, statement: str,
                    _parameters: object, _context: object, _many: bool) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                reads.append(statement)

        event.listen(engine, "before_cursor_execute", capture)
        try:
            value = await repository.get_operation(operation.tool_operation_id, tenant_id="tenant")
            assert value.result_payload == payload
            # No archive prewarming: when discovery is needed the first owner
            # observation is discarded, then the joined reader captures it anew.
            joined = [sql for sql in reads if "content_owners" in sql]
            assert len(joined) == 1 and "ai_step_tool_operations" in joined[0]
            if archive_open:
                assert len(reads) == (2 if native else 1)
            else:
                assert "ai_step_tool_operations" in reads[0] and "content_owners" not in reads[0]
            reads.clear()
            assert await repository.get_operations((operation.tool_operation_id, "missing", operation.tool_operation_id),
                                                   tenant_id="tenant") == (value,)
            assert len(reads) == (2 if native else 1)
            reads.clear()
            assert await repository.get_operation(operation.tool_operation_id, tenant_id="other") is None
            assert await repository.get_operations((), tenant_id="tenant") == ()
            assert not reads
        finally:
            event.remove(engine, "before_cursor_execute", capture)
    # The joined reader must not treat an absent fixed payload as an empty value.
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        engine = state.checkpoint.tools.state_store.storage_group.context.engine
        async with engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_messages WHERE store_key = :store"),
                                     {"store": (native_store_key if native else payload_store_key)("native-result", "tenant")})
        with pytest.raises(AIError) as missing:
            await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
