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
            assert await repository.list_by_execution(operation.execution_id, tenant_id="tenant") == (value,)
            assert len(reads) == (2 if native else 1)
            reads.clear()
            assert await repository.list_by_agent_run(operation.agent_run_id, tenant_id="tenant") == (value,)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("corruption", ("missing", "schema"))
async def test_scoped_tool_join_preserves_empty_owners_order_and_limit(tmp_path: Path, corruption: str) -> None:
    from linktools.ai.persistence.state import RecordQuery, StoredRecord
    from linktools.ai.storage import StoredPayload
    from .test_tool_execution_boundary import _request

    config = PersistenceStorage.sqlite(tmp_path / "listed.db")
    async with Persistence.open("list-join", tenant_id="tenant", storage=config) as state:
        repository = state.checkpoint.tools
        for identity, execution, run in (("a", "execution", "run"), ("b", "execution", "run"),
                                         ("c", "other-execution", "other-run")):
            operation = await repository.admit(replace(_request(), tool_operation_id=identity,
                tool_call_id=identity, execution_id=execution, agent_run_id=run))
            if identity != "b":
                await repository.complete_payload(identity, tenant_id="tenant", owner=operation.owner,
                    fence=operation.fence, result_payload=StoredPayload.inline_text("shared receipt"))
    async with Persistence.open("list-join", tenant_id="other", storage=config) as state:
        other = await state.checkpoint.tools.admit(replace(_request(), tool_operation_id="foreign"))
        await state.checkpoint.tools.complete_payload(other.tool_operation_id, tenant_id="other",
            owner=other.owner, fence=other.fence, result_payload=StoredPayload.inline_text("foreign receipt"))
    async with Persistence.open("list-join", tenant_id="tenant", storage=config, read_only=True) as state:
        repository = state.checkpoint.tools
        # Discover a genuinely cold archive through the ordinary retry path.
        values = await repository.list_by_execution("execution", tenant_id="tenant")
        assert {value.tool_operation_id for value in values} == {"a", "b"}
        assert next(value for value in values if value.tool_operation_id == "b").result_payload is None
        reads: list[str] = []
        def capture(_connection: object, _cursor: object, statement: str,
                    _parameters: object, _context: object, _many: bool) -> None:
            if statement.lstrip().upper().startswith(("SELECT", "WITH")):
                reads.append(statement)
        engine = repository.state_store.storage_group.context.engine.sync_engine
        event.listen(engine, "before_cursor_execute", capture)
        try:
            assert await repository.list_by_agent_run("run", tenant_id="tenant") == values
            assert len(reads) == 1 and "content_owners" in reads[0]
            reads.clear()
            assert await repository.list_by_execution("missing", tenant_id="tenant") == ()
            assert len(reads) == 1
            reads.clear()
            assert await repository.list_by_execution("execution", tenant_id="other") == ()
            assert await repository.list_by_agent_run("run", tenant_id="other") == ()
            assert not reads
        finally:
            event.remove(engine, "before_cursor_execute", capture)
        scope = repository._scope("tool_operation", "agent_run", ("run",))
        query = RecordQuery(kind="tool_operation", scope_key=scope, limit=1)
        expected = await repository.state_store.read(lambda tx: tx.list_records(query))
        async def read(_transaction: object) -> tuple[StoredRecord, ...] | None:
            return await repository._tool_content.prefetch_owners((), query=query)
        assert await repository._read_content(read) == expected
        assert len(expected) == 1
        next_query = replace(query, after_sort_key=expected[0].sort_key, after_key=expected[0].key)
        expected_next = await repository.state_store.read(lambda tx: tx.list_records(next_query))
        async def read_next(_transaction: object) -> tuple[StoredRecord, ...] | None:
            return await repository._tool_content.prefetch_owners((), query=next_query)
        assert await repository._read_content(read_next) == expected_next
        assert len(expected_next) == 1 and expected_next[0].key != expected[0].key
    async with Persistence.open("list-join", tenant_id="tenant", storage=config) as state:
        repository = state.checkpoint.tools
        if corruption == "missing":
            async with repository.state_store.storage_group.context.engine.begin() as connection:
                await connection.execute(text("DELETE FROM ai_step_messages WHERE store_key = :store"),
                                         {"store": payload_store_key("list-join", "tenant")})
        else:
            from copy import deepcopy
            from linktools.ai.persistence.codec import decode_envelope, encode_envelope
            record, = await repository.state_store.read(lambda tx: tx.list_records(query))
            value = deepcopy(decode_envelope(record.data).value)
            value["payload"]["schema"] = 999
            from sqlalchemy import update
            from linktools.ai.core import canonical_json_bytes
            group = repository.state_store.storage_group
            table = group.metadata.tables["ai_step_tool_operations"]
            async with group.context.engine.begin() as connection:
                await connection.execute(update(table).where(table.c.key == record.key).values(
                    payload_json=canonical_json_bytes(encode_envelope(value)).decode("utf-8")))
        for method, identity in ((repository.list_by_execution, "execution"), (repository.list_by_agent_run, "run")):
            with pytest.raises(AIError) as missing:
                await method(identity, tenant_id="tenant")
            assert missing.value.code is (ErrorCode.STORAGE_INTEGRITY_ERROR if corruption == "missing"
                                          else ErrorCode.STORAGE_VERSION_UNSUPPORTED)
