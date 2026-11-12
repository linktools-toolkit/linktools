#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Effect ledger physical ownership without a Trace/Step adapter."""

from dataclasses import replace
from pathlib import Path

import pytest
from sqlalchemy import text

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.state import StateTransaction
from linktools.ai.storage import StoredPayload
from tests.ai.test_tool_execution_boundary import _request


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("memory", "sqlite"))
async def test_effect_ledger_is_the_only_physical_owner(tmp_path: Path, backend: str) -> None:
    storage = PersistenceStorage.in_memory() if backend == "memory" else PersistenceStorage.sqlite(tmp_path / "state.db")
    async with Persistence.open("ledger", tenant_id="tenant", storage=storage) as state:
        tools = state.checkpoint.tools
        store = tools.state_store
        admitted = await tools.admit(_request())
        done = await tools.complete_payload("operation", tenant_id="tenant", owner=admitted.owner,
                                            fence=admitted.fence, result_payload=StoredPayload.inline_text("result"))
        assert await tools.get_by_call("run", "call", tenant_id="tenant") == done
        key = tools._tool_key("operation")
        records = await store.read(lambda transaction: transaction.scan_records())
        assert sum(record.kind == "tool_operation" for record in records) == 1
        if backend == "sqlite":
            async with store.context.engine.connect() as connection:
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_tool_operations")) == 1
                assert await connection.scalar(text("SELECT count(*) FROM ai_step_messages")) == 1
                assert await connection.scalar(text("SELECT count(*) FROM ai_state_records WHERE kind='tool_operation'")) == 0
        else:
            assert store.storage_group.opened_archive is not None
            _, content_store = store.storage_group.opened_archive
            assert len(content_store._record_owners.payload_rows) == 1
            assert len(store._step_tool_operations) == 1
            assert not any(record.kind == "tool_operation" for record in store._records.values())
        raw = await store.read_record(key)
        assert raw is not None

        async def abort_after_delete(transaction: StateTransaction) -> None:
            assert await transaction.delete_record(key, expected_storage_revision=raw.storage_revision)
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        with pytest.raises(AIError) as error:
            await store.mutate(abort_after_delete)
        assert error.value.code is ErrorCode.STORAGE_CONFLICT
        assert await tools.get_operation("operation", tenant_id="tenant") == done
        await store.validate_integrity()
        await store.mutate(lambda transaction: transaction.delete_records((key,)))
        assert await tools.get_by_call("run", "call", tenant_id="tenant") is None


@pytest.mark.asyncio
async def test_subject_lookup_does_not_join_independent_owner_ids(tmp_path: Path) -> None:
    async with Persistence.open("ledger", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "state.db")) as state:
        tools = state.checkpoint.tools
        admitted = await tools.admit(_request())
        store = tools.state_store
        current = await store.read_record(tools._tool_key(admitted.tool_operation_id))
        assert current is not None
        opaque = replace(current, key=b"opaque", scope_key=b"different-scope", subject_id="different-call")
        await store.mutate(lambda transaction: transaction.insert_record(opaque))
        async with store.context.engine.begin() as connection:
            # Independently generated primary keys may coincide across owners.
            await connection.execute(text("UPDATE ai_state_records SET id=100 WHERE key=:key"), {"key": opaque.key})
            await connection.execute(text("UPDATE ai_step_tool_operations SET id=100"))
        selected = await store.read(lambda transaction: transaction.read_record_subjects(
            scope_key=opaque.scope_key, kind="tool_operation", subject_ids=("different-call",)))
        assert tuple(record.key for record in selected) == (opaque.key,)
        selected = await store.read(lambda transaction: transaction.read_record_subjects(
            scope_key=current.scope_key, kind="tool_operation", subject_ids=(current.subject_id,)))
        assert tuple(record.key for record in selected) == (current.key,)
