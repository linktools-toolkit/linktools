#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool error receipts retain fixed, fully verified event sources."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai_harness.step_persistence import StepEvent
from sqlalchemy import update

from linktools.ai.core import canonical_json_bytes, deterministic_id
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import NativeToolErrorReference, Persistence, PersistenceStorage, SqlStepStore
from linktools.ai.persistence._archive import ContentArchive
from linktools.ai.persistence._step_schema import build_step_sql_metadata
from linktools.ai.persistence.codec import decode_envelope, native_tool_error_reference
from linktools.ai.persistence.state import StateGroupTransaction
from linktools.ai.storage import StoredPayload

from .test_tool_execution_boundary import _admit, _commands


@pytest.mark.asyncio
@pytest.mark.parametrize("split", (False, True))
@pytest.mark.parametrize("failure", ("failed", "retry", "noncanonical"))
async def test_fixed_tool_error_reconstructs_original_receipt(
    tmp_path: Path, split: bool, failure: str,
) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "errors.db")
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=config) as state:
        admission = await _admit(state)
        tools = state.checkpoint.tools
        native = SqlStepStore.for_state_store(tools.state_store, namespace="tool-boundary", tenant_id="tenant")
        assert native is not None
        scope, archive_store = await tools.state_store.storage_group.open_archive(create=True)
        native.bind_message_archive(ContentArchive(archive_store, scope, namespace="tool-boundary",
                                                   tenant_id="tenant", object_store=None))
        event = StepEvent(run_id="run", kind="tool_call_failed", step_index=1,
                          tool_name="tool", tool_call_id="call", error="original failure",
                          idempotency_key=deterministic_id("tool", "operation", 1, 1, "tool_call_failed"))
        reference = NativeToolErrorReference(scope, native.store_key, event.run_id, event.idempotency_key)
        code = ErrorCode.TOOL_RETRY_REQUIRED if failure == "retry" else ErrorCode.TOOL_EXECUTION_FAILED
        raw = canonical_json_bytes({"version": 1, "kind": "tool_call_rejected" if failure == "retry"
                                    else "tool_call_failed", "message": event.error})
        payload = StoredPayload.inline_bytes(raw + (b"\n" if failure == "noncanonical" else b""))
        if split:
            await tools.fail_payload("operation", tenant_id="tenant", owner="worker", fence=1,
                                     error_code=code.value, error_payload=payload)
            await native.append_event(event)
            await tools.share_published_error("operation", tenant_id="tenant", source=reference)
        else:
            prepared = await native.prepare_step(events=(event,))

            async def write(group: StateGroupTransaction) -> None:
                await native.write_prepared_step(group.transaction(tools.state_store), prepared)

            await _commands(state).commit_tool_terminal_boundary("operation", tenant_id="tenant", owner="worker",
                fence=1, execution=admission.execution, observed=admission.operation, error_code=code.value,
                error_payload=payload, error_source=reference, write=write, stores=(tools.state_store,))
        records = await tools.state_store.read(lambda transaction: transaction.scan_records())
        stored = next(record for record in records if record.kind == "tool_operation")
        actual = native_tool_error_reference(decode_envelope(stored.data).value["payload"]["fields"]["error_payload"])
        assert actual == (None if failure == "noncanonical" else reference)
        assert (await tools.get_operation("operation", tenant_id="tenant")).error_payload == payload
        with pytest.raises(AIError) as conflict:
            await native.append_event(replace(event, error="different source"))
        assert conflict.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=config, read_only=True) as cold:
        assert (await cold.checkpoint.tools.get_operation("operation", tenant_id="tenant")).error_payload == payload
        assert (await cold.checkpoint.tools.get_operations(("operation",), tenant_id="tenant"))[0].error_payload == payload


@pytest.mark.asyncio
async def test_fixed_error_read_rejects_corrupt_source_metadata(tmp_path: Path) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "corrupt.db")
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=config) as state:
        await _admit(state)
        tools = state.checkpoint.tools
        native = SqlStepStore.for_state_store(tools.state_store, namespace="tool-boundary", tenant_id="tenant")
        assert native is not None
        scope, archive_store = await tools.state_store.storage_group.open_archive(create=True)
        native.bind_message_archive(ContentArchive(archive_store, scope, namespace="tool-boundary",
                                                   tenant_id="tenant", object_store=None))
        event = StepEvent(run_id="run", kind="tool_call_failed", step_index=1, tool_name="tool", tool_call_id="call",
            error="known failure", metadata={"proof": "original"},
            idempotency_key=deterministic_id("tool", "operation", 1, 1, "tool_call_failed"))
        payload = StoredPayload.inline_bytes(canonical_json_bytes(
            {"version": 1, "kind": "tool_call_failed", "message": event.error}))
        await tools.fail_payload("operation", tenant_id="tenant", owner="worker", fence=1,
                                error_code=ErrorCode.TOOL_EXECUTION_FAILED.value, error_payload=payload)
        await native.append_event(event)
        await tools.share_published_error("operation", tenant_id="tenant", source=NativeToolErrorReference(
            scope, native.store_key, event.run_id, event.idempotency_key))
        # The error text still matches. Complete source integrity must reject
        # corrupted metadata rather than treating the text projection as proof.
        table = build_step_sql_metadata().tables["ai_step_events"]
        async with native.storage_context.engine.begin() as connection:
            await connection.execute(update(table).where(table.c.store_key == native.store_key).values(
                metadata={"proof": "changed"}))
        with pytest.raises(AIError) as corrupt:
            await tools.get_operation("operation", tenant_id="tenant")
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    async with Persistence.open("tool-boundary", tenant_id="tenant", storage=config, read_only=True) as cold:
        with pytest.raises(AIError) as corrupt:
            await cold.checkpoint.tools.get_operations(("operation",), tenant_id="tenant")
        assert corrupt.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
