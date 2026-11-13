#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Published tool text shares its immutable Step body without changing receipts."""

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ToolReturnPart

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceDomain, PersistenceStorage
from linktools.ai.persistence.codec import decode_envelope, native_tool_result_reference
from linktools.ai.persistence.contracts import ToolOperationRecord
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.storage import StoredPayload

from .test_native_text_result_runtime import _prepare, _publish
from .test_tool_execution_boundary import _request


async def _settle(state: Persistence, body: object) -> tuple[ToolOperationRecord, StoredPayload]:
    operation = await state.checkpoint.tools.admit(_request())
    payload = StoredPayload.inline_bytes(encode_model_messages((ModelRequest(parts=[
        ToolReturnPart("runtime", body, tool_call_id=operation.tool_operation_id)]),)))
    await state.checkpoint.tools.complete_payload(operation.tool_operation_id, tenant_id="tenant",
        owner=operation.owner, fence=operation.fence, result_payload=payload)
    return operation, payload


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("large_equal", "small_equal", "different", "structured_equal",
                                 "large_structured_equal", "typed_different"))
async def test_tool_receipt_reuses_only_exact_published_content(tmp_path: Path, case: str) -> None:
    body = ({"items": [None, True, 1, 1.0], "text": "structured " * (9000 if case.startswith("large") else 1)}
            if "structured" in case else {"value": 1} if case == "typed_different"
            else "tool body " * (9000 if case != "small_equal" else 1))
    config = PersistenceStorage.sqlite(tmp_path / "tool.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        operation, payload = await _settle(state, body)
        source = ({"value": True} if case == "typed_different" else
                  "different " * 9000 if case == "different" else body)
        checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[ToolReturnPart("tool", source, tool_call_id="call")]),
                                                   *checkpoint.messages])
        # The original effect is durable before any Step publication.
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
        await _publish(commands, commit, current, checkpoint)
        records = await state.checkpoint.tools.state_store.read(lambda tx: tx.scan_records())
        record = next(record for record in records if record.kind == "tool_operation")
        reference = native_tool_result_reference(decode_envelope(record.data).value["payload"]["fields"]["result_payload"])
        assert (reference is not None) is case.endswith("equal")
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
        values = await state.checkpoint.tools.get_operations((operation.tool_operation_id,), tenant_id="tenant")
        assert values[0].result_payload == payload


@pytest.mark.asyncio
async def test_tool_reference_and_step_publication_roll_back_together(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "tool.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        body = "atomic tool body " * 9000
        operation, payload = await _settle(state, body)
        checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[ToolReturnPart("tool", body, tool_call_id="call")]),
                                                   *checkpoint.messages])
        async def transaction_type(transaction: StateTransaction) -> type[StateTransaction]:
            return type(transaction)
        target = await state.checkpoint.tools.state_store.mutate(transaction_type)
        original = target.replace_record
        failed = 0
        async def fail_after_owner(transaction: StateTransaction, record: StoredRecord, *, expected_storage_revision: int) -> bool:
            nonlocal failed
            replaced = await original(transaction, record, expected_storage_revision=expected_storage_revision)
            if record.kind == "tool_operation":
                reference = native_tool_result_reference(decode_envelope(record.data).value["payload"]["fields"]["result_payload"])
                if reference is not None:
                    assert replaced
                    assert await transaction.get_record(record.key) == record
                    failed += 1
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return replaced
        with monkeypatch.context() as patch:
            patch.setattr(target, "replace_record", fail_after_owner)
            with pytest.raises(AIError):
                await _publish(commands, commit, current, checkpoint)
        assert failed == 1
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).status == current.status


@pytest.mark.asyncio
@pytest.mark.parametrize("boundary", ("direct", "prepared"))
async def test_nonterminal_projection_shares_tool_receipt(tmp_path: Path, boundary: str) -> None:
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.sqlite(tmp_path / "tool.db")) as state:
        _commands, _commit, _current, checkpoint = await _prepare(state, "done", "done")
        body = "published tool body " * 9000
        operation, payload = await _settle(state, body)
        checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[ToolReturnPart("tool", body, tool_call_id="call")]),
                                                   *checkpoint.messages])
        from linktools.ai.persistence._step_contracts import AgentRunRecord
        trace = state.run_store.read_store(PersistenceDomain.TRACE)
        run = AgentRunRecord("run")
        if boundary == "direct":
            await trace.sync_projection(run, events=(), checkpoints=(checkpoint,), execution_id="execution")
        else:
            await trace.prepare_run_registration(run)
            prepared = await trace.prepare_checkpoints(run, (checkpoint,))
            await trace.sync_prepared_projection(run, events=(), checkpoints=prepared.checkpoints,
                                                execution_id="execution", producer_generation=1)
        records = await state.checkpoint.tools.state_store.read(lambda tx: tx.scan_records())
        record = next(record for record in records if record.kind == "tool_operation")
        assert native_tool_result_reference(decode_envelope(record.data).value["payload"]["fields"]["result_payload"]) is not None
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).result_payload == payload


def test_noncanonical_or_opaque_receipt_uses_original_payload() -> None:
    from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice
    from linktools.ai.persistence._native_tool_result import prepare_native_tool_result
    from linktools.ai.steps import native_store_key
    body = "receipt " * 10000
    raw = encode_model_messages((ModelRequest(parts=[ToolReturnPart("tool", body, tool_call_id="call")]),))
    canonical = encode_model_messages((ModelRequest(parts=[ToolReturnPart("runtime", body, tool_call_id="operation")]),))
    source = NativeMessageRoot("a" * 32, native_store_key("native-result", "tenant"), (NativeMessageSlice("b" * 64, 1, 0, 1),))
    for original in (json.dumps(json.loads(canonical), indent=2).encode(), b"opaque receipt" * 10000):
        assert prepare_native_tool_result(source, raw, 0, StoredPayload.inline_bytes(original), operation_id="operation") is None
