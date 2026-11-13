#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Fixed Step calls replace only byte-equivalent admitted tool arguments."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelResponse, ToolCallPart

from linktools.ai.core import canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_envelope, native_tool_arguments_reference, tool_payload_slots
from linktools.ai.persistence._message_root import NativeMessageRoot, NativeMessageSlice
from linktools.ai.persistence._native_tool_arguments import prepare_native_tool_arguments, restore_native_tool_arguments_payload
from linktools.ai.steps import native_store_key
from linktools.ai.storage import StoredPayload

from .test_native_text_result_runtime import _prepare, _publish
from .test_tool_execution_boundary import _request


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("small", "large", "defaulted", "coerced"))
async def test_published_call_preserves_exact_admitted_arguments(tmp_path: Path, case: str) -> None:
    args = {"value": "x" * (90000 if case == "large" else 1), "count": 1}
    source = dict(args)
    if case == "defaulted":
        source.pop("count")
    elif case == "coerced":
        source["count"] = "1"
    payload = StoredPayload.inline_bytes(canonical_json_bytes(args))
    config = PersistenceStorage.sqlite(tmp_path / "arguments.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        request = replace(_request(), arguments_payload=payload, arguments_digest=canonical_sha256(args))
        operation = await state.checkpoint.tools.admit(request)
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).arguments_payload == payload
        checkpoint = replace(checkpoint, messages=[ModelResponse(parts=[ToolCallPart("tool", source, tool_call_id="call")]),
                                                   *checkpoint.messages])
        await _publish(commands, commit, current, checkpoint)
        records = await state.checkpoint.tools.state_store.read(lambda tx: tx.scan_records())
        stored = next(record for record in records if record.kind == "tool_operation")
        wire = decode_envelope(stored.data).value["payload"]["fields"]["arguments_payload"]
        assert (native_tool_arguments_reference(wire) is not None) is (case in {"small", "large"})
        # Updating the effect after publication must retain its accepted arguments.
        await state.checkpoint.tools.complete_payload(operation.tool_operation_id, tenant_id="tenant",
            owner=operation.owner, fence=operation.fence, result_payload=StoredPayload.inline_text("done"))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config, read_only=True) as state:
        values = await state.checkpoint.tools.get_operations((operation.tool_operation_id,), tenant_id="tenant")
        assert values[0].arguments_payload == payload
        assert await state.checkpoint.tools.get_operations((operation.tool_operation_id,), tenant_id="other") == ()


def test_native_arguments_identity_schema_and_slot_are_strict() -> None:
    root = NativeMessageRoot("a" * 32, native_store_key("native-result", "tenant"),
                             (NativeMessageSlice("b" * 64, 1, 0, 1),))
    raw = encode_model_messages((ModelResponse(parts=[ToolCallPart("tool", '{"x": 1}', tool_call_id="call")]),))
    payload = StoredPayload.inline_bytes(canonical_json_bytes({"x": 1}))
    reference = prepare_native_tool_arguments(root, raw, 0, payload)
    assert reference is not None
    assert set(reference.to_json()) == {"$native_tool_arguments", "part_index", "tool_name", "tool_call_id"}
    assert restore_native_tool_arguments_payload(reference, raw) == payload
    with pytest.raises(AIError) as rejected:
        restore_native_tool_arguments_payload(replace(reference, tool_call_id="foreign"), raw)
    assert rejected.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    wrapper = {"$dataclass": "stored_payload", "schema": 1, "fields": {"content": reference.to_json()}}
    for invalid in ({**wrapper, "schema": 999}, {key: value for key, value in wrapper.items() if key != "schema"}):
        with pytest.raises(AIError):
            native_tool_arguments_reference(invalid)
    with pytest.raises(AIError):
        tuple(tool_payload_slots({"$dataclass": "tool_operation", "fields": {"result_payload": wrapper}}))
    assert prepare_native_tool_arguments(root, raw, 0, StoredPayload.inline_bytes(b'{ "x" : 1 }')) is None


@pytest.mark.asyncio
async def test_argument_owner_and_step_publication_rollback_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from linktools.ai.persistence.state import StateTransaction, StoredRecord

    async with Persistence.open("native-result", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "rollback.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        args = {"x": 1}
        payload = StoredPayload.inline_bytes(canonical_json_bytes(args))
        operation = await state.checkpoint.tools.admit(replace(
            _request(), arguments_payload=payload, arguments_digest=canonical_sha256(args)))
        checkpoint = replace(checkpoint, messages=[ModelResponse(parts=[ToolCallPart("tool", args, tool_call_id="call")]),
                                                   *checkpoint.messages])

        async def transaction_type(tx: StateTransaction) -> type[StateTransaction]:
            return type(tx)

        target = await state.checkpoint.tools.state_store.mutate(transaction_type)
        original = target.replace_record
        failures = 0

        async def fail_after_reference(tx: StateTransaction, record: StoredRecord, *, expected_storage_revision: int) -> bool:
            nonlocal failures
            replaced = await original(tx, record, expected_storage_revision=expected_storage_revision)
            if record.kind == "tool_operation":
                wire = decode_envelope(record.data).value["payload"]["fields"].get("arguments_payload")
                if wire is not None and native_tool_arguments_reference(wire) is not None:
                    assert replaced
                    failures += 1
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return replaced

        with monkeypatch.context() as patch:
            patch.setattr(target, "replace_record", fail_after_reference)
            with pytest.raises(AIError):
                await _publish(commands, commit, current, checkpoint)
        assert failures == 1
        records = await state.checkpoint.tools.state_store.read(lambda tx: tx.scan_records())
        stored = next(record for record in records if record.kind == "tool_operation")
        wire = decode_envelope(stored.data).value["payload"]["fields"]["arguments_payload"]
        assert native_tool_arguments_reference(wire) is None
        assert (await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")).arguments_payload == payload
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).status == current.status


@pytest.mark.asyncio
async def test_fixed_arguments_still_reject_corrupt_owner_digest(tmp_path: Path) -> None:
    from copy import deepcopy
    from linktools.ai.persistence.state import StateTransaction

    async with Persistence.open("native-result", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "owner-digest.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        args = {"x": 1}
        payload = StoredPayload.inline_bytes(canonical_json_bytes(args))
        operation = await state.checkpoint.tools.admit(replace(
            _request(), arguments_payload=payload, arguments_digest=payload.digest))
        checkpoint = replace(checkpoint, messages=[ModelResponse(parts=[ToolCallPart("tool", args, tool_call_id="call")]),
                                                   *checkpoint.messages])
        await _publish(commands, commit, current, checkpoint)

        async def corrupt(transaction: StateTransaction) -> None:
            records = await transaction.scan_records()
            record = next(value for value in records if value.kind == "tool_operation")
            data = deepcopy(dict(record.data))
            fields = data["value"]["payload"]["fields"]
            assert native_tool_arguments_reference(fields["arguments_payload"]) is not None
            fields["arguments_digest"] = "0" * 64
            assert await transaction.replace_record(replace(record, data=data,
                storage_revision=record.storage_revision + 1), expected_storage_revision=record.storage_revision)

        await state.checkpoint.tools.state_store.mutate(corrupt)
        with pytest.raises(AIError, match=ErrorCode.STORAGE_INTEGRITY_ERROR.value):
            await state.checkpoint.tools.get_operation(operation.tool_operation_id, tenant_id="tenant")
