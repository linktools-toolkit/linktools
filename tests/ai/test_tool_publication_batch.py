#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""One published source pass preserves independent argument and receipt eligibility."""
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from linktools.ai.core import canonical_json_bytes, canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model.messages import encode_model_messages
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence._tool_operation_content import ToolOperationContent
from linktools.ai.persistence._message_root import NativeMessageRoot
from linktools.ai.persistence.codec import decode_envelope, native_tool_arguments_reference, native_tool_result_reference
from linktools.ai.persistence.state import StateTransaction, StoredRecord
from linktools.ai.storage import StoredPayload
from .test_native_text_result_runtime import _prepare, _publish
from .test_tool_execution_boundary import _request


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("arguments", "result", "both", "none", "rollback"))
async def test_tool_publication_shares_eligible_slots_in_one_cas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, case: str,
) -> None:
    config = PersistenceStorage.sqlite(tmp_path / "tools.db")
    args = {"count": 1}
    argument_payload = StoredPayload.inline_bytes(canonical_json_bytes(args))
    async with Persistence.open("native-result", tenant_id="tenant", storage=config) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        repository = state.checkpoint.tools
        operation = await repository.admit(replace(_request(), arguments_payload=argument_payload,
                                                   arguments_digest=canonical_sha256(args)))
        result_payload = StoredPayload.inline_bytes(encode_model_messages((ModelRequest(parts=[
            ToolReturnPart("runtime", "receipt", tool_call_id=operation.tool_operation_id)]),)))
        await repository.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
                                          fence=operation.fence, result_payload=result_payload)
        async def stored() -> StoredRecord:
            return next(record for record in await repository.state_store.read(lambda tx: tx.scan_records())
                        if record.kind == "tool_operation")
        before = await stored()
        match_args, match_result = case in {"arguments", "both", "rollback"}, case in {"result", "both", "rollback"}
        checkpoint = replace(checkpoint, messages=[
            ModelResponse(parts=[ToolCallPart("tool", args if match_args else {"count": "1"}, tool_call_id="call")]),
            ModelRequest(parts=[ToolReturnPart("tool", "receipt" if match_result else "different", tool_call_id="call")]),
            *checkpoint.messages])
        captures = 0
        original_capture = ToolOperationContent.capture_native_sources
        async def capture(self: ToolOperationContent, roots: Sequence[NativeMessageRoot]) -> dict[NativeMessageRoot, tuple[bytes, ...]]:
            nonlocal captures
            captures += 1
            return await original_capture(self, roots)
        with monkeypatch.context() as patch:
            patch.setattr(ToolOperationContent, "capture_native_sources", capture)
            if case == "rollback":
                async def transaction_type(tx: StateTransaction) -> type[StateTransaction]:
                    return type(tx)
                target = await repository.state_store.mutate(transaction_type)
                original_replace = target.replace_record
                async def reject(tx: StateTransaction, record: StoredRecord, *, expected_storage_revision: int) -> bool:
                    changed = await original_replace(tx, record, expected_storage_revision=expected_storage_revision)
                    if record.kind == "tool_operation":
                        fields = decode_envelope(record.data).value["payload"]["fields"]
                        if native_tool_arguments_reference(fields.get("arguments_payload")) is not None:
                            assert native_tool_result_reference(fields.get("result_payload")) is not None
                            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    return changed
                patch.setattr(target, "replace_record", reject)
                with pytest.raises(AIError) as failed:
                    await _publish(commands, commit, current, checkpoint)
                assert failed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
            else:
                await _publish(commands, commit, current, checkpoint)
        assert captures == 1
        after = await stored()
        fields = decode_envelope(after.data).value["payload"]["fields"]
        assert (native_tool_arguments_reference(fields["arguments_payload"]) is not None) is (match_args and case != "rollback")
        assert (native_tool_result_reference(fields["result_payload"]) is not None) is (match_result and case != "rollback")
        assert after.storage_revision - before.storage_revision == int(case in {"arguments", "result", "both"})
        value = await repository.get_operation(operation.tool_operation_id, tenant_id="tenant")
        assert value.arguments_payload == argument_payload and value.result_payload == result_payload


@pytest.mark.asyncio
async def test_publication_expands_distinct_archived_tools_as_one_batch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from collections.abc import Callable
    from linktools.ai.persistence._archive import ContentArchive
    from linktools.ai.persistence.codec import TOOL_PAYLOAD_CODEC, tool_payload_reference

    async with Persistence.open("native-result", tenant_id="tenant",
                                storage=PersistenceStorage.sqlite(tmp_path / "batch.db")) as state:
        commands, commit, current, checkpoint = await _prepare(state, "done", "done")
        repository = state.checkpoint.tools
        expected = {}
        calls, returns = [], []
        for index in range(2):
            args = {"count": index}
            argument = StoredPayload.inline_bytes(canonical_json_bytes(args))
            operation = await repository.admit(replace(_request(), tool_operation_id=f"operation-{index}",
                tool_call_id=f"call-{index}", arguments_payload=argument, arguments_digest=canonical_sha256(args)))
            text = f"receipt-{index}"
            result = StoredPayload.inline_bytes(encode_model_messages((ModelRequest(parts=[
                ToolReturnPart("runtime", text, tool_call_id=operation.tool_operation_id)]),)))
            await repository.complete_payload(operation.tool_operation_id, tenant_id="tenant", owner=operation.owner,
                                              fence=operation.fence, result_payload=result)
            expected[operation.tool_operation_id] = (argument, result)
            calls.append(ToolCallPart("tool", args, tool_call_id=operation.tool_call_id))
            returns.append(ToolReturnPart("tool", text, tool_call_id=operation.tool_call_id))
        before = [record for record in await repository.state_store.read(lambda tx: tx.scan_records())
                  if record.kind == "tool_operation"]
        assert len(before) == 2
        for record in before:
            fields = decode_envelope(record.data).value["payload"]["fields"]
            assert tool_payload_reference(fields["arguments_payload"]) is not None
            assert tool_payload_reference(fields["result_payload"]) is not None
        checkpoint = replace(checkpoint, messages=[ModelResponse(parts=calls), ModelRequest(parts=returns),
                                                   *checkpoint.messages])
        batches = []
        expansions = []
        expanding_tools = False
        original_expand = ToolOperationContent.expand_many
        async def expand_many(self: ToolOperationContent, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
            nonlocal expanding_tools
            previous = expanding_tools
            expanding_tools = True
            expansions.append(len(records))
            try:
                return await original_expand(self, records)
            finally:
                expanding_tools = previous
        original = ContentArchive.read_contents
        async def read_contents(self: ContentArchive, identities: Sequence[str], *, codec: str,
                                reserve: Callable[[str, int], None] | None = None) -> tuple[bytes, ...]:
            if expanding_tools and codec == TOOL_PAYLOAD_CODEC:
                batches.append(tuple(identities))
            return await original(self, identities, codec=codec, reserve=reserve)
        with monkeypatch.context() as patch:
            patch.setattr(ContentArchive, "read_contents", read_contents)
            patch.setattr(ToolOperationContent, "expand_many", expand_many)
            await _publish(commands, commit, current, checkpoint)
        assert expansions == [2]
        assert len(batches) == 1 and len(set(batches[0])) == 4
        after = [record for record in await repository.state_store.read(lambda tx: tx.scan_records())
                 if record.kind == "tool_operation"]
        revisions = {record.key: record.storage_revision for record in before}
        for record in after:
            fields = decode_envelope(record.data).value["payload"]["fields"]
            assert native_tool_arguments_reference(fields["arguments_payload"]) is not None
            assert native_tool_result_reference(fields["result_payload"]) is not None
            assert record.storage_revision == revisions[record.key] + 1
        for identity, (argument, result) in expected.items():
            recovered = await repository.get_operation(identity, tenant_id="tenant")
            assert recovered.arguments_payload == argument and recovered.result_payload == result
