#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Memory turn codecs and native SQL snapshots preserve exact content ownership."""
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import BaseModel
from pydantic_ai.messages import BinaryContent, ModelResponse, ToolCallPart

from linktools.ai.core import ExecutionStatus, PromptLimits, TenantAuthorizationPolicy
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.model import StoredUserInput
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.inputs import ExecutionInputMaterializer
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence._session_transcript import decode_session_transcript_root
from linktools.ai.persistence.codec import (
    decode_domain, decode_session_turn_commit, encode_session_turn_commit,
    iter_runtime_object_refs,
)
from linktools.ai.persistence.contracts import SessionTurnCommit
from linktools.ai.persistence._snapshot_validation import canonical_snapshot_indexes, validate_snapshot_domain
from linktools.ai.storage import InMemoryObjectStore, read_object

from ._runtime_test_helpers import _Models, _UsageFunctionModel, model_features
from .test_session_timeline import _session
from .test_conversation_output_retention import (
    _NAMESPACE, _features, _storage,
)


def _commit(*, start: int = 0, end: int = 0) -> SessionTurnCommit:
    return SessionTurnCommit("session", 1, "execution", start, end)


@pytest.mark.parametrize("start,end", ((0, 0), (2, 5)))
def test_current_turn_codec_roundtrips_execution_reference_and_message_span(start: int, end: int) -> None:
    commit = _commit(start=start, end=end)
    encoded = encode_session_turn_commit(commit)
    assert encoded["version"] == 1
    assert decode_session_turn_commit(encoded) == commit
    assert not {"stored_user_input", "output", "status", "created_at", "updated_at",
                "error_code", "safe_error_details"}.intersection(encoded)
    assert tuple(iter_runtime_object_refs(encoded, default_domain=PersistenceDomain.CONVERSATION)) == ()


@pytest.mark.parametrize("field", (
    "session_id", "turn_seq", "execution_id", "start_message_index", "end_message_index",
))
def test_current_turn_codec_requires_correlation_identity_and_message_span(field: str) -> None:
    encoded = dict(encode_session_turn_commit(_commit()))
    del encoded[field]
    with pytest.raises(AIError) as raised:
        decode_session_turn_commit(encoded)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_current_turn_codec_preserves_forward_compatible_extra_fields() -> None:
    commit = _commit()
    encoded = {**encode_session_turn_commit(commit), "display_hint": "future"}
    assert decode_session_turn_commit(encoded) == commit


@pytest.mark.asyncio
@pytest.mark.parametrize("field,value", (
    ("version", True), ("version", 2), ("session_id", "different-session"),
    ("execution_id", ""), ("turn_seq", True), ("turn_seq", 2),
    ("start_message_index", -1), ("start_message_index", 1),
    ("end_message_index", True),
))
async def test_repository_and_snapshot_reject_same_corrupt_turn_metadata(
    field: str, value: object,
) -> None:
    # The receipt codec is still used by real agent-bound Memory sessions.
    # SQL Session turns have a different physical owner and manifest format.
    storage = Persistence(PersistenceStorage.in_memory())
    await storage.initialize(namespace="turn-codec", tenant_id="tenant")
    try:
        repository = storage.conversation.sessions
        await repository.create(_session())
        row = repository._stored_timeline_commit(_commit())
        source_records = (*await repository.state_store.read(lambda tx: tx.scan_records()), row)
        sequences = canonical_snapshot_indexes(
            namespace="turn-codec", tenant_id="tenant", domain=PersistenceDomain.CONVERSATION,
            records=source_records, facts=(), operations=(),
        )
        row = replace(row, data={**row.data, field: value})
        with pytest.raises(AIError) as raised:
            repository._decode_timeline_commit("session", 1, row)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        records = (*await repository.state_store.read(lambda tx: tx.scan_records()), row)
        with pytest.raises(AIError) as raised:
            validate_snapshot_domain(namespace="turn-codec", tenant_id="tenant",
                domain=PersistenceDomain.CONVERSATION, records=records, facts=(), operations=(),
                sequences=sequences)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


class _NativeHostAuthorization(TenantAuthorizationPolicy):
    """Authorize ordinary SQL Session operations for this test host."""


class _CanonicalOutput(BaseModel):
    canonical: str


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "local_sqlite"))
async def test_snapshot_roundtrips_turn_references_and_native_payload_owners(
    tmp_path: Path, backend: str,
) -> None:
    output = {"canonical": "x" * (80 * 1024)}
    attachment = b"input attachment" * 8192

    async def model(_messages, info):
        [tool] = info.output_tools
        return ModelResponse(parts=[ToolCallPart(tool.name, output, tool_call_id="structured-output")])

    policy = _NativeHostAuthorization()
    storage = _storage(tmp_path, backend)
    async with (
        Persistence.open(_NAMESPACE, storage=storage) as state,
        Runtime.open(persistence=state, features=model_features(_Models(_UsageFunctionModel(model)), _features()),
                     authorization=policy) as runtime,
    ):
        principal = runtime.default_principal
        await runtime.sessions.create_turn_session("session", principal=principal)
        turn = await runtime.sessions.start_turn(
            "session", ("original input", BinaryContent(attachment, media_type="text/plain")),
            agent="default", principal=principal, idempotency_key="capture", output=_CanonicalOutput,
        )
        result = await turn.wait_final(timeout_seconds=10)
        assert result.status is ExecutionStatus.SUCCEEDED
        expected = (await runtime.history.turn_session("session", principal=principal))["timeline"]["items"]
        [item] = expected
        assert item["turn_id"] == turn.key.turn_id and item["output"] == output
        live = await state.session_turns.get_turn(turn.key)
        assert live["result_json"]["output_ref"] is not None
        assert decode_session_transcript_root(live["result_json"]).count >= 2
        assert "output" not in live["result_json"] and "canonical" not in live["result_json"]
        assert state.conversation.sessions is None
        assert await state.execution.executions.get(turn.key.turn_id, tenant_id=principal.tenant_id) is None

    archive = InMemoryObjectStore("turn-snapshot")
    limits = SnapshotLimits(max_entries=1000, max_bytes=4 * 1024 * 1024)
    async with Persistence.open(_NAMESPACE, storage=_storage(tmp_path, backend), read_only=True) as reader:
        snapshot = await reader.export_snapshot(object_store=archive, limits=limits)
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=archive, root=root, limits=limits)
    async with (
        Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_root(root),
                         tenant_id=principal.tenant_id, read_only=True) as restored,
        History.open(persistence=restored, authorization=policy) as history,
    ):
        assert (await history.turn_session("session", principal=principal))["timeline"]["items"] == expected
        owner = await restored.session_turn_reader.get_turn(turn.key)
        assert owner["result_json"]["output_ref"] == live["result_json"]["output_ref"]
        assert decode_session_transcript_root(owner["result_json"]).slices == decode_session_transcript_root(live["result_json"]).slices
        assert await restored.execution.executions.get(turn.key.turn_id, tenant_id=principal.tenant_id) is None
        stored = decode_domain(owner["request_json"]["input"], StoredUserInput)
        [ref] = stored.payload.object_refs()
        objects = restored.object_store(PersistenceDomain.EXECUTION)
        assert await read_object(objects, ref.key, expected_digest=ref.digest, expected_size=ref.size) == attachment
        prompt = await ExecutionInputMaterializer(None, PromptLimits(), object_store=objects).restore(stored)
        assert prompt[0] == "original input"
        assert [part.data for part in prompt if isinstance(part, BinaryContent)] == [attachment]
