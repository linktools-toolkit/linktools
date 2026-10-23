#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Current conversation envelopes and portable snapshots preserve terminal meaning."""
from ._runtime_test_helpers import model_features

from linktools.ai.runtime.state import RuntimeState
import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.core import ExecutionStatus, PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import StoredUserInput
from linktools.ai.runtime import Runtime, RuntimeHistory, RuntimeStorage
from linktools.ai.runtime._input import ExecutionInputMaterializer
from linktools.ai.runtime.state import RuntimeDomain, SnapshotLimits
from linktools.ai.runtime.state._codec import (
    decode_session_turn_commit, encode_domain, encode_session_turn_commit,
    iter_runtime_object_refs,
)
from linktools.ai.runtime.state._contracts import SessionTurnCommit
from linktools.ai.runtime.state._snapshot_validation import canonical_snapshot_indexes
from linktools.ai.storage import InMemoryObjectStore, StoredPayload, read_object

from .test_session_timeline import _session
from .test_conversation_output_retention import (
    _NAMESPACE, _canonical_output, _erase_execution_owners, _features, _models, _storage,
)


def _commit(status: ExecutionStatus = ExecutionStatus.SUCCEEDED) -> SessionTurnCommit:
    now = datetime.now(timezone.utc)
    return SessionTurnCommit(
        "session", 1, "execution", 0, 0,
        StoredUserInput("text", StoredPayload.inline_text("original input")),
        StoredPayload.inline_json(None) if status is ExecutionStatus.SUCCEEDED else None,
        status, now, now, None if status is ExecutionStatus.SUCCEEDED else "MODEL_ERROR",
        {} if status is ExecutionStatus.SUCCEEDED else {"reason": "safe", "retry": False},
    )


@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
def test_current_flat_turn_codec_roundtrips_all_terminal_fields(status: ExecutionStatus) -> None:
    commit = _commit(status)
    encoded = encode_session_turn_commit(commit)
    assert encoded["version"] == 1
    assert decode_session_turn_commit(encoded) == commit
    assert (decode_session_turn_commit(encoded).output is not None) == (status is ExecutionStatus.SUCCEEDED)


@pytest.mark.parametrize("field", (
    "stored_user_input", "output", "status", "created_at", "updated_at",
    "error_code", "safe_error_details",
))
def test_current_turn_codec_requires_terminal_fields(field: str) -> None:
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
    ("version", True), ("turn_seq", True), ("end_message_index", -1),
    ("status", encode_domain(ExecutionStatus.STARTED)),
    ("created_at", {"$datetime": "2026-10-19T00:00:00"}),
    ("error_code", 42), ("safe_error_details", []), ("output", None),
    ("stored_user_input", None),
))
async def test_repository_and_snapshot_reject_same_corrupt_turn_metadata(
    tmp_path: Path, field: str, value: object,
) -> None:
    storage = RuntimeStorage.sqlite(tmp_path / "state.db")
    storage = RuntimeState(storage)
    await storage.initialize(namespace="turn-codec", tenant_id="tenant")
    try:
        repository = storage.conversation.sessions
        await repository.create(_session())
        row = repository._stored_timeline_commit(_commit())
        row = replace(row, data={**row.data, field: value})
        with pytest.raises(AIError) as raised:
            repository._decode_timeline_commit("session", 1, row)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        records = (*await repository.state_store.read(lambda tx: tx.scan_records()), row)
        with pytest.raises(AIError) as raised:
            canonical_snapshot_indexes(namespace="turn-codec", tenant_id="tenant",
                domain=RuntimeDomain.CONVERSATION, records=records, facts=(), operations=())
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    finally:
        await storage.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "filesystem"))
async def test_snapshot_roundtrips_retained_turn_payloads_after_execution_prune(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    output = {"canonical": "x" * (80 * 1024)}
    attachment = b"input attachment" * 8192
    _canonical_output(monkeypatch, output)
    storage = _storage(tmp_path, backend)
    async with Runtime.open(_NAMESPACE, storage=storage, features=model_features(_models(), _features())) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        await session.run(("original input", BinaryContent(attachment, media_type="text/plain")), timeout_seconds=5)
        expected = (await session.timeline()).items
    storage = _storage(tmp_path, backend)
    storage = RuntimeState(storage)
    await storage.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        await _erase_execution_owners(storage)
    finally:
        await storage.close()
    reader = _storage(tmp_path, backend)
    reader = RuntimeState(reader)
    await reader.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id, read_only=True)
    archive = InMemoryObjectStore("turn-snapshot")
    limits = SnapshotLimits(max_entries=1000, max_bytes=4 * 1024 * 1024)
    try:
        snapshot = await reader.export_snapshot(object_store=archive, limits=limits)
    finally:
        await reader.close()
    root = tmp_path / "restored"
    await RuntimeState.restore_snapshot(snapshot, object_store=archive, root=root, limits=limits)
    async with RuntimeHistory.open(_NAMESPACE, storage=RuntimeStorage.from_root(root)) as history:
        assert (await history.session_timeline("session", principal=principal)).items == expected
    restored = RuntimeStorage.from_root(root)
    restored = RuntimeState(restored)
    await restored.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        commits = await restored.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
        )
        assert len(commits) == 1
        commit = commits[0]
        refs = tuple(iter_runtime_object_refs(encode_session_turn_commit(commit), default_domain=RuntimeDomain.CONVERSATION))
        assert len(refs) >= 2 and all(domain is RuntimeDomain.CONVERSATION for domain, _ in refs)
        objects = restored.object_store(RuntimeDomain.CONVERSATION)
        ref = commit.output.ref
        assert json.loads(await read_object(objects, ref.key, expected_digest=ref.digest, expected_size=ref.size)) == output
        prompt = await ExecutionInputMaterializer(None, PromptLimits(), object_store=objects).restore(commit.stored_user_input)
        assert [part.data for part in prompt if isinstance(part, BinaryContent)] == [attachment]
    finally:
        await restored.close()
