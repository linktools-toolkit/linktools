#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Thin conversation receipts and snapshots preserve canonical Execution ownership."""
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import BinaryContent

from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.history import History
from linktools.ai.runtime import Runtime
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.inputs import ExecutionInputMaterializer
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.persistence.codec import (
    _encode_persisted_domain, decode_session_turn_commit, encode_session_turn_commit,
    iter_runtime_object_refs,
)
from linktools.ai.persistence.contracts import SessionTurnCommit
from linktools.ai.persistence._snapshot_validation import canonical_snapshot_indexes, validate_snapshot_domain
from linktools.ai.storage import InMemoryObjectStore, read_object

from ._runtime_test_helpers import model_features
from .test_session_timeline import _session
from .test_conversation_output_retention import (
    _NAMESPACE, _canonical_output, _features, _models, _storage,
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
    tmp_path: Path, field: str, value: object,
) -> None:
    storage = PersistenceStorage.sqlite(tmp_path / "state.db")
    storage = Persistence(storage)
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


@pytest.mark.asyncio
@pytest.mark.parametrize("backend", ("sqlite", "local_sqlite"))
async def test_snapshot_roundtrips_turn_references_and_execution_payload_owners(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    output = {"canonical": "x" * (80 * 1024)}
    attachment = b"input attachment" * 8192
    _canonical_output(monkeypatch, output)
    storage = _storage(tmp_path, backend)
    async with Persistence.open(_NAMESPACE, storage=storage) as persistence_113, Runtime.attach(persistence=persistence_113, features=model_features(_models(), _features())) as runtime:
        principal = runtime.default_principal
        session = await runtime.agents.get().create_session("session")
        result = (await session.run(("original input", BinaryContent(attachment, media_type="text/plain")), timeout_seconds=5)).result
        expected = (await session.timeline()).items
    reader = _storage(tmp_path, backend)
    reader = Persistence(reader)
    await reader.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id, read_only=True)
    archive = InMemoryObjectStore("turn-snapshot")
    limits = SnapshotLimits(max_entries=1000, max_bytes=4 * 1024 * 1024)
    try:
        snapshot = await reader.export_snapshot(object_store=archive, limits=limits)
    finally:
        await reader.close()
    root = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=archive, root=root, limits=limits)
    async with Persistence.open(_NAMESPACE, storage=PersistenceStorage.from_root(root), tenant_id="default", read_only=True) as history_persistence_139, History.open(persistence=history_persistence_139) as history:
        assert (await history.session_timeline("session", principal=principal)).items == expected
    restored = PersistenceStorage.from_root(root)
    restored = Persistence(restored)
    await restored.initialize(namespace=_NAMESPACE, tenant_id=principal.tenant_id)
    try:
        commits = await restored.conversation.sessions.list_timeline_commits(
            "session", tenant_id=principal.tenant_id, start_turn_seq=1, end_turn_seq=2,
        )
        assert len(commits) == 1
        commit = commits[0]
        refs = tuple(iter_runtime_object_refs(encode_session_turn_commit(commit), default_domain=PersistenceDomain.CONVERSATION))
        assert refs == ()
        assert commit.execution_id == result.execution_id
        owner = await restored.execution.executions.get(commit.execution_id, tenant_id=principal.tenant_id)
        assert owner is not None
        refs = tuple(iter_runtime_object_refs(_encode_persisted_domain(owner), default_domain=PersistenceDomain.EXECUTION))
        assert len(refs) == 1 and refs[0][0] is PersistenceDomain.EXECUTION
        objects = restored.object_store(PersistenceDomain.EXECUTION)
        assert owner.result.output.kind == "inline"
        assert owner.result.output.decode() == output
        ref = refs[0][1]
        assert await read_object(objects, ref.key, expected_digest=ref.digest, expected_size=ref.size) == attachment
        prompt = await ExecutionInputMaterializer(None, PromptLimits(), object_store=objects).restore(owner.stored_user_input)
        assert [part.data for part in prompt if isinstance(part, BinaryContent)] == [attachment]
    finally:
        await restored.close()
