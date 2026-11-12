#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical text-list input shares only an exact fixed prompt representation."""

from dataclasses import replace
from pathlib import Path

import pytest
from pydantic_ai.messages import ModelRequest, UserPromptPart
from sqlalchemy import text

from linktools.ai.core import PromptLimits
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.inputs import ExecutionInputMaterializer
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.persistence.codec import decode_envelope, native_user_input_reference
from linktools.ai.persistence.snapshots import SnapshotLimits
from linktools.ai.storage import InMemoryObjectStore

from .test_native_input_runtime import _prepare_input
from .test_native_text_result_runtime import _publish
from .test_native_result_portable_gc import Guard


@pytest.mark.asyncio
@pytest.mark.parametrize("case", ("equal", "reordered", "custom_view", "files", "version_bool", "version_float"))
async def test_public_text_list_capture_reuses_only_complete_canonical_input(tmp_path: Path, case: str) -> None:
    original = ExecutionInputMaterializer(None, PromptLimits()).capture(("a", "b", "界"))
    if case in {"custom_view", "files", "version_bool", "version_float"}:
        view = dict(original.view)
        if case in {"version_bool", "version_float"}:
            view["version"] = True if case == "version_bool" else 1.0
        elif case == "files":
            view["files"] = [{"path": "accepted.txt", "digest": "a" * 64}]
        else:
            view["accepted_provenance"] = "must survive"
        original = replace(original, view=view)
    source = ["b", "a", "界"] if case == "reordered" else ["a", "b", "界"]
    storage = PersistenceStorage.sqlite(tmp_path / "input.db")
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original, "placeholder")
        checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[UserPromptPart(source)]), checkpoint.messages[-1]])
        await _publish(commands, commit, current, checkpoint)
        repository = state.execution.executions
        record = await repository.state_store.read_record(repository.execution_record_key("execution"))
        reference = native_user_input_reference(decode_envelope(record.data).value["payload"]["fields"]["stored_user_input"])
        assert (reference is not None) is (case == "equal")
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        assert (await state.execution.executions.get_fresh("execution", tenant_id="tenant")).stored_user_input == original


@pytest.mark.asyncio
async def test_text_list_input_readonly_snapshot_restores_and_rejects_missing_source(tmp_path: Path) -> None:
    original = ExecutionInputMaterializer(None, PromptLimits()).capture(("first", "", "first"))
    storage = PersistenceStorage.sqlite(tmp_path / "source.db")
    objects = InMemoryObjectStore("text-list-input")
    limits = SnapshotLimits(10000, 20_000_000)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage) as state:
        commands, commit, current, checkpoint = await _prepare_input(state, original, "placeholder")
        checkpoint = replace(checkpoint, messages=[ModelRequest(parts=[UserPromptPart(["first", "", "first"])]), checkpoint.messages[-1]])
        await _publish(commands, commit, current, checkpoint)
    async with Persistence.open("native-result", tenant_id="tenant", storage=storage, read_only=True) as state:
        snapshot = await state.export_snapshot(object_store=objects, limits=limits)
    target = tmp_path / "restored"
    await Persistence.restore_snapshot(snapshot, object_store=objects, root=target, limits=limits)
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(target)) as state:
        repository = state.execution.executions
        assert (await repository.get_fresh("execution", tenant_id="tenant")).stored_user_input == original
        await state.purge_unreferenced_objects((), exclusive=Guard())
        assert (await repository.get_fresh("execution", tenant_id="tenant")).stored_user_input == original
        raw = await repository.state_store.read_record(repository.execution_record_key("execution"))
        reference = native_user_input_reference(decode_envelope(raw.data).value["payload"]["fields"]["stored_user_input"])
        assert reference is not None
        context = repository.state_store.storage_group.context
        async with context.engine.begin() as connection:
            await connection.execute(text("DELETE FROM ai_step_messages WHERE store_key=:store AND sha256 IN "
                "(SELECT message_sha256 FROM ai_step_sequences WHERE store_key=:store AND sha256=:root)"),
                {"store": reference.root.store_key, "root": reference.root.slices[0].first})
    async with Persistence.open("native-result", tenant_id="tenant", storage=PersistenceStorage.from_root(target), read_only=True) as state:
        with pytest.raises(AIError) as missing:
            await state.execution.executions.get_fresh("execution", tenant_id="tenant")
        assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
async def test_attachment_input_is_not_a_plain_text_projection_candidate() -> None:
    import hashlib
    from pydantic_ai.messages import BinaryContent
    from linktools.ai.core import canonical_json_bytes
    from linktools.ai.inputs import encode_user_content
    from linktools.ai.model import StoredUserInput, plain_text_input_content
    from linktools.ai.model.attachments import store_attachment_payload
    from linktools.ai.storage import ObjectRef

    raw = canonical_json_bytes(encode_user_content(("read", BinaryContent(b"accepted file", media_type="text/plain"))))

    async def write(data: bytes) -> ObjectRef:
        digest = hashlib.sha256(data).hexdigest()
        return ObjectRef("files", digest, digest, len(data))

    payload = await store_attachment_payload(raw, format="user-content", write_object=write)
    assert payload is not None
    original = StoredUserInput("user-content-v1", payload)
    before = payload.to_json()
    assert plain_text_input_content(original) is None
    assert payload.to_json() == before
    assert original.payload.object_refs() == payload.object_refs()
