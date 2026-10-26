#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Completed preparation captures bind missing observations to exact owner keys."""
from dataclasses import replace
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain, Persistence
from linktools.ai.persistence.state import active_state_scope
from linktools.ai.persistence._step_contracts import AgentRunRecord
from linktools.ai.persistence._model_interaction_store import ModelInteractionStateStepArchive
from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_incremental_projection_boundaries import _storage

pytestmark = pytest.mark.asyncio


async def _capture(archive, run, checkpoint):
    requests = await archive.conversation_preparation_record_keys(run, checkpoint)
    assert requests is not None
    async def read(group):
        rows = await group.get_records(requests)
        return {store: {key: rows[store].get(key) for key in keys} for store, keys in requests.items()}
    return await archive.state_store.storage_group.read_many(tuple(requests), read)


@pytest.mark.parametrize("damage", ("wrong_store", "missing_key", "extra_key", "record_identity", "other_run", "foreign_namespace", "foreign_tenant"))
async def test_capture_rejects_mismatched_store_scope_or_keys(tmp_path: Path, damage: str) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="capture", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    try:
        observed = await _capture(archive, run, checkpoint)
        owner = archive.state_store
        if damage == "wrong_store":
            observed[state.execution.executions.state_store] = observed.pop(owner)
        elif damage == "missing_key":
            observed[owner].pop(next(iter(observed[owner])))
        elif damage == "extra_key":
            observed[owner][b"x" * 32] = None
        elif damage == "record_identity":
            await archive.materialize_checkpoint(run, checkpoint)
            observed = await _capture(archive, run, checkpoint)
            key, record = next((k, v) for k, v in observed[owner].items() if v is not None)
            observed[owner][key] = replace(record, key_digest=b"x" * 32)
        elif damage == "other_run":
            run = replace(run, agent_run_id="another")
            checkpoint = replace(checkpoint, agent_run_id="another")
        elif damage in {"foreign_namespace", "foreign_tenant"}:
            archive = ModelInteractionStateStepArchive(
                owner, object_store=None,
                namespace="foreign" if damage == "foreign_namespace" else "capture",
                tenant_id="foreign" if damage == "foreign_tenant" else "tenant",
                runtime_domain=PersistenceDomain.CONVERSATION,
            )
        with pytest.raises(AIError) as raised:
            await archive.prepare_conversation_checkpoint(run, checkpoint, observed_records=observed)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert active_state_scope() is None
    finally:
        await state.close()


async def test_completed_capture_matches_uncaptured_preparation_outside_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="capture", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    try:
        expected = await archive.prepare_conversation_checkpoint(run, checkpoint)
        observed = await _capture(archive, run, checkpoint)
        content = await archive.transcript_repository.content_archive(create=True)
        prepare_content = content.prepare_append
        checked = []
        async def outside_scope(*args, **kwargs):
            assert active_state_scope() is None
            checked.append(True)
            return await prepare_content(*args, **kwargs)
        monkeypatch.setattr(content, "prepare_append", outside_scope)
        actual = await archive.prepare_conversation_checkpoint(run, checkpoint, observed_records=observed)
        assert actual == expected and checked
        async def nested(group):
            with pytest.raises(AIError) as raised:
                await archive.prepare_conversation_checkpoint(run, checkpoint, observed_records=observed)
            assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        await archive.state_store.storage_group.read_many(tuple(observed), nested)
    finally:
        await state.close()


@pytest.mark.parametrize("override", ("relocate", "prepare"))
async def test_custom_preparation_keys_do_not_open_archive_or_encode_bodies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, override: str,
) -> None:
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="capture", tenant_id="tenant")
    original = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    class RelocatingArchive(ModelInteractionStateStepArchive):
        async def relocate_conversation_checkpoint(self, run, checkpoint):
            return await super().relocate_conversation_checkpoint(run, checkpoint)
    class PreparingArchive(ModelInteractionStateStepArchive):
        async def prepare_checkpoints(self, run, checkpoints, **kwargs):
            return await super().prepare_checkpoints(run, checkpoints, **kwargs)
    custom = (RelocatingArchive if override == "relocate" else PreparingArchive)(
        original.state_store, object_store=None, namespace="capture", tenant_id="tenant",
        runtime_domain=PersistenceDomain.CONVERSATION,
    )
    async def forbidden(*args, **kwargs):
        pytest.fail("Custom preparation must not be pre-read by the built-in batching path")
    import linktools.ai.persistence._step_archive as step_module
    def no_encoding(*args, **kwargs):
        pytest.fail("Custom preparation must not be pre-encoded by the built-in batching path")
    monkeypatch.setattr(custom.transcript_repository, "content_archive", forbidden)
    monkeypatch.setattr(step_module, "encode_model_messages", no_encoding)
    try:
        assert await custom.conversation_preparation_record_keys(AgentRunRecord("run"), _checkpoint(2)) is None
    finally:
        await state.close()


@pytest.mark.parametrize("observation", ("changed_body", "omitted_content"))
async def test_unobserved_actual_content_is_freshly_checked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, observation: str,
) -> None:
    from linktools.ai.model.messages import encode_model_messages
    from linktools.ai.persistence._archive import PreparedArchive
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="capture", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    try:
        observed = await _capture(archive, run, checkpoint)
        content = await archive.transcript_repository.content_archive(create=True)
        if observation == "changed_body":
            checkpoint.messages[0].parts[0].content = "x"
        else:
            observed[content.store] = {}
        actual = PreparedArchive()
        values = (("model-message", encode_model_messages((checkpoint.messages[0],))),)
        await content.prepare_content(values, actual)
        record = actual.records[content.content_record_keys(values)[0]]
        from linktools.ai.storage import StoredPayload
        corrupt = replace(record, data={**record.data, "payload": StoredPayload.inline_bytes(b"bad").to_json()})
        await content.store.mutate(lambda transaction: transaction.insert_record(corrupt))
        read_content = content._read_content
        checked = []
        async def checked_body(value):
            checked.append(value)
            return await read_content(value)
        monkeypatch.setattr(content, "_read_content", checked_body)
        with pytest.raises(AIError) as raised:
            await archive.prepare_conversation_checkpoint(run, checkpoint, observed_records=observed)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert checked, "The actual unobserved key must load and validate its durable body"
        assert not await archive.get_agent_run(agent_run_id="run")
    finally:
        await state.close()


async def test_changed_body_does_not_reuse_unrelated_missing_observations(tmp_path: Path) -> None:
    from linktools.ai.persistence._archive import PreparedArchive
    state = Persistence(_storage("sqlite", tmp_path))
    await state.initialize(namespace="capture", tenant_id="tenant")
    archive = state.run_store.read_store(PersistenceDomain.CONVERSATION)
    run, checkpoint = AgentRunRecord("run"), _checkpoint(2)
    try:
        observed = await _capture(archive, run, checkpoint)
        checkpoint.messages[0].parts[0].content = "x"
        expected = await archive.prepare_conversation_checkpoint(run, checkpoint)
        prepared = PreparedArchive()
        actual = await archive.prepare_conversation_checkpoint(
            run, checkpoint, observed_records=observed, archive=prepared,
        )
        assert actual == expected
        assert not prepared._observed_content
    finally:
        await state.close()
