#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hot producer receipts never replace recovery source validation."""

from contextlib import suppress
from dataclasses import replace

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime import RuntimeDomain
from linktools.ai.runtime.state import RuntimeState
from linktools.ai.runtime.state._step_archive import StateStepArchive
from linktools.ai.runtime.state._step_contracts import AgentRunRecord

from .test_checkpoint_metadata_snapshot_io import _checkpoint
from .test_incremental_projection_boundaries import _storage


@pytest.mark.asyncio
async def test_confirmed_checkpoint_prefix_avoids_old_bodies_but_rejects_mutation(tmp_path, monkeypatch):
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="prefix-receipt", tenant_id="tenant")
    try:
        store = state.run_store
        run = AgentRunRecord("run")
        await store.register_agent_run(run)
        caller_checkpoint = _checkpoint(2)
        materialize = StateStepArchive.materialize_checkpoint_in_transaction

        async def mutate_caller_after_prepare(self, *args, **kwargs):
            await materialize(self, *args, **kwargs)
            caller_checkpoint.messages[0].parts[0].content = "caller changed during commit"

        monkeypatch.setattr(StateStepArchive, "materialize_checkpoint_in_transaction", mutate_caller_after_prepare)
        await store.save_checkpoint(caller_checkpoint)
        archive = store.read_store(RuntimeDomain.CHECKPOINT)
        assert "run" in archive._checkpoint_prefix_receipts

        async def missing_source(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_NOT_FOUND)

        monkeypatch.setattr(archive.transcript_repository, "load_message_span", missing_source)
        relocated, _ = await archive.prepare_run_checkpoint(run, _checkpoint(3))
        assert relocated.transcript_message_count_before == 2
        changed = _checkpoint(3)
        changed.messages[0].parts[0].content = "same-count mutation"
        with pytest.raises(AIError) as caught:
            await archive.prepare_run_checkpoint(run, changed)
        assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        # A public relocation is a recovery consumer even on the warm producer.
        with pytest.raises(AIError) as caught:
            await archive.relocate_run_checkpoint(run, _checkpoint(3))
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
        archive.release_runtime_cache("run")
        with pytest.raises(AIError) as caught:
            await archive.prepare_run_checkpoint(run, _checkpoint(3))
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await state.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("unknown", [False, True])
async def test_unconfirmed_checkpoint_cannot_advance_prefix_receipt(tmp_path, monkeypatch, unknown):
    state = RuntimeState(_storage("sqlite", tmp_path))
    await state.initialize(namespace="prefix-rollback", tenant_id="tenant")
    try:
        store = state.run_store
        run = AgentRunRecord("run")
        await store.register_agent_run(run)
        await store.save_checkpoint(_checkpoint(1))
        archive = store.read_store(RuntimeDomain.CHECKPOINT)
        receipt = archive._checkpoint_prefix_receipts["run"]
        original = StateStepArchive.materialize_checkpoint_in_transaction

        async def fail_after_write(self, *args, **kwargs):
            await original(self, *args, **kwargs)
            raise OSError("lost transaction before commit")

        async def unknown_readback(*args, **kwargs):
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)

        if unknown:
            mutate = type(archive.state_store).mutate

            async def lose_ack_after_commit(self, callback):
                result = await mutate(self, callback)
                if self is archive.state_store:
                    raise OSError("committed but acknowledgement lost")
                return result

            monkeypatch.setattr(type(archive.state_store), "mutate", lose_ack_after_commit)
            monkeypatch.setattr(archive, "verify_checkpoint_range", unknown_readback)
        else:
            monkeypatch.setattr(StateStepArchive, "materialize_checkpoint_in_transaction", fail_after_write)
        with pytest.raises((AIError, OSError)):
            await store.save_checkpoint(_checkpoint(2))
        assert archive._checkpoint_prefix_receipts["run"] == receipt
        # A receipt for another immutable fact cannot authorize a newer prefix.
        archive._checkpoint_prefix_receipts["run"] = replace(
            receipt, checkpoint=replace(receipt.checkpoint, step_index=999),
        )
        called = []
        load = archive.transcript_repository.load_message_span

        async def observe(*args, **kwargs):
            called.append(True)
            return await load(*args, **kwargs)

        monkeypatch.setattr(archive.transcript_repository, "load_message_span", observe)
        await archive.prepare_run_checkpoint(run, _checkpoint(2))
        assert called
    finally:
        with suppress(AIError):
            await state.close()


@pytest.mark.asyncio
async def test_prefix_receipt_never_hides_deleted_recovery_body_from_fresh_reader(tmp_path):
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from linktools.ai.runtime import RuntimeStorage
    from linktools.ai.runtime.state._step_contracts import AgentRunCheckpoint
    from linktools.ai.storage import InMemoryObjectStore

    objects = InMemoryObjectStore()
    path = tmp_path / "deleted-prefix.db"
    state = RuntimeState(RuntimeStorage.sqlite(path, object_store=objects))
    fresh = RuntimeState(RuntimeStorage.sqlite(path, object_store=objects))
    await state.initialize(namespace="deleted-prefix", tenant_id="tenant")
    await fresh.initialize(namespace="deleted-prefix", tenant_id="tenant")
    try:
        run = AgentRunRecord("run")
        await state.run_store.register_agent_run(run)
        checkpoint = AgentRunCheckpoint("run", 1, [ModelRequest(parts=[UserPromptPart("body " * 10000)])])
        await state.run_store.save_checkpoint(checkpoint)
        warm = state.run_store.read_store(RuntimeDomain.CHECKPOINT)
        cold = fresh.run_store.read_store(RuntimeDomain.CHECKPOINT)
        assert "run" in warm._checkpoint_prefix_receipts
        assert not cold._checkpoint_prefix_receipts
        refs = [ref async for ref in objects.list_objects()]
        assert refs
        for ref in refs:
            await objects.delete_object(ref.key, expected_digest=ref.digest)
        for archive in (warm, cold):
            with pytest.raises(AIError) as caught:
                await archive.relocate_run_checkpoint(run, checkpoint)
            assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
        with pytest.raises(AIError) as caught:
            await cold.prepare_run_checkpoint(run, checkpoint)
        assert caught.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await fresh.close()
        await state.close()
