#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pytest

from linktools.ai.core import Principal
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._attachment import DefaultAttachmentService
from linktools.ai.runtime._object import RuntimeObjectKeyFactory
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentSourceRecord,
    InputPrepareRecord,
    InputPrepareSlot,
    PathOrigin,
    RuntimeState,
    managed_attachment_path,
)
from linktools.ai.runtime.state._attachment_repository import AttachmentRepository


async def _runtime() -> tuple[RuntimeState, AttachmentRepository, DefaultAttachmentService]:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="workspace", tenant_id="tenant")
    repository = AttachmentRepository(
        state.execution.executions.state_store,
        namespace="workspace",
        tenant_id="tenant",
    )
    service = DefaultAttachmentService(
        repository,
        state,
        RuntimeObjectKeyFactory("workspace"),
    )
    return state, repository, service


@pytest.mark.asyncio
async def test_upload_release_replay_keeps_released_tombstone() -> None:
    state, repository, service = await _runtime()
    principal = Principal("user", "tenant", "local_trusted")
    key = "upload-key-" + "a" * 24
    try:
        first = await service.upload(
            b"body",
            media_type="application/octet-stream",
            name="evidence.bin",
            principal=principal,
            idempotency_key=key,
        )
        kind, owner_key, slot = __import__(
            "linktools.ai.runtime.state",
            fromlist=["managed_attachment_locator"],
        ).managed_attachment_locator(first.path)
        assert (kind, slot) == ("u", 0)
        held = await repository.get_upload(owner_key, tenant_id="tenant")
        assert held is not None and held.status == "HELD" and held.held_content is not None

        await service.release(first.path, principal=principal)
        released = await repository.get_upload(owner_key, tenant_id="tenant")
        assert released is not None
        assert released.status == "RELEASED"
        assert released.held_content is None

        replayed = await service.upload(
            b"body",
            media_type="application/octet-stream",
            name="evidence.bin",
            principal=principal,
            idempotency_key=key,
        )
        assert replayed == first
        current = await repository.get_upload(owner_key, tenant_id="tenant")
        assert current is not None
        assert current.status == "RELEASED"
        assert current.held_content is None
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_release_rejects_non_upload_and_hides_unknown_upload() -> None:
    state, _repository, service = await _runtime()
    principal = Principal("user", "tenant", "local_trusted")
    try:
        with pytest.raises(AIError) as wrong_kind:
            await service.release(
                managed_attachment_path("p", "a" * 64, 0),
                principal=principal,
            )
        assert wrong_kind.value.code is ErrorCode.REQUEST_FIELD_INVALID

        with pytest.raises(AIError) as unknown:
            await service.release(
                managed_attachment_path("u", "b" * 64, 0),
                principal=principal,
            )
        assert unknown.value.code is ErrorCode.AUTHORIZATION_DENIED
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_granted_upload_survives_upload_release_in_prepare_hold() -> None:
    state, repository, service = await _runtime()
    principal = Principal("user", "tenant", "local_trusted")
    upload_key = "upload-key-" + "c" * 24
    prepare_key = "prepare-key-" + "d" * 23
    try:
        uploaded = await service.upload(
            b"body",
            media_type="application/octet-stream",
            name="evidence.bin",
            principal=principal,
            idempotency_key=upload_key,
        )
        _kind, upload_owner, _slot = __import__(
            "linktools.ai.runtime.state",
            fromlist=["managed_attachment_locator"],
        ).managed_attachment_locator(uploaded.path)
        held = await repository.get_upload(upload_owner, tenant_id="tenant")
        assert held is not None and held.held_content is not None

        initial = InputPrepareRecord(
            1,
            "1" * 64,
            PathOrigin(1, "workspace", "posix", "/workspace"),
            "PREPARING",
            (),
            None,
            None,
            None,
        )
        prepare_owner, current = await repository.reserve_prepare(
            "execution.run",
            prepare_key,
            initial,
        )
        entry = AttachmentEntry(
            managed_attachment_path("p", prepare_owner, 0),
            held.descriptor.name,
            held.descriptor.media_type,
            held.descriptor.presentation,
            held.held_content,
        )
        next_prepare = InputPrepareRecord(
            1,
            current.intent_digest,
            current.path_origin,
            "PREPARING",
            (InputPrepareSlot(0, None, entry),),
            None,
            None,
            None,
        )
        granted = await repository.grant_upload_slot(
            upload_owner,
            prepare_owner,
            principal=principal,
            expected_prepare=current,
            next_prepare=next_prepare,
        )
        assert granted.slots == next_prepare.slots

        await service.release(uploaded.path, principal=principal)
        released = await repository.get_upload(upload_owner, tenant_id="tenant")
        assert released is not None and released.held_content is None
        retained = await repository.get_prepare(prepare_owner, tenant_id="tenant")
        assert retained is not None
        assert retained.slots[0].entry.content == entry.content
    finally:
        await state.close()


@pytest.mark.asyncio
async def test_source_freeze_requires_existing_execution() -> None:
    state, repository, service = await _runtime()
    principal = Principal("user", "tenant", "local_trusted")
    try:
        uploaded = await service.upload(
            b"body",
            media_type="application/octet-stream",
            principal=principal,
            idempotency_key="upload-key-" + "e" * 24,
        )
        _kind, upload_owner, _slot = __import__(
            "linktools.ai.runtime.state",
            fromlist=["managed_attachment_locator"],
        ).managed_attachment_locator(uploaded.path)
        held = await repository.get_upload(upload_owner, tenant_id="tenant")
        assert held is not None and held.held_content is not None

        source_owner = repository.source_key("missing-execution", "evidence.bin")
        source = AttachmentSourceRecord(
            1,
            "missing-execution",
            "evidence.bin",
            AttachmentEntry(
                managed_attachment_path("e", source_owner, 0),
                held.descriptor.name,
                held.descriptor.media_type,
                held.descriptor.presentation,
                held.held_content,
            ),
        )
        with pytest.raises(AIError) as raised:
            await repository.freeze_source(source)
        assert raised.value.code is ErrorCode.STORAGE_NOT_FOUND
    finally:
        await state.close()
