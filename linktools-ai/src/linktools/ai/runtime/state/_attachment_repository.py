#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-domain persistence for portable attachment owners."""

import re
from dataclasses import replace

from ...core import Principal
from ...errors import AIError, ErrorCode
from ._attachments import (
    AttachmentSourceRecord,
    AttachmentUploadRecord,
    InputPrepareRecord,
    managed_attachment_path,
)
from ._plan import RuntimeDomain
from ._repositories import RepositoryBase, projected_record, replace_checked
from ._store import StateStore, StateTransaction, StoredRecord

_HEX_KEY = re.compile(r"[0-9a-f]{64}")


class AttachmentRepository(RepositoryBase):
    """Own upload, input preparation, and frozen source records."""

    def __init__(
        self,
        store: StateStore,
        *,
        namespace: str,
        tenant_id: str,
    ) -> None:
        super().__init__(
            store,
            namespace=namespace,
            tenant_id=tenant_id,
            domain=RuntimeDomain.EXECUTION,
        )

    def upload_key(self, principal: Principal, idempotency_key: str) -> str:
        if not isinstance(principal, Principal) or principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_identity(idempotency_key, "upload idempotency key")
        return self._key(
            "attachment_upload",
            [principal.principal_id, principal.kind, idempotency_key],
        ).hex()

    def prepare_key(self, scope: str, idempotency_key: str) -> str:
        _require_identity(scope, "input prepare scope")
        _require_identity(idempotency_key, "input prepare idempotency key")
        return self._key("input_prepare", [scope, idempotency_key]).hex()

    def source_key(self, execution_id: str, relative: str) -> str:
        _require_identity(execution_id, "source execution id")
        _require_identity(relative, "source relative path")
        return self._key("attachment_source", [execution_id, relative]).hex()

    async def reserve_upload(
        self,
        principal: Principal,
        idempotency_key: str,
        candidate: AttachmentUploadRecord,
    ) -> tuple[str, AttachmentUploadRecord]:
        owner_key = self.upload_key(principal, idempotency_key)
        if candidate.owner_principal != principal or candidate.status != "HELD":
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        _require_upload_path(candidate, owner_key)
        key = bytes.fromhex(owner_key)

        async def mutate(transaction: StateTransaction) -> AttachmentUploadRecord:
            stored = await transaction.get_record(key)
            if stored is not None:
                current = await self._decode_kind(
                    stored,
                    AttachmentUploadRecord,
                    "attachment_upload",
                )
                if not _same_upload_request(current, candidate):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return current
            await transaction.insert_record(
                self._stored(
                    "attachment_upload",
                    [principal.principal_id, principal.kind, idempotency_key],
                    candidate,
                    state=candidate.status,
                )
            )
            return candidate

        return owner_key, await self._store.mutate(mutate)

    async def get_upload(
        self,
        owner_key: str,
        *,
        tenant_id: str,
    ) -> AttachmentUploadRecord | None:
        _require_tenant(tenant_id, self._tenant_id)
        stored = await self._stored_by_key(owner_key, "attachment_upload")
        if stored is None:
            return None
        value = await self._decode_kind(
            stored,
            AttachmentUploadRecord,
            "attachment_upload",
        )
        if not isinstance(value, AttachmentUploadRecord):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_upload_path(value, owner_key)
        return value

    async def release_upload(
        self,
        owner_key: str,
        *,
        principal: Principal,
    ) -> AttachmentUploadRecord:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        key = _owner_bytes(owner_key)

        async def mutate(transaction: StateTransaction) -> AttachmentUploadRecord:
            stored = await transaction.get_record(key)
            if stored is None or stored.kind != "attachment_upload":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            decoded = await self._decode_kind(
                stored,
                AttachmentUploadRecord,
                "attachment_upload",
            )
            if not isinstance(decoded, AttachmentUploadRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = decoded
            _require_upload_path(current, owner_key)
            if current.owner_principal != principal:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            if current.status == "RELEASED":
                return current
            if current.status != "HELD" or current.held_content is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            next_value = replace(current, held_content=None, status="RELEASED")
            candidate = replace(
                projected_record(self, stored, next_value),
                state="RELEASED",
            )
            await replace_checked(transaction, candidate, stored.storage_version)
            return next_value

        return await self._store.mutate(mutate)

    async def reserve_prepare(
        self,
        scope: str,
        idempotency_key: str,
        candidate: InputPrepareRecord,
    ) -> tuple[str, InputPrepareRecord]:
        if candidate.status != "PREPARING":
            raise ValueError("new input preparation must be PREPARING")
        owner_key = self.prepare_key(scope, idempotency_key)
        _require_prepare_paths(candidate, owner_key)
        key = bytes.fromhex(owner_key)

        async def mutate(transaction: StateTransaction) -> InputPrepareRecord:
            stored = await transaction.get_record(key)
            if stored is not None:
                decoded = await self._decode_kind(
                    stored,
                    InputPrepareRecord,
                    "input_prepare",
                )
                if not isinstance(decoded, InputPrepareRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                current = decoded
                if (
                    current.intent_digest != candidate.intent_digest
                    or current.path_origin != candidate.path_origin
                ):
                    raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
                return current
            await transaction.insert_record(
                self._stored(
                    "input_prepare",
                    [scope, idempotency_key],
                    candidate,
                    state=candidate.status,
                )
            )
            return candidate

        return owner_key, await self._store.mutate(mutate)

    async def get_prepare(
        self,
        owner_key: str,
        *,
        tenant_id: str,
    ) -> InputPrepareRecord | None:
        _require_tenant(tenant_id, self._tenant_id)
        stored = await self._stored_by_key(owner_key, "input_prepare")
        if stored is None:
            return None
        value = await self._decode_kind(stored, InputPrepareRecord, "input_prepare")
        if not isinstance(value, InputPrepareRecord):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_prepare_paths(value, owner_key)
        return value

    async def compare_and_swap_prepare(
        self,
        owner_key: str,
        *,
        expected: InputPrepareRecord,
        next_record: InputPrepareRecord,
    ) -> InputPrepareRecord:
        _require_prepare_transition(expected, next_record)
        _require_prepare_paths(next_record, owner_key)
        key = _owner_bytes(owner_key)

        async def mutate(transaction: StateTransaction) -> InputPrepareRecord:
            stored = await transaction.get_record(key)
            if stored is None or stored.kind != "input_prepare":
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            decoded = await self._decode_kind(
                stored,
                InputPrepareRecord,
                "input_prepare",
            )
            if not isinstance(decoded, InputPrepareRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if decoded != expected:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            candidate = replace(
                projected_record(self, stored, next_record),
                state=next_record.status,
            )
            await replace_checked(transaction, candidate, stored.storage_version)
            return next_record

        return await self._store.mutate(mutate)

    async def grant_upload_slot(
        self,
        upload_owner_key: str,
        prepare_owner_key: str,
        *,
        principal: Principal,
        expected_prepare: InputPrepareRecord,
        next_prepare: InputPrepareRecord,
    ) -> InputPrepareRecord:
        if principal.tenant_id != self._tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        _require_prepare_transition(expected_prepare, next_prepare)
        _require_prepare_paths(next_prepare, prepare_owner_key)
        upload_key = _owner_bytes(upload_owner_key)
        prepare_key = _owner_bytes(prepare_owner_key)

        async def mutate(transaction: StateTransaction) -> InputPrepareRecord:
            upload_stored = await transaction.get_record(upload_key)
            if upload_stored is None or upload_stored.kind != "attachment_upload":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            upload_value = await self._decode_kind(
                upload_stored,
                AttachmentUploadRecord,
                "attachment_upload",
            )
            if not isinstance(upload_value, AttachmentUploadRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            _require_upload_path(upload_value, upload_owner_key)
            if upload_value.owner_principal != principal or upload_value.status != "HELD":
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            guarded = await transaction.guard_record(
                upload_stored.key_digest,
                expected_storage_version=upload_stored.storage_version,
            )
            if guarded is None:
                raise AIError(ErrorCode.STORAGE_CONFLICT)

            prepare_stored = await transaction.get_record(prepare_key)
            if prepare_stored is None or prepare_stored.kind != "input_prepare":
                raise AIError(ErrorCode.STORAGE_NOT_FOUND)
            prepare_value = await self._decode_kind(
                prepare_stored,
                InputPrepareRecord,
                "input_prepare",
            )
            if not isinstance(prepare_value, InputPrepareRecord):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if prepare_value != expected_prepare:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            candidate = replace(
                projected_record(self, prepare_stored, next_prepare),
                state=next_prepare.status,
            )
            await replace_checked(
                transaction,
                candidate,
                prepare_stored.storage_version,
            )
            return next_prepare

        return await self._store.mutate(mutate)

    async def freeze_source(
        self,
        candidate: AttachmentSourceRecord,
    ) -> tuple[str, AttachmentSourceRecord]:
        owner_key = self.source_key(candidate.execution_id, candidate.relative)
        _require_source_path(candidate, owner_key)
        key = bytes.fromhex(owner_key)

        async def mutate(transaction: StateTransaction) -> AttachmentSourceRecord:
            stored = await transaction.get_record(key)
            if stored is not None:
                decoded = await self._decode_kind(
                    stored,
                    AttachmentSourceRecord,
                    "attachment_source",
                )
                if not isinstance(decoded, AttachmentSourceRecord):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    decoded.execution_id != candidate.execution_id
                    or decoded.relative != candidate.relative
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return decoded
            await transaction.insert_record(
                self._stored(
                    "attachment_source",
                    [candidate.execution_id, candidate.relative],
                    candidate,
                )
            )
            return candidate

        return owner_key, await self._store.mutate(mutate)

    async def get_source(
        self,
        execution_id: str,
        relative: str,
        *,
        tenant_id: str,
    ) -> AttachmentSourceRecord | None:
        _require_tenant(tenant_id, self._tenant_id)
        owner_key = self.source_key(execution_id, relative)
        return await self.get_source_by_key(owner_key, tenant_id=tenant_id)

    async def get_source_by_key(
        self,
        owner_key: str,
        *,
        tenant_id: str,
    ) -> AttachmentSourceRecord | None:
        _require_tenant(tenant_id, self._tenant_id)
        stored = await self._stored_by_key(owner_key, "attachment_source")
        if stored is None:
            return None
        value = await self._decode_kind(
            stored,
            AttachmentSourceRecord,
            "attachment_source",
        )
        if not isinstance(value, AttachmentSourceRecord):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _require_source_path(value, owner_key)
        return value

    async def _stored_by_key(
        self,
        owner_key: str,
        kind: str,
    ) -> StoredRecord | None:
        key = _owner_bytes(owner_key)
        stored = await self._store.read(lambda transaction: transaction.get_record(key))
        if stored is not None and stored.kind != kind:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return stored

    async def _decode_kind(
        self,
        stored: StoredRecord,
        target: type,
        kind: str,
    ) -> object:
        if stored.kind != kind:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._decode(stored, target)


def _same_upload_request(
    current: AttachmentUploadRecord,
    candidate: AttachmentUploadRecord,
) -> bool:
    return (
        current.owner_principal == candidate.owner_principal
        and current.intent_digest == candidate.intent_digest
        and current.descriptor == candidate.descriptor
    )


def _require_prepare_transition(
    current: InputPrepareRecord,
    candidate: InputPrepareRecord,
) -> None:
    if (
        current.intent_digest != candidate.intent_digest
        or current.path_origin != candidate.path_origin
    ):
        raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
    allowed = {
        "PREPARING": {"PREPARING", "READY", "ABORTED"},
        "READY": {"READY", "ADOPTED", "ABORTED"},
        "ADOPTED": {"ADOPTED"},
        "ABORTED": {"ABORTED"},
    }
    if candidate.status not in allowed[current.status]:
        raise AIError(ErrorCode.STORAGE_CONFLICT)


def _require_upload_path(record: AttachmentUploadRecord, owner_key: str) -> None:
    if record.descriptor.path != managed_attachment_path("u", owner_key, 0):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _require_prepare_paths(record: InputPrepareRecord, owner_key: str) -> None:
    for item in record.slots:
        if item.entry.path != managed_attachment_path("p", owner_key, item.slot):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if record.input is not None:
        for slot, entry in enumerate(record.input.attachment_manifest):
            if entry.path != managed_attachment_path("p", owner_key, slot):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _require_source_path(record: AttachmentSourceRecord, owner_key: str) -> None:
    if record.entry.path != managed_attachment_path("e", owner_key, 0):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _owner_bytes(owner_key: str) -> bytes:
    if not isinstance(owner_key, str) or _HEX_KEY.fullmatch(owner_key) is None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return bytes.fromhex(owner_key)


def _require_identity(value: str, field: str) -> None:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} is required")


def _require_tenant(actual: str, expected: str) -> None:
    if actual != expected:
        raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)


__all__ = ["AttachmentRepository"]
