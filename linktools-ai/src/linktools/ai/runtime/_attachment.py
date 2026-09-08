#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime attachment upload, preparation, and release owners."""

import hashlib
import mimetypes
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, TypeAlias, cast

from pydantic_ai.messages import BinaryContent, UploadedFile, UserContent

from ..capability import WorkspaceAccess
from ..core import JsonValue, Principal, canonical_sha256, validate_idempotency_key
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef
from ._input import _encode_user_content, input_intent_digest
from ._object import RuntimeObjectKeyFactory
from .service_api import AttachmentInfo
from .state import (
    AttachmentEntry,
    AttachmentPresentation,
    AttachmentSemanticEntry,
    AttachmentUploadRecord,
    ContentRef,
    InputAttachmentPart,
    InputNativePart,
    InputPrepareRecord,
    InputPrepareSlot,
    InputSource,
    InputTextPart,
    InputV2,
    PathOrigin,
    PreparedInput,
    RuntimeDomain,
    input_v2_digest,
    managed_attachment_locator,
    managed_attachment_path,
)
from .state._attachment_repository import AttachmentRepository

if TYPE_CHECKING:
    from ..workspace import Workspace
    from .state import RuntimeState

_UserPromptInput: TypeAlias = str | Sequence[UserContent]


class DefaultAttachmentService:
    """Own application uploads without granting them to an Execution."""

    def __init__(
        self,
        repository: AttachmentRepository,
        state: "RuntimeState",
        object_key_factory: RuntimeObjectKeyFactory,
    ) -> None:
        self._repository = repository
        self._state = state
        self._object_key_factory = object_key_factory

    async def upload(
        self,
        data: bytes,
        *,
        media_type: str,
        name: str | None = None,
        principal: Principal,
        idempotency_key: str,
    ) -> AttachmentInfo:
        if not isinstance(data, bytes) or not data:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(media_type, str) or not media_type:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if name is not None and (not isinstance(name, str) or not name):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        validate_idempotency_key(idempotency_key)
        if principal.tenant_id != self._state.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)

        owner_key = self._repository.upload_key(principal, idempotency_key)
        digest = hashlib.sha256(data).hexdigest()
        size = len(data)
        path = managed_attachment_path("u", owner_key, 0)
        presentation = AttachmentPresentation(None, None)
        descriptor = AttachmentSemanticEntry(
            path,
            name,
            media_type,
            presentation,
            digest,
            size,
        )
        intent_digest = canonical_sha256(
            {
                "version": 1,
                "name": name,
                "media_type": media_type,
                "digest": digest,
                "size": size,
            }
        )

        existing = await self._repository.get_upload(
            owner_key,
            tenant_id=principal.tenant_id,
        )
        if existing is not None:
            if (
                existing.owner_principal != principal
                or existing.intent_digest != intent_digest
                or existing.descriptor != descriptor
            ):
                raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
            return _attachment_info(existing.descriptor)

        owner_scope = f"attachment-upload:{owner_key}"
        content = await _store_content(
            self._state,
            self._object_key_factory,
            data,
            tenant_id=principal.tenant_id,
            owner_scope=owner_scope,
        )
        candidate = AttachmentUploadRecord(
            1,
            principal,
            intent_digest,
            descriptor,
            content,
            "HELD",
        )
        _owner_key, current = await self._repository.reserve_upload(
            principal,
            idempotency_key,
            candidate,
        )
        if current.intent_digest != intent_digest or current.descriptor != descriptor:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
        return _attachment_info(current.descriptor)

    async def release(
        self,
        path: str,
        *,
        principal: Principal,
    ) -> None:
        try:
            kind, owner_key, slot = managed_attachment_locator(path)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if kind != "u" or slot != 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if principal.tenant_id != self._state.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        current = await self._repository.release_upload(
            owner_key,
            principal=principal,
        )
        if current.descriptor.path != path:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        await self._state._release_object_scope(
            RuntimeDomain.EXECUTION,
            owner_scope=f"attachment-upload:{owner_key}",
        )


class InputPreparer:
    """Freeze one managed SDK input before durable Execution admission."""

    def __init__(
        self,
        repository: AttachmentRepository,
        state: "RuntimeState",
        object_key_factory: RuntimeObjectKeyFactory,
        workspace: "Workspace",
    ) -> None:
        self._repository = repository
        self._state = state
        self._object_key_factory = object_key_factory
        self._workspace = workspace

    def requires_managed_input(
        self,
        user_prompt: _UserPromptInput,
        attachments: Sequence[str],
    ) -> bool:
        if tuple(attachments):
            return True
        if isinstance(user_prompt, str):
            return False
        if not isinstance(user_prompt, Sequence) or isinstance(
            user_prompt,
            (str, bytes, bytearray),
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return any(isinstance(item, BinaryContent) for item in user_prompt)

    async def prepare(
        self,
        user_prompt: _UserPromptInput,
        attachments: Sequence[str],
        *,
        principal: Principal,
        scope: str,
        idempotency_key: str,
    ) -> PreparedInput:
        validate_idempotency_key(idempotency_key)
        if principal.tenant_id != self._state.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        prompt = _prompt_items(user_prompt)
        sources = _attachment_sources(attachments)
        intent_digest = input_intent_digest(user_prompt, tuple(attachments))
        origin = _path_origin(self._workspace)
        candidate = InputPrepareRecord(
            1,
            intent_digest,
            origin,
            "PREPARING",
            (),
            None,
            None,
            None,
        )
        owner_key, current = await self._repository.reserve_prepare(
            scope,
            idempotency_key,
            candidate,
        )
        if current.status == "READY":
            if current.input is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current.input
        if current.status == "ADOPTED":
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        if current.status == "ABORTED":
            raise _stable_prepare_error(current.error_code)
        if current.status != "PREPARING":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        plan = _build_input_plan(prompt, sources)
        access: WorkspaceAccess | None = None
        owner_scope = f"input-prepare:{owner_key}"
        try:
            for slot_id, source in enumerate(plan.unique_sources):
                if slot_id < len(current.slots):
                    slot = current.slots[slot_id]
                    _verify_replayed_slot(slot, slot_id, source, owner_key)
                    continue
                if slot_id != len(current.slots):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if isinstance(source, _BinarySource):
                    entry = await self._binary_entry(
                        source,
                        owner_key=owner_key,
                        slot=slot_id,
                        owner_scope=owner_scope,
                        tenant_id=principal.tenant_id,
                    )
                    next_record = replace(
                        current,
                        slots=(*current.slots, InputPrepareSlot(slot_id, None, entry)),
                    )
                    current = await self._repository.compare_and_swap_prepare(
                        owner_key,
                        expected=current,
                        next_record=next_record,
                    )
                    continue
                if isinstance(source, _UploadSource):
                    upload = await self._repository.get_upload(
                        source.owner_key,
                        tenant_id=principal.tenant_id,
                    )
                    if (
                        upload is None
                        or upload.owner_principal != principal
                        or upload.status != "HELD"
                        or upload.held_content is None
                        or upload.descriptor.path != source.path
                    ):
                        raise AIError(ErrorCode.AUTHORIZATION_DENIED)
                    entry = AttachmentEntry(
                        managed_attachment_path("p", owner_key, slot_id),
                        upload.descriptor.name,
                        upload.descriptor.media_type,
                        upload.descriptor.presentation,
                        upload.held_content,
                    )
                    next_record = replace(
                        current,
                        slots=(*current.slots, InputPrepareSlot(slot_id, None, entry)),
                    )
                    current = await self._repository.grant_upload_slot(
                        source.owner_key,
                        owner_key,
                        principal=principal,
                        expected_prepare=current,
                        next_prepare=next_record,
                    )
                    continue
                if not isinstance(source, _PathSource):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if access is None:
                    access = WorkspaceAccess.for_workspace(self._workspace)
                body = await access.read_bytes(source.relative)
                if not body:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                content = await _store_content(
                    self._state,
                    self._object_key_factory,
                    body,
                    tenant_id=principal.tenant_id,
                    owner_scope=owner_scope,
                )
                entry = AttachmentEntry(
                    managed_attachment_path("p", owner_key, slot_id),
                    PurePosixPath(source.relative).name,
                    source.media_type,
                    AttachmentPresentation(None, None),
                    content,
                )
                next_record = replace(
                    current,
                    slots=(
                        *current.slots,
                        InputPrepareSlot(slot_id, source.relative, entry),
                    ),
                )
                current = await self._repository.compare_and_swap_prepare(
                    owner_key,
                    expected=current,
                    next_record=next_record,
                )
        finally:
            if access is not None:
                await access.close()

        manifest = tuple(slot.entry for slot in current.slots)
        input_value = _build_input_v2(plan, manifest)
        prepared = PreparedInput(
            1,
            "linktools-input-v2",
            input_value,
            manifest,
            intent_digest,
            input_v2_digest(input_value, manifest),
            origin,
        )
        ready = InputPrepareRecord(
            1,
            intent_digest,
            origin,
            "READY",
            (),
            prepared,
            None,
            None,
        )
        committed = await self._repository.compare_and_swap_prepare(
            owner_key,
            expected=current,
            next_record=ready,
        )
        if committed.input is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return committed.input

    async def _binary_entry(
        self,
        source: "_BinarySource",
        *,
        owner_key: str,
        slot: int,
        owner_scope: str,
        tenant_id: str,
    ) -> AttachmentEntry:
        if not source.data:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        content = await _store_content(
            self._state,
            self._object_key_factory,
            source.data,
            tenant_id=tenant_id,
            owner_scope=owner_scope,
        )
        return AttachmentEntry(
            managed_attachment_path("p", owner_key, slot),
            None,
            source.media_type,
            AttachmentPresentation(source.identifier, source.vendor_metadata),
            content,
        )


class _BinarySource:
    __slots__ = ("data", "media_type", "identifier", "vendor_metadata", "key")

    def __init__(
        self,
        data: bytes,
        media_type: str,
        identifier: str | None,
        vendor_metadata: Mapping[str, JsonValue] | None,
        key: str,
    ) -> None:
        self.data = data
        self.media_type = media_type
        self.identifier = identifier
        self.vendor_metadata = vendor_metadata
        self.key = key


class _UploadSource:
    __slots__ = ("path", "owner_key", "key")

    def __init__(self, path: str, owner_key: str) -> None:
        self.path = path
        self.owner_key = owner_key
        self.key = f"upload:{path}"


class _PathSource:
    __slots__ = ("relative", "media_type", "key")

    def __init__(self, relative: str, media_type: str) -> None:
        self.relative = relative
        self.media_type = media_type
        self.key = f"path:{relative}"


_Source = _BinarySource | _UploadSource | _PathSource


class _InputPlan:
    __slots__ = ("prompt_parts", "available_keys", "source_relatives", "unique_sources")

    def __init__(
        self,
        prompt_parts: tuple[InputTextPart | InputNativePart | str, ...],
        available_keys: tuple[str, ...],
        source_relatives: tuple[tuple[str, str], ...],
        unique_sources: tuple[_Source, ...],
    ) -> None:
        self.prompt_parts = prompt_parts
        self.available_keys = available_keys
        self.source_relatives = tuple(dict.fromkeys(source_relatives))
        self.unique_sources = unique_sources


def _prompt_items(
    value: _UserPromptInput,
) -> tuple[str | UserContent, ...]:
    if isinstance(value, str):
        return (value,)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result = tuple(value)
    if not result:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if any(isinstance(item, UploadedFile) for item in result):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={"field": "user_prompt", "reason": "uploaded_file_not_durable"},
        )
    return cast(tuple[str | UserContent, ...], result)


def _attachment_sources(value: Sequence[str]) -> tuple[_UploadSource | _PathSource, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result: list[_UploadSource | _PathSource] = []
    for raw in value:
        if not isinstance(raw, str) or not raw:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if raw.startswith("virtual:"):
            try:
                kind, owner_key, slot = managed_attachment_locator(raw)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            if kind != "u" or slot != 0:
                raise AIError(ErrorCode.AUTHORIZATION_DENIED)
            result.append(_UploadSource(raw, owner_key))
            continue
        if raw.startswith("file:") or "://" in raw or "\\" in raw:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        path = PurePosixPath(raw)
        if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        relative = path.as_posix()
        if relative != raw:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        media_type, _encoding = mimetypes.guess_type(relative, strict=False)
        if media_type is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "attachments", "reason": "media_type_unknown"},
            )
        result.append(_PathSource(relative, media_type))
    return tuple(result)


def _build_input_plan(
    prompt: tuple[str | UserContent, ...],
    attachments: tuple[_UploadSource | _PathSource, ...],
) -> _InputPlan:
    unique: list[_Source] = []
    key_to_index: dict[str, int] = {}
    prompt_parts: list[InputTextPart | InputNativePart | str] = []
    binary_ordinal = 0
    for item in prompt:
        if isinstance(item, str):
            prompt_parts.append(InputTextPart("text", item))
            continue
        if isinstance(item, BinaryContent):
            if (
                not isinstance(item.data, bytes)
                or not item.data
                or not isinstance(item.media_type, str)
                or not item.media_type
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            metadata = _vendor_metadata(item.vendor_metadata)
            key = f"binary:{binary_ordinal}"
            binary_ordinal += 1
            source = _BinarySource(
                item.data,
                item.media_type,
                item.identifier,
                metadata,
                key,
            )
            key_to_index[key] = len(unique)
            unique.append(source)
            prompt_parts.append(key)
            continue
        payload = _encode_user_content((item,))
        prompt_parts.append(
            InputNativePart(
                "native",
                "pydantic-user-content-v1",
                payload,
            )
        )

    available_keys: list[str] = []
    source_relatives: list[tuple[str, str]] = []
    for source in attachments:
        index = key_to_index.get(source.key)
        if index is None:
            index = len(unique)
            key_to_index[source.key] = index
            unique.append(source)
        if source.key not in available_keys:
            available_keys.append(source.key)
        if isinstance(source, _PathSource):
            source_relatives.append((source.relative, source.key))
    source_relatives.sort(key=lambda item: item[0])
    return _InputPlan(
        tuple(prompt_parts),
        tuple(available_keys),
        tuple(source_relatives),
        tuple(unique),
    )


def _build_input_v2(
    plan: _InputPlan,
    manifest: tuple[AttachmentEntry, ...],
) -> InputV2:
    key_to_index = {source.key: index for index, source in enumerate(plan.unique_sources)}
    parts: list[InputTextPart | InputNativePart | InputAttachmentPart] = []
    for part in plan.prompt_parts:
        if isinstance(part, str):
            parts.append(InputAttachmentPart("attachment", key_to_index[part]))
        else:
            parts.append(part)
    available = tuple(key_to_index[key] for key in plan.available_keys)
    sources = tuple(
        InputSource(relative, key_to_index[key])
        for relative, key in plan.source_relatives
    )
    if len(manifest) != len(plan.unique_sources):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return InputV2(2, tuple(parts), available, sources)


def _verify_replayed_slot(
    slot: InputPrepareSlot,
    expected_slot: int,
    source: _Source,
    owner_key: str,
) -> None:
    if slot.slot != expected_slot:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if slot.entry.path != managed_attachment_path("p", owner_key, expected_slot):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if isinstance(source, _PathSource):
        if slot.relative != source.relative or slot.entry.media_type != source.media_type:
            raise AIError(ErrorCode.IDEMPOTENCY_CONFLICT)
    elif slot.relative is not None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _path_origin(workspace: "Workspace") -> PathOrigin:
    return PathOrigin(
        1,
        workspace.workspace_id,
        "windows" if os.name == "nt" else "posix",
        str(workspace.root),
    )


def _vendor_metadata(value: object) -> Mapping[str, JsonValue] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return cast(Mapping[str, JsonValue], value)


async def _store_content(
    state: "RuntimeState",
    factory: RuntimeObjectKeyFactory,
    data: bytes,
    *,
    tenant_id: str,
    owner_scope: str,
) -> ContentRef:
    if not data:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    digest = hashlib.sha256(data).hexdigest()
    store = state.working_object_store(
        RuntimeDomain.EXECUTION,
        owner_scope=owner_scope,
    )
    key = factory.key(RuntimeDomain.EXECUTION, tenant_id, digest)
    stat = await store.stat(key)
    if stat is not None:
        if stat.digest != digest or stat.size != len(data):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        reference = ObjectRef(store.store_id, key, stat.digest, stat.size)
    else:
        async def chunks():
            yield data

        created = await store.put(
            key,
            chunks(),
            expected_size=len(data),
            expected_digest=digest,
        )
        reference = ObjectRef(store.store_id, key, created.digest, created.size)
    return ContentRef(RuntimeDomain.EXECUTION.value, owner_scope, reference)


def _stable_prepare_error(error_code: str | None) -> AIError:
    if error_code is None:
        return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return AIError(ErrorCode(error_code))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _attachment_info(value: AttachmentSemanticEntry) -> AttachmentInfo:
    return AttachmentInfo(
        value.path,
        value.name,
        value.media_type,
        value.size,
    )


__all__ = ["DefaultAttachmentService", "InputPreparer"]
