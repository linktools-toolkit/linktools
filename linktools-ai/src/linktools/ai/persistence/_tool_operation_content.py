#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool ledger payloads and fixed Native receipt projections."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload
from ._archive import ContentArchive, ContentRecord, PreparedArchive, archive_content_identity
from ._payload import decode_payload_bytes
from ._message_root import NativeMessageRoot, NativeToolResultReference
from ._native_tool_result import restore_native_tool_result_payload
from ._step_store import SqlStepStore
from ..steps import native_store_key
from .codec import (TOOL_PAYLOAD_CODEC, decode_envelope, encode_envelope,
                    tool_payload_slots, tool_payload_reference, map_tool_payload_slots, native_tool_result_reference, _encode_persisted_domain)
from .state import StateStore, StoredRecord, active_state_scope

class ToolContentReadRequired(AIError):
    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _map_record(record: StoredRecord, mapper: Callable[[Mapping], JsonValue]) -> StoredRecord:
    envelope = decode_envelope(record.data)
    return replace(record, data=encode_envelope({**envelope.value,
        "payload": map_tool_payload_slots(envelope.value["payload"], mapper)}))


def _decode(raw: bytes, identity: str) -> Mapping:
    if archive_content_identity(TOOL_PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest()) != identity:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(raw)
        if (not isinstance(value, Mapping) or value.get("$dataclass") != "stored_payload"
                or tool_payload_reference(value) is not None):
            raise ValueError("invalid archived tool payload")
        return value
    except (UnicodeError, ValueError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def tool_content_size(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> int:
    if record.kind != "tool_operation":
        return 0
    total = 0
    for _, payload in tool_payload_slots(decode_envelope(record.data).value["payload"]):
        reference = tool_payload_reference(payload)
        if reference is not None:
            content = contents.get(reference)
            if (content is None or content.identity != reference[2]
                    or content.codec != TOOL_PAYLOAD_CODEC or content.payload is not None):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            total += content.raw_size
    return total


def expand_tool_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord], *,
                        content_scope: str | None = None, reserve: Callable[[int], None] | None = None) -> StoredRecord:
    if record.kind != "tool_operation":
        return record
    def expand(payload: Mapping) -> JsonValue:
        reference = tool_payload_reference(payload)
        if reference is None:
            return payload
        content = contents.get(reference)
        if (content is None or content.identity != reference[2] or content.codec != TOOL_PAYLOAD_CODEC
                or content.payload is not None or content_scope is not None and reference[1] != content_scope):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if reserve is not None:
            reserve(content.raw_size)
        raw = decode_payload_bytes(content.body, codec=content.compression,
                                   raw_digest=content.raw_digest, raw_size=content.raw_size)
        return _decode(raw, reference[2])
    return _map_record(record, expand)


def expand_native_tool_content(
    record: StoredRecord, payloads: Mapping[NativeToolResultReference, StoredPayload],
) -> StoredRecord:
    """Expand validated native receipt projections before ordinary domain decoding."""
    if record.kind != "tool_operation":
        return record
    def expand(payload: Mapping) -> JsonValue:
        reference = native_tool_result_reference(payload)
        if reference is None:
            return payload
        value = payloads.get(reference)
        if (not isinstance(value, StoredPayload) or value.kind != "inline" or value.encoding != "base64"
                or (value.digest, value.size) != (reference.digest, reference.size)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _encode_persisted_domain(value)
    return _map_record(record, expand)


class ToolOperationContent:
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        self.store, self.namespace, self.tenant_id = store, namespace, tenant_id

    async def _archive(self) -> ContentArchive:
        group = self.store.storage_group
        opened = group.opened_archive
        active = active_state_scope()
        if opened is None and active is None:
            opened = await group.open_archive()
        if active is not None and not active.writable and active.group is group and (opened is None or opened[1] not in active.transactions):
            raise ToolContentReadRequired()
        if opened is None or active is not None and (active.group is not group or opened[1] not in active.transactions):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ContentArchive(opened[1], opened[0], namespace=self.namespace, tenant_id=self.tenant_id, object_store=None)

    async def encode(self, record: StoredRecord, *, previous: StoredRecord | None = None) -> StoredRecord:
        native_retained = {} if previous is None else {name: reference
            for name, payload in tool_payload_slots(decode_envelope(previous.data).value["payload"])
            if (reference := native_tool_result_reference(payload)) is not None}
        retained_fields = {}
        for name, payload in tool_payload_slots(decode_envelope(record.data).value["payload"]):
            fields = payload.get("fields", {})
            reference = native_retained.get(name)
            if (reference is not None and fields.get("kind") == "inline" and fields.get("encoding") == "base64"
                    and (fields.get("digest"), fields.get("size")) == (reference.digest, reference.size)):
                retained_fields[name] = {"$dataclass": "stored_payload", "schema": 1,
                                         "fields": {"content": reference.to_json()}}
        if not retained_fields:
            return record
        envelope = decode_envelope(record.data)
        value = envelope.value["payload"]
        return replace(record, data=encode_envelope({**envelope.value, "payload": {
            **value, "fields": {**value["fields"], **retained_fields}}}))

    async def capture_native_sources(self, roots: Sequence[NativeMessageRoot]) -> dict[NativeMessageRoot, tuple[bytes, ...]]:
        """Read fixed published messages inside the same owner write transaction."""
        archive = await self._archive()
        roots = tuple(dict.fromkeys(roots))
        if any(root.group_id != archive.scope_id or root.store_key != native_store_key(self.namespace, self.tenant_id)
               for root in roots):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native = SqlStepStore.for_state_store(self.store, namespace=self.namespace, tenant_id=self.tenant_id)
        active = active_state_scope()
        transaction = None if active is None or not active.writable else active.transactions.get(self.store)
        if native is None or transaction is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native.bind_message_archive(archive)
        values = await (await native.capture_message_roots(roots, transaction=transaction))()
        return dict(zip(roots, values, strict=True))

    async def read_native_payloads(
        self, records: Sequence[StoredRecord], *, reserve: Callable[[int], None] | None = None,
    ) -> dict[NativeToolResultReference, StoredPayload]:
        references = tuple(dict.fromkeys(reference for record in records
            for _, payload in tool_payload_slots(decode_envelope(record.data).value["payload"])
            if (reference := native_tool_result_reference(payload)) is not None))
        if not references:
            return {}
        archive = await self._archive()
        if any(reference.root.group_id != archive.scope_id
               or reference.root.store_key != native_store_key(self.namespace, self.tenant_id)
               for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native = SqlStepStore.for_state_store(self.store, namespace=self.namespace, tenant_id=self.tenant_id)
        active = active_state_scope()
        transaction = None if active is None else active.transactions.get(self.store)
        if native is None or transaction is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native.bind_message_archive(archive)
        roots = tuple(dict.fromkeys(reference.root for reference in references))
        captured = await native.capture_message_roots(roots, transaction=transaction, reserve=reserve)
        messages = dict(zip(roots, await captured(), strict=True))
        values = {}
        for reference in references:
            raw = messages[reference.root]
            if len(raw) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values[reference] = restore_native_tool_result_payload(reference, raw[0])
        return values

    async def expand(self, record: StoredRecord) -> StoredRecord:
        return (await self.expand_many((record,)))[0]

    async def expand_many(self, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
        native_payloads = await self.read_native_payloads(records)
        if native_payloads:
            records = tuple(expand_native_tool_content(record, native_payloads) for record in records)
        references = tuple(dict.fromkeys(reference for record in records
            for _, payload in tool_payload_slots(decode_envelope(record.data).value["payload"])
            if (reference := tool_payload_reference(payload)) is not None))
        if not references:
            return tuple(records)
        archive = await self._archive()
        if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        raw = await archive.read_contents(tuple(reference[2] for reference in references), codec=TOOL_PAYLOAD_CODEC)
        values = {reference: _decode(body, reference[2]) for reference, body in zip(references, raw, strict=True)}
        return tuple(_map_record(record, lambda payload: values.get(tool_payload_reference(payload), payload)) for record in records)


__all__ = ["expand_native_tool_content", "ToolContentReadRequired", "ToolOperationContent", "tool_content_size", "expand_tool_content"]
