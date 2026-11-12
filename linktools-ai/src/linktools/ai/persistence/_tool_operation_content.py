#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Immutable large tool payloads owned by the database content archive."""

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import PayloadPolicy
from ._archive import ContentArchive, ContentRecord, PreparedArchive, archive_content_identity
from ._payload import decode_payload_bytes
from .codec import (TOOL_PAYLOAD_CODEC, decode_envelope, encode_envelope,
                    tool_payload_slots, tool_payload_reference, map_tool_payload_slots)
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
        selected = {}
        for name, payload in tool_payload_slots(decode_envelope(record.data).value["payload"]):
            raw = canonical_json_bytes(payload)
            if len(raw) > PayloadPolicy().inline_limit_bytes:
                selected[name] = raw
        if not selected:
            return record
        # Standalone repositories without shared content retain their inline contract.
        opened, active = self.store.storage_group.opened_archive, active_state_scope()
        if opened is None or active is not None and opened[1] not in active.transactions:
            return record
        archive = await self._archive()
        retained = set() if previous is None else {tool_payload_reference(payload)
            for _, payload in tool_payload_slots(decode_envelope(previous.data).value["payload"])}
        references = {name: (archive.scope_id, archive.content_scope,
            archive_content_identity(TOOL_PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest())) for name, raw in selected.items()}
        missing = tuple((TOOL_PAYLOAD_CODEC, raw) for name, raw in selected.items() if references[name] not in retained)
        if missing:
            prepared = PreparedArchive()
            await archive.prepare_content(missing, prepared)
            await archive.store.mutate(lambda tx: archive.publish(tx, prepared))
        envelope = decode_envelope(record.data)
        value = envelope.value["payload"]
        fields = dict(value["fields"])
        for name, (scope, owner, identity) in references.items():
            fields[name] = {"$dataclass": "stored_payload", "schema": 1, "fields": {"content": {
                "$tool_payload": identity, "archive_scope": scope, "content_scope": owner}}}
        return replace(record, data=encode_envelope({**envelope.value, "payload": {**value, "fields": fields}}))

    async def expand(self, record: StoredRecord) -> StoredRecord:
        return (await self.expand_many((record,)))[0]

    async def expand_many(self, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
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


__all__ = ["ToolContentReadRequired", "ToolOperationContent", "tool_content_size", "expand_tool_content"]
