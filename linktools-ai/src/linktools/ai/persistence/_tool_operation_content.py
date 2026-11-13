#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tool ledger payloads and fixed Native receipt projections."""

import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import replace
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai_harness.step_persistence import StepEvent

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, StoredPayload
from ._archive import ContentArchive, ContentRecord, PreparedArchive, archive_content_identity, archive_scope_key
from ._archive_read import ContentReferenceReader
from ._payload import decode_payload_bytes
from ._message_root import NativeMessageRoot, NativeToolResultReference, NativeToolArgumentsReference
from ._native_tool_error import NativeToolErrorReference, prepare_native_tool_error, restore_native_tool_error_payload
from ._native_tool_result import restore_native_tool_result_payload
from ._native_tool_arguments import prepare_native_tool_arguments, restore_native_tool_arguments_payload
from ._step_store import SqlStepStore
from ..steps import native_store_key
from .codec import (TOOL_PAYLOAD_CODEC, decode_envelope, encode_envelope,
                    tool_payload_slots, tool_payload_reference, map_tool_payload_slots, native_tool_result_reference, native_tool_arguments_reference, native_tool_error_reference,
                    decode_domain, encode_domain, _encode_persisted_domain)
from .state import RecordQuery, StateStore, StoredRecord, active_state_scope

class ToolContentReadRequired(AIError):
    def __init__(self) -> None:
        super().__init__(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _map_record(record: StoredRecord, mapper: Callable[[Mapping], JsonValue]) -> StoredRecord:
    envelope = decode_envelope(record.data)
    return replace(record, data=encode_envelope({**envelope.value,
        "payload": map_tool_payload_slots(envelope.value["payload"], mapper)}))


def _decode(raw: bytes) -> Mapping:
    try:
        value = json.loads(raw)
        if not isinstance(value, Mapping) or value.get("$dataclass") != "stored_payload":
            raise ValueError("invalid archived tool payload")
        return _encode_persisted_domain(decode_domain(value, StoredPayload))
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
        return _decode(raw)
    return _map_record(record, expand)


def expand_native_tool_content(
    record: StoredRecord, payloads: Mapping[NativeToolResultReference | NativeToolArgumentsReference | NativeToolErrorReference, StoredPayload],
) -> StoredRecord:
    """Expand validated native receipt projections before ordinary domain decoding."""
    if record.kind != "tool_operation":
        return record
    def expand(payload: Mapping) -> JsonValue:
        reference = (native_tool_arguments_reference(payload) or native_tool_result_reference(payload)
                     or native_tool_error_reference(payload))
        if reference is None:
            return payload
        value = payloads.get(reference)
        attached = isinstance(reference, NativeToolResultReference) and reference.attachment_format is not None
        if (not isinstance(value, StoredPayload) or value.kind != "inline"
                or value.encoding != ("json-attachments" if attached else "base64")
                or isinstance(reference, NativeToolResultReference)
                and value.size != reference.size
                or isinstance(reference, NativeToolArgumentsReference)
                and value.digest != decode_envelope(record.data).value["payload"]["fields"].get("arguments_digest")):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if attached and (value.value["format"] != reference.attachment_format
                         or canonical_json_bytes(value.value["attachments"]) != canonical_json_bytes(reference.attachment_entries())):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _encode_persisted_domain(value)
    return _map_record(record, expand)


class ToolOperationContent:
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        self.store, self.namespace, self.tenant_id = store, namespace, tenant_id
        self._archive_object_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None

    def bind_archive_object_reader(self, reader: Callable[[ObjectRef], Awaitable[bytes]]) -> None:
        """Bind verified file reads through the owning physical archive route."""
        self._archive_object_reader = reader

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
        return ContentArchive(opened[1], opened[0], namespace=self.namespace, tenant_id=self.tenant_id, object_store=None,
                              object_reader=self._archive_object_reader)

    async def prefetch_owners(
        self, keys: Sequence[bytes], *, query: RecordQuery | None = None,
    ) -> tuple[StoredRecord, ...] | None:
        """Capture fixed payload slots with keyed owners in the current snapshot."""
        active = active_state_scope()
        if (not keys and query is None or active is None or active.writable
                or active.group is not self.store.storage_group
                or self.store not in active.transactions
                or not isinstance(active.record_reader, ContentReferenceReader)
                or self.store.storage_group.opened_archive is None):
            return
        archive = await self._archive()
        found = await active.record_reader.read_content_references(
            owner_store=self.store, owner_keys=keys, content_store=archive.store,
            content_scope_key=archive_scope_key(self.namespace, self.tenant_id),
            content_paths=tuple(("value", "payload", "fields", name, "fields", "content", "$execution_payload")
                                for name in ("arguments_payload", "result_payload", "error_payload")),
            owner_query=query,
            native_root_paths=tuple(("value", "payload", "fields", name, "fields", "content", marker)
                for name, marker in (("arguments_payload", "$native_tool_arguments"),
                                     ("result_payload", "$native_tool_result"))),
        )
        return (None if found is None else
                tuple(sorted(found[self.store].values(), key=lambda row: (row.sort_key, row.key))))

    async def encode(
        self, record: StoredRecord, *, previous: StoredRecord | None = None,
        error_source: NativeToolErrorReference | None = None,
        arguments_source: tuple[NativeMessageRoot, bytes, int] | None = None,
    ) -> StoredRecord:
        native_retained = {} if previous is None else {name: reference
            for name, payload in tool_payload_slots(decode_envelope(previous.data).value["payload"])
            if (reference := (native_tool_arguments_reference(payload) or native_tool_result_reference(payload))) is not None}
        value = decode_envelope(record.data).value["payload"]
        if error_source is None and previous is not None:
            error_source = next((reference for name, payload in tool_payload_slots(
                decode_envelope(previous.data).value["payload"])
                if name == "error_payload" and (reference := native_tool_error_reference(payload)) is not None), None)
        error_events = {} if error_source is None else await self.capture_error_sources((error_source,))
        attached_previous = (await self.read_native_payloads((previous,)) if previous is not None and any(
            isinstance(reference, NativeToolResultReference) and reference.attachment_format is not None
            for reference in native_retained.values()) else {})
        retained_fields = {}
        selected = {}
        for name, payload in tool_payload_slots(decode_envelope(record.data).value["payload"]):
            fields = payload.get("fields", {})
            if name == "arguments_payload" and arguments_source is not None:
                root, raw, part_index = arguments_source
                reference = prepare_native_tool_arguments(root, raw, part_index, decode_domain(payload, StoredPayload))
                owner = value["fields"]
                if (reference is not None and reference.tool_name == owner["tool_name"]
                        and reference.tool_call_id == owner["tool_call_id"]):
                    retained_fields[name] = {"$dataclass": "stored_payload", "schema": 1,
                                             "fields": {"content": reference.to_json()}}
                    continue
            if name == "error_payload" and error_source is not None:
                owner = value["fields"]
                selected_error = prepare_native_tool_error(error_source, error_events[error_source],
                    decode_domain(payload, StoredPayload), operation_id=owner["tool_operation_id"],
                    tool_name=owner["tool_name"], tool_call_id=owner["tool_call_id"],
                    fence=record.lease_fence, error_code=owner["error_code"])
                if selected_error is not None:
                    retained_fields[name] = {"$dataclass": "stored_payload", "schema": 1,
                                             "fields": {"content": selected_error.to_json()}}
                    continue
            reference = native_retained.get(name)
            if (reference in attached_previous and canonical_json_bytes(encode_domain(decode_domain(payload, StoredPayload)))
                    == canonical_json_bytes(encode_domain(attached_previous[reference]))):
                retained_fields[name] = {"$dataclass": "stored_payload", "schema": 1,
                                         "fields": {"content": reference.to_json()}}
                continue
            if (reference is not None
                    and (not isinstance(reference, NativeToolResultReference) or reference.attachment_format is None)
                    and fields.get("kind") == "inline" and fields.get("encoding") == "base64"
                    and (fields.get("digest") == decode_envelope(previous.data).value["payload"]["fields"]["arguments_digest"]
                         if isinstance(reference, NativeToolArgumentsReference)
                         else (fields.get("digest"), fields.get("size")) == (reference.digest, reference.size))):
                retained_fields[name] = {"$dataclass": "stored_payload", "schema": 1,
                                         "fields": {"content": reference.to_json()}}
                continue
            selected[name] = canonical_json_bytes(encode_domain(decode_domain(payload, StoredPayload)))
        if selected:
            archive = await self._archive()
            retained = set() if previous is None else {tool_payload_reference(payload)
                for _, payload in tool_payload_slots(decode_envelope(previous.data).value["payload"])}
            if any(reference is not None and reference[:2] != (archive.scope_id, archive.content_scope)
                   for reference in retained):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            prepared = PreparedArchive()
            missing = []
            for name, raw in selected.items():
                identity = archive_content_identity(TOOL_PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest())
                reference = archive.scope_id, archive.content_scope, identity
                if reference not in retained:
                    missing.append((TOOL_PAYLOAD_CODEC, raw))
                retained_fields[name] = {"$dataclass": "stored_payload", "schema": 1, "fields": {"content": {
                    "$execution_payload": identity, "archive_scope": archive.scope_id,
                    "content_scope": archive.content_scope}}}
            if missing:
                await archive.prepare_content(tuple(missing), prepared)
                await archive.store.mutate(lambda transaction: archive.publish(transaction, prepared))
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

    async def capture_error_sources(
        self, references: Sequence[NativeToolErrorReference],
    ) -> dict[NativeToolErrorReference, "StepEvent"]:
        """Capture complete immutable events in the same scope as their tool owners."""
        archive = await self._archive()
        if any(reference.group_id != archive.scope_id
               or reference.store_key != native_store_key(self.namespace, self.tenant_id)
               for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native = SqlStepStore.for_state_store(self.store, namespace=self.namespace, tenant_id=self.tenant_id)
        active = active_state_scope()
        transaction = None if active is None else active.transactions.get(self.store)
        if native is None or transaction is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        events = await native.read_fixed_events(tuple((ref.run_id, ref.replay_key) for ref in references),
                                              transaction=transaction)
        return {ref: events[(ref.run_id, ref.replay_key)] for ref in references}

    async def read_native_payloads(
        self, records: Sequence[StoredRecord], *, reserve: Callable[[int], None] | None = None,
    ) -> dict[NativeToolResultReference | NativeToolArgumentsReference | NativeToolErrorReference, StoredPayload]:
        references = tuple(dict.fromkeys(reference for record in records
            for _, payload in tool_payload_slots(decode_envelope(record.data).value["payload"])
            if (reference := (native_tool_arguments_reference(payload) or native_tool_result_reference(payload))) is not None))
        errors = tuple((reference, record, decode_envelope(record.data).value["payload"]["fields"])
            for record in records for _, payload in tool_payload_slots(decode_envelope(record.data).value["payload"])
            if (reference := native_tool_error_reference(payload)) is not None)
        values = {}
        if errors:
            events = await self.capture_error_sources(tuple(dict.fromkeys(ref for ref, _, _ in errors)))
            for reference, record, owner in errors:
                value = restore_native_tool_error_payload(reference, events[reference],
                    operation_id=owner["tool_operation_id"], tool_name=owner["tool_name"],
                    tool_call_id=owner["tool_call_id"], fence=record.lease_fence, error_code=owner["error_code"])
                if reserve is not None:
                    reserve(value.size)
                values[reference] = value
        if not references:
            return values
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
        for reference in references:
            raw = messages[reference.root]
            if len(raw) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values[reference] = (restore_native_tool_arguments_payload(reference, raw[0])
                                 if isinstance(reference, NativeToolArgumentsReference)
                                 else restore_native_tool_result_payload(reference, raw[0]))
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
        values = {reference: _decode(body) for reference, body in zip(references, raw, strict=True)}
        return tuple(_map_record(record, lambda payload: values.get(tool_payload_reference(payload), payload)) for record in records)


__all__ = ["expand_native_tool_content", "ToolContentReadRequired", "ToolOperationContent", "tool_content_size", "expand_tool_content"]
