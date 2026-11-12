#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution bindings and immutable payload bodies in the shared content archive."""

import hashlib
import json
from collections import Counter
from weakref import WeakKeyDictionary, ref
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import replace

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import PayloadPolicy, StoredPayload
from ..execution import ExecutionRecord, ExecutionTimelineRecord
from ..model import StoredUserInput
from ..agent import AgentBindingContract
from ..task import TaskBindingContract
from ._archive import archive_content_scope, archive_content_identity, ContentArchive, ContentRecord, PreparedArchive
from ._payload import decode_payload_bytes
from .codec import (_execution_binding_summary, binding_reference, decode_envelope, encode_envelope, _encode_persisted_domain,
    encode_execution_content, decode_execution_content, execution_payload_reference, decode_domain, execution_payload_slots, payload_reference, map_execution_payload_slots)
from .state import StateStorageGroup, StateStore, StateTransaction, StoredRecord, active_state_scope

BINDING_CODEC = "execution-binding"
PAYLOAD_CODEC = "execution-payload"


class _ContentReadRequired(AIError):
    """A declared Execution reference needs an unopened snapshot member."""

    def __init__(self, group: StateStorageGroup) -> None:
        super().__init__(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self.group = group


def _payloads(record: StoredRecord) -> tuple[tuple[tuple[str, ...], Mapping], ...]:
    return execution_payload_slots(decode_envelope(record.data).value["payload"])


def _map_payloads(record: StoredRecord, mapper: Callable[[Mapping], JsonValue]) -> StoredRecord:
    envelope = decode_envelope(record.data)
    return replace(record, data=encode_envelope({**envelope.value,
        "payload": map_execution_payload_slots(envelope.value["payload"], mapper)}))


def _content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> Iterator[tuple[tuple[str, str, str], ContentRecord]]:
    if record.kind != "execution":
        return
    references = [(binding_reference(_binding(record)), BINDING_CODEC)]
    references.extend((payload_reference(payload), PAYLOAD_CODEC) for _, payload in _payloads(record))
    for reference, codec in references:
        if reference is None:
            continue
        content = contents.get(reference)
        if (content is None or content.identity != reference[2] or content.codec != codec
                or content.payload is not None):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        yield reference, content


def _decode_payload(raw: bytes, identity: str) -> StoredPayload | StoredUserInput:
    if archive_content_identity(PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest()) != identity:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        value = json.loads(raw)
        if not isinstance(value, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return decode_domain(value, StoredUserInput if value.get("$dataclass") == "stored_user_input" else StoredPayload)
    except (UnicodeError, ValueError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error



def _binding(record: StoredRecord) -> object:
    try:
        payload = decode_envelope(record.data).value["payload"]
        if not isinstance(payload, Mapping) or payload.get("$dataclass") != "execution_record":
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        fields = payload["fields"]
        if not isinstance(fields, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        binding = fields["binding"]
        if not isinstance(binding, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return binding
    except (KeyError, TypeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _with_binding(record: StoredRecord, binding: JsonValue) -> StoredRecord:
    envelope = decode_envelope(record.data)
    payload = envelope.value["payload"]
    return replace(record, data=encode_envelope({**envelope.value, "payload": {
        **payload, "fields": {**payload["fields"], "binding": binding},
    }}))


def expand_binding(record: StoredRecord, raw: bytes) -> StoredRecord:
    """Restore the original wire bytes only after checking their content identity."""
    reference = binding_reference(_binding(record))
    if reference is None or archive_content_identity(BINDING_CODEC, hashlib.sha256(raw).hexdigest()) != reference[2]:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        binding = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(binding, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decoded = decode_domain(binding, AgentBindingContract | TaskBindingContract)
    if _binding(record)["summary"] != _execution_binding_summary(decoded):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _with_binding(record, binding)


def snapshot_execution_content_size(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> int:
    """Return decoded bytes retained for this execution, without hydrating them."""
    return sum(content.raw_size for _reference, content in _content(record, contents))


def expand_snapshot_execution_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord],
                                     *, namespace: str, tenant_id: str,
                                     reserve: Callable[[int], None] | None = None) -> StoredRecord:
    return _expand_execution_content(record, contents, content_scope=archive_content_scope(namespace, tenant_id),
                                     reserve=reserve)


def expand_retained_execution_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord]) -> StoredRecord:
    """Resolve retained owners across all scopes in one offline physical store."""
    return _expand_execution_content(record, contents, content_scope=None)


def _expand_execution_content(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord],
                              *, content_scope: str | None,
                              reserve: Callable[[int], None] | None = None) -> StoredRecord:
    if record.kind != "execution":
        return record
    decoded = {}
    for reference, content in _content(record, contents):
        if content_scope is not None and reference[1] != content_scope:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if reserve is not None:
            reserve(content.raw_size)
        decoded[reference] = decode_payload_bytes(content.body, codec=content.compression,
            raw_digest=content.raw_digest, raw_size=content.raw_size)
    binding = binding_reference(_binding(record))
    if binding is not None:
        record = expand_binding(record, decoded[binding])
    def expand(payload: Mapping) -> JsonValue:
        reference = payload_reference(payload)
        return payload if reference is None else _encode_persisted_domain(_decode_payload(decoded[reference], reference[2]))
    return _map_payloads(record, expand)


class ExecutionContent:
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        self._store, self._namespace, self._tenant_id = store, namespace, tenant_id
        self._content_archive: ContentArchive | None = None
        self._share_content = False
        # This cache saves immutable encoding work, never proves source existence.
        self._payload_encodings: dict[int, tuple[ref, str, bytes]] = {}
        self._binding_encodings: dict[int, tuple[ref, str, bytes]] = {}
        self._projections: dict[int, tuple[ref, StoredRecord]] = {}
        self._binding_reads: WeakKeyDictionary[StateTransaction, dict[str, tuple[ContentRecord, bytes]]] = WeakKeyDictionary()

    def enable_sharing(self) -> None:
        """Allow immutable Execution content to be prepared at its next write boundary."""
        self._share_content = True

    async def initialize(self, *, create: bool, share_content: bool = True) -> None:
        if share_content:
            self.enable_sharing()
        if self._content_archive is not None:
            return
        group = self._store.storage_group
        if not isinstance(group, StateStorageGroup):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        active = active_state_scope()
        if active is not None:
            opened = group.opened_archive
            if opened is None:
                return
            if active.group is not group or opened[1] not in active.transactions:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        else:
            opened = group.opened_archive or await group.open_archive(create=create)
        if opened is None:
            return
        scope, store = opened
        # Immutable Execution bodies stay in the database for atomic publication
        # and complete offline snapshot validation before state is admitted.
        self._content_archive = ContentArchive(store, scope, namespace=self._namespace,
                                               tenant_id=self._tenant_id, object_store=None)

    async def _archive(self, *, create: bool = False) -> ContentArchive:
        active = active_state_scope()
        group = self._store.storage_group
        if not create and active is not None and not active.writable and active.group is group:
            opened = group.opened_archive
            if opened is None or opened[1] not in active.transactions:
                raise _ContentReadRequired(group)
        await self.initialize(create=create, share_content=False)
        if self._content_archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        active = active_state_scope()
        if active is not None and (active.group is not self._store.storage_group
                                   or self._content_archive.store not in active.transactions):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._content_archive

    async def expand(self, record: StoredRecord) -> StoredRecord:
        return (await self.expand_many((record,)))[0]

    async def expand_bindings(self, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
        return await self._expand_bindings(records)

    async def _expand_bindings(self, records: Sequence[StoredRecord], *,
                          reserve: Callable[[int], None] | None = None) -> tuple[StoredRecord, ...]:
        references = tuple(binding_reference(_binding(record)) for record in records)
        if not any(reference is not None for reference in references):
            return tuple(records)
        archive = await self._archive()
        if any(reference is not None and reference[:2] != (archive.scope_id, archive.content_scope)
               for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identities = tuple(dict.fromkeys(reference[2] for reference in references if reference is not None))
        counts = Counter(reference[2] for reference in references if reference is not None)
        def reserve_content(identity: str, size: int) -> None:
            if reserve is not None:
                reserve(counts[identity] * size)
        active = active_state_scope()
        transaction = None if active is None else active.transactions.get(archive.store)
        cached = None
        if transaction is not None:
            try:
                cached = self._binding_reads.setdefault(transaction, {})
            except TypeError:
                # Custom transaction objects need not support weak references.
                pass
        if cached is None:
            contents = dict(zip(identities, await archive.read_contents(
                identities, codec=BINDING_CODEC, reserve=reserve_content if reserve is not None else None,
            )))
        else:
            # Cached bytes never stand in for source existence or integrity.
            # Recheck current metadata even within a writable transaction, where
            # an earlier read may precede deletion or replacement of content.
            records_by_id = dict(zip(identities, await archive.content_records(identities)))
            contents = {}
            missing = []
            for identity, content in records_by_id.items():
                if content.codec != BINDING_CODEC or content.payload is not None:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                reserve_content(identity, content.raw_size)
                previous = cached.get(identity)
                if previous is not None and previous[0] == content:
                    contents[identity] = previous[1]
                else:
                    missing.append(identity)
            raws = await archive.read_contents(missing, codec=BINDING_CODEC) if missing else ()
            for identity, raw in zip(missing, raws, strict=True):
                contents[identity] = raw
                cached[identity] = records_by_id[identity], raw
        return tuple(record if reference is None else expand_binding(record, contents[reference[2]])
                     for record, reference in zip(records, references))


    def remember_projection(self, value: ExecutionRecord, record: StoredRecord) -> None:
        key = id(value)
        existing = self._projections.get(key)
        if existing is None or existing[0]() is not value:
            # A later uncommitted representation must not replace this value's
            # original source projection. New control values get their own entry.
            self._projections[key] = (ref(value, lambda _value: self._projections.pop(key, None)), record)

    def projection(self, value: ExecutionRecord) -> StoredRecord | None:
        cached = self._projections.get(id(value))
        return cached[1] if cached is not None and cached[0]() is value else None

    @staticmethod
    def payload_values(value: ExecutionRecord) -> tuple[StoredPayload | StoredUserInput, ...]:
        values = [value.stored_user_input]
        if value.result is not None and value.result.output is not None:
            values.append(value.result.output)
        recovery = value.recovery
        handoff = None if recovery is None else recovery.terminal_handoff
        if handoff is not None and handoff.outcome.output is not None:
            values.append(handoff.outcome.output)
        return tuple(values)

    def _encoding(self, payload: StoredPayload | StoredUserInput) -> tuple[str, bytes] | None:
        key = id(payload)
        cached = self._payload_encodings.get(key)
        if cached is not None and cached[0]() is payload:
            return cached[1:]
        body = payload.payload if isinstance(payload, StoredUserInput) else payload
        if body.kind != "inline" or (body.size <= PayloadPolicy().inline_limit_bytes
                and not (isinstance(payload, StoredUserInput) and payload.view is not None)):
            return None
        from .codec import encode_domain
        raw = canonical_json_bytes(encode_domain(payload))
        if len(raw) <= PayloadPolicy().inline_limit_bytes:
            return None
        identity = archive_content_identity(PAYLOAD_CODEC, hashlib.sha256(raw).hexdigest())
        self._payload_encodings[key] = (ref(payload, lambda _value: self._payload_encodings.pop(key, None)), identity, raw)
        return identity, raw

    def _binding_encoding(self, binding: AgentBindingContract | TaskBindingContract) -> tuple[str, bytes]:
        key = id(binding)
        cached = self._binding_encodings.get(key)
        if cached is not None and cached[0]() is binding:
            return cached[1:]
        raw = canonical_json_bytes(_encode_persisted_domain(binding))
        identity = archive_content_identity(BINDING_CODEC, hashlib.sha256(raw).hexdigest())
        self._binding_encodings[key] = (ref(binding, lambda _value: self._binding_encodings.pop(key, None)), identity, raw)
        return identity, raw

    async def prepare_value(self, value: ExecutionRecord) -> None:
        if self._share_content:
            await self.initialize(create=True)

    async def encode(self, value: ExecutionRecord, *, previous: StoredRecord | None = None) -> JsonValue:
        """Publish new immutable bodies with the same transaction as their owner."""
        if not self._share_content:
            return _encode_persisted_domain(value)
        selected = [(payload, self._encoding(payload)) for payload in self.payload_values(value)]
        selected = [(payload, encoded) for payload, encoded in selected if encoded is not None]
        binding = self._binding_encoding(value.binding)
        active = active_state_scope()
        opened = self._store.storage_group.opened_archive
        if active is not None and (opened is None or opened[1] not in active.transactions):
            return _encode_persisted_domain(value)
        archive = await self._archive(create=True)
        retained = {reference for _, payload in _payloads(previous)
                    if (reference := payload_reference(payload)) is not None} if previous is not None else set()
        if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in retained):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        previous_binding = None if previous is None else binding_reference(_binding(previous))
        if previous_binding is not None and previous_binding[:2] != (archive.scope_id, archive.content_scope):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        prepared = PreparedArchive()
        encoded_payloads = {}
        missing = []
        for payload, (identity, raw) in selected:
            reference = archive.scope_id, archive.content_scope, identity
            if reference not in retained:
                missing.append((PAYLOAD_CODEC, raw))
            encoded_payloads[id(payload)] = {"$dataclass": "stored_user_input" if isinstance(payload, StoredUserInput) else "stored_payload", "schema": 1,
                "fields": {"content": {"$execution_payload": identity,
                    "archive_scope": archive.scope_id, "content_scope": archive.content_scope}}}
        identity, raw = binding
        if previous_binding != (archive.scope_id, archive.content_scope, identity):
            missing.append((BINDING_CODEC, raw))
        encoded_binding = {"$execution_binding": identity, "archive_scope": archive.scope_id,
                           "content_scope": archive.content_scope,
                           "summary": _execution_binding_summary(value.binding)}
        if missing:
            await archive.prepare_content(tuple(missing), prepared)
            await archive.store.mutate(lambda transaction: archive.publish(transaction, prepared))
        return encode_execution_content(value, encoded_payloads, binding=encoded_binding)

    async def read_payloads(self, records: Sequence[StoredRecord], *,
                            reserve: Callable[[int], None] | None = None, result_only: bool = False) -> dict[tuple[str, str, str], StoredPayload | StoredUserInput]:
        references = [reference for record in records for path, payload in _payloads(record)
                      if (not result_only or path == ("result", "fields", "output"))
                      if (reference := payload_reference(payload)) is not None]
        if not references:
            return {}
        archive = await self._archive()
        if any(reference[:2] != (archive.scope_id, archive.content_scope) for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        counts = Counter(reference[2] for reference in references)
        identities = tuple(counts)
        def reserve_content(identity: str, size: int) -> None:
            if reserve is not None:
                reserve(counts[identity] * size)
        raws = await archive.read_contents(identities, codec=PAYLOAD_CODEC,
            reserve=reserve_content if reserve is not None else None)
        values = {}
        for identity, raw in zip(identities, raws, strict=True):
            value = _decode_payload(raw, identity)
            key = id(value)
            self._payload_encodings[key] = (ref(value, lambda _value, key=key: self._payload_encodings.pop(key, None)), identity, raw)
            values[(archive.scope_id, archive.content_scope, identity)] = value
        return values

    def terminal_matches(self, current: StoredRecord, candidate: StoredRecord, value: ExecutionRecord) -> bool:
        values = {("stored_user_input",): value.stored_user_input,
                  ("result", "fields", "output"): None if value.result is None else value.result.output}
        decoded = {reference: values[path] for path, payload in _payloads(candidate)
                   if path in values and (reference := payload_reference(payload)) is not None}
        if any(reference not in decoded for path, payload in _payloads(current)
               if path in values and (reference := payload_reference(payload)) is not None):
            return False
        previous = decode_execution_content(current.data, decoded, timeline=True)
        return previous.completed_at is None or previous.terminal_matches(value)

    async def decode(self, record: StoredRecord, *, timeline: bool = False) -> ExecutionRecord | ExecutionTimelineRecord:
        return (await self.decode_many((record,), timeline=timeline))[0]

    async def decode_many(self, records: Sequence[StoredRecord], *,
                          timeline: bool = False) -> tuple[ExecutionRecord | ExecutionTimelineRecord, ...]:
        payloads = await self.read_payloads(records)
        expanded = records if timeline else await self.expand_bindings(records)
        values = []
        for original, record in zip(records, expanded, strict=True):
            value = decode_execution_content(record.data, payloads, timeline=timeline)
            if not timeline:
                self.remember_projection(value, original)
            values.append(value)
        return tuple(values)

    async def expand_many(self, records: Sequence[StoredRecord], *,
                          reserve: Callable[[int], None] | None = None) -> tuple[StoredRecord, ...]:
        payloads = await self.read_payloads(records, reserve=reserve)
        records = await self._expand_bindings(records, reserve=reserve)
        def expand(payload: Mapping) -> JsonValue:
            reference = payload_reference(payload)
            return payload if reference is None else _encode_persisted_domain(payloads[reference])
        return tuple(_map_payloads(record, expand) for record in records)


__all__ = ["ExecutionContent", "expand_snapshot_execution_content", "expand_retained_execution_content", "snapshot_execution_content_size",
           ]
