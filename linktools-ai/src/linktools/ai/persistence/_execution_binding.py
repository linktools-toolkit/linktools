#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Complete execution bindings owned once by the shared content archive."""

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import replace

from ..core import JsonValue, canonical_json_bytes, canonical_sha256
from ..errors import AIError, ErrorCode
from ._archive import ContentArchive, ContentRecord, PreparedArchive
from ._payload import decode_payload_bytes
from .codec import binding_reference, decode_envelope, encode_envelope
from .state import StateStorageGroup, StateStore, StoredRecord, active_state_scope

BINDING_CODEC = "execution-binding"


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
    if reference is None or canonical_sha256([BINDING_CODEC, hashlib.sha256(raw).hexdigest()]) != reference[2]:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        binding = json.loads(raw)
    except (UnicodeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(binding, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return _with_binding(record, binding)


def expand_snapshot_binding(record: StoredRecord, contents: Mapping[tuple[str, str, str], ContentRecord],
                            *, namespace: str, tenant_id: str) -> StoredRecord:
    if record.kind != "execution":
        return record
    reference = binding_reference(_binding(record))
    if reference is None:
        return record
    if reference[1] != canonical_sha256([namespace, tenant_id]):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = contents.get(reference)
    if (content is None or content.identity != reference[2] or content.codec != BINDING_CODEC
            or content.payload.ref is not None):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    encoded = content.payload.decode()
    if (not isinstance(encoded, bytes) or len(encoded) != content.payload.size
            or hashlib.sha256(encoded).hexdigest() != content.payload.digest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return expand_binding(record, decode_payload_bytes(encoded, codec=content.compression,
                          raw_digest=content.raw_digest, raw_size=content.raw_size))


class ExecutionBindingContent:
    def __init__(self, store: StateStore, *, namespace: str, tenant_id: str) -> None:
        self._store, self._namespace, self._tenant_id = store, namespace, tenant_id
        self._content_archive: ContentArchive | None = None
        self._initialized = False
        self._share_bindings = False

    async def initialize(self, *, create: bool, share_bindings: bool = True) -> None:
        if self._initialized:
            if share_bindings and self._content_archive is not None:
                self._share_bindings = True
            return
        group = self._store.storage_group
        if not isinstance(group, StateStorageGroup):
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)
        if active_state_scope() is not None:
            return
        opened = await group.open_archive(create=create)
        if opened is None:
            self._initialized = True
            return
        scope, store = opened
        # Binding bodies are shared compressed metadata. Keeping them inline also
        # permits complete offline snapshot validation before any state is admitted.
        self._content_archive = ContentArchive(store, scope, namespace=self._namespace,
                                               tenant_id=self._tenant_id, object_store=None)
        self._initialized = True
        self._share_bindings = share_bindings

    async def _archive(self, *, create: bool = False) -> ContentArchive:
        await self.initialize(create=create, share_bindings=False)
        if self._content_archive is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._content_archive

    async def compact(self, record: StoredRecord, *, previous: StoredRecord | None = None) -> StoredRecord:
        if record.kind != "execution":
            return record
        if not self._share_bindings or self._content_archive is None:
            return record
        scope = active_state_scope()
        if scope is not None:
            if self._content_archive.store not in scope.transactions:
                # A handle opened after callback entry is not enlisted in that
                # transaction. Keep the complete binding with its original owner.
                return record
        raw = canonical_json_bytes(_binding(record))
        identity = canonical_sha256([BINDING_CODEC, hashlib.sha256(raw).hexdigest()])
        if previous is not None:
            reference = binding_reference(_binding(previous))
            if reference is not None and reference[2] == identity:
                if reference[1] != canonical_sha256([self._namespace, self._tenant_id]):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return _with_binding(record, _binding(previous))
        archive = await self._archive(create=True)
        prepared = PreparedArchive()
        await archive.prepare_content(((BINDING_CODEC, raw),), prepared)
        await archive.store.mutate(lambda transaction: archive.publish(transaction, prepared))
        return _with_binding(record, {"$execution_binding": identity, "archive_scope": archive.scope_id,
                                      "content_scope": archive.content_scope})

    async def expand(self, record: StoredRecord) -> StoredRecord:
        return (await self.expand_many((record,)))[0]

    async def expand_many(self, records: Sequence[StoredRecord]) -> tuple[StoredRecord, ...]:
        references = tuple(binding_reference(_binding(record)) for record in records)
        if not any(reference is not None for reference in references):
            return tuple(records)
        archive = await self._archive()
        if any(reference is not None and reference[:2] != (archive.scope_id, archive.content_scope)
               for reference in references):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identities = tuple(dict.fromkeys(reference[2] for reference in references if reference is not None))
        contents = dict(zip(identities, await archive.read_contents(identities, codec=BINDING_CODEC)))
        return tuple(record if reference is None else expand_binding(record, contents[reference[2]])
                     for record, reference in zip(records, references))



__all__ = ["ExecutionBindingContent", "expand_snapshot_binding"]
