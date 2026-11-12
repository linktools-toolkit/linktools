#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group-owned immutable content and exact persistent sequence ranges."""

import base64
import binascii
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from weakref import WeakKeyDictionary

from ..core import JsonValue, canonical_json_bytes, canonical_sha256, validate_persistence_namespace, validate_tenant_id
from ..errors import AIError, ErrorCode
from ..model.attachments import read_attachment_payload, store_attachment_payload, validate_attachment_payload
from ..storage import ObjectRef, ObjectStore, StoredPayload, read_object
from ._archive_read import ArchiveFrontierReader, ArchiveNodeRead, ArchiveNodeSlot
from ._record_key import decode_key_components, encode_key_components
from ._payload import decode_payload_bytes, encode_payload_bytes
from .state import RECORD_KEY_MAX_BYTES, StateStore, StateTransaction, StoredRecord, record_key, active_state_transaction


if TYPE_CHECKING:
    from ._message_root import NativeMessageSequence
    from ._step_message_content import NativeMessageContent, PreparedNativeMessages


def _integrity() -> AIError:
    return AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _digest(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise _integrity()
    try:
        if bytes.fromhex(value).hex() != value:
            raise _integrity()
    except ValueError as error:
        raise _integrity() from error
    return value


def validate_archive_group_id(value: str) -> str:
    """Keep an exact, NUL-free storage-group identity of at most 128 UTF-8 bytes."""
    try:
        if not isinstance(value, str) or "\0" in value or not 1 <= len(value.encode("utf-8")) <= 128:
            raise _integrity()
    except UnicodeError as error:
        raise _integrity() from error
    return value


def archive_scope_key(namespace: str, tenant_id: str) -> bytes:
    """Encode the physical archive partition directly from its exact owners."""
    return encode_key_components(("archive_scope", validate_persistence_namespace(namespace),
                                  validate_tenant_id(tenant_id)), 403)


def archive_scope_parts(value: bytes) -> tuple[str, str]:
    """Validate persisted archive scope bytes and recover their exact owners."""
    try:
        parts = decode_key_components(value, 403)
        if len(parts) != 3 or parts[0] != "archive_scope":
            raise _integrity()
        namespace, tenant_id = parts[1:]
        if archive_scope_key(namespace, tenant_id) != value:
            raise _integrity()
        return namespace, tenant_id
    except (AIError, TypeError, ValueError, UnicodeError) as error:
        raise _integrity() from error


def archive_scope_token(value: bytes) -> str:
    archive_scope_parts(value)
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def decode_archive_scope(value: str) -> bytes:
    """Accept only the single canonical unpadded URL-safe scope transport."""
    try:
        if not isinstance(value, str) or not 1 <= len(value) <= 538:
            raise _integrity()
        raw = base64.b64decode(value + "=" * (-len(value) % 4), altchars=b"-_", validate=True)
        if archive_scope_token(raw) != value:
            raise _integrity()
        return raw
    except (AIError, TypeError, ValueError, UnicodeError, binascii.Error) as error:
        raise _integrity() from error


def archive_content_scope(namespace: str, tenant_id: str) -> str:
    return archive_scope_token(archive_scope_key(namespace, tenant_id))


def _content_codec(value: str) -> str:
    try:
        if not isinstance(value, str) or "\0" in value or not 1 <= len(value.encode("utf-8")) <= 64:
            raise _integrity()
    except UnicodeError as error:
        raise _integrity() from error
    return value


def archive_content_identity(codec: str, raw_digest: str) -> str:
    return _content_codec(codec) + ":" + _digest(raw_digest)


def validate_archive_content_identity(value: str) -> str:
    if not isinstance(value, str) or len(value) < 66 or value[-65] != ":":
        raise _integrity()
    return archive_content_identity(value[:-65], value[-64:])


def _count(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise _integrity()
    return value


@dataclass(frozen=True, slots=True)
class ArchiveSlice:
    first: str
    count: int
    start: int
    end: int

    def __post_init__(self) -> None:
        _digest(self.first)
        if not 0 <= _count(self.start) < _count(self.end) <= _count(self.count):
            raise _integrity()

    def to_json(self) -> dict[str, JsonValue]:
        return {"first": self.first, "count": self.count, "start": self.start, "end": self.end}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "ArchiveSlice":
        try:
            return cls(_digest(value["first"]), _count(value["count"]), _count(value["start"]), _count(value["end"]))
        except (KeyError, TypeError) as error:
            raise _integrity() from error


@dataclass(frozen=True, slots=True)
class ArchiveRoot:
    scope_id: str
    content_scope: str = field(repr=False)
    slices: tuple[ArchiveSlice, ...] = ()

    def __post_init__(self) -> None:
        validate_archive_group_id(self.scope_id)
        decode_archive_scope(self.content_scope)

    @property
    def count(self) -> int:
        return sum(item.end - item.start for item in self.slices)

    def to_json(self) -> dict[str, JsonValue]:
        return {"archive_scope": self.scope_id, "content_scope": self.content_scope, "slices": [item.to_json() for item in self.slices]}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "ArchiveRoot":
        try:
            scope, slices = value["archive_scope"], value["slices"]
            if not isinstance(scope, str) or not isinstance(slices, list):
                raise _integrity()
            if not all(isinstance(item, Mapping) for item in slices):
                raise _integrity()
            return cls(scope, value["content_scope"], tuple(ArchiveSlice.from_json(item) for item in slices))
        except (KeyError, TypeError) as error:
            raise _integrity() from error

    def slice(self, start: int, end: int) -> "ArchiveRoot":
        if not 0 <= _count(start) <= _count(end) <= self.count:
            raise _integrity()
        result: list[ArchiveSlice] = []
        offset = 0
        for item in self.slices:
            size = item.end - item.start
            lo, hi = max(0, start - offset), min(size, end - offset)
            if hi > lo:
                result.append(ArchiveSlice(item.first, item.count, item.start + lo, item.start + hi))
            offset += size
            if offset >= end:
                break
        return ArchiveRoot(self.scope_id, self.content_scope, tuple(result))


@dataclass(frozen=True, slots=True)
class ImmutableSequenceNode:
    content_id: str
    tree_count: int = 1
    left: str | None = None
    right: str | None = None
    next_forest: str | None = None

    def __post_init__(self) -> None:
        # Native message nodes retain their raw byte checksum as content identity.
        if isinstance(self.content_id, str) and len(self.content_id) == 64:
            _digest(self.content_id)
        else:
            validate_archive_content_identity(self.content_id)
        if _count(self.tree_count) < 1 or self.tree_count & (self.tree_count + 1):
            raise _integrity()
        for value in (self.left, self.right, self.next_forest):
            if value is not None:
                _digest(value)
        if (self.left is None) != (self.right is None):
            raise _integrity()
        if (self.tree_count == 1) != (self.left is None):
            raise _integrity()

    @classmethod
    def forest_weights(cls, count: int) -> tuple[int, ...]:
        """Return the bounded tree sizes of a canonical occurrence sequence."""
        count = _count(count)
        values: list[int] = []
        while count:
            size = (1 << ((count + 1).bit_length() - 1)) - 1
            values.append(size)
            count -= size
        return tuple(reversed(values))

    @classmethod
    def extend(
        cls, content_id: str, first: "ImmutableSequenceNode | None",
        second: "ImmutableSequenceNode | None",
    ) -> "ImmutableSequenceNode":
        """Append one occurrence using the captured immutable forest frontier."""
        if first is not None and second is not None and first.tree_count == second.tree_count:
            return cls(content_id, 1 + first.tree_count + second.tree_count,
                       first.identity, second.identity, second.next_forest)
        return cls(content_id, next_forest=None if first is None else first.identity)

    def to_json(self) -> dict[str, JsonValue]:
        return {"content": self.content_id, "count": self.tree_count,
                "left": self.left, "right": self.right, "next": self.next_forest}

    @property
    def identity(self) -> str:
        return canonical_sha256(self.to_json())

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "ImmutableSequenceNode":
        try:
            return cls(validate_archive_content_identity(value["content"]), _count(value["count"]),
                       None if value["left"] is None else _digest(value["left"]),
                       None if value["right"] is None else _digest(value["right"]),
                       None if value["next"] is None else _digest(value["next"]))
        except (KeyError, TypeError) as error:
            raise _integrity() from error


@dataclass(frozen=True, slots=True)
class ContentRecord:
    codec: str
    raw_digest: str
    raw_size: int
    compression: str
    payload: StoredPayload | None = None
    body: bytes | None = None

    def __post_init__(self) -> None:
        _content_codec(self.codec)
        if self.compression not in {"raw", "zlib"}:
            raise _integrity()
        _digest(self.raw_digest)
        _count(self.raw_size)
        if self.payload is None:
            if type(self.body) is not bytes:
                raise _integrity()
        elif (self.body is not None or self.payload.encoding != "json-attachments"
              or self.compression != "raw" or self.payload.digest != self.raw_digest
              or self.payload.size != self.raw_size):
            raise _integrity()

    @property
    def identity(self) -> str:
        return archive_content_identity(self.codec, self.raw_digest)

    def object_refs(self) -> tuple[ObjectRef, ...]:
        return () if self.payload is None else self.payload.object_refs()

    def to_json(self) -> dict[str, JsonValue]:
        return {"codec": self.codec, "raw_digest": self.raw_digest, "raw_size": self.raw_size,
                "compression": self.compression, "payload": None if self.payload is None else self.payload.to_json()}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue], *, body: bytes | None = None) -> "ContentRecord":
        try:
            payload = value["payload"]
            return cls(value["codec"], _digest(value["raw_digest"]), _count(value["raw_size"]),
                       value["compression"], None if payload is None else StoredPayload.from_json(dict(payload)), body)
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity() from error

    @classmethod
    def from_record(cls, record: StoredRecord) -> "ContentRecord":
        namespace, tenant_id = archive_scope_parts(record.scope_key)
        content = cls.from_json(record.data, body=record.payload_bytes)
        if (record.kind != "archive_content" or record.sort_key != content.identity.encode("utf-8")
                or record.key != record_key(namespace, tenant_id, "archive", record.kind, (content.identity,))):
            raise _integrity()
        return content

    def to_native_values(self) -> dict[str, object]:
        return {"message": self.body, "compression": self.compression, "raw_size": self.raw_size,
                "payload_json": None if self.payload is None else canonical_json_bytes(self.payload.to_json()).decode("utf-8")}

    @classmethod
    def from_native_row(cls, row: Mapping[str, object]) -> "ContentRecord":
        try:
            payload = row["payload_json"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            return cls("model-message", row.get("message_sha256", row.get("sha256")), row["raw_size"],
                       row["compression"], None if payload is None else StoredPayload.from_json(payload), row["message"])
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity() from error


@dataclass(slots=True)
class PreparedArchive:
    """Publication records and immutable nodes known during one preparation."""

    owner: tuple[str, str, str] | None = None
    records: dict[bytes, StoredRecord] = field(default_factory=dict)
    nodes: dict[str, ImmutableSequenceNode] = field(default_factory=dict)
    native_messages: dict["NativeMessageSequence", "PreparedNativeMessages"] = field(default_factory=dict)
    native_publishers: dict["NativeMessageSequence", Callable[[StateTransaction, "PreparedArchive"], Awaitable[None]]] = field(default_factory=dict, repr=False, compare=False)
    # Uncommitted representation reuse belongs only to this preparation and owner epoch.
    native_payloads: dict["NativeMessageContent", tuple[int, dict[str, ContentRecord]]] = field(default_factory=dict, repr=False, compare=False)
    _verified_content: dict[bytes, StoredRecord] = field(default_factory=dict, repr=False, compare=False)
    _observed_content: dict[bytes, StoredRecord | None] = field(default_factory=dict, repr=False, compare=False)

    def content_verified(self, record: StoredRecord) -> bool:
        return self._verified_content.get(record.key) is record

    def verify_content(self, record: StoredRecord) -> None:
        self._verified_content[record.key] = record

    def clear_verified_content(self) -> None:
        self._verified_content.clear()

    def clear_content_observation(self) -> None:
        self._observed_content.clear()

    @classmethod
    def combine(cls, values: Sequence["PreparedArchive"]) -> "PreparedArchive":
        combined = cls()
        visited: set[int] = set()
        for value in values:
            if id(value) in visited:
                continue
            visited.add(id(value))
            if value.owner is not None:
                if combined.owner is not None and combined.owner != value.owner:
                    raise _integrity()
                combined.owner = value.owner
            combined.native_messages.update(value.native_messages)
            combined.native_publishers.update(value.native_publishers)
            for key, record in value.records.items():
                previous = combined.records.get(key)
                if previous is not None and previous != record:
                    raise _integrity()
                combined.records[key] = record
            for key, node in value.nodes.items():
                previous_node = combined.nodes.get(key)
                if previous_node is not None and previous_node != node:
                    raise _integrity()
                combined.nodes[key] = node
        for value in values:
            value.clear_verified_content()
            value.clear_content_observation()
        return combined


@dataclass(frozen=True, slots=True)
class _RangeNode:
    span: int
    identity: str
    offset: int
    size: int
    forest_position: int | None
    expand: bool = True



class ContentArchive:
    def __init__(self, store: StateStore, scope_id: str, *, namespace: str,
                 tenant_id: str, object_store: ObjectStore | None,
                 object_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None) -> None:
        self._content_reads: WeakKeyDictionary[StateTransaction, dict[str, tuple[ContentRecord, bytes]]] = WeakKeyDictionary()
        self._message_publisher: Callable[[StateTransaction, PreparedArchive], Awaitable[None]] | None = None
        self.store = store
        self.scope_id = validate_archive_group_id(scope_id)
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._objects = object_store
        self._object_reader = object_reader
        self._scope_key = archive_scope_key(namespace, tenant_id)
        self.content_scope = archive_scope_token(self._scope_key)
        self._object_prefix = "v1/archive/" + scope_id + "/" + self.content_scope + "/"

    def _bind(self, prepared: PreparedArchive) -> None:
        owner = self.scope_id, self._namespace, self._tenant_id
        if prepared.owner is None:
            prepared.owner = owner
        elif prepared.owner != owner:
            raise _integrity()

    def empty_root(self) -> ArchiveRoot:
        return ArchiveRoot(self.scope_id, self.content_scope)

    def _key(self, kind: str, identity: str) -> bytes:
        return record_key(self._namespace, self._tenant_id, "archive", kind, (identity,))

    def _record(self, kind: str, identity: str, data: Mapping[str, JsonValue],
                *, payload_bytes: bytes | None = None) -> StoredRecord:
        return StoredRecord(self._key(kind, identity), self._scope_key, None, kind, identity.encode("utf-8"),
                            None, 0, None, 0, None, data, payload_bytes=payload_bytes)

    def content_record_keys(self, values: Sequence[tuple[str, bytes]]) -> tuple[bytes, ...]:
        return tuple(dict.fromkeys(self._key("archive_content", archive_content_identity(
            codec, hashlib.sha256(raw).hexdigest())) for codec, raw in values))

    def observe_content(
        self, prepared: PreparedArchive, requested_keys: Sequence[bytes],
        records: Mapping[bytes, StoredRecord],
    ) -> None:
        """Retain one completed metadata observation, without proving body integrity."""
        self._bind(prepared)
        keys = frozenset(requested_keys)
        if (any(not isinstance(key, bytes) or not 1 <= len(key) <= RECORD_KEY_MAX_BYTES for key in keys)
                or not set(records).issubset(keys)):
            raise _integrity()
        for key, record in records.items():
            if (record.key != key or record.kind != "archive_content"
                    or record.scope_key != self._scope_key):
                raise _integrity()
            content = self._decode_content_record(record)
            if record.sort_key != content.identity.encode("utf-8") or self._key("archive_content", content.identity) != key:
                raise _integrity()
        observed = {key: records.get(key) for key in keys}
        for key, record in observed.items():
            if key in prepared._observed_content and prepared._observed_content[key] != record:
                raise _integrity()
        prepared._observed_content.update(observed)

    async def prepare_content(self, values: Sequence[tuple[str, bytes]],
                              prepared: PreparedArchive) -> tuple[str, ...]:
        return await self._prepare_content(values, prepared)

    async def ensure_content(self, values: Sequence[tuple[str, bytes]]) -> tuple[str, ...]:
        """Verify current immutable bytes and publish only contents not yet present."""
        keys = self.content_record_keys(values)

        async def ensure(transaction: StateTransaction) -> tuple[str, ...]:
            existing = await transaction.get_records(keys)
            prepared = PreparedArchive()
            self.observe_content(prepared, keys, existing)
            identities = await self.prepare_content(values, prepared)
            prepared.records = {key: record for key, record in prepared.records.items() if key not in existing}
            if prepared.records:
                await self.publish(transaction, prepared)
            return identities

        return await self.store.mutate(ensure)

    async def _prepare_content(self, values: Sequence[tuple[str, bytes]], prepared: PreparedArchive,
                               existing: Mapping[bytes, StoredRecord] | None = None) -> tuple[str, ...]:
        self._bind(prepared)
        identities: list[str] = []
        unique: dict[str, tuple[str, bytes, str]] = {}
        for codec, raw in values:
            digest = hashlib.sha256(raw).hexdigest()
            identity = archive_content_identity(codec, digest)
            identities.append(identity)
            unique[identity] = (codec, raw, digest)
        keys = {identity: self._key("archive_content", identity) for identity in unique}
        missing = [key for key in keys.values() if key not in prepared.records]
        observed = {key: prepared._observed_content.pop(key) for key in keys.values()
                    if key in prepared._observed_content}
        if existing is None:
            unread = tuple(key for key in missing if key not in observed)
            existing = dict(await self.store.read_records(unread)) if unread else {}
            existing.update((key, record) for key, record in observed.items() if record is not None)
        for identity, (codec, raw, digest) in unique.items():
            key = keys[identity]
            stored = existing.get(key) or prepared.records.get(key)
            if stored is not None:
                if stored.kind != "archive_content":
                    raise _integrity()
                content = self._decode_content_record(stored)
                if content.identity != identity or content.raw_size != len(raw):
                    raise _integrity()
                if not prepared.content_verified(stored):
                    if await self._load_content(content) != raw:
                        raise _integrity()
                    prepared.verify_content(stored)
                prepared.records[key] = stored
                continue
            compression, payload, encoded = await self.prepare_payload(raw, codec=codec)
            content = ContentRecord(codec, digest, len(raw), compression, payload, encoded if payload is None else None)
            prepared.records[key] = self._record("archive_content", identity, content.to_json(), payload_bytes=content.body)
            prepared.verify_content(prepared.records[key])
        return tuple(identities)

    async def prepare_payload(
        self, raw: bytes, *, existing: ContentRecord | None = None, codec: str = "model-message",
    ) -> tuple[str, StoredPayload | None, bytes]:
        """Prepare the shared representation without reading or publishing SQL rows."""
        _content_codec(codec)
        if existing is not None:
            if (existing.codec != codec or existing.raw_size != len(raw)
                    or existing.raw_digest != hashlib.sha256(raw).hexdigest()):
                raise _integrity()
            if existing.payload is not None:
                if any(ref.store_id != "runtime" or ref.key != self._object_prefix + ref.digest
                       for ref in existing.object_refs()):
                    raise _integrity()
                validate_attachment_payload(raw, existing.payload)
                return existing.compression, existing.payload, raw
            if decode_payload_bytes(existing.body, codec=existing.compression,
                                    raw_digest=existing.raw_digest, raw_size=existing.raw_size) != raw:
                raise _integrity()
            return existing.compression, None, existing.body
        if codec == "model-message" and self._objects is not None:
            async def write(data: bytes) -> ObjectRef:
                digest = hashlib.sha256(data).hexdigest()
                key = self._object_prefix + digest
                async def chunks() -> AsyncIterator[bytes]:
                    yield data
                await self._objects.put(key, chunks(), expected_size=len(data), expected_digest=digest)
                return ObjectRef("runtime", key, digest, len(data))
            payload = await store_attachment_payload(raw, format="model-messages", write_object=write)
            if payload is not None:
                return "raw", payload, raw
        compression, encoded = encode_payload_bytes(raw)
        return compression, None, encoded

    async def read_payload_object(self, reference: ObjectRef) -> bytes:
        """Resolve a canonical message's object through this physical archive route."""
        if (reference.store_id != "runtime" or reference.key != self._object_prefix + reference.digest):
            raise _integrity()
        if self._object_reader is not None:
            return await self._object_reader(reference)
        if self._objects is None:
            raise _integrity()
        return await read_object(self._objects, reference.key,
                                 expected_digest=reference.digest, expected_size=reference.size)

    async def _frontiers(
        self, transaction: StateTransaction, requests: Sequence[ArchiveNodeRead],
        content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord]:
        if isinstance(transaction, ArchiveFrontierReader):
            result = await transaction.read_archive_frontiers(
                scope_key=self._scope_key, requests=requests,
                content_keys=content_keys,
            )
            if result is not None:
                return result
        result = dict(await transaction.get_records(tuple(dict.fromkeys(
            [request.key for request in requests] + list(content_keys)
        ))))
        related: set[bytes] = set()
        body_nodes: set[bytes] = set()
        for request in requests:
            record = result.get(request.key)
            if record is None:
                continue
            node = self._decode_node_record(record)
            slots = ((ArchiveNodeSlot.SELF, node.identity), (ArchiveNodeSlot.LEFT, node.left),
                     (ArchiveNodeSlot.RIGHT, node.right), (ArchiveNodeSlot.NEXT, node.next_forest))
            for slot, identity in slots:
                if identity is None:
                    continue
                key = self._key("archive_sequence", identity)
                if (request.metadata_slots | request.content_slots) & slot:
                    related.add(key)
                if request.content_slots & slot:
                    body_nodes.add(key)
        missing = tuple(related.difference(result))
        if missing:
            result.update(await transaction.get_records(missing))
        bodies = set(content_keys)
        for key in body_nodes:
            record = result.get(key)
            if record is not None:
                node = self._decode_node_record(record)
                bodies.add(self._key("archive_content", node.content_id))
        missing = tuple(bodies.difference(result))
        if missing:
            result.update(await transaction.get_records(missing))
        return result

    def _decode_content_record(self, record: StoredRecord) -> ContentRecord:
        if record.scope_key != self._scope_key:
            raise _integrity()
        return ContentRecord.from_record(record)

    def _decode_node_record(self, record: StoredRecord) -> ImmutableSequenceNode:
        if record.kind != "archive_sequence" or record.scope_key != self._scope_key:
            raise _integrity()
        node = ImmutableSequenceNode.from_json(record.data)
        if record.sort_key != node.identity.encode("utf-8") or record.key != self._key(record.kind, node.identity):
            raise _integrity()
        return node

    def prepared_frontier(self, root: ArchiveRoot, prepared: PreparedArchive) -> tuple[StoredRecord, ...]:
        """Return the at-most-two append nodes already held by this publication."""
        self._bind(prepared)
        if root.scope_id != self.scope_id or root.content_scope != self.content_scope:
            raise _integrity()
        if not root.slices:
            return ()
        tail = root.slices[-1]
        node = prepared.nodes.get(tail.first)
        if tail.start != 0 or tail.end != tail.count or node is None:
            return ()
        identities = (tail.first,) if node.next_forest is None else (tail.first, node.next_forest)
        records = tuple(prepared.records.get(self._key("archive_sequence", identity)) for identity in identities)
        if any(record is None for record in records):
            return ()
        return records

    def reuse_frontier(
        self, root: ArchiveRoot, records: tuple[StoredRecord, ...], prepared: PreparedArchive,
    ) -> None:
        """Seed only the exact append frontier captured by a completed range read."""
        self._bind(prepared)
        if root.scope_id != self.scope_id or root.content_scope != self.content_scope:
            raise _integrity()
        tail = root.slices[-1] if root.slices else None
        if tail is None or tail.start != 0 or tail.end != tail.count:
            raise _integrity()
        weights = ImmutableSequenceNode.forest_weights(tail.count)
        if len(records) != min(2, len(weights)):
            raise _integrity()
        identity = tail.first
        decoded: dict[str, ImmutableSequenceNode] = {}
        for position, record in enumerate(records):
            node = self._decode_node_record(record)
            if (node.identity != identity or node.tree_count != weights[position]
                    or (node.next_forest is None) != (position + 1 == len(weights))):
                raise _integrity()
            previous = prepared.records.get(record.key)
            previous_node = prepared.nodes.get(node.identity)
            if ((previous is not None and previous != record)
                    or (previous_node is not None and previous_node != node)):
                raise _integrity()
            decoded[node.identity] = node
            identity = node.next_forest
        prepared.records.update((record.key, record) for record in records)
        prepared.nodes.update(decoded)

    async def prepare_append(
        self, root: ArchiveRoot, values: Sequence[tuple[str, bytes]], prepared: PreparedArchive,
    ) -> ArchiveRoot:
        """Capture content existence and append frontier in one bounded read plan."""
        self._bind(prepared)
        if root.scope_id != self.scope_id or root.content_scope != self.content_scope:
            raise _integrity()
        if not values:
            return root
        keys = self.content_record_keys(values)
        missing = tuple(key for key in keys if key not in prepared.records)
        observed = {key: prepared._observed_content.pop(key) for key in keys
                    if key in prepared._observed_content}
        unread = tuple(key for key in missing if key not in observed)
        tail = root.slices[-1] if root.slices else None
        first = tail.first if tail is not None and tail.start == 0 and tail.end == tail.count else None
        requests: tuple[ArchiveNodeRead, ...] = ()
        if first is not None:
            known = prepared.nodes.get(first)
            if known is None:
                requests = (ArchiveNodeRead(self._key("archive_sequence", first),
                                           ArchiveNodeSlot.SELF | ArchiveNodeSlot.NEXT),)
            elif known.next_forest is not None and known.next_forest not in prepared.nodes:
                requests = (ArchiveNodeRead(self._key("archive_sequence", known.next_forest)),)
        async def capture(transaction: StateTransaction):
            return await self._frontiers(transaction, requests, unread)
        existing = dict(await self.store.read(capture)) if requests or unread else {}
        existing.update((key, record) for key, record in observed.items() if record is not None)
        cache: dict[str, ImmutableSequenceNode] = {}
        for key, record in existing.items():
            if record.kind != "archive_sequence":
                continue
            node = self._decode_node_record(record)
            if key != record.key:
                raise _integrity()
            previous = prepared.records.get(key)
            if previous is not None and previous != record:
                raise _integrity()
            previous_node = prepared.nodes.get(node.identity)
            if previous_node is not None and previous_node != node:
                raise _integrity()
            prepared.records[key] = record
            prepared.nodes[node.identity] = cache[node.identity] = node
        identities = await self._prepare_content(values, prepared, existing)
        return await self._append(root, identities, prepared, cache)

    async def _nodes(self, identities: Sequence[str], prepared: PreparedArchive,
                     cache: dict[str, ImmutableSequenceNode]) -> None:
        missing = tuple(dict.fromkeys(identity for identity in identities if identity not in cache))
        pending = [identity for identity in missing if identity not in prepared.nodes]
        records = await self.store.read_records([self._key("archive_sequence", identity) for identity in pending]) if pending else {}
        for identity in missing:
            node = prepared.nodes.get(identity)
            if node is None:
                record = records.get(self._key("archive_sequence", identity))
                if record is None:
                    raise _integrity()
                node = self._decode_node_record(record)
                if node.identity != identity:
                    raise _integrity()
            prepared.nodes[identity] = cache[identity] = node

    async def append(self, root: ArchiveRoot, contents: Sequence[str],
                     prepared: PreparedArchive) -> ArchiveRoot:
        return await self._append(root, contents, prepared, {})

    async def _append(self, root: ArchiveRoot, contents: Sequence[str], prepared: PreparedArchive,
                      cache: dict[str, ImmutableSequenceNode]) -> ArchiveRoot:
        self._bind(prepared)
        if root.scope_id != self.scope_id or root.content_scope != self.content_scope:
            raise _integrity()
        slices = list(root.slices)
        for content_id in contents:
            validate_archive_content_identity(content_id)
            tail = slices[-1] if slices and slices[-1].start == 0 and slices[-1].end == slices[-1].count else None
            first_id = None if tail is None else tail.first
            first = second = None
            if first_id is not None:
                await self._nodes((first_id,), prepared, cache)
                first = cache[first_id]
                if first.next_forest is not None:
                    await self._nodes((first.next_forest,), prepared, cache)
                    second = cache[first.next_forest]
            node = ImmutableSequenceNode.extend(content_id, first, second)
            identity = node.identity
            prepared.nodes[identity] = cache[identity] = node
            record = self._record("archive_sequence", identity, node.to_json())
            prepared.records[record.key] = record
            count = 1 if tail is None else tail.count + 1
            value = ArchiveSlice(identity, count, 0, count)
            if tail is None:
                slices.append(value)
            else:
                slices[-1] = value
        return ArchiveRoot(self.scope_id, self.content_scope, tuple(slices))

    async def rewrite(
        self, baseline: ArchiveRoot, values: Sequence[tuple[str, bytes]], prepared: PreparedArchive,
    ) -> ArchiveRoot:
        """Reuse ordered old occurrences as exact slices, retaining only new content."""
        self._bind(prepared)
        previous = await self.content_ids(baseline, prepared=prepared)
        positions: dict[str, deque[int]] = defaultdict(deque)
        for index, identity in enumerate(previous):
            positions[identity].append(index)
        selected: list[int | None] = []
        cursor = 0
        for codec, raw in values:
            identity = archive_content_identity(codec, hashlib.sha256(raw).hexdigest())
            candidates = positions[identity]
            while candidates and candidates[0] < cursor:
                candidates.popleft()
            position = candidates.popleft() if candidates else None
            selected.append(position)
            if position is not None:
                cursor = position + 1
        root = self.empty_root()
        index = 0
        while index < len(values):
            position = selected[index]
            end = index + 1
            if position is None:
                while end < len(values) and selected[end] is None:
                    end += 1
                root = await self.prepare_append(root, values[index:end], prepared)
            else:
                while end < len(values) and selected[end] == position + end - index:
                    end += 1
                slices = list(root.slices)
                for span in baseline.slice(position, position + end - index).slices:
                    if (slices and slices[-1].first == span.first and slices[-1].count == span.count
                            and slices[-1].end == span.start):
                        old = slices[-1]
                        slices[-1] = ArchiveSlice(old.first, old.count, old.start, span.end)
                    else:
                        slices.append(span)
                root = ArchiveRoot(self.scope_id, self.content_scope, tuple(slices))
            index = end
        return root

    def publication_record_keys(self, prepared: PreparedArchive) -> tuple[bytes, ...]:
        self._bind(prepared)
        return tuple(set(prepared.records) | self._publication_dependencies(prepared))

    def _publication_dependencies(self, prepared: PreparedArchive) -> set[bytes]:
        dependencies: set[bytes] = set()
        # Previously confirmed nodes are read knowledge, not pending writes.
        for record in prepared.records.values():
            if record.kind != "archive_sequence":
                continue
            try:
                node = prepared.nodes[record.sort_key.decode("ascii")]
            except (UnicodeError, KeyError) as error:
                raise _integrity() from error
            dependencies.add(self._key("archive_content", node.content_id))
            dependencies.update(self._key("archive_sequence", child) for child in
                                (node.left, node.right, node.next_forest) if child is not None)
        return dependencies

    def bind_message_publisher(
        self, publisher: Callable[[StateTransaction, PreparedArchive], Awaitable[None]],
    ) -> None:
        """Bind native dependencies without making the generic archive their owner."""
        self._message_publisher = publisher

    async def publish_native(self, transaction: StateTransaction, prepared: PreparedArchive) -> None:
        """Keep each provisional native sequence with the producer that prepared it."""
        for root, value in prepared.native_messages.items():
            publisher = prepared.native_publishers.get(root, self._message_publisher)
            if publisher is None:
                raise _integrity()
            await publisher(transaction, PreparedArchive(native_messages={root: value}))

    async def publish(self, transaction: StateTransaction, prepared: PreparedArchive, *, native: bool = True) -> None:
        """Use the archive transaction enlisted with every owner publication."""
        if native and prepared.native_messages:
            await self.publish_native(transaction, prepared)
        prepared.clear_verified_content()
        prepared.clear_content_observation()
        self._bind(prepared)
        dependencies = self._publication_dependencies(prepared)
        if dependencies.issubset(prepared.records):
            await transaction.insert_immutable_records(tuple(prepared.records.values()))
            return
        existing = await transaction.get_records(tuple(set(prepared.records) | dependencies))
        if not dependencies.issubset(set(prepared.records) | set(existing)):
            raise _integrity()
        missing: list[StoredRecord] = []
        for key, record in prepared.records.items():
            current = existing.get(key)
            if current is None:
                missing.append(record)
            elif (current.kind != record.kind or current.data != record.data
                  or current.payload_bytes != record.payload_bytes):
                raise _integrity()
        if missing:
            await transaction.insert_records(missing)

    async def content_ids(self, root: ArchiveRoot, *, start: int = 0,
                          end: int | None = None, prepared: PreparedArchive | None = None) -> tuple[str, ...]:
        selected = root.slice(start, root.count if end is None else end)
        return (await self.content_ids_many((selected,), prepared=prepared))[0]

    async def content_ids_many(self, roots: Sequence[ArchiveRoot], *,
                               prepared: PreparedArchive | None = None) -> tuple[tuple[str, ...], ...]:
        if prepared is not None or not roots:
            resolved = await self._resolve_ranges(None, roots, prepared=prepared)
            if resolved is not None:
                return resolved[0]

        async def capture(transaction: StateTransaction):
            resolved = await self._resolve_ranges(transaction, roots, prepared=prepared)
            assert resolved is not None
            return resolved[0]
        return await self.store.read(capture)

    async def _resolve_ranges(
        self, transaction: StateTransaction | None, roots: Sequence[ArchiveRoot], *,
        prepared: PreparedArchive | None = None, include_content: bool = False,
        append_frontiers: dict[ArchiveRoot, tuple[StoredRecord, ...]] | None = None,
    ) -> tuple[tuple[tuple[str, ...], ...], dict[str, ContentRecord]] | None:
        # A prepared-only probe stops before its first durable dependency. The
        # caller then resolves all dependent reads in the original single scope.
        prepared = PreparedArchive() if prepared is None else prepared
        self._bind(prepared)
        nodes: dict[str, ImmutableSequenceNode] = {}
        contents: dict[str, ContentRecord] = {}
        spans: list[ArchiveSlice] = []
        positions: list[tuple[int, int]] = []
        frontier_first: set[str] = set()
        frontier_records: dict[str, StoredRecord] = {}
        for root in roots:
            if root.scope_id != self.scope_id or root.content_scope != self.content_scope:
                raise _integrity()
            positions.append((len(spans), len(root.slices)))
            spans.extend(root.slices)
            if append_frontiers is not None and root.slices:
                tail = root.slices[-1]
                if tail.start == 0 and tail.end == tail.count:
                    frontier_first.add(tail.first)
        weights = [ImmutableSequenceNode.forest_weights(span.count) for span in spans]
        ranges = [(span.count - span.end, span.count - span.start) for span in spans]
        matches: list[list[str | None]] = [[None] * (span.end - span.start) for span in spans]
        queue = deque(_RangeNode(index, span.first, 0, weights[index][0], 0)
                      for index, span in enumerate(spans))
        processed: set[_RangeNode] = set()
        while queue:
            pending: list[_RangeNode] = []
            body_pending: set[str] = set()
            while queue:
                job = queue.popleft()
                if job in processed:
                    continue
                node = nodes.get(job.identity) or prepared.nodes.get(job.identity)
                if node is None:
                    pending.append(job)
                    continue
                nodes[job.identity] = node
                processed.add(job)
                if node.tree_count != job.size:
                    raise _integrity()
                lo, hi = ranges[job.span]
                if job.forest_position is not None:
                    following = job.forest_position + 1
                    if following < len(weights[job.span]):
                        if node.next_forest is None:
                            raise _integrity()
                        queue.append(_RangeNode(job.span, node.next_forest, job.offset + job.size,
                                                weights[job.span][following], following))
                    elif node.next_forest is not None:
                        raise _integrity()
                intersects = job.expand and hi > job.offset and lo < job.offset + job.size
                if not intersects:
                    continue
                if lo <= job.offset < hi:
                    position = job.offset - lo
                    if matches[job.span][position] is not None:
                        raise _integrity()
                    matches[job.span][position] = node.content_id
                    if include_content and node.content_id not in contents:
                        body_pending.add(job.identity)
                if job.size > 1:
                    child_size = (job.size - 1) // 2
                    for identity, offset in ((node.left, job.offset + 1),
                                             (node.right, job.offset + 1 + child_size)):
                        if identity is None:
                            raise _integrity()
                        queue.append(_RangeNode(job.span, identity, offset, child_size, None,
                                                hi > offset and lo < offset + child_size))
            requests: dict[str, ArchiveNodeRead] = {}
            for job in pending:
                lo, hi = ranges[job.span]
                intersects = job.expand and hi > job.offset and lo < job.offset + job.size
                meta = ArchiveNodeSlot.SELF
                body = ArchiveNodeSlot.SELF if intersects and lo <= job.offset < hi else ArchiveNodeSlot(0)
                if intersects and job.size > 1:
                    meta |= ArchiveNodeSlot.LEFT | ArchiveNodeSlot.RIGHT
                    child_size = (job.size - 1) // 2
                    if lo <= job.offset + 1 < hi:
                        body |= ArchiveNodeSlot.LEFT
                    if lo <= job.offset + 1 + child_size < hi:
                        body |= ArchiveNodeSlot.RIGHT
                if (job.forest_position is not None
                        and job.forest_position + 1 < len(weights[job.span])):
                    meta |= ArchiveNodeSlot.NEXT
                    if lo <= job.offset + job.size < hi:
                        body |= ArchiveNodeSlot.NEXT
                if not include_content:
                    body = ArchiveNodeSlot(0)
                key = self._key("archive_sequence", job.identity)
                previous = requests.get(job.identity, ArchiveNodeRead(key, ArchiveNodeSlot(0)))
                requests[job.identity] = ArchiveNodeRead(key, meta | previous.metadata_slots,
                                                         body | previous.content_slots)
            for identity in body_pending:
                key = self._key("archive_sequence", identity)
                previous = requests.get(identity, ArchiveNodeRead(key, ArchiveNodeSlot(0)))
                requests[identity] = ArchiveNodeRead(key, previous.metadata_slots,
                                                     previous.content_slots | ArchiveNodeSlot.SELF)
            if requests:
                if transaction is None:
                    return None
                rows = await self._frontiers(transaction, tuple(requests.values()))
                for key, record in rows.items():
                    if key != record.key or record.scope_key != self._scope_key:
                        raise _integrity()
                    if record.kind == "archive_sequence":
                        node = self._decode_node_record(record)
                        nodes[node.identity] = node
                    elif record.kind == "archive_content":
                        content = self._decode_content_record(record)
                        if (record.sort_key != content.identity.encode("utf-8")
                                or key != self._key(record.kind, content.identity)):
                            raise _integrity()
                        contents[content.identity] = content
                    else:
                        raise _integrity()
                if append_frontiers is not None:
                    wanted = frontier_first | {
                        nodes[identity].next_forest for identity in frontier_first
                        if identity in nodes and nodes[identity].next_forest is not None
                    }
                    for record in rows.values():
                        if record.kind == "archive_sequence" and record.sort_key.decode("ascii") in wanted:
                            frontier_records[record.sort_key.decode("ascii")] = record
                for identity in body_pending:
                    content_id = nodes[identity].content_id
                    record = prepared.records.get(self._key("archive_content", content_id))
                    if record is not None:
                        content = self._decode_content_record(record)
                        if content.identity != content_id:
                            raise _integrity()
                        contents[content_id] = content
            for job in pending:
                if job.identity not in nodes:
                    raise _integrity()
            for identity in body_pending:
                if nodes[identity].content_id not in contents:
                    raise _integrity()
            queue.extend(pending)
        if any(value is None for group in matches for value in group):
            raise _integrity()
        if append_frontiers is not None:
            for root in roots:
                if not root.slices:
                    continue
                tail = root.slices[-1]
                if tail.start != 0 or tail.end != tail.count:
                    continue
                first = nodes[tail.first]
                identities = ((tail.first,) if first.next_forest is None
                              else (tail.first, first.next_forest))
                append_frontiers[root] = tuple(frontier_records[identity] for identity in identities)
        groups = tuple(tuple(value for group in matches[start:start + count]
                             for value in reversed(group) if value is not None)
                       for start, count in positions)
        prepared.nodes.update(nodes)
        return groups, {identity: contents[identity] for group in groups for identity in group} if include_content else {}

    def content_identity_keys(self, identities: Sequence[str]) -> tuple[bytes, ...]:
        """Resolve fixed content identities without preparing or loading bytes."""
        return tuple(self._key("archive_content", validate_archive_content_identity(identity)) for identity in identities)

    async def content_records(self, identities: Sequence[str]) -> tuple[ContentRecord, ...]:
        """Read content metadata in a batch without opening object bodies."""
        keys = {identity: self._key("archive_content", validate_archive_content_identity(identity)) for identity in identities}
        records = await self.store.read_records(tuple(keys.values())) if keys else {}
        values: dict[str, ContentRecord] = {}
        for identity, key in keys.items():
            record = records.get(key)
            if record is None or record.kind != "archive_content":
                raise _integrity()
            content = self._decode_content_record(record)
            if content.identity != identity:
                raise _integrity()
            values[identity] = content
        return tuple(values[identity] for identity in identities)

    async def read_ranges(
        self, roots: Sequence[ArchiveRoot], *, codec: str,
        append_frontiers: dict[ArchiveRoot, tuple[StoredRecord, ...]] | None = None,
    ) -> tuple[tuple[bytes, ...], ...]:
        load = await self.capture_ranges(roots, codec=codec, append_frontiers=append_frontiers)
        return await load()

    async def capture_ranges(
        self, roots: Sequence[ArchiveRoot], *, codec: str,
        append_frontiers: dict[ArchiveRoot, tuple[StoredRecord, ...]] | None = None,
    ) -> Callable[[], Awaitable[tuple[tuple[bytes, ...], ...]]]:
        """Capture immutable metadata; defer object I/O until the owner scope closes."""
        frontiers = {} if append_frontiers is not None else None

        async def capture(transaction: StateTransaction):
            resolved = await self._resolve_ranges(transaction, roots, include_content=True,
                                                  append_frontiers=frontiers)
            assert resolved is not None
            return resolved

        groups, records = await self.store.read(capture) if roots else ((), {})
        async def load() -> tuple[tuple[bytes, ...], ...]:
            decoded: dict[str, bytes] = {}
            for identity, content in records.items():
                if content.codec != codec:
                    raise _integrity()
                decoded[identity] = await self._load_content(content)
            result = tuple(tuple(decoded[identity] for identity in group) for group in groups)
            if append_frontiers is not None and frontiers is not None:
                append_frontiers.update(frontiers)
            return result

        return load

    async def read_contents(self, identities: Sequence[str], *, codec: str,
                            reserve: Callable[[str, int], None] | None = None) -> tuple[bytes, ...]:
        keys = {identity: self._key("archive_content", validate_archive_content_identity(identity)) for identity in identities}
        records = await self.store.read_records(tuple(keys.values())) if keys else {}
        decoded: dict[str, bytes] = {}
        for identity, key in keys.items():
            record = records.get(key)
            if record is None or record.kind != "archive_content":
                raise _integrity()
            content = self._decode_content_record(record)
            if content.identity != identity or content.codec != codec:
                raise _integrity()
            if reserve is not None:
                reserve(identity, content.raw_size)
            decoded[identity] = await self._load_content(content)
        return tuple(decoded[identity] for identity in identities)

    async def capture_content_records(
        self, identities: Sequence[str],
    ) -> tuple[tuple[ContentRecord, ...], Callable[[Sequence[str]], Awaitable[tuple[bytes, ...]]]]:
        """Capture actual owner-scope metadata and bind its later byte loading.

        The caller enlists this archive with its owner before capture. The loader
        accepts only captured identities, never caller-provided content records.
        """
        records = await self.content_records(identities)
        captured = {content.identity: content for content in records}

        async def load(selected: Sequence[str]) -> tuple[bytes, ...]:
            if any(identity not in captured for identity in selected):
                raise _integrity()
            return tuple([await self._load_content(captured[identity]) for identity in selected])

        return records, load

    async def _load_content(self, content: ContentRecord) -> bytes:
        """Reuse validation only for the same captured inline source in this transaction.

        Every caller first captures source metadata. Changed or removed sources
        therefore remain failures; external object bodies are never cached here.
        """
        transaction = active_state_transaction(self.store)
        cache = None
        if transaction is not None and content.payload is None:
            try:
                cache = self._content_reads.setdefault(transaction, {})
            except TypeError:
                pass  # Custom transactions may not support weak references.
        previous = None if cache is None else cache.get(content.identity)
        if previous is not None and previous[0] == content:
            return previous[1]
        raw = await self._read_content(content)
        if cache is not None:
            cache[content.identity] = content, raw
        return raw

    async def _read_content(self, content: ContentRecord) -> bytes:
        raw = (content.body if content.payload is None else
               await read_attachment_payload(content.payload, read_object=self.read_payload_object))
        return decode_payload_bytes(raw, codec=content.compression,
                                    raw_digest=content.raw_digest, raw_size=content.raw_size)


__all__ = ["ArchiveSlice", "ArchiveRoot", "ImmutableSequenceNode", "ContentRecord", "PreparedArchive", "ContentArchive", "archive_content_scope", "archive_scope_key", "archive_scope_parts", "archive_scope_token", "decode_archive_scope", "archive_content_identity", "validate_archive_content_identity", "validate_archive_group_id"]
