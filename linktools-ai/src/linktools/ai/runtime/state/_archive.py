#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Group-owned immutable content and exact persistent sequence ranges."""

import hashlib
from collections import defaultdict, deque
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field

from ...core import JsonValue, canonical_json_bytes, canonical_sha256
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, ObjectStore, StoredPayload, read_object
from ._archive_read import ArchiveFrontierReader, ArchiveNodeRead, ArchiveNodeSlot
from ._payload import PAYLOAD_COMPRESS_MINIMUM, decode_payload_bytes, encode_payload_bytes
from ._store import StateStore, StateTransaction, StoredRecord, record_key_digest


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
    content_scope: str
    slices: tuple[ArchiveSlice, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.scope_id, str) or not self.scope_id:
            raise _integrity()
        _digest(self.content_scope)

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
            return cls(scope, _digest(value["content_scope"]), tuple(ArchiveSlice.from_json(item) for item in slices))
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
        _digest(self.content_id)
        if _count(self.tree_count) < 1 or self.tree_count & (self.tree_count + 1):
            raise _integrity()
        for value in (self.left, self.right, self.next_forest):
            if value is not None:
                _digest(value)
        if (self.left is None) != (self.right is None):
            raise _integrity()
        if (self.tree_count == 1) != (self.left is None):
            raise _integrity()

    def to_json(self) -> dict[str, JsonValue]:
        return {"content": self.content_id, "count": self.tree_count,
                "left": self.left, "right": self.right, "next": self.next_forest}

    @property
    def identity(self) -> str:
        return canonical_sha256(self.to_json())

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "ImmutableSequenceNode":
        try:
            return cls(_digest(value["content"]), _count(value["count"]),
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
    payload: StoredPayload

    @property
    def identity(self) -> str:
        return canonical_sha256([self.codec, self.raw_digest])

    def to_json(self) -> dict[str, JsonValue]:
        return {"codec": self.codec, "raw_digest": self.raw_digest, "raw_size": self.raw_size,
                "compression": self.compression, "payload": self.payload.to_json()}

    @classmethod
    def from_json(cls, value: Mapping[str, JsonValue]) -> "ContentRecord":
        try:
            codec, compression = value["codec"], value["compression"]
            if not isinstance(codec, str) or not codec or compression not in {"raw", "zlib"}:
                raise _integrity()
            return cls(codec, _digest(value["raw_digest"]), _count(value["raw_size"]),
                       compression, StoredPayload.from_json(dict(value["payload"])))
        except (KeyError, TypeError, ValueError) as error:
            raise _integrity() from error


@dataclass(slots=True)
class PreparedArchive:
    """Unpublished immutable records; publish with the owning domain's CAS."""

    owner: tuple[str, str, str] | None = None
    records: dict[bytes, StoredRecord] = field(default_factory=dict)
    nodes: dict[str, ImmutableSequenceNode] = field(default_factory=dict)
    _verified_content: dict[bytes, str] = field(default_factory=dict, repr=False, compare=False)
    _observed_content: dict[bytes, StoredRecord | None] = field(default_factory=dict, repr=False, compare=False)

    def content_verified(self, record: StoredRecord) -> bool:
        return self._verified_content.get(record.key_digest) == hashlib.sha256(record.kind.encode() + b"\0" + canonical_json_bytes(record.data)).hexdigest()

    def verify_content(self, record: StoredRecord) -> None:
        self._verified_content[record.key_digest] = hashlib.sha256(record.kind.encode() + b"\0" + canonical_json_bytes(record.data)).hexdigest()

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


def _forest_weights(count: int) -> tuple[int, ...]:
    values: list[int] = []
    while count:
        size = (1 << ((count + 1).bit_length() - 1)) - 1
        values.append(size)
        count -= size
    return tuple(reversed(values))


class ContentArchive:
    def __init__(self, store: StateStore, scope_id: str, *, namespace: str,
                 tenant_id: str, object_store: ObjectStore | None,
                 object_reader: Callable[[ObjectRef], Awaitable[bytes]] | None = None) -> None:
        self.store = store
        self.scope_id = scope_id
        self._namespace = namespace
        self._tenant_id = tenant_id
        self._objects = object_store
        self._object_reader = object_reader
        self.content_scope = canonical_sha256([namespace, tenant_id])
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
        return record_key_digest("", "", "archive", kind, [self.content_scope, identity])

    def _record(self, kind: str, identity: str, data: Mapping[str, JsonValue]) -> StoredRecord:
        return StoredRecord(self._key(kind, identity), bytes.fromhex(self.content_scope), None, kind, identity,
                            None, 0, None, 0, None, data)

    def content_record_keys(self, values: Sequence[tuple[str, bytes]]) -> tuple[bytes, ...]:
        return tuple(dict.fromkeys(self._key("archive_content", canonical_sha256(
            [codec, hashlib.sha256(raw).hexdigest()])) for codec, raw in values))

    def observe_content(
        self, prepared: PreparedArchive, requested_keys: Sequence[bytes],
        records: Mapping[bytes, StoredRecord],
    ) -> None:
        """Retain one completed metadata observation, without proving body integrity."""
        self._bind(prepared)
        keys = frozenset(requested_keys)
        if (any(not isinstance(key, bytes) or len(key) != 32 for key in keys)
                or not set(records).issubset(keys)):
            raise _integrity()
        for key, record in records.items():
            if (record.key_digest != key or record.kind != "archive_content"
                    or record.scope_digest != bytes.fromhex(self.content_scope)):
                raise _integrity()
            content = ContentRecord.from_json(record.data)
            if record.sort_key != content.identity or self._key("archive_content", content.identity) != key:
                raise _integrity()
        observed = {key: records.get(key) for key in keys}
        for key, record in observed.items():
            if key in prepared._observed_content and prepared._observed_content[key] != record:
                raise _integrity()
        prepared._observed_content.update(observed)

    async def prepare_content(self, values: Sequence[tuple[str, bytes]],
                              prepared: PreparedArchive) -> tuple[str, ...]:
        return await self._prepare_content(values, prepared)

    async def _prepare_content(self, values: Sequence[tuple[str, bytes]], prepared: PreparedArchive,
                               existing: Mapping[bytes, StoredRecord] | None = None) -> tuple[str, ...]:
        self._bind(prepared)
        identities = tuple(canonical_sha256([codec, hashlib.sha256(raw).hexdigest()]) for codec, raw in values)
        unique = dict(zip(identities, values))
        keys = {identity: self._key("archive_content", identity) for identity in unique}
        missing = [key for key in keys.values() if key not in prepared.records]
        observed = {key: prepared._observed_content.pop(key) for key in keys.values()
                    if key in prepared._observed_content}
        if existing is None:
            unread = tuple(key for key in missing if key not in observed)
            existing = dict(await self.store.read_records(unread)) if unread else {}
            existing.update((key, record) for key, record in observed.items() if record is not None)
        for identity, (codec, raw) in unique.items():
            if not isinstance(codec, str) or not codec:
                raise ValueError("content codec is required")
            key = keys[identity]
            stored = existing.get(key) or prepared.records.get(key)
            if stored is not None:
                if stored.kind != "archive_content":
                    raise _integrity()
                content = ContentRecord.from_json(stored.data)
                if content.identity != identity or content.raw_size != len(raw):
                    raise _integrity()
                if not prepared.content_verified(stored):
                    if await self._read_content(content) != raw:
                        raise _integrity()
                    prepared.verify_content(stored)
                prepared.records[key] = stored
                continue
            compression, digest, size, encoded = encode_payload_bytes(raw)
            if self._objects is None or size < PAYLOAD_COMPRESS_MINIMUM:
                payload = StoredPayload.inline_bytes(encoded)
            else:
                object_key = self._object_prefix + hashlib.sha256(encoded).hexdigest()
                async def chunks(value: bytes = encoded) -> AsyncIterator[bytes]:
                    yield value
                stat = await self._objects.put(object_key, chunks(), expected_size=len(encoded),
                                               expected_digest=hashlib.sha256(encoded).hexdigest())
                payload = StoredPayload.object(ObjectRef("runtime", object_key, stat.digest, stat.size))
            content = ContentRecord(codec, digest, size, compression, payload)
            prepared.records[key] = self._record("archive_content", identity, content.to_json())
            prepared.verify_content(prepared.records[key])
        return identities

    async def _frontiers(
        self, transaction: StateTransaction, requests: Sequence[ArchiveNodeRead],
        content_keys: Sequence[bytes] = (),
    ) -> Mapping[bytes, StoredRecord]:
        if isinstance(transaction, ArchiveFrontierReader):
            result = await transaction.read_archive_frontiers(
                scope_digest=bytes.fromhex(self.content_scope), requests=requests,
                content_keys=content_keys,
            )
            if result is not None:
                return result
        result = dict(await transaction.get_records(tuple(dict.fromkeys(
            [request.key_digest for request in requests] + list(content_keys)
        ))))
        related: set[bytes] = set()
        body_nodes: set[bytes] = set()
        for request in requests:
            record = result.get(request.key_digest)
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

    def _decode_node_record(self, record: StoredRecord) -> ImmutableSequenceNode:
        if record.kind != "archive_sequence" or record.scope_digest != bytes.fromhex(self.content_scope):
            raise _integrity()
        node = ImmutableSequenceNode.from_json(record.data)
        if record.sort_key != node.identity or record.key_digest != self._key(record.kind, node.identity):
            raise _integrity()
        return node

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
        weights = _forest_weights(tail.count)
        if len(records) != min(2, len(weights)):
            raise _integrity()
        identity = tail.first
        decoded: dict[str, ImmutableSequenceNode] = {}
        for position, record in enumerate(records):
            node = self._decode_node_record(record)
            if (node.identity != identity or node.tree_count != weights[position]
                    or (node.next_forest is None) != (position + 1 == len(weights))):
                raise _integrity()
            previous = prepared.records.get(record.key_digest)
            previous_node = prepared.nodes.get(node.identity)
            if ((previous is not None and previous != record)
                    or (previous_node is not None and previous_node != node)):
                raise _integrity()
            decoded[node.identity] = node
            identity = node.next_forest
        prepared.records.update((record.key_digest, record) for record in records)
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
            if key != record.key_digest:
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
            cache[identity] = node

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
            _digest(content_id)
            tail = slices[-1] if slices and slices[-1].start == 0 and slices[-1].end == slices[-1].count else None
            first_id = None if tail is None else tail.first
            first = second = None
            if first_id is not None:
                await self._nodes((first_id,), prepared, cache)
                first = cache[first_id]
                if first.next_forest is not None:
                    await self._nodes((first.next_forest,), prepared, cache)
                    second = cache[first.next_forest]
            if first is not None and second is not None and first.tree_count == second.tree_count:
                node = ImmutableSequenceNode(content_id, 1 + first.tree_count + second.tree_count,
                                             first_id, first.next_forest, second.next_forest)
            else:
                node = ImmutableSequenceNode(content_id, next_forest=first_id)
            identity = node.identity
            prepared.nodes[identity] = cache[identity] = node
            record = self._record("archive_sequence", identity, node.to_json())
            prepared.records[record.key_digest] = record
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
            identity = canonical_sha256([codec, hashlib.sha256(raw).hexdigest()])
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
        for node in prepared.nodes.values():
            dependencies.add(self._key("archive_content", node.content_id))
            dependencies.update(self._key("archive_sequence", child) for child in
                                (node.left, node.right, node.next_forest) if child is not None)
        return dependencies

    async def publish(self, transaction: StateTransaction, prepared: PreparedArchive) -> None:
        """Use the archive transaction enlisted with every owner publication."""
        prepared.clear_verified_content()
        prepared.clear_content_observation()
        self._bind(prepared)
        dependencies = self._publication_dependencies(prepared)
        existing = await transaction.get_records(tuple(set(prepared.records) | dependencies))
        if not dependencies.issubset(set(prepared.records) | set(existing)):
            raise _integrity()
        missing: list[StoredRecord] = []
        for key, record in prepared.records.items():
            current = existing.get(key)
            if current is None:
                missing.append(record)
            elif current.kind != record.kind or current.data != record.data:
                raise _integrity()
        if missing:
            await transaction.insert_records(missing)

    async def content_ids(self, root: ArchiveRoot, *, start: int = 0,
                          end: int | None = None, prepared: PreparedArchive | None = None) -> tuple[str, ...]:
        selected = root.slice(start, root.count if end is None else end)
        return (await self.content_ids_many((selected,), prepared=prepared))[0]

    async def content_ids_many(self, roots: Sequence[ArchiveRoot], *,
                               prepared: PreparedArchive | None = None) -> tuple[tuple[str, ...], ...]:
        async def capture(transaction: StateTransaction):
            groups, _ = await self._resolve_ranges(transaction, roots, prepared=prepared)
            return groups
        return await self.store.read(capture)

    async def _resolve_ranges(
        self, transaction: StateTransaction, roots: Sequence[ArchiveRoot], *,
        prepared: PreparedArchive | None = None, include_content: bool = False,
        append_frontiers: dict[ArchiveRoot, tuple[StoredRecord, ...]] | None = None,
    ) -> tuple[tuple[tuple[str, ...], ...], dict[str, ContentRecord]]:
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
        weights = [_forest_weights(span.count) for span in spans]
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
                rows = await self._frontiers(transaction, tuple(requests.values()))
                for key, record in rows.items():
                    if key != record.key_digest or record.scope_digest != bytes.fromhex(self.content_scope):
                        raise _integrity()
                    if record.kind == "archive_sequence":
                        node = self._decode_node_record(record)
                        nodes[node.identity] = node
                    elif record.kind == "archive_content":
                        content = ContentRecord.from_json(record.data)
                        if (record.sort_key != content.identity
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
                        if record.kind == "archive_sequence" and record.sort_key in wanted:
                            frontier_records[record.sort_key] = record
                for identity in body_pending:
                    content_id = nodes[identity].content_id
                    record = prepared.records.get(self._key("archive_content", content_id))
                    if record is not None:
                        content = ContentRecord.from_json(record.data)
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
        return groups, {identity: contents[identity] for group in groups for identity in group} if include_content else {}

    async def content_records(self, identities: Sequence[str]) -> tuple[ContentRecord, ...]:
        """Read content metadata in a batch without opening object bodies."""
        keys = {identity: self._key("archive_content", _digest(identity)) for identity in identities}
        records = await self.store.read_records(tuple(keys.values())) if keys else {}
        values: dict[str, ContentRecord] = {}
        for identity, key in keys.items():
            record = records.get(key)
            if record is None or record.kind != "archive_content":
                raise _integrity()
            content = ContentRecord.from_json(record.data)
            if content.identity != identity:
                raise _integrity()
            values[identity] = content
        return tuple(values[identity] for identity in identities)

    async def read_ranges(
        self, roots: Sequence[ArchiveRoot], *, codec: str,
        append_frontiers: dict[ArchiveRoot, tuple[StoredRecord, ...]] | None = None,
    ) -> tuple[tuple[bytes, ...], ...]:
        """Capture dependent immutable metadata together, then read bodies outside the state scope."""
        if not roots:
            return ()

        frontiers = {} if append_frontiers is not None else None

        async def capture(transaction: StateTransaction):
            return await self._resolve_ranges(transaction, roots, include_content=True,
                                              append_frontiers=frontiers)

        groups, records = await self.store.read(capture)
        decoded: dict[str, bytes] = {}
        for identity, content in records.items():
            if content.codec != codec:
                raise _integrity()
            decoded[identity] = await self._read_content(content)
        if append_frontiers is not None and frontiers is not None:
            append_frontiers.update(frontiers)
        return tuple(tuple(decoded[identity] for identity in group) for group in groups)

    async def read_contents(self, identities: Sequence[str], *, codec: str) -> tuple[bytes, ...]:
        keys = {identity: self._key("archive_content", _digest(identity)) for identity in identities}
        records = await self.store.read_records(tuple(keys.values())) if keys else {}
        decoded: dict[str, bytes] = {}
        for identity, key in keys.items():
            record = records.get(key)
            if record is None or record.kind != "archive_content":
                raise _integrity()
            content = ContentRecord.from_json(record.data)
            if content.identity != identity or content.codec != codec:
                raise _integrity()
            decoded[identity] = await self._read_content(content)
        return tuple(decoded[identity] for identity in identities)

    async def _read_content(self, content: ContentRecord) -> bytes:
        payload = content.payload
        if payload.ref is None:
            raw = payload.decode()
            if not isinstance(raw, bytes):
                raise _integrity()
        else:
            if self._objects is None or not payload.ref.key.startswith(self._object_prefix):
                raise _integrity()
            raw = (await self._object_reader(payload.ref) if self._object_reader is not None
                   else await read_object(self._objects, payload.ref.key, expected_digest=payload.digest,
                                          expected_size=payload.size))
            if len(raw) != payload.size or hashlib.sha256(raw).hexdigest() != payload.digest:
                raise _integrity()
        return decode_payload_bytes(raw, codec=content.compression,
                                    raw_digest=content.raw_digest, raw_size=content.raw_size)


__all__ = ["ArchiveSlice", "ArchiveRoot", "ImmutableSequenceNode", "ContentRecord", "PreparedArchive", "ContentArchive"]
