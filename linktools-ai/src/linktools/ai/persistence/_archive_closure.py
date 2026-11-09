#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact retained archive closure shared by offline collection and snapshots."""

import hashlib
import json
from collections.abc import Iterator, Mapping, Sequence

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..storage import ObjectRef, StoredPayload
from ._archive import ArchiveRoot, ArchiveSlice, ContentRecord, ImmutableSequenceNode
from ._message_root import NativeMessageRoot, NativeMessageSlice
from ._payload import decode_payload_bytes
from .codec import (
    _decode_enveloped_domain,
    _encode_persisted_domain,
    decode_domain,
    encode_domain,
    encode_envelope,
)
from .state import StoredRecord, record_key_digest


def _context_payload(value: Mapping[str, object]) -> tuple[Mapping[str, object], StoredPayload] | None:
    """Compact references belong only to the execution input-context field."""
    if value.get("$dataclass") != "execution_record":
        return None
    fields = value.get("fields")
    reference = fields.get("input_context") if isinstance(fields, Mapping) else None
    if not isinstance(reference, Mapping) or reference.get("$dataclass") != "runtime_payload_ref":
        return None
    fields = reference.get("fields")
    wire = fields.get("payload") if isinstance(fields, Mapping) else None
    if not isinstance(wire, Mapping):
        return None
    payload = (_decode_enveloped_domain(encode_envelope({"type": "stored_payload", "payload": wire}), StoredPayload)
               if "schema" in wire else decode_domain(wire, StoredPayload))
    if payload.kind != "inline" or payload.encoding != "json":
        return None
    manifest = payload.decode()
    if not isinstance(manifest, Mapping) or manifest.get("kind") != "execution-context":
        return None
    if (manifest.get("version") != 1 or set(manifest) != {
        "kind", "version", "history", "semantic_digest", "replace_history_system_prompt",
        "unavailable_reason", "repository_instructions", "session_metadata", "memory", "preload_skills",
    }):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _context_history(manifest)
    return wire, payload


def _context_history(manifest: Mapping[str, object]) -> ArchiveRoot | NativeMessageRoot:
    history = manifest["history"]
    try:
        if isinstance(history, Mapping) and history.get("type") == "native_message_root":
            return NativeMessageRoot.from_json(history)
        return ArchiveRoot.from_json(history)
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _context_references(manifest: Mapping[str, object]) -> Iterator[tuple[str, str, str]]:
    instructions = manifest.get("repository_instructions")
    values = [] if instructions is None else [instructions]
    skills = manifest["preload_skills"]
    if not isinstance(skills, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values.extend(skills)
    for name in ("session_metadata", "memory"):
        mapping = manifest.get(name)
        if mapping is not None:
            if not isinstance(mapping, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.extend(mapping.values())
    for value in values:
        if not isinstance(value, Mapping) or set(value) != {"archive_scope", "content_scope", "content_id"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scope, content_scope, identity = value["archive_scope"], value["content_scope"], value["content_id"]
        if not all(isinstance(item, str) and item for item in (scope, content_scope, identity)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        yield scope, content_scope, identity


def iter_archive_roots(value: object) -> Iterator[ArchiveRoot]:
    if isinstance(value, ArchiveRoot):
        yield value
    elif isinstance(value, Mapping):
        if value.get("$dataclass") == "archive_root":
            yield (_decode_enveloped_domain(encode_envelope({"type": "archive_root", "payload": value}), ArchiveRoot)
                   if "schema" in value else decode_domain(value, ArchiveRoot))
        else:
            context = _context_payload(value)
            if context is not None:
                root = _context_history(context[1].decode())
                if isinstance(root, ArchiveRoot):
                    yield root
            for item in value.values():
                yield from iter_archive_roots(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_archive_roots(item)


def iter_archive_content_refs(value: object) -> Iterator[tuple[str, str, str]]:
    if isinstance(value, Mapping):
        context = _context_payload(value)
        if context is not None:
            yield from _context_references(context[1].decode())
        for item in value.values():
            yield from iter_archive_content_refs(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_archive_content_refs(item)


_NATIVE_ROOT_SLOTS = {
    "transcript_chunk": "content",
    "archived_context_block": "root",
    "loaded_model_context": "archive_root",
}


def iter_native_message_roots(value: object) -> Iterator[NativeMessageRoot]:
    """Read declared owner slots, never infer roots from business field names.

    Business mappings are escaped by the durable codec. Payload bytes and the
    dictionaries they decode to are opaque here, even if they resemble a root.
    """
    from .contracts import ArchivedContextBlock, LoadedModelContext, TranscriptChunk

    if isinstance(value, (TranscriptChunk, ArchivedContextBlock, LoadedModelContext)):
        root = (value.content if isinstance(value, TranscriptChunk) else
                value.root if isinstance(value, ArchivedContextBlock) else value.archive_root)
        if isinstance(root, NativeMessageRoot):
            yield root
        return
    if isinstance(value, Mapping):
        kind = value.get("$dataclass")
        context = _context_payload(value)
        if context is not None:
            root = _context_history(context[1].decode())
            if isinstance(root, NativeMessageRoot):
                yield root
        if kind in {"stored_payload", "runtime_payload_ref", "object_ref"}:
            return
        if kind in _NATIVE_ROOT_SLOTS:
            fields = value.get("fields")
            slot = _NATIVE_ROOT_SLOTS[kind]
            if not isinstance(fields, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if slot not in fields and kind == "loaded_model_context":
                return
            if slot not in fields:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            root = fields[slot]
            if root is None and kind == "loaded_model_context":
                return
            if not isinstance(root, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if root.get("$dataclass") == "native_message_root":
                try:
                    yield (_decode_enveloped_domain(encode_envelope({"type": "native_message_root", "payload": root}), NativeMessageRoot)
                           if "schema" in root else decode_domain(root, NativeMessageRoot))
                except (KeyError, TypeError, ValueError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            elif root.get("$dataclass") != "archive_root":
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return
        for item in value.values():
            yield from iter_native_message_roots(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from iter_native_message_roots(item)


def require_known_retention(value: object, *, state: object = None) -> None:
    """Unresolved publication and dependency holds cannot authorize collection."""
    if state is not None and (not isinstance(state, str) or "unknown" in state.lower()):
        raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
    if isinstance(value, Mapping):
        kind = value.get("$dataclass")
        if kind in {"stored_payload", "runtime_payload_ref", "object_ref"}:
            return
        if kind == "execution_record":
            fields = value.get("fields")
            if not isinstance(fields, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            holds = fields.get("dependency_hold_ids", {"$tuple": []})
            if not isinstance(holds, Mapping) or holds != {"$tuple": []}:
                # The dependency owner, not this scanner, decides when a hold
                # has a complete terminal handoff. Never infer that from a CAS.
                raise AIError(ErrorCode.STORAGE_CONFLICT, "Dependency holds require owner settlement before cleanup")
        if "$enum" in value and isinstance(value.get("value"), str) and "unknown" in value["value"].lower():
            raise AIError(ErrorCode.STORAGE_COMMIT_UNKNOWN)
        for item in value.values():
            require_known_retention(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_known_retention(item)


class _SequenceClosure:
    """One-operation DAG memoization shared by archive and native owners."""

    _historical_links = True

    def __init__(self) -> None:
        self._structural: set[tuple[str, str, bool]] = set()
        self._complete: set[tuple[str, str]] = set()
        self._forest_info: dict[tuple[str, str], tuple[int, int, bool]] = {}
        self._complete_forests: set[tuple[str, str]] = set()

    def _node(self, scope: str, identity: str) -> ImmutableSequenceNode:
        raise NotImplementedError

    def _retain_sequence(self, scope: str, identity: str) -> None:
        raise NotImplementedError

    def _retain_content(self, scope: str, identity: str) -> None:
        raise NotImplementedError

    def _mark_structure(self, scope: str, identity: str, active: set[str], *, forest: bool = True) -> None:
        key = scope, identity, forest
        if identity in active:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if key in self._structural:
            return
        active.add(identity)
        node = self._node(scope, identity)
        for child in (node.left, node.right):
            if child is not None:
                if self._node(scope, child).tree_count != (node.tree_count - 1) // 2:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                self._mark_structure(scope, child, active, forest=self._historical_links)
        if forest and node.next_forest is not None:
            self._mark_structure(scope, node.next_forest, active)
        active.remove(identity)
        self._structural.add(key)
        self._retain_sequence(scope, identity)

    def _mark_tree(self, scope: str, identity: str, start: int, end: int) -> None:
        key = scope, identity
        node = self._node(scope, identity)
        if not 0 <= start < end <= node.tree_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if key in self._complete:
            return
        if start == 0:
            self._retain_content(scope, node.content_id)
        offset = 1
        for child in (node.left, node.right):
            if child is None:
                continue
            count = self._node(scope, child).tree_count
            lo, hi = max(0, start - offset), min(count, end - offset)
            if hi > lo:
                self._mark_tree(scope, child, lo, hi)
            offset += count
        if start == 0 and end == node.tree_count:
            self._complete.add(key)

    def _forest(self, scope: str, identity: str) -> tuple[int, int, bool]:
        key = scope, identity
        if key not in self._forest_info:
            node = self._node(scope, identity)
            total, equal = node.tree_count, False
            if node.next_forest is not None:
                count, weight, next_equal = self._forest(scope, node.next_forest)
                if node.tree_count > weight or next_equal:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                total += count
                equal = node.tree_count == weight
            self._forest_info[key] = total, node.tree_count, equal
        return self._forest_info[key]

    def _mark_full_forest(self, scope: str, identity: str) -> None:
        key = scope, identity
        if key in self._complete_forests:
            return
        node = self._node(scope, identity)
        self._mark_tree(scope, identity, 0, node.tree_count)
        if node.next_forest is not None:
            self._mark_full_forest(scope, node.next_forest)
        self._complete_forests.add(key)

    def _mark_slices(self, scope: str, slices: Sequence[ArchiveSlice | NativeMessageSlice]) -> None:
        for span in slices:
            self._mark_structure(scope, span.first, set())
            if self._forest(scope, span.first)[0] != span.count:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if span.start == 0 and span.end == span.count:
                self._mark_full_forest(scope, span.first)
                continue
            lo, hi = span.count - span.end, span.count - span.start
            identity: str | None = span.first
            offset = 0
            while identity is not None and offset < hi:
                if lo <= offset and hi == span.count:
                    self._mark_full_forest(scope, identity)
                    break
                node = self._node(scope, identity)
                left, right = max(0, lo - offset), min(node.tree_count, hi - offset)
                if right > left:
                    self._mark_tree(scope, identity, left, right)
                offset += node.tree_count
                identity = node.next_forest



class ArchiveClosure(_SequenceClosure):
    """Exact retained archive metadata and object references."""

    def __init__(self, scope_id: str, records: Sequence[StoredRecord]) -> None:
        super().__init__()
        self.scope_id = scope_id
        self._has_roots = False
        self._records: dict[tuple[str, str, str], StoredRecord] = {}
        self._nodes: dict[tuple[str, str], ImmutableSequenceNode] = {}
        self._retained: set[bytes] = set()
        for record in records:
            if record.kind not in {"archive_content", "archive_sequence"}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if record.scope_digest is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            scope = record.scope_digest.hex()
            identity = scope, record.kind, record.sort_key
            if record.key_digest != record_key_digest("", "", "archive", record.kind, [scope, record.sort_key]):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if identity in self._records:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._records[identity] = record

    def _record(self, scope: str, kind: str, identity: str) -> StoredRecord:
        try:
            return self._records[scope, kind, identity]
        except KeyError as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _node(self, scope: str, identity: str) -> ImmutableSequenceNode:
        key = scope, identity
        if key not in self._nodes:
            record = self._record(scope, "archive_sequence", identity)
            node = ImmutableSequenceNode.from_json(record.data)
            if node.identity != identity:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._nodes[key] = node
        return self._nodes[key]

    def _retain_sequence(self, scope: str, identity: str) -> None:
        self._retained.add(self._record(scope, "archive_sequence", identity).key_digest)

    def _retain_content(self, scope: str, identity: str) -> None:
        self.mark_content(self.scope_id, scope, identity)

    def mark_content(self, scope_id: str, content_scope: str, identity: str) -> None:
        self._has_roots = True
        if scope_id != self.scope_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        record = self._record(content_scope, "archive_content", identity)
        content = ContentRecord.from_json(record.data)
        if content.identity != identity:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if content.payload.ref is not None and not content.payload.ref.key.startswith(
            f"v1/archive/{scope_id}/{content_scope}/"
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._retained.add(record.key_digest)

    def mark(self, root: ArchiveRoot) -> None:
        self._has_roots = True
        if root.scope_id != self.scope_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        self._mark_slices(root.content_scope, root.slices)

    @property
    def has_roots(self) -> bool:
        return self._has_roots

    @property
    def retained_records(self) -> tuple[StoredRecord, ...]:
        return tuple(record for record in self._records.values() if record.key_digest in self._retained)

    @property
    def unreachable_records(self) -> tuple[StoredRecord, ...]:
        return tuple(record for record in self._records.values() if record.key_digest not in self._retained)

    @property
    def object_refs(self) -> tuple[ObjectRef, ...]:
        return tuple(
            content.payload.ref
            for record in self.retained_records if record.kind == "archive_content"
            for content in (ContentRecord.from_json(record.data),)
            if content.payload.ref is not None
        )



class NativeMessageClosure(_SequenceClosure):
    """Native structural closure without following a subtree's historical tail."""

    _historical_links = False

    def __init__(self, rows: Sequence[Mapping[str, object]]) -> None:
        super().__init__()
        self._rows: dict[tuple[str, str], Mapping[str, object]] = {}
        self._nodes: dict[tuple[str, str], ImmutableSequenceNode] = {}
        self.retained_sequences: set[tuple[str, str]] = set()
        self.retained_messages: set[tuple[str, str]] = set()
        for row in rows:
            if not isinstance(row, Mapping) or not {
                "store_digest", "sha256", "message_sha256", "tree_count",
                "left_sha256", "right_sha256", "next_sha256",
            }.issubset(row):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for field in ("store_digest", "sha256"):
                value = row[field]
                if (not isinstance(value, str) or len(value) != 64
                        or any(char not in "0123456789abcdef" for char in value)):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            key = row["store_digest"], row["sha256"]
            if key in self._rows:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._rows[key] = row

    def _node(self, scope: str, identity: str) -> ImmutableSequenceNode:
        key = scope, identity
        if key not in self._nodes:
            row = self._rows.get(key)
            if row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            try:
                node = ImmutableSequenceNode(row["message_sha256"], row["tree_count"],
                                             row["left_sha256"], row["right_sha256"], row["next_sha256"])
            except (KeyError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            if node.identity != identity:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            self._nodes[key] = node
        return self._nodes[key]

    def _retain_sequence(self, scope: str, identity: str) -> None:
        self.retained_sequences.add((scope, identity))

    def _retain_content(self, scope: str, identity: str) -> None:
        self.retained_messages.add((scope, identity))

    def mark(self, store_digest: str, slices: Sequence[NativeMessageSlice]) -> None:
        try:
            self._mark_slices(store_digest, slices)
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


    def message_identities(self, store_digest: str, slices: Sequence[NativeMessageSlice]) -> tuple[str, ...]:
        """Read validated message identities in chronological occurrence order."""
        self.mark(store_digest, slices)

        def tree(identity: str, start: int, end: int) -> Iterator[str]:
            node = self._node(store_digest, identity)
            if start == 0:
                yield node.content_id
            offset = 1
            for child in (node.left, node.right):
                if child is None:
                    continue
                count = self._node(store_digest, child).tree_count
                lo, hi = max(0, start - offset), min(count, end - offset)
                if hi > lo:
                    yield from tree(child, lo, hi)
                offset += count

        result = []
        for span in slices:
            lo, hi = span.count - span.end, span.count - span.start
            identity, offset = span.first, 0
            reversed_messages = []
            while identity is not None and offset < hi:
                node = self._node(store_digest, identity)
                start, end = max(0, lo - offset), min(node.tree_count, hi - offset)
                if end > start:
                    reversed_messages.extend(tree(identity, start, end))
                offset += node.tree_count
                identity = node.next_forest
            result.extend(reversed(reversed_messages))
        return tuple(result)

    def mark_snapshot(self, row: Mapping[str, object]) -> None:
        """Mark a native snapshot's declared full sequence, including tombstones."""
        try:
            root = row["sequence_root"]
            if type(row["pruned"]) is not bool:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if row["pruned"]:
                if root is not None or row["messages"] != "{}":
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return
            descriptor = json.loads(row["messages"])
            if not isinstance(descriptor, dict) or set(descriptor) != {"stored_messages", "checkpoint"}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            checkpoint = descriptor["checkpoint"]
            if checkpoint is not None:
                if (not isinstance(checkpoint, dict)
                        or set(checkpoint) != {"output_retries_used", "message_sha256"}
                        or type(checkpoint["output_retries_used"]) is not int
                        or checkpoint["output_retries_used"] < 0
                        or not isinstance(checkpoint["message_sha256"], str)
                        or len(checkpoint["message_sha256"]) != 64
                        or any(char not in "0123456789abcdef" for char in checkpoint["message_sha256"])):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            count = descriptor["stored_messages"]
            if type(count) is not int or count < 0 or (root is None) != (count == 0):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if count:
                self.mark(row["store_digest"], (NativeMessageSlice(root, count, 0, count),))
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def validate_native_message_content(
    row: Mapping[str, object], *, group_id: str | None = None,
) -> ContentRecord:
    """Verify native metadata and inline bytes without reinterpreting SDK JSON."""
    try:
        content = ContentRecord.from_json(json.loads(row["message"]))
        if content.codec != "model-message" or content.raw_digest != row["sha256"]:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (group_id is not None and content.payload.ref is not None
                and not content.payload.ref.key.startswith(f"v1/archive/{group_id}/")):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if content.payload.kind == "inline":
            encoded = content.payload.decode()
            if content.payload.encoding == "json":
                encoded = canonical_json_bytes(encoded)
            elif isinstance(encoded, str):
                encoded = encoded.encode("utf-8")
            if (not isinstance(encoded, bytes) or len(encoded) != content.payload.size
                    or hashlib.sha256(encoded).hexdigest() != content.payload.digest):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decode_payload_bytes(encoded, codec=content.compression,
                                 raw_digest=content.raw_digest, raw_size=content.raw_size)
        return content
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def rekey_native_message_roots(
    value: JsonValue, groups: Mapping[str, str],
    stores: Mapping[tuple[str, str], str] | None = None,
) -> JsonValue:
    """Rewrite declared native root slots; leave business payloads opaque."""
    def rewrite(root: NativeMessageRoot) -> NativeMessageRoot:
        if root.group_id not in groups:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        store = root.store_digest if stores is None else stores.get((root.group_id, root.store_digest))
        if store is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return NativeMessageRoot(groups[root.group_id], store, root.slices)

    if isinstance(value, dict):
        kind = value.get("$dataclass")
        if kind in {"stored_payload", "runtime_payload_ref", "object_ref"}:
            return value
        # Validate all declared slots before rewriting even an empty root.
        roots = tuple(iter_native_message_roots(value)) if kind in _NATIVE_ROOT_SLOTS else ()
        if roots:
            fields = value["fields"]
            slot = _NATIVE_ROOT_SLOTS[kind]
            wire = fields[slot]
            root = rewrite(roots[0])
            encoded = _encode_persisted_domain(root) if "schema" in wire else encode_domain(root)
            return {**value, "fields": {**fields, slot: encoded}}
        context = _context_payload(value)
        result = {key: rekey_native_message_roots(item, groups, stores) for key, item in value.items()}
        if context is not None:
            wire, payload = context
            manifest = payload.decode()
            history = _context_history(manifest)
            if isinstance(history, NativeMessageRoot):
                payload = StoredPayload.inline_json({**manifest, "history": rewrite(history).to_json()})
                encoded = _encode_persisted_domain(payload) if "schema" in wire else encode_domain(payload)
                result["fields"]["input_context"]["fields"]["payload"] = encoded
        return result
    if isinstance(value, list):
        return [rekey_native_message_roots(item, groups, stores) for item in value]
    return value


def rekey_archive_roots(value: JsonValue, scopes: Mapping[str, str]) -> JsonValue:
    """Rewrite only typed archive roots and execution input-context references."""
    if isinstance(value, dict):
        if value.get("$dataclass") == "archive_root":
            fields = value.get("fields")
            if not isinstance(fields, dict) or fields.get("scope_id") not in scopes:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return {**value, "fields": {**fields, "scope_id": scopes[fields["scope_id"]]}}
        context = _context_payload(value)
        result = {key: rekey_archive_roots(item, scopes) for key, item in value.items()}
        if context is not None:
            wire, payload = context
            manifest = payload.decode()

            def rewrite(reference: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
                scope = reference.get("archive_scope")
                if not isinstance(scope, str) or scope not in scopes:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                return {**reference, "archive_scope": scopes[scope]}

            # Validate the closed set of reference-bearing slots before rewriting.
            history = _context_history(manifest)
            tuple(_context_references(manifest))
            manifest = {**manifest, "history": (rewrite(manifest["history"])
                        if isinstance(history, ArchiveRoot) else manifest["history"])}
            for name in ("session_metadata", "memory"):
                if manifest.get(name) is not None:
                    manifest[name] = {key: rewrite(item) for key, item in manifest[name].items()}
            if manifest.get("repository_instructions") is not None:
                manifest["repository_instructions"] = rewrite(manifest["repository_instructions"])
            manifest["preload_skills"] = [rewrite(item) for item in manifest["preload_skills"]]
            payload = StoredPayload.inline_json(manifest)
            encoded = _encode_persisted_domain(payload) if "schema" in wire else encode_domain(payload)
            result["fields"]["input_context"]["fields"]["payload"] = encoded
        return result
    if isinstance(value, list):
        return [rekey_archive_roots(item, scopes) for item in value]
    return value


__all__ = [
    "ArchiveClosure", "NativeMessageClosure", "iter_archive_roots", "iter_archive_content_refs",
    "iter_native_message_roots", "rekey_archive_roots", "rekey_native_message_roots",
    "require_known_retention", "validate_native_message_content",
]
