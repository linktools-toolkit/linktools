#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exact retained archive closure shared by offline collection and snapshots."""

from collections.abc import Iterator, Mapping, Sequence

from ...core import JsonValue
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef, StoredPayload
from ._archive import ArchiveRoot, ContentRecord, ImmutableSequenceNode
from ._codec import _decode_enveloped_domain, _encode_persisted_domain, decode_domain, encode_domain, encode_envelope
from ._store import StoredRecord, record_key_digest


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
        "unavailable_reason", "repository_instructions", "session_metadata", "memory",
    }):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    ArchiveRoot.from_json(manifest["history"])
    return wire, payload


def _context_references(manifest: Mapping[str, object]) -> Iterator[tuple[str, str, str]]:
    values = [manifest.get("repository_instructions")]
    for name in ("session_metadata", "memory"):
        mapping = manifest.get(name)
        if mapping is not None:
            if not isinstance(mapping, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.extend(mapping.values())
    for value in values:
        if value is None:
            continue
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
                yield ArchiveRoot.from_json(context[1].decode()["history"])
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


class ArchiveClosure:
    """Mark structural DAG nodes and only content visible through exact slices.

    Structural verification and complete tree coverage are memoized separately.
    Prefix roots therefore share work without keeping a descendant set per root.
    """

    def __init__(self, scope_id: str, records: Sequence[StoredRecord]) -> None:
        self.scope_id = scope_id
        self._has_roots = False
        self._records: dict[tuple[str, str, str], StoredRecord] = {}
        self._nodes: dict[tuple[str, str], ImmutableSequenceNode] = {}
        self._structural: set[tuple[str, str]] = set()
        self._complete: set[tuple[str, str]] = set()
        self._forest_info: dict[tuple[str, str], tuple[int, int, bool]] = {}
        self._complete_forests: set[tuple[str, str]] = set()
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

    def _mark_structure(self, scope: str, identity: str, active: set[str]) -> None:
        key = scope, identity
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
                self._mark_structure(scope, child, active)
        if node.next_forest is not None:
            self._mark_structure(scope, node.next_forest, active)
        active.remove(identity)
        self._structural.add(key)
        self._retained.add(self._record(scope, "archive_sequence", identity).key_digest)

    def _mark_tree(self, scope: str, identity: str, start: int, end: int) -> None:
        key = scope, identity
        node = self._node(scope, identity)
        if not 0 <= start < end <= node.tree_count:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if key in self._complete:
            return
        if start == 0:
            self.mark_content(self.scope_id, scope, node.content_id)
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

    def mark(self, root: ArchiveRoot) -> None:
        self._has_roots = True
        if root.scope_id != self.scope_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        scope = root.content_scope
        for span in root.slices:
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
            ArchiveRoot.from_json(manifest["history"])
            tuple(_context_references(manifest))
            manifest = {**manifest, "history": rewrite(manifest["history"])}
            for name in ("session_metadata", "memory"):
                if manifest.get(name) is not None:
                    manifest[name] = {key: rewrite(item) for key, item in manifest[name].items()}
            if manifest.get("repository_instructions") is not None:
                manifest["repository_instructions"] = rewrite(manifest["repository_instructions"])
            payload = StoredPayload.inline_json(manifest)
            encoded = _encode_persisted_domain(payload) if "schema" in wire else encode_domain(payload)
            result["fields"]["input_context"]["fields"]["payload"] = encoded
        return result
    if isinstance(value, list):
        return [rekey_archive_roots(item, scopes) for item in value]
    return value


__all__ = ["ArchiveClosure", "iter_archive_roots", "iter_archive_content_refs", "rekey_archive_roots"]
