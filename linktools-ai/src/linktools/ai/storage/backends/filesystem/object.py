#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FilesystemObjectBackend: filesystem-backed ObjectWriterBackend + history,
with an operation journal for crash recovery.

Layout:

    .storage/
      revision                   # the namespace's monotonic revision counter
      operations/                # the operation journal
        <operation-id>/
          intent.json            # the planned mutation (keys, versions, hashes)
          state                  # PREPARED | VERSIONS_PUBLISHED |
                                 # REVISION_PUBLISHED | COMMITTED | ABORTED
      history/
        <encoded-key>/
          versions/
            <version>/           # one immutable directory per version
              metadata.json      # always present
              content.bin        # absent for a tombstone
      idempotency/
        <encoded-op-key>.json    # journal idempotency record (immutable result ref)

Every mutation runs inside ONE namespace lock
(``coordinator.hold("object-namespace")``) so cross-process observers cannot
see intermediate state, and a crash leaves a single operation record the next
operation's recovery resolves. The state machine is:

    PREPARED → VERSIONS_PUBLISHED → REVISION_PUBLISHED → COMMITTED

Recovery on every operation entry applies the recovery table: PREPARED
records are aborted (any temp dir cleaned up); records past PREPARED are
forward-completed (publish remaining version dirs from the source-of-truth in
the intent, advance revision if not yet advanced, write idempotency). The
"never regress an already-published revision" rule is enforced: recovery
never overwrites a higher revision with a lower one.

Reads IGNORE the operations/ directory entirely -- only history/ + revision/
participate in reads -- so a finished journal record (whether cleaned up or
not) cannot distort a live read.

All filesystem access is dirfd-relative via :class:`SecureDirectory`; no
``resolve_secure_path`` style path-check-then-use remains."""

from __future__ import annotations

import asyncio
import json
import urllib.parse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any, Mapping

from ...coordination.protocols import KeyedCoordinator
from ...object.errors import (
    StorageIdempotencyConflictError,
    StorageIntegrityError,
    StorageObjectNotFoundError,
    StoragePreconditionFailedError,
)
from ...object.models import (
    Depth,
    Found,
    Masked,
    Missing,
    ObjectInfo,
    ObjectPage,
    ObjectVersionPage,
    StorageKey,
    StoredObject,
    WriteOptions,
)
from .secure_directory import FilesystemSecurityMode, SecureDirectory, TrustedLocalDirectory


# --- path-encoding helpers ---------------------------------------------------

def _encoded(key: StorageKey) -> str:
    # Percent-encode so the mapping from StorageKey -> directory name is
    # reversible: "/" and "%" are escaped, so distinct keys can never
    # collide on one directory.
    return urllib.parse.quote(key.value.strip("/") or "__root__", safe="")


def _matches_depth(prefix: StorageKey, candidate: StorageKey, depth: "Depth") -> bool:
    if not candidate.is_under(prefix):
        return False
    if depth is Depth.INFINITY:
        return True
    if prefix.is_root:
        rel_depth = len(candidate._segments)
    else:
        if candidate.value == prefix.value:
            rel_depth = 0
        else:
            rel_depth = len(candidate._segments) - len(prefix._segments)
    if depth is Depth.ZERO:
        return rel_depth == 0
    return rel_depth <= 1


# --- journal state + record shapes ------------------------------------------

class _OpState(str, Enum):
    PREPARED = "PREPARED"
    TARGET_PUBLISHED = "TARGET_PUBLISHED"
    SOURCE_PUBLISHED = "SOURCE_PUBLISHED"
    VERSIONS_PUBLISHED = "VERSIONS_PUBLISHED"
    REVISION_PUBLISHED = "REVISION_PUBLISHED"
    COMMITTED = "COMMITTED"
    ABORTED = "ABORTED"


@dataclass
class _VersionIntent:
    """One version-directory publication within an operation. For a move, the
    operation has two of these: the target result + the source tombstone. For
    a put, one; for a delete, one (the tombstone)."""

    key_value: str
    version: int
    tombstone: bool
    etag: str
    content_type: "str | None"
    size: int
    modified_at: str
    metadata: "dict[str, Any]"
    # For a move's target: the (key, version) the content is sourced from.
    # recovery reads source content from history/<src>/versions/<v>/content.bin.
    source_key: "str | None" = None
    source_version: "int | None" = None
    commit_revision: "int | None" = None
    content_sha256: "str | None" = None
    operation_id: "str | None" = None


@dataclass
class _OperationIntent:
    operation: str  # "put" | "delete" | "move"
    operation_id: str
    request_hash: "str | None"
    idempotency_key: "str | None"
    new_revision: int
    versions: "list[_VersionIntent]"

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "operation": self.operation,
                "operation_id": self.operation_id,
                "request_hash": self.request_hash,
                "idempotency_key": self.idempotency_key,
                "new_revision": self.new_revision,
                "versions": [v.__dict__ for v in self.versions],
            },
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "_OperationIntent":
        data = json.loads(raw)
        return cls(
            operation=data["operation"],
            operation_id=data["operation_id"],
            request_hash=data.get("request_hash"),
            idempotency_key=data.get("idempotency_key"),
            new_revision=int(data["new_revision"]),
            versions=[_VersionIntent(**v) for v in data.get("versions", [])],
        )


@dataclass
class _IdempotencyRecord:
    """The idempotency record: the replay reads from the immutable
    version directory (referenced by result_key + result_version), NOT from
    the current live state."""

    operation: str
    request_hash: str
    result_key: "str | None"  # None for delete
    result_version: "int | None"
    commit_revision: int

    def to_json(self) -> bytes:
        return json.dumps(
            {
                "operation": self.operation,
                "request_hash": self.request_hash,
                "result_key": self.result_key,
                "result_version": self.result_version,
                "commit_revision": self.commit_revision,
            },
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, raw: bytes) -> "_IdempotencyRecord":
        data = json.loads(raw)
        return cls(
            operation=data["operation"],
            request_hash=data["request_hash"],
            result_key=data.get("result_key"),
            result_version=data.get("result_version"),
            commit_revision=int(data["commit_revision"]),
        )


# --- the backend ------------------------------------------------------------

class FilesystemObjectBackend:
    """Filesystem ObjectWriterBackend + VersionedObjectBackend with an
    operation journal for crash recovery. NOT a TransactionalObjectBackend
    (multi-object transactions are refused rather than faked).

    Every op runs inside ``coordinator.hold("object-namespace")``; every
    mutation runs the journal state machine PREPARED → VERSIONS_PUBLISHED →
    REVISION_PUBLISHED → COMMITTED so a crash mid-mutation leaves a record
    the next op's recovery resolves."""

    backend_id: str = "primary"
    _NAMESPACE_KEY = "object-namespace"

    def __init__(
        self,
        *,
        root: Path,
        coordinator: "KeyedCoordinator | None" = None,
        mode: FilesystemSecurityMode = FilesystemSecurityMode.SECURE_POSIX,
    ) -> None:
        self._sd = (
            TrustedLocalDirectory(root)
            if mode is FilesystemSecurityMode.TRUSTED_LOCAL
            else SecureDirectory(root, mode=mode)
        )
        # Coordinator: the coordination spec says the backend is constructed WITH one
        # (or constructs a default FilesystemKeyedCoordinator). Construction
        # of the default is left to the convenience wrapper so the bare
        # backend can be assembled into a different coordination topology.
        if coordinator is None:
            coordinator = self._build_default_coordinator()
        self._coordinator = coordinator
        self._recovery_error: BaseException | None = None
        self._last_generation = -1
        # Lazy: created on first writer that needs them.
        self._sd.ensure_directory(".storage", "history")
        self._sd.ensure_directory(".storage", "operations")
        self._sd.ensure_directory(".storage", "idempotency")

    def _build_default_coordinator(self) -> "KeyedCoordinator":
        # Late import: FilesystemKeyedCoordinator needs fcntl; constructing
        # the default coordinator only on a backend that did not have one
        # injected keeps the import out of the read path on platforms that
        # inject their own coordinator.
        from ...coordination.file import FilesystemKeyedCoordinator

        return FilesystemKeyedCoordinator(
            root=self._sd.root / ".storage" / "coordination"
        )

    @property
    def root(self) -> Path:
        return self._sd.root

    @property
    def security_mode(self) -> FilesystemSecurityMode:
        return self._sd.security_mode

    @property
    def coordinator(self) -> "KeyedCoordinator":
        return self._coordinator

    # --- component helpers ------------------------------------------------

    @staticmethod
    def _key_versions_dir_components(key: StorageKey) -> "tuple[str, ...]":
        return (".storage", "history", _encoded(key), "versions")

    @staticmethod
    def _version_dir_components(key: StorageKey, version: int) -> "tuple[str, ...]":
        return FilesystemObjectBackend._key_versions_dir_components(key) + (
            str(version),
        )

    @staticmethod
    def _version_metadata_components(key: StorageKey, version: int) -> "tuple[str, ...]":
        return FilesystemObjectBackend._version_dir_components(key, version) + (
            "metadata.json",
        )

    @staticmethod
    def _version_content_components(key: StorageKey, version: int) -> "tuple[str, ...]":
        return FilesystemObjectBackend._version_dir_components(key, version) + (
            "content.bin",
        )

    @staticmethod
    def _operation_dir_components(operation_id: str) -> "tuple[str, ...]":
        return (".storage", "operations", operation_id)

    @staticmethod
    def _operation_intent_components(operation_id: str) -> "tuple[str, ...]":
        return FilesystemObjectBackend._operation_dir_components(operation_id) + (
            "intent.json",
        )

    @staticmethod
    def _operation_state_components(operation_id: str) -> "tuple[str, ...]":
        return FilesystemObjectBackend._operation_dir_components(operation_id) + (
            "state",
        )

    @staticmethod
    def _idempotency_components(op_key: str) -> "tuple[str, ...]":
        return (".storage", "idempotency", urllib.parse.quote(op_key, safe="") + ".json")

    @staticmethod
    def _revision_components() -> "tuple[str, ...]":
        return (".storage", "revision")

    @staticmethod
    def _generation_components() -> "tuple[str, ...]":
        return (".storage", "operations_generation")

    # --- public ops (all wrapped in coordinator.hold) ----------------------

    async def raw_get(self, key: StorageKey, *, include_content: bool = True):
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(self._raw_get_sync, key, include_content=include_content)

    async def raw_stat(self, key: StorageKey) -> "ObjectInfo | None":
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(self._raw_stat_sync, key)

    async def raw_list(
        self, prefix: StorageKey, *, depth: "Depth", limit: int, cursor: "str | None"
    ) -> ObjectPage:
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(
                self._raw_list_sync, prefix, depth=depth, limit=limit, cursor=cursor
            )

    async def revision(self) -> str:
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return str(await asyncio.to_thread(self._read_revision_sync))

    async def raw_put_checked(
        self, key: StorageKey, content: bytes, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(
                self._raw_put_checked_sync,
                key,
                content,
                options=options,
                request_hash=request_hash,
            )

    async def raw_delete_checked(
        self, key: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> None:
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            await asyncio.to_thread(
                self._raw_delete_checked_sync, key, options=options, request_hash=request_hash
            )

    async def raw_move_checked(
        self, source: StorageKey, target: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(
                self._raw_move_checked_sync,
                source,
                target,
                options=options,
                request_hash=request_hash,
            )

    async def raw_get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(self._raw_get_version_sync, key, version)

    async def raw_get_at_revision(self, key: StorageKey, revision: int) -> "StoredObject | None":
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(self._raw_get_at_revision_sync, key, revision)

    async def raw_list_versions(
        self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None
    ) -> ObjectVersionPage:
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(
                self._raw_list_versions_sync, key, limit=limit, cursor=cursor
            )

    async def raw_list_at_revision(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]":
        async with self._coordinator.hold(self._NAMESPACE_KEY):
            return await asyncio.to_thread(self._raw_list_at_revision_sync, prefix, revision)

    # --- recovery (the never-regress state machine) --------------------------------------------

    def _recover(self) -> None:
        """Strictly recover every unfinished operation before serving work."""
        if self._recovery_error is not None:
            raise self._recovery_error
        operations_components = (".storage", "operations")
        if self._sd.stat(*operations_components) is None:
            return
        try:
            for op_id in self._sd.list_names(*operations_components):
                self._recover_one(op_id)
        except BaseException as exc:
            self._recovery_error = exc
            raise

    def _recover_one(self, op_id: str) -> None:
        intent_components = self._operation_intent_components(op_id)
        if self._sd.stat(*intent_components) is None:
            raise StorageIntegrityError(f"operation {op_id!r} has no intent")
        intent = _OperationIntent.from_json(self._sd.read_bytes(*intent_components))
        state = self._read_state(op_id)
        if state is None:
            raise StorageIntegrityError(f"operation {op_id!r} has invalid state")
        if state in (_OpState.COMMITTED, _OpState.ABORTED):
            return
        if state is _OpState.PREPARED:
            # The intent was written but no version directory was ever
            # published (the rename in step 7 hadn't run). Clean up any
            # half-built temp dir and mark ABORTED. We DO NOT delete an
            # already-published version dir here -- if it exists, the rename
            # won the race against the state write, so forward-complete
            # instead (see below).
            if any(self._version_exists(v) for v in intent.versions):
                self._finalize_after_versions_published(op_id, intent)
            else:
                self._abort_operation(op_id, intent)
            return
        if state in (_OpState.TARGET_PUBLISHED, _OpState.SOURCE_PUBLISHED, _OpState.VERSIONS_PUBLISHED):
            self._finalize_after_versions_published(op_id, intent)
            return
        if state is _OpState.REVISION_PUBLISHED:
            # Revision already advanced; only idempotency remains.
            self._write_idempotency_from_intent(intent)
            self._set_state(op_id, _OpState.COMMITTED)
            return
        raise StorageIntegrityError(f"unknown recovery state for operation {op_id!r}")

    def _version_exists(self, v: _VersionIntent) -> bool:
        key = StorageKey(v.key_value)
        return self._sd.stat(*self._version_metadata_components(key, v.version)) is not None

    def _all_versions_published(self, intent: _OperationIntent) -> bool:
        for v in intent.versions:
            key = StorageKey(v.key_value)
            if self._sd.stat(*self._version_metadata_components(key, v.version)) is None:
                return False
        return True

    def _finalize_after_versions_published(self, op_id: str, intent: _OperationIntent) -> None:
        """Forward-complete an operation whose versions exist but which has
        not yet reached COMMITTED. Publish any missing version dir from the
        intent (verifying against intent); advance the revision if not yet
        advanced; write idempotency."""
        # 1. Ensure every version dir exists with intent-matching content.
        for v in intent.versions:
            self._ensure_version_published(v)
        self._set_state(op_id, _OpState.VERSIONS_PUBLISHED)
        # 2. Advance revision if not already at or beyond intent.new_revision.
        # The "never regress" rule: only ever raise.
        current = self._read_revision_sync()
        if current < intent.new_revision:
            self._sd.atomic_write(
                *self._revision_components(),
                content=str(intent.new_revision).encode("utf-8"),
            )
        self._set_state(op_id, _OpState.REVISION_PUBLISHED)
        # 3. Idempotency.
        self._write_idempotency_from_intent(intent)
        self._set_state(op_id, _OpState.COMMITTED)

    def _ensure_version_published(self, v: _VersionIntent) -> None:
        """The version directory for ``v`` must exist with intent-matching
        metadata. If it is missing, publish it from the source-of-truth
        (history/<src>/versions/<v> for a move's target; an empty tombstone
        dir for a delete). If it exists, verify the metadata matches."""
        key = StorageKey(v.key_value)
        meta_components = self._version_metadata_components(key, v.version)
        existing_meta = self._sd.stat(*meta_components)
        metadata_payload = self._version_metadata_payload(v)
        if existing_meta is None:
            # Publish the missing version directory atomically.
            files: "dict[str, bytes]" = {"metadata.json": metadata_payload}
            content_bytes = self._materialize_content(v)
            if content_bytes is not None:
                files["content.bin"] = content_bytes
            self._sd.ensure_directory(*self._key_versions_dir_components(key))
            self._sd.atomic_publish_directory(
                *self._version_dir_components(key, v.version),
                files=files,
            )
        else:
            # Verify metadata matches (spec: "验证内容").
            actual = json.loads(self._sd.read_bytes(*meta_components))
            if not _metadata_matches_intent(actual, v):
                raise StorageIntegrityError(
                    f"version {v.version} of {v.key_value!r} on disk does not "
                    f"match its operation intent"
                )
            if not v.tombstone:
                try:
                    content = self._read_version_bytes(key, v.version)
                except FileNotFoundError as exc:
                    raise StorageIntegrityError(f"missing content for {v.key_value!r}") from exc
                if v.content_sha256 and sha256(content).hexdigest() != v.content_sha256:
                    raise StorageIntegrityError(f"content digest mismatch for {v.key_value!r}")

    def _materialize_content(self, v: _VersionIntent) -> "bytes | None":
        """For a tombstone: no content (None). For a put: content was in the
        caller's hand at crash time and is unrecoverable, so a put-version
        that is missing at recovery is an integrity error. For a move's
        target: read content from source_key@source_version in history/."""
        if v.tombstone:
            return None
        if v.source_key is None:
            if not v.operation_id:
                raise StorageIntegrityError(f"missing operation id for {v.key_value!r}")
            payload = self._operation_dir_components(v.operation_id) + ("payloads", str(v.version), "content.bin")
            try:
                content = self._sd.read_bytes(*payload)
            except FileNotFoundError as exc:
                raise StorageIntegrityError(f"missing staged payload for {v.key_value!r}") from exc
            if v.content_sha256 and sha256(content).hexdigest() != v.content_sha256:
                raise StorageIntegrityError(f"content digest mismatch for {v.key_value!r}")
            return content
        src_key = StorageKey(v.source_key)
        content = self._sd.read_bytes(
            *self._version_content_components(src_key, v.source_version)
        )
        if v.content_sha256 and sha256(content).hexdigest() != v.content_sha256:
            raise StorageIntegrityError(f"content digest mismatch for {v.key_value!r}")
        return content

    def _version_metadata_payload(self, v: _VersionIntent) -> bytes:
        return json.dumps(
            {
                "key": v.key_value,
                "version": v.version,
                "commit_revision": v.commit_revision,
                "etag": v.etag,
                "content_type": v.content_type,
                "size": v.size,
                "modified_at": v.modified_at,
                "metadata": v.metadata,
                "tombstone": v.tombstone,
            },
            sort_keys=True,
        ).encode("utf-8")

    def _abort_operation(self, op_id: str, intent: _OperationIntent) -> None:
        """A PREPARED op that did NOT finish publishing its versions: discard
        any temp artifacts and mark ABORTED. The published versions (there
        should be none at PREPARED) stay; live state is unchanged."""
        # Nothing temp to clean here in our atomic_publish_directory flow --
        # a half-built temp dir lives under versions/ with a .NAME.tmp prefix
        # and is removed by atomic_publish_directory itself on its own
        # exception path. We could sweep them here as defense-in-depth.
        self._set_state(op_id, _OpState.ABORTED)

    def _try_mark_aborted(self, op_id: str) -> None:
        try:
            self._set_state(op_id, _OpState.ABORTED)
        except Exception:
            pass

    def _write_idempotency_from_intent(self, intent: _OperationIntent) -> None:
        if intent.idempotency_key is None:
            return
        # The result references the (non-tombstone) version produced by the
        # operation. For move: target result. For put: the new version. For
        # delete: None.
        result = next(
            (v for v in intent.versions if not v.tombstone),
            None,
        )
        op_key = self._idempotency_op_key(
            intent.operation, intent.idempotency_key, intent.versions
        )
        record = _IdempotencyRecord(
            operation=intent.operation,
            request_hash=intent.request_hash or "",
            result_key=result.key_value if result is not None else None,
            result_version=result.version if result is not None else None,
            commit_revision=intent.new_revision,
        )
        self._sd.atomic_write(
            *self._idempotency_components(op_key),
            content=record.to_json(),
        )

    @staticmethod
    def _idempotency_op_key(
        operation: str,
        idempotency_key: str,
        versions: "list[_VersionIntent]",
    ) -> str:
        """Reproduce the live path's idempotency-op-key, which is keyed by
        operation + the keys it touches + the caller-supplied idempotency
        key. Used identically on the live path and in recovery so a replay
        after a crashed-and-recovered op finds the record."""
        if operation == "put":
            return f"put:{versions[0].key_value}:{idempotency_key}"
        if operation == "delete":
            return f"delete:{versions[0].key_value}:{idempotency_key}"
        if operation == "move":
            keys = sorted(v.key_value for v in versions)
            return f"move:{':'.join(keys)}:{idempotency_key}"
        return f"{operation}:{idempotency_key}"

    # --- internal: low-level journal I/O ----------------------------------

    def _read_state(self, op_id: str) -> "_OpState | None":
        components = self._operation_state_components(op_id)
        if self._sd.stat(*components) is None:
            return None
        raw = self._sd.read_bytes(*components).decode("utf-8").strip()
        try:
            return _OpState(raw)
        except ValueError:
            return None

    def _set_state(self, op_id: str, state: _OpState) -> None:
        self._sd.ensure_directory(*self._operation_dir_components(op_id))
        self._sd.atomic_write(
            *self._operation_state_components(op_id),
            content=state.value.encode("utf-8"),
        )

    def _begin_operation(self, intent: _OperationIntent) -> None:
        """Spec the journal step sequence step 4: write intent.json + state=PREPARED."""
        op_dir = self._operation_dir_components(intent.operation_id)
        self._sd.ensure_directory(*op_dir)
        self._sd.atomic_write(
            *self._operation_intent_components(intent.operation_id),
            content=intent.to_json(),
        )
        self._set_state(intent.operation_id, _OpState.PREPARED)
        generation = 0
        if self._sd.stat(*self._generation_components()) is not None:
            generation = int(self._sd.read_bytes(*self._generation_components()) or b"0")
        self._sd.atomic_write(*self._generation_components(), content=str(generation + 1).encode())

    def _read_revision_sync(self) -> int:
        components = self._revision_components()
        if self._sd.stat(*components) is None:
            return 0
        return int(self._sd.read_bytes(*components).decode("utf-8").strip() or "0")

    def _advance_revision_to(self, new_revision: int) -> None:
        self._sd.atomic_write(
            *self._revision_components(),
            content=str(new_revision).encode("utf-8"),
        )

    # --- internal: version directory reads ---------------------------------

    def _list_version_numbers(self, key: StorageKey) -> "list[int]":
        versions_dir = self._key_versions_dir_components(key)
        if self._sd.stat(*versions_dir) is None:
            return []
        out: "list[int]" = []
        for name in self._sd.list_names(*versions_dir):
            try:
                out.append(int(name))
            except ValueError:
                continue
        return sorted(out)

    def _latest_version_number(self, key: StorageKey) -> "int | None":
        versions = self._list_version_numbers(key)
        return versions[-1] if versions else None

    def _read_version_metadata(self, key: StorageKey, version: int) -> "dict | None":
        components = self._version_metadata_components(key, version)
        if self._sd.stat(*components) is None:
            return None
        return json.loads(self._sd.read_bytes(*components))

    def _read_version_bytes(self, key: StorageKey, version: int) -> bytes:
        return self._sd.read_bytes(*self._version_content_components(key, version))

    def _info_from_metadata(self, key: StorageKey, raw: dict) -> ObjectInfo:
        return ObjectInfo(
            key=key,
            etag=raw["etag"],
            version=raw["version"],
            commit_revision=raw.get("commit_revision"),
            content_type=raw["content_type"],
            size=raw["size"],
            modified_at=datetime.fromisoformat(raw["modified_at"]),
            metadata=raw.get("metadata") or {},
        )

    def _live_version_metadata(self, key: StorageKey) -> "tuple[int, dict] | None":
        latest = self._latest_version_number(key)
        if latest is None:
            return None
        return latest, self._read_version_metadata(key, latest)

    def _read_idempotency(self, op_key: str) -> "_IdempotencyRecord | None":
        components = self._idempotency_components(op_key)
        if self._sd.stat(*components) is None:
            return None
        return _IdempotencyRecord.from_json(self._sd.read_bytes(*components))

    def _write_idempotency(self, op_key: str, record: _IdempotencyRecord) -> None:
        self._sd.atomic_write(
            *self._idempotency_components(op_key),
            content=record.to_json(),
        )

    # --- internal: publish version dirs (the live path) -------------------

    def _publish_version_dir(
        self,
        *,
        key: StorageKey,
        version: int,
        metadata: dict,
        content: "bytes | None",
    ) -> None:
        """Spec the journal step sequence step 5-7: build the version directory atomically (all
        files land together via atomic_publish_directory). The directory
        either fully exists with both files or does not exist at all."""
        self._sd.ensure_directory(*self._key_versions_dir_components(key))
        files: "dict[str, bytes]" = {
            "metadata.json": json.dumps(metadata, sort_keys=True).encode("utf-8"),
        }
        if content is not None:
            files["content.bin"] = content
        self._sd.atomic_publish_directory(
            *self._version_dir_components(key, version),
            files=files,
        )

    def _live_metadata_dict(
        self,
        *,
        key: StorageKey,
        version: int,
        commit_revision: int,
        content: "bytes | None",
        content_type: "str | None",
        metadata: "dict[str, Any]",
        tombstone: bool,
        operation_id: str,
    ) -> dict:
        return {
            "key": key.value,
            "version": version,
            "commit_revision": commit_revision,
            "etag": sha256(content).hexdigest() if content is not None else "",
            "content_type": content_type,
            "size": len(content) if content is not None else 0,
            "modified_at": datetime.now(timezone.utc).isoformat(),
            "metadata": dict(metadata),
            "tombstone": tombstone,
            "operation_id": operation_id,
        }

    # --- read paths -------------------------------------------------------

    def _raw_get_sync(self, key: StorageKey, *, include_content: bool):
        self._recover()
        live = self._live_version_metadata(key)
        if live is None:
            return Missing
        version, raw = live
        if raw["tombstone"]:
            return Masked(
                key=key, version=version, commit_revision=raw.get("commit_revision")
            )
        content = b""
        if include_content:
            content = self._read_version_bytes(key, version)
        return Found(
            StoredObject(info=self._info_from_metadata(key, raw), content=content)
        )

    def _raw_stat_sync(self, key: StorageKey) -> "ObjectInfo | None":
        self._recover()
        live = self._live_version_metadata(key)
        if live is None or live[1]["tombstone"]:
            return None
        return self._info_from_metadata(key, live[1])

    def _raw_list_sync(
        self, prefix: StorageKey, *, depth: "Depth", limit: int, cursor: "str | None"
    ) -> ObjectPage:
        self._recover()
        history_components = (".storage", "history")
        if self._sd.stat(*history_components) is None:
            return ObjectPage(items=(), next_cursor=None)
        candidates: "list[StorageKey]" = []
        for name in self._sd.list_names(*history_components):
            key = StorageKey("/" + urllib.parse.unquote(name))
            if not _matches_depth(prefix, key, depth):
                continue
            live = self._live_version_metadata(key)
            if live is None or live[1]["tombstone"]:
                continue
            if cursor is not None and key.value <= cursor:
                continue
            candidates.append(key)
        candidates.sort(key=lambda k: k.value)
        page_keys = candidates[: limit + 1]
        items = []
        for key in page_keys[:limit]:
            _, raw = self._live_version_metadata(key)
            items.append(self._info_from_metadata(key, raw))
        next_cursor = page_keys[limit - 1].value if len(page_keys) > limit else None
        return ObjectPage(items=tuple(items), next_cursor=next_cursor)

    def _raw_get_version_sync(self, key: StorageKey, version: int) -> "StoredObject | None":
        self._recover()
        raw = self._read_version_metadata(key, version)
        if raw is None or raw["tombstone"]:
            return None
        content = self._read_version_bytes(key, version)
        return StoredObject(
            info=self._info_from_metadata(key, raw), content=content
        )

    def _raw_get_at_revision_sync(self, key: StorageKey, revision: int) -> "StoredObject | None":
        self._recover()
        best: "tuple[int, dict] | None" = None
        for v in self._list_version_numbers(key):
            raw = self._read_version_metadata(key, v)
            cr = raw.get("commit_revision")
            if cr is not None and cr <= revision:
                best = (v, raw)
            else:
                break
        if best is None or best[1]["tombstone"]:
            return None
        version, raw = best
        content = self._read_version_bytes(key, version)
        return StoredObject(info=self._info_from_metadata(key, raw), content=content)

    def _raw_list_versions_sync(
        self, key: StorageKey, *, limit: int, cursor: "str | None"
    ) -> ObjectVersionPage:
        self._recover()
        versions = self._list_version_numbers(key)
        start = 0 if cursor is None else int(cursor)
        page_versions = versions[start : start + limit]
        items = []
        for v in page_versions:
            raw = self._read_version_metadata(key, v)
            items.append(self._info_from_metadata(key, raw))
        next_start = start + len(page_versions)
        next_cursor = None if next_start >= len(versions) else str(next_start)
        return ObjectVersionPage(items=tuple(items), next_cursor=next_cursor)

    def _raw_list_at_revision_sync(self, prefix: StorageKey, revision: int) -> "tuple[ObjectInfo, ...]":
        self._recover()
        history_components = (".storage", "history")
        if self._sd.stat(*history_components) is None:
            return ()
        out: "list[ObjectInfo]" = []
        for name in self._sd.list_names(*history_components):
            key = StorageKey("/" + urllib.parse.unquote(name))
            if not key.is_under(prefix):
                continue
            best: "tuple[int, dict] | None" = None
            for v in self._list_version_numbers(key):
                raw = self._read_version_metadata(key, v)
                cr = raw.get("commit_revision")
                if cr is not None and cr <= revision:
                    best = (v, raw)
                else:
                    break
            if best is not None and not best[1]["tombstone"]:
                out.append(self._info_from_metadata(key, best[1]))
        return tuple(out)

    # --- write paths (single-key put + delete) ----------------------------

    def _raw_put_checked_sync(
        self, key: StorageKey, content: bytes, *, options: WriteOptions, request_hash: str
    ) -> StoredObject:
        # 1. recovery
        self._recover()
        # 2. idempotency replay
        idem_key = (
            f"put:{key.value}:{options.idempotency_key}"
            if options.idempotency_key
            else None
        )
        if idem_key is not None:
            record = self._read_idempotency(idem_key)
            if record is not None:
                if record.request_hash != request_hash:
                    raise StorageIdempotencyConflictError(
                        f"idempotency key {options.idempotency_key!r} replayed with a different request"
                    )
                if record.result_key is None or record.result_version is None:
                    raise StorageObjectNotFoundError(key.value)
                # Replay reads from the IMMUTABLE version directory (spec
                # the idempotency record spec): never from the current live state.
                replay_key = StorageKey(record.result_key)
                raw = self._read_version_metadata(replay_key, record.result_version)
                if raw is None:
                    raise StorageIntegrityError(
                        f"idempotency record points at missing version "
                        f"{record.result_version} of {record.result_key!r}"
                    )
                content_bytes = (
                    self._read_version_bytes(replay_key, record.result_version)
                    if not raw["tombstone"]
                    else content
                )
                return StoredObject(
                    info=self._info_from_metadata(replay_key, raw),
                    content=content_bytes,
                )
        # 3. CAS + next version + new revision
        live = self._live_version_metadata(key)
        live_info = None if live is None or live[1]["tombstone"] else live[1]
        if options.if_none_match and live_info is not None:
            raise StoragePreconditionFailedError(
                f"if_none_match failed: {key.value!r} already exists"
            )
        if options.if_match is not None:
            if live_info is None or live_info["etag"] != options.if_match:
                raise StoragePreconditionFailedError(
                    f"if_match failed: {key.value!r} etag mismatch"
                )
        next_version = (live[0] + 1) if live is not None else 1
        new_revision = self._read_revision_sync() + 1
        operation_id = _new_operation_id()
        # 4. write intent + state=PREPARED
        intent = _OperationIntent(
            operation="put",
            operation_id=operation_id,
            request_hash=request_hash,
            idempotency_key=options.idempotency_key,
            new_revision=new_revision,
            versions=[
                _VersionIntent(
                    key_value=key.value,
                    version=next_version,
                    tombstone=False,
                    etag=sha256(content).hexdigest(),
                    content_type=options.content_type,
                    size=len(content),
                    modified_at=datetime.now(timezone.utc).isoformat(),
                    metadata=dict(options.metadata or {}),
                    commit_revision=new_revision,
                    content_sha256=sha256(content).hexdigest(),
                    operation_id=operation_id,
                )
            ],
        )
        self._sd.ensure_directory(*self._operation_dir_components(operation_id), "payloads", str(next_version))
        self._sd.atomic_write(
            *self._operation_dir_components(operation_id), "payloads", str(next_version), "content.bin",
            content=content,
        )
        self._begin_operation(intent)
        # 5-7. publish the version directory atomically.
        metadata = self._live_metadata_dict(
            key=key,
            version=next_version,
            commit_revision=new_revision,
            content=content,
            content_type=options.content_type,
            metadata=dict(options.metadata or {}),
            tombstone=False,
            operation_id=operation_id,
        )
        self._publish_version_dir(
            key=key,
            version=next_version,
            metadata=metadata,
            content=content,
        )
        # 8. state=VERSIONS_PUBLISHED
        self._set_state(operation_id, _OpState.VERSIONS_PUBLISHED)
        # 9. publish revision
        self._advance_revision_to(new_revision)
        # 10. state=REVISION_PUBLISHED
        self._set_state(operation_id, _OpState.REVISION_PUBLISHED)
        # 11. idempotency
        if idem_key is not None:
            self._write_idempotency(
                idem_key,
                _IdempotencyRecord(
                    operation="put",
                    request_hash=request_hash,
                    result_key=key.value,
                    result_version=next_version,
                    commit_revision=new_revision,
                ),
            )
        # 12. state=COMMITTED
        self._set_state(operation_id, _OpState.COMMITTED)
        return StoredObject(
            info=self._info_from_metadata(key, metadata), content=content
        )

    def _raw_delete_checked_sync(
        self, key: StorageKey, *, options: WriteOptions, request_hash: str
    ) -> None:
        self._recover()
        idem_key = (
            f"delete:{key.value}:{options.idempotency_key}"
            if options.idempotency_key
            else None
        )
        if idem_key is not None:
            record = self._read_idempotency(idem_key)
            if record is not None:
                if record.request_hash != request_hash:
                    raise StorageIdempotencyConflictError(
                        f"idempotency key {options.idempotency_key!r} replayed with a different request"
                    )
                return None
        live = self._live_version_metadata(key)
        if live is None or live[1]["tombstone"]:
            # Deleting a missing key is a no-op (no tombstone, no bump). We
            # DO record idempotency so a replay is a no-op too.
            if idem_key is not None:
                self._write_idempotency(
                    idem_key,
                    _IdempotencyRecord(
                        operation="delete",
                        request_hash=request_hash,
                        result_key=None,
                        result_version=None,
                        commit_revision=self._read_revision_sync(),
                    ),
                )
            return None
        if options.if_match is not None and live[1]["etag"] != options.if_match:
            raise StoragePreconditionFailedError(
                f"if_match failed: {key.value!r} etag mismatch"
            )
        next_version = live[0] + 1
        new_revision = self._read_revision_sync() + 1
        operation_id = _new_operation_id()
        intent = _OperationIntent(
            operation="delete",
            operation_id=operation_id,
            request_hash=request_hash,
            idempotency_key=options.idempotency_key,
            new_revision=new_revision,
            versions=[
                _VersionIntent(
                    key_value=key.value,
                    version=next_version,
                    tombstone=True,
                    etag="",
                    content_type=None,
                    size=0,
                    modified_at=datetime.now(timezone.utc).isoformat(),
                    metadata={},
                    commit_revision=new_revision,
                    operation_id=operation_id,
                )
            ],
        )
        self._begin_operation(intent)
        metadata = self._live_metadata_dict(
            key=key,
            version=next_version,
            commit_revision=new_revision,
            content=None,
            content_type=None,
            metadata={},
            tombstone=True,
            operation_id=operation_id,
        )
        # Tombstone: publish with metadata only (no content.bin).
        self._publish_version_dir(
            key=key, version=next_version, metadata=metadata, content=None
        )
        self._set_state(operation_id, _OpState.VERSIONS_PUBLISHED)
        self._advance_revision_to(new_revision)
        self._set_state(operation_id, _OpState.REVISION_PUBLISHED)
        if idem_key is not None:
            self._write_idempotency(
                idem_key,
                _IdempotencyRecord(
                    operation="delete",
                    request_hash=request_hash,
                    result_key=None,
                    result_version=None,
                    commit_revision=new_revision,
                ),
            )
        self._set_state(operation_id, _OpState.COMMITTED)

    def _raw_move_checked_sync(
        self,
        source: StorageKey,
        target: StorageKey,
        *,
        options: WriteOptions,
        request_hash: str,
    ) -> StoredObject:
        self._recover()
        idem_key = (
            f"move:{source.value}:{target.value}:{options.idempotency_key}"
            if options.idempotency_key
            else None
        )
        if idem_key is not None:
            record = self._read_idempotency(idem_key)
            if record is not None:
                if record.request_hash != request_hash:
                    raise StorageIdempotencyConflictError(
                        f"idempotency key {options.idempotency_key!r} replayed with a different request"
                    )
                if record.result_key is None or record.result_version is None:
                    raise StorageObjectNotFoundError(source.value)
                replay_key = StorageKey(record.result_key)
                raw = self._read_version_metadata(replay_key, record.result_version)
                if raw is None:
                    raise StorageIntegrityError(
                        f"idempotency record points at missing version "
                        f"{record.result_version} of {record.result_key!r}"
                    )
                content_bytes = self._read_version_bytes(replay_key, record.result_version)
                return StoredObject(
                    info=self._info_from_metadata(replay_key, raw),
                    content=content_bytes,
                )
        src_live = self._live_version_metadata(source)
        if src_live is None or src_live[1]["tombstone"]:
            raise StorageObjectNotFoundError(source.value)
        src_version, src_raw = src_live
        content = self._read_version_bytes(source, src_version)

        tgt_live = self._live_version_metadata(target)
        tgt_info = None if tgt_live is None or tgt_live[1]["tombstone"] else tgt_live[1]
        if options.if_none_match and tgt_info is not None:
            raise StoragePreconditionFailedError(
                f"if_none_match failed: {target.value!r} already exists"
            )
        if options.if_match is not None:
            if tgt_info is None or tgt_info["etag"] != options.if_match:
                raise StoragePreconditionFailedError(
                    f"if_match failed: {target.value!r} etag mismatch"
                )

        next_target_version = (tgt_live[0] + 1) if tgt_live is not None else 1
        next_source_version = src_version + 1
        new_revision = self._read_revision_sync() + 1
        operation_id = _new_operation_id()
        now_iso = datetime.now(timezone.utc).isoformat()

        # Spec the recovery table: the move intent records BOTH version dirs (target
        # result + source tombstone) sharing ONE commit_revision.
        target_intent = _VersionIntent(
            key_value=target.value,
            version=next_target_version,
            tombstone=False,
            etag=src_raw["etag"],
            content_type=src_raw["content_type"],
            size=src_raw["size"],
            modified_at=now_iso,
            metadata=dict(src_raw.get("metadata") or {}),
            # Content is sourced from source's CURRENT version, so recovery
            # can re-materialize the target even if it crashed before the
            # target publish completed.
            source_key=source.value,
            source_version=src_version,
            commit_revision=new_revision,
            content_sha256=sha256(content).hexdigest(),
            operation_id=operation_id,
        )
        source_tombstone_intent = _VersionIntent(
            key_value=source.value,
            version=next_source_version,
            tombstone=True,
            etag="",
            content_type=None,
            size=0,
            modified_at=now_iso,
            metadata={},
            commit_revision=new_revision,
            operation_id=operation_id,
        )
        intent = _OperationIntent(
            operation="move",
            operation_id=operation_id,
            request_hash=request_hash,
            idempotency_key=options.idempotency_key,
            new_revision=new_revision,
            versions=[target_intent, source_tombstone_intent],
        )
        self._begin_operation(intent)
        # Publish target version dir (with content sourced from source's
        # current version).
        target_metadata = self._live_metadata_dict(
            key=target,
            version=next_target_version,
            commit_revision=new_revision,
            content=content,
            content_type=src_raw["content_type"],
            metadata=dict(src_raw.get("metadata") or {}),
            tombstone=False,
            operation_id=operation_id,
        )
        self._publish_version_dir(
            key=target,
            version=next_target_version,
            metadata=target_metadata,
            content=content,
        )
        # Publish source tombstone (no content.bin).
        source_metadata = self._live_metadata_dict(
            key=source,
            version=next_source_version,
            commit_revision=new_revision,
            content=None,
            content_type=None,
            metadata={},
            tombstone=True,
            operation_id=operation_id,
        )
        self._publish_version_dir(
            key=source,
            version=next_source_version,
            metadata=source_metadata,
            content=None,
        )
        self._set_state(operation_id, _OpState.VERSIONS_PUBLISHED)
        self._advance_revision_to(new_revision)
        self._set_state(operation_id, _OpState.REVISION_PUBLISHED)
        if idem_key is not None:
            self._write_idempotency(
                idem_key,
                _IdempotencyRecord(
                    operation="move",
                    request_hash=request_hash,
                    result_key=target.value,
                    result_version=next_target_version,
                    commit_revision=new_revision,
                ),
            )
        self._set_state(operation_id, _OpState.COMMITTED)
        return StoredObject(
            info=self._info_from_metadata(target, target_metadata), content=content
        )


def _metadata_matches_intent(actual: dict, v: _VersionIntent) -> bool:
    """Recovery-time check that a published version dir matches what the
    intent said it should be. commit_revision is filled in by the live path
    and verify the operation revision as well as the immutable fields."""
    pairs = (
        ("etag", v.etag),
        ("size", v.size),
        ("tombstone", v.tombstone),
        ("version", v.version),
        ("key", v.key_value),
        ("commit_revision", v.commit_revision),
        ("metadata", v.metadata),
    )
    for actual_key, expected in pairs:
        if actual.get(actual_key) != expected:
            return False
    # content_type: tolerate None on both sides; treat None==None as a match
    # even though we wrote it through the live path's dict.
    if actual.get("content_type") != v.content_type and actual.get("content_type") is not None:
        return False
    if not v.tombstone and v.content_sha256:
        return True
    return True


def _new_operation_id() -> str:
    """A unique operation id. Uses secrets.token_hex so two operations in the
    same nanosecond do not collide on a directory name."""
    import secrets

    return secrets.token_hex(16)


class FilesystemObjectStore:
    """Convenience: an ObjectStore pre-wired to a fresh FilesystemObjectBackend
    + its default FilesystemKeyedCoordinator."""

    def __init__(
        self,
        *,
        root: Path,
        coordinator: "KeyedCoordinator | None" = None,
        mode: FilesystemSecurityMode = FilesystemSecurityMode.SECURE_POSIX,
    ) -> None:
        from ...object.store import ObjectStore

        self._backend = FilesystemObjectBackend(
            root=root, coordinator=coordinator, mode=mode
        )
        self._store = ObjectStore(primary=self._backend)

    @property
    def backend(self) -> FilesystemObjectBackend:
        return self._backend

    async def get(self, key: StorageKey) -> "StoredObject | None":
        return await self._store.get(key)

    async def stat(self, key: StorageKey) -> "ObjectInfo | None":
        return await self._store.stat(key)

    async def list(self, prefix: StorageKey, **kwargs) -> ObjectPage:
        return await self._store.list(prefix, **kwargs)

    async def revision(self) -> str:
        return await self._store.revision()

    async def put(self, key: StorageKey, content: bytes, **kwargs) -> StoredObject:
        return await self._store.put(key, content, **kwargs)

    async def delete(self, key: StorageKey, **kwargs) -> None:
        await self._store.delete(key, **kwargs)

    async def move(self, source: StorageKey, target: StorageKey, **kwargs) -> StoredObject:
        return await self._store.move(source, target, **kwargs)

    async def get_version(self, key: StorageKey, version: int) -> "StoredObject | None":
        return await self._backend.raw_get_version(key, version)

    async def list_versions(
        self, key: StorageKey, *, limit: int = 100, cursor: "str | None" = None
    ) -> ObjectVersionPage:
        return await self._backend.raw_list_versions(key, limit=limit, cursor=cursor)
