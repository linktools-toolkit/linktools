#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical Runtime run and portable runtime snapshot contracts."""

import asyncio
import hashlib
import json
import os
import posixpath
import shutil
import secrets
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Mapping, cast

from filelock import FileLock, Timeout
from linktools.core import environ

from ..core import (
    JsonValue,
    canonical_json_bytes,
    canonical_sha256,
    validate_persistence_namespace,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode
from ..storage import FilesystemMutationLock, ObjectRef, ObjectStore, read_object
from ._snapshot_contract import RunSnapshot, snapshot_digest
from ._runtime_history import RuntimeHistory
from .state import OfflineExclusiveStorage, RuntimeState, SnapshotLimits

if TYPE_CHECKING:
    from ..workspace import Workspace

_logger = environ.get_logger("ai.runtime.snapshot")


@dataclass(frozen=True, slots=True)
class SnapshotTargetInspection:
    status: str
    generation: str | None
    snapshot_digest: str


@dataclass(frozen=True, slots=True)
class RestoredRuntime:
    snapshot_digest: str
    namespace: str
    tenant_id: str
    state_root: Path
    workspace_root: Path | None
    workspace_id: str | None
    generation: str

    def open_history(self) -> RuntimeHistory:
        """Return a read-only RuntimeHistory context for this generation."""
        return RuntimeHistory.open(
            self.namespace,
            state=RuntimeState.from_root(self.state_root),
            tenant_id=self.tenant_id,
        )


class RuntimeSnapshot:
    """Offline portable snapshot publisher and local generation restorer."""

    @classmethod
    async def create(
        cls,
        namespace: str,
        *,
        tenant_id: str,
        state: "RuntimeState",
        object_store: ObjectStore,
        workspace: "Workspace | None" = None,
        exclusive: OfflineExclusiveStorage,
        metadata: Mapping[str, JsonValue] | None = None,
        limits: SnapshotLimits,
    ) -> ObjectRef:
        resolved_namespace = validate_persistence_namespace(namespace)
        resolved_tenant = validate_tenant_id(tenant_id)
        if not isinstance(limits, SnapshotLimits):
            raise TypeError("limits must be SnapshotLimits")
        async def publish() -> ObjectRef:
            initialized = False
            try:
                if state.ready:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                await state.initialize(
                    namespace=resolved_namespace,
                    tenant_id=resolved_tenant,
                    read_only=True,
                )
                initialized = True
                if (
                    state.namespace != resolved_namespace
                    or state.tenant_id != resolved_tenant
                ):
                    raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
                state_ref = await state.export_snapshot(
                    object_store=object_store,
                    limits=limits,
                )
                if state_ref.size > limits.max_bytes:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                state_payload = await read_object(
                    object_store,
                    state_ref.key,
                    expected_digest=state_ref.digest,
                    expected_size=state_ref.size,
                )
                state_manifest = _parse_state_manifest(state_payload)
                state_entries, state_object_bytes = _state_snapshot_usage(
                    state_manifest
                )
                workspace_entries = await _capture_workspace(
                    workspace,
                    state=state,
                    object_store=object_store,
                    limits=limits,
                )
                manifest: dict[str, JsonValue] = {
                    "kind": "runtime-snapshot",
                    "format_version": 1,
                    "namespace": resolved_namespace,
                    "tenant_id": resolved_tenant,
                    "state": _object_ref_payload(state_ref),
                    "workspace": workspace_entries,
                    "metadata": dict(metadata or {}),
                }
                payload = canonical_json_bytes(manifest)
                workspace_count, workspace_bytes = _workspace_snapshot_usage(
                    workspace_entries
                )
                if (
                    state_entries + workspace_count > limits.max_entries
                    or len(payload)
                    + state_ref.size
                    + state_object_bytes
                    + workspace_bytes
                    > limits.max_bytes
                ):
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                digest = _digest_bytes(payload)
                key = f"v1/runtime-snapshot/{digest}"
                await _put_object(object_store, key, payload)
                _logger.info(
                    "runtime snapshot published: namespace=%s tenant=%s digest=%s",
                    resolved_namespace,
                    resolved_tenant,
                    digest,
                )
                return ObjectRef(object_store.store_id, key, digest, len(payload))
            finally:
                if initialized:
                    await state.close()

        async with exclusive.offline_exclusivity():
            return await publish()

    @classmethod
    async def verify(
        cls,
        ref: ObjectRef,
        *,
        object_store: ObjectStore,
        limits: SnapshotLimits,
    ) -> None:
        manifest = await _read_manifest(ref, object_store, limits)
        state = _object_ref_from_payload(manifest["state"])
        if state.size > limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        state_payload = await read_object(
            object_store,
            state.key,
            expected_digest=state.digest,
            expected_size=state.size,
        )
        state_manifest = _parse_state_manifest(state_payload)
        if (
            state_manifest.get("namespace") != manifest.get("namespace")
            or state_manifest.get("tenant_id") != manifest.get("tenant_id")
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        state_entries, state_object_bytes = _state_snapshot_usage(
            state_manifest
        )
        state_objects = cast(list[object], state_manifest["objects"])

        workspace = manifest.get("workspace")
        if not isinstance(workspace, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        present = workspace.get("present")
        workspace_id = workspace.get("workspace_id")
        entries = workspace.get("entries")
        if (
            not isinstance(present, bool)
            or (workspace_id is not None and not isinstance(workspace_id, str))
            or not isinstance(entries, list)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if (not present and (workspace_id is not None or entries)) or (
            present and (workspace_id is None or not workspace_id)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kinds = _validate_workspace_entries(entries)
        workspace_count, workspace_bytes = _workspace_snapshot_usage(workspace)
        total_bytes = (
            len(canonical_json_bytes(cast(JsonValue, manifest)))
            + state.size
            + state_object_bytes
            + workspace_bytes
        )
        if (
            state_entries + workspace_count > limits.max_entries
            or total_bytes > limits.max_bytes
        ):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        for raw_object in state_objects:
            if not isinstance(raw_object, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source = _object_ref_from_payload(raw_object.get("source"))
            content = _object_ref_from_payload(raw_object.get("content"))
            if source.digest != content.digest or source.size != content.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await _verify_object(object_store, content)
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if kinds[cast(str, entry["path"])] != "file":
                continue
            content = _object_ref_from_payload(entry.get("content"))
            await _verify_object(object_store, content)


    @classmethod
    async def copy(
        cls,
        ref: ObjectRef,
        source_store: ObjectStore,
        target_store: ObjectStore,
        *,
        limits: SnapshotLimits,
    ) -> ObjectRef:
        manifest = await _read_manifest(ref, source_store, limits)
        await cls.verify(ref, object_store=source_store, limits=limits)
        state_ref = _object_ref_from_payload(manifest["state"])
        state_payload = await read_object(
            source_store,
            state_ref.key,
            expected_digest=state_ref.digest,
            expected_size=state_ref.size,
        )
        refs = [state_ref]
        state_manifest = _parse_state_manifest(state_payload)
        state_objects = state_manifest.get("objects", [])
        if not isinstance(state_objects, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        refs.extend(
            _object_ref_from_payload(cast(Mapping[str, object], entry)["content"])
            for entry in state_objects
            if isinstance(entry, Mapping)
        )
        raw_workspace = manifest.get("workspace")
        if not isinstance(raw_workspace, Mapping) or not isinstance(
            raw_workspace.get("entries"), list
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        entries = cast(list[object], raw_workspace["entries"])
        kinds = _validate_workspace_entries(entries)
        refs.extend(
            _object_ref_from_payload(entry["content"])
            for entry in cast(list[Mapping[str, object]], entries)
            if kinds[cast(str, entry["path"])] == "file"
        )
        for item in refs:
            await _copy_object(source_store, target_store, item)
        await _copy_object(source_store, target_store, ref)
        return ObjectRef(target_store.store_id, ref.key, ref.digest, ref.size)


    @classmethod
    async def restore(
        cls,
        ref: ObjectRef,
        *,
        object_store: ObjectStore,
        target: str | Path,
        namespace: str,
        tenant_id: str,
        limits: SnapshotLimits,
        replace_policy: str = "same",
        expected_generation: str | None = None,
        exclusive: OfflineExclusiveStorage | None = None,
    ) -> RestoredRuntime:
        manifest = await cls._verified_manifest(ref, object_store, limits)
        resolved_namespace = validate_persistence_namespace(namespace)
        resolved_tenant = validate_tenant_id(tenant_id)
        if (
            manifest["namespace"] != resolved_namespace
            or manifest["tenant_id"] != resolved_tenant
        ):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        if replace_policy not in {"missing", "same", "replace"}:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        root = Path(target).expanduser().resolve(strict=False)
        root.mkdir(parents=True, exist_ok=True)
        current_file = root / "current.json"
        current = _read_current(current_file)
        if replace_policy == "missing" and current is not None:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        if replace_policy == "same" and current is not None:
            if current.get("snapshot_digest") != ref.digest:
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
            inspection = await cls.inspect_target(
                root,
                ref,
                object_store=object_store,
                limits=limits,
            )
            if inspection.status != "matching":
                raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
            return _restored_runtime(root, current)
        if replace_policy == "replace" and (
            expected_generation is None or exclusive is None
        ):
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)

        async def restore_generation() -> RestoredRuntime:
            generation = secrets.token_hex(16)
            staging = root / ".staging" / generation
            staging.mkdir(parents=True, exist_ok=False)
            (staging / ".runtime-snapshot-staging").write_text(
                generation,
                encoding="utf-8",
            )
            state_root = staging / "state"
            await RuntimeState.restore_snapshot(
                _object_ref_from_payload(manifest["state"]),
                object_store=object_store,
                root=state_root,
                limits=limits,
            )
            workspace_root = await _restore_workspace(
                manifest.get("workspace"),
                object_store=object_store,
                target=staging / "workspace",
                limits=limits,
            )
            snapshot_payload = canonical_json_bytes(
                cast(dict[str, JsonValue], manifest)
            )
            (staging / "snapshot.json").write_bytes(snapshot_payload)
            generation_root = root / "generations" / generation
            generation_root.parent.mkdir(parents=True, exist_ok=True)
            staging.rename(generation_root)

            publish_lock = root / ".runtime-snapshot-locks" / "publish.lock"
            async with FilesystemMutationLock(publish_lock):
                latest = _read_current(current_file)
                if replace_policy == "missing" and latest is not None:
                    raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
                if replace_policy == "same" and latest is not None:
                    if latest.get("snapshot_digest") != ref.digest:
                        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
                    raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
                if replace_policy == "replace":
                    if latest is None or latest.get("generation") != expected_generation:
                        raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
                value = {
                    "snapshot_digest": ref.digest,
                    "generation": generation,
                    "namespace": resolved_namespace,
                    "tenant_id": resolved_tenant,
                    "state_root": str(generation_root / "state"),
                    "workspace_root": (
                        None
                        if workspace_root is None
                        else str(generation_root / "workspace")
                    ),
                    "workspace_id": (
                        cast(Mapping[str, object], manifest["workspace"]).get(
                            "workspace_id"
                        )
                    ),
                }
                _write_current(current_file, value)
                return _restored_runtime(root, value)

        if replace_policy != "replace":
            return await restore_generation()
        if exclusive is None:
            raise AIError(ErrorCode.SNAPSHOT_CONFLICT)
        async with exclusive.offline_exclusivity():
            return await restore_generation()

    @classmethod
    async def _verified_manifest(
        cls,
        ref: ObjectRef,
        object_store: ObjectStore,
        limits: SnapshotLimits,
    ) -> Mapping[str, object]:
        await cls.verify(ref, object_store=object_store, limits=limits)
        return await _read_manifest(ref, object_store, limits)

    @classmethod
    async def inspect_target(
        cls,
        target: str | Path,
        ref: ObjectRef,
        *,
        object_store: ObjectStore,
        limits: SnapshotLimits,
    ) -> SnapshotTargetInspection:
        await cls.verify(ref, object_store=object_store, limits=limits)
        current = _read_current(Path(target) / "current.json")
        if current is None:
            return SnapshotTargetInspection("missing", None, ref.digest)
        if current.get("snapshot_digest") != ref.digest:
            return SnapshotTargetInspection(
                "conflict",
                cast(str | None, current.get("generation")),
                ref.digest,
            )
        generation = cast(str, current.get("generation"))
        generation_root = Path(target) / "generations" / generation
        status = "matching" if generation_root.exists() else "modified"
        if status == "matching":
            try:
                generation_manifest = _read_generation_manifest(generation_root)
                if (
                    canonical_sha256(cast(JsonValue, generation_manifest))
                    != ref.digest
                ):
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                await _verify_restored_generation(
                    generation_root,
                    manifest=generation_manifest,
                    object_store=object_store,
                    limits=limits,
                )
            except AIError as error:
                if error.code not in {
                    ErrorCode.STORAGE_CONFLICT,
                    ErrorCode.STORAGE_NOT_FOUND,
                }:
                    raise
                status = "modified"
        return SnapshotTargetInspection(status, generation, ref.digest)

    @classmethod
    async def collect_temporary(cls, target: str | Path) -> int:
        target_path = Path(target).expanduser().resolve(strict=False)
        staging = target_path / ".staging"
        if not staging.is_dir():
            return 0
        removed = 0
        locks = target_path / ".runtime-snapshot-locks"
        locks.mkdir(parents=True, exist_ok=True)
        for child in staging.iterdir():
            marker = child / ".runtime-snapshot-staging"
            if not child.is_dir() or not marker.is_file():
                continue
            try:
                generation = marker.read_text(encoding="utf-8").strip()
            except OSError:
                continue
            if not generation:
                continue
            lock = FileLock(str(locks / f"{generation}.lock"), thread_local=False)
            try:
                await asyncio.to_thread(lock.acquire, timeout=0)
            except Timeout:
                continue
            try:
                if child.is_dir() and marker.is_file():
                    shutil.rmtree(child)
                    removed += 1
            finally:
                await asyncio.to_thread(lock.release)
        return removed


async def _capture_workspace(
    workspace: "Workspace | None",
    *,
    state: "RuntimeState",
    object_store: ObjectStore,
    limits: SnapshotLimits,
) -> dict[str, JsonValue]:
    if workspace is None:
        return {
            "present": False,
            "workspace_id": None,
            "entries": [],
        }
    root = workspace.root.resolve()
    excluded = tuple(
        path.resolve()
        for path in (*state.local_paths(), *object_store.local_paths())
    )
    if any(path == root for path in excluded):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    entries: list[dict[str, JsonValue]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()):
        relative = _safe_relative_path(path.relative_to(root).as_posix())
        try:
            resolved = path.resolve(strict=False)
        except RuntimeError as error:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
        if any(_is_child_or_same(resolved, item) for item in excluded):
            if path.is_symlink():
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            continue
        if len(entries) >= limits.max_entries:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

        if path.is_symlink():
            try:
                target = os.readlink(path)
                target_path = (path.parent / target).resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
            _safe_symlink_target(relative, target)
            if (
                not _is_child_or_same(target_path, root)
                or any(_is_child_or_same(target_path, item) for item in excluded)
            ):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            entries.append({"path": relative, "symlink": target})
            continue

        if path.is_dir():
            entries.append({"path": relative, "directory": True})
            continue
        if not path.is_file():
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

        try:
            before = path.stat()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        remaining = limits.max_bytes - total_bytes
        if before.st_size > remaining:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        digest, size = await _read_file_digest(path, remaining)
        try:
            after_digest = path.stat()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if not _same_file_snapshot(before, after_digest, size):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        key = f"v1/runtime-workspace/{digest}"
        await object_store.put(
            key,
            _file_chunks(path),
            expected_size=size,
            expected_digest=digest,
        )
        try:
            after_publish = path.stat()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if not _same_file_snapshot(before, after_publish, size):
            raise AIError(ErrorCode.STORAGE_CONFLICT)

        total_bytes += size
        entries.append(
            {
                "path": relative,
                "mode": after_publish.st_mode & 0o111,
                "content": {
                    "store_id": object_store.store_id,
                    "key": key,
                    "digest": digest,
                    "size": size,
                },
            }
        )

    _validate_workspace_entries(entries)
    return {
        "present": True,
        "workspace_id": workspace.workspace_id,
        "entries": entries,
    }


async def _restore_workspace(
    raw: object,
    *,
    object_store: ObjectStore,
    target: Path,
    limits: SnapshotLimits,
) -> Path | None:
    if not isinstance(raw, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    present = raw.get("present")
    workspace_id = raw.get("workspace_id")
    entries = raw.get("entries")
    if (
        not isinstance(present, bool)
        or (workspace_id is not None and not isinstance(workspace_id, str))
        or not isinstance(entries, list)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not present:
        if workspace_id is not None or entries:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return None
    if workspace_id is None or not workspace_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if len(entries) > limits.max_entries:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    kinds = _validate_workspace_entries(entries)
    total_bytes = sum(
        _object_ref_from_payload(entry.get("content")).size
        for entry in entries
        if isinstance(entry, Mapping) and kinds[cast(str, entry["path"])] == "file"
    )
    if total_bytes > limits.max_bytes:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    target.mkdir(parents=True, exist_ok=False)

    for relative in sorted(
        (path for path, kind in kinds.items() if kind == "directory"),
        key=lambda value: (value.count("/"), value),
    ):
        (target / relative).mkdir(parents=False, exist_ok=False)

    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        relative = cast(str, entry["path"])
        if kinds[relative] != "file":
            continue
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        content = _object_ref_from_payload(entry.get("content"))
        await _write_object_file(
            object_store,
            content,
            destination,
            max_bytes=limits.max_bytes,
        )
        mode = entry.get("mode", 0)
        if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o111:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        try:
            current_mode = destination.stat().st_mode
            os.chmod(destination, (current_mode & ~0o111) | mode)
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    entry_by_path = {
        cast(str, entry["path"]): entry
        for entry in entries
        if isinstance(entry, Mapping)
    }
    links: list[tuple[Path, str, bool]] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        relative = cast(str, entry["path"])
        if kinds[relative] != "symlink":
            continue
        link_target = cast(str, entry["symlink"])
        logical_target = _safe_symlink_target(relative, link_target)
        final_kind = _resolve_workspace_entry_kind(logical_target, entry_by_path, kinds)
        links.append((target / relative, link_target, final_kind == "directory"))

    for destination, link_target, target_is_directory in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            destination.symlink_to(
                link_target,
                target_is_directory=target_is_directory,
            )
        except OSError as error:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error

    for destination, _link_target, _target_is_directory in links:
        try:
            resolved = destination.resolve(strict=True)
        except (OSError, RuntimeError) as error:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
        if not _is_child_or_same(resolved, target):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    return target


def _safe_relative_path(value: str) -> str:
    if (
        not isinstance(value, str)
        or not value
        or "\x00" in value
        or "\\" in value
        or value.startswith("/")
        or value.startswith("./")
        or value.endswith("/")
        or "//" in value
    ):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    path = PurePosixPath(value)
    parts = path.parts
    if (
        not parts
        or any(part in {"", ".", ".."} for part in parts)
        or (len(parts[0]) == 2 and parts[0][1] == ":" and parts[0][0].isalpha())
        or path.as_posix() != value
    ):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    return value


def _safe_symlink_target(source: str, target: str) -> str:
    if (
        not isinstance(target, str)
        or not target
        or "\x00" in target
        or "\\" in target
        or target.startswith("/")
    ):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    first = PurePosixPath(target).parts[:1]
    if first and len(first[0]) == 2 and first[0][1] == ":" and first[0][0].isalpha():
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    combined = posixpath.normpath(
        posixpath.join(posixpath.dirname(source), target)
    )
    if combined in {"", ".", ".."} or combined.startswith("../"):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    return _safe_relative_path(combined)


def _workspace_entry_kind(entry: Mapping[str, object]) -> str:
    is_directory = entry.get("directory") is True
    has_symlink = "symlink" in entry
    has_content = "content" in entry
    if sum((is_directory, has_symlink, has_content)) != 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if is_directory:
        if set(entry) - {"path", "directory"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return "directory"
    if has_symlink:
        if set(entry) - {"path", "symlink"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return "symlink"
    if set(entry) - {"path", "mode", "content"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    mode = entry.get("mode", 0)
    if isinstance(mode, bool) or not isinstance(mode, int) or mode < 0 or mode > 0o111:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _object_ref_from_payload(entry.get("content"))
    return "file"


def _validate_workspace_entry(entry: Mapping[str, object]) -> str:
    value = entry.get("path")
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    path = _safe_relative_path(value)
    kind = _workspace_entry_kind(entry)
    if kind == "symlink":
        target = entry.get("symlink")
        if not isinstance(target, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _safe_symlink_target(path, target)
    return kind


def _validate_workspace_entries(entries: list[object]) -> dict[str, str]:
    kinds: dict[str, str] = {}
    collision_keys: dict[str, str] = {}
    values: dict[str, Mapping[str, object]] = {}
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        kind = _validate_workspace_entry(raw_entry)
        path = cast(str, raw_entry["path"])
        if path in kinds:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        collision_key = os.path.normcase(path)
        previous = collision_keys.get(collision_key)
        if previous is not None and previous != path:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        collision_keys[collision_key] = path
        kinds[path] = kind
        values[path] = raw_entry

    for path, kind in kinds.items():
        parts = path.split("/")
        for index in range(1, len(parts)):
            parent = "/".join(parts[:index])
            parent_kind = kinds.get(parent)
            if parent_kind is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if parent_kind != "directory":
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        if kind == "symlink":
            target = cast(str, values[path]["symlink"])
            logical_target = _safe_symlink_target(path, target)
            _resolve_workspace_entry_kind(logical_target, values, kinds)
    return kinds


def _resolve_workspace_entry_kind(
    path: str,
    entries: Mapping[str, Mapping[str, object]],
    kinds: Mapping[str, str],
) -> str:
    current = path
    seen: set[str] = set()
    while True:
        kind = kinds.get(current)
        if kind is None:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        if kind != "symlink":
            return kind
        if current in seen:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        seen.add(current)
        entry = entries[current]
        target = entry.get("symlink")
        if not isinstance(target, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        current = _safe_symlink_target(current, target)


async def _verify_restored_generation(
    root: Path,
    *,
    manifest: Mapping[str, object],
    object_store: ObjectStore,
    limits: SnapshotLimits,
) -> None:
    state_root = root / "state"
    if not state_root.is_dir():
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    namespace = manifest.get("namespace")
    tenant_id = manifest.get("tenant_id")
    if not isinstance(namespace, str) or not isinstance(tenant_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    restored_state = RuntimeState.from_root(state_root)
    try:
        await restored_state.initialize(
            namespace=namespace,
            tenant_id=tenant_id,
            read_only=True,
        )
    except AIError as error:
        if error.code is ErrorCode.STORAGE_NOT_FOUND:
            raise AIError(ErrorCode.STORAGE_CONFLICT) from error
        raise
    finally:
        if restored_state.ready:
            await restored_state.close()

    workspace_root = root / "workspace"
    workspace = manifest.get("workspace")
    if not isinstance(workspace, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    present = workspace.get("present")
    workspace_id = workspace.get("workspace_id")
    expected = workspace.get("entries")
    if (
        not isinstance(present, bool)
        or (workspace_id is not None and not isinstance(workspace_id, str))
        or not isinstance(expected, list)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not present:
        if workspace_id is not None or expected or workspace_root.exists():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return
    if workspace_id is None or not workspace_id:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not workspace_root.is_dir():
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    kinds = _validate_workspace_entries(expected)
    expected_by_path = {
        cast(str, entry["path"]): entry
        for entry in expected
        if isinstance(entry, Mapping)
    }
    actual = {
        path.relative_to(workspace_root).as_posix(): path
        for path in workspace_root.rglob("*")
    }
    if set(actual) != set(expected_by_path):
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if len(actual) > limits.max_entries:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)

    total_bytes = 0
    for relative, entry in expected_by_path.items():
        path = actual[relative]
        kind = kinds[relative]
        if kind == "directory":
            if path.is_symlink() or not path.is_dir():
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            continue
        if kind == "symlink":
            if not path.is_symlink() or os.readlink(path) != entry["symlink"]:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            try:
                resolved = path.resolve(strict=True)
            except (OSError, RuntimeError) as error:
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            if not _is_child_or_same(resolved, workspace_root):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            continue
        if path.is_symlink() or not path.is_file():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        content = _object_ref_from_payload(entry.get("content"))
        total_bytes += content.size
        if total_bytes > limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        digest, size = await _read_file_digest(path, content.size)
        if digest != content.digest or size != content.size:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        mode = entry.get("mode", 0)
        if path.stat().st_mode & 0o111 != mode:
            raise AIError(ErrorCode.STORAGE_CONFLICT)


def _read_generation_manifest(root: Path) -> Mapping[str, object]:
    try:
        payload = (root / "snapshot.json").read_bytes()
        value = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, Mapping) or value.get("kind") != "runtime-snapshot":
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return value


def _state_snapshot_usage(
    manifest: Mapping[str, object],
) -> tuple[int, int]:
    domains = manifest.get("domains")
    objects = manifest.get("objects")
    if not isinstance(domains, Mapping) or not isinstance(objects, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    entries = len(objects)
    object_bytes = 0
    for raw_domain in domains.values():
        if not isinstance(raw_domain, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for name in ("records", "aliases", "facts", "operations", "sequences"):
            values = raw_domain.get(name)
            if not isinstance(values, list):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            entries += len(values)
    for raw_object in objects:
        if not isinstance(raw_object, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        source = _object_ref_from_payload(raw_object.get("source"))
        content = _object_ref_from_payload(raw_object.get("content"))
        if source.digest != content.digest or source.size != content.size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        object_bytes += content.size
    return entries, object_bytes


def _workspace_snapshot_usage(
    workspace: Mapping[str, object],
) -> tuple[int, int]:
    entries = workspace.get("entries")
    if not isinstance(entries, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    total_bytes = 0
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if "content" in entry:
            total_bytes += _object_ref_from_payload(entry["content"]).size
    return len(entries), total_bytes


def _parse_state_manifest(payload: bytes) -> Mapping[str, object]:
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if (
        not isinstance(value, Mapping)
        or value.get("kind") != "runtime-state-snapshot"
        or value.get("format_version") != 1
        or not isinstance(value.get("domains"), Mapping)
    ):
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return value


def _same_file_snapshot(before: os.stat_result, after: os.stat_result, size: int) -> bool:
    return (
        before.st_size == size
        and after.st_size == size
        and before.st_mtime_ns == after.st_mtime_ns
        and before.st_ino == after.st_ino
    )


async def _file_chunks(path: Path):
    handle = await asyncio.to_thread(path.open, "rb")
    try:
        while True:
            chunk = await asyncio.to_thread(handle.read, 1024 * 1024)
            if not chunk:
                return
            yield chunk
    finally:
        await asyncio.to_thread(handle.close)


async def _verify_object(store: ObjectStore, ref: ObjectRef) -> None:
    digest = hashlib.sha256()
    size = 0
    async for chunk in store.open(ref.key):
        if not isinstance(chunk, bytes) or not chunk:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        size += len(chunk)
        if size > ref.size:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest.update(chunk)
    if size != ref.size or digest.hexdigest() != ref.digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _copy_object(
    source: ObjectStore,
    target: ObjectStore,
    ref: ObjectRef,
) -> None:
    if source is target:
        return
    await target.put(
        ref.key,
        source.open(ref.key),
        expected_size=ref.size,
        expected_digest=ref.digest,
    )


async def _write_object_file(
    store: ObjectStore,
    ref: ObjectRef,
    path: Path,
    *,
    max_bytes: int,
) -> None:
    if ref.size > max_bytes:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    digest = hashlib.sha256()
    size = 0
    handle = await asyncio.to_thread(path.open, "wb")
    try:
        async for chunk in store.open(ref.key):
            if not isinstance(chunk, bytes) or not chunk:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            size += len(chunk)
            if size > ref.size or size > max_bytes:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            digest.update(chunk)
            await asyncio.to_thread(handle.write, chunk)
        await asyncio.to_thread(handle.flush)
        await asyncio.to_thread(os.fsync, handle.fileno())
    finally:
        await asyncio.to_thread(handle.close)
    if size != ref.size or digest.hexdigest() != ref.digest:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


async def _read_file_digest(path: Path, limit: int) -> tuple[str, int]:
    try:
        return await asyncio.to_thread(_read_file_digest_sync, path, limit)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _read_file_digest_sync(path: Path, limit: int) -> tuple[str, int]:
    size = 0
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            size += len(chunk)
            if size > limit:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            digest.update(chunk)
    return digest.hexdigest(), size


def _is_child_or_same(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


async def _read_manifest(
    ref: ObjectRef,
    object_store: ObjectStore,
    limits: SnapshotLimits,
) -> Mapping[str, object]:
    if not isinstance(limits, SnapshotLimits):
        raise TypeError("limits must be SnapshotLimits")
    if ref.size > limits.max_bytes:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    payload = await read_object(
        object_store,
        ref.key,
        expected_digest=ref.digest,
        expected_size=ref.size,
    )
    try:
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if (
        not isinstance(value, Mapping)
        or value.get("kind") != "runtime-snapshot"
        or value.get("format_version") != 1
    ):
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return value


def _object_ref_payload(ref: ObjectRef) -> dict[str, JsonValue]:
    return {
        "store_id": ref.store_id,
        "key": ref.key,
        "digest": ref.digest,
        "size": ref.size,
    }


def _object_ref_from_payload(value: object) -> ObjectRef:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return ObjectRef(
            str(value["store_id"]),
            str(value["key"]),
            str(value["digest"]),
            int(value["size"]),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _digest_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


async def _one_chunk(value: bytes):
    yield value


async def _put_object(object_store: ObjectStore, key: str, value: bytes) -> None:
    digest = _digest_bytes(value)
    current = await object_store.stat(key)
    if current is not None:
        if current.digest != digest or current.size != len(value):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        await read_object(
            object_store,
            key,
            expected_digest=digest,
            expected_size=len(value),
        )
        return
    await object_store.put(
        key,
        _one_chunk(value),
        expected_size=len(value),
        expected_digest=digest,
    )


def _read_current(path: Path) -> Mapping[str, object] | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _write_current(path: Path, value: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError as error:
        temporary.unlink(missing_ok=True)
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


def _restored_runtime(root: Path, value: Mapping[str, object]) -> RestoredRuntime:
    return RestoredRuntime(
        cast(str, value["snapshot_digest"]),
        cast(str, value["namespace"]),
        cast(str, value["tenant_id"]),
        Path(cast(str, value["state_root"])),
        (
            None
            if value.get("workspace_root") is None
            else Path(cast(str, value["workspace_root"]))
        ),
        cast(str | None, value.get("workspace_id")),
        cast(str, value["generation"]),
    )


__all__ = [
    "RestoredRuntime",
    "RunSnapshot",
    "RuntimeSnapshot",
    "SnapshotLimits",
    "SnapshotTargetInspection",
    "snapshot_digest",
]
