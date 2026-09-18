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
from pathlib import Path
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
from .state import OfflineExclusiveStorage, RuntimeState

if TYPE_CHECKING:
    from ..workspace import Workspace

_logger = environ.get_logger("ai.runtime.snapshot")


@dataclass(frozen=True, slots=True)
class SnapshotLimits:
    """Local admission limits for portable snapshot operations."""

    max_entries: int
    max_bytes: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_entries, bool)
            or not isinstance(self.max_entries, int)
            or self.max_entries < 1
            or isinstance(self.max_bytes, bool)
            or not isinstance(self.max_bytes, int)
            or self.max_bytes < 1
        ):
            raise ValueError("snapshot limits must be positive integers")


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
                state_ref = await state.export_snapshot(object_store=object_store)
                if state_ref.size > limits.max_bytes:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
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
                if len(payload) > limits.max_bytes:
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
        state_objects = state_manifest.get("objects", [])
        if not isinstance(state_objects, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        workspace = manifest.get("workspace")
        if (
            not isinstance(workspace, list)
            or len(workspace) + len(state_objects) > limits.max_entries
        ):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        _validate_workspace_entries(workspace)
        for raw_object in state_objects:
            if not isinstance(raw_object, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            source = _object_ref_from_payload(raw_object.get("source"))
            content = _object_ref_from_payload(raw_object.get("content"))
            if source.digest != content.digest or source.size != content.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            await read_object(
                object_store,
                content.key,
                expected_digest=content.digest,
                expected_size=content.size,
            )
        for entry in workspace:
            if not isinstance(entry, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if "symlink" not in entry:
                content = _object_ref_from_payload(entry.get("content"))
                await read_object(
                    object_store,
                    content.key,
                    expected_digest=content.digest,
                    expected_size=content.size,
                )

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
        refs.extend(
            _object_ref_from_payload(entry["content"])
            for entry in cast(list[Mapping[str, object]], manifest["workspace"])
            if "symlink" not in entry
        )
        for item in refs:
            value = await read_object(
                source_store,
                item.key,
                expected_digest=item.digest,
                expected_size=item.size,
            )
            await _put_object(target_store, item.key, value)
        payload = await read_object(
            source_store,
            ref.key,
            expected_digest=ref.digest,
            expected_size=ref.size,
        )
        await _put_object(target_store, ref.key, payload)
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
            except AIError:
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
) -> list[dict[str, JsonValue]]:
    if workspace is None:
        return []
    root = workspace.root.resolve()
    excluded = tuple(
        path.resolve()
        for path in (*state.local_paths(), *object_store.local_paths())
    )
    if any(path == root for path in excluded):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    entries: list[dict[str, JsonValue]] = []
    total_bytes = 0
    for path in sorted(root.rglob("*")):
        try:
            resolved = path.resolve(strict=False)
        except RuntimeError as error:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
        if any(_is_child_or_same(resolved, item) for item in excluded):
            if path.is_symlink():
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            continue
        relative = path.relative_to(root)
        if path.is_symlink():
            try:
                target = os.readlink(path)
                target_path = (path.parent / target).resolve(strict=False)
            except (OSError, RuntimeError) as error:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
            if (
                Path(target).is_absolute()
                or not _is_child_or_same(target_path, root)
                or any(_is_child_or_same(target_path, item) for item in excluded)
                or not target_path.exists()
            ):
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            entries.append(
                {
                    "path": relative.as_posix(),
                    "symlink": target,
                }
            )
            if len(entries) > limits.max_entries:
                raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
            continue
        if path.is_dir():
            continue
        if not path.is_file():
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        try:
            before = path.stat()
            value = path.read_bytes()
            after = path.stat()
        except OSError as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        if (
            before.st_size != after.st_size
            or before.st_mtime_ns != after.st_mtime_ns
            or before.st_ino != after.st_ino
        ):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        total_bytes += len(value)
        if len(entries) >= limits.max_entries or total_bytes > limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        digest = _digest_bytes(value)
        key = f"v1/runtime-workspace/{digest}"
        await _put_object(object_store, key, value)
        entries.append(
            {
                "path": relative.as_posix(),
                "mode": after.st_mode & 0o111,
                "content": {
                    "store_id": object_store.store_id,
                    "key": key,
                    "digest": digest,
                    "size": len(value),
                },
            }
        )
    return entries


async def _restore_workspace(
    raw: object,
    *,
    object_store: ObjectStore,
    target: Path,
    limits: SnapshotLimits,
) -> Path | None:
    if not isinstance(raw, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if not raw:
        return None
    if len(raw) > limits.max_entries:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    seen: set[str] = set()
    links: list[tuple[Path, str]] = []
    relative_paths: list[str] = []
    for entry in raw:
        if not isinstance(entry, Mapping) or not isinstance(entry.get("path"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_workspace_entry(entry)
        relative = _safe_relative_path(entry["path"])
        if relative in seen:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        seen.add(relative)
        relative_paths.append(relative)
    for relative in relative_paths:
        parts = relative.split("/")
        if any("/".join(parts[:index]) in seen for index in range(1, len(parts))):
            raise AIError(ErrorCode.STORAGE_CONFLICT)
    target.mkdir(parents=True, exist_ok=False)
    total_bytes = 0
    seen.clear()
    for entry in raw:
        relative = _safe_relative_path(cast(str, entry["path"]))
        seen.add(relative)
        destination = target / relative
        if "symlink" in entry:
            links.append((destination, cast(str, entry["symlink"])))
            continue
        content = _object_ref_from_payload(entry.get("content"))
        total_bytes += content.size
        if total_bytes > limits.max_bytes:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(
            await read_object(
                object_store,
                content.key,
                expected_digest=content.digest,
                expected_size=content.size,
            )
        )
        if int(entry.get("mode", 0)):
            os.chmod(destination, 0o755)
    for destination, link_target in links:
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            resolved = (destination.parent / link_target).resolve(strict=False)
        except RuntimeError as error:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
        if not _is_child_or_same(resolved, target) or not os.path.lexists(resolved):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        destination.symlink_to(link_target)
    for destination, _ in links:
        try:
            resolved = destination.resolve(strict=False)
        except RuntimeError as error:
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED) from error
        if not _is_child_or_same(resolved, target):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    return target


def _safe_relative_path(value: str) -> str:
    path = Path(value)
    if (
        not value
        or path.is_absolute()
        or "\x00" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    return path.as_posix()


def _validate_workspace_entry(entry: Mapping[str, object]) -> None:
    value = entry.get("path")
    if not isinstance(value, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _safe_relative_path(value)
    if "symlink" in entry:
        target = entry["symlink"]
        if not isinstance(target, str) or not target or Path(target).is_absolute():
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        combined = posixpath.normpath(
            posixpath.join(posixpath.dirname(value), target)
        )
        if combined == ".." or combined.startswith("../"):
            raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
        return
    if "content" not in entry:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _validate_workspace_entries(entries: list[object]) -> None:
    paths: set[str] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_workspace_entry(raw_entry)
        path = cast(str, raw_entry["path"])
        if path in paths:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        paths.add(path)
    for path in paths:
        parts = path.split("/")
        if any("/".join(parts[:index]) in paths for index in range(1, len(parts))):
            raise AIError(ErrorCode.STORAGE_CONFLICT)


async def _verify_restored_generation(
    root: Path,
    *,
    manifest: Mapping[str, object],
    object_store: ObjectStore,
    limits: SnapshotLimits,
) -> None:
    state_root = root / "state"
    if not state_root.is_dir():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    workspace_root = root / "workspace"
    expected = manifest.get("workspace")
    if not isinstance(expected, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    _validate_workspace_entries(expected)
    expected_by_path: dict[str, Mapping[str, object]] = {}
    for entry in expected:
        if not isinstance(entry, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        path = cast(str, entry["path"])
        if path in expected_by_path:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        expected_by_path[path] = entry
    if not expected_by_path:
        if workspace_root.exists():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return
    if not workspace_root.is_dir():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    actual = {
        path.relative_to(workspace_root).as_posix(): path
        for path in workspace_root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    if set(actual) != set(expected_by_path):
        raise AIError(ErrorCode.STORAGE_CONFLICT)
    if len(actual) > limits.max_entries:
        raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
    for relative, entry in expected_by_path.items():
        path = actual[relative]
        if "symlink" in entry:
            if not path.is_symlink() or os.readlink(path) != entry["symlink"]:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            continue
        if path.is_symlink():
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        content = _object_ref_from_payload(entry.get("content"))
        digest, size = await _read_file_digest(path, limits.max_bytes)
        if digest != content.digest or size != content.size:
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


async def _read_file_digest(path: Path, limit: int) -> tuple[str, int]:
    size = 0
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise AIError(ErrorCode.SNAPSHOT_UNSUPPORTED)
                digest.update(chunk)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
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
