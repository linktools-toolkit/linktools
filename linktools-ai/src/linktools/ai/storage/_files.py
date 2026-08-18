#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Root-contained atomic file primitives."""

import hashlib
import json
import os
import re
import shutil
import tempfile
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from pathlib import Path

from ..core import JsonValue
from ..errors import AIError, ErrorCode, InvalidStoragePathError


def validate_root_path(root: Path, relative: str) -> Path:
    if not relative or "\x00" in relative or "\\" in relative:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from exc
    return candidate


def read_bytes(path: Path) -> bytes:
    return path.read_bytes()


def write_bytes_atomic(path: Path, value: bytes, *, fsync: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    replaced = False
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            if fsync:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        replaced = True
        if fsync:
            sync_directory(path.parent)
    except BaseException as error:
        Path(temporary).unlink(missing_ok=True)
        if replaced:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
        raise


def read_json(path: Path) -> "dict[str, JsonValue]":
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("JSON root must be an object")
    return value


def write_json_atomic(path: Path, value: "dict[str, JsonValue]", *, fsync: bool = False) -> None:
    write_bytes_atomic(
        path,
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"),
        fsync=fsync,
    )


def unlink_if_exists(path: Path) -> None:
    path.unlink(missing_ok=True)


def sync_directory(path: Path) -> None:
    """Flush a directory when the platform exposes directory descriptors."""
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        try:
            os.fsync(descriptor)
        except OSError:
            return
    finally:
        os.close(descriptor)


def _directory_chain(path: Path, root: Path) -> tuple[Path, ...]:
    root = root.resolve()
    current = path.resolve()
    try:
        current.relative_to(root)
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    values: list[Path] = []
    while True:
        values.append(current)
        if current == root:
            return tuple(values)
        current = current.parent


def _sync_directories(paths: Collection[Path]) -> None:
    for path in sorted(set(paths), key=lambda value: len(value.parts), reverse=True):
        sync_directory(path)


class FilesystemJournal:
    """Shared crash-safe journal for granular filesystem stores."""

    def __init__(self, root: Path, *, error_code: ErrorCode) -> None:
        self._root = root
        self._transaction = root / ".txn"
        self._error_code = error_code

    def stage(
        self,
        writes: Mapping[str, bytes],
        deletes: Collection[str],
        *,
        base_generation: int,
        target_generation: int,
    ) -> dict[str, object]:
        if target_generation != base_generation + 1:
            raise AIError(self._error_code)
        try:
            if (self._transaction / "commit").exists():
                raise AIError(self._error_code)
            if self._transaction.exists():
                shutil.rmtree(self._transaction)
                sync_directory(self._root)
            stage = self._transaction / "stage"
            stage.mkdir(parents=True)
            write_entries: list[dict[str, str]] = []
            for relative, content in writes.items():
                path = validate_root_path(
                    stage,
                    _safe_relative(relative, self._error_code),
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)
                _sync_file(path)
                write_entries.append(
                    {
                        "path": relative,
                        "sha256": _sha256(content),
                    }
                )
            _sync_tree(stage)
            plan: dict[str, object] = {
                "journal_version": 1,
                "base_generation": base_generation,
                "target_generation": target_generation,
                "writes": write_entries,
                "deletes": sorted(deletes),
            }
            self.validate(plan, base_generation=base_generation)
            _write_journal_json(self._transaction / "plan.json", plan)
            sync_directory(self._transaction)
            _write_journal_text(self._transaction / "commit", "1")
            sync_directory(self._transaction)
            sync_directory(self._root)
            return plan
        except AIError:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise AIError(self._error_code) from error

    def validate(
        self,
        plan: Mapping[str, object],
        *,
        base_generation: int | None = None,
    ) -> None:
        if plan.get("journal_version") != 1:
            raise AIError(self._error_code)
        base = plan.get("base_generation")
        target = plan.get("target_generation")
        if not isinstance(base, int) or not isinstance(target, int) or target != base + 1:
            raise AIError(self._error_code)
        if base_generation is not None and base != base_generation:
            raise AIError(self._error_code)
        writes = plan.get("writes")
        deletes = plan.get("deletes")
        if not isinstance(writes, list) or not isinstance(deletes, list):
            raise AIError(self._error_code)
        written_paths: set[str] = set()
        deleted_paths: set[str] = set()
        for item in writes:
            if not isinstance(item, Mapping):
                raise AIError(self._error_code)
            relative = _safe_relative(str(item.get("path", "")), self._error_code)
            digest = item.get("sha256")
            if (
                relative != str(item.get("path"))
                or relative in written_paths
                or not isinstance(digest, str)
                or len(digest) != 64
            ):
                raise AIError(self._error_code)
            try:
                int(digest, 16)
            except ValueError as error:
                raise AIError(self._error_code) from error
            written_paths.add(relative)
        for value in deletes:
            relative = _safe_relative(str(value), self._error_code)
            if relative in deleted_paths or relative in written_paths:
                raise AIError(self._error_code)
            deleted_paths.add(relative)

    def publish(self, plan: Mapping[str, object]) -> None:
        affected: set[Path] = set()
        try:
            self.validate(plan)
            stage = self._transaction / "stage"
            writes = plan["writes"]
            deletes = plan["deletes"]
            if not isinstance(writes, list) or not isinstance(deletes, list):
                raise AIError(self._error_code)
            for item in writes:
                if not isinstance(item, Mapping):
                    raise AIError(self._error_code)
                relative = _safe_relative(str(item["path"]), self._error_code)
                source = validate_root_path(stage, relative)
                destination = validate_root_path(self._root, relative)
                affected.update(_directory_chain(destination.parent, self._root))
                destination.parent.mkdir(parents=True, exist_ok=True)
                if source.is_file():
                    os.replace(source, destination)
                elif not destination.is_file() or _sha256(destination.read_bytes()) != str(item["sha256"]):
                    raise AIError(self._error_code)
            for value in deletes:
                relative = _safe_relative(str(value), self._error_code)
                path = validate_root_path(self._root, relative)
                affected.update(_directory_chain(path.parent, self._root))
                path.unlink(missing_ok=True)
        except AIError:
            raise
        except (OSError, TypeError, ValueError) as error:
            raise AIError(self._error_code) from error
        finally:
            _sync_directories(affected)

    def complete(self) -> None:
        try:
            if self._transaction.exists():
                shutil.rmtree(self._transaction)
                sync_directory(self._root)
        except OSError as error:
            raise AIError(self._error_code) from error

    def recover(
        self,
        read_generation: Callable[[], int],
        write_generation: Callable[[int], None],
    ) -> None:
        marker = self._transaction / "commit"
        if not marker.exists():
            if self._transaction.exists():
                shutil.rmtree(self._transaction)
                sync_directory(self._root)
            return
        try:
            plan = _read_journal_json(self._transaction / "plan.json")
            current = read_generation()
            base = plan.get("base_generation")
            self.validate(
                plan,
                base_generation=current if current == base else None,
            )
            if current not in {int(plan["base_generation"]), int(plan["target_generation"])}:
                raise AIError(self._error_code)
            self.publish(plan)
            target = int(plan["target_generation"])
            if current != target:
                write_generation(target)
                sync_directory(self._root)
            shutil.rmtree(self._transaction)
            sync_directory(self._root)
        except AIError:
            raise
        except (OSError, KeyError, TypeError, ValueError) as error:
            raise AIError(self._error_code) from error


def atomic_write_bytes(path: "str | Path", value: bytes) -> None:
    write_bytes_atomic(Path(path), value)


def atomic_write_json(path: "str | Path", value: JsonValue) -> None:
    atomic_write_bytes(
        path,
        json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8"),
    )


_STORAGE_ID = re.compile(r"[A-Za-z0-9._-]+")


@dataclass(frozen=True, slots=True)
class StorageId:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "StorageId":
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) > 255
            or raw in {".", ".."}
            or _STORAGE_ID.fullmatch(raw) is None
        ):
            raise InvalidStoragePathError(f"invalid storage id: {raw!r}")
        return cls(raw)


@dataclass(frozen=True, slots=True)
class StoragePath:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "StoragePath":
        parts = raw.split("/") if isinstance(raw, str) else []
        if (
            not isinstance(raw, str)
            or not raw
            or len(raw) > 512
            or any(_STORAGE_ID.fullmatch(part) is None for part in parts)
        ):
            raise InvalidStoragePathError(f"invalid storage path: {raw!r}")
        return cls(raw)


@dataclass(frozen=True, slots=True)
class Sha256Digest:
    value: str

    @classmethod
    def parse(cls, raw: str) -> "Sha256Digest":
        if not isinstance(raw, str) or re.fullmatch(r"[0-9a-f]{64}", raw) is None:
            raise InvalidStoragePathError(f"invalid sha256 digest: {raw!r}")
        return cls(raw)


def safe_child(
    root: "str | Path",
    *validated_parts: "StorageId | StoragePath | Sha256Digest | str",
) -> Path:
    root_path = Path(root).resolve(strict=False)
    values = tuple(
        part.value if isinstance(part, (StorageId, StoragePath, Sha256Digest)) else part for part in validated_parts
    )
    candidate = root_path.joinpath(*values).resolve(strict=False)
    try:
        candidate.relative_to(root_path)
    except ValueError as error:
        raise InvalidStoragePathError(f"path escapes storage root: {candidate}") from error
    return candidate


__all__ = [
    "FilesystemJournal",
    "Sha256Digest",
    "StorageId",
    "StoragePath",
    "atomic_write_bytes",
    "atomic_write_json",
    "read_bytes",
    "read_json",
    "safe_child",
    "sync_directory",
    "unlink_if_exists",
    "validate_root_path",
    "write_bytes_atomic",
    "write_json_atomic",
]


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_relative(value: str, error_code: ErrorCode) -> str:
    path = Path(value)
    if (
        path.is_absolute()
        or not value
        or "\x00" in value
        or "\\" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise AIError(error_code)
    return value


def _write_journal_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    _sync_file(path)


def _write_journal_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")
    _sync_file(path)


def _read_journal_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("filesystem journal JSON must be an object")
    return value


def _sync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _sync_tree(root: Path) -> None:
    for path in sorted((value for value in root.rglob("*") if value.is_dir()), reverse=True):
        sync_directory(path)
    sync_directory(root)
