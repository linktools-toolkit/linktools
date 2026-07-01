from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class InMemoryCapabilityRepository:
    """Reference CapabilityRepositoryProtocol implementation backed by a plain dict.

    Not for production use (no persistence, no concurrency control) — intended for
    linktools-ai's own tests and as a starting point for lightweight consumers
    (e.g. a quant-agent backtest harness) that don't need MySQL.
    """

    rows: dict[tuple[str, str], dict[str, Any]] = field(default_factory=dict)
    _next_id: int = 1

    async def upsert_file(self, *, kind, file_path, content, checksum, updated_by):
        key = (kind, file_path)
        existing = self.rows.get(key)
        if existing is not None and existing.get("checksum") == checksum:
            return existing["id"], existing["version"], False
        version = (existing["version"] + 1) if existing else 1
        file_id = existing["id"] if existing else self._next_id
        if existing is None:
            self._next_id += 1
        self.rows[key] = {
            "id": file_id, "kind": kind, "file_path": file_path, "content": content,
            "checksum": checksum, "version": version, "status": "active",
            "updated_by": updated_by, "updated_at": datetime.now(),
        }
        return file_id, version, True

    async def tombstone_file(self, kind, file_path, *, checksum, updated_by):
        key = (kind, file_path)
        existing = self.rows.get(key)
        if existing is None or existing.get("status") == "deleted":
            return False
        existing["status"] = "deleted"
        existing["checksum"] = checksum
        existing["updated_by"] = updated_by
        existing["version"] += 1
        return True

    async def apply_batch(self, *, kind, capability_id, primary_rel_path, primary_content,
                           supplementary_files, deleted_rel_paths, expected_revision, updated_by):
        raise NotImplementedError("apply_batch is exercised via integration tests with a real repository")

    async def list_capabilities_active(self, kind=None):
        return [row for row in self.rows.values() if row["status"] == "active" and (kind is None or row["kind"] == kind)]

    async def capability_exists(self, kind, capability_id):
        prefix = f"{capability_id}/"
        return any(k[0] == kind and k[1].startswith(prefix) for k in self.rows)

    async def get_file(self, kind, file_path):
        row = self.rows.get((kind, file_path))
        return dict(row) if row and row["status"] == "active" else None

    async def get_file_at_version(self, kind, file_path, version):
        row = self.rows.get((kind, file_path))
        return dict(row) if row and row["version"] == version else None

    async def get_primary_files(self, kind, primary_rel):
        return [row for (k, fp), row in self.rows.items() if k == kind and fp.endswith(f"/{primary_rel}")]

    async def list_files(self, kind, capability_id):
        prefix = f"{capability_id}/"
        return [row for (k, fp), row in self.rows.items() if k == kind and fp.startswith(prefix)]

    async def list_file_states(self, kind, capability_id):
        return await self.list_files(kind, capability_id)

    async def list_files_since(self, since):
        if since is None:
            return list(self.rows.values())
        return [row for row in self.rows.values() if row["updated_at"] >= since]

    async def delete_files_all(self, kind, capability_id):
        prefix = f"{capability_id}/"
        keys = [k for k in self.rows if k[0] == kind and k[1].startswith(prefix)]
        for k in keys:
            del self.rows[k]
        return len(keys)

    async def restore_builtin_files(self, kind, capability_id):
        return 0

    async def move_files(self, kind, old_capability_id, new_capability_id):
        old_prefix, new_prefix = f"{old_capability_id}/", f"{new_capability_id}/"
        keys = [k for k in self.rows if k[0] == kind and k[1].startswith(old_prefix)]
        for k in keys:
            row = self.rows.pop(k)
            new_fp = new_prefix + k[1][len(old_prefix):]
            row["file_path"] = new_fp
            self.rows[(kind, new_fp)] = row
        return len(keys)

    async def was_capability_renamed(self, kind, old_capability_id, new_capability_id):
        return False

    async def delete_file(self, kind, file_path):
        return self.rows.pop((kind, file_path), None) is not None

    async def rename_file(self, kind, old_file_path, new_file_path):
        row = self.rows.pop((kind, old_file_path), None)
        if row is None:
            return None
        row["file_path"] = new_file_path
        row["version"] += 1
        self.rows[(kind, new_file_path)] = row
        return row["id"], row["version"]


@dataclass
class _CacheConfig:
    enabled: bool = True


@dataclass
class InMemoryCapabilityCache:
    """Reference CapabilityCacheProtocol implementation — a plain dict, no TTL enforcement."""

    config: _CacheConfig = field(default_factory=_CacheConfig)
    _store: dict[str, str] = field(default_factory=dict)
    _locks: dict[str, str] = field(default_factory=dict)

    async def get(self, key: str) -> str | None:
        return self._store.get(key)

    async def incr(self, key: str) -> int:
        value = int(self._store.get(key, "0")) + 1
        self._store[key] = str(value)
        return value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def setex(self, key: str, ttl: int, value: str) -> None:
        del ttl
        self._store[key] = value

    async def try_acquire(self, key: str, value: str, ttl: int) -> bool:
        del ttl
        if key in self._locks:
            return False
        self._locks[key] = value
        return True

    async def release_if_owner(self, key: str, value: str) -> bool:
        if self._locks.get(key) == value:
            del self._locks[key]
            return True
        return False
