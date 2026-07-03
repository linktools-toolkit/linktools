import asyncio
import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from linktools.ai.resource_store.database import DatabaseBackend, _RawRow
from linktools.ai.resource_store.protocols import DeleteOp, MoveOp, PutOp, ResourceBackend


@dataclass
class _CacheConfig:
    enabled: bool = False


@dataclass
class _FakeRedis:
    """No-op stand-in -- disabled by default so tests exercise the redis-absent path
    (matching how registry_store's own tests covered the same fallback)."""

    config: _CacheConfig = field(default_factory=_CacheConfig)
    _store: "dict[str, str]" = field(default_factory=dict)

    async def get(self, key: str) -> "str | None":
        return self._store.get(key)

    async def incr(self, key: str) -> int:
        value = int(self._store.get(key, "0")) + 1
        self._store[key] = str(value)
        return value

    async def delete(self, key: str) -> None:
        self._store.pop(key, None)

    async def try_acquire(self, key: str, value: str, ttl: int) -> bool:
        return True

    async def release_if_owner(self, key: str, value: str) -> bool:
        return True


class _InMemoryDatabaseBackend(DatabaseBackend):
    """Test double implementing the _raw_* contract with a plain dict, so this test
    file exercises DatabaseBackend's concrete cache/idempotency logic without a real
    SQL dependency."""

    def __init__(self, redis) -> None:
        super().__init__(redis=redis, workspace_root=None)
        self._rows: "dict[str, _RawRow]" = {}
        self._next_id = 1

    def _checksum(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    async def _raw_get(self, path: str) -> "_RawRow | None":
        row = self._rows.get(path)
        return row if row is not None and row.status == "active" else None

    async def _raw_get_by_name(self, namespace: str, name: str) -> "list[_RawRow]":
        prefix, suffix = f"/{namespace}/", f"/{name}"
        return [r for p, r in self._rows.items() if p.startswith(prefix) and p.endswith(suffix) and r.status == "active"]

    async def _raw_upsert(self, path: str, content: str, checksum: str, updated_by: str) -> "tuple[int, int, bool]":
        existing = self._rows.get(path)
        if existing is not None and existing.status == "active" and self._checksum(existing.content) == checksum:
            return existing.row_id, existing.version, False
        row_id = existing.row_id if existing else self._next_id
        if existing is None:
            self._next_id += 1
        version = (existing.version + 1) if existing else 1
        self._rows[path] = _RawRow(row_id=row_id, path=path, content=content, version=version, status="active", updated_at=datetime.now())
        return row_id, version, True

    async def _raw_delete(self, path: str, updated_by: str) -> bool:
        existing = self._rows.get(path)
        if existing is None or existing.status != "active":
            return False
        self._rows[path] = _RawRow(row_id=existing.row_id, path=path, content=existing.content, version=existing.version + 1, status="deleted", updated_at=datetime.now())
        return True

    async def _raw_move(self, src_path: str, dst_path: str, updated_by: str) -> "tuple[int, int] | None":
        src = self._rows.get(src_path)
        dst = self._rows.get(dst_path)
        src_active = src is not None and src.status == "active"
        dst_active = dst is not None and dst.status == "active"
        if dst_active and not src_active:
            return dst.row_id, dst.version
        if not src_active:
            return None
        new_version = src.version + 1
        self._rows[dst_path] = _RawRow(row_id=src.row_id, path=dst_path, content=src.content, version=new_version, status="active", updated_at=datetime.now())
        self._rows[src_path] = _RawRow(row_id=src.row_id, path=src_path, content=src.content, version=src.version, status="deleted", updated_at=datetime.now())
        return src.row_id, new_version

    async def _raw_list_since(self, since) -> "list[_RawRow]":
        if since is None:
            return list(self._rows.values())
        return [r for r in self._rows.values() if r.updated_at >= since]

    async def _raw_apply_batch(self, ops, *, expected_revision, updated_by) -> "tuple[str, list[_RawRow]]":
        # Return only the rows touched by THIS batch's ops -- not the whole table.
        # (Old CapabilityStore.save_batch's `final_rows` returned everything under one
        # capability_id, including untouched files, because its batch was scoped to a
        # single capability. apply_batch here can span arbitrary unrelated paths, so
        # "the complete state of one capability" no longer has a clean analog -- only
        # per-op results generalize.)
        touched_paths = set()
        for op in ops:
            if isinstance(op, PutOp):
                await self._raw_upsert(op.path, op.content, self._checksum(op.content), updated_by)
                touched_paths.add(op.path)
            elif isinstance(op, DeleteOp):
                await self._raw_delete(op.path, updated_by)
                touched_paths.add(op.path)
            elif isinstance(op, MoveOp):
                await self._raw_move(op.src_path, op.dst_path, updated_by)
                touched_paths.add(op.dst_path)
        return "rev-1", [self._rows[p] for p in touched_paths if p in self._rows]


def test_backend_satisfies_protocol():
    assert isinstance(_InMemoryDatabaseBackend(_FakeRedis()), ResourceBackend)


def test_put_then_get_roundtrip():
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        await backend.put("/skill/a/SKILL.md", "hello")
        result = await backend.get("/skill/a/SKILL.md")
        assert result is not None
        assert result.content == "hello"
        assert result.version == 1

    asyncio.run(run())


def test_put_same_content_is_idempotent():
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        await backend.put("/skill/a/SKILL.md", "hello")
        second = await backend.put("/skill/a/SKILL.md", "hello")
        assert second.version == 1

    asyncio.run(run())


def test_delete_nonexistent_path_is_idempotent_no_op():
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        assert await backend.delete("/skill/never/SKILL.md") is False

    asyncio.run(run())


def test_move_already_applied_is_idempotent():
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        await backend.put("/skill/a/SKILL.md", "hello")
        await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        second = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert second is not None
        assert second.content == "hello"

    asyncio.run(run())


def test_get_by_name_result_stays_resident_after_lru_pressure():
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        await backend.put("/skill/a/SKILL.md", "primary content")
        await backend.get_by_name("skill", "SKILL.md")  # touches the resident tier

        for i in range(300):
            await backend.put(f"/skill/filler-{i}/notes.md", "x")

        # A plain get() (not via get_by_name) for the filler files went through the
        # bounded LRU (cap 256); the get_by_name-touched entry must survive regardless.
        assert backend._resident.get("/skill/a/SKILL.md") == "primary content"
        assert (await backend.get("/skill/a/SKILL.md")).content == "primary content"

    asyncio.run(run())


def test_propfind_and_list_since_and_apply_batch_and_revision():
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")

        listed = await backend.propfind("/skill/")
        assert {r.path for r in listed} == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

        since = await backend.list_since(None)
        assert len(since) == 2

        results = await backend.apply_batch([
            PutOp(path="/skill/c/SKILL.md", content="c"),
            DeleteOp(path="/skill/a/SKILL.md"),
        ])
        assert {r.path for r in results} == {"/skill/c/SKILL.md"}

        revision = await backend.get_revision()
        assert isinstance(revision, int)

    asyncio.run(run())


def test_get_at_version_returns_historical_content_via_disk_cache(tmp_path):
    async def run():
        backend = _InMemoryDatabaseBackend(_FakeRedis())
        backend._workspace_root = tmp_path
        await backend.put("/skill/a/SKILL.md", "v1")
        await backend.put("/skill/a/SKILL.md", "v2")
        v1 = await backend.get_at_version("/skill/a/SKILL.md", 1)
        assert v1 is not None
        assert v1.content == "v1"

    asyncio.run(run())
