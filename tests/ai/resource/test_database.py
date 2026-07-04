import asyncio
import hashlib
from datetime import datetime, timedelta
from fnmatch import fnmatch

from linktools.ai.resource.database import DatabaseBackend, _RawRow
from linktools.ai.resource.protocols import DeleteOp, MoveOp, PutOp, ResourceBackend


class _InMemoryDatabaseBackend(DatabaseBackend):
    """Test double implementing the _raw_* contract with a plain dict, so this test
    file exercises DatabaseBackend's concrete cache/idempotency logic without a real
    SQL dependency."""

    def __init__(self, cluster_lock) -> None:
        super().__init__(cluster_lock=cluster_lock, cache_dir=None)
        self._rows: "dict[str, _RawRow]" = {}
        self._history: "dict[str, dict[int, _RawRow]]" = {}
        self._next_id = 1

    def _checksum(self, content: str) -> str:
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _record_version(self, row: "_RawRow") -> None:
        self._history.setdefault(row.path, {})[row.version] = row

    async def _raw_get(self, path: str, version: "int | None" = None) -> "_RawRow | None":
        if version is None:
            row = self._rows.get(path)
            return row if row is not None and row.status == "active" else None
        return self._history.get(path, {}).get(version)

    async def _raw_upsert(self, path: str, content: str, checksum: str, updated_by: str) -> "tuple[int, int, bool]":
        existing = self._rows.get(path)
        if existing is not None and existing.status == "active" and self._checksum(existing.content) == checksum:
            return existing.row_id, existing.version, False
        row_id = existing.row_id if existing else self._next_id
        if existing is None:
            self._next_id += 1
        version = (existing.version + 1) if existing else 1
        self._rows[path] = _RawRow(row_id=row_id, path=path, content=content, version=version, status="active", updated_at=datetime.now())
        self._record_version(self._rows[path])
        return row_id, version, True

    async def _raw_delete(self, path: str, updated_by: str) -> bool:
        existing = self._rows.get(path)
        if existing is None or existing.status != "active":
            return False
        self._rows[path] = _RawRow(row_id=existing.row_id, path=path, content=existing.content, version=existing.version + 1, status="deleted", updated_at=datetime.now())
        self._record_version(self._rows[path])
        return True

    async def _raw_move(self, src_path: str, dst_path: str, updated_by: str) -> "tuple[int, int, bool] | None":
        src = self._rows.get(src_path)
        dst = self._rows.get(dst_path)
        src_active = src is not None and src.status == "active"
        dst_active = dst is not None and dst.status == "active"
        if dst_active and not src_active:
            return dst.row_id, dst.version, False
        if not src_active:
            return None
        new_version = src.version + 1
        self._rows[dst_path] = _RawRow(row_id=src.row_id, path=dst_path, content=src.content, version=new_version, status="active", updated_at=datetime.now())
        self._rows[src_path] = _RawRow(row_id=src.row_id, path=src_path, content=src.content, version=src.version, status="deleted", updated_at=datetime.now())
        self._record_version(self._rows[dst_path])
        self._record_version(self._rows[src_path])
        return src.row_id, new_version, True

    async def _raw_list(self, *, pattern=None, since=None, include_deleted=False) -> "list[_RawRow]":
        rows = self._rows.values()
        if pattern is not None:
            rows = [r for r in rows if fnmatch(r.path, pattern)]
        if since is not None:
            rows = [r for r in rows if r.updated_at >= since]
        if not include_deleted:
            rows = [r for r in rows if r.status == "active"]
        return list(rows)

    async def _raw_apply_batch(self, ops, *, expected_revision, updated_by) -> "tuple[str, list[_RawRow], bool]":
        # Return only the rows touched by THIS batch's ops -- not the whole table.
        # (Old CapabilityStore.save_batch's `final_rows` returned everything under one
        # capability_id, including untouched files, because its batch was scoped to a
        # single capability. apply_batch here can span arbitrary unrelated paths, so
        # "the complete state of one capability" no longer has a clean analog -- only
        # per-op results generalize.)
        touched_paths = set()
        changed = False
        for op in ops:
            if isinstance(op, PutOp):
                _, _, op_changed = await self._raw_upsert(op.path, op.content, self._checksum(op.content), updated_by)
                changed = changed or op_changed
                touched_paths.add(op.path)
            elif isinstance(op, DeleteOp):
                op_changed = await self._raw_delete(op.path, updated_by)
                changed = changed or op_changed
                touched_paths.add(op.path)
            elif isinstance(op, MoveOp):
                result = await self._raw_move(op.src_path, op.dst_path, updated_by)
                if result is not None:
                    changed = changed or result[2]
                touched_paths.add(op.dst_path)
        return "rev-1", [self._rows[p] for p in touched_paths if p in self._rows], changed


def test_backend_satisfies_protocol():
    assert isinstance(_InMemoryDatabaseBackend(None), ResourceBackend)


def test_put_then_get_roundtrip():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "hello")
        result = await backend.get("/skill/a/SKILL.md")
        assert result is not None
        assert result.content == "hello"
        assert result.version == 1

    asyncio.run(run())


def test_put_same_content_is_idempotent():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "hello")
        second = await backend.put("/skill/a/SKILL.md", "hello")
        assert second.version == 1

    asyncio.run(run())


def test_delete_nonexistent_path_is_idempotent_no_op():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        assert await backend.delete("/skill/never/SKILL.md") is False

    asyncio.run(run())


def test_move_already_applied_is_idempotent():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "hello")
        await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        second = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert second is not None
        assert second.content == "hello"

    asyncio.run(run())


def test_move_already_applied_does_not_bump_revision():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "hello")
        await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        revision_after_first_move = await backend.revision()

        second = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert second is not None

        revision_after_second_move = await backend.revision()
        assert revision_after_second_move == revision_after_first_move

    asyncio.run(run())


def test_move_commits_destination_state_before_returning():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "hello")
        moved = await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")
        assert moved is not None

        # By the time move() returns, dst_path's index/cache/underlying store must
        # already fully agree -- this is what locking both src and dst protects.
        fetched = await backend.get("/skill/b/SKILL.md")
        assert fetched is not None
        assert fetched.content == "hello"
        assert fetched.version == moved.version
        assert backend._index["/skill/b/SKILL.md"][1] == moved.version
        assert "/skill/a/SKILL.md" not in backend._index

    asyncio.run(run())


def test_move_locks_both_src_and_dst_paths():
    async def run():
        held_paths_during_move: "list[set[str]]" = []

        class _InstrumentedBackend(_InMemoryDatabaseBackend):
            async def _raw_move(self, src_path, dst_path, updated_by):
                # Snapshot which write locks are currently held (locked) while inside
                # the raw move call -- this is the window Finding 2's fix protects.
                held_paths_during_move.append(
                    {p for p, lock in self._write_locks.items() if lock.locked()}
                )
                return await super()._raw_move(src_path, dst_path, updated_by)

        backend = _InstrumentedBackend(None)
        await backend.put("/skill/a/SKILL.md", "hello")
        await backend.move("/skill/a/SKILL.md", "/skill/b/SKILL.md")

        assert held_paths_during_move
        assert held_paths_during_move[-1] == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

    asyncio.run(run())


def test_list_pattern_result_stays_resident_after_lru_pressure():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "primary content")
        await backend.list(pattern="/skill/*/SKILL.md")  # touches the resident tier

        for i in range(300):
            await backend.put(f"/skill/filler-{i}/notes.md", "x")

        # A plain get() (not via list(pattern=...)) for the filler files went through
        # the bounded LRU (cap 256); the list()-touched entry must survive regardless.
        assert backend._resident.get("/skill/a/SKILL.md") == "primary content"
        assert (await backend.get("/skill/a/SKILL.md")).content == "primary content"

    asyncio.run(run())


def test_list_and_apply_batch_and_revision():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/skill/b/SKILL.md", "b")

        listed = await backend.list(pattern="/skill/*")
        assert {r.path for r in listed} == {"/skill/a/SKILL.md", "/skill/b/SKILL.md"}

        since = await backend.list()
        assert len(since) == 2

        results = await backend.apply_batch([
            PutOp(path="/skill/c/SKILL.md", content="c"),
            DeleteOp(path="/skill/a/SKILL.md"),
        ])
        assert {r.path for r in results} == {"/skill/c/SKILL.md"}

        revision = await backend.revision()
        assert isinstance(revision, int)

    asyncio.run(run())


def test_list_since_filters_older_rows():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "a")
        future = datetime.now() + timedelta(days=1)
        results = await backend.list(since=future)
        assert results == []

    asyncio.run(run())


def test_list_matches_glob_across_namespace_when_initialized():
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        await backend.put("/skill/a/SKILL.md", "a")
        await backend.put("/mcp/b/mcp.yaml", "b")
        await backend.put("/skill/b/notes.md", "not a match")
        await backend.get("/skill/a/SKILL.md")  # forces _initialized via _ensure_fresh -> _sync

        results = await backend.list(pattern="*/SKILL.md")
        assert {r.path for r in results} == {"/skill/a/SKILL.md"}

    asyncio.run(run())


def test_list_falls_back_to_raw_storage_before_initialized():
    async def run():
        writer = _InMemoryDatabaseBackend(None)
        await writer.put("/skill/a/SKILL.md", "a")
        await writer.put("/mcp/b/mcp.yaml", "b")

        reader = _InMemoryDatabaseBackend(None)
        reader._rows = writer._rows

        results = await reader.list(pattern="*/SKILL.md")
        assert {r.path for r in results} == {"/skill/a/SKILL.md"}

    asyncio.run(run())


def test_get_at_version_returns_historical_content_via_disk_cache(tmp_path):
    async def run():
        backend = _InMemoryDatabaseBackend(None)
        backend._cache_dir = tmp_path
        await backend.put("/skill/a/SKILL.md", "v1")
        await backend.put("/skill/a/SKILL.md", "v2")
        v1 = await backend.get("/skill/a/SKILL.md", 1)
        assert v1 is not None
        assert v1.content == "v1"

    asyncio.run(run())


def test_get_at_version_falls_back_to_raw_storage_when_caches_miss():
    async def run():
        # Write two versions with one backend instance (this populates the real
        # backing store via _raw_upsert, which _record_version mirrors into history).
        writer = _InMemoryDatabaseBackend(None)
        await writer.put("/skill/a/SKILL.md", "v1")
        await writer.put("/skill/a/SKILL.md", "v2")

        # Fresh backend instance sharing the same underlying rows/history dicts, but
        # with empty in-memory caches (_hist, _index) and no cache_dir (disk L2
        # absent) -- this forces both the hist-LRU tier and disk-L2 tier to miss, so
        # the older version can only come back via the new _raw_get(path, version) tier.
        reader = _InMemoryDatabaseBackend(None)
        reader._rows = writer._rows
        reader._history = writer._history

        v1 = await reader.get("/skill/a/SKILL.md", 1)
        assert v1 is not None
        assert v1.content == "v1"
        assert v1.version == 1

        # And it should now be cached in the hist-LRU for next time.
        assert reader._hist_get(reader._rows["/skill/a/SKILL.md"].row_id, 1) == "v1"

    asyncio.run(run())
