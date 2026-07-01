import asyncio
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any

from linktools.ai.capabilities.store import CapabilityStore
from linktools.ai.capabilities.memory import InMemoryCapabilityCache


class _FakeRepo:
    """Test-specific fake: exposes `since_args` to assert the incremental-vs-full-reconcile
    sync path, which InMemoryCapabilityRepository (a plain dict, no call-log) doesn't cover.
    """

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.since_args: list[datetime | None] = []

    async def list_files_since(self, since: datetime | None = None) -> list[dict[str, Any]]:
        self.since_args.append(since)
        return [dict(row) for row in self.rows]

    async def get_file(self, kind: str, file_path: str) -> dict[str, Any] | None:
        for row in self.rows:
            if row["kind"] == kind and row["file_path"] == file_path and row.get("status") == "active":
                return dict(row)
        return None


def _row(file_id: int, capability_id: str, rel_path: str, content: str, *, version: int = 1) -> dict[str, Any]:
    return {
        "id": file_id,
        "kind": "skill",
        "file_path": f"{capability_id}/{rel_path}",
        "content": content,
        "checksum": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "version": version,
        "status": "active",
        "updated_by": "test",
        "updated_at": datetime(2026, 1, 1, 0, 0, version),
    }


def test_sync_reconciles_removed_rows_when_remote_version_changes(tmp_path: Path) -> None:
    async def run() -> None:
        repo = _FakeRepo([_row(1, "old_skill", "SKILL.md", "old")])
        redis = InMemoryCapabilityCache()
        await redis.incr("capabilities:version")
        store = CapabilityStore(repo, redis, tmp_path)

        await store.sync_if_stale()
        assert store.get_file_ids("skill", "old_skill") == {"SKILL.md": (1, 1)}

        repo.rows = [_row(2, "new_skill", "SKILL.md", "new", version=2)]
        await redis.incr("capabilities:version")

        await store.sync_if_stale()

        assert store.get_file_ids("skill", "old_skill") == {}
        assert store.get_file_ids("skill", "new_skill") == {"SKILL.md": (2, 2)}
        assert repo.since_args[-1] is None

    asyncio.run(run())


def test_get_file_row_checks_freshness_before_returning_cached_row(tmp_path: Path) -> None:
    async def run() -> None:
        repo = _FakeRepo([_row(1, "cap", "SKILL.md", "v1")])
        redis = InMemoryCapabilityCache()
        await redis.incr("capabilities:version")
        store = CapabilityStore(repo, redis, tmp_path)

        await store.sync_if_stale()
        assert (await store.get_file_row("skill", "cap", "SKILL.md"))["version"] == 1

        repo.rows = [_row(1, "cap", "SKILL.md", "v2", version=2)]
        await redis.incr("capabilities:version")

        row = await store.get_file_row("skill", "cap", "SKILL.md")

        assert row is not None
        assert row["version"] == 2
        assert row["checksum"] == hashlib.sha256(b"v2").hexdigest()

    asyncio.run(run())
