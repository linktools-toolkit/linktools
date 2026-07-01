from __future__ import annotations

"""Distributed capability file store: MySQL as storage, in-process memory as cache.

Write path  (model generates skill → persisted in MySQL, version bumped in Redis):
    await store.save(kind, capability_id, file_path, content)
    await store.save_file(kind, capability_id, rel_path, content)

Read path  (before each trace — O(1) Redis check, MySQL only on version change):
    await store.sync_if_stale()
"""

import asyncio
import hashlib
import json
import logging
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import AsyncIterator, Any

from .protocols import CapabilityCacheProtocol, CapabilityRepositoryProtocol
from ..support.utils import json_ready

logger = logging.getLogger("linktools.ai.capabilities.store")

_REDIS_VERSION_KEY = "capabilities:version"
_HASH_METHOD = "sha256"
_SYNC_LOOKBACK = timedelta(minutes=1)

_LRU_MAX_FILES = 256  # upper bound for supplementary-file content / row caches
_MAX_CACHE_BYTES = 1 * 1024 * 1024  # 1 MiB: larger supplementary/historical files are served from disk/DB, never held in memory

_PRIMARY_REL: dict[str, str] = {
    "worker": "agent.md",
    "stage": "agent.md",
    "subagent": "agent.md",
    "skill": "SKILL.md",
    "mcp": "mcp.yaml",
    "plugin": "plugin.yaml",
}


def _exceeds_cache_limit(content: str) -> bool:
    """True when content is too large to hold in memory.

    Char count is a cheap lower bound on UTF-8 byte length, so a string whose char
    count already exceeds the limit is rejected without encoding; only strings within
    the char limit are measured exactly.
    """
    return len(content) > _MAX_CACHE_BYTES or len(content.encode("utf-8")) > _MAX_CACHE_BYTES


# DB storage path: {capability_id}/{rel_path}  (kind is a separate column)
def _db_path(capability_id: str, rel_path: str) -> str:
    return f"{capability_id}/{rel_path}"


def _db_prefix(capability_id: str) -> str:
    return f"{capability_id}/"


# In-process memory key: {kind}/{capability_id}/{rel_path}  (kind kept for uniqueness)
def _mem_key(kind: str, capability_id: str, rel_path: str) -> str:
    return f"{kind}/{capability_id}/{rel_path}"


def _mem_prefix(kind: str, capability_id: str) -> str:
    return f"{kind}/{capability_id}/"


def _extract_rel(kind: str, capability_id: str, file_path: str) -> str:
    id_prefix = _db_prefix(capability_id)
    if file_path.startswith(id_prefix):
        return file_path[len(id_prefix):]
    source_root = {
        "worker": "worker",
        "stage": "stage",
        "subagent": "subagent",
        "skill": "skill",
        "mcp": "adapter",
        "plugin": "plugin",
    }.get(kind, kind)
    source_prefix = f"{source_root}/{capability_id}/"
    if file_path.startswith(source_prefix):
        return file_path[len(source_prefix):]
    return file_path


def _legacy_display_rel(kind: str, capability_id: str, file_path: str, rel_path: str) -> str | None:
    if file_path == rel_path:
        return None
    id_prefix = _db_prefix(capability_id)
    if file_path.startswith(id_prefix):
        return None
    source_root = {
        "worker": "worker",
        "stage": "stage",
        "subagent": "subagent",
        "skill": "skill",
        "mcp": "adapter",
        "plugin": "plugin",
    }.get(kind, kind)
    if file_path.startswith(f"{source_root}/{capability_id}/"):
        return file_path
    return None


def _split_db_path(db_path: str) -> tuple[str, str] | None:
    if "/" not in db_path:
        return None
    cap_id, rel = db_path.split("/", 1)
    return (cap_id, rel) if cap_id and rel else None


def _with_file_parts(kind: str, row: dict[str, Any]) -> dict[str, Any]:
    parts = _split_db_path(str(row.get("file_path") or ""))
    if parts is not None:
        row["capability_id"], row["rel_path"] = parts
    return row


def _load_version(value: str | None) -> int:
    if not value:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


_ABSENT_REVISION = "absent"


class CapabilityConflictError(RuntimeError):
    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(f"capability revision mismatch: expected={expected} actual={actual}")
        self.expected = expected
        self.actual = actual


def _revision_token(rows: list[dict[str, Any]]) -> str:
    active_rows = [row for row in rows if str(row.get("status") or "active") == "active"]
    if not active_rows:
        return _ABSENT_REVISION
    payload = [
        {
            "file_path": str(row.get("file_path") or row.get("rel_path") or ""),
            "version": int(row.get("version") or 0),
        }
        for row in sorted(active_rows, key=lambda item: str(item.get("file_path") or item.get("rel_path") or ""))
    ]
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")).hexdigest()
    return digest


def revision_token(rows: list[dict[str, Any]]) -> str:
    return _revision_token(rows)


class CapabilityStore:
    """Sync model-generated capability files to in-process memory.

    MySQL stores the content (source of truth).
    Redis stores only a monotonic version counter — workers do a single GET
    per trace and skip the DB entirely when the version hasn't changed.

    One instance per worker process; asyncio-safe.
    """

    def __init__(self, repo: "CapabilityRepositoryProtocol", redis: "CapabilityCacheProtocol", workspace_root: "Path | None" = None) -> None:
        self.repo = repo
        self.redis = redis
        self._workspace_root = workspace_root
        self._local_version: int = 0
        self._last_sync_at: datetime | None = None
        self._initialized: bool = False
        self._lock = asyncio.Lock()
        # Content caches: primary files stay resident (registry preload depends on them);
        # supplementary files go through a bounded LRU. Disk ({id}.v{ver}) is the L2 tier.
        self._mem: dict[str, str] = {}
        self._lru: "OrderedDict[str, str]" = OrderedDict()
        # Resident index of every known file → (file_id, version). Tiny; powers disk-path
        # resolution and has()/list_file_states() without a DB round-trip.
        self._mem_versions: dict[str, tuple[int, int]] = {}
        self._deleted: set[str] = set()
        self._mem_rows: "OrderedDict[str, dict]" = OrderedDict()  # bounded LRU of full DB rows
        # Bounded LRU of historical-version content, keyed by (file_id, version). Immutable,
        # so cached entries never go stale — spares repeated disk reads of old versions.
        self._hist: "OrderedDict[tuple[int, int], str]" = OrderedDict()
        self._cap_locks: dict[str, asyncio.Lock] = {}

    # ── Content cache helpers (mem: primary resident + supplementary LRU) ──

    def _content_get(self, fp: str) -> str | None:
        cached = self._mem.get(fp)
        if cached is not None:
            return cached
        cached = self._lru.get(fp)
        if cached is not None:
            self._lru.move_to_end(fp)
        return cached

    def _content_put(self, fp: str, content: str, *, primary: bool) -> None:
        if primary:
            self._mem[fp] = content  # primary spec files stay resident regardless of size
            return
        # Oversized supplementary files are never cached: evict any stale entry and let
        # reads fall through to the disk/DB tiers.
        if _exceeds_cache_limit(content):
            self._lru.pop(fp, None)
            return
        self._lru[fp] = content
        self._lru.move_to_end(fp)
        while len(self._lru) > _LRU_MAX_FILES:
            self._lru.popitem(last=False)

    def _content_pop(self, fp: str) -> None:
        self._mem.pop(fp, None)
        self._lru.pop(fp, None)

    def _hist_get(self, file_id: int, version: int) -> str | None:
        cached = self._hist.get((file_id, version))
        if cached is not None:
            self._hist.move_to_end((file_id, version))
        return cached

    def _hist_put(self, file_id: int, version: int, content: str) -> None:
        if _exceeds_cache_limit(content):
            return
        self._hist[(file_id, version)] = content
        self._hist.move_to_end((file_id, version))
        while len(self._hist) > _LRU_MAX_FILES:
            self._hist.popitem(last=False)

    def _rows_get(self, fp: str) -> dict | None:
        cached = self._mem_rows.get(fp)
        if cached is not None:
            self._mem_rows.move_to_end(fp)
        return cached

    def _rows_put(self, fp: str, row: dict) -> None:
        self._mem_rows[fp] = row
        self._mem_rows.move_to_end(fp)
        while len(self._mem_rows) > _LRU_MAX_FILES:
            self._mem_rows.popitem(last=False)

    async def _ensure_fresh(self) -> None:
        """Cheap cross-process freshness check: single Redis GET, sync only when version advanced."""
        if not self._initialized:
            await self.sync_if_stale()
            return
        if not self.redis.config.enabled:
            return
        if await self._remote_version() != self._local_version:
            await self.sync_if_stale()

    # ── Locking ───────────────────────────────────────────────────────────

    @asynccontextmanager
    async def cap_write_lock(self, kind: str, capability_id: str) -> AsyncIterator[None]:
        """Distributed per-capability write lock via Redis SET NX EX.

        Falls back to asyncio.Lock when Redis is disabled (single-instance dev).
        Retries for up to 10s before raising TimeoutError.
        """
        if not self.redis.config.enabled:
            key = _mem_prefix(kind, capability_id)
            if key not in self._cap_locks:
                self._cap_locks[key] = asyncio.Lock()
            async with self._cap_locks[key]:
                yield
            return

        lock_key = f"cap:wlock:{_mem_prefix(kind, capability_id)}"
        lock_val = uuid.uuid4().hex
        ttl = 30  # seconds; long enough for any single materialize+write cycle

        acquired = False
        for _ in range(100):  # 100 × 0.1s = 10s max wait
            if await self.redis.try_acquire(lock_key, lock_val, ttl):
                acquired = True
                break
            await asyncio.sleep(0.1)

        if not acquired:
            raise TimeoutError(f"cap_write_lock timeout for {kind}/{capability_id}")

        try:
            yield
        finally:
            try:
                await self.redis.release_if_owner(lock_key, lock_val)
            except Exception as exc:
                logger.warning("cap_write_lock release failed (%s/%s): %s", kind, capability_id, exc)

    # ── Write ─────────────────────────────────────────────────────────────

    async def save(
        self,
        *,
        kind: str,
        capability_id: str,
        file_path: str,
        content: str,
        updated_by: str = "engine",
    ) -> bool:
        """Persist any capability file to MySQL. Returns True if content changed."""
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        rel = _extract_rel(kind, capability_id, file_path)
        legacy_rel = _legacy_display_rel(kind, capability_id, file_path, rel)
        db_fp = _db_path(capability_id, rel)
        fp = _mem_key(kind, capability_id, rel)
        file_id, version, changed = await self.repo.upsert_file(
            kind=kind, file_path=db_fp,
            content=content, checksum=checksum,
            updated_by=updated_by,
        )
        legacy_changed = False
        if legacy_rel is not None:
            try:
                legacy_changed = await self.repo.tombstone_file(
                    kind, _db_path(capability_id, legacy_rel),
                    checksum=hashlib.sha256(b"").hexdigest(),
                    updated_by=updated_by,
                )
            except Exception as exc:
                logger.warning(
                    "capability legacy path cleanup failed (%s/%s/%s): %s",
                    kind, capability_id, legacy_rel, exc,
                )
            legacy_fp = _mem_key(kind, capability_id, legacy_rel)
            self._mem_versions.pop(legacy_fp, None)
            self._content_pop(legacy_fp)
            self._mem_rows.pop(legacy_fp, None)
            self._deleted.add(legacy_fp)
        self._mem_versions[fp] = (file_id, version)
        self._deleted.discard(fp)
        self._content_put(fp, content, primary=rel == _PRIMARY_REL.get(kind))
        self._mem_rows.pop(fp, None)
        self._write_workspace(file_id, version, content, kind=kind, capability_id=capability_id, rel_path=rel, checksum=checksum)
        if changed or legacy_changed:
            await self._bump_redis_version()
            logger.info("capability saved: kind=%s capability_id=%s file=%s", kind, capability_id, fp)
        return changed

    async def save_file(
        self, *, kind: str, capability_id: str, rel_path: str, content: str, updated_by: str = "ui",
    ) -> bool:
        """Persist a supplementary file. Returns True if content changed."""
        checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
        db_fp = _db_path(capability_id, rel_path)
        fp = _mem_key(kind, capability_id, rel_path)
        file_id, version, changed = await self.repo.upsert_file(
            kind=kind, file_path=db_fp,
            content=content, checksum=checksum, updated_by=updated_by,
        )
        self._mem_versions[fp] = (file_id, version)
        self._deleted.discard(fp)
        self._content_put(fp, content, primary=rel_path == _PRIMARY_REL.get(kind))
        self._mem_rows.pop(fp, None)
        self._write_workspace(file_id, version, content, kind=kind, capability_id=capability_id, rel_path=rel_path, checksum=checksum)
        if changed:
            await self._bump_redis_version()
            logger.info("capability file saved: kind=%s capability_id=%s rel=%s", kind, capability_id, rel_path)
        return changed

    async def save_batch(
        self,
        *,
        kind: str,
        capability_id: str,
        primary_rel_path: str,
        primary_content: str,
        supplementary_files: list[dict[str, str | None]],
        deleted_rel_paths: list[str],
        expected_revision: str,
        updated_by: str = "ui",
    ) -> str:
        try:
            revision, final_rows, changed = await self.repo.apply_batch(
                kind=kind,
                capability_id=capability_id,
                primary_rel_path=primary_rel_path,
                primary_content=primary_content,
                supplementary_files=supplementary_files,
                deleted_rel_paths=deleted_rel_paths,
                expected_revision=expected_revision,
                updated_by=updated_by,
            )
        except CapabilityConflictError:
            raise
        except Exception as exc:
            logger.warning("capability_store.save_batch failed (%s/%s): %s", kind, capability_id, exc)
            raise
        # Write-through: reconcile this process's memory to the post-commit DB state, so files
        # left unchanged by the batch (e.g. an untouched primary) keep a valid index entry
        # instead of being dropped by a blanket purge that incremental sync can't restore.
        for row in final_rows:
            self._apply_db_row_to_mem(row)
        if changed:
            await self._bump_redis_version()
            logger.info(
                "capability batch saved: kind=%s capability_id=%s files=%d deleted=%d",
                kind, capability_id, len(supplementary_files) + 1, len(deleted_rel_paths),
            )
        return revision

    # ── Read ──────────────────────────────────────────────────────────────

    async def list_active(self, kind: str | None = None) -> list[dict]:
        try:
            return await self.repo.list_capabilities_active(kind=kind)
        except Exception as exc:
            logger.warning("capability_store.list_active failed: %s", exc)
            return []

    async def has(self, kind: str, capability_id: str) -> bool:
        await self._ensure_fresh()
        prefix = _mem_prefix(kind, capability_id)
        if any(fp.startswith(prefix) for fp in self._mem_versions):
            return True
        if self._initialized:
            return False  # full sync completed — not in memory means not in DB
        try:
            return await self.repo.capability_exists(kind, capability_id)
        except Exception as exc:
            logger.warning("capability_store.has failed (%s/%s): %s", kind, capability_id, exc)
            return False

    async def get_file(self, kind: str, capability_id: str, rel_path: str) -> str | None:
        """Read current-version content via the three-tier cache: mem → disk → DB.

        Freshness is driven by the global ``capabilities:version`` counter: a cheap
        check refreshes the local index when another process has written. The local
        version index then locates the disk file, whose integrity is validated by its
        sidecar hash.
        """
        await self._ensure_fresh()

        fp = _mem_key(kind, capability_id, rel_path)
        db_fp = _db_path(capability_id, rel_path)
        primary = rel_path == _PRIMARY_REL.get(kind)

        # State 3: mem hit (sync evicts stale entries, so a hit is current).
        cached = self._content_get(fp)
        if cached is not None:
            return cached

        # State 2: disk hit — located by the fresh {file_id}.v{version}, sidecar-validated.
        entry = self._mem_versions.get(fp)
        if entry is not None:
            disk_content = self._read_workspace(*entry)
            if disk_content is not None:
                self._content_put(fp, disk_content, primary=primary)
                return disk_content

        # State 1: DB.
        try:
            row = await self.repo.get_file(kind, db_fp)
        except Exception as exc:
            logger.warning("capability_store.get_file failed (%s/%s/%s): %s", kind, capability_id, rel_path, exc)
            return None
        if row is None:
            return None
        content = str(row["content"])
        db_id, db_ver = row.get("id"), row.get("version")
        if db_id is not None and db_ver is not None:
            cur = self._mem_versions.get(fp)
            # A concurrent sync/save may have cached a newer version while we awaited the
            # DB read — don't clobber it with this older snapshot (and don't re-poison disk).
            if cur is not None and cur[1] > int(db_ver):
                return self._content_get(fp) or content
            self._mem_versions[fp] = (int(db_id), int(db_ver))
            self._write_workspace(int(db_id), int(db_ver), content, kind=kind, capability_id=capability_id, rel_path=rel_path, checksum=row.get("checksum"))
        self._content_put(fp, content, primary=primary)
        return content

    async def get_file_row(self, kind: str, capability_id: str, rel_path: str) -> dict | None:
        await self._ensure_fresh()
        fp = _mem_key(kind, capability_id, rel_path)
        cached = self._rows_get(fp)
        if cached is not None:
            return cached
        try:
            row = await self.repo.get_file(kind, _db_path(capability_id, rel_path))
        except Exception as exc:
            logger.warning("capability_store.get_file_row failed (%s/%s/%s): %s", kind, capability_id, rel_path, exc)
            return None
        if row is not None:
            self._rows_put(fp, row)
        return row

    async def get_file_at_version(self, kind: str, capability_id: str, rel_path: str, version: int) -> str | None:
        """Historical read: hist LRU → disk ({file_id}.v{version}, hash-validated) → DB.

        Historical versions are immutable, so the LRU never goes stale.
        """
        fp = _mem_key(kind, capability_id, rel_path)
        entry = self._mem_versions.get(fp)
        file_id = entry[0] if entry else None

        if entry and entry[1] == version:
            cached = self._content_get(fp)
            if cached is not None:
                return cached

        if file_id is not None:
            hist = self._hist_get(file_id, version)
            if hist is not None:
                return hist
            disk_content = self._read_workspace(file_id, version)
            if disk_content is not None:
                self._hist_put(file_id, version, disk_content)
                return disk_content

        try:
            row = await self.repo.get_file_at_version(kind, _db_path(capability_id, rel_path), version)
        except Exception as exc:
            logger.warning(
                "capability_store.get_file_at_version failed (%s/%s/%s@v%d): %s",
                kind, capability_id, rel_path, version, exc,
            )
            return None
        if row is None:
            return None
        content = str(row["content"])
        if file_id is not None:
            self._write_workspace(
                file_id, version, content,
                kind=kind, capability_id=capability_id, rel_path=rel_path,
                checksum=row.get("checksum"),
            )
            self._hist_put(file_id, version, content)
        return content

    def get_file_ids(self, kind: str, capability_id: str) -> dict[str, tuple[int, int]]:
        """Return {rel_path: (file_id, version)} for every in-memory file of a capability.

        Reads the in-process version index — no DB round-trip. Used by snapshot writers
        to record all files (primary + supplementary) for future scene restoration.
        """
        prefix = _mem_prefix(kind, capability_id)
        return {
            fp[len(prefix):]: entry
            for fp, entry in self._mem_versions.items()
            if fp.startswith(prefix)
        }

    def get_deleted_rels(self, kind: str, capability_id: str) -> set[str]:
        """Rel-paths that are explicitly deleted in DB for this capability (no DB round-trip)."""
        prefix = _mem_prefix(kind, capability_id)
        return {fp[len(prefix):] for fp in self._deleted if fp.startswith(prefix)}

    def cache_path(self, file_id: int, version: int) -> "Path | None":
        """Absolute path of a file's flat L2 cache entry ({file_id}.v{version}), or None.

        The file is written there by every sync, so it is a stable symlink target.
        """
        if self._workspace_root is None:
            return None
        return self._workspace_root / f"{file_id}.v{version}"

    def db_managed_ids(self, kind: str) -> set[str]:
        """Capability ids with at least one primary file in memory cache."""
        prefix = f"{kind}/"
        result: set[str] = set()
        for fp in self._mem:
            if fp.startswith(prefix):
                cap_id = fp[len(prefix):].split("/")[0]
                if cap_id:
                    result.add(cap_id)
        return result

    async def list_files(self, kind: str, capability_id: str) -> list[dict]:
        try:
            return await self.repo.list_files(kind, capability_id)
        except Exception as exc:
            logger.warning("capability_store.list_files failed (%s/%s): %s", kind, capability_id, exc)
            return []

    async def list_file_states(self, kind: str, capability_id: str) -> list[dict]:
        await self._ensure_fresh()
        if self._initialized:
            prefix = _mem_prefix(kind, capability_id)
            active = [
                {
                    "file_path": _db_path(capability_id, fp[len(prefix):]),
                    "rel_path": fp[len(prefix):],
                    "status": "active",
                    "version": self._mem_versions[fp][1],
                }
                for fp in self._mem_versions
                if fp.startswith(prefix)
            ]
            deleted = [
                {
                    "file_path": _db_path(capability_id, fp[len(prefix):]),
                    "rel_path": fp[len(prefix):],
                    "status": "deleted",
                }
                for fp in self._deleted
                if fp.startswith(prefix)
            ]
            return sorted(active + deleted, key=lambda item: (str(item["rel_path"]), str(item["status"])))
        try:
            return await self.repo.list_file_states(kind, capability_id)
        except Exception as exc:
            logger.warning("capability_store.list_file_states failed (%s/%s): %s", kind, capability_id, exc)
            return []

    async def list_kind_meta(self, kind: str) -> tuple[set[str], dict[str, list[str]], dict[str, list[str]], str]:
        """Derive (active ids, active rels, deleted rels) for a kind from the synced memory state.

        Shares the single incremental sync with iter_primaries — no separate DB query.
        """
        await self._ensure_fresh()
        prefix = f"{kind}/"
        db_ids: set[str] = set()
        files_by_id: dict[str, list[str]] = {}
        deleted_by_id: dict[str, list[str]] = {}
        for fp in self._mem_versions:
            if not fp.startswith(prefix):
                continue
            cap_id, _, rel = fp[len(prefix):].partition("/")
            if not cap_id or not rel:
                continue
            db_ids.add(cap_id)
            files_by_id.setdefault(cap_id, []).append(rel)
        for fp in self._deleted:
            if not fp.startswith(prefix):
                continue
            cap_id, _, rel = fp[len(prefix):].partition("/")
            if not cap_id or not rel:
                continue
            deleted_by_id.setdefault(cap_id, []).append(rel)
        for rels in files_by_id.values():
            rels.sort()
        for rels in deleted_by_id.values():
            rels.sort()
        return db_ids, files_by_id, deleted_by_id, "mem"


    def _write_workspace(
        self,
        file_id: int,
        version: int,
        content: str,
        *,
        kind: str | None = None,
        capability_id: str | None = None,
        rel_path: str | None = None,
        checksum: str | None = None,
    ) -> None:
        """Write content file plus .meta.json with checksum (reuses caller's checksum if given)."""
        if self._workspace_root is None:
            return
        dest = self._workspace_root / f"{file_id}.v{version}"
        meta = self._workspace_root / f"{file_id}.v{version}.meta.json"
        try:
            self._workspace_root.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")
            if checksum is None:
                checksum = hashlib.sha256(content.encode("utf-8")).hexdigest()
            if kind and capability_id and rel_path:
                meta.write_text(json.dumps({
                    "file_id": file_id,
                    "version": version,
                    "kind": kind,
                    "capability_id": capability_id,
                    "rel_path": rel_path,
                    "checksum": checksum,
                }, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError as exc:
            logger.warning("capability_store: workspace write failed (%s.v%s): %s", file_id, version, exc)

    def _read_workspace(self, file_id: int, version: int) -> str | None:
        """Return disk content only if the .meta.json checksum matches; else None.

        A mismatch means a corrupt/partial write — the pair is removed so the caller
        falls through to the DB and rewrites it.
        """
        if self._workspace_root is None:
            return None
        dest = self._workspace_root / f"{file_id}.v{version}"
        meta = self._workspace_root / f"{file_id}.v{version}.meta.json"
        try:
            if not dest.exists() or not meta.exists():
                return None
            content = dest.read_text(encoding="utf-8")
            meta_data = json.loads(meta.read_text(encoding="utf-8"))
            expected_checksum = meta_data.get("checksum")
            if not expected_checksum or hashlib.sha256(content.encode("utf-8")).hexdigest() != expected_checksum:
                dest.unlink(missing_ok=True)
                meta.unlink(missing_ok=True)
                return None
            return content
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("capability_store: workspace read failed (%s.v%s): %s", file_id, version, exc)
            return None

    def _delete_workspace(self, file_id: int, version: int) -> None:
        if self._workspace_root is None:
            return
        for name in (f"{file_id}.v{version}", f"{file_id}.v{version}.meta.json"):
            try:
                (self._workspace_root / name).unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("capability_store: workspace delete failed (%s): %s", name, exc)

    async def iter_primaries(self, kind: str) -> list[tuple[str, str, str]]:
        """Return (capability_id, file_path, content) for all primary files of a kind."""
        primary_rel = _PRIMARY_REL.get(kind)
        prefix = f"{kind}/"

        await self._ensure_fresh()

        # Fast path: sync has populated _mem; no DB round-trip needed.
        if self._initialized:
            return [
                (fp[len(prefix):].split("/")[0], fp, content)
                for fp, content in self._mem.items()
                if fp.startswith(prefix)
            ]

        if primary_rel:
            try:
                rows = await self.repo.get_primary_files(kind, primary_rel)
            except Exception as exc:
                logger.warning("capability_store.iter_primaries DB failed (%s): %s", kind, exc)
                rows = []
            result = []
            for row in rows:
                capability_id = str(row["capability_id"])
                content = str(row["content"])
                file_id, new_ver = int(row["id"]), int(row["version"])
                fp = _mem_key(kind, capability_id, primary_rel)
                self._mem[fp] = content  # primary: resident
                self._rows_put(fp, row)
                self._mem_versions[fp] = (file_id, new_ver)
                self._deleted.discard(fp)
                self._write_workspace(file_id, new_ver, content, kind=kind, capability_id=capability_id, rel_path=primary_rel, checksum=row.get("checksum"))
                result.append((capability_id, fp, content))
            if result:
                logger.info("capability_store.iter_primaries: loaded %d %s from DB", len(result), kind)
            return result

        return [
            (fp[len(prefix):].split("/")[0], fp, content)
            for fp, content in self._mem.items()
            if fp.startswith(prefix)
        ]

    # ── Delete / Rename ───────────────────────────────────────────────────

    async def delete(self, kind: str, capability_id: str) -> bool:
        if await self.repo.delete_files_all(kind, capability_id) == 0:
            return False
        prefix = _mem_prefix(kind, capability_id)
        keys = [k for k in self._mem_versions if k.startswith(prefix)]
        for key in keys:
            self._content_pop(key)
            entry = self._mem_versions.pop(key, None)
            if entry:
                self._delete_workspace(*entry)
            self._mem_rows.pop(key, None)
            self._deleted.add(key)
        await self._bump_redis_version()
        logger.info("capability deleted: kind=%s capability_id=%s", kind, capability_id)
        return True

    async def restore_builtin(self, kind: str, capability_id: str) -> bool:
        if await self.repo.restore_builtin_files(kind, capability_id) == 0:
            return False
        prefix = _mem_prefix(kind, capability_id)
        keys = [k for k in self._mem_versions if k.startswith(prefix)]
        for key in keys:
            self._content_pop(key)
            entry = self._mem_versions.pop(key, None)
            if entry:
                self._delete_workspace(*entry)
            self._mem_rows.pop(key, None)
            self._deleted.discard(key)
        await self._bump_redis_version()
        logger.info("builtin capability restored: kind=%s capability_id=%s", kind, capability_id)
        return True

    async def rename(self, kind: str, old_capability_id: str, new_capability_id: str) -> bool:
        try:
            if await self.repo.move_files(kind, old_capability_id, new_capability_id) == 0:
                return False
        except ValueError as exc:
            logger.warning("capability_store.rename rejected: %s", exc)
            return False
        old_prefix = _mem_prefix(kind, old_capability_id)
        new_prefix = _mem_prefix(kind, new_capability_id)
        # Disk files are keyed by {file_id}.v{version} (path-independent), so a rename only
        # remaps cache keys — no disk move needed.
        for old_key in [k for k in self._mem_versions if k.startswith(old_prefix)]:
            new_key = old_key.replace(old_prefix, new_prefix, 1)
            if old_key in self._mem:
                self._mem[new_key] = self._mem.pop(old_key)
            if old_key in self._lru:
                self._lru[new_key] = self._lru.pop(old_key)
            file_id, version = self._mem_versions.pop(old_key)
            new_version = version + 1
            self._mem_versions[new_key] = (file_id, new_version)
            rel = new_key[len(new_prefix):]
            content = self._content_get(new_key) or self._read_workspace(file_id, version)
            if content is not None:
                self._write_workspace(file_id, new_version, content, kind=kind, capability_id=new_capability_id, rel_path=rel)
                self._delete_workspace(file_id, version)
            if old_key in self._deleted:
                self._deleted.remove(old_key)
                self._deleted.add(new_key)
            self._mem_rows.pop(old_key, None)
            self._mem_rows.pop(new_key, None)
        await self._bump_redis_version()
        logger.info("capability renamed: kind=%s %s→%s", kind, old_capability_id, new_capability_id)
        return True

    async def rename_already_applied(self, kind: str, old_capability_id: str, new_capability_id: str) -> bool:
        try:
            return await self.repo.was_capability_renamed(kind, old_capability_id, new_capability_id)
        except Exception as exc:
            logger.warning(
                "capability_store.rename_already_applied failed (%s/%s→%s): %s",
                kind, old_capability_id, new_capability_id, exc,
            )
            return False

    async def delete_file(self, kind: str, capability_id: str, rel_path: str) -> bool:
        fp = _mem_key(kind, capability_id, rel_path)
        try:
            if not await self.repo.delete_file(kind, _db_path(capability_id, rel_path)):
                return False
        except Exception as exc:
            logger.warning("capability_store.delete_file failed (%s/%s/%s): %s", kind, capability_id, rel_path, exc)
            return False
        entry = self._mem_versions.pop(fp, None)
        if entry:
            self._delete_workspace(*entry)
        self._content_pop(fp)
        self._mem_rows.pop(fp, None)
        self._deleted.add(fp)
        await self._bump_redis_version()
        logger.info("capability file deleted: kind=%s capability_id=%s rel=%s", kind, capability_id, rel_path)
        return True

    async def tombstone_file(self, kind: str, capability_id: str, rel_path: str, *, updated_by: str = "ui") -> bool:
        checksum = hashlib.sha256(b"").hexdigest()
        fp = _mem_key(kind, capability_id, rel_path)
        try:
            ok = await self.repo.tombstone_file(kind, _db_path(capability_id, rel_path), checksum=checksum, updated_by=updated_by)
        except Exception as exc:
            logger.warning("capability_store.tombstone_file failed (%s/%s/%s): %s", kind, capability_id, rel_path, exc)
            return False
        if not ok:
            return False
        entry = self._mem_versions.pop(fp, None)
        if entry:
            self._delete_workspace(*entry)
        self._content_pop(fp)
        self._mem_rows.pop(fp, None)
        self._deleted.add(fp)
        await self._bump_redis_version()
        logger.info("capability file tombstoned: kind=%s capability_id=%s rel=%s", kind, capability_id, rel_path)
        return True

    async def rename_file(self, kind: str, capability_id: str, old_rel: str, new_rel: str) -> bool:
        old_key = _mem_key(kind, capability_id, old_rel)
        new_key = _mem_key(kind, capability_id, new_rel)
        try:
            renamed = await self.repo.rename_file(kind, _db_path(capability_id, old_rel), _db_path(capability_id, new_rel))
            if renamed is None:
                return False
        except Exception as exc:
            logger.warning("capability_store.rename_file failed: %s", exc)
            return False
        # Disk file is keyed by {file_id}.v{version}, unchanged by a path rename.
        if old_key in self._mem:
            self._mem[new_key] = self._mem.pop(old_key)
        if old_key in self._lru:
            self._lru[new_key] = self._lru.pop(old_key)
        entry = self._mem_versions.pop(old_key, None)
        if entry is not None:
            file_id, old_version = entry
            _renamed_id, new_version = renamed
            self._mem_versions[new_key] = (file_id, new_version)
            content = self._content_get(new_key) or self._read_workspace(file_id, old_version)
            if content is not None:
                self._write_workspace(file_id, new_version, content, kind=kind, capability_id=capability_id, rel_path=new_rel)
                self._delete_workspace(file_id, old_version)
        self._deleted.discard(old_key)
        self._deleted.discard(new_key)
        self._mem_rows.pop(old_key, None)
        self._mem_rows.pop(new_key, None)
        await self._bump_redis_version()
        logger.info("capability file renamed: kind=%s capability_id=%s %s→%s", kind, capability_id, old_rel, new_rel)
        return True

    async def copy_file(self, kind: str, capability_id: str, old_rel: str, new_rel: str) -> bool:
        content = await self.get_file(kind, capability_id, old_rel)
        if content is None:
            return False
        return await self.save_file(kind=kind, capability_id=capability_id, rel_path=new_rel, content=content, updated_by="ui")

    # ── Sync (worker process) ─────────────────────────────────────────────

    async def sync_if_stale(self) -> bool:
        """Pull capability files updated since last sync from MySQL to in-process memory.

        Fast path: single Redis GET per trace (O(1)).
        Slow path: MySQL query + memory update, only when version changed.
        """
        async with self._lock:
            try:
                return await self._do_sync()
            except Exception as exc:
                logger.warning("capability_store: sync failed, using cached memory: %s", exc)
                return False

    # ── Internal ─────────────────────────────────────────────────────────

    def _apply_db_row_to_mem(self, row: dict[str, Any]) -> bool:
        """Apply one DB row to in-process memory; returns True if active content was cached.

        Shared by incremental sync and post-write reconciliation so both keep the memory
        index exactly consistent with the DB (active → cache, deleted → tombstone).
        """
        parts = _split_db_path(str(row["file_path"]))
        if parts is None:
            return False
        kind = row["kind"]
        cap_id, rel = parts
        fp = _mem_key(kind, cap_id, rel)
        if row.get("status", "active") == "deleted":
            entry = self._mem_versions.pop(fp, None)
            if entry:
                self._delete_workspace(*entry)
            self._content_pop(fp)
            self._mem_rows.pop(fp, None)
            self._deleted.add(fp)
            return False
        content = row["content"]
        file_id, new_ver = int(row["id"]), int(row["version"])
        self._mem_versions[fp] = (file_id, new_ver)
        self._deleted.discard(fp)
        self._write_workspace(file_id, new_ver, content, kind=kind, capability_id=cap_id, rel_path=rel, checksum=row.get("checksum"))
        if rel == _PRIMARY_REL.get(kind):
            self._mem[fp] = content  # primary: resident
        else:
            self._lru.pop(fp, None)  # supplementary: drop stale, load lazily from disk
        self._rows_put(fp, _with_file_parts(kind, json_ready(dict(row))))
        return True

    async def _do_sync(self) -> bool:
        remote_version = await self._remote_version()
        if self._initialized and remote_version == self._local_version:
            return False

        # A version advance can include path moves or physical restores, where the old
        # rows no longer exist in capability_files. Reconcile from the full current table
        # so vanished keys are removed from this process instead of lingering as ghosts.
        full_reconcile = self._initialized and remote_version != self._local_version
        if full_reconcile or self._last_sync_at is None:
            since = None
        else:
            since = self._last_sync_at - _SYNC_LOOKBACK
        rows = await self.repo.list_files_since(since)

        if not rows:
            if full_reconcile:
                self._mem.clear()
                self._lru.clear()
                self._mem_versions.clear()
                self._deleted.clear()
                self._mem_rows.clear()
            self._initialized = True
            self._local_version = remote_version
            return False

        if full_reconcile:
            self._mem.clear()
            self._lru.clear()
            self._mem_versions.clear()
            self._deleted.clear()
            self._mem_rows.clear()

        max_updated_at: datetime | None = self._last_sync_at
        wrote = 0

        for row in rows:
            try:
                if self._apply_db_row_to_mem(row):
                    wrote += 1
            except Exception as exc:
                # Deterministic bad row (malformed data): log and skip.
                logger.error("capability_store: skipping bad row %s: %s", row, exc)

            ts = row.get("updated_at")
            if isinstance(ts, datetime):
                max_updated_at = ts if max_updated_at is None else max(max_updated_at, ts)

        self._last_sync_at = max_updated_at
        self._local_version = remote_version
        self._initialized = True
        logger.info(
            "capability_store: synced %d file(s) from DB, version=%s updated_at=%s",
            wrote, remote_version, max_updated_at,
        )
        return wrote > 0

    async def current_version(self) -> int:
        """全局 capability 版本号（跨进程/跨机器一致，每次写操作 INCR；redis 关闭返回 0）。"""
        return await self._remote_version()

    async def _remote_version(self) -> int:
        if not self.redis.config.enabled:
            return 0
        try:
            return _load_version(await self.redis.get(_REDIS_VERSION_KEY))
        except Exception as exc:
            logger.warning("capability_store: redis get failed, forcing DB sync: %s", exc)
            return self._local_version + 1

    async def _bump_redis_version(self) -> None:
        if not self.redis.config.enabled:
            return
        try:
            await self.redis.incr(_REDIS_VERSION_KEY)
        except Exception as exc:
            # Only reset key when the cache backend explicitly says the value is not an
            # integer (e.g. a legacy JSON string, surfaced by real Redis as a
            # ResponseError). Transient errors (network, failover) must NOT reset the
            # key — that would silently roll back the global counter. Matched by
            # exception class name (rather than importing the redis package's
            # exception type) to keep this module free of a hard Redis dependency —
            # any CapabilityCacheProtocol implementation can signal this the same way.
            if type(exc).__name__ != "ResponseError" or "not an integer" not in str(exc).lower():
                logger.warning("capability_store: redis version bump failed (transient): %s", exc)
                return
            logger.warning("capability_store: redis version key has non-integer value, resetting: %s", exc)
            try:
                await self.redis.delete(_REDIS_VERSION_KEY)
                await self.redis.incr(_REDIS_VERSION_KEY)
            except Exception as reset_exc:
                logger.warning("capability_store: redis version reset failed: %s", reset_exc)
