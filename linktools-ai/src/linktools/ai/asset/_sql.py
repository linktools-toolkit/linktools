"""Three-table SQL Asset backend with optimistic global revision CAS."""

import hashlib
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from linktools.core import environ

from ..core import JsonValue, canonical_json_bytes, validate_asset_namespace
from ..errors import AIError, ErrorCode
from ..storage import (
    MetadataChange,
    MetadataLoad,
    MetadataLoadMode,
    ObjectRef,
    ObjectStore,
    PayloadPolicy,
    SqlObjectStore,
    StorageBatchResult,
    StorageChange,
    StorageDeleteResult,
    StorageEntryRevision,
    StorageEntryStatus,
    StorageOperation,
    StoragePutResult,
    StorageResetResult,
    StorageRevision,
    StoredPayload,
    VersionSummary,
    build_object_sql_metadata,
    create_sql_storage_context,
    dialect_for_name,
    payload_fits_inline,
    sql_audit_columns,
    sql_audit_indexes,
    sql_id_column,
    sql_query_index,
    sql_sha256,
    sql_table_options,
    sql_unique,
)
from ._domain import AssetInfo, AssetKey, AssetRoot

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine

_logger = environ.get_logger("ai.asset.sql")
_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
_RETRY_LIMIT = 8


def build_asset_sql_metadata(*, metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import JSON, BigInteger, Column, MetaData, Table

    if metadata is None:
        metadata = MetaData()
    if "ai_asset_heads" in metadata.tables:
        return metadata
    digest = sql_sha256()
    heads = Table(
        "ai_asset_heads",
        metadata,
        sql_id_column(),
        Column(
            "namespace_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the AssetStore namespace.",
        ),
        Column(
            "store_revision", BigInteger, nullable=False, comment="Last committed namespace-wide AssetStore revision."
        ),
        *sql_audit_columns(),
        comment="Namespace-level optimistic revision head that serializes AssetStore mutations.",
        **sql_table_options(),
    )
    sql_unique(heads, "namespace_digest")
    sql_audit_indexes(heads)
    entries = Table(
        "ai_asset_entries",
        metadata,
        sql_id_column(),
        Column(
            "key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the asset key including namespace identity.",
        ),
        Column(
            "namespace_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the AssetStore namespace.",
        ),
        Column("entry_revision", BigInteger, nullable=False, comment="Latest committed per-asset revision."),
        Column(
            "store_revision",
            BigInteger,
            nullable=False,
            comment="Namespace-wide revision that produced this current projection.",
        ),
        Column(
            "payload_json",
            JSON,
            nullable=False,
            comment=(
                "Versioned canonical current AssetInfo payload including status, metadata, "
                "content reference, and semantic timestamps."
            ),
        ),
        *sql_audit_columns(),
        comment="Current AssetStore projection for the latest committed revision of each asset key.",
        **sql_table_options(),
    )
    sql_unique(entries, "key_digest")
    sql_query_index(entries, "namespace_digest", "store_revision")
    sql_audit_indexes(entries)
    changes = Table(
        "ai_asset_changes",
        metadata,
        sql_id_column(),
        Column(
            "key_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the asset key including namespace identity.",
        ),
        Column(
            "entry_revision",
            BigInteger,
            nullable=False,
            comment="Immutable per-asset revision number represented by this history row.",
        ),
        Column(
            "namespace_digest",
            digest,
            nullable=False,
            comment="Canonical SHA-256 identity of the AssetStore namespace.",
        ),
        Column(
            "store_revision",
            BigInteger,
            nullable=False,
            comment="Namespace-wide revision in which this asset change committed.",
        ),
        Column(
            "payload_json", JSON, nullable=False, comment="Versioned canonical immutable AssetInfo history payload."
        ),
        *sql_audit_columns(),
        comment="Immutable AssetStore history containing every committed per-asset revision.",
        **sql_table_options(),
    )
    sql_unique(changes, "key_digest", "entry_revision")
    sql_unique(changes, "namespace_digest", "store_revision", "key_digest")
    sql_audit_indexes(changes)
    return metadata


class SqlAssetBackend:
    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        namespace: str,
        object_store: ObjectStore | None = None,
        payload_policy: "PayloadPolicy | None" = None,
    ) -> None:
        validate_asset_namespace(namespace)
        dialect_for_name(engine.dialect.name)
        self._namespace = namespace
        self._namespace_digest = hashlib.sha256(namespace.encode("utf-8")).digest()
        self._root = AssetRoot(
            f"sql:{self._namespace_digest.hex()[:16]}",
            "sql",
            namespace,
            self._namespace_digest.hex(),
        )
        self._context = create_sql_storage_context(engine)
        self._metadata = build_asset_sql_metadata()
        self._object_store = object_store or SqlObjectStore.from_context(self._context)
        self._payload_policy = payload_policy or PayloadPolicy()
        if object_store is None:
            build_object_sql_metadata(metadata=self._metadata)
        self._ready = False

    @property
    def root(self) -> AssetRoot:
        return self._root

    @property
    def writable(self) -> bool:
        return True

    @property
    def atomic_batch(self) -> bool:
        return True

    async def initialize(self) -> None:
        await self._context.initialize(metadata=self._metadata)
        table = self._metadata.tables["ai_asset_heads"]

        async def initialize_head(session) -> None:
            from sqlalchemy import insert, select

            existing = await session.scalar(
                select(table.c.namespace_digest).where(table.c.namespace_digest == self._namespace_digest.hex())
            )
            if existing is None:
                await session.execute(
                    insert(table).values(
                        namespace_digest=self._namespace_digest.hex(),
                        store_revision=0,
                    )
                )

        await self._context.run_mutation(initialize_head)
        self._ready = True
        _logger.info("SQL Asset backend initialized: namespace=%s", self._root.digest[:16])

    async def close(self) -> None:
        self._ready = False
        await self._context.close()

    async def head_revision(self) -> StorageRevision:
        row = await self._head()
        return StorageRevision(str(row))

    async def load_metadata(self, after_revision: StorageRevision | None) -> MetadataLoad[AssetKey, AssetInfo]:
        self._ensure_ready()
        current = int((await self.head_revision()).value)
        if after_revision is not None and int(after_revision.value) == current:
            return MetadataLoad(MetadataLoadMode.PATCH, StorageRevision(str(current)), ())
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            rows = (
                (
                    await session.execute(
                        select(entries).where(entries.c.namespace_digest == self._namespace_digest.hex())
                    )
                )
                .mappings()
                .all()
            )
        finally:
            await session.close()
        return MetadataLoad(
            MetadataLoadMode.REPLACE,
            StorageRevision(str(current)),
            tuple(
                MetadataChange(_key_from_data(row["payload_json"]), _info_from_data(row["payload_json"]))
                for row in rows
            ),
        )

    async def get(self, key: AssetKey) -> bytes | None:
        info = await self.stat(key)
        if info is None or info.status is not StorageEntryStatus.NORMAL:
            return None
        return await _read_asset_object(self._object_store, self._namespace_digest, key, info)

    async def get_many(self, keys: Sequence[AssetKey]) -> dict[AssetKey, bytes]:
        current = await self._load_current(tuple(dict.fromkeys(keys)))
        result: dict[AssetKey, bytes] = {}
        for key in dict.fromkeys(keys):
            info = current[key]
            if info is not None and info.status is StorageEntryStatus.NORMAL:
                result[key] = await _read_asset_object(
                    self._object_store,
                    self._namespace_digest,
                    key,
                    info,
                )
        return result

    async def stat(self, key: AssetKey) -> AssetInfo | None:
        self._ensure_ready()
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            row = (
                await session.execute(
                    select(entries.c.payload_json).where(
                        entries.c.key_digest == _asset_key_digest(self._namespace_digest, key).hex(),
                        entries.c.namespace_digest == self._namespace_digest.hex(),
                    )
                )
            ).scalar_one_or_none()
        finally:
            await session.close()
        return None if row is None else _info_from_data(row)

    async def put(
        self,
        key: AssetKey,
        value: bytes,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StoragePutResult[AssetInfo]:
        result = await self.apply_batch(
            (StorageChange(StorageOperation.PUT, key, value, expected_revision, metadata or {}),)
        )
        return result.results[0]

    async def delete(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageDeleteResult[AssetKey]:
        result = await self.apply_batch(
            (StorageChange(StorageOperation.DELETE, key, None, expected_revision, metadata or {}),)
        )
        return result.results[0]

    async def reset(
        self,
        key: AssetKey,
        *,
        expected_revision: StorageEntryRevision | None = None,
        metadata: Mapping[str, JsonValue] | None = None,
    ) -> StorageResetResult[AssetKey]:
        result = await self.apply_batch(
            (StorageChange(StorageOperation.RESET, key, None, expected_revision, metadata or {}),)
        )
        return result.results[0]

    async def apply_batch(
        self, changes: Sequence[StorageChange[AssetKey, bytes]], *, expected_revision: StorageRevision | None = None
    ) -> StorageBatchResult[AssetInfo, AssetKey]:
        self._ensure_ready()
        if len({change.key for change in changes}) != len(changes):
            raise AIError(ErrorCode.STORAGE_BATCH_DUPLICATE_KEY)
        for _ in range(_RETRY_LIMIT):
            result = await self._apply_once(changes, expected_revision)
            if result is not None:
                return result
        raise AIError(ErrorCode.STORAGE_CONFLICT)

    async def _apply_once(
        self, changes: Sequence[StorageChange[AssetKey, bytes]], expected_revision: StorageRevision | None
    ) -> StorageBatchResult[AssetInfo, AssetKey] | None:
        try:
            return await self._apply_once_transaction(changes, expected_revision)
        except AIError:
            raise
        except Exception as error:
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def _apply_once_transaction(
        self, changes: Sequence[StorageChange[AssetKey, bytes]], expected_revision: StorageRevision | None
    ) -> StorageBatchResult[AssetInfo, AssetKey] | None:
        current_revision = await self._head()
        if expected_revision is not None and int(expected_revision.value) != current_revision:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        current = await self._load_current(tuple(change.key for change in changes))
        for change in changes:
            previous = current[change.key]
            if change.expected_revision is not None and (
                previous is None or previous.revision != change.expected_revision
            ):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
        next_revision = (
            current_revision + 1
            if any(_mutates(change, current[change.key]) for change in changes)
            else current_revision
        )
        if next_revision == current_revision:
            return StorageBatchResult(
                StorageRevision(str(current_revision)),
                True,
                tuple(_unchanged(change, current[change.key], current_revision) for change in changes),
            )
        prepared: list[tuple[StorageChange[AssetKey, bytes], AssetInfo]] = []
        for change in changes:
            info = _next_info(change, current[change.key], next_revision, self._root)
            if change.operation is StorageOperation.PUT:
                content = bytes(change.value or b"")
                inline = StoredPayload.inline_bytes(content)
                if payload_fits_inline(inline, self._payload_policy):
                    info = replace(info, content=inline)
                else:
                    object_key = _asset_object_key(self._namespace_digest, info.etag)
                    await _put_asset_object(self._object_store, object_key, content)
                    info = replace(
                        info,
                        content=StoredPayload.object(
                            ObjectRef(
                                self._object_store.store_id,
                                object_key,
                                info.etag,
                                info.size,
                            )
                        ),
                    )
            prepared.append((change, info))
        entries = self._metadata.tables["ai_asset_entries"]
        history = self._metadata.tables["ai_asset_changes"]
        heads = self._metadata.tables["ai_asset_heads"]
        values: list[object] = []
        async def execute(session) -> StorageBatchResult[AssetInfo, AssetKey] | None:
            from sqlalchemy import func, update

            head_result = await session.execute(
                update(heads)
                .where(
                    heads.c.namespace_digest == self._namespace_digest.hex(),
                    heads.c.store_revision == current_revision,
                )
                .values(
                    store_revision=next_revision,
                    updated_at=func.current_timestamp(),
                )
            )
            if head_result.rowcount != 1:
                return None
            for change, info in prepared:
                data = _info_data(info)
                key_digest = _asset_key_digest(self._namespace_digest, change.key).hex()
                await session.execute(
                    history.insert().values(
                        key_digest=key_digest,
                        entry_revision=info.revision.value,
                        namespace_digest=self._namespace_digest.hex(),
                        store_revision=next_revision,
                        payload_json=data,
                    )
                )
                await self._context.dialect.upsert(
                    session,
                    table=entries,
                    values={
                        "key_digest": key_digest,
                        "namespace_digest": self._namespace_digest.hex(),
                        "entry_revision": info.revision.value,
                        "store_revision": next_revision,
                        "payload_json": data,
                    },
                    set_values={
                        "namespace_digest": self._namespace_digest.hex(),
                        "entry_revision": info.revision.value,
                        "store_revision": next_revision,
                        "payload_json": data,
                        "updated_at": func.current_timestamp(),
                    },
                    index_elements=("key_digest",),
                )
                values.append(_result(change, info, next_revision))
            return StorageBatchResult(StorageRevision(str(next_revision)), True, tuple(values))

        result = await self._context.run_mutation(execute)
        if result is None:
            return None
        return result

    async def _load_current(self, keys: Sequence[AssetKey]) -> dict[AssetKey, AssetInfo | None]:
        entries = self._metadata.tables["ai_asset_entries"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            digests = tuple(_asset_key_digest(self._namespace_digest, key).hex() for key in dict.fromkeys(keys))
            rows = (
                await session.execute(
                    select(entries.c.key_digest, entries.c.payload_json).where(
                        entries.c.namespace_digest == self._namespace_digest.hex(),
                        entries.c.key_digest.in_(digests),
                    )
                )
            ).all()
        finally:
            await session.close()
        by_digest = {str(row.key_digest): _info_from_data(row.payload_json) for row in rows}
        return {key: by_digest.get(_asset_key_digest(self._namespace_digest, key).hex()) for key in keys}

    async def list_versions(self, key: AssetKey) -> tuple[VersionSummary, ...]:
        self._ensure_ready()
        history = self._metadata.tables["ai_asset_changes"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            rows = (
                (
                    await session.execute(
                        select(history.c.payload_json)
                        .where(
                            history.c.key_digest == _asset_key_digest(self._namespace_digest, key).hex(),
                            history.c.namespace_digest == self._namespace_digest.hex(),
                        )
                        .order_by(history.c.entry_revision)
                    )
                )
                .scalars()
                .all()
            )
        finally:
            await session.close()
        return tuple(_summary(_info_from_data(row)) for row in rows)

    async def get_at_revision(self, key: AssetKey, entry_revision: StorageEntryRevision) -> bytes | None:
        history = self._metadata.tables["ai_asset_changes"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            data = await session.scalar(
                select(history.c.payload_json).where(
                    history.c.key_digest == _asset_key_digest(self._namespace_digest, key).hex(),
                    history.c.entry_revision == entry_revision.value,
                    history.c.namespace_digest == self._namespace_digest.hex(),
                )
            )
        finally:
            await session.close()
        if data is None:
            return None
        info = _info_from_data(data)
        return (
            None
            if info.status is not StorageEntryStatus.NORMAL
            else await _read_asset_object(self._object_store, self._namespace_digest, key, info)
        )

    async def get_at_version(self, key: AssetKey, version: int) -> bytes | None:
        return await self.get_at_revision(key, StorageEntryRevision(version))

    async def _head(self) -> int:
        heads = self._metadata.tables["ai_asset_heads"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            value = await session.scalar(
                select(heads.c.store_revision).where(heads.c.namespace_digest == self._namespace_digest.hex())
            )
        finally:
            await session.close()
        if value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return int(value)

    async def validate_integrity(self) -> None:
        self._ensure_ready()
        entries = self._metadata.tables["ai_asset_entries"]
        history = self._metadata.tables["ai_asset_changes"]
        heads = self._metadata.tables["ai_asset_heads"]
        session = self._context.sessions()
        try:
            from sqlalchemy import select

            namespace = self._namespace_digest.hex()
            head = await session.scalar(select(heads.c.store_revision).where(heads.c.namespace_digest == namespace))
            if head is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            head_revision = int(head)
            if head_revision < 0:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            entry_rows = (
                (
                    await session.execute(
                        select(entries.c.key_digest, entries.c.entry_revision, entries.c.payload_json).where(
                            entries.c.namespace_digest == namespace
                        )
                    )
                )
                .mappings()
                .all()
            )
            history_rows = (
                (
                    await session.execute(
                        select(history.c.key_digest, history.c.entry_revision, history.c.payload_json)
                        .where(history.c.namespace_digest == namespace)
                        .order_by(history.c.key_digest, history.c.entry_revision)
                    )
                )
                .mappings()
                .all()
            )
        finally:
            await session.close()
        parsed_entries = [(row, self._audit_info(row, head_revision)) for row in entry_rows]
        parsed_history = [(row, self._audit_info(row, head_revision)) for row in history_rows]
        checked_objects: set[str] = set()
        for _, info in (*parsed_entries, *parsed_history):
            object_key = _asset_object_key(self._namespace_digest, info.etag)
            if info.status is not StorageEntryStatus.NORMAL or object_key in checked_objects:
                continue
            checked_objects.add(object_key)
            if info.content is not None and info.content.kind == "inline":
                continue
            stat = await self._object_store.stat(object_key)
            if stat is None or stat.digest != info.etag or stat.size != info.size:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        grouped: dict[str, list[tuple[Mapping[str, object], AssetInfo]]] = {}
        for row, info in parsed_history:
            grouped.setdefault(str(row["key_digest"]), []).append((row, info))
        current_by_key = {str(row["key_digest"]): (row, info) for row, info in parsed_entries}
        for key, rows in grouped.items():
            revisions = [info.revision.value for _, info in rows]
            if revisions != list(range(1, len(revisions) + 1)):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            current = current_by_key.get(key)
            if current is None or current[1].revision.value != revisions[-1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if current[1] != rows[-1][1]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            store_revisions = [int(info.store_revision.value) for _, info in rows]
            if store_revisions != sorted(store_revisions) or len(set(store_revisions)) != len(store_revisions):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if set(current_by_key) != set(grouped):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        store_revisions = [int(info.store_revision.value) for _, info in (*parsed_entries, *parsed_history)]
        if head_revision == 0 and store_revisions:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if head_revision > 0 and (not store_revisions or max(store_revisions) != head_revision):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _logger.info("SQL Asset integrity validated: namespace=%s", self._root.digest[:16])

    def _audit_info(self, row: Mapping[str, object], head: int) -> AssetInfo:
        try:
            info = _info_from_data(row["payload_json"])
            if str(row["key_digest"]) != _asset_key_digest(self._namespace_digest, info.key).hex():
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if int(row["entry_revision"]) != info.revision.value:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if info.root_id != self._root.root_id or info.root_digest != self._root.digest:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            store_revision = int(info.store_revision.value)
            if store_revision < 1 or store_revision > head:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return info
        except AIError:
            raise
        except (KeyError, TypeError, ValueError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _ensure_ready(self) -> None:
        if not self._ready:
            raise AIError(ErrorCode.STORAGE_DEPENDENCY_NOT_READY)


def _asset_key_digest(namespace_digest: bytes, key: AssetKey) -> bytes:
    return hashlib.sha256(
        canonical_json_bytes({"namespace_digest": namespace_digest.hex(), "kind": key.kind, "id": key.id})
    ).digest()


def _info_data(info: AssetInfo) -> dict[str, JsonValue]:
    return {
        "v": 1,
        "value": {
            "kind": info.key.kind,
            "id": info.key.id,
            "revision": info.revision.value,
            "store_revision": info.store_revision.value,
            "etag": info.etag,
            "size": info.size,
            "status": info.status.value,
            "root_id": info.root_id,
            "root_digest": info.root_digest,
            "modified_at": info.modified_at.isoformat(),
            "metadata": dict(info.metadata),
            "content": None if info.content is None else info.content.to_json(),
        },
    }


def _info_from_data(data: Mapping[str, object]) -> AssetInfo:
    value = data["value"] if isinstance(data.get("value"), Mapping) else data
    return AssetInfo(
        AssetKey(str(value["kind"]), str(value["id"])),
        StorageEntryRevision(int(value["revision"])),
        StorageRevision(str(value["store_revision"])),
        str(value["etag"]),
        int(value["size"]),
        StorageEntryStatus(str(value["status"])),
        str(value["root_id"]),
        str(value["root_digest"]),
        datetime.fromisoformat(str(value["modified_at"])),
        dict(value.get("metadata", {})),
        None if value.get("content") is None else StoredPayload.from_json(value["content"]),
    )


def _key_from_data(data: Mapping[str, object]) -> AssetKey:
    value = data["value"]
    return AssetKey(str(value["kind"]), str(value["id"]))


def _next_info(
    change: StorageChange[AssetKey, bytes], previous: AssetInfo | None, store_revision: int, root: AssetRoot
) -> AssetInfo:
    revision = 1 if previous is None else previous.revision.value + 1
    value = bytes(change.value or b"") if change.operation is StorageOperation.PUT else b""
    status = {
        StorageOperation.PUT: StorageEntryStatus.NORMAL,
        StorageOperation.DELETE: StorageEntryStatus.DELETED,
        StorageOperation.RESET: StorageEntryStatus.RESET,
    }[change.operation]
    return AssetInfo(
        change.key,
        StorageEntryRevision(revision),
        StorageRevision(str(store_revision)),
        hashlib.sha256(value).hexdigest(),
        len(value),
        status,
        root.root_id,
        root.digest,
        datetime.now(timezone.utc),
        change.metadata,
    )


def _summary(info: AssetInfo) -> VersionSummary:
    return VersionSummary(info.revision, info.etag, info.size, info.modified_at, info.status, info.metadata)


def _result(change: StorageChange[AssetKey, bytes], info: AssetInfo, store_revision: int) -> object:
    if change.operation is StorageOperation.PUT:
        return StoragePutResult(info, info.revision, StorageRevision(str(store_revision)), True)
    if change.operation is StorageOperation.DELETE:
        return StorageDeleteResult(change.key, True, info.revision, StorageRevision(str(store_revision)))
    return StorageResetResult(change.key, True, StorageRevision(str(store_revision)))


def _unchanged(change: StorageChange[AssetKey, bytes], info: AssetInfo | None, store_revision: int) -> object:
    if change.operation is StorageOperation.PUT and info is not None:
        return StoragePutResult(info, info.revision, StorageRevision(str(store_revision)), False)
    if change.operation is StorageOperation.DELETE:
        return StorageDeleteResult(change.key, False, None, StorageRevision(str(store_revision)))
    return StorageResetResult(change.key, False, StorageRevision(str(store_revision)))


def _mutates(change: StorageChange[AssetKey, bytes], previous: AssetInfo | None) -> bool:
    if previous is None:
        return True
    if change.operation is StorageOperation.PUT:
        return (
            previous.status is not StorageEntryStatus.NORMAL
            or previous.etag != hashlib.sha256(bytes(change.value or b"")).hexdigest()
        )
    target = {
        StorageOperation.DELETE: StorageEntryStatus.DELETED,
        StorageOperation.RESET: StorageEntryStatus.RESET,
    }[change.operation]
    return previous.status is not target


async def _put_asset_object(store: ObjectStore, key: str, value: bytes) -> None:
    digest = hashlib.sha256(value).hexdigest()

    async def chunks() -> AsyncIterator[bytes]:
        yield value

    await store.put(key, chunks(), expected_size=len(value), expected_digest=digest)


async def _read_asset_object(store: ObjectStore, namespace_digest: bytes, key: AssetKey, info: AssetInfo) -> bytes:
    if info.content is not None and info.content.kind == "inline":
        value = info.content.decode()
        if not isinstance(value, bytes):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value
    if info.content is not None and info.content.ref is not None:
        return await _read_object_bytes(
            store,
            info.content.ref.key,
            info.etag,
            info.size,
        )
    return await _read_object_bytes(
        store, _asset_object_key(namespace_digest, info.etag), info.etag, info.size
    )


async def _read_object_bytes(store: ObjectStore, key: str, digest: str, size: int) -> bytes:
    from ..storage import read_object

    return await read_object(store, key, expected_digest=digest, expected_size=size)


def _asset_object_key(namespace_digest: bytes, digest: str) -> str:
    return (
        "asset/"
        + hashlib.sha256(
            canonical_json_bytes(
                {
                    "namespace_digest": namespace_digest.hex(),
                    "content_digest": digest,
                }
            )
        ).hexdigest()
    )


__all__ = ["SqlAssetBackend", "build_asset_sql_metadata"]
