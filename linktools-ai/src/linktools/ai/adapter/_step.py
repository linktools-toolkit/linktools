#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL adapters for the public Pydantic AI Harness StepStore contract."""

import asyncio
import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from linktools.core import environ
from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaContext, media_uri_for, parse_media_uri
from pydantic_ai_harness.step_persistence import (
    ContinuableSnapshot,
    RunRecord,
    StepEvent,
    ToolEffectRecord,
)

from ..core import canonical_json_bytes, validate_persistence_namespace
from ..errors import AIError, ErrorCode
from ..storage import (
    FilesystemWriterLock,
    SqlErrorKind,
    classify_sql_error,
    read_json,
    resolve_dialect,
    sql_blob,
    sql_digest,
    sql_index,
    sql_integer_id,
    sql_table_options,
    storage_name,
    validate_schema,
    write_json_atomic,
)
from ._schema import new_step_metadata

if TYPE_CHECKING:
    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker


_logger = environ.get_logger("ai.adapter.step")


class DurableFilesystemStepStore:
    """Immutable-file StepStore with crash recovery checks and fsync barriers."""

    def __init__(self, root: str | Path, namespace: str, *, writer_lock: FilesystemWriterLock | None = None) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("StepStore namespace is invalid") from error
        runtime_root = Path(root).expanduser().resolve()
        self._namespace = namespace
        self._root = runtime_root / "steps" / _file_digest(namespace)
        self._lock = asyncio.Lock()
        self._writer_lock = writer_lock or FilesystemWriterLock(runtime_root / "step.lock")
        self._owns_writer_lock = writer_lock is None
        self._closed = True

    async def initialize(self) -> None:
        async with self._lock:
            self._root.mkdir(parents=True, exist_ok=True)
            if any(path.name.endswith(".tmp") for path in self._root.rglob("*")):
                raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
            if self._owns_writer_lock:
                await self._writer_lock.acquire()
            self._closed = False
            _logger.debug("filesystem step store initialized: root=%s namespace=%s", self._root, self._namespace)

    async def close(self) -> None:
        if self._owns_writer_lock:
            await self._writer_lock.release()
        self._closed = True

    async def register_run(self, record: RunRecord) -> None:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(record.run_id).parent
            try:
                directory.mkdir(parents=True, exist_ok=False)
            except FileExistsError as error:
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            write_json_atomic(directory / "run.json", _file_run_json(record), fsync=True)
            _fsync_directory(directory)

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        async with self._lock:
            self._ensure_open()
            path = self._run_path(run_id)
            return None if not path.is_file() else _file_run_from_json(_file_read(path))

    async def list_runs(self, *, parent_run_id: str | None = None, conversation_id: str | None = None) -> list[RunRecord]:
        async with self._lock:
            self._ensure_open()
            values: list[RunRecord] = []
            for path in self._root.glob("runs/*/run.json"):
                record = _file_run_from_json(_file_read(path))
                if parent_run_id is not None and record.parent_run_id != parent_run_id:
                    continue
                if conversation_id is not None and record.conversation_id != conversation_id:
                    continue
                values.append(record)
            return sorted(values, key=lambda item: (item.started_at, item.run_id))

    async def append_event(self, event: StepEvent) -> None:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(event.run_id).parent / "events"
            directory.mkdir(parents=True, exist_ok=True)
            sequence = len(tuple(directory.glob("event-*.json")))
            path = directory / f"event-{sequence:020d}.json"
            write_json_atomic(path, _file_event_json(event), fsync=True)
            _fsync_directory(directory)

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(run_id).parent / "events"
            return [_file_event_from_json(_file_read(path)) for path in sorted(directory.glob("event-*.json"))]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(snapshot.run_id).parent / "snapshots"
            directory.mkdir(parents=True, exist_ok=True)
            index = len(tuple(directory.glob("snapshot-*.json")))
            payload = {
                "run_id": snapshot.run_id,
                "step_index": snapshot.step_index,
                "conversation_id": snapshot.conversation_id,
                "parent_run_id": snapshot.parent_run_id,
                "agent_name": snapshot.agent_name,
                "timestamp": _file_time_json(snapshot.timestamp),
                "state": "complete",
                "messages": json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages)),
            }
            write_json_atomic(directory / f"snapshot-{index:020d}.json", payload, fsync=True)
            _fsync_directory(directory)

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        async with self._lock:
            self._ensure_open()
            paths = sorted((self._run_path(run_id).parent / "snapshots").glob("snapshot-*.json"))
            for path in reversed(paths):
                value = _file_read(path)
                if include_interrupted or value.get("state", "complete") == "complete":
                    return _file_snapshot_from_json(value)
            return None

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(run_id).parent / "snapshots"
            snapshots = [
                _file_snapshot_from_json(value)
                for path in sorted(directory.glob("snapshot-*.json"))
                if (value := _file_read(path)).get("state", "complete") == "complete"
            ]
            return snapshots

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(record.run_id).parent / "effects"
            directory.mkdir(parents=True, exist_ok=True)
            index = len(tuple(directory.glob("effect-*.json")))
            write_json_atomic(directory / f"effect-{index:020d}.json", _file_effect_json(record), fsync=True)
            _fsync_directory(directory)

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        values = await self._effects(run_id)
        matching = [item for item in values if item.tool_call_id == tool_call_id]
        return matching[-1] if matching else None

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        values = await self._effects(run_id)
        latest: dict[str, ToolEffectRecord] = {}
        for item in values:
            latest[item.tool_call_id] = item
        return [item for item in latest.values() if item.status == "started"]

    async def _effects(self, run_id: str) -> list[ToolEffectRecord]:
        async with self._lock:
            self._ensure_open()
            directory = self._run_path(run_id).parent / "effects"
            return [_file_effect_from_json(_file_read(path)) for path in sorted(directory.glob("effect-*.json"))]

    def _run_path(self, run_id: str) -> Path:
        _validate_file_id(run_id)
        return self._root / "runs" / _file_digest(run_id) / "run.json"

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.STORAGE_CLOSED)


def _file_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _validate_file_id(value: str) -> None:
    if not value or len(value) > 200 or value in {".", ".."} or any(char in value for char in "/\\\x00"):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _file_read(path: Path) -> dict[str, object]:
    try:
        value = read_json(path)
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if not isinstance(value, dict):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return value


def _file_time_json(value: datetime) -> str:
    if value.tzinfo is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.astimezone(timezone.utc).isoformat()


def _file_datetime(value: object) -> datetime:
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED) from error
    if parsed.tzinfo is None:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return parsed.astimezone(timezone.utc)


def _file_run_json(record: RunRecord) -> dict[str, object]:
    return {"run_id": record.run_id, "conversation_id": record.conversation_id, "parent_run_id": record.parent_run_id, "agent_name": record.agent_name, "metadata": dict(record.metadata), "started_at": _file_time_json(record.started_at)}


def _file_run_from_json(value: dict[str, object]) -> RunRecord:
    metadata = value.get("metadata", {})
    if not isinstance(metadata, dict) or any(not isinstance(key, str) or not isinstance(item, str) for key, item in metadata.items()):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return RunRecord(run_id=str(value["run_id"]), conversation_id=_file_optional(value.get("conversation_id")), parent_run_id=_file_optional(value.get("parent_run_id")), agent_name=_file_optional(value.get("agent_name")), metadata=metadata, started_at=_file_datetime(value["started_at"]))


def _file_event_json(event: StepEvent) -> dict[str, object]:
    return {**asdict(event), "timestamp": _file_time_json(event.timestamp), "metadata": dict(event.metadata)}


def _file_event_from_json(value: dict[str, object]) -> StepEvent:
    return StepEvent(run_id=str(value["run_id"]), kind=value["kind"], step_index=int(value["step_index"]), timestamp=_file_datetime(value["timestamp"]), conversation_id=_file_optional(value.get("conversation_id")), parent_run_id=_file_optional(value.get("parent_run_id")), agent_name=_file_optional(value.get("agent_name")), tool_call_id=_file_optional(value.get("tool_call_id")), tool_name=_file_optional(value.get("tool_name")), error=_file_optional(value.get("error")), metadata=_file_string_map(value.get("metadata", {})))


def _file_snapshot_from_json(value: dict[str, object]) -> ContinuableSnapshot:
    state = str(value.get("state", "complete"))
    if state not in {"complete", "interrupted"}:
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return ContinuableSnapshot(run_id=str(value["run_id"]), step_index=int(value["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(value.get("messages", [])), conversation_id=_file_optional(value.get("conversation_id")), parent_run_id=_file_optional(value.get("parent_run_id")), agent_name=_file_optional(value.get("agent_name")), timestamp=_file_datetime(value["timestamp"]))


def _file_effect_json(record: ToolEffectRecord) -> dict[str, object]:
    return {**asdict(record), "started_at": _file_time_json(record.started_at), "ended_at": None if record.ended_at is None else _file_time_json(record.ended_at)}


def _file_effect_from_json(value: dict[str, object]) -> ToolEffectRecord:
    return ToolEffectRecord(tool_call_id=str(value["tool_call_id"]), tool_name=str(value["tool_name"]), run_id=str(value["run_id"]), status=value["status"], started_at=_file_datetime(value["started_at"]), ended_at=None if value.get("ended_at") is None else _file_datetime(value["ended_at"]), idempotency_key=_file_optional(value.get("idempotency_key")), effect_summary=_file_optional(value.get("effect_summary")))


def _file_optional(value: object) -> str | None:
    return None if value is None else str(value)


def _file_string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()):
        raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
    return dict(value)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    except OSError as error:
        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error


class SqlStepStore:
    """Namespace-bound MySQL/PostgreSQL StepStore implementation."""

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]", namespace: str) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("StepStore namespace is invalid") from error
        self._sessions = session_factory
        self._namespace = namespace
        self._namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self._metadata, self._tables = _build_tables()
        self._schema_digest = _schema_digest(self._metadata)
        self._media = SqlMediaStore(session_factory, namespace, metadata=self._metadata, tables=self._tables)

    @property
    def schema_digest(self) -> str:
        return self._schema_digest

    async def initialize(self) -> None:
        await validate_schema(self._sessions, self._metadata)

    async def close(self) -> None:
        return None

    async def register_run(self, record: RunRecord) -> None:
        from sqlalchemy import insert
        table = self._tables["runs"]
        try:
            async with self._sessions() as session:
                async with session.begin():
                    await session.execute(insert(table).values(_run_values(self._namespace_key, record)))
        except Exception as error:
            if classify_sql_error(error) is SqlErrorKind.INTEGRITY:
                raise AIError(ErrorCode.STORAGE_CONFLICT) from error
            raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        from sqlalchemy import select
        table = self._tables["runs"]
        async with self._sessions() as session:
            row = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id))).mappings().first()
        return None if row is None else _run_from_row(row)

    async def list_runs(self, *, parent_run_id: "str | None" = None, conversation_id: "str | None" = None) -> "list[RunRecord]":
        from sqlalchemy import select
        table = self._tables["runs"]
        predicates = [table.c.namespace_key == self._namespace_key]
        if parent_run_id is not None:
            predicates.append(table.c.parent_run_id == parent_run_id)
        if conversation_id is not None:
            predicates.append(table.c.conversation_id == conversation_id)
        async with self._sessions() as session:
            rows = (await session.execute(select(table).where(*predicates).order_by(table.c.started_at, table.c.run_id))).mappings().all()
        return [_run_from_row(row) for row in rows]

    async def append_event(self, event: StepEvent) -> None:
        from sqlalchemy import insert
        table = self._tables["events"]
        async with self._sessions() as session:
            async with session.begin():
                await session.execute(insert(table).values(_event_values(self._namespace_key, event)))

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        from sqlalchemy import select
        table = self._tables["events"]
        async with self._sessions() as session:
            rows = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id).order_by(table.c.id))).mappings().all()
        return [_event_from_row(row) for row in rows]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        from sqlalchemy import insert
        table = self._tables["snapshots"]
        async with self._sessions() as session:
            async with session.begin():
                await session.execute(insert(table).values(_snapshot_values(self._namespace_key, snapshot)))

    async def latest_snapshot(self, *, run_id: str, include_interrupted: bool = False) -> ContinuableSnapshot | None:
        from sqlalchemy import select
        table = self._tables["snapshots"]
        async with self._sessions() as session:
            statement = select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id)
            if not include_interrupted:
                statement = statement.where(table.c.state == "complete")
            row = (await session.execute(statement.order_by(table.c.id.desc()).limit(1))).mappings().first()
        return None if row is None else _snapshot_from_row(row)

    async def list_snapshots(self, *, run_id: str) -> list[ContinuableSnapshot]:
        from sqlalchemy import select
        table = self._tables["snapshots"]
        async with self._sessions() as session:
            rows = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id, table.c.state == "complete").order_by(table.c.id))).mappings().all()
        snapshots = [_snapshot_from_row(row) for row in rows]
        return snapshots

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        values = _effect_values(self._namespace_key, record)
        table = self._tables["effects"]
        async with self._sessions() as session:
            async with session.begin():
                await resolve_dialect(session).upsert(
                    session,
                    table=table,
                    values=values,
                    set_values={
                        "tool_name": values["tool_name"],
                        "status": values["status"],
                        "started_at": values["started_at"],
                        "ended_at": values["ended_at"],
                        "idempotency_key": values["idempotency_key"],
                        "effect_summary": values["effect_summary"],
                        "updated_at": values["updated_at"],
                    },
                    index_elements=("namespace_key", "run_id", "tool_call_id"),
                )

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        from sqlalchemy import select
        table = self._tables["effects"]
        async with self._sessions() as session:
            row = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id, table.c.tool_call_id == tool_call_id))).mappings().first()
        return None if row is None else _effect_from_row(row)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        from sqlalchemy import select
        table = self._tables["effects"]
        async with self._sessions() as session:
            rows = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id, table.c.status == "started"))).mappings().all()
        return [_effect_from_row(row) for row in rows]


class SqlMediaStore:
    """Content-addressed SQL media store used by Harness snapshots."""

    def __init__(self, session_factory: "async_sessionmaker[AsyncSession]", namespace: str, *, metadata: "object | None" = None, tables: "dict[str, object] | None" = None) -> None:
        try:
            validate_persistence_namespace(namespace)
        except AIError as error:
            raise ValueError("media namespace is invalid") from error
        self._sessions = session_factory
        self._namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        if metadata is None or tables is None:
            self._metadata, self._tables = _build_tables()
        else:
            self._metadata, self._tables = metadata, tables
        self._schema_digest = _schema_digest(self._metadata)

    @property
    def schema_digest(self) -> str:
        return self._schema_digest

    async def put(self, data: bytes, *, context: MediaContext = MediaContext()) -> str:
        from sqlalchemy import insert
        uri = media_uri_for(data)
        digest = parse_media_uri(uri)
        table = self._tables["media"]
        values = {"namespace_key": self._namespace_key, "sha256": digest, "media_type": context.media_type, "bytes": data, "size_bytes": len(data), "metadata_json": dict(context.metadata)}
        async with self._sessions() as session:
            async with session.begin():
                try:
                    await session.execute(insert(table).values(values))
                except Exception as error:
                    if classify_sql_error(error) is not SqlErrorKind.INTEGRITY:
                        raise AIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        return uri

    async def get(self, uri: str, *, context: MediaContext = MediaContext()) -> bytes:
        row = await self._media_row(uri)
        if row is None:
            raise FileNotFoundError(uri)
        data = bytes(row["bytes"])
        _validate_media(row, data)
        return data

    async def exists(self, uri: str, *, context: MediaContext = MediaContext()) -> bool:
        from sqlalchemy import select
        digest = parse_media_uri(uri)
        table = self._tables["media"]
        async with self._sessions() as session:
            row_id = await session.scalar(
                select(table.c.id).where(
                    table.c.namespace_key == self._namespace_key,
                    table.c.sha256 == digest,
                )
            )
        return row_id is not None

    async def public_url(self, uri: str, *, context: MediaContext = MediaContext()) -> str | None:
        return None

    async def get_metadata(self, uri: str, *, context: MediaContext = MediaContext()) -> Mapping[str, str]:
        row = await self._media_row(uri)
        if row is None:
            raise FileNotFoundError(uri)
        data = bytes(row["bytes"])
        _validate_media(row, data)
        metadata = row["metadata_json"]
        if not isinstance(metadata, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in metadata.items()):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return metadata

    async def _media_row(self, uri: str) -> object | None:
        from sqlalchemy import select
        digest = parse_media_uri(uri)
        table = self._tables["media"]
        async with self._sessions() as session:
            return (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.sha256 == digest))).mappings().first()


def _build_tables() -> tuple[object, dict[str, object]]:
    from sqlalchemy import (
        JSON,
        BigInteger,
        Column,
        DateTime,
        Index,
        Integer,
        String,
        Table,
        Text,
        UniqueConstraint,
    )
    from sqlalchemy.sql import func

    metadata = new_step_metadata()
    key = sql_digest()
    integer_id = sql_integer_id()
    tables = {
        "runs": Table(
            storage_name("step_runs"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("namespace_key", key, nullable=False),
            Column("run_id", String(200), nullable=False),
            Column("conversation_id", String(200), nullable=False),
            Column("parent_run_id", String(200)),
            Column("agent_name", String(256)),
            Column("metadata_json", JSON, nullable=False),
            Column("started_at", DateTime(timezone=True), nullable=False),
            Column("run_key", key, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
            Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
            UniqueConstraint("namespace_key", "run_id", name="uk_namespace_key_run_id"),
            UniqueConstraint("namespace_key", "run_key", name="uk_namespace_key_run_key"),
            **sql_table_options(),
        ),
        "events": Table(
            storage_name("step_events"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("namespace_key", key, nullable=False),
            Column("run_id", String(200), nullable=False),
            Column("kind", String(64), nullable=False),
            Column("step_index", Integer, nullable=False),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("conversation_id", String(200)),
            Column("parent_run_id", String(200)),
            Column("agent_name", String(256)),
            Column("tool_call_id", String(256)),
            Column("tool_name", String(256)),
            Column("error", Text),
            Column("metadata_json", JSON, nullable=False),
            sql_index(Index("ix_namespace_key_run_id", "namespace_key", "run_id")),
            Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
            Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
            **sql_table_options(),
        ),
        "snapshots": Table(
            storage_name("step_snapshots"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("namespace_key", key, nullable=False),
            Column("run_id", String(200), nullable=False),
            Column("step_index", Integer, nullable=False),
            Column("conversation_id", String(200)),
            Column("parent_run_id", String(200)),
            Column("agent_name", String(256)),
            Column("timestamp", DateTime(timezone=True), nullable=False),
            Column("state", String(16), nullable=False, server_default="complete"),
            Column("messages_json", JSON, nullable=False),
            sql_index(Index("ix_namespace_key_run_id", "namespace_key", "run_id")),
            Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
            Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
            **sql_table_options(),
        ),
        "effects": Table(
            storage_name("step_effects"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("namespace_key", key, nullable=False),
            Column("run_id", String(200), nullable=False),
            Column("tool_call_id", String(256), nullable=False),
            Column("tool_name", String(256), nullable=False),
            Column("status", String(64), nullable=False),
            Column("started_at", DateTime(timezone=True), nullable=False),
            Column("ended_at", DateTime(timezone=True)),
            Column("idempotency_key", String(256)),
            Column("effect_summary", Text),
            Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
            Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
            UniqueConstraint("namespace_key", "run_id", "tool_call_id", name="uk_namespace_key_run_id_tool_call_id"),
            **sql_table_options(),
        ),
        "media": Table(
            storage_name("step_media"),
            metadata,
            Column("id", integer_id, primary_key=True, autoincrement=True),
            Column("namespace_key", key, nullable=False),
            Column("sha256", key, nullable=False),
            Column("media_type", String(256)),
            Column("bytes", sql_blob(), nullable=False),
            Column("size_bytes", BigInteger, nullable=False),
            Column("metadata_json", JSON, nullable=False),
            Column("updated_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp(), onupdate=func.current_timestamp()),
            Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.current_timestamp()),
            UniqueConstraint("namespace_key", "sha256", name="uk_namespace_key_sha256"),
            **sql_table_options(),
        ),
    }
    for name, table in tables.items():
        sql_index(Index("ix_updated_at", table.c.updated_at))
        sql_index(Index("ix_created_at", table.c.created_at))
    return metadata, tables


def build_step_schema() -> "MetaData":
    """Build StepStore metadata for explicit deployment schema provisioning."""
    metadata, _ = _build_tables()
    return metadata


def _run_values(namespace_key: str, record: RunRecord) -> dict[str, object]:
    if record.conversation_id is None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    started_at = _utc(record.started_at)
    return {"namespace_key": namespace_key, "run_id": record.run_id, "conversation_id": record.conversation_id, "parent_run_id": record.parent_run_id, "agent_name": record.agent_name, "metadata_json": dict(record.metadata), "started_at": started_at, "run_key": _file_digest(f"{record.conversation_id}\x00{started_at.isoformat()}\x00{record.run_id}")}


def _event_values(namespace_key: str, event: StepEvent) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": event.run_id, "kind": event.kind, "step_index": event.step_index, "timestamp": _utc(event.timestamp), "conversation_id": event.conversation_id, "parent_run_id": event.parent_run_id, "agent_name": event.agent_name, "tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "error": event.error, "metadata_json": dict(event.metadata)}


def _snapshot_values(namespace_key: str, snapshot: ContinuableSnapshot) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": snapshot.run_id, "step_index": snapshot.step_index, "conversation_id": snapshot.conversation_id, "parent_run_id": snapshot.parent_run_id, "agent_name": snapshot.agent_name, "timestamp": _utc(snapshot.timestamp), "state": "complete", "messages_json": json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages))}


def _effect_values(namespace_key: str, record: ToolEffectRecord) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": record.run_id, "tool_call_id": record.tool_call_id, "tool_name": record.tool_name, "status": record.status, "started_at": _utc(record.started_at), "ended_at": None if record.ended_at is None else _utc(record.ended_at), "idempotency_key": record.idempotency_key, "effect_summary": record.effect_summary}


def _run_from_row(row: Mapping[str, object]) -> RunRecord:
    return RunRecord(run_id=str(row["run_id"]), conversation_id=_optional(row["conversation_id"]), parent_run_id=_optional(row["parent_run_id"]), agent_name=_optional(row["agent_name"]), metadata=_string_map(row["metadata_json"]), started_at=_utc(row["started_at"]))


def _event_from_row(row: Mapping[str, object]) -> StepEvent:
    return StepEvent(run_id=str(row["run_id"]), kind=row["kind"], step_index=int(row["step_index"]), timestamp=_utc(row["timestamp"]), conversation_id=_optional(row["conversation_id"]), parent_run_id=_optional(row["parent_run_id"]), agent_name=_optional(row["agent_name"]), tool_call_id=_optional(row["tool_call_id"]), tool_name=_optional(row["tool_name"]), error=_optional(row["error"]), metadata=_string_map(row["metadata_json"]))


def _snapshot_from_row(row: Mapping[str, object]) -> ContinuableSnapshot:
    state = str(row.get("state", "complete"))
    if state not in {"complete", "interrupted"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return ContinuableSnapshot(run_id=str(row["run_id"]), step_index=int(row["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(row["messages_json"]), conversation_id=_optional(row["conversation_id"]), parent_run_id=_optional(row["parent_run_id"]), agent_name=_optional(row["agent_name"]), timestamp=_utc(row["timestamp"]))


def _effect_from_row(row: Mapping[str, object]) -> ToolEffectRecord:
    return ToolEffectRecord(tool_call_id=str(row["tool_call_id"]), tool_name=str(row["tool_name"]), run_id=str(row["run_id"]), status=row["status"], started_at=_utc(row["started_at"]), ended_at=None if row["ended_at"] is None else _utc(row["ended_at"]), idempotency_key=_optional(row["idempotency_key"]), effect_summary=_optional(row["effect_summary"]))


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.astimezone(timezone.utc)


def _validate_media(row: Mapping[str, object], data: bytes) -> None:
    if len(data) != int(row["size_bytes"]) or hashlib.sha256(data).hexdigest() != str(row["sha256"]):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _schema_digest(metadata: "MetaData") -> str:
    manifest = {
        str(name): [str(column.name) + ":" + str(column.type) for column in table.columns]
        for name, table in sorted(metadata.tables.items())
    }
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


__all__ = ["DurableFilesystemStepStore", "SqlMediaStore", "SqlStepStore", "build_step_schema"]
