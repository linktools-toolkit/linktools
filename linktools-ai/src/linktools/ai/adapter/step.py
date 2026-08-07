#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL adapters for the public Pydantic AI Harness StepStore contract."""

import hashlib
import json
from datetime import datetime, timezone
from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai_harness.media import MediaContext, media_uri_for, parse_media_uri
from pydantic_ai_harness.step_persistence import ContinuableSnapshot, RunRecord, StepEvent, ToolEffectRecord

from ..core.errors import ErrorCode, LinktoolsAIError
from ..core.json import canonical_json_bytes
from ..storage.database import StorageDatabase
from ..storage.names import storage_name
from .schema import new_step_metadata

if TYPE_CHECKING:
    from sqlalchemy import MetaData


class SqlStepStore:
    """Namespace-bound MySQL/PostgreSQL StepStore implementation."""

    def __init__(self, database: StorageDatabase, namespace: str) -> None:
        self._database = database
        self._namespace = namespace
        self._namespace_key = hashlib.sha256(namespace.encode("utf-8")).hexdigest()
        self._metadata, self._tables = _build_tables()
        self._schema_digest = _schema_digest(self._metadata)
        self._media = SqlMediaStore(database, namespace, metadata=self._metadata, tables=self._tables)

    @property
    def schema_digest(self) -> str:
        return self._schema_digest

    async def initialize(self) -> None:
        async with self._database.engine.begin() as connection:
            await connection.run_sync(self._metadata.create_all)

    async def close(self) -> None:
        return None

    async def register_run(self, record: RunRecord) -> None:
        from sqlalchemy import insert
        table = self._tables["runs"]
        try:
            async with self._database.session_factory() as session:
                async with session.begin():
                    await session.execute(insert(table).values(_run_values(self._namespace_key, record)))
        except Exception as error:
            if _is_integrity(error):
                raise LinktoolsAIError(ErrorCode.STORAGE_CONFLICT) from error
            raise LinktoolsAIError(ErrorCode.STORAGE_UNAVAILABLE) from error

    async def get_run(self, *, run_id: str) -> RunRecord | None:
        from sqlalchemy import select
        table = self._tables["runs"]
        async with self._database.session_factory() as session:
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
        async with self._database.session_factory() as session:
            rows = (await session.execute(select(table).where(*predicates).order_by(table.c.started_at, table.c.run_id))).mappings().all()
        return [_run_from_row(row) for row in rows]

    async def append_event(self, event: StepEvent) -> None:
        from sqlalchemy import insert
        table = self._tables["events"]
        async with self._database.session_factory() as session:
            async with session.begin():
                await session.execute(insert(table).values(_event_values(self._namespace_key, event)))

    async def list_events(self, *, run_id: str) -> list[StepEvent]:
        from sqlalchemy import select
        table = self._tables["events"]
        async with self._database.session_factory() as session:
            rows = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id).order_by(table.c.seq))).mappings().all()
        return [_event_from_row(row) for row in rows]

    async def save_snapshot(self, snapshot: ContinuableSnapshot) -> None:
        from sqlalchemy import insert
        table = self._tables["snapshots"]
        async with self._database.session_factory() as session:
            async with session.begin():
                await session.execute(insert(table).values(_snapshot_values(self._namespace_key, snapshot)))

    async def latest_snapshot(self, *, run_id: str) -> ContinuableSnapshot | None:
        from sqlalchemy import select
        table = self._tables["snapshots"]
        async with self._database.session_factory() as session:
            row = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id).order_by(table.c.seq.desc()).limit(1))).mappings().first()
        return None if row is None else _snapshot_from_row(row)

    async def record_tool_effect(self, record: ToolEffectRecord) -> None:
        from sqlalchemy import delete, insert
        table = self._tables["effects"]
        async with self._database.session_factory() as session:
            async with session.begin():
                await session.execute(delete(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == record.run_id, table.c.tool_call_id == record.tool_call_id))
                await session.execute(insert(table).values(_effect_values(self._namespace_key, record)))

    async def get_tool_effect(self, *, run_id: str, tool_call_id: str) -> ToolEffectRecord | None:
        from sqlalchemy import select
        table = self._tables["effects"]
        async with self._database.session_factory() as session:
            row = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id, table.c.tool_call_id == tool_call_id))).mappings().first()
        return None if row is None else _effect_from_row(row)

    async def list_unresolved_tool_effects(self, *, run_id: str) -> list[ToolEffectRecord]:
        from sqlalchemy import select
        table = self._tables["effects"]
        async with self._database.session_factory() as session:
            rows = (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.run_id == run_id, table.c.status == "started"))).mappings().all()
        return [_effect_from_row(row) for row in rows]


class SqlMediaStore:
    """Content-addressed SQL media store used by Harness snapshots."""

    def __init__(self, database: StorageDatabase, namespace: str, *, metadata: "object | None" = None, tables: "dict[str, object] | None" = None) -> None:
        self._database = database
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
        async with self._database.session_factory() as session:
            async with session.begin():
                try:
                    await session.execute(insert(table).values(values))
                except Exception as error:
                    if not _is_integrity(error):
                        raise LinktoolsAIError(ErrorCode.STORAGE_UNAVAILABLE) from error
        return uri

    async def get(self, uri: str, *, context: MediaContext = MediaContext()) -> bytes:
        row = await self._media_row(uri)
        if row is None:
            raise FileNotFoundError(uri)
        data = bytes(row["bytes"])
        _validate_media(row, data)
        return data

    async def exists(self, uri: str, *, context: MediaContext = MediaContext()) -> bool:
        return await self._media_row(uri) is not None

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
            raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return metadata

    async def _media_row(self, uri: str) -> object | None:
        from sqlalchemy import select
        digest = parse_media_uri(uri)
        table = self._tables["media"]
        async with self._database.session_factory() as session:
            return (await session.execute(select(table).where(table.c.namespace_key == self._namespace_key, table.c.sha256 == digest))).mappings().first()


def _build_tables() -> tuple[object, dict[str, object]]:
    from sqlalchemy import BigInteger, CHAR, DateTime, Integer, JSON, LargeBinary, String, Table, Text, Column, PrimaryKeyConstraint, UniqueConstraint
    from sqlalchemy.dialects import mysql
    metadata = new_step_metadata()
    key = CHAR(64).with_variant(mysql.CHAR(64, charset="ascii", collation="ascii_bin"), "mysql")
    tables = {
        "runs": Table(storage_name("step_runs"), metadata, Column("namespace_key", key, nullable=False), Column("run_id", String(200), nullable=False), Column("conversation_id", String(200)), Column("parent_run_id", String(200)), Column("agent_name", String(256)), Column("metadata_json", JSON, nullable=False), Column("started_at", DateTime(timezone=True), nullable=False), PrimaryKeyConstraint("namespace_key", "run_id"), UniqueConstraint("namespace_key", "conversation_id", "started_at", "run_id"), mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"),
        "events": Table(storage_name("step_events"), metadata, Column("seq", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True), Column("namespace_key", key, nullable=False), Column("run_id", String(200), nullable=False), Column("kind", String(64), nullable=False), Column("step_index", Integer, nullable=False), Column("timestamp", DateTime(timezone=True), nullable=False), Column("conversation_id", String(200)), Column("parent_run_id", String(200)), Column("agent_name", String(256)), Column("tool_call_id", String(256)), Column("tool_name", String(256)), Column("error", Text), Column("metadata_json", JSON, nullable=False), mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"),
        "snapshots": Table(storage_name("step_snapshots"), metadata, Column("seq", BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True), Column("namespace_key", key, nullable=False), Column("run_id", String(200), nullable=False), Column("step_index", Integer, nullable=False), Column("conversation_id", String(200)), Column("parent_run_id", String(200)), Column("agent_name", String(256)), Column("timestamp", DateTime(timezone=True), nullable=False), Column("messages_json", JSON, nullable=False), mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"),
        "effects": Table(storage_name("step_effects"), metadata, Column("namespace_key", key, nullable=False), Column("run_id", String(200), nullable=False), Column("tool_call_id", String(256), nullable=False), Column("tool_name", String(256), nullable=False), Column("status", String(64), nullable=False), Column("started_at", DateTime(timezone=True), nullable=False), Column("ended_at", DateTime(timezone=True)), Column("idempotency_key", String(256)), Column("effect_summary", Text), PrimaryKeyConstraint("namespace_key", "run_id", "tool_call_id"), mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"),
        "media": Table(storage_name("step_media"), metadata, Column("namespace_key", key, nullable=False), Column("sha256", key, nullable=False), Column("media_type", String(256)), Column("bytes", LargeBinary().with_variant(mysql.LONGBLOB(), "mysql"), nullable=False), Column("size_bytes", BigInteger, nullable=False), Column("metadata_json", JSON, nullable=False), PrimaryKeyConstraint("namespace_key", "sha256"), mysql_engine="InnoDB", mysql_charset="utf8mb4", mysql_collate="utf8mb4_bin"),
    }
    return metadata, tables


def _run_values(namespace_key: str, record: RunRecord) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": record.run_id, "conversation_id": record.conversation_id, "parent_run_id": record.parent_run_id, "agent_name": record.agent_name, "metadata_json": dict(record.metadata), "started_at": _utc(record.started_at)}


def _event_values(namespace_key: str, event: StepEvent) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": event.run_id, "kind": event.kind, "step_index": event.step_index, "timestamp": _utc(event.timestamp), "conversation_id": event.conversation_id, "parent_run_id": event.parent_run_id, "agent_name": event.agent_name, "tool_call_id": event.tool_call_id, "tool_name": event.tool_name, "error": event.error, "metadata_json": dict(event.metadata)}


def _snapshot_values(namespace_key: str, snapshot: ContinuableSnapshot) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": snapshot.run_id, "step_index": snapshot.step_index, "conversation_id": snapshot.conversation_id, "parent_run_id": snapshot.parent_run_id, "agent_name": snapshot.agent_name, "timestamp": _utc(snapshot.timestamp), "messages_json": json.loads(ModelMessagesTypeAdapter.dump_json(snapshot.messages))}


def _effect_values(namespace_key: str, record: ToolEffectRecord) -> dict[str, object]:
    return {"namespace_key": namespace_key, "run_id": record.run_id, "tool_call_id": record.tool_call_id, "tool_name": record.tool_name, "status": record.status, "started_at": _utc(record.started_at), "ended_at": None if record.ended_at is None else _utc(record.ended_at), "idempotency_key": record.idempotency_key, "effect_summary": record.effect_summary}


def _run_from_row(row: Mapping[str, object]) -> RunRecord:
    return RunRecord(run_id=str(row["run_id"]), conversation_id=_optional(row["conversation_id"]), parent_run_id=_optional(row["parent_run_id"]), agent_name=_optional(row["agent_name"]), metadata=_string_map(row["metadata_json"]), started_at=_utc(row["started_at"]))


def _event_from_row(row: Mapping[str, object]) -> StepEvent:
    return StepEvent(run_id=str(row["run_id"]), kind=row["kind"], step_index=int(row["step_index"]), timestamp=_utc(row["timestamp"]), conversation_id=_optional(row["conversation_id"]), parent_run_id=_optional(row["parent_run_id"]), agent_name=_optional(row["agent_name"]), tool_call_id=_optional(row["tool_call_id"]), tool_name=_optional(row["tool_name"]), error=_optional(row["error"]), metadata=_string_map(row["metadata_json"]))


def _snapshot_from_row(row: Mapping[str, object]) -> ContinuableSnapshot:
    return ContinuableSnapshot(run_id=str(row["run_id"]), step_index=int(row["step_index"]), messages=ModelMessagesTypeAdapter.validate_python(row["messages_json"]), conversation_id=_optional(row["conversation_id"]), parent_run_id=_optional(row["parent_run_id"]), agent_name=_optional(row["agent_name"]), timestamp=_utc(row["timestamp"]))


def _effect_from_row(row: Mapping[str, object]) -> ToolEffectRecord:
    return ToolEffectRecord(tool_call_id=str(row["tool_call_id"]), tool_name=str(row["tool_name"]), run_id=str(row["run_id"]), status=row["status"], started_at=_utc(row["started_at"]), ended_at=None if row["ended_at"] is None else _utc(row["ended_at"]), idempotency_key=_optional(row["idempotency_key"]), effect_summary=_optional(row["effect_summary"]))


def _optional(value: object) -> str | None:
    return None if value is None else str(value)


def _string_map(value: object) -> dict[str, str]:
    if not isinstance(value, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in value.items()):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return dict(value)


def _utc(value: object) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value.astimezone(timezone.utc)


def _validate_media(row: Mapping[str, object], data: bytes) -> None:
    if len(data) != int(row["size_bytes"]) or hashlib.sha256(data).hexdigest() != str(row["sha256"]):
        raise LinktoolsAIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _is_integrity(error: BaseException) -> bool:
    try:
        from sqlalchemy.exc import IntegrityError
    except ModuleNotFoundError:
        return False
    return isinstance(error, IntegrityError)


def _schema_digest(metadata: "MetaData") -> str:
    manifest = {
        str(name): [str(column.name) + ":" + str(column.type) for column in table.columns]
        for name, table in sorted(metadata.tables.items())
    }
    return hashlib.sha256(canonical_json_bytes(manifest)).hexdigest()


__all__ = ["SqlMediaStore", "SqlStepStore"]
