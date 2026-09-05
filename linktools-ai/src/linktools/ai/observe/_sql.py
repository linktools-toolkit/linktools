#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL MetricStore and canonical metrics metadata."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast

from ..core import Page
from ..errors import AIError, ErrorCode
from ..storage import (
    create_sql_storage_context,
    namespace_digest,
    sql_audit_columns,
    sql_audit_indexes,
    sql_id_column,
    sql_query_index,
    sql_sha256,
    sql_table_options,
    sql_text_key,
    sql_unique,
)
from ._codec import (
    decode_definition_envelope,
    decode_observation_envelope,
    definition_envelope,
    definition_semantic_digest,
    observation_digest,
    observation_envelope,
    observation_payload_digest,
)
from ._model import MetricDefinition, Observation
from ._store import _parse_scan_cursor, _scan_cursor

if TYPE_CHECKING:
    from collections.abc import Mapping

    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

    from ..storage import SqlValue


def build_metrics_sql_metadata(*, metadata: "MetaData | None" = None) -> "MetaData":
    from sqlalchemy import BigInteger, Column, DateTime, JSON, MetaData, Table
    from sqlalchemy.dialects import mysql

    target = metadata if metadata is not None else MetaData()
    timestamp_type = DateTime(timezone=True).with_variant(
        mysql.DATETIME(fsp=6), "mysql"
    )
    definitions = Table(
        "ai_metric_definitions",
        target,
        sql_id_column(),
        Column(
            "namespace_digest",
            sql_sha256(),
            nullable=False,
            comment="SHA-256 partition identity of the Metrics namespace.",
        ),
        Column(
            "metric_name",
            sql_text_key(128),
            nullable=False,
            comment="Versioned metric definition name.",
        ),
        Column(
            "revision",
            BigInteger,
            nullable=False,
            comment="Metric semantic revision.",
        ),
        Column(
            "definition_digest",
            sql_sha256(),
            nullable=False,
            comment="SHA-256 of the normalized metric definition.",
        ),
        Column(
            "observation_kind",
            sql_text_key(128),
            nullable=False,
            comment="Canonical observation kind consumed by the metric.",
        ),
        Column(
            "payload_json",
            JSON,
            nullable=False,
            comment="Versioned canonical MetricDefinitionEnvelope payload.",
        ),
        *sql_audit_columns(),
        comment="Custom metric definitions.",
        **sql_table_options(),
    )
    sql_unique(definitions, "namespace_digest", "metric_name", "revision")
    sql_audit_indexes(definitions)

    observations = Table(
        "ai_metric_observations",
        target,
        sql_id_column(),
        Column(
            "namespace_digest",
            sql_sha256(),
            nullable=False,
            comment="SHA-256 partition identity of the Metrics namespace.",
        ),
        Column(
            "observation_digest",
            sql_sha256(),
            nullable=False,
            comment="SHA-256 identity of namespace plus observation_id.",
        ),
        Column(
            "payload_digest",
            sql_sha256(),
            nullable=False,
            comment="SHA-256 of the canonical ObservationEnvelope payload.",
        ),
        Column(
            "kind",
            sql_text_key(128),
            nullable=False,
            comment="Canonical observation kind query projection.",
        ),
        Column(
            "occurred_at",
            timestamp_type,
            nullable=False,
            comment="Canonical UTC observation occurrence time.",
        ),
        Column(
            "payload_json",
            JSON,
            nullable=False,
            comment="Versioned canonical immutable ObservationEnvelope payload.",
        ),
        *sql_audit_columns(),
        comment="Immutable metric observations.",
        **sql_table_options(),
    )
    sql_unique(observations, "observation_digest")
    sql_query_index(observations, "namespace_digest", "kind", "occurred_at")
    sql_query_index(observations, "namespace_digest", "occurred_at")
    sql_audit_indexes(observations)
    return target


class SqlMetricStore:
    def __init__(self, engine: "AsyncEngine") -> None:
        self._metadata = build_metrics_sql_metadata()
        self._definitions = self._metadata.tables["ai_metric_definitions"]
        self._observations = self._metadata.tables["ai_metric_observations"]
        self._context = create_sql_storage_context(engine)

    async def _initialize(self) -> None:
        await self._context.initialize(metadata=self._metadata)

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> MetricDefinition:
        from sqlalchemy import select

        await self._initialize()
        namespace_key = namespace_digest(namespace)
        semantic_digest = definition_semantic_digest(definition)
        values = {
            "namespace_digest": namespace_key,
            "metric_name": definition.name,
            "revision": definition.revision,
            "definition_digest": semantic_digest,
            "observation_kind": definition.observation_kind,
            "payload_json": definition_envelope(namespace, definition),
        }

        async def write_and_read(session: "AsyncSession") -> Mapping[str, object]:
            await self._context.dialect.insert_ignore_conflict(
                session,
                table=self._definitions,
                values=cast("Mapping[str, SqlValue]", values),
                index_elements=("namespace_digest", "metric_name", "revision"),
            )
            row = (
                await session.execute(
                    select(self._definitions).where(
                        self._definitions.c.namespace_digest == namespace_key,
                        self._definitions.c.metric_name == definition.name,
                        self._definitions.c.revision == definition.revision,
                    )
                )
            ).mappings().first()
            if row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return row

        row = await self._context.run_mutation(
            write_and_read,
            domain="metrics.definition",
        )
        stored = self._decode_definition_row(namespace, namespace_key, row)
        if str(row["definition_digest"]) != semantic_digest:
            raise AIError(ErrorCode.STORAGE_CONFLICT)
        return stored

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> MetricDefinition | None:
        from sqlalchemy import select

        await self._initialize()
        namespace_key = namespace_digest(namespace)
        statement = (
            select(self._definitions)
            .where(
                self._definitions.c.namespace_digest == namespace_key,
                self._definitions.c.metric_name == name,
                self._definitions.c.revision == revision,
            )
            .limit(1)
        )
        async with self._context.sessions() as session:
            row = (await session.execute(statement)).mappings().first()
        if row is None:
            return None
        return self._decode_definition_row(namespace, namespace_key, row)

    async def latest_definition(
        self,
        namespace: str,
        name: str,
    ) -> MetricDefinition | None:
        from sqlalchemy import select

        await self._initialize()
        namespace_key = namespace_digest(namespace)
        statement = (
            select(self._definitions)
            .where(
                self._definitions.c.namespace_digest == namespace_key,
                self._definitions.c.metric_name == name,
            )
            .order_by(self._definitions.c.revision.desc())
            .limit(1)
        )
        async with self._context.sessions() as session:
            row = (await session.execute(statement)).mappings().first()
        if row is None:
            return None
        return self._decode_definition_row(namespace, namespace_key, row)

    def _decode_definition_row(
        self,
        namespace: str,
        namespace_key: str,
        row: "Mapping[str, object]",
    ) -> MetricDefinition:
        definition = decode_definition_envelope(
            row["payload_json"], expected_namespace=namespace
        )
        if (
            row["namespace_digest"] != namespace_key
            or definition.name != row["metric_name"]
            or definition.revision != row["revision"]
            or definition.observation_kind != row["observation_kind"]
            or definition_semantic_digest(definition) != row["definition_digest"]
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return definition

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        from sqlalchemy import select

        await self._initialize()
        collapsed: dict[str, tuple[str, Observation]] = {}
        for observation in observations:
            identity = observation_digest(namespace, observation.observation_id)
            payload = observation_payload_digest(namespace, observation)
            current = collapsed.get(identity)
            if current is not None and current[0] != payload:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            collapsed[identity] = (payload, observation)
        if not collapsed:
            return
        namespace_key = namespace_digest(namespace)
        rows = [
            {
                "namespace_digest": namespace_key,
                "observation_digest": identity,
                "payload_digest": payload,
                "kind": observation.kind,
                "occurred_at": observation.occurred_at,
                "payload_json": observation_envelope(namespace, observation),
            }
            for identity, (payload, observation) in collapsed.items()
        ]
        identities = tuple(collapsed)

        async def write_and_validate(session: "AsyncSession") -> None:
            await self._context.dialect.insert_ignore_conflict_many(
                session,
                table=self._observations,
                rows=cast("list[Mapping[str, SqlValue]]", rows),
                index_elements=("observation_digest",),
            )
            result = (
                await session.execute(
                    select(
                        self._observations.c.observation_digest,
                        self._observations.c.payload_digest,
                    ).where(self._observations.c.observation_digest.in_(identities))
                )
            ).all()
            existing = {str(row[0]): str(row[1]) for row in result}
            if len(existing) != len(collapsed):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for identity, (payload, _) in collapsed.items():
                if existing.get(identity) != payload:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)

        await self._context.run_mutation(
            write_and_validate,
            domain="metrics.observation",
        )

    async def scan_observations(
        self,
        namespace: str,
        kind: str,
        start: datetime,
        end: datetime,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[Observation]:
        from sqlalchemy import and_, or_, select

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        await self._initialize()
        namespace_key = namespace_digest(namespace)
        statement = select(self._observations).where(
            self._observations.c.namespace_digest == namespace_key,
            self._observations.c.kind == kind,
            self._observations.c.occurred_at >= start,
            self._observations.c.occurred_at < end,
        )
        if cursor is not None:
            after_time, after_digest = _parse_scan_cursor(cursor)
            statement = statement.where(
                or_(
                    self._observations.c.occurred_at > after_time,
                    and_(
                        self._observations.c.occurred_at == after_time,
                        self._observations.c.observation_digest > after_digest,
                    ),
                )
            )
        statement = statement.order_by(
            self._observations.c.occurred_at,
            self._observations.c.observation_digest,
        ).limit(limit + 1)
        async with self._context.sessions() as session:
            rows = (await session.execute(statement)).mappings().all()

        has_more = len(rows) > limit
        selected = rows[:limit]
        values: list[Observation] = []
        for row in selected:
            observation = decode_observation_envelope(
                row["payload_json"], expected_namespace=namespace
            )
            expected_identity = observation_digest(
                namespace, observation.observation_id
            )
            expected_payload = observation_payload_digest(namespace, observation)
            if (
                row["namespace_digest"] != namespace_key
                or row["kind"] != observation.kind
                or _utc_database_datetime(cast("datetime", row["occurred_at"]))
                != observation.occurred_at
                or row["observation_digest"] != expected_identity
                or row["payload_digest"] != expected_payload
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            values.append(observation)

        next_cursor = None
        if has_more and values:
            last = values[-1]
            next_cursor = _scan_cursor(
                last.occurred_at,
                observation_digest(namespace, last.observation_id),
            )
        return Page(tuple(values), next_cursor)

    async def prune_observations(self, namespace: str, *, before: datetime) -> int:
        from sqlalchemy import delete

        await self._initialize()
        namespace_key = namespace_digest(namespace)

        async def delete_rows(session: "AsyncSession") -> int:
            result = await session.execute(
                delete(self._observations).where(
                    self._observations.c.namespace_digest == namespace_key,
                    self._observations.c.occurred_at < before,
                )
            )
            return int(result.rowcount or 0)

        return await self._context.run_mutation(delete_rows, domain="metrics.prune")


def _utc_database_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "SqlMetricStore",
    "build_metrics_sql_metadata",
]
