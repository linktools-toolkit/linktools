#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL MetricStore and canonical metrics metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, cast

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
from ._model import MetricDefinition, MetricSourceKind, Observation
from ._store import (
    _MetricQueryPushdownPlan,
    _MetricQueryPushdownResult,
    _MetricQueryPushdownRow,
    _parse_scan_cursor,
    _scan_cursor,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sqlalchemy import MetaData
    from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession
    from sqlalchemy.pool import ConnectionPoolEntry, PoolProxiedConnection

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
    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        validate_schema: bool = True,
    ) -> None:
        if not isinstance(validate_schema, bool):
            raise TypeError("validate_schema must be bool")
        self._metadata = build_metrics_sql_metadata()
        self._definitions = self._metadata.tables["ai_metric_definitions"]
        self._observations = self._metadata.tables["ai_metric_observations"]
        self._context = create_sql_storage_context(engine)
        self._validate_schema = validate_schema
        if engine.dialect.name == "sqlite":
            from sqlalchemy import event

            if not event.contains(engine.sync_engine, "checkout", _configure_sqlite_verifier):
                event.listen(engine.sync_engine, "checkout", _configure_sqlite_verifier)

    async def _initialize(self) -> None:
        await self._context.initialize(
            metadata=self._metadata if self._validate_schema else None
        )

    async def _execute_metric_query(
        self,
        namespace: str,
        plan: _MetricQueryPushdownPlan,
    ) -> _MetricQueryPushdownResult | None:
        from ._sql_query import (
            _sql_features_available,
            execute_sql_metric_query,
            sql_query_pushdown_supported,
        )

        await self._initialize()
        dialect_name = self._context.dialect.name
        namespace_key = namespace_digest(namespace)
        async with self._context.sessions() as session:
            if dialect_name in {"mysql", "postgresql"}:
                if not sql_query_pushdown_supported(plan):
                    return None
                # Verification and reduction must see the same immutable records,
                # including when the borrowed engine normally uses READ COMMITTED.
                await session.connection(
                    execution_options={"isolation_level": "REPEATABLE READ"}
                )
                if not await _sql_features_available(session, dialect_name, plan):
                    return None
                scanned_count = await self._verify_query_records(
                    session, namespace, namespace_key, plan
                )
                if (
                    plan.source_kind is MetricSourceKind.OBSERVATION_COUNT
                    and not plan.filters
                    and not plan.correlation_filters
                    and not plan.group_by
                    and plan.bucket_microseconds is None
                ):
                    # The fully verified stream already carries this SQL count.
                    return _MetricQueryPushdownResult(rows=(
                        _MetricQueryPushdownRow(
                            group=(), bucket_index=None,
                            sample_count=scanned_count, sample_sum=scanned_count,
                        ),
                    ))
            return await execute_sql_metric_query(
                session,
                namespace_key=namespace_key,
                dialect_name=dialect_name,
                plan=plan,
                namespace=namespace,
                verify=dialect_name == "sqlite",
            )

    async def _verify_query_records(
        self,
        session: "AsyncSession",
        namespace: str,
        namespace_key: str,
        plan: _MetricQueryPushdownPlan,
    ) -> int:
        from sqlalchemy import Text, cast as sql_cast, func, select

        start = plan.start.astimezone(timezone.utc)
        end = plan.end.astimezone(timezone.utc)
        if self._context.dialect.name == "mysql":
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
        columns = self._observations.c
        bounded = (
            select(
                columns.namespace_digest,
                columns.observation_digest,
                columns.payload_digest,
                columns.kind,
                columns.occurred_at,
                sql_cast(columns.payload_json, Text).label("payload_json"),
            )
            .where(
                columns.namespace_digest == namespace_key,
                columns.kind == plan.observation_kind,
                columns.occurred_at >= start,
                columns.occurred_at < end,
            )
            .limit(plan.max_scanned_observations + 1)
            .subquery("metric_verification")
        )
        statement = select(bounded, func.count().over().label("scanned_count"))
        result = await session.stream(statement, execution_options={"yield_per": 512})
        scanned_count = 0
        try:
            async for row in result.mappings():
                scanned_count = int(row["scanned_count"])
                if scanned_count > plan.max_scanned_observations:
                    raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
                try:
                    # Read JSON text so driver decoding cannot erase duplicate
                    # members before the codec and SQL extract different values.
                    payload = json.loads(
                        row["payload_json"],
                        object_pairs_hook=_unique_json_pairs,
                        parse_constant=_reject_json_constant,
                    )
                    _decode_observation_record(
                        namespace, namespace_key, {**row, "payload_json": payload}
                    )
                except (TypeError, ValueError, RecursionError) as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        finally:
            await result.close()
        return scanned_count

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

        await self._context.run_mutation(write_and_validate, domain="metrics.observation")

    async def get_observation(
        self,
        namespace: str,
        observation_id: str,
    ) -> Observation | None:
        from sqlalchemy import select

        await self._initialize()
        namespace_key = namespace_digest(namespace)
        identity = observation_digest(namespace, observation_id)
        statement = (
            select(self._observations)
            .where(self._observations.c.observation_digest == identity)
            .limit(1)
        )
        async with self._context.sessions() as session:
            row = (await session.execute(statement)).mappings().first()
        if row is None:
            return None
        observation = self._decode_observation_row(namespace, namespace_key, row)
        if observation.observation_id != observation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return observation

    def _decode_observation_row(
        self,
        namespace: str,
        namespace_key: str,
        row: "Mapping[str, object]",
    ) -> Observation:
        return _decode_observation_record(namespace, namespace_key, row)


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
        values = [
            self._decode_observation_row(namespace, namespace_key, row)
            for row in selected
        ]

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



class _SqliteFunctionConnection(Protocol):
    def create_function(
        self,
        name: str,
        arguments: int,
        function: "Callable[..., str | None]",
    ) -> None: ...


def _configure_sqlite_verifier(
    connection: _SqliteFunctionConnection,
    record: "ConnectionPoolEntry",
    _proxy: "PoolProxiedConnection",
) -> None:
    # Pool entry info is cleared when its physical connection is replaced.
    if record.info.get("linktools_metric_verifier") is not _sqlite_record_error:
        connection.create_function("linktools_metric_verify", 8, _sqlite_record_error)
        record.info["linktools_metric_verifier"] = _sqlite_record_error


def _unique_json_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = dict(pairs)
    # SQLite JSON extraction and Python disagree on duplicate object keys.
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON object key")
    return result


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-JSON numeric constant: {value}")


def _sqlite_record_error(
    namespace: str,
    namespace_key: str,
    stored_namespace: str,
    stored_identity: str,
    stored_payload: str,
    kind: str,
    occurred_at: str,
    payload: str,
) -> str | None:
    try:
        decoded = json.loads(
            payload,
            object_pairs_hook=_unique_json_pairs,
            parse_constant=_reject_json_constant,
        )
        timestamp = datetime.fromisoformat(occurred_at)
        _decode_observation_record(
            namespace,
            namespace_key,
            {
                "namespace_digest": stored_namespace,
                "observation_digest": stored_identity,
                "payload_digest": stored_payload,
                "kind": kind,
                "occurred_at": timestamp,
                "payload_json": decoded,
            },
        )
    except AIError as error:
        # Return typed failures to SQL; SQLite otherwise erases their code.
        return error.code.value
    except (TypeError, ValueError, RecursionError):
        return ErrorCode.STORAGE_INTEGRITY_ERROR.value
    return None


def _decode_observation_record(
    namespace: str,
    namespace_key: str,
    row: "Mapping[str, object]",
) -> Observation:
    observation = decode_observation_envelope(
        row["payload_json"], expected_namespace=namespace
    )
    expected_identity = observation_digest(namespace, observation.observation_id)
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
    return observation


def _utc_database_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


__all__ = [
    "SqlMetricStore",
    "build_metrics_sql_metadata",
]
