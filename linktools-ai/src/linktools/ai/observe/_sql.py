#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SQL MetricStore and canonical metrics metadata."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol, cast

from ..core import Page, validate_persistence_namespace, validate_resource_id
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
    same_definition_contract,
    definition_envelope,
    observation_digest,
    observation_envelope,
    observation_payload_digest,
)
from ._model import MetricDefinition, MetricSourceKind, Observation, _identifier
from ._store import (
    _MetricQueryPushdownPlan,
    _MetricQueryPushdownResult,
    _MetricQueryPushdownRow,
    _parse_scan_cursor,
    _scan_cursor,
    _observation_pruned,
    _SHA256,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from sqlalchemy import MetaData, Select
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
            "namespace", sql_text_key(256), nullable=False,
            comment="Original Metrics namespace retained after pruning.",
        ),
        Column(
            "observation_id", sql_text_key(128), nullable=False,
            comment="Original observation identity retained after pruning.",
        ),
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
            JSON(none_as_null=True),
            nullable=True,
            comment="Versioned canonical immutable ObservationEnvelope payload.",
        ),
        Column(
            "active_occurred_at", timestamp_type, nullable=True,
            comment="Derived occurrence time for active observations only.",
        ),
        Column(
            "pruned_at", timestamp_type, nullable=True,
            comment="Backend time of the first payload pruning transition.",
        ),
        *sql_audit_columns(),
        comment="Immutable metric observations and retained pruning receipts.",
        **sql_table_options(),
    )
    sql_unique(observations, "observation_digest")
    sql_query_index(observations, "namespace_digest", "kind", "active_occurred_at")
    sql_query_index(observations, "namespace_digest", "active_occurred_at")
    sql_audit_indexes(observations)
    return target


class SqlMetricStore:
    def __init__(
        self,
        engine: "AsyncEngine",
        *,
        validate_schema: bool = False,
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

    def _select_observations(self) -> "Select":
        from sqlalchemy import Text, cast as sql_cast, select

        # Raw JSON preserves duplicate members for integrity verification.
        return select(*(
            sql_cast(column, Text).label("payload_json")
            if column.name == "payload_json" else column
            for column in self._observations.c
        ))

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
                await self._context.dialect.begin_consistent_read(session)
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
            return await self._context.run_statement(execute_sql_metric_query(
                session,
                namespace_key=namespace_key,
                dialect_name=dialect_name,
                plan=plan,
                namespace=namespace,
                verify=dialect_name == "sqlite",
            ))

    async def _verify_query_records(
        self,
        session: "AsyncSession",
        namespace: str,
        namespace_key: str,
        plan: _MetricQueryPushdownPlan,
    ) -> int:
        from sqlalchemy import func, select

        start = plan.start.astimezone(timezone.utc)
        end = plan.end.astimezone(timezone.utc)
        if self._context.dialect.name == "mysql":
            start = start.replace(tzinfo=None)
            end = end.replace(tzinfo=None)
        columns = self._observations.c
        bounded = (
            self._select_observations()
            .where(
                columns.namespace_digest == namespace_key,
                columns.kind == plan.observation_kind,
                columns.pruned_at.is_(None),
                columns.active_occurred_at >= start,
                columns.active_occurred_at < end,
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
                    _decode_observation_record(namespace, namespace_key, row)
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
        values = {
            "namespace_digest": namespace_key,
            "metric_name": definition.name,
            "revision": definition.revision,
            "observation_kind": definition.observation_kind,
            "payload_json": definition_envelope(namespace, definition),
        }

        async def write_and_read(session: "AsyncSession") -> Mapping[str, object]:
            result = await self._context.run_statement(self._context.dialect.insert_ignore_conflict(
                session,
                table=self._definitions,
                values=cast("Mapping[str, SqlValue]", values),
                index_elements=("namespace_digest", "metric_name", "revision"),
            ))
            if result.inserted:
                return values
            row = (
                await self._context.run_statement(session.execute(
                    select(self._definitions).where(
                        self._definitions.c.namespace_digest == namespace_key,
                        self._definitions.c.metric_name == definition.name,
                        self._definitions.c.revision == definition.revision,
                    )
                ))
            ).mappings().first()
            if row is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return row

        row = await self._context.run_mutation(
            write_and_read,
            domain="metrics.definition",
        )
        stored = self._decode_definition_row(namespace, namespace_key, row)
        if not same_definition_contract(stored, definition):
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
            row = (await self._context.run_statement(session.execute(statement))).mappings().first()
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
            row = (await self._context.run_statement(session.execute(statement))).mappings().first()
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
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return definition

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
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
                "namespace": namespace,
                "observation_id": observation.observation_id,
                "namespace_digest": namespace_key,
                "observation_digest": identity,
                "payload_digest": payload,
                "kind": observation.kind,
                "occurred_at": observation.occurred_at,
                "active_occurred_at": observation.occurred_at,
                "pruned_at": None,
                "payload_json": observation_envelope(namespace, observation),
            }
            for identity, (payload, observation) in collapsed.items()
        ]
        identities = tuple(collapsed)

        async def write_and_validate(session: "AsyncSession") -> None:
            inserted = await self._context.run_statement(self._context.dialect.insert_ignore_conflict_many(
                session,
                table=self._observations,
                rows=cast("list[Mapping[str, SqlValue]]", rows),
                index_elements=("observation_digest",),
            ))
            if inserted == len(rows):
                return
            result = (
                await self._context.run_statement(session.execute(
                    self._select_observations().where(
                        self._observations.c.observation_digest.in_(identities)
                    )
                ))
            ).mappings().all()
            existing = {str(row["observation_digest"]): row for row in result}
            if len(existing) != len(collapsed):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            for identity, (payload, observation) in collapsed.items():
                row = existing[identity]
                if row["payload_digest"] != payload:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
                _validate_observation_identity(namespace, namespace_key, row)
                if row["observation_id"] != observation.observation_id:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                if (
                    row["kind"] != observation.kind
                    or _utc_database_datetime(cast(datetime, row["occurred_at"]))
                    != observation.occurred_at
                ):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            pruned = False
            for row in existing.values():
                if self._decode_observation_row(namespace, namespace_key, row) is None:
                    pruned = True
            if pruned:
                raise _observation_pruned()

        await self._context.run_mutation(write_and_validate, domain="metrics.observation")

    async def get_observation(
        self,
        namespace: str,
        observation_id: str,
    ) -> Observation | None:
        await self._initialize()
        namespace_key = namespace_digest(namespace)
        identity = observation_digest(namespace, observation_id)
        statement = (
            self._select_observations()
            .where(self._observations.c.observation_digest == identity)
            .limit(1)
        )
        async with self._context.sessions() as session:
            row = (await self._context.run_statement(session.execute(statement))).mappings().first()
        if row is None:
            return None
        if row["observation_id"] != observation_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return self._decode_observation_row(namespace, namespace_key, row)

    def _decode_observation_row(
        self,
        namespace: str,
        namespace_key: str,
        row: "Mapping[str, object]",
    ) -> Observation | None:
        if row["pruned_at"] is not None:
            _validate_observation_identity(namespace, namespace_key, row)
            if (
                not isinstance(row["pruned_at"], datetime)
                or row["payload_json"] is not None
                or row["active_occurred_at"] is not None
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return None
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
        from sqlalchemy import and_, or_

        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        await self._initialize()
        namespace_key = namespace_digest(namespace)
        statement = self._select_observations().where(
            self._observations.c.namespace_digest == namespace_key,
            self._observations.c.kind == kind,
            self._observations.c.pruned_at.is_(None),
            self._observations.c.active_occurred_at >= start,
            self._observations.c.active_occurred_at < end,
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
            rows = (await self._context.run_statement(session.execute(statement))).mappings().all()

        has_more = len(rows) > limit
        selected = rows[:limit]
        values = [
            _decode_observation_record(namespace, namespace_key, row)
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
        from sqlalchemy import and_, or_, update

        await self._initialize()
        namespace_key = namespace_digest(namespace)

        async def prune_rows(session: "AsyncSession") -> int:
            await self._context.dialect.begin_consistent_read(session)
            pruned_at = await self._context.dialect.database_now(session)
            columns = self._observations.c
            after: tuple[datetime, str] | None = None
            count = 0
            while True:
                statement = self._select_observations().where(
                    columns.namespace_digest == namespace_key,
                    columns.pruned_at.is_(None),
                    columns.active_occurred_at < before,
                )
                if after is not None:
                    statement = statement.where(or_(
                        columns.active_occurred_at > after[0],
                        and_(columns.active_occurred_at == after[0],
                             columns.observation_digest > after[1]),
                    ))
                statement = statement.order_by(
                    columns.active_occurred_at, columns.observation_digest,
                ).limit(256)
                rows = (await self._context.run_statement(
                    session.execute(statement)
                )).mappings().all()
                if not rows:
                    return count
                for row in rows:
                    _decode_observation_record(namespace, namespace_key, row)
                identities = tuple(row["observation_digest"] for row in rows)
                result = await self._context.run_statement(session.execute(
                    update(self._observations).where(
                        columns.namespace_digest == namespace_key,
                        columns.observation_digest.in_(identities),
                        columns.pruned_at.is_(None),
                    ).values(
                        pruned_at=pruned_at, payload_json=None,
                        active_occurred_at=None,
                    )
                ))
                count += int(result.rowcount or 0)
                last = rows[-1]
                after = (last["active_occurred_at"], last["observation_digest"])

        return await self._context.run_mutation(prune_rows, domain="metrics.prune")



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
        connection.create_function("linktools_metric_verify", 12, _sqlite_record_error)
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
    original_namespace: str,
    observation_id: str,
    active_occurred_at: str | None,
    pruned_at: str | None,
) -> str | None:
    try:
        timestamp = datetime.fromisoformat(occurred_at)
        _decode_observation_record(
            namespace,
            namespace_key,
            {
                "namespace": original_namespace,
                "observation_id": observation_id,
                "namespace_digest": stored_namespace,
                "observation_digest": stored_identity,
                "payload_digest": stored_payload,
                "kind": kind,
                "occurred_at": timestamp,
                "payload_json": payload,
                "active_occurred_at": (
                    None if active_occurred_at is None else datetime.fromisoformat(active_occurred_at)
                ),
                "pruned_at": None if pruned_at is None else datetime.fromisoformat(pruned_at),
            },
        )
    except AIError as error:
        # Return typed failures to SQL; SQLite otherwise erases their code.
        return error.code.value
    except (TypeError, ValueError, RecursionError):
        return ErrorCode.STORAGE_INTEGRITY_ERROR.value
    return None


def _validate_observation_identity(
    namespace: str,
    namespace_key: str,
    row: "Mapping[str, object]",
) -> None:
    try:
        validate_persistence_namespace(cast(str, row["namespace"]))
        validate_resource_id(cast(str, row["observation_id"]))
        _identifier(row["kind"], name="observation kind")
    except (AIError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if (
        row["namespace"] != namespace
        or row["namespace_digest"] != namespace_key
        or row["observation_digest"] != observation_digest(namespace, cast(str, row["observation_id"]))
        or not isinstance(row["payload_digest"], str)
        or _SHA256.fullmatch(row["payload_digest"]) is None
        or not isinstance(row["occurred_at"], datetime)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _decode_observation_record(
    namespace: str,
    namespace_key: str,
    row: "Mapping[str, object]",
) -> Observation:
    _validate_observation_identity(namespace, namespace_key, row)
    payload = row["payload_json"]
    if isinstance(payload, str):
        try:
            payload = json.loads(
                payload, object_pairs_hook=_unique_json_pairs,
                parse_constant=_reject_json_constant,
            )
        except (TypeError, ValueError, RecursionError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    observation = decode_observation_envelope(
        payload, expected_namespace=namespace
    )
    expected_identity = observation_digest(namespace, observation.observation_id)
    expected_payload = observation_payload_digest(namespace, observation)
    if (
        row["namespace_digest"] != namespace_key
        or row["observation_id"] != observation.observation_id
        or row["pruned_at"] is not None
        or not isinstance(row["active_occurred_at"], datetime)
        or _utc_database_datetime(row["active_occurred_at"]) != observation.occurred_at
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
