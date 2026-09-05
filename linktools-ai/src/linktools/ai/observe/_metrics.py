#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Close-free Metrics facade and package-owned built-in definitions."""

from __future__ import annotations

import uuid
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import canonical_json_bytes, validate_persistence_namespace
from ..errors import AIError, ErrorCode
from ._codec import observation_envelope
from ._memory import InMemoryMetricStore
from ._model import (
    MetricAggregation,
    MetricDefinition,
    MetricMeasurement,
    MetricQuery,
    MetricQueryResult,
    MetricSource,
    MetricSourceKind,
    MetricType,
    Observation,
    validate_metric_value,
)
from ._query import execute_query
from ._sqlite import SQLiteMetricStore
from ._store import MetricStore

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

_CANONICAL_QUERY_FIELDS = frozenset(
    {"source_namespace", "tenant_id", "status", "error_code"}
)
_RESERVED_PREFIX = "linktools."
_MAX_RECORD_BATCH = 256
_MAX_RECORD_BATCH_CANONICAL_BYTES = 16 * 1024 * 1024


class Metrics:
    """Metrics dataset facade without a user-visible lifecycle."""

    def __init__(self, store: MetricStore, *, namespace: str = "default") -> None:
        self._store = store
        self._namespace = validate_persistence_namespace(namespace)

    @property
    def namespace(self) -> str:
        return self._namespace

    @classmethod
    def in_memory(cls, *, namespace: str = "default") -> Metrics:
        return cls(InMemoryMetricStore(), namespace=namespace)

    @classmethod
    def sqlite(cls, path: str | Path, *, namespace: str = "default") -> Metrics:
        return cls(SQLiteMetricStore(path), namespace=namespace)

    @classmethod
    def sql(cls, engine: AsyncEngine, *, namespace: str = "default") -> Metrics:
        from ._sql import SqlMetricStore

        return cls(SqlMetricStore(engine), namespace=namespace)

    @classmethod
    def from_store(cls, store: MetricStore, *, namespace: str = "default") -> Metrics:
        return cls(store, namespace=namespace)

    async def define(self, definition: MetricDefinition) -> MetricDefinition:
        if not isinstance(definition, MetricDefinition):
            raise TypeError("definition must be MetricDefinition")
        if definition.name.startswith(_RESERVED_PREFIX):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "metric"},
            )
        return await self._store.put_definition(self._namespace, definition)

    async def record(
        self,
        metric: str,
        value: int | float,
        *,
        revision: int | None = None,
        observation_id: str | None = None,
        occurred_at: datetime | None = None,
        source_namespace: str | None = None,
        tenant_id: str | None = None,
        status: str | None = None,
        error_code: str | None = None,
        dimensions: Mapping[str, str] | None = None,
        correlation: Mapping[str, str | int] | None = None,
    ) -> str:
        if not isinstance(metric, str) or metric.startswith(_RESERVED_PREFIX):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "metric"},
            )
        definition = await self._resolve_definition(metric, revision)
        source = definition.source
        if source.kind is not MetricSourceKind.MEASUREMENT:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "metric"},
            )
        if definition.observation_kind.startswith(_RESERVED_PREFIX):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "observation_kind"},
            )
        selected_dimensions = dict(dimensions or {})
        allowed_dimensions = set(definition.query_fields) - _CANONICAL_QUERY_FIELDS
        if not set(selected_dimensions).issubset(allowed_dimensions):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "dimensions"},
            )
        if source.measurement_name is None or source.measurement_revision is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        sample = validate_metric_value(definition.metric_type, value)
        resolved_id = observation_id or uuid.uuid4().hex
        observation = Observation(
            version=1,
            observation_id=resolved_id,
            kind=definition.observation_kind,
            occurred_at=occurred_at or datetime.now(timezone.utc),
            source_namespace=source_namespace,
            tenant_id=tenant_id,
            status=status,
            error_code=error_code,
            correlation=dict(correlation or {}),
            dimensions=selected_dimensions,
            measurements=(
                MetricMeasurement(
                    source.measurement_name,
                    source.measurement_revision,
                    sample,
                ),
            ),
        )
        await self.record_observations((observation,))
        return resolved_id

    async def record_observations(
        self,
        observations: Sequence[Observation],
    ) -> None:
        if not isinstance(observations, Sequence) or isinstance(
            observations, (str, bytes, bytearray)
        ):
            raise TypeError("observations must be a sequence")
        batch = tuple(observations)
        if len(batch) > _MAX_RECORD_BATCH:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        total_bytes = 0
        for observation in batch:
            if not isinstance(observation, Observation):
                raise TypeError("observations must contain Observation values")
            total_bytes += len(
                canonical_json_bytes(observation_envelope(self._namespace, observation))
            )
            if total_bytes > _MAX_RECORD_BATCH_CANONICAL_BYTES:
                raise AIError(ErrorCode.OBSERVATION_PAYLOAD_TOO_LARGE)
        try:
            await self._store.put_observations(self._namespace, batch)
        except AIError as error:
            if error.code is not ErrorCode.STORAGE_COMMIT_UNKNOWN:
                raise
            await self._store.put_observations(self._namespace, batch)

    async def query(self, query: MetricQuery) -> MetricQueryResult:
        if not isinstance(query, MetricQuery):
            raise TypeError("query must be MetricQuery")
        definition = await self._resolve_definition(query.metric, query.revision)
        return await execute_query(self._store, self._namespace, definition, query)

    async def prune(self, *, before: datetime) -> int:
        if not isinstance(before, datetime) or before.tzinfo is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "before"},
            )
        return await self._store.prune_observations(
            self._namespace,
            before=before.astimezone(timezone.utc),
        )

    async def _resolve_definition(
        self,
        name: str,
        revision: int | None,
    ) -> MetricDefinition:
        built_in = _BUILTINS.get(name)
        if built_in is not None:
            if revision is None or revision == built_in.revision:
                return built_in
            raise AIError(ErrorCode.METRIC_NOT_FOUND, safe_details={"metric": name})
        definition = (
            await self._store.latest_definition(self._namespace, name)
            if revision is None
            else await self._store.get_definition(self._namespace, name, revision)
        )
        if definition is None:
            raise AIError(ErrorCode.METRIC_NOT_FOUND, safe_details={"metric": name})
        return definition


def _count(name: str, kind: str, fields: tuple[str, ...]) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=kind,
        source=MetricSource.observation_count(),
        metric_type=MetricType.COUNTER,
        unit="1",
        default_aggregation=MetricAggregation.SUM,
        query_fields=fields,
    )


def _latency(name: str, kind: str, fields: tuple[str, ...]) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=kind,
        source=MetricSource.measurement("latency_ns"),
        metric_type=MetricType.DISTRIBUTION,
        unit="ns",
        default_aggregation=MetricAggregation.MEAN,
        query_fields=fields,
    )


def _token(
    name: str,
    measurement: str,
    kind: str,
    fields: tuple[str, ...],
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=kind,
        source=MetricSource.measurement(measurement),
        metric_type=MetricType.COUNTER,
        unit="token",
        default_aggregation=MetricAggregation.SUM,
        query_fields=fields,
    )


def _ratio(
    name: str,
    kind: str,
    field: str,
    values: tuple[str, ...],
    fields: tuple[str, ...],
) -> MetricDefinition:
    return MetricDefinition(
        name=name,
        revision=1,
        observation_kind=kind,
        source=MetricSource.indicator(field, values),
        metric_type=MetricType.RATIO,
        unit="1",
        default_aggregation=MetricAggregation.MEAN,
        query_fields=fields,
    )


_CANONICAL = ("source_namespace", "tenant_id", "status", "error_code")
_MODEL_FIELDS = (*_CANONICAL, "agent_id", "provider", "model_identity", "route_id")
_TOOL_FIELDS = (*_CANONICAL, "agent_id", "tool_name")
_AGENT_FIELDS = (*_CANONICAL, "agent_id")
_EXECUTION_FIELDS = (*_CANONICAL, "agent_id", "lineage_kind")
_TASK_ATTEMPT_FIELDS = _CANONICAL
_TASK_GRAPH_FIELDS = _CANONICAL
_STORAGE_FIELDS = (*_CANONICAL, "domain", "target")

_BUILTIN_DEFINITIONS = (
    _count("linktools.model.request.count", "linktools.model.request", _MODEL_FIELDS),
    _latency("linktools.model.request.latency", "linktools.model.request", _MODEL_FIELDS),
    _ratio(
        "linktools.model.request.failure_ratio",
        "linktools.model.request",
        "status",
        ("FAILED",),
        _MODEL_FIELDS,
    ),
    _ratio(
        "linktools.model.request.timeout_ratio",
        "linktools.model.request",
        "error_code",
        ("MODEL_TIMEOUT",),
        _MODEL_FIELDS,
    ),
    _ratio(
        "linktools.model.request.rate_limit_ratio",
        "linktools.model.request",
        "error_code",
        ("MODEL_RATE_LIMITED",),
        _MODEL_FIELDS,
    ),
    _token(
        "linktools.model.input_tokens",
        "input_tokens",
        "linktools.model.request",
        _MODEL_FIELDS,
    ),
    _token(
        "linktools.model.output_tokens",
        "output_tokens",
        "linktools.model.request",
        _MODEL_FIELDS,
    ),
    _token(
        "linktools.model.cache_read_tokens",
        "cache_read_tokens",
        "linktools.model.request",
        _MODEL_FIELDS,
    ),
    _token(
        "linktools.model.cache_write_tokens",
        "cache_write_tokens",
        "linktools.model.request",
        _MODEL_FIELDS,
    ),
    _count(
        "linktools.tool.execution.count",
        "linktools.tool.execution",
        _TOOL_FIELDS,
    ),
    _latency(
        "linktools.tool.execution.latency",
        "linktools.tool.execution",
        _TOOL_FIELDS,
    ),
    _ratio(
        "linktools.tool.execution.failure_ratio",
        "linktools.tool.execution",
        "status",
        ("FAILED",),
        _TOOL_FIELDS,
    ),
    _count("linktools.agent.run.count", "linktools.agent.run", _AGENT_FIELDS),
    _latency("linktools.agent.run.latency", "linktools.agent.run", _AGENT_FIELDS),
    _ratio(
        "linktools.agent.run.failure_ratio",
        "linktools.agent.run",
        "status",
        ("FAILED",),
        _AGENT_FIELDS,
    ),
    _count(
        "linktools.execution.count",
        "linktools.execution.terminal",
        _EXECUTION_FIELDS,
    ),
    _ratio(
        "linktools.execution.failure_ratio",
        "linktools.execution.terminal",
        "status",
        ("FAILED",),
        _EXECUTION_FIELDS,
    ),
    _ratio(
        "linktools.execution.cancel_ratio",
        "linktools.execution.terminal",
        "status",
        ("CANCELLED",),
        _EXECUTION_FIELDS,
    ),
    _token(
        "linktools.execution.input_tokens",
        "input_tokens",
        "linktools.execution.terminal",
        _EXECUTION_FIELDS,
    ),
    _token(
        "linktools.execution.output_tokens",
        "output_tokens",
        "linktools.execution.terminal",
        _EXECUTION_FIELDS,
    ),
    _count(
        "linktools.task.node.attempt.count",
        "linktools.task.node.attempt",
        _TASK_ATTEMPT_FIELDS,
    ),
    _latency(
        "linktools.task.node.attempt.latency",
        "linktools.task.node.attempt",
        _TASK_ATTEMPT_FIELDS,
    ),
    _ratio(
        "linktools.task.node.attempt.failure_ratio",
        "linktools.task.node.attempt",
        "status",
        ("FAILED", "BLOCKED"),
        _TASK_ATTEMPT_FIELDS,
    ),
    _count(
        "linktools.task.graph.count",
        "linktools.task.graph.terminal",
        _TASK_GRAPH_FIELDS,
    ),
    _latency(
        "linktools.task.graph.latency",
        "linktools.task.graph.terminal",
        _TASK_GRAPH_FIELDS,
    ),
    _ratio(
        "linktools.task.graph.failure_ratio",
        "linktools.task.graph.terminal",
        "status",
        ("FAILED", "BLOCKED"),
        _TASK_GRAPH_FIELDS,
    ),
    _ratio(
        "linktools.task.graph.cancel_ratio",
        "linktools.task.graph.terminal",
        "status",
        ("CANCELLED",),
        _TASK_GRAPH_FIELDS,
    ),
    _count(
        "linktools.storage.operation.count",
        "linktools.storage.operation",
        _STORAGE_FIELDS,
    ),
    _latency(
        "linktools.storage.operation.latency",
        "linktools.storage.operation",
        _STORAGE_FIELDS,
    ),
    _ratio(
        "linktools.storage.operation.failure_ratio",
        "linktools.storage.operation",
        "status",
        ("FAILED",),
        _STORAGE_FIELDS,
    ),
)
_BUILTINS = {definition.name: definition for definition in _BUILTIN_DEFINITIONS}


__all__ = ["Metrics"]
