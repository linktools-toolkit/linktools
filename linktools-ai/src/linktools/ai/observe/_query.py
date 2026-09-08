#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral metric query engine."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime

from ..core import canonical_json_bytes
from ..errors import AIError, ErrorCode
from ._codec import observation_digest, observation_envelope
from ._model import (
    MetricAggregation,
    MetricDefinition,
    MetricPoint,
    MetricQuery,
    MetricQueryResult,
    MetricSourceKind,
    Observation,
    validate_metric_value,
)
from ._store import (
    MetricStore,
    _MetricQueryPushdownPlan,
    _MetricQueryPushdownResult,
    _MetricQueryPushdownRow,
    _MetricQueryPushdownStore,
)

_SCAN_PAGE_SIZE = 512
_MAX_SCANNED_OBSERVATIONS = 100_000
_MAX_SCANNED_CANONICAL_BYTES = 64 * 1024 * 1024
_MAX_EXTRACTED_SAMPLES = 100_000
_MAX_GROUPS = 256
_MAX_BUCKETS = 2_048
_MAX_RESULT_POINTS = 16_384
_RUNTIME_CONTEXT_FIELD_PREFIX = "context."


def _facet(observation: Observation, field: str) -> str | None:
    if field == "source_namespace":
        return observation.source_namespace
    if field == "tenant_id":
        return observation.tenant_id
    if field == "status":
        return observation.status
    if field == "error_code":
        return observation.error_code
    return observation.dimensions.get(field)


def _query_field_allowed(definition: MetricDefinition, field: str) -> bool:
    return field in definition.query_fields or (
        definition.name.startswith("linktools.")
        and field.startswith(_RUNTIME_CONTEXT_FIELD_PREFIX)
        and len(field) > len(_RUNTIME_CONTEXT_FIELD_PREFIX)
    )


def _group_sort_key(group: tuple[str | None, ...]) -> tuple[tuple[int, str], ...]:
    return tuple((0, "") if value is None else (1, value) for value in group)


def _sample(
    definition: MetricDefinition,
    observation: Observation,
) -> int | float | None:
    source = definition.source
    if source.kind is MetricSourceKind.OBSERVATION_COUNT:
        return 1
    if source.kind is MetricSourceKind.INDICATOR:
        value = _facet(observation, source.indicator_field or "")
        return 1 if value in source.indicator_values else 0
    for measurement in observation.measurements:
        if (
            measurement.name == source.measurement_name
            and measurement.revision == source.measurement_revision
        ):
            return measurement.value
    return None


def _validated_sample(
    definition: MetricDefinition,
    observation: Observation,
) -> int | float | None:
    sample = _sample(definition, observation)
    if sample is None:
        return None
    try:
        return validate_metric_value(definition.metric_type, sample)
    except AIError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _empty_value(aggregation: MetricAggregation) -> int | float | None:
    if aggregation in {
        MetricAggregation.COUNT,
        MetricAggregation.SUM,
        MetricAggregation.RATE,
    }:
        return 0
    return None


@dataclass(slots=True)
class _Accumulator:
    aggregation: MetricAggregation
    count: int = 0
    value: int | float | None = None
    latest_at: datetime | None = None
    latest_digest: str | None = None
    samples: list[int | float] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.aggregation is MetricAggregation.PERCENTILE:
            self.samples = []

    def add(
        self,
        sample: int | float,
        *,
        occurred_at: datetime,
        digest: str | None,
    ) -> None:
        self.count += 1
        if self.aggregation is MetricAggregation.COUNT:
            return
        if self.aggregation in {
            MetricAggregation.SUM,
            MetricAggregation.MEAN,
            MetricAggregation.RATE,
        }:
            self.value = sample if self.value is None else self.value + sample
            return
        if self.aggregation is MetricAggregation.MIN:
            self.value = sample if self.value is None else min(self.value, sample)
            return
        if self.aggregation is MetricAggregation.MAX:
            self.value = sample if self.value is None else max(self.value, sample)
            return
        if self.aggregation is MetricAggregation.LATEST:
            if digest is None:
                raise RuntimeError("latest aggregation requires observation digest")
            if self.latest_at is None or (occurred_at, digest) > (
                self.latest_at,
                self.latest_digest or "",
            ):
                self.latest_at = occurred_at
                self.latest_digest = digest
                self.value = sample
            return
        if self.samples is None:
            raise RuntimeError("percentile accumulator is invalid")
        self.samples.append(sample)

    def result(
        self,
        *,
        percentile: float | None,
        seconds: float,
    ) -> int | float | None:
        if self.count == 0:
            return _empty_value(self.aggregation)
        if self.aggregation is MetricAggregation.COUNT:
            return self.count
        if self.aggregation is MetricAggregation.SUM:
            return self.value
        if self.aggregation is MetricAggregation.MEAN:
            if self.value is None:
                raise RuntimeError("mean accumulator is invalid")
            return self.value / self.count
        if self.aggregation in {
            MetricAggregation.MIN,
            MetricAggregation.MAX,
            MetricAggregation.LATEST,
        }:
            return self.value
        if self.aggregation is MetricAggregation.RATE:
            if self.value is None:
                raise RuntimeError("rate accumulator is invalid")
            return self.value / seconds
        if percentile is None or self.samples is None:
            raise RuntimeError("percentile is required")
        ordered = sorted(self.samples)
        rank = math.ceil(percentile * len(ordered))
        return ordered[rank - 1]


async def execute_query(
    store: MetricStore,
    namespace: str,
    definition: MetricDefinition,
    query: MetricQuery,
) -> MetricQueryResult:
    from ._model import _ALLOWED_AGGREGATIONS

    aggregation = query.aggregation or definition.default_aggregation
    if aggregation not in _ALLOWED_AGGREGATIONS[definition.metric_type]:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if aggregation is MetricAggregation.PERCENTILE and query.percentile is None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if aggregation is not MetricAggregation.PERCENTILE and query.percentile is not None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    for query_field in (*query.filters.keys(), *query.group_by):
        if not _query_field_allowed(definition, query_field):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    start, end = query.window.resolve()
    window_delta = end - start
    bucket_count = 1
    if query.bucket is not None:
        bucket_count = math.ceil(window_delta / query.bucket)
        if bucket_count > _MAX_BUCKETS:
            raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    pushdown = await _try_pushdown(
        store,
        namespace,
        definition,
        query,
        aggregation=aggregation,
        start=start,
        end=end,
        bucket_count=bucket_count,
    )
    if pushdown is not None:
        return _pushdown_result(
            definition,
            query,
            aggregation,
            start,
            end,
            bucket_count,
            pushdown,
        )

    groups: dict[tuple[str | None, ...], _Accumulator] = {}
    bucket_groups: dict[tuple[tuple[str | None, ...], int], _Accumulator] = {}
    actual_groups: set[tuple[str | None, ...]] = set()
    scanned_count = 0
    scanned_bytes = 0
    extracted_count = 0
    cursor: str | None = None

    while True:
        page = await store.scan_observations(
            namespace,
            definition.observation_kind,
            start,
            end,
            cursor=cursor,
            limit=_SCAN_PAGE_SIZE,
        )
        if len(page.items) > _SCAN_PAGE_SIZE:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not page.items and page.next_cursor is not None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

        for observation in page.items:
            scanned_count += 1
            if scanned_count > _MAX_SCANNED_OBSERVATIONS:
                raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
            scanned_bytes += len(
                canonical_json_bytes(observation_envelope(namespace, observation))
            )
            if scanned_bytes > _MAX_SCANNED_CANONICAL_BYTES:
                raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

            if any(
                _facet(observation, key) != value
                for key, value in query.filters.items()
            ):
                continue
            if any(
                observation.correlation.get(key) != value
                for key, value in query.correlation_filters.items()
            ):
                continue
            sample = _validated_sample(definition, observation)
            if sample is None:
                continue
            extracted_count += 1
            if extracted_count > _MAX_EXTRACTED_SAMPLES:
                raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

            group = tuple(_facet(observation, field) for field in query.group_by)
            if query.group_by:
                actual_groups.add(group)
                if len(actual_groups) > _MAX_GROUPS:
                    raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

            digest = (
                observation_digest(namespace, observation.observation_id)
                if aggregation is MetricAggregation.LATEST
                else None
            )
            if query.bucket is None:
                accumulator = groups.get(group)
                if accumulator is None:
                    accumulator = _Accumulator(aggregation)
                    groups[group] = accumulator
                accumulator.add(
                    sample,
                    occurred_at=observation.occurred_at,
                    digest=digest,
                )
            else:
                index = int((observation.occurred_at - start) // query.bucket)
                if 0 <= index < bucket_count:
                    key = (group, index)
                    accumulator = bucket_groups.get(key)
                    if accumulator is None:
                        accumulator = _Accumulator(aggregation)
                        bucket_groups[key] = accumulator
                    accumulator.add(
                        sample,
                        occurred_at=observation.occurred_at,
                        digest=digest,
                    )

        next_cursor = page.next_cursor
        if next_cursor is None:
            break
        if next_cursor == cursor:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        cursor = next_cursor

    result_unit = _result_unit(definition, aggregation)
    points: list[MetricPoint] = []

    result_groups = (
        sorted(actual_groups, key=_group_sort_key) if query.group_by else [()]
    )
    if query.bucket is not None:
        expected_points = len(result_groups) * bucket_count
    else:
        expected_points = len(result_groups)
    if expected_points > _MAX_RESULT_POINTS:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    if query.bucket is None:
        for group in result_groups:
            accumulator = groups.get(group) or _Accumulator(aggregation)
            points.append(
                MetricPoint(
                    dimensions=tuple(zip(query.group_by, group, strict=True)),
                    bucket_start=None,
                    bucket_end=None,
                    value=accumulator.result(
                        percentile=query.percentile,
                        seconds=window_delta.total_seconds(),
                    ),
                    sample_count=accumulator.count,
                )
            )
    else:
        for group in result_groups:
            for index in range(bucket_count):
                bucket_start = start + query.bucket * index
                bucket_end = min(end, bucket_start + query.bucket)
                accumulator = bucket_groups.get((group, index)) or _Accumulator(
                    aggregation
                )
                points.append(
                    MetricPoint(
                        dimensions=tuple(zip(query.group_by, group, strict=True)),
                        bucket_start=bucket_start,
                        bucket_end=bucket_end,
                        value=accumulator.result(
                            percentile=query.percentile,
                            seconds=(bucket_end - bucket_start).total_seconds(),
                        ),
                        sample_count=accumulator.count,
                    )
                )

    return MetricQueryResult(
        metric=definition.name,
        revision=definition.revision,
        unit=result_unit,
        aggregation=aggregation,
        window_start=start,
        window_end=end,
        points=tuple(points),
    )


async def _try_pushdown(
    store: MetricStore,
    namespace: str,
    definition: MetricDefinition,
    query: MetricQuery,
    *,
    aggregation: MetricAggregation,
    start: datetime,
    end: datetime,
    bucket_count: int,
) -> _MetricQueryPushdownResult | None:
    if not isinstance(store, _MetricQueryPushdownStore):
        return None
    bucket_microseconds = None
    if query.bucket is not None:
        bucket_microseconds = (
            (query.bucket.days * 86_400 + query.bucket.seconds) * 1_000_000
            + query.bucket.microseconds
        )
    source = definition.source
    plan = _MetricQueryPushdownPlan(
        observation_kind=definition.observation_kind,
        source_kind=source.kind,
        metric_type=definition.metric_type,
        measurement_name=source.measurement_name,
        measurement_revision=source.measurement_revision,
        indicator_field=source.indicator_field,
        indicator_values=source.indicator_values,
        aggregation=aggregation,
        percentile=query.percentile,
        start=start,
        end=end,
        filters=tuple(query.filters.items()),
        correlation_filters=tuple(query.correlation_filters.items()),
        group_by=query.group_by,
        bucket_microseconds=bucket_microseconds,
        bucket_count=bucket_count,
        max_scanned_observations=_MAX_SCANNED_OBSERVATIONS,
        max_extracted_samples=_MAX_EXTRACTED_SAMPLES,
        max_groups=_MAX_GROUPS,
        max_result_points=_MAX_RESULT_POINTS,
    )
    return await store._execute_metric_query(namespace, plan)


def _pushdown_result(
    definition: MetricDefinition,
    query: MetricQuery,
    aggregation: MetricAggregation,
    start: datetime,
    end: datetime,
    bucket_count: int,
    result: _MetricQueryPushdownResult,
) -> MetricQueryResult:
    by_partition: dict[
        tuple[tuple[str | None, ...], int | None], _MetricQueryPushdownRow
    ] = {}
    actual_groups: set[tuple[str | None, ...]] = set()
    for row in result.rows:
        key = (row.group, row.bucket_index)
        if key in by_partition:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        by_partition[key] = row
        if query.group_by:
            actual_groups.add(row.group)

    result_groups = (
        sorted(actual_groups, key=_group_sort_key) if query.group_by else [()]
    )
    expected_points = len(result_groups) * bucket_count
    if expected_points > _MAX_RESULT_POINTS:
        raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    points: list[MetricPoint] = []
    if query.bucket is None:
        seconds = (end - start).total_seconds()
        for group in result_groups:
            row = by_partition.get((group, None))
            points.append(
                MetricPoint(
                    dimensions=tuple(zip(query.group_by, group, strict=True)),
                    bucket_start=None,
                    bucket_end=None,
                    value=_pushdown_value(row, aggregation, seconds=seconds),
                    sample_count=0 if row is None else row.sample_count,
                )
            )
    else:
        for group in result_groups:
            for index in range(bucket_count):
                bucket_start = start + query.bucket * index
                bucket_end = min(end, bucket_start + query.bucket)
                row = by_partition.get((group, index))
                points.append(
                    MetricPoint(
                        dimensions=tuple(zip(query.group_by, group, strict=True)),
                        bucket_start=bucket_start,
                        bucket_end=bucket_end,
                        value=_pushdown_value(
                            row,
                            aggregation,
                            seconds=(bucket_end - bucket_start).total_seconds(),
                        ),
                        sample_count=0 if row is None else row.sample_count,
                    )
                )

    return MetricQueryResult(
        metric=definition.name,
        revision=definition.revision,
        unit=_result_unit(definition, aggregation),
        aggregation=aggregation,
        window_start=start,
        window_end=end,
        points=tuple(points),
    )


def _pushdown_value(
    row: _MetricQueryPushdownRow | None,
    aggregation: MetricAggregation,
    *,
    seconds: float,
) -> int | float | None:
    if row is None or row.sample_count == 0:
        return _empty_value(aggregation)
    if aggregation is MetricAggregation.COUNT:
        return row.sample_count
    if aggregation in {
        MetricAggregation.MIN,
        MetricAggregation.MAX,
        MetricAggregation.LATEST,
        MetricAggregation.PERCENTILE,
    }:
        if row.selected_value is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return row.selected_value
    if row.sample_sum is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if aggregation is MetricAggregation.SUM:
        return row.sample_sum
    if aggregation is MetricAggregation.MEAN:
        return row.sample_sum / row.sample_count
    if aggregation is MetricAggregation.RATE:
        return row.sample_sum / seconds
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _result_unit(
    definition: MetricDefinition,
    aggregation: MetricAggregation,
) -> str:
    if aggregation is MetricAggregation.COUNT:
        return "1"
    if aggregation is MetricAggregation.RATE:
        return f"{definition.unit}/s"
    return definition.unit


__all__: list[str] = []
