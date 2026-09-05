#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral metric query engine."""

from __future__ import annotations

import math
from collections import defaultdict
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
from ._store import MetricStore

_SCAN_PAGE_SIZE = 512
_MAX_SCANNED_OBSERVATIONS = 100_000
_MAX_SCANNED_CANONICAL_BYTES = 64 * 1024 * 1024
_MAX_EXTRACTED_SAMPLES = 100_000
_MAX_GROUPS = 256
_MAX_BUCKETS = 2_048
_MAX_RESULT_POINTS = 16_384


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


def _aggregate(
    aggregation: MetricAggregation,
    samples: list[tuple[int | float, datetime, str]],
    *,
    percentile: float | None,
    seconds: float,
) -> int | float | None:
    if not samples:
        return _empty_value(aggregation)
    values = [sample[0] for sample in samples]
    if aggregation is MetricAggregation.COUNT:
        return len(values)
    if aggregation is MetricAggregation.SUM:
        return sum(values)
    if aggregation is MetricAggregation.MEAN:
        return sum(values) / len(values)
    if aggregation is MetricAggregation.MIN:
        return min(values)
    if aggregation is MetricAggregation.MAX:
        return max(values)
    if aggregation is MetricAggregation.RATE:
        return sum(values) / seconds
    if aggregation is MetricAggregation.LATEST:
        return max(samples, key=lambda item: (item[1], item[2]))[0]
    if percentile is None:
        raise RuntimeError("percentile is required")
    ordered = sorted(values)
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

    allowed_fields = set(definition.query_fields)
    for field in (*query.filters.keys(), *query.group_by):
        if field not in allowed_fields:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    start, end = query.window.resolve()
    window_delta = end - start
    bucket_count = 1
    if query.bucket is not None:
        if query.bucket > window_delta:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        bucket_count = math.ceil(window_delta / query.bucket)
        if bucket_count > _MAX_BUCKETS:
            raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    groups: dict[
        tuple[str | None, ...],
        list[tuple[int | float, datetime, str]],
    ] = defaultdict(list)
    bucket_groups: dict[
        tuple[tuple[str | None, ...], int],
        list[tuple[int | float, datetime, str]],
    ] = defaultdict(list)
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

            item = (
                sample,
                observation.occurred_at,
                observation_digest(namespace, observation.observation_id),
            )
            if query.bucket is None:
                groups[group].append(item)
            else:
                index = int((observation.occurred_at - start) // query.bucket)
                if 0 <= index < bucket_count:
                    bucket_groups[(group, index)].append(item)

        next_cursor = page.next_cursor
        if next_cursor is None:
            break
        if next_cursor == cursor:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        cursor = next_cursor

    result_unit = (
        "1"
        if aggregation is MetricAggregation.COUNT
        else f"{definition.unit}/s"
        if aggregation is MetricAggregation.RATE
        else definition.unit
    )
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
        if not query.group_by and () not in groups:
            groups[()] = []
        for group in result_groups:
            samples = groups.get(group, [])
            points.append(
                MetricPoint(
                    dimensions=tuple(zip(query.group_by, group, strict=True)),
                    bucket_start=None,
                    bucket_end=None,
                    value=_aggregate(
                        aggregation,
                        samples,
                        percentile=query.percentile,
                        seconds=window_delta.total_seconds(),
                    ),
                    sample_count=len(samples),
                )
            )
    else:
        for group in result_groups:
            for index in range(bucket_count):
                bucket_start = start + query.bucket * index
                bucket_end = min(end, bucket_start + query.bucket)
                samples = bucket_groups.get((group, index), [])
                points.append(
                    MetricPoint(
                        dimensions=tuple(zip(query.group_by, group, strict=True)),
                        bucket_start=bucket_start,
                        bucket_end=bucket_end,
                        value=_aggregate(
                            aggregation,
                            samples,
                            percentile=query.percentile,
                            seconds=(bucket_end - bucket_start).total_seconds(),
                        ),
                        sample_count=len(samples),
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


__all__: list[str] = []
