#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Backend-neutral metric query engine."""

from __future__ import annotations

import math
from collections import defaultdict
from datetime import datetime

from ..errors import AIError, ErrorCode
from ._codec import observation_digest
from ._model import (
    MetricAggregation,
    MetricDefinition,
    MetricQuery,
    MetricQueryPoint,
    MetricQueryResult,
    MetricSourceKind,
    Observation,
)
from ._store import MetricStore

_MAX_SCANNED_OBSERVATIONS = 100_000
_MAX_EXTRACTED_SAMPLES = 100_000
_MAX_GROUPS = 1_000
_MAX_BUCKETS = 10_000


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
    aggregation = query.aggregation or definition.default_aggregation
    from ._model import _ALLOWED_AGGREGATIONS

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
    bucket_count = 1
    if query.bucket is not None:
        bucket_count = math.ceil((end - start) / query.bucket)
        if bucket_count > _MAX_BUCKETS:
            raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)

    observations = await store.scan_observations(
        namespace,
        kind=definition.observation_kind,
        start=start,
        end=end,
        limit=_MAX_SCANNED_OBSERVATIONS + 1,
    )
    if len(observations) > _MAX_SCANNED_OBSERVATIONS:
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
    sample_count = 0

    for observation in observations:
        if any(
            _facet(observation, key) != value
            for key, value in query.filters.items()
        ):
            continue
        group = tuple(_facet(observation, field) for field in query.group_by)
        if query.group_by:
            actual_groups.add(group)
            if len(actual_groups) > _MAX_GROUPS:
                raise AIError(ErrorCode.METRIC_QUERY_LIMIT_EXCEEDED)
        sample = _sample(definition, observation)
        if sample is None:
            continue
        sample_count += 1
        if sample_count > _MAX_EXTRACTED_SAMPLES:
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

    result_unit = (
        f"{definition.unit}/s"
        if aggregation is MetricAggregation.RATE
        else definition.unit
    )
    points: list[MetricQueryPoint] = []

    if query.bucket is None:
        result_groups = (
            sorted(actual_groups, key=_group_sort_key) if query.group_by else [()]
        )
        if not query.group_by and () not in groups:
            groups[()] = []
        for group in result_groups:
            samples = groups.get(group, [])
            points.append(
                MetricQueryPoint(
                    dimensions=dict(zip(query.group_by, group)),
                    value=_aggregate(
                        aggregation,
                        samples,
                        percentile=query.percentile,
                        seconds=(end - start).total_seconds(),
                    ),
                    sample_count=len(samples),
                )
            )
    else:
        result_groups = (
            sorted(actual_groups, key=_group_sort_key) if query.group_by else [()]
        )
        for group in result_groups:
            for index in range(bucket_count):
                bucket_start = start + query.bucket * index
                bucket_end = min(end, bucket_start + query.bucket)
                samples = bucket_groups.get((group, index), [])
                points.append(
                    MetricQueryPoint(
                        dimensions=dict(zip(query.group_by, group)),
                        value=_aggregate(
                            aggregation,
                            samples,
                            percentile=query.percentile,
                            seconds=(bucket_end - bucket_start).total_seconds(),
                        ),
                        sample_count=len(samples),
                        bucket_start=bucket_start,
                        bucket_end=bucket_end,
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
