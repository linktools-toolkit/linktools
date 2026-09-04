#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Versioned metric contracts."""

from __future__ import annotations

import math
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from types import MappingProxyType
from typing import Protocol

from ..core import (
    validate_persistence_namespace,
    validate_resource_id,
    validate_tenant_id,
)
from ..errors import AIError, ErrorCode

_METRIC_NAME_MAX = 128
_KIND_MAX = 128
_FIELD_MAX = 128
_DIMENSION_VALUE_MAX = 256
_DIMENSIONS_MAX = 16
_CORRELATIONS_MAX = 16
_MEASUREMENTS_MAX = 32
_CANONICAL_QUERY_FIELDS = frozenset(
    {"source_namespace", "tenant_id", "status", "error_code"}
)


class MetricType(str, Enum):
    COUNTER = "COUNTER"
    GAUGE = "GAUGE"
    DISTRIBUTION = "DISTRIBUTION"
    RATIO = "RATIO"


class MetricAggregation(str, Enum):
    COUNT = "COUNT"
    SUM = "SUM"
    MEAN = "MEAN"
    MIN = "MIN"
    MAX = "MAX"
    LATEST = "LATEST"
    RATE = "RATE"
    PERCENTILE = "PERCENTILE"


class MetricSourceKind(str, Enum):
    OBSERVATION_COUNT = "OBSERVATION_COUNT"
    MEASUREMENT = "MEASUREMENT"
    INDICATOR = "INDICATOR"


_ALLOWED_AGGREGATIONS: dict[MetricType, frozenset[MetricAggregation]] = {
    MetricType.COUNTER: frozenset(
        {MetricAggregation.COUNT, MetricAggregation.SUM, MetricAggregation.RATE}
    ),
    MetricType.GAUGE: frozenset(
        {
            MetricAggregation.COUNT,
            MetricAggregation.MEAN,
            MetricAggregation.MIN,
            MetricAggregation.MAX,
            MetricAggregation.LATEST,
        }
    ),
    MetricType.DISTRIBUTION: frozenset(
        {
            MetricAggregation.COUNT,
            MetricAggregation.MEAN,
            MetricAggregation.MIN,
            MetricAggregation.MAX,
            MetricAggregation.PERCENTILE,
        }
    ),
    MetricType.RATIO: frozenset(
        {MetricAggregation.COUNT, MetricAggregation.MEAN}
    ),
}


def _required_text(value: object, *, name: str, maximum: int) -> str:
    if (
        not isinstance(value, str)
        or not value
        or value != value.strip()
        or any(unicodedata.category(character) == "Cc" for character in value)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    try:
        size = len(value.encode("utf-8"))
    except UnicodeEncodeError as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name}
        ) from error
    if size > maximum:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return value


def _identifier(value: object, *, name: str, maximum: int = _METRIC_NAME_MAX) -> str:
    text = _required_text(value, name=name, maximum=maximum)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return text


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value.astimezone(timezone.utc)


def _finite_number(value: object, *, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if isinstance(value, float) and not math.isfinite(value):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


def validate_metric_value(metric_type: MetricType, value: object) -> int | float:
    normalized = _finite_number(value, name="metric value")
    if metric_type is MetricType.COUNTER and normalized < 0:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    if metric_type is MetricType.RATIO and not 0 <= normalized <= 1:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return normalized


@dataclass(frozen=True, slots=True)
class MetricSource:
    kind: MetricSourceKind
    measurement_name: str | None = None
    measurement_revision: int | None = None
    indicator_field: str | None = None
    indicator_values: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, MetricSourceKind):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.kind is MetricSourceKind.OBSERVATION_COUNT:
            if (
                self.measurement_name is not None
                or self.measurement_revision is not None
                or self.indicator_field is not None
                or self.indicator_values
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return
        if self.kind is MetricSourceKind.MEASUREMENT:
            _identifier(self.measurement_name, name="measurement_name")
            if (
                isinstance(self.measurement_revision, bool)
                or not isinstance(self.measurement_revision, int)
                or self.measurement_revision < 1
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if self.indicator_field is not None or self.indicator_values:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            return
        if self.measurement_name is not None or self.measurement_revision is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.indicator_field not in {"status", "error_code"}:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not self.indicator_values:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        values = tuple(
            sorted(
                {
                    _required_text(
                        item,
                        name="indicator value",
                        maximum=_DIMENSION_VALUE_MAX,
                    )
                    for item in self.indicator_values
                }
            )
        )
        object.__setattr__(self, "indicator_values", values)

    @classmethod
    def observation_count(cls) -> MetricSource:
        return cls(MetricSourceKind.OBSERVATION_COUNT)

    @classmethod
    def measurement(cls, name: str, *, revision: int = 1) -> MetricSource:
        return cls(
            MetricSourceKind.MEASUREMENT,
            measurement_name=name,
            measurement_revision=revision,
        )

    @classmethod
    def indicator(cls, field: str, values: tuple[str, ...]) -> MetricSource:
        return cls(
            MetricSourceKind.INDICATOR,
            indicator_field=field,
            indicator_values=values,
        )


@dataclass(frozen=True, slots=True)
class MetricMeasurement:
    name: str
    revision: int
    value: int | float

    def __post_init__(self) -> None:
        _identifier(self.name, name="measurement name")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(
            self, "value", _finite_number(self.value, name="measurement value")
        )


@dataclass(frozen=True, slots=True)
class Observation:
    version: int
    observation_id: str
    kind: str
    occurred_at: datetime
    source_namespace: str | None
    tenant_id: str | None
    status: str | None
    error_code: str | None
    correlation: Mapping[str, str | int]
    dimensions: Mapping[str, str]
    measurements: tuple[MetricMeasurement, ...]

    def __post_init__(self) -> None:
        if self.version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        validate_resource_id(self.observation_id)
        _identifier(self.kind, name="kind", maximum=_KIND_MAX)
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at))
        if self.source_namespace is not None:
            validate_persistence_namespace(self.source_namespace)
        if self.tenant_id is not None:
            validate_tenant_id(self.tenant_id)
        for name in ("status", "error_code"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, name=name, maximum=_DIMENSION_VALUE_MAX)
        correlation = dict(self.correlation)
        if len(correlation) > _CORRELATIONS_MAX:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for key, value in correlation.items():
            _identifier(key, name="correlation key", maximum=_FIELD_MAX)
            if isinstance(value, bool) or not isinstance(value, (str, int)):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if isinstance(value, str):
                _required_text(
                    value, name="correlation value", maximum=_DIMENSION_VALUE_MAX
                )
        dimensions = dict(self.dimensions)
        if len(dimensions) > _DIMENSIONS_MAX:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for key, value in dimensions.items():
            _identifier(key, name="dimension key", maximum=_FIELD_MAX)
            _required_text(
                value, name="dimension value", maximum=_DIMENSION_VALUE_MAX
            )
            if key in _CANONICAL_QUERY_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        measurements = tuple(self.measurements)
        if len(measurements) > _MEASUREMENTS_MAX:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        identities = [(item.name, item.revision) for item in measurements]
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "correlation", MappingProxyType(correlation))
        object.__setattr__(self, "dimensions", MappingProxyType(dimensions))
        object.__setattr__(self, "measurements", measurements)


@dataclass(frozen=True, slots=True)
class MetricDefinition:
    name: str
    revision: int
    observation_kind: str
    source: MetricSource
    metric_type: MetricType
    unit: str
    default_aggregation: MetricAggregation
    query_fields: tuple[str, ...] = ()
    description: str | None = None

    def __post_init__(self) -> None:
        _identifier(self.name, name="metric name")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _identifier(self.observation_kind, name="observation_kind", maximum=_KIND_MAX)
        if not isinstance(self.source, MetricSource):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(self.metric_type, MetricType):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _required_text(self.unit, name="unit", maximum=64)
        if not isinstance(self.default_aggregation, MetricAggregation):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.default_aggregation not in _ALLOWED_AGGREGATIONS[self.metric_type]:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        fields = tuple(sorted(set(self.query_fields)))
        for field in fields:
            _identifier(field, name="query field", maximum=_FIELD_MAX)
        if self.description is not None and (
            not isinstance(self.description, str) or len(self.description) > 1024
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            self.source.kind is MetricSourceKind.OBSERVATION_COUNT
            and self.metric_type is not MetricType.COUNTER
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            self.source.kind is MetricSourceKind.INDICATOR
            and self.metric_type not in {MetricType.COUNTER, MetricType.RATIO}
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "query_fields", fields)


@dataclass(frozen=True, slots=True)
class MetricWindow:
    start: datetime | None = None
    end: datetime | None = None
    recent_delta: timedelta | None = None

    def __post_init__(self) -> None:
        explicit = self.start is not None or self.end is not None
        if explicit:
            if self.start is None or self.end is None or self.recent_delta is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            start = _utc(self.start)
            end = _utc(self.end)
            if start >= end:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            object.__setattr__(self, "start", start)
            object.__setattr__(self, "end", end)
        else:
            if self.recent_delta is None or self.recent_delta <= timedelta(0):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    @classmethod
    def recent(
        cls,
        *,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
    ) -> MetricWindow:
        return cls(recent_delta=timedelta(minutes=minutes, hours=hours, days=days))

    @classmethod
    def between(cls, start: datetime, end: datetime) -> MetricWindow:
        return cls(start=start, end=end)

    def resolve(self, *, now: datetime | None = None) -> tuple[datetime, datetime]:
        if self.recent_delta is None:
            if self.start is None or self.end is None:
                raise RuntimeError("invalid metric window")
            return self.start, self.end
        resolved_end = _utc(now or datetime.now(timezone.utc))
        return resolved_end - self.recent_delta, resolved_end


@dataclass(frozen=True, slots=True)
class MetricQuery:
    metric: str
    window: MetricWindow
    revision: int | None = None
    aggregation: MetricAggregation | None = None
    percentile: float | None = None
    filters: Mapping[str, str] = MappingProxyType({})
    group_by: tuple[str, ...] = ()
    bucket: timedelta | None = None

    def __post_init__(self) -> None:
        _identifier(self.metric, name="metric")
        if self.revision is not None and (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(self.window, MetricWindow):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.aggregation is not None and not isinstance(
            self.aggregation, MetricAggregation
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.aggregation is MetricAggregation.PERCENTILE:
            if (
                isinstance(self.percentile, bool)
                or not isinstance(self.percentile, (int, float))
                or not 0 < self.percentile <= 1
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        elif self.percentile is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        filters = dict(self.filters)
        for key, value in filters.items():
            _identifier(key, name="filter key", maximum=_FIELD_MAX)
            _required_text(value, name="filter value", maximum=_DIMENSION_VALUE_MAX)
        groups = tuple(self.group_by)
        if len(groups) != len(set(groups)):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for field in groups:
            _identifier(field, name="group field", maximum=_FIELD_MAX)
        if self.bucket is not None and self.bucket <= timedelta(0):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "filters", MappingProxyType(filters))
        object.__setattr__(self, "group_by", groups)


@dataclass(frozen=True, slots=True)
class MetricQueryPoint:
    dimensions: Mapping[str, str | None]
    value: int | float | None
    sample_count: int
    bucket_start: datetime | None = None
    bucket_end: datetime | None = None

    def __post_init__(self) -> None:
        if self.sample_count < 0:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "dimensions", MappingProxyType(dict(self.dimensions)))


@dataclass(frozen=True, slots=True)
class MetricQueryResult:
    metric: str
    revision: int
    unit: str
    aggregation: MetricAggregation
    window_start: datetime
    window_end: datetime
    points: tuple[MetricQueryPoint, ...]


class MetricRecorder(Protocol):
    def try_record(self, observation: Observation) -> bool: ...


__all__ = [
    "MetricAggregation",
    "MetricDefinition",
    "MetricMeasurement",
    "MetricQuery",
    "MetricQueryPoint",
    "MetricQueryResult",
    "MetricRecorder",
    "MetricSource",
    "MetricSourceKind",
    "MetricType",
    "MetricWindow",
    "Observation",
]
