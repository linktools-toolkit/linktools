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

_IDENTIFIER_MAX = 128
_UNIT_MAX = 32
_CANONICAL_FIELD_MAX = 128
_VALUE_MAX = 256
_DIMENSIONS_MAX = 16
_CORRELATIONS_MAX = 16
_MEASUREMENTS_MAX = 32
_QUERY_FIELDS_MAX = 20
_INDICATOR_VALUES_MAX = 16
_INT64_MIN = -(2**63)
_INT64_MAX = 2**63 - 1
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
        value.encode("utf-8")
    except UnicodeEncodeError as error:
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name}
        ) from error
    if len(value) > maximum:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return value


def _identifier(value: object, *, name: str) -> str:
    text = _required_text(value, name=name, maximum=_IDENTIFIER_MAX)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", text):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return text


def _utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value.astimezone(timezone.utc)


def _int64(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    if not _INT64_MIN <= value <= _INT64_MAX:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return value


def _finite_number(value: object, *, name: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    if isinstance(value, int):
        return _int64(value, name=name)
    if not math.isfinite(value):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return 0.0 if value == 0.0 else value


def _string_tuple(
    value: object,
    *,
    name: str,
    maximum: int,
) -> tuple[str, ...]:
    if not isinstance(value, tuple):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": name})
    return tuple(_required_text(item, name=name, maximum=maximum) for item in value)


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
        values = _string_tuple(
            self.indicator_values,
            name="indicator value",
            maximum=_CANONICAL_FIELD_MAX,
        )
        normalized = tuple(sorted(set(values)))
        if not normalized or len(normalized) > _INDICATOR_VALUES_MAX:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "indicator_values", normalized)

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
        if isinstance(self.version, bool) or not isinstance(self.version, int):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.version != 1:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        validate_resource_id(self.observation_id)
        _identifier(self.kind, name="kind")
        object.__setattr__(self, "occurred_at", _utc(self.occurred_at))
        if self.source_namespace is not None:
            validate_persistence_namespace(self.source_namespace)
        if self.tenant_id is not None:
            validate_tenant_id(self.tenant_id)
        for name in ("status", "error_code"):
            value = getattr(self, name)
            if value is not None:
                _required_text(value, name=name, maximum=_CANONICAL_FIELD_MAX)

        if not isinstance(self.correlation, Mapping):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        correlation = dict(self.correlation)
        if len(correlation) > _CORRELATIONS_MAX:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for key, value in correlation.items():
            _identifier(key, name="correlation key")
            if key in _CANONICAL_QUERY_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if isinstance(value, str):
                _required_text(value, name="correlation value", maximum=_VALUE_MAX)
            else:
                _int64(value, name="correlation value")

        if not isinstance(self.dimensions, Mapping):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        dimensions = dict(self.dimensions)
        if len(dimensions) > _DIMENSIONS_MAX:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for key, value in dimensions.items():
            _identifier(key, name="dimension key")
            if key in _CANONICAL_QUERY_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            _required_text(value, name="dimension value", maximum=_VALUE_MAX)

        if not isinstance(self.measurements, tuple):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        measurements = self.measurements
        if len(measurements) > _MEASUREMENTS_MAX or any(
            not isinstance(item, MetricMeasurement) for item in measurements
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        identities = [(item.name, item.revision) for item in measurements]
        if len(set(identities)) != len(identities):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        object.__setattr__(self, "correlation", MappingProxyType(correlation))
        object.__setattr__(self, "dimensions", MappingProxyType(dimensions))


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

    def __post_init__(self) -> None:
        _identifier(self.name, name="metric name")
        if (
            isinstance(self.revision, bool)
            or not isinstance(self.revision, int)
            or self.revision < 1
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _identifier(self.observation_kind, name="observation_kind")
        if not isinstance(self.source, MetricSource):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(self.metric_type, MetricType):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _required_text(self.unit, name="unit", maximum=_UNIT_MAX)
        if not isinstance(self.default_aggregation, MetricAggregation):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.default_aggregation not in _ALLOWED_AGGREGATIONS[self.metric_type]:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        fields = _string_tuple(
            self.query_fields,
            name="query field",
            maximum=_IDENTIFIER_MAX,
        )
        if len(fields) > _QUERY_FIELDS_MAX or len(fields) != len(set(fields)):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for field in fields:
            _identifier(field, name="query field")

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
            return
        if (
            not isinstance(self.recent_delta, timedelta)
            or self.recent_delta <= timedelta(0)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

    @classmethod
    def recent(
        cls,
        *,
        minutes: int = 0,
        hours: int = 0,
        days: int = 0,
    ) -> MetricWindow:
        for value in (minutes, hours, days):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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

        if not isinstance(self.filters, Mapping):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        filters = dict(self.filters)
        for key, value in filters.items():
            _identifier(key, name="filter key")
            _required_text(value, name="filter value", maximum=_VALUE_MAX)

        groups = _string_tuple(
            self.group_by,
            name="group field",
            maximum=_IDENTIFIER_MAX,
        )
        if len(groups) != len(set(groups)):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        for field in groups:
            _identifier(field, name="group field")

        if self.bucket is not None and (
            not isinstance(self.bucket, timedelta) or self.bucket <= timedelta(0)
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        object.__setattr__(self, "filters", MappingProxyType(filters))
        object.__setattr__(self, "group_by", groups)


@dataclass(frozen=True, slots=True)
class MetricPoint:
    dimensions: tuple[tuple[str, str | None], ...]
    bucket_start: datetime | None
    bucket_end: datetime | None
    value: int | float | None
    sample_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.dimensions, tuple) or any(
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or (item[1] is not None and not isinstance(item[1], str))
            for item in self.dimensions
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            isinstance(self.sample_count, bool)
            or not isinstance(self.sample_count, int)
            or self.sample_count < 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if self.bucket_start is not None:
            object.__setattr__(self, "bucket_start", _utc(self.bucket_start))
        if self.bucket_end is not None:
            object.__setattr__(self, "bucket_end", _utc(self.bucket_end))
        if (self.bucket_start is None) != (self.bucket_end is None):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            self.bucket_start is not None
            and self.bucket_end is not None
            and self.bucket_start >= self.bucket_end
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


@dataclass(frozen=True, slots=True)
class MetricQueryResult:
    metric: str
    revision: int
    unit: str
    aggregation: MetricAggregation
    window_start: datetime
    window_end: datetime
    points: tuple[MetricPoint, ...]


class MetricRecorder(Protocol):
    def try_record(self, observation: Observation) -> bool: ...


__all__ = [
    "MetricAggregation",
    "MetricDefinition",
    "MetricMeasurement",
    "MetricPoint",
    "MetricQuery",
    "MetricQueryResult",
    "MetricRecorder",
    "MetricSource",
    "MetricSourceKind",
    "MetricType",
    "MetricWindow",
    "Observation",
]
