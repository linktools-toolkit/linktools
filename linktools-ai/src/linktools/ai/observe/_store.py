#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MetricStore contract."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Protocol, runtime_checkable

from ..core import Page
from ..errors import AIError, ErrorCode
from ._model import (
    MetricAggregation,
    MetricDefinition,
    MetricSourceKind,
    Observation,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")


def _scan_cursor(occurred_at: datetime, observation_digest: str) -> str:
    if occurred_at.tzinfo is None or _SHA256.fullmatch(observation_digest) is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return f"{occurred_at.astimezone(timezone.utc).isoformat()}|{observation_digest}"


def _parse_scan_cursor(cursor: str) -> tuple[datetime, str]:
    if not isinstance(cursor, str) or not cursor:
        raise AIError(ErrorCode.CURSOR_INVALID)
    occurred_raw, separator, digest = cursor.partition("|")
    if not separator or _SHA256.fullmatch(digest) is None:
        raise AIError(ErrorCode.CURSOR_INVALID)
    try:
        occurred_at = datetime.fromisoformat(occurred_raw)
    except ValueError as error:
        raise AIError(ErrorCode.CURSOR_INVALID) from error
    if occurred_at.tzinfo is None:
        raise AIError(ErrorCode.CURSOR_INVALID)
    return occurred_at.astimezone(timezone.utc), digest


@dataclass(frozen=True, slots=True)
class _MetricQueryPushdownPlan:
    observation_kind: str
    source_kind: MetricSourceKind
    measurement_name: str | None
    measurement_revision: int | None
    indicator_field: str | None
    indicator_values: tuple[str, ...]
    aggregation: MetricAggregation
    percentile: float | None
    start: datetime
    end: datetime
    filters: tuple[tuple[str, str], ...]
    correlation_filters: tuple[tuple[str, str | int], ...]
    group_by: tuple[str, ...]
    bucket_microseconds: int | None
    bucket_count: int
    max_scanned_observations: int
    max_extracted_samples: int
    max_groups: int
    max_result_points: int


@dataclass(frozen=True, slots=True)
class _MetricQueryPushdownRow:
    group: tuple[str | None, ...]
    bucket_index: int | None
    sample_count: int
    sample_sum: int | float | None = None
    selected_value: int | float | None = None


@dataclass(frozen=True, slots=True)
class _MetricQueryPushdownResult:
    rows: tuple[_MetricQueryPushdownRow, ...]


@runtime_checkable
class _MetricQueryPushdownStore(Protocol):
    async def _execute_metric_query(
        self,
        namespace: str,
        plan: _MetricQueryPushdownPlan,
    ) -> _MetricQueryPushdownResult | None: ...


class MetricStore(Protocol):
    """Persist Metrics facts without performing metric aggregation.

    Definition identity is ``(namespace, name, revision)``. Rewriting the same
    identity with identical semantics is idempotent; different semantics must
    fail with ``STORAGE_CONFLICT``.

    Observation identity is the namespace-scoped ``observation_id``. A batch
    write is atomic: identical replays are idempotent and any conflicting
    identity fails the whole batch with ``STORAGE_CONFLICT``.

    ``scan_observations`` returns only the requested namespace/kind and the
    half-open interval ``[start, end)`` in strict ascending
    ``(occurred_at, observation_digest)`` order. ``cursor`` is an opaque
    continuation token produced by the same store; ``next_cursor=None`` means
    the scan is exhausted. A page never contains more than ``limit`` items.
    """

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> MetricDefinition: ...

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> MetricDefinition | None: ...

    async def latest_definition(
        self,
        namespace: str,
        name: str,
    ) -> MetricDefinition | None: ...

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None: ...

    async def get_observation(
        self,
        namespace: str,
        observation_id: str,
    ) -> Observation | None: ...

    async def scan_observations(
        self,
        namespace: str,
        kind: str,
        start: datetime,
        end: datetime,
        *,
        cursor: str | None,
        limit: int,
    ) -> Page[Observation]: ...

    async def prune_observations(
        self,
        namespace: str,
        *,
        before: datetime,
    ) -> int: ...


__all__ = ["MetricStore"]
