#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory MetricStore."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from ..core import Page
from ..errors import AIError, ErrorCode
from ._codec import (
    same_definition_contract,
    observation_digest,
    observation_payload_digest,
)
from ._model import MetricDefinition, Observation
from ._store import _parse_scan_cursor, _scan_cursor, _observation_pruned


@dataclass(frozen=True, slots=True)
class _ObservationRecord:
    namespace: str
    observation_id: str
    payload_digest: str
    kind: str
    occurred_at: datetime
    observation: Observation | None
    pruned_at: datetime | None = None


class InMemoryMetricStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._definitions: dict[tuple[str, str, int], MetricDefinition] = {}
        self._observations: dict[str, _ObservationRecord] = {}

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> MetricDefinition:
        key = (namespace, definition.name, definition.revision)
        async with self._lock:
            current = self._definitions.get(key)
            if current is None:
                self._definitions[key] = definition
                return definition
            if not same_definition_contract(current, definition):
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return current

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> MetricDefinition | None:
        async with self._lock:
            return self._definitions.get((namespace, name, revision))

    async def latest_definition(
        self,
        namespace: str,
        name: str,
    ) -> MetricDefinition | None:
        async with self._lock:
            candidates = [
                value
                for key, value in self._definitions.items()
                if key[0] == namespace and key[1] == name
            ]
            if not candidates:
                return None
            return max(candidates, key=lambda value: value.revision)

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None:
        pending: dict[str, _ObservationRecord] = {}
        for observation in observations:
            identity = observation_digest(namespace, observation.observation_id)
            payload = observation_payload_digest(namespace, observation)
            candidate = _ObservationRecord(
                namespace, observation.observation_id, payload,
                observation.kind, observation.occurred_at, observation,
            )
            existing = pending.get(identity)
            if existing is not None and existing.payload_digest != payload:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            pending[identity] = candidate
        async with self._lock:
            for identity, candidate in pending.items():
                current = self._observations.get(identity)
                if current is not None:
                    if (current.namespace, current.observation_id) != (
                        namespace, candidate.observation_id,
                    ):
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    if current.payload_digest != candidate.payload_digest:
                        raise AIError(ErrorCode.STORAGE_CONFLICT)
            if any(
                self._observations[identity].pruned_at is not None
                for identity in pending if identity in self._observations
            ):
                raise _observation_pruned()
            for identity, candidate in pending.items():
                self._observations.setdefault(identity, candidate)

    async def get_observation(
        self,
        namespace: str,
        observation_id: str,
    ) -> Observation | None:
        identity = observation_digest(namespace, observation_id)
        async with self._lock:
            current = self._observations.get(identity)
            if current is None:
                return None
            if (current.namespace, current.observation_id) != (namespace, observation_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current.observation

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
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise AIError(ErrorCode.PAGE_LIMIT_INVALID)
        after = None if cursor is None else _parse_scan_cursor(cursor)
        async with self._lock:
            values = [
                value.observation
                for value in self._observations.values()
                if value.namespace == namespace
                and value.pruned_at is None
                and value.observation is not None
                and value.kind == kind
                and start <= value.occurred_at < end
            ]
        values.sort(
            key=lambda item: (
                item.occurred_at,
                observation_digest(namespace, item.observation_id),
            )
        )
        if after is not None:
            values = [
                item
                for item in values
                if (
                    item.occurred_at,
                    observation_digest(namespace, item.observation_id),
                )
                > after
            ]
        selected = tuple(values[: limit + 1])
        has_more = len(selected) > limit
        items = selected[:limit]
        next_cursor = None
        if has_more and items:
            last = items[-1]
            next_cursor = _scan_cursor(
                last.occurred_at,
                observation_digest(namespace, last.observation_id),
            )
        return Page(items, next_cursor)

    async def prune_observations(self, namespace: str, *, before: datetime) -> int:
        async with self._lock:
            keys = [
                key
                for key, value in self._observations.items()
                if value.namespace == namespace and value.pruned_at is None
                and value.occurred_at < before
            ]
            pruned_at = datetime.now(timezone.utc)
            for key in keys:
                self._observations[key] = replace(
                    self._observations[key], observation=None, pruned_at=pruned_at,
                )
            return len(keys)


__all__ = ["InMemoryMetricStore"]
