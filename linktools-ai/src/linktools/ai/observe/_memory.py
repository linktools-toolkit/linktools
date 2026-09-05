#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""In-memory MetricStore."""

from __future__ import annotations

import asyncio
from datetime import datetime

from ..core import Page
from ..errors import AIError, ErrorCode
from ._codec import (
    definition_semantic_digest,
    observation_digest,
    observation_payload_digest,
)
from ._model import MetricDefinition, Observation
from ._store import _parse_scan_cursor, _scan_cursor


class InMemoryMetricStore:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._definitions: dict[
            tuple[str, str, int], tuple[str, MetricDefinition]
        ] = {}
        self._observations: dict[str, tuple[str, str, Observation]] = {}

    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> MetricDefinition:
        key = (namespace, definition.name, definition.revision)
        digest = definition_semantic_digest(definition)
        async with self._lock:
            current = self._definitions.get(key)
            if current is None:
                self._definitions[key] = (digest, definition)
                return definition
            if current[0] != digest:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            return current[1]

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int,
    ) -> MetricDefinition | None:
        async with self._lock:
            current = self._definitions.get((namespace, name, revision))
            return current[1] if current is not None else None

    async def latest_definition(
        self,
        namespace: str,
        name: str,
    ) -> MetricDefinition | None:
        async with self._lock:
            candidates = [
                value[1]
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
        pending: dict[str, tuple[str, str, Observation]] = {}
        for observation in observations:
            identity = observation_digest(namespace, observation.observation_id)
            payload = observation_payload_digest(namespace, observation)
            candidate = (namespace, payload, observation)
            existing = pending.get(identity)
            if existing is not None and existing[1] != payload:
                raise AIError(ErrorCode.STORAGE_CONFLICT)
            pending[identity] = candidate
        async with self._lock:
            for identity, candidate in pending.items():
                current = self._observations.get(identity)
                if current is not None and current[1] != candidate[1]:
                    raise AIError(ErrorCode.STORAGE_CONFLICT)
            for identity, candidate in pending.items():
                self._observations.setdefault(identity, candidate)

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
                value[2]
                for value in self._observations.values()
                if value[0] == namespace
                and value[2].kind == kind
                and start <= value[2].occurred_at < end
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
                if value[0] == namespace and value[2].occurred_at < before
            ]
            for key in keys:
                del self._observations[key]
            return len(keys)


__all__ = ["InMemoryMetricStore"]
