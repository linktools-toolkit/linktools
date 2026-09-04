#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""MetricStore contract."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol

from ._model import MetricDefinition, Observation


class MetricStore(Protocol):
    async def put_definition(
        self,
        namespace: str,
        definition: MetricDefinition,
    ) -> None: ...

    async def get_definition(
        self,
        namespace: str,
        name: str,
        revision: int | None,
    ) -> MetricDefinition | None: ...

    async def put_observations(
        self,
        namespace: str,
        observations: tuple[Observation, ...],
    ) -> None: ...

    async def scan_observations(
        self,
        namespace: str,
        *,
        kind: str,
        start: datetime,
        end: datetime,
        limit: int,
    ) -> tuple[Observation, ...]: ...

    async def prune(self, namespace: str, *, before: datetime) -> int: ...


__all__ = ["MetricStore"]
