#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task-domain metrics.

A minimal Protocol (counter / duration / gauge) plus two implementations:
``NoopTaskMetrics`` (the default -- observability is opt-in) and
``CountersTaskMetrics`` (records into in-memory dicts for tests / lightweight
export). Label cardinality is the caller's responsibility: only low-cardinality
dimensions (handler, status, failure_kind) are used -- never job_id / task_id /
user_id / raw error text (those would blow up a real metrics backend)."""

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class TaskMetrics(Protocol):
    async def inc_counter(
        self, name: str, *, labels: "Mapping[str, str] | None" = None
    ) -> None: ...

    async def observe_duration(
        self,
        name: str,
        value: float,
        *,
        labels: "Mapping[str, str] | None" = None,
    ) -> None: ...

    async def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: "Mapping[str, str] | None" = None,
    ) -> None: ...


class NoopTaskMetrics:
    """Default no-op metrics: observability is opt-in, so the core loop pays no
    overhead and needs no backend when metrics are off."""

    async def inc_counter(
        self, name: str, *, labels: "Mapping[str, str] | None" = None
    ) -> None:
        pass

    async def observe_duration(
        self,
        name: str,
        value: float,
        *,
        labels: "Mapping[str, str] | None" = None,
    ) -> None:
        pass

    async def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: "Mapping[str, str] | None" = None,
    ) -> None:
        pass


def _key(name: str, labels: "Mapping[str, str] | None") -> "tuple[str, tuple[tuple[str, str], ...]]":
    items = tuple(sorted(labels.items())) if labels else ()
    return name, items


class CountersTaskMetrics:
    """In-memory metrics: counters accumulate, durations collect samples, gauges
    keep the last value. Useful for tests and for a lightweight exporter."""

    def __init__(self) -> None:
        self._counters: "dict[tuple[str, tuple[tuple[str, str], ...]], float]" = {}
        self._durations: "dict[tuple[str, tuple[tuple[str, str], ...]], list[float]]" = {}
        self._gauges: "dict[tuple[str, tuple[tuple[str, str], ...]], float]" = {}

    async def inc_counter(
        self, name: str, *, labels: "Mapping[str, str] | None" = None
    ) -> None:
        key = _key(name, labels)
        self._counters[key] = self._counters.get(key, 0.0) + 1.0

    async def observe_duration(
        self,
        name: str,
        value: float,
        *,
        labels: "Mapping[str, str] | None" = None,
    ) -> None:
        self._durations.setdefault(_key(name, labels), []).append(value)

    async def set_gauge(
        self,
        name: str,
        value: float,
        *,
        labels: "Mapping[str, str] | None" = None,
    ) -> None:
        self._gauges[_key(name, labels)] = value

    def counter(self, name: str, *, labels: "Mapping[str, str] | None" = None) -> float:
        return sum(
            v for (n, _), v in self._counters.items() if n == name and _matches(_, labels)
        )

    def samples(self, name: str, *, labels: "Mapping[str, str] | None" = None) -> "list[float]":
        out: "list[float]" = []
        for (n, items), vals in self._durations.items():
            if n == name and _matches(items, labels):
                out.extend(vals)
        return out


def _matches(
    items: "tuple[tuple[str, str], ...]", labels: "Mapping[str, str] | None"
) -> bool:
    if labels is None:
        return True
    want = tuple(sorted(labels.items()))
    return all(k_v in items for k_v in want)


__all__: "list[str]" = [
    "TaskMetrics",
    "NoopTaskMetrics",
    "CountersTaskMetrics",
]
