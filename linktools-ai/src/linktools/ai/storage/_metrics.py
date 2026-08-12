#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Process-local storage metrics owned by one composed runtime."""

from collections import Counter
from time import monotonic
from typing import Any


class StorageMetrics:
    """Collect the stable storage metric vocabulary without global state."""

    def __init__(self) -> None:
        self._counts: Counter[tuple[str, tuple[tuple[str, str], ...]]] = Counter()
        self._durations: dict[tuple[str, tuple[tuple[str, str], ...]], list[float]] = {}

    def count(self, name: str, **labels: str) -> None:
        self._counts[(name, tuple(sorted(labels.items())))] += 1

    def duration(self, name: str, elapsed: float, **labels: str) -> None:
        key = name, tuple(sorted(labels.items()))
        self._durations.setdefault(key, []).append(elapsed)

    def operation(self, domain: str, target: str, result: str, started_at: float) -> None:
        self.count("storage.operation.count", domain=domain, target=target, result=result)
        self.duration("storage.operation.duration", monotonic() - started_at, domain=domain, target=target)
        if result == "failure":
            self.count("storage.failure.count", domain=domain, target=target)

    def snapshot(self) -> dict[str, Any]:
        return {
            "counts": dict(self._counts),
            "durations": {key: tuple(values) for key, values in self._durations.items()},
        }


__all__ = ["StorageMetrics"]
