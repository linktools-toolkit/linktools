#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime root context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..core import RunContextData, normalize_run_context, overlay_run_context

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class RuntimeContext(Generic[AppT]):
    """Combine process-local application dependencies with portable Runtime defaults."""

    app: AppT
    values: RunContextData = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", normalize_run_context(self.values))

    def overlay(
        self,
        values: "Mapping[str, object] | None",
    ) -> RuntimeContext[AppT]:
        return RuntimeContext(self.app, overlay_run_context(self.values, values))


__all__ = ["RuntimeContext"]
