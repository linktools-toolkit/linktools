#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime root context."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..core import RunContextData, normalize_run_context

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class RuntimeContext(Generic[AppT]):
    """Combine process-local application dependencies with portable Runtime defaults."""

    app: AppT
    values: RunContextData = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", normalize_run_context(self.values))

    def overlay(self, values: "Mapping[str, object] | None") -> RunContextData:
        from ..core import overlay_run_context

        return overlay_run_context(self.values, values)


__all__ = ["RuntimeContext"]
