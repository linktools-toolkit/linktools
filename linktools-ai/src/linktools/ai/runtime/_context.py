#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime root context."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..core import (
    CorrelationData,
    normalize_correlation,
    overlay_correlation,
    validate_tenant_id,
)

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class RuntimeContext(Generic[AppT]):
    """Runtime-lifetime application, tenant, and default correlation metadata."""

    app: AppT
    tenant_id: str = "default"
    correlation: CorrelationData = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(
            self,
            "correlation",
            normalize_correlation(self.correlation),
        )

    def overlay(
        self,
        correlation: "Mapping[str, object] | None",
    ) -> RuntimeContext[AppT]:
        return RuntimeContext(
            self.app,
            self.tenant_id,
            overlay_correlation(self.correlation, correlation),
        )


__all__ = ["RuntimeContext"]
