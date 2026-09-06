#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime root context."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Generic, TypeVar

from ..core import (
    CorrelationData,
    normalize_correlation,
    overlay_correlation,
    validate_tenant_id,
)

AppT = TypeVar("AppT")
_METRIC_DIMENSIONS_MAX = 8
_METRIC_DIMENSION_KEY_MAX = 64
_METRIC_DIMENSION_VALUE_MAX = 128
_METRIC_DIMENSION_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
_RESERVED_METRIC_DIMENSION_PREFIXES = ("linktools.", "context.")
_HIGH_CARDINALITY_METRIC_DIMENSIONS = frozenset(
    {
        "attempt_index",
        "execution_id",
        "fence",
        "graph_id",
        "node_id",
        "principal_id",
        "request_id",
        "resource_id",
        "run_id",
        "session_id",
        "span_id",
        "step_run_id",
        "tenant_id",
        "tool_call_id",
        "trace_id",
        "user_id",
    }
)


def _normalize_metric_dimensions(value: Mapping[str, str]) -> Mapping[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("metric_dimensions must be a mapping")
    if len(value) > _METRIC_DIMENSIONS_MAX:
        raise ValueError("metric_dimensions contains too many entries")
    normalized: dict[str, str] = {}
    for key, item in value.items():
        if (
            not isinstance(key, str)
            or len(key) > _METRIC_DIMENSION_KEY_MAX
            or _METRIC_DIMENSION_KEY_RE.fullmatch(key) is None
            or key.startswith(_RESERVED_METRIC_DIMENSION_PREFIXES)
            or key in _HIGH_CARDINALITY_METRIC_DIMENSIONS
        ):
            raise ValueError("metric dimension key is invalid")
        if (
            not isinstance(item, str)
            or not item
            or item != item.strip()
            or len(item) > _METRIC_DIMENSION_VALUE_MAX
            or any(unicodedata.category(character) == "Cc" for character in item)
        ):
            raise ValueError("metric dimension value is invalid")
        try:
            item.encode("utf-8")
        except UnicodeEncodeError as error:
            raise ValueError("metric dimension value is invalid") from error
        normalized[key] = item
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class RuntimeContext(Generic[AppT]):
    """Runtime-lifetime application, tenant, correlation, and metric metadata."""

    app: AppT
    tenant_id: str = "default"
    correlation: CorrelationData = field(default_factory=dict)
    metric_dimensions: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", validate_tenant_id(self.tenant_id))
        object.__setattr__(
            self,
            "correlation",
            normalize_correlation(self.correlation),
        )
        object.__setattr__(
            self,
            "metric_dimensions",
            _normalize_metric_dimensions(self.metric_dimensions),
        )

    def overlay(
        self,
        correlation: "Mapping[str, object] | None",
    ) -> RuntimeContext[AppT]:
        return RuntimeContext(
            app=self.app,
            tenant_id=self.tenant_id,
            correlation=overlay_correlation(self.correlation, correlation),
            metric_dimensions=self.metric_dimensions,
        )


__all__ = ["RuntimeContext"]
