#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Per-execution application context exposed to LinkTools tools."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from ..core import (
    CorrelationData,
    ImmutableJsonMapping,
    JsonValue,
    Principal,
    normalize_correlation,
    validate_memory_scope,
)
from ..workspace import Workspace

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class AgentContext(Generic[AppT]):
    app: AppT
    principal: Principal
    workspace: Workspace
    session_id: "str | None"
    execution_id: str
    session_metadata: Mapping[str, JsonValue]
    memory_scope: "str | None" = None
    correlation: CorrelationData = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.principal, Principal):
            raise TypeError("principal must be Principal")
        if not isinstance(self.workspace, Workspace):
            raise TypeError("workspace must be Workspace")
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise ValueError("session_id must be a non-empty string or None")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if self.memory_scope is not None:
            object.__setattr__(
                self,
                "memory_scope",
                validate_memory_scope(self.memory_scope),
            )
        object.__setattr__(
            self,
            "session_metadata",
            ImmutableJsonMapping(dict(self.session_metadata)),
        )
        object.__setattr__(
            self,
            "correlation",
            normalize_correlation(self.correlation),
        )


__all__ = ["AgentContext"]
