#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped observable context."""

from contextvars import ContextVar, Token
from dataclasses import dataclass

from ..core import (
    Principal,
    validate_principal_id,
    validate_resource_id,
    validate_tenant_id,
)
from ..errors import AIError


@dataclass(frozen=True, slots=True)
class ObservationContext:
    tenant_id: str
    principal_id: str
    execution_id: str
    session_id: "str | None"
    run_id: str
    agent_id: str
    parent_execution_id: "str | None" = None
    parent_run_id: "str | None" = None
    subagent_id: "str | None" = None
    depth: int = 0

    def __post_init__(self) -> None:
        try:
            validate_tenant_id(self.tenant_id)
            validate_principal_id(self.principal_id)
            validate_resource_id(self.execution_id)
        except AIError as error:
            raise ValueError("run context identity is invalid") from error
        if self.session_id is not None and (
            not isinstance(self.session_id, str) or not self.session_id.strip()
        ):
            raise ValueError("run context is incomplete")
        if (
            not self.run_id.strip()
            or not self.agent_id.strip()
            or self.depth < 0
        ):
            raise ValueError("run context is incomplete")


_current: ContextVar[ObservationContext | None] = ContextVar(
    "linktools_ai_run_context",
    default=None,
)


def set_context(context: ObservationContext) -> 'Token[ObservationContext | None]':
    return _current.set(context)


def reset_context(token: 'Token[ObservationContext | None]') -> None:
    _current.reset(token)


def current_context() -> 'ObservationContext | None':
    return _current.get()


def context_for(
    principal: Principal,
    execution_id: str,
    session_id: "str | None",
    run_id: str,
    agent_id: str,
) -> ObservationContext:
    return ObservationContext(principal.tenant_id, principal.principal_id, execution_id, session_id, run_id, agent_id)


__all__ = ["ObservationContext", "context_for", "current_context", "reset_context", "set_context"]
