#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped observable context."""

from contextvars import ContextVar, Token
from dataclasses import dataclass

from ..core import Principal


@dataclass(frozen=True, slots=True)
class RunContext:
    tenant_id: str
    principal_id: str
    execution_id: str
    session_id: str
    run_id: str
    agent_id: str
    parent_execution_id: "str | None" = None
    parent_run_id: "str | None" = None
    subagent_id: "str | None" = None
    depth: int = 0

    def __post_init__(self) -> None:
        if (
            not self.tenant_id.strip()
            or not self.principal_id.strip()
            or not self.execution_id.strip()
            or not self.session_id.strip()
            or not self.run_id.strip()
            or not self.agent_id.strip()
            or self.depth < 0
        ):
            raise ValueError("run context is incomplete")


_current: ContextVar[RunContext | None] = ContextVar("linktools_ai_run_context", default=None)


def set_context(context: RunContext) -> 'Token[RunContext | None]':
    return _current.set(context)


def reset_context(token: 'Token[RunContext | None]') -> None:
    _current.reset(token)


def current_context() -> 'RunContext | None':
    return _current.get()


def context_for(principal: Principal, execution_id: str, session_id: str, run_id: str, agent_id: str) -> RunContext:
    return RunContext(principal.tenant_id, principal.principal_id, execution_id, session_id, run_id, agent_id)


__all__ = ["RunContext", "context_for", "current_context", "reset_context", "set_context"]
