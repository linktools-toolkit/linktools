#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped execution management declaration."""

from __future__ import annotations

from copy import copy
from typing import Any

from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import RunContext

from ..core import CommitPolicy
from ..errors import AIError, ErrorCode
from ..storage import CacheBackend, StreamBackend, TurnCoordinationBackend
from ._context import AgentContext


class ExecutionCapability(AbstractCapability[AgentContext[object]]):
    """Configure the required execution owner and its admission timing.

    A terminal preference is conditional on coordination and transaction eligibility.
    Coordination resources are borrowed independently of storage read caches.
    All modes retain shared admission coordination and mandatory safety writes.
    retain_history selects terminal continuation retention independently of sessions;
    the effective policy is captured at admission and bound afresh for each run.
    """

    id = "runtime.execution"
    _service: AbstractCapability[AgentContext[object]] | None = None

    def __init__(
        self, *, commit: CommitPolicy = CommitPolicy.sync(),
        retain_history: bool = False,
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> None:
        if not isinstance(retain_history, bool):
            raise TypeError("retain_history must be bool")
        if not isinstance(commit, CommitPolicy) or commit.mode not in {
            "sync",
            "terminal",
        }:
            raise ValueError("Execution requires sync or terminal commit")
        if coordination is not None and not isinstance(
            coordination, (CacheBackend, TurnCoordinationBackend, StreamBackend),
        ):
            raise TypeError("coordination must implement cache, turn coordination or stream operations")
        for names in (
            ("get", "put", "delete"),
            ("claim_turn", "read_turn", "renew_turn", "release_turn", "cancel_turn"),
            ("open_stream", "read_stream"),
        ):
            methods = tuple(getattr(coordination, name, None) for name in names)
            if any(method is not None for method in methods) and not all(callable(method) for method in methods):
                raise TypeError(f"coordination must implement {', '.join(names)} together")
        self._coordination = coordination
        self._commit = commit
        self._retain_history = retain_history

    @property
    def retain_history(self) -> bool:
        """Whether this execution retains its terminal continuation context."""
        return self._retain_history

    def bind(
        self, service: AbstractCapability[AgentContext[object]], *, retain_history: bool,
    ) -> ExecutionCapability:
        """Bind one run to its admitted retention promise without changing the declaration."""
        bound = copy(self)
        bound._service = service
        bound._retain_history = retain_history
        return bound

    def _bound_service(self) -> AbstractCapability[AgentContext[object]]:
        if self._service is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._service

    async def after_run(
        self, ctx: RunContext[AgentContext[object]], *, result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        if not self.retain_history:
            return result
        return await self._bound_service().after_run(ctx, result=result)

    async def on_run_error(
        self, ctx: RunContext[AgentContext[object]], *, error: BaseException,
    ) -> AgentRunResult[Any]:
        if not self.retain_history:
            raise error
        return await self._bound_service().on_run_error(ctx, error=error)

    @property
    def commit(self) -> CommitPolicy:
        return self._commit

    @property
    def coordination(self) -> CacheBackend | TurnCoordinationBackend | StreamBackend | None:
        return self._coordination


__all__ = ["ExecutionCapability"]
