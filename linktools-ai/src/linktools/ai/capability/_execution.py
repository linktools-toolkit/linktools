#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped execution management declaration."""

from pydantic_ai.capabilities import AbstractCapability

from ..core import CommitPolicy
from ..storage import CacheBackend, StreamBackend, TurnCoordinationBackend
from ._context import AgentContext


class ExecutionCapability(AbstractCapability[AgentContext[object]]):
    """Configure the required execution owner and its admission timing.

    A terminal preference is conditional on coordination and transaction eligibility.
    Coordination resources are borrowed independently of storage read caches.
    All modes retain shared admission coordination and mandatory safety writes.
    """

    id = "runtime.execution"

    def __init__(
        self, *, commit: CommitPolicy = CommitPolicy.sync(),
        coordination: CacheBackend | TurnCoordinationBackend | StreamBackend | None = None,
    ) -> None:
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

    @property
    def commit(self) -> CommitPolicy:
        return self._commit

    @property
    def coordination(self) -> CacheBackend | TurnCoordinationBackend | StreamBackend | None:
        return self._coordination


__all__ = ["ExecutionCapability"]
