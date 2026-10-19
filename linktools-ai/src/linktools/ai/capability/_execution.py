#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-scoped execution management declaration."""

from pydantic_ai.capabilities import AbstractCapability

from ..core import CommitPolicy
from ._context import AgentContext


class ExecutionCapability(AbstractCapability[AgentContext[object]]):
    """Configure the required execution owner and its admission timing.

    A terminal preference is conditional on cache and transaction eligibility.
    All modes retain shared admission coordination and mandatory safety writes.
    """

    id = "runtime.execution"

    def __init__(self, *, commit: CommitPolicy = CommitPolicy.sync()) -> None:
        if not isinstance(commit, CommitPolicy) or commit.mode not in {
            "sync",
            "terminal",
        }:
            raise ValueError("Execution requires sync or terminal commit")
        self._commit = commit

    @property
    def commit(self) -> CommitPolicy:
        return self._commit


__all__ = ["ExecutionCapability"]
