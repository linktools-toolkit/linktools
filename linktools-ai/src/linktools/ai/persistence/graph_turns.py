#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Persistence-owned adapters for joining Graph execution to a canonical turn."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..core import OperationLedgerRecord
    from ..task import TaskLease
    from ._task_repository import TaskRepositoryImpl
    from .turns import SessionTurnKey, SqlSessionTurns
    from ._step_store import SqlStepStore


@dataclass(frozen=True, slots=True)
class GraphDeferredWait:
    """The exact grouped native checkpoint and task/session owners."""

    turns: "SqlSessionTurns"
    key: "SessionTurnKey"
    generation: int
    step_store: "SqlStepStore"
    step_conversation_id: str
    checkpoint_id: str
    task_repository: "TaskRepositoryImpl"
    task_lease: "TaskLease | None"
    max_bytes: int


@dataclass(frozen=True, slots=True)
class GraphCancelOwner:
    """Authorized cold cancellation proof; never a runnable Task lease."""

    node_id: str
    operation: "OperationLedgerRecord"

    @property
    def tenant_id(self) -> str:
        return self.operation.tenant_id


def graph_task_repository(repository: object) -> "TaskRepositoryImpl | None":
    """Select the native grouped Task adapter, not an arbitrary implementation."""
    from ._task_repository import TaskRepositoryImpl

    return repository if type(repository) is TaskRepositoryImpl else None


__all__ = ["GraphCancelOwner", "GraphDeferredWait", "graph_task_repository"]
