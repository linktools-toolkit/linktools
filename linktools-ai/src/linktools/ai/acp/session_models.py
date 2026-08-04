#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP session state models shared by lifecycle services."""

import asyncio
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from .mcp_resources import SessionMcpResources


CloseReason = Literal["client", "eof", "signal", "error"]


@dataclass(slots=True)
class ActiveAcpSession:
    record: "AcpSessionRecord"
    lock: asyncio.Lock
    active_execution_id: "str | None"
    mcp_resources: "SessionMcpResources"
    terminal_handles: "set[str]"
    pending_elicitation_ids: "set[str]"
    operation_epoch: int = 0
    operation: "SessionOperationToken | None" = None
    closing_requested: bool = False
    cleanup_required: bool = False
    close_task: "asyncio.Task[SessionCloseResult] | None" = None
    pending_permission: "PendingPermissionToken | None" = None
    pending_permission_task: "asyncio.Task[Any] | None" = None
    pending_elicitation_tasks: "dict[str, asyncio.Task[Any]]" = field(default_factory=dict)
    terminal_create_tasks: "set[asyncio.Task[Any]]" = field(default_factory=set)
    terminal_release_tasks: "dict[str, asyncio.Task[Any]]" = field(default_factory=dict)

    @property
    def closing(self) -> bool:
        """Return whether the session is closing or requires cleanup."""
        return self.closing_requested or self.cleanup_required


@dataclass(frozen=True, slots=True)
class PendingPermissionToken:
    session_id: str
    execution_id: str
    approval_id: str
    tool_call_id: str
    epoch: int


@dataclass(frozen=True, slots=True)
class SessionCloseFailure:
    resource_type: "Literal['operation', 'execution', 'permission', 'elicitation', 'terminal', 'mcp', 'persistence']"
    resource_id: "str | None"
    error_id: str


@dataclass(frozen=True, slots=True)
class SessionCloseResult:
    closed: bool
    failures: "tuple[SessionCloseFailure, ...]"


if TYPE_CHECKING:
    from .persistence import AcpSessionRecord
    from .session_state import SessionOperationToken


__all__ = [
    "ActiveAcpSession",
    "CloseReason",
    "PendingPermissionToken",
    "SessionCloseFailure",
    "SessionCloseResult",
]
