#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pure Session state, revision and exclusive operation lease rules."""

from datetime import datetime, timedelta, timezone
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.errors import ErrorCode, LinktoolsAIError


class SessionStatus(StrEnum):
    OPEN = "OPEN"
    BUSY = "BUSY"
    CLOSING = "CLOSING"
    CLOSED = "CLOSED"
    CLEANUP_REQUIRED = "CLEANUP_REQUIRED"


class Session(BaseModel):
    """Immutable Session projection; callers replace it after each CAS."""

    model_config = ConfigDict(frozen=True)

    session_id: str
    owner_id: str
    project_id: str
    agent_id: str
    agent_revision: int = Field(ge=1)
    bundle_digest: "str | None" = None
    profile: str
    settings: "dict[str, object]" = Field(default_factory=dict)
    workspace_ref: "str | None" = None
    mcp_resource_refs: "tuple[str, ...]" = ()
    current_execution_id: "str | None" = None
    status: SessionStatus = SessionStatus.OPEN
    revision: int = Field(default=1, ge=1)

    def transition_to(self, target: SessionStatus) -> "Session":
        allowed = {
            SessionStatus.OPEN: {SessionStatus.BUSY, SessionStatus.CLOSING},
            SessionStatus.BUSY: {SessionStatus.OPEN, SessionStatus.CLOSING},
            SessionStatus.CLOSING: {SessionStatus.CLOSED, SessionStatus.CLEANUP_REQUIRED},
            SessionStatus.CLEANUP_REQUIRED: {SessionStatus.CLOSING, SessionStatus.CLOSED},
            SessionStatus.CLOSED: set(),
        }
        if target not in allowed[self.status]:
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT, f"invalid session transition: {self.status} -> {target}")
        return self.model_copy(update={"status": target, "revision": self.revision + 1})


class SessionLease(BaseModel):
    """Exclusive Session operation lease with a monotonically increasing fence."""

    model_config = ConfigDict(frozen=True)

    owner: "str | None" = None
    operation_id: "str | None" = None
    fence: int = 0
    expires_at: "datetime | None" = None

    def claim(self, owner: str, operation_id: str, now: datetime, duration: timedelta) -> "SessionLease":
        if self.is_active(now):
            if self.owner == owner and self.operation_id == operation_id:
                return self
            raise LinktoolsAIError(ErrorCode.SESSION_BUSY, "session has an active operation")
        return self.model_copy(update={"owner": owner, "operation_id": operation_id, "fence": self.fence + 1, "expires_at": now + duration})

    def renew(self, owner: str, fence: int, now: datetime, duration: timedelta) -> "SessionLease":
        self._assert(owner, fence, now)
        return self.model_copy(update={"expires_at": now + duration})

    def release(self, owner: str, fence: int, now: datetime) -> "SessionLease":
        self._assert(owner, fence, now)
        return self.model_copy(update={"owner": None, "operation_id": None, "expires_at": None})

    def is_active(self, now: datetime) -> bool:
        if self.owner is None or self.expires_at is None:
            return False
        expiry = self.expires_at if self.expires_at.tzinfo else self.expires_at.replace(tzinfo=timezone.utc)
        current = now if now.tzinfo else now.replace(tzinfo=timezone.utc)
        return expiry > current

    def _assert(self, owner: str, fence: int, now: datetime) -> None:
        if self.owner != owner or self.fence != fence or not self.is_active(now):
            raise LinktoolsAIError(ErrorCode.SESSION_CONFLICT, "session lease is stale")


__all__ = ["Session", "SessionLease", "SessionStatus"]
