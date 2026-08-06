#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Workspace and sandbox lifecycle values."""

from enum import StrEnum
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class WorkspaceDataStatus(StrEnum):
    """Workspace data states."""

    NONE = "NONE"
    IMPORTING = "IMPORTING"
    READY = "READY"
    CHECKPOINTING = "CHECKPOINTING"
    CHECKPOINTED = "CHECKPOINTED"
    RESTORING = "RESTORING"
    FINALIZED = "FINALIZED"
    LOST = "LOST"


class SandboxResourceStatus(StrEnum):
    """Remote sandbox resource states."""

    NONE = "NONE"
    PROVISIONING = "PROVISIONING"
    ACTIVE = "ACTIVE"
    DESTROYING = "DESTROYING"
    DESTROYED = "DESTROYED"
    LOST = "LOST"


class SandboxLimits(BaseModel):
    """Bounded resources and network policy for one Sandbox lease."""

    model_config = ConfigDict(frozen=True)

    cpu: float = Field(gt=0)
    memory_mb: int = Field(gt=0)
    disk_mb: int = Field(gt=0)
    wall_timeout_seconds: int = Field(gt=0)
    idle_timeout_seconds: int = Field(gt=0)
    block_network: bool = True
    allowed_domains: "tuple[str, ...]" = ()


class SandboxLease(BaseModel):
    """Tenant and execution-bound Sandbox resource lease."""

    model_config = ConfigDict(frozen=True)

    lease_id: str
    execution_id: str
    tenant_id: str
    status: SandboxResourceStatus = SandboxResourceStatus.NONE
    limits: SandboxLimits
    expires_at: "datetime | None" = None

    def can_use(self, tenant_id: str, execution_id: str) -> bool:
        """Return whether this lease belongs to the requested scope."""
        return self.tenant_id == tenant_id and self.execution_id == execution_id

    def is_expired(self, now: datetime) -> bool:
        """Return whether a caller-provided deadline has passed."""
        return self.expires_at is not None and now >= self.expires_at
