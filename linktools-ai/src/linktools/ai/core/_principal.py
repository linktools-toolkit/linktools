#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Principal lookup boundary."""

from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol, runtime_checkable

from ..errors import AIError, ErrorCode
from ._value import Principal, PrincipalKind, ResourceKind


class AuthorizationAction(StrEnum):
    EXECUTION_RUN = "execution.run"
    EXECUTION_READ = "execution.read"
    EXECUTION_CANCEL = "execution.cancel"
    SESSION_CREATE = "session.create"
    SESSION_READ = "session.read"
    SESSION_UPDATE = "session.update"
    SESSION_CLOSE = "session.close"
    TASK_RUN = "task.run"
    TASK_READ = "task.read"
    TASK_CANCEL = "task.cancel"
    EVALUATION_RUN = "evaluation.run"
    EVALUATION_READ = "evaluation.read"
    EVALUATION_COMPARE = "evaluation.compare"
    APPROVAL_READ = "approval.read"
    APPROVAL_DECIDE = "approval.decide"
    EXTERNAL_READ = "external.read"
    EXTERNAL_SUPPLY = "external.supply"
    EVENT_READ = "event.read"
    ARTIFACT_READ = "artifact.read"
    MEMORY_READ = "memory.read"
    MEMORY_WRITE = "memory.write"
    TOOL_EXECUTE = "tool.execute"


@dataclass(frozen=True, slots=True)
class ResourceRef:
    kind: ResourceKind
    id: str
    tenant_id: str
    owner_principal_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.tenant_id.strip():
            raise ValueError("resource reference is incomplete")


class AuthorizationPolicy(Protocol):
    async def authorize(
        self,
        principal: Principal,
        action: AuthorizationAction,
        resource: ResourceRef,
    ) -> None: ...


class TenantAuthorizationPolicy:
    """Default deny-by-tenant policy used by local and service composition roots."""

    async def authorize(
        self,
        principal: Principal,
        action: AuthorizationAction,
        resource: ResourceRef,
    ) -> None:
        if principal.tenant_id != resource.tenant_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if resource.owner_principal_id is not None and resource.owner_principal_id != principal.principal_id:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)
        if principal.kind == PrincipalKind.LOCAL_TRUSTED.value and action not in {
            AuthorizationAction.EXECUTION_RUN,
            AuthorizationAction.EXECUTION_READ,
            AuthorizationAction.EXECUTION_CANCEL,
            AuthorizationAction.SESSION_CREATE,
            AuthorizationAction.SESSION_READ,
            AuthorizationAction.SESSION_UPDATE,
            AuthorizationAction.SESSION_CLOSE,
            AuthorizationAction.TOOL_EXECUTE,
        }:
            raise AIError(ErrorCode.AUTHORIZATION_DENIED)


@runtime_checkable
class PrincipalProvider(Protocol):
    async def current(self) -> Principal: ...


__all__ = ["AuthorizationAction", "AuthorizationPolicy", "PrincipalProvider", "ResourceRef", "TenantAuthorizationPolicy"]
