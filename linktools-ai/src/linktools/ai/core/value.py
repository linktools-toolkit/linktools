#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure values shared by every AI subsystem."""

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

from .paging import Page
from .errors import ErrorCode, LinktoolsAIError


class ExecutionProfile(StrEnum):
    PRODUCTION_SERVICE = "production-service"
    PRODUCTION_SANDBOXED = "production-sandboxed"
    LOCAL_CODING = "local-coding"


@dataclass(frozen=True, slots=True)
class Principal:
    principal_id: str
    tenant_id: str
    kind: str = "user"

    def __post_init__(self) -> None:
        if not self.principal_id.strip() or not self.tenant_id.strip() or not self.kind.strip():
            raise ValueError("principal is incomplete")


PrincipalKind: TypeAlias = str


def profile_available(profile: ExecutionProfile) -> bool:
    return profile is not ExecutionProfile.PRODUCTION_SANDBOXED


def require_profile_available(profile: ExecutionProfile) -> None:
    if not profile_available(profile):
        raise LinktoolsAIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "production-sandboxed is blocked by the Harness release")


__all__ = ["ExecutionProfile", "Page", "Principal", "PrincipalKind", "profile_available", "require_profile_available"]
