#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Server-owned execution profile selection."""

from dataclasses import dataclass

from ...domain.agent import AgentRelease
from ...domain.execution import ExecutionProfile, ExecutionRequest
from ...foundation.errors import ErrorCode, LinktoolsAIError


@dataclass(frozen=True, slots=True)
class ProfilePolicy:
    sandbox_available: bool = False
    default_profile: ExecutionProfile = ExecutionProfile.PRODUCTION_SERVICE

    def select(self, request: ExecutionRequest, release: AgentRelease) -> ExecutionProfile:
        profile = request.requested_profile or self.default_profile
        if profile is ExecutionProfile.PRODUCTION_SANDBOXED and not self.sandbox_available:
            raise LinktoolsAIError(ErrorCode.CAPABILITY_DISABLED_FOR_PROFILE, "sandbox capability is not released")
        allowed = {item.value for item in release.allowed_profiles}
        if profile.value not in allowed:
            raise LinktoolsAIError(ErrorCode.PROFILE_NOT_ALLOWED, "profile is not allowed by the release")
        return profile


__all__ = ["ProfilePolicy"]
