#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Sandbox resource and network policy validation."""

from ...domain.sandbox import SandboxLimits
from ...foundation.errors import ErrorCode, LinktoolsAIError


class SandboxPolicy:
    def validate(self, limits: SandboxLimits) -> SandboxLimits:
        if not limits.block_network and not limits.allowed_domains:
            raise LinktoolsAIError(ErrorCode.SANDBOX_UNAVAILABLE, "network allowlist is required")
        return limits


__all__ = ["SandboxPolicy"]
