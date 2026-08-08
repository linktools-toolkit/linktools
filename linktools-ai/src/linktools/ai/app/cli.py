#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local CLI composition."""

from dataclasses import dataclass

from ..workspace import trusted_workspace_principal
from ..runtime.services import ExecutionHandle, ExecutionRequest
from .services import AppServices


@dataclass(frozen=True, slots=True)
class CliApplication:
    services: AppServices

    async def run(self, prompt: str, *, principal_id: str = "local") -> ExecutionHandle:
        principal = trusted_workspace_principal(principal_id)
        runtime = await self.services.runtime_factory.build_for_request(ExecutionRequest(prompt, principal))
        return await runtime.execution.run(ExecutionRequest(prompt, principal))


__all__ = ["CliApplication"]
