#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local CLI composition."""

from dataclasses import dataclass

from ..local.principal import trusted_local_principal
from ..runtime.services import ExecutionHandle, ExecutionRequest
from .services import EntryServices


@dataclass(frozen=True, slots=True)
class CliApplication:
    services: EntryServices

    async def run(self, prompt: str, *, principal_id: str = "local") -> ExecutionHandle:
        principal = trusted_local_principal(principal_id)
        return await self.services.runtime.execution.run(ExecutionRequest(prompt, principal))


__all__ = ["CliApplication"]
