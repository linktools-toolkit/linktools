#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Local CLI composition."""

from dataclasses import dataclass
from uuid import uuid4

from ..workspace import trusted_workspace_principal
from ..runtime import ExecutionHandle, ExecutionRequest
from ._assembly import AppServices


@dataclass(frozen=True, slots=True)
class CliApplication:
    services: AppServices

    async def run(self, prompt: str, *, binding_digest: str, memory_namespace: str, principal_id: str = "local") -> ExecutionHandle:
        principal = trusted_workspace_principal(principal_id)
        request = ExecutionRequest(prompt=prompt, principal=principal, idempotency_key=uuid4().hex, memory_namespace=memory_namespace)
        return await self.services.runtime_services.execution.run(binding_digest, request)


__all__ = ["CliApplication"]
