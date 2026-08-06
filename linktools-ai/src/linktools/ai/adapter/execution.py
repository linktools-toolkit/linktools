#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution service adapter boundary."""

from typing import Protocol

from ..runtime.services import ExecutionHandle, ExecutionRequest


class ExecutionGateway(Protocol):
    async def start_execution(self, binding_digest: str, request: ExecutionRequest) -> ExecutionHandle: ...


__all__ = ["ExecutionGateway"]
