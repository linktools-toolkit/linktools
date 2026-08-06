#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution query and mutation API."""

from typing import Protocol

from ..core import Page, Principal
from .services import (
    CancelExecutionRequest,
    CancelExecutionResult,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionView,
    ForkExecutionRequest,
    RetryExecutionRequest,
    TraceItem,
    TranscriptItem,
)


class ExecutionQueryApi(Protocol):
    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView: ...
    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult: ...
    async def trace(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TraceItem]': ...
    async def transcript(self, execution_id: str, *, principal: Principal, cursor: 'str | None' = None, limit: int = 100) -> 'Page[TranscriptItem]': ...


class ExecutionApi(ExecutionQueryApi, Protocol):
    async def run(self, request: ExecutionRequest) -> ExecutionHandle: ...
    async def retry(self, execution_id: str, request: RetryExecutionRequest) -> ExecutionHandle: ...
    async def fork(self, execution_id: str, request: ForkExecutionRequest) -> ExecutionHandle: ...
    async def cancel(self, execution_id: str, request: CancelExecutionRequest) -> CancelExecutionResult: ...


__all__ = ["ExecutionApi", "ExecutionQueryApi"]
