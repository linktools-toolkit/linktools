#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RunStore: the Protocol every Run persistence backend implements. transition()
is the ONLY way status changes -- callers never set record.status directly."""

from typing import Protocol, runtime_checkable

from .models import RunErrorInfo, RunRecord, RunResult, RunStatus


@runtime_checkable
class RunStore(Protocol):
    async def create(self, run: RunRecord) -> RunRecord: ...

    async def get(self, run_id: str) -> "RunRecord | None": ...

    async def transition(
        self,
        run_id: str,
        target: RunStatus,
        *,
        expected_version: int,
        result: "RunResult | None" = None,
        error: "RunErrorInfo | None" = None,
    ) -> RunRecord: ...

    async def list_children(self, run_id: str) -> "tuple[RunRecord, ...]": ...
