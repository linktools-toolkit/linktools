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
        cancel_requested_at: "datetime | None" = None,
        cancel_requested_by: "str | None" = None,
        cancel_reason: "str | None" = None,
    ) -> RunRecord: ...

    async def list_children(self, run_id: str) -> "tuple[RunRecord, ...]": ...

    async def claim_execution(self, run_id: str, *, worker_id: str, execution_token: str) -> RunRecord: ...

    async def heartbeat_execution(self, run_id: str, *, worker_id: str, execution_token: str) -> RunRecord: ...
