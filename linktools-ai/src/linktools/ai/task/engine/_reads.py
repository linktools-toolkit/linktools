#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Route mixed-task reads and retention to the execution's producing service."""
from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from ...core import BudgetUsage, ExecutionStatus, Principal
from ...errors import AIError, ErrorCode
from ...execution import ExecutionResult, ExecutionService, ExecutionView, TaskExecutionService
if TYPE_CHECKING:
    from ...runtime.agent_execution import AgentExecutionPort


class TaskExecutionReads:
    def __init__(self, tasks: TaskExecutionService, ports: Mapping[str, AgentExecutionPort[object]]) -> None:
        self._tasks = tasks
        self._ports = ports

    async def _service(self, execution_id: str, tenant_id: str) -> ExecutionService:
        if not self._ports:
            return self._tasks
        direct = self._ports.get(execution_id)
        if direct is not None:
            return direct.service
        record = await self._tasks.load_execution(execution_id, tenant_id=tenant_id)
        if record is None or record.binding_kind == "task":
            return self._tasks
        port = self._ports.get(execution_id) or self._ports.get(record.root_execution_id)
        if port is None:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED)
        return port.service

    async def _authorized(self, execution_id: str, principal: Principal) -> ExecutionService:
        if not self._ports:
            return self._tasks
        view = await self._tasks.inspect(execution_id, principal=principal)
        if view.binding_kind == "task":
            return self._tasks
        port = self._ports.get(execution_id) or self._ports.get(view.root_execution_id)
        if port is None:
            raise AIError(ErrorCode.BINDING_NOT_REGISTERED)
        return port.service

    async def inspect(self, execution_id: str, *, principal: Principal) -> ExecutionView:
        service = await self._authorized(execution_id, principal)
        return await service.inspect(execution_id, principal=principal)

    async def result(self, execution_id: str, *, principal: Principal) -> ExecutionResult:
        service = await self._authorized(execution_id, principal)
        return await service.result(execution_id, principal=principal)

    async def result_payload_size(self, execution_id: str, *, principal: Principal) -> int:
        service = await self._authorized(execution_id, principal)
        return await service.result_payload_size(execution_id, principal=principal)

    async def wait(self, execution_id: str, *, principal: Principal, timeout_seconds: float | None = None) -> ExecutionResult:
        service = await self._authorized(execution_id, principal)
        return await service.wait(execution_id, principal=principal, timeout_seconds=timeout_seconds)

    async def list_children(self, execution_id: str, *, principal: Principal) -> tuple[ExecutionView, ...]:
        service = await self._authorized(execution_id, principal)
        return await service.list_children(execution_id, principal=principal)

    async def budget_usage(self, execution_id: str, *, principal: Principal) -> BudgetUsage | None:
        service = await self._authorized(execution_id, principal)
        return await service.budget_usage(execution_id, principal=principal)

    async def acquire_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> bool:
        service = await self._service(execution_id, tenant_id)
        return await service.acquire_dependency_hold(execution_id, tenant_id=tenant_id, hold_id=hold_id)

    async def release_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> None:
        service = await self._service(execution_id, tenant_id)
        await service.release_dependency_hold(execution_id, tenant_id=tenant_id, hold_id=hold_id)

    async def release_settled_dependency_hold(self, execution_id: str, *, tenant_id: str, hold_id: str) -> None:
        """Repeat terminal Task cleanup without requiring an already-finished producer."""
        try:
            service = await self._service(execution_id, tenant_id)
        except AIError as error:
            if error.code is not ErrorCode.BINDING_NOT_REGISTERED:
                raise
            record = await self._tasks.load_source_execution(execution_id, tenant_id=tenant_id)
            if (record is not None and record.execution_id == execution_id
                    and record.binding_kind == "agent" and self._ports
                    and execution_id not in self._ports and record.root_execution_id not in self._ports
                    and record.status in {ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED}
                    and record.retention_closed and hold_id not in record.dependency_hold_ids):
                # Only the durable hold obligation is complete. This does not
                # substitute for producer-owned result reads or object cleanup.
                return
            raise
        await service.release_dependency_hold(execution_id, tenant_id=tenant_id, hold_id=hold_id)

    async def request_terminal_handoff(self, execution_id: str, *, tenant_id: str, terminal_commit: bool = False) -> None:
        service = await self._service(execution_id, tenant_id)
        await service.request_terminal_handoff(execution_id, tenant_id=tenant_id, terminal_commit=terminal_commit)
