#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Task runner recovery for agent nodes with durable executions."""

from typing import Generic, TypeVar

from ..core import ExecutionStatus, canonical_sha256
from ..errors import AIError, ErrorCode
from ..task import TaskNodeInvocation, TaskNodeRunControl, TaskNodeRunResult
from ._planner import RuntimeTaskNodeRunner, _execution_failure

AppT = TypeVar("AppT")


class RecoveryRuntimeTaskNodeRunner(RuntimeTaskNodeRunner[AppT], Generic[AppT]):
    """Resume a recovery-required agent execution instead of replacing it."""

    async def run(
        self,
        invocation: TaskNodeInvocation,
        *,
        control: TaskNodeRunControl,
    ) -> TaskNodeRunResult:
        snapshot = await self._task_state.snapshot_graph(
            invocation.graph_id,
            tenant_id=invocation.principal.tenant_id,
        )
        if snapshot is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        state = next(
            (
                value
                for value in snapshot.node_states
                if value.node_id == invocation.node.node_id
            ),
            None,
        )
        if state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        execution_id = state.execution_id
        if execution_id is None:
            return await super().run(invocation, control=control)
        view = await self._execution.inspect(
            execution_id,
            principal=invocation.principal,
        )
        if view.status is not ExecutionStatus.RECOVERY_REQUIRED:
            return await super().run(invocation, control=control)
        await control.bind_execution(execution_id)
        await self._execution.recover(
            execution_id,
            principal=invocation.principal,
        )
        result = await self._execution.wait(
            execution_id,
            principal=invocation.principal,
        )
        if result.status is not ExecutionStatus.SUCCEEDED:
            raise _execution_failure(result)
        if result.output is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        digest = canonical_sha256(result.output)
        payload = await self._materialize_result(
            result.output,
            tenant_id=invocation.principal.tenant_id,
            graph_id=invocation.graph_id,
            node_id=invocation.node.node_id,
        )
        if payload.digest != digest:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return TaskNodeRunResult(digest, execution_id, payload)


__all__ = ["RecoveryRuntimeTaskNodeRunner"]
