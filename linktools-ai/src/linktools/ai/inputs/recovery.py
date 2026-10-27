#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Read-only admission guard for cold Agent recovery in persisted task graphs.

No live TaskEngine is required: the admitted principal, invocation binding and
absolute deadline remain owned by the existing durable records.
"""

from datetime import datetime, timezone
from typing import TYPE_CHECKING

from ..core import OperationKind, OperationStatus, Principal, ResourceKind, TaskStatus
from ..errors import AIError, ErrorCode
from ..task.deadlines import deadline_policy

if TYPE_CHECKING:
    from ..execution import ExecutionRecord
    from ..persistence import Persistence
    from ._input_capture import InputCaptures

__all__ = ["TaskRecoveryGuard"]

_TERMINAL = frozenset({TaskStatus.SUCCEEDED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.BLOCKED})


class TaskRecoveryGuard:
    """Borrow durable readers; never dispatch work or own producer state."""

    def __init__(self, persistence: "Persistence", captures: "InputCaptures") -> None:
        self._persistence = persistence
        self._captures = captures

    async def __call__(self, execution: "ExecutionRecord") -> bool:
        ancestors: set[str] = set()
        while True:
            if execution.execution_id in ancestors:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            ancestors.add(execution.execution_id)
            if execution.requires_task_invocation_capture and not await self._allow_task(execution):
                return False
            if execution.parent_execution_id is None:
                return True
            parent = await self._persistence.execution.executions.get(
                execution.parent_execution_id, tenant_id=self._persistence.tenant_id,
            )
            if (parent is None or parent.execution_id != execution.parent_execution_id
                    or parent.root_execution_id != execution.root_execution_id):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            execution = parent

    async def _allow_task(self, execution: "ExecutionRecord") -> bool:
        tenant_id = self._persistence.tenant_id
        invocation = await self._captures.read_invocation_node(
            execution.execution_id,
            principal=Principal(execution.principal_id, tenant_id, execution.principal_kind),
        )
        if invocation is None:
            # An interrupted handoff cannot recover unknown policy by guessing.
            return False
        graph_id, captured_node = invocation
        admission = await self._persistence.task.admissions.get(graph_id, tenant_id=tenant_id)
        state = await self._persistence.task.tasks.graph_state(graph_id, tenant_id=tenant_id)
        if admission is None or state is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        node = next((item for item in state.nodes if item.node_id == captured_node.node_id), None)
        binding = next((item for item in state.node_states if item.node_id == captured_node.node_id), None)
        if (node != captured_node or binding is None or binding.execution_id != execution.execution_id
                or admission.principal.tenant_id != tenant_id
                or admission.principal.principal_id != execution.principal_id
                or admission.principal.kind != execution.principal_kind):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if state.status in _TERMINAL or binding.status in _TERMINAL:
            return False
        if await self._has_cancel_intent(graph_id, execution.execution_id):
            return False
        scope = None if admission.budget_scope_id is None else (admission.budget_scope_id, admission.budget)
        policy = deadline_policy(node, execution.started_at, scope)
        return policy is None or policy[0] > datetime.now(timezone.utc)

    async def _has_cancel_intent(self, graph_id: str, execution_id: str) -> bool:
        # Cancellation admission is durable before node cancellation settles.
        # Read the existing ledger, not a second producer-owned cancellation flag.
        cursor = None
        while True:
            page = await self._persistence.task.operations.list_pending(
                ResourceKind.TASK_GRAPH, graph_id,
                tenant_id=self._persistence.tenant_id, limit=128,
                states=frozenset({OperationStatus.PENDING, OperationStatus.RUNNING,
                                  OperationStatus.EFFECT_UNKNOWN}),
                after_sequence=cursor,
            )
            if any(item.operation_kind is OperationKind.TASK_CANCEL
                   and item.execution_id in {None, execution_id} for item in page):
                return True
            if len(page) < 128:
                return False
            cursor = page[-1].sequence
