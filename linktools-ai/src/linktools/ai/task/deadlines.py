#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure deadline policy shared by task waiting and durable recovery."""

from datetime import datetime, timedelta

from ..core import JsonValue, RunBudget
from ..errors import ErrorCode
from ._graph import TaskNode

__all__ = ["deadline_policy"]


def deadline_policy(
    node: TaskNode,
    started_at: datetime | None,
    budget_scope: "tuple[str, RunBudget] | None",
) -> tuple[datetime, ErrorCode, dict[str, JsonValue]] | None:
    deadline = None if budget_scope is None else budget_scope[1].deadline_at
    code = ErrorCode.EXECUTION_USAGE_LIMIT_EXCEEDED
    details: dict[str, JsonValue] = {
        "dimension": "deadline_at", "reason": "deadline_exceeded",
    }
    if budget_scope is not None:
        details["scope_id"] = budget_scope[0]
    if node.timeout_seconds is not None and started_at is not None:
        # The first claim and the admitted node are the durable owners. Recovery
        # derives the same absolute deadline without granting a fresh interval.
        node_deadline = started_at + timedelta(seconds=node.timeout_seconds)
        if deadline is None or node_deadline < deadline:
            deadline = node_deadline
            code = ErrorCode.EXECUTION_WAIT_TIMEOUT
            details = {"reason": "node_deadline_exceeded", "timeout_seconds": node.timeout_seconds}
    if deadline is None:
        return None
    details["deadline_at"] = deadline.isoformat()
    return deadline, code, details
