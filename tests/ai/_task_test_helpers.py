#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Task test admission helpers."""

from linktools.ai.core import Principal
from linktools.ai.runtime import RuntimeState
from linktools.ai.task import (
    TaskGraph,
    TaskGraphAdmission,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphView,
)


async def admit_graph(
    state: RuntimeState,
    graph: TaskGraph,
    *,
    tenant_id: str = "tenant",
) -> TaskGraphView:
    request = TaskGraphRequest(
        graph,
        Principal("task-test", tenant_id),
        f"test:{graph.graph_id}",
        TaskGraphLimits(),
    )
    return await state.task.admissions.admit(
        TaskGraphAdmission.from_request(request),
        graph,
    )
