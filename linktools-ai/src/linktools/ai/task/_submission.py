#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepared TaskGraph identity and cancellation outcome."""

import re
from dataclasses import dataclass

from ..core import Principal, TaskStatus
from ._graph import TaskGraph, TaskGraphAdmission, TaskGraphResult


@dataclass(frozen=True, slots=True)
class TaskSubmissionRef:
    namespace: str
    tenant_id: str
    graph_id: str
    request_digest: str
    operation_id: str
    principal: Principal

    def __post_init__(self) -> None:
        if (
            not self.namespace
            or not self.graph_id
            or self.tenant_id != self.principal.tenant_id
            or re.fullmatch(r"[0-9a-f]{64}", self.request_digest) is None
            or re.fullmatch(r"[0-9a-f]{64}", self.operation_id) is None
        ):
            raise ValueError("task submission identity is invalid")


@dataclass(frozen=True, slots=True)
class TaskGraphSubmission:
    namespace: str
    admission: TaskGraphAdmission
    graph: TaskGraph

    def __post_init__(self) -> None:
        if not self.namespace:
            raise ValueError("task submission namespace is required")
        self.admission.launch()
        self.admission.validate_graph(self.graph)

    @property
    def ref(self) -> TaskSubmissionRef:
        return TaskSubmissionRef(
            self.namespace,
            self.admission.principal.tenant_id,
            self.admission.graph_id,
            self.admission.initial_request_digest,
            self.admission.operation_id,
            self.admission.principal,
        )


@dataclass(frozen=True, slots=True)
class TaskSubmissionResult:
    submission: TaskSubmissionRef
    admitted: bool
    result: TaskGraphResult


@dataclass(frozen=True, slots=True)
class TaskSubmissionCancellation:
    submission: TaskSubmissionRef
    admitted: bool
    status: TaskStatus


__all__ = [
    "TaskGraphSubmission",
    "TaskSubmissionCancellation",
    "TaskSubmissionRef",
    "TaskSubmissionResult",
]
