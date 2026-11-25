#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""TaskGraph submission identity and cancellation outcome."""

import re
from dataclasses import dataclass, field

from ..core import Principal, TaskStatus, validate_idempotency_key
from ..errors import AIError, ErrorCode
from ._graph import TaskGraph, TaskGraphAdmission, TaskGraphResult


@dataclass(frozen=True, slots=True)
class TaskSubmissionRef:
    namespace: str
    tenant_id: str
    graph_id: str
    request_digest: str
    operation_id: str = field(repr=False)
    principal: Principal

    def __post_init__(self) -> None:
        if (
            not self.namespace
            or not self.graph_id
            or self.tenant_id != self.principal.tenant_id
            or re.fullmatch(r"[0-9a-f]{64}", self.request_digest) is None
        ):
            raise ValueError("task submission identity is invalid")
        try:
            validate_idempotency_key(self.operation_id)
        except AIError as error:
            raise ValueError("task submission operation id is invalid") from error

    @classmethod
    def from_admission(cls, namespace: str, admission: TaskGraphAdmission) -> "TaskSubmissionRef":
        """Derive the retained identity from its admission owner."""
        return cls(
            namespace, admission.principal.tenant_id, admission.graph_id,
            admission.initial_request_digest, admission.operation_id, admission.principal,
        )


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
        return TaskSubmissionRef.from_admission(self.namespace, self.admission)


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
