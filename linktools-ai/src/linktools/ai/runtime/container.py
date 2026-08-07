#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime containers with explicit query/mutation separation."""

from dataclasses import dataclass

from ..agent.context import AgentBinding
from ..task.service import TaskApi, TaskQueryApi
from .approval import ApprovalApi
from .artifact import ArtifactApi
from .event import EventApi
from .evaluation import EvaluationApi, EvaluationQueryApi
from .execution import ExecutionApi, ExecutionQueryApi
from .session import SessionApi, SessionQueryApi
from .services import RuntimeServiceIdentity


@dataclass(frozen=True, slots=True)
class Runtime:
    service_identity: RuntimeServiceIdentity
    binding: AgentBinding
    execution: ExecutionApi
    session: SessionApi
    task: TaskApi
    evaluation: EvaluationApi
    approval: ApprovalApi
    event: EventApi
    artifact: ArtifactApi


@dataclass(frozen=True, slots=True)
class RuntimeAccess:
    service_identity: RuntimeServiceIdentity
    execution: ExecutionQueryApi
    session: SessionQueryApi
    task: TaskQueryApi
    evaluation: EvaluationQueryApi
    approval: ApprovalApi
    event: EventApi
    artifact: ArtifactApi


__all__ = ["Runtime", "RuntimeAccess"]
