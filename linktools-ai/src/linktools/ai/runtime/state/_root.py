#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeState lifecycle owner."""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import canonical_sha256, validate_persistence_namespace
from ...errors import AIError, ErrorCode
from ._contracts import (
    ArtifactState,
    ConversationState,
    EvaluationState,
    ExecutionState,
    MemoryState,
    RecoveryState,
    RuntimeDomain,
    RuntimeRetentionMode,
    TaskState,
)
from ._plan import RuntimeStatePlan, RuntimeStateRoute

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ...storage import ObjectStore
    from .._persistence import RuntimeDomainStates


class RuntimeState:
    """Own the materialized Runtime domain state and its resources."""

    def __init__(self, plan: RuntimeStatePlan, *, object_store: "ObjectStore | None" = None) -> None:
        self._plan = plan
        self._object_store = object_store
        self._lifecycle = "new"
        self._lock = asyncio.Lock()
        self._close_cursor = 0
        self._close_actions: "tuple[Callable[[], Awaitable[None]], ...]" = ()
        self._resources: "RuntimeDomainStates | None" = None
        self._steps: "object | None" = None
        self._retention: "object | None" = None
        self._handoff_contract_digest: "str | None" = None

    @classmethod
    def memory(cls) -> "RuntimeState":
        return cls(RuntimeStatePlan())

    @classmethod
    def filesystem(cls, path: "str | Path", *, object_store: "ObjectStore | None" = None) -> "RuntimeState":
        base = _normalize_path(path)
        return cls(
            RuntimeStatePlan(
                **{domain.value: RuntimeStateRoute.filesystem(base / domain.value) for domain in RuntimeDomain}
            ),
            object_store=object_store,
        )

    @classmethod
    def sqlite(cls, path: "str | Path", *, object_store: "ObjectStore | None" = None) -> "RuntimeState":
        route = RuntimeStateRoute.sqlite(path)
        return cls(RuntimeStatePlan(**{domain.value: route for domain in RuntimeDomain}), object_store=object_store)

    @classmethod
    def sql(cls, engine: "AsyncEngine", *, object_store: "ObjectStore | None" = None) -> "RuntimeState":
        route = RuntimeStateRoute.sql(engine)
        return cls(RuntimeStatePlan(**{domain.value: route for domain in RuntimeDomain}), object_store=object_store)

    @classmethod
    def from_plan(
        cls,
        plan: RuntimeStatePlan,
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        if not isinstance(plan, RuntimeStatePlan):
            raise TypeError("plan must be a RuntimeStatePlan")
        object_domains = {
            RuntimeDomain.CONVERSATION,
            RuntimeDomain.EXECUTION,
            RuntimeDomain.MEMORY,
            RuntimeDomain.ARTIFACT,
            RuntimeDomain.RECOVERY,
        }
        if object_store is not None and not any(
            domain in object_domains and plan.route(domain).retention is RuntimeRetentionMode.DURABLE
            for domain in RuntimeDomain
        ):
            raise ValueError("object_store requires at least one durable object-capable RuntimeDomain")
        return cls(plan, object_store=object_store)

    @property
    def plan(self) -> RuntimeStatePlan:
        return self._plan

    @property
    def lifecycle(self) -> str:
        return self._lifecycle

    @property
    def handoff_contract_digest(self) -> str:
        if self._handoff_contract_digest is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._handoff_contract_digest

    @property
    def resources(self) -> "RuntimeDomainStates":
        if self._resources is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._resources

    @property
    def conversation(self) -> ConversationState:
        return ConversationState(self.resources.conversation.sessions, self.resources.conversation.operations)

    @property
    def execution(self) -> ExecutionState:
        return ExecutionState(
            self.resources.execution.executions,
            self.resources.execution.events,
            self.resources.execution.idempotency,
            self.resources.execution.operations,
        )

    @property
    def memory(self) -> MemoryState:
        return MemoryState(self.resources.memory.records, self.resources.memory.operations)

    @property
    def artifact(self) -> ArtifactState:
        return ArtifactState(self.resources.artifact.records, self.resources.artifact.operations)

    @property
    def task(self) -> TaskState:
        return TaskState(self.resources.task.tasks, self.resources.task.operations)

    @property
    def evaluation(self) -> EvaluationState:
        return EvaluationState(
            self.resources.evaluation.records,
            self.resources.evaluation.idempotency,
            self.resources.evaluation.operations,
        )

    @property
    def recovery(self) -> RecoveryState:
        return RecoveryState(
            self.resources.recovery.approvals,
            self.resources.recovery.external_calls,
            self.resources.recovery.checkpoints,
            self.resources.recovery.operations,
            self.resources.recovery.tools,
        )

    @property
    def steps(self) -> object:
        if self._steps is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._steps

    @property
    def retention(self) -> object:
        if self._retention is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._retention

    async def initialize(self, *, namespace: str, tenant_id: str) -> None:
        async with self._lock:
            if self._lifecycle != "new":
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY, "RuntimeState must be NEW")
            validate_persistence_namespace(namespace)
            if not tenant_id.strip():
                raise ValueError("tenant_id is required")
            self._lifecycle = "initializing"
            try:
                from ._materializer import materialize_runtime_state

                resources, steps, retention, actions = await materialize_runtime_state(
                    self._plan,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    object_store=self._object_store,
                )
                self._resources = resources
                self._steps = steps
                self._retention = retention
                self._close_actions = actions
                self._handoff_contract_digest = _handoff_digest(self._plan, self._object_store)
                self._lifecycle = "ready"
            except BaseException:
                self._lifecycle = "closed"
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._lifecycle == "new":
                self._lifecycle = "closed"
                return
            if self._lifecycle == "closed":
                return
            self._lifecycle = "closing"
            while self._close_cursor < len(self._close_actions):
                action = self._close_actions[self._close_cursor]
                task = asyncio.create_task(action(), name="linktools-runtime-state-close")
                try:
                    await asyncio.shield(task)
                except asyncio.CancelledError:
                    await asyncio.shield(task)
                    raise
                except BaseException:
                    raise
                self._close_cursor += 1
            self._lifecycle = "closed"


def _normalize_path(value: "str | Path") -> Path:
    from pathlib import Path

    return Path(value).expanduser().resolve(strict=False)


def _handoff_digest(plan: RuntimeStatePlan, object_store: "ObjectStore | None") -> str:
    object_domains = {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
    }
    routes = {}
    for domain in (RuntimeDomain.CONVERSATION, RuntimeDomain.EXECUTION, RuntimeDomain.RECOVERY):
        route = plan.route(domain)
        routes[domain.value] = {
            "retention": route.retention.value,
            "route_kind": route.kind,
            "route_identity": route.route_identity,
            "object_store_id": object_store.store_id
            if object_store is not None and domain in object_domains and route.retention is RuntimeRetentionMode.DURABLE
            else "builtin" if route.retention is RuntimeRetentionMode.DURABLE else "transient" if route.retention is RuntimeRetentionMode.TRANSIENT else "memory",
        }
    return canonical_sha256({"version": 3, **routes})


__all__ = ["RuntimeState"]
