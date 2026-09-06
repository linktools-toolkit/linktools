#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""RuntimeState lifecycle owner."""

import asyncio
from collections.abc import Awaitable, Callable
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from ...core import canonical_sha256, validate_persistence_namespace
from ...errors import AIError, ErrorCode
from ...storage import ObjectStore
from ._contracts import (
    ArtifactState,
    ConversationState,
    EvaluationState,
    ExecutionState,
    MemoryState,
    RecoveryState,
    TaskState,
)
from ._plan import (
    RuntimeDomain,
    RuntimeRetentionMode,
    RuntimeStatePlan,
    RuntimeStateRoute,
)

if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncEngine

    from ._maintenance import RuntimeStorageMaintenance
    from ._materializer import _MaterializedRuntimeState, _RuntimeObjectRouter
    from ._retention import RuntimeRetentionController
    from ._steps import RuntimeStepStore


class _RuntimeStateLifecycle(str, Enum):
    __str__ = str.__str__
    __format__ = str.__format__
    NEW = "new"
    INITIALIZING = "initializing"
    READY = "ready"
    CLOSING = "closing"
    CLOSED = "closed"


class RuntimeState:
    """Own materialized domain states and every resource acquired for them."""

    def __init__(
        self,
        plan: RuntimeStatePlan,
        *,
        object_store: "ObjectStore | None" = None,
    ) -> None:
        _validate_state_configuration(plan, object_store)
        self._plan = plan
        self._external_object_store = object_store
        self._lifecycle = _RuntimeStateLifecycle.NEW
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._close_cursor = 0
        self._close_actions: tuple[Callable[[], Awaitable[None]], ...] = ()
        self._namespace: str | None = None
        self._tenant_id: str | None = None
        self._conversation: ConversationState | None = None
        self._execution: ExecutionState | None = None
        self._memory: MemoryState | None = None
        self._artifact: ArtifactState | None = None
        self._task: TaskState | None = None
        self._evaluation: EvaluationState | None = None
        self._recovery: RecoveryState | None = None
        self._objects: _RuntimeObjectRouter | None = None
        self._steps: RuntimeStepStore | None = None
        self._retention: RuntimeRetentionController | None = None
        self._maintenance: RuntimeStorageMaintenance | None = None
        self._handoff_contract_digest: str | None = None

    @classmethod
    def in_memory(cls) -> "RuntimeState":
        return cls(RuntimeStatePlan())

    @classmethod
    def filesystem(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        base = _normalize_path(path)
        return cls(
            RuntimeStatePlan(
                **{
                    domain.value: RuntimeStateRoute.filesystem(
                        base / domain.value,
                        transaction_root=base,
                    )
                    for domain in RuntimeDomain
                }
            ),
            object_store=object_store,
        )

    @classmethod
    def sqlite(
        cls,
        path: "str | Path",
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        route = RuntimeStateRoute.sqlite(path)
        return cls(
            RuntimeStatePlan(
                **{domain.value: route for domain in RuntimeDomain}
            ),
            object_store=object_store,
        )

    @classmethod
    def sql(
        cls,
        engine: "AsyncEngine",
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        route = RuntimeStateRoute.sql(engine)
        return cls(
            RuntimeStatePlan(
                **{domain.value: route for domain in RuntimeDomain}
            ),
            object_store=object_store,
        )

    @classmethod
    def from_plan(
        cls,
        plan: RuntimeStatePlan,
        *,
        object_store: "ObjectStore | None" = None,
    ) -> "RuntimeState":
        return cls(plan, object_store=object_store)

    @property
    def plan(self) -> RuntimeStatePlan:
        return self._plan

    @property
    def ready(self) -> bool:
        return self._lifecycle is _RuntimeStateLifecycle.READY

    @property
    def namespace(self) -> str:
        self._require_ready()
        if self._namespace is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._namespace

    @property
    def tenant_id(self) -> str:
        self._require_ready()
        if self._tenant_id is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._tenant_id

    @property
    def handoff_contract_digest(self) -> str:
        self._require_ready()
        if self._handoff_contract_digest is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._handoff_contract_digest

    @property
    def conversation(self) -> ConversationState:
        return self._require_state(self._conversation)

    @property
    def execution(self) -> ExecutionState:
        return self._require_state(self._execution)

    @property
    def memory(self) -> MemoryState:
        return self._require_state(self._memory)

    @property
    def artifact(self) -> ArtifactState:
        return self._require_state(self._artifact)

    @property
    def task(self) -> TaskState:
        return self._require_state(self._task)

    @property
    def evaluation(self) -> EvaluationState:
        return self._require_state(self._evaluation)

    @property
    def recovery(self) -> RecoveryState:
        return self._require_state(self._recovery)

    @property
    def steps(self) -> "RuntimeStepStore":
        return self._require_state(self._steps)

    @property
    def retention(self) -> "RuntimeRetentionController":
        return self._require_state(self._retention)

    @property
    def maintenance(self) -> "RuntimeStorageMaintenance":
        return self._require_state(self._maintenance)

    async def initialize(self, *, namespace: str, tenant_id: str) -> None:
        async with self._lock:
            if self._lifecycle is not _RuntimeStateLifecycle.NEW:
                raise AIError(
                    ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                    "RuntimeState must be NEW",
                )
            validate_persistence_namespace(namespace)
            if not tenant_id.strip():
                raise ValueError("tenant_id is required")
            self._lifecycle = _RuntimeStateLifecycle.INITIALIZING
            try:
                from ._materializer import materialize_runtime_state

                materialized = await materialize_runtime_state(
                    self._plan,
                    namespace=namespace,
                    tenant_id=tenant_id,
                    object_store=self._external_object_store,
                )
                self._assign_materialized(
                    materialized,
                    namespace,
                    tenant_id,
                )
                self._handoff_contract_digest = _handoff_digest(
                    self._plan,
                    self._external_object_store,
                )
                self._lifecycle = _RuntimeStateLifecycle.READY
            except BaseException:
                self._lifecycle = _RuntimeStateLifecycle.CLOSED
                raise

    async def close(self) -> None:
        async with self._lock:
            if self._lifecycle is _RuntimeStateLifecycle.NEW:
                self._lifecycle = _RuntimeStateLifecycle.CLOSED
                return
            if self._lifecycle is _RuntimeStateLifecycle.CLOSED:
                return
            self._lifecycle = _RuntimeStateLifecycle.CLOSING
            task = self._close_task
            if task is None or task.done():
                task = asyncio.create_task(
                    self._run_close_actions(),
                    name="linktools-runtime-state-close",
                )
                task.add_done_callback(self._consume_close_result)
                self._close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                self._consume_close_result(task)
            raise

    def _consume_close_result(self, task: "asyncio.Task[None]") -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            # A later close() resumes from _close_cursor and retries the failed action.
            pass

    async def _run_close_actions(self) -> None:
        while self._close_cursor < len(self._close_actions):
            action = self._close_actions[self._close_cursor]
            await action()
            self._close_cursor += 1
        self._lifecycle = _RuntimeStateLifecycle.CLOSED

    def _assign_materialized(
        self,
        value: "_MaterializedRuntimeState",
        namespace: str,
        tenant_id: str,
    ) -> None:
        self._conversation = value.conversation
        self._execution = value.execution
        self._memory = value.memory
        self._artifact = value.artifact
        self._task = value.task
        self._evaluation = value.evaluation
        self._recovery = value.recovery
        self._objects = value.objects
        self._steps = value.steps
        self._retention = value.retention
        self._maintenance = value.maintenance
        self._close_actions = value.close_actions
        self._namespace = namespace
        self._tenant_id = tenant_id

    def _require_ready(self) -> None:
        if self._lifecycle is not _RuntimeStateLifecycle.READY:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "RuntimeState is not ready",
            )

    def _require_state(self, value: object) -> object:
        self._require_ready()
        if value is None:
            raise AIError(
                ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                "RuntimeState is not ready",
            )
        return value

    def object_store(self, domain: RuntimeDomain) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._objects.object_store(domain)

    def _object_store(self, domain: RuntimeDomain) -> ObjectStore:
        return self.object_store(domain)

    def working_object_store(
        self,
        domain: RuntimeDomain,
        *,
        owner_scope: str,
    ) -> ObjectStore:
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._objects.working_object_store(
            domain,
            owner_scope=owner_scope,
        )

    def _working_object_store(
        self,
        domain: RuntimeDomain,
        *,
        owner_scope: str,
    ) -> ObjectStore:
        return self.working_object_store(domain, owner_scope=owner_scope)

    async def _release_object_scope(
        self,
        domain: RuntimeDomain,
        *,
        owner_scope: str,
    ) -> None:
        self._require_ready()
        if self._objects is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        await self._objects.release_object_scope(
            domain,
            owner_scope=owner_scope,
        )


def _validate_state_configuration(
    plan: RuntimeStatePlan,
    object_store: "ObjectStore | None",
) -> None:
    if not isinstance(plan, RuntimeStatePlan):
        raise TypeError("plan must be a RuntimeStatePlan")
    object_domains = {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
        RuntimeDomain.TASK,
    }
    if object_store is not None and not any(
        domain in object_domains
        and plan.route(domain).retention is RuntimeRetentionMode.DURABLE
        for domain in RuntimeDomain
    ):
        raise ValueError(
            "object_store requires at least one durable object-capable RuntimeDomain"
        )
    if (
        plan.route(RuntimeDomain.CONVERSATION).retention
        is RuntimeRetentionMode.DURABLE
    ):
        if (
            plan.route(RuntimeDomain.EXECUTION).retention
            is not RuntimeRetentionMode.DURABLE
        ):
            raise ValueError("durable conversation requires durable execution")
        if (
            plan.route(RuntimeDomain.RECOVERY).retention
            is not RuntimeRetentionMode.DURABLE
        ):
            raise ValueError("durable conversation requires durable recovery")
    if plan.route(RuntimeDomain.TASK).retention is RuntimeRetentionMode.DURABLE:
        if plan.route(RuntimeDomain.EXECUTION).retention is not RuntimeRetentionMode.DURABLE:
            raise ValueError("durable task requires durable execution")
        if plan.route(RuntimeDomain.RECOVERY).retention is not RuntimeRetentionMode.DURABLE:
            raise ValueError("durable task requires durable recovery")


def _normalize_path(value: "str | Path") -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise ValueError("RuntimeState path is required")
    return Path(value).expanduser().resolve(strict=False)


def _handoff_digest(
    plan: RuntimeStatePlan,
    object_store: "ObjectStore | None",
) -> str:
    object_domains = {
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.MEMORY,
        RuntimeDomain.ARTIFACT,
        RuntimeDomain.RECOVERY,
    }
    routes = {}
    for domain in (
        RuntimeDomain.CONVERSATION,
        RuntimeDomain.EXECUTION,
        RuntimeDomain.RECOVERY,
    ):
        route = plan.route(domain)
        routes[domain.value] = {
            "retention": route.retention.value,
            "route_kind": route.kind,
            "route_identity": route.route_identity,
            "object_store_id": (
                object_store.store_id
                if object_store is not None
                and domain in object_domains
                and route.retention is RuntimeRetentionMode.DURABLE
                else (
                    "builtin"
                    if route.retention is RuntimeRetentionMode.DURABLE
                    else (
                        "transient"
                        if route.retention is RuntimeRetentionMode.TRANSIENT
                        else "memory"
                    )
                )
            ),
        }
    return canonical_sha256({"version": 6, **routes})


__all__ = ["RuntimeState"]
