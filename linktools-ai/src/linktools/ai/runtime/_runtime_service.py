#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime composition boundary."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Protocol, TypeGuard

from linktools.core import environ
from pydantic import BaseModel

from ..agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    AgentDefinition,
)
from ..asset import AssetRepository
from ..capability import RuntimeCapability
from ..core import (
    JsonValue,
    Principal,
    PrincipalKind,
    SessionStatus,
    validate_agent_id,
    validate_idempotency_key,
    validate_memory_scope,
    validate_resource_id,
    validate_tenant_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..spec import AgentSpec
from ..task import (
    TaskGraph,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphResult,
    TaskNode,
)
from ._agent import AgentHandle
from .service_api import (
    ApprovalService,
    ArtifactService,
    CancelExecutionRequest,
    CancelExecutionResult,
    CreateSessionRequest,
    EvaluationHandle,
    EvaluationService,
    EventService,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ExecutionStreamEvent,
    ReplayEvaluationRequest,
    ResumeSessionRequest,
    RunEvaluationRequest,
    SessionService,
    SessionView,
    TaskService,
)

_logger = environ.get_logger("ai.runtime")
_AGENT_TASK_V1_FIELDS = frozenset(
    {"type", "version", "binding", "user_prompt", "planning", "thinking"}
)


class _LocalRuntimeCoordinatorPort(Protocol):
    async def run(
        self, binding_digest: str, request: ExecutionRequest
    ) -> ExecutionHandle: ...

    async def resume(
        self,
        agent_digest: str,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
    ) -> ExecutionHandle: ...

    def _stream_for_agent(
        self,
        agent_digest: str,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        principal = self._resolve_principal(principal)
        return self._stream_agent(
            agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )

    async def _stream_agent(
        self,
        agent_digest: str,
        user_prompt: str,
        *,
        output: "type[BaseModel] | None",
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        handle = await self._start_for_agent(
            agent_digest,
            user_prompt,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
            local_stream=self._local_coordinator is not None,
        )
        stream = (
            self.event.stream(handle.execution_id, principal=principal)
            if self._local_coordinator is None
            else self._local_coordinator.stream(handle.execution_id, principal=principal)
        )
        async for event in stream:
            yield event

    async def _create_session_for_agent(
        self,
        agent_digest: str,
        session_id: str,
        *,
        principal: "Principal | None",
        cwd: "str | None",
        metadata: "Mapping[str, JsonValue] | None",
    ) -> SessionView:
        principal = self._resolve_principal(principal)
        definition = self._catalog.definition(agent_digest)
        values = dict(metadata or {})
        if any(key.startswith("linktools.ai.") for key in values):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        values["linktools.ai.agent_id"] = definition.spec.id
        return await self.session.create(
            definition.digest,
            CreateSessionRequest(
                principal,
                session_id,
                secrets.token_urlsafe(32),
                cwd,
                values,
            ),
        )

    async def _run_evaluation_for_agent(
        self,
        agent_digest: str,
        request: RunEvaluationRequest,
        *,
        output: "type[BaseModel] | None",
    ) -> EvaluationHandle:
        binding = self._bind_agent(agent_digest, output=output)
        return await self.evaluation.run(binding.digest, binding.output_schema_fingerprint, request)

    async def _replay_evaluation_for_agent(
        self,
        agent_digest: str,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
        *,
        output: "type[BaseModel] | None",
    ) -> ExecutionHandle:
        binding = self._bind_agent(agent_digest, output=output)
        return await self.evaluation.replay(binding.digest, snapshot_id, request)

    async def run_graph(
        self,
        graph: TaskGraph,
        *,
        principal: "Principal | None" = None,
        idempotency_key: str,
        limits: "TaskGraphLimits | None" = None,
    ) -> TaskGraphResult:
        request = await self._admit_graph(
            graph,
            principal=principal,
            idempotency_key=idempotency_key,
            limits=limits,
        )
        return await self.task.run_graph(request)

    async def run_graph_and_wait(
        self,
        graph: TaskGraph,
        *,
        principal: "Principal | None" = None,
        idempotency_key: str,
        limits: "TaskGraphLimits | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> TaskGraphResult:
        request = await self._admit_graph(
            graph,
            principal=principal,
            idempotency_key=idempotency_key,
            limits=limits,
        )
        return await self.task.run_graph_and_wait(
            request,
            timeout_seconds=timeout_seconds,
        )

    async def _admit_graph(
        self,
        graph: TaskGraph,
        *,
        principal: "Principal | None",
        idempotency_key: str,
        limits: "TaskGraphLimits | None",
    ) -> TaskGraphRequest:
        self._ensure_open()
        principal = self._resolve_principal(principal)
        selected_limits = limits or TaskGraphLimits()
        validate_idempotency_key(idempotency_key)
        graph.validate_limits(selected_limits)
        admitted_nodes: list[TaskNode] = []
        agent_ids: set[str] = set()
        for node in graph.nodes:
            payload = node.input
            if frozenset(payload) != _AGENT_TASK_V1_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if payload.get("type") != "linktools.ai.agent" or payload.get("version") != 1:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            user_prompt = payload.get("user_prompt")
            planning = payload.get("planning")
            thinking = payload.get("thinking")
            if (
                not isinstance(user_prompt, str)
                or not isinstance(planning, bool)
                or not isinstance(thinking, bool)
            ):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                snapshot = AgentBindingSnapshot.from_payload(payload.get("binding"))
                binding = self._restore_binding(snapshot)
            except AIError as error:
                if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                    raise
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            validate_user_prompt(user_prompt)
            agent_ids.add(binding.definition.spec.id)
            admitted_nodes.append(
                TaskNode(
                    node.node_id,
                    node.dependencies,
                    input={
                        "type": "linktools.ai.agent",
                        "version": 1,
                        "binding": binding.snapshot.to_payload(),
                        "user_prompt": user_prompt,
                        "planning": planning,
                        "thinking": thinking,
                    },
                    budget_cost=node.budget_cost,
                )
            )
        admitted = TaskGraph(graph.graph_id, tuple(admitted_nodes))
        _logger.info(
            "agent task graph admitted: graph=%s tenant=%s agents=%s nodes=%s",
            graph.graph_id,
            principal.tenant_id,
            tuple(sorted(agent_ids)),
            len(admitted.nodes),
        )
        return TaskGraphRequest(admitted, principal, idempotency_key, selected_limits)

    async def _ensure_session(
        self,
        definition: AgentDefinition,
        session_id: str,
        principal: Principal,
    ) -> None:
        try:
            session = await self.session.get(session_id, principal=principal)
        except AIError as error:
            if error.code not in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
                raise
            try:
                await self.session.create(
                    definition.digest,
                    CreateSessionRequest(
                        principal,
                        session_id,
                        secrets.token_urlsafe(32),
                        None,
                        {"linktools.ai.agent_id": definition.spec.id},
                    ),
                )
                return
            except AIError as create_error:
                if create_error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
            session = await self.session.get(session_id, principal=principal)
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.agent_digest != definition.digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _resolve_principal(self, principal: "Principal | None") -> Principal:
        if principal is not None:
            return principal
        _logger.debug(
            "runtime default principal selected: tenant=%s principal=%s",
            self._default_principal.tenant_id,
            self._default_principal.principal_id,
        )
        return self._default_principal

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            if self._close_task is None:
                self._close_task = asyncio.create_task(
                    self._cleanup(),
                    name="linktools-runtime-close",
                )
            task = self._close_task
        cancelled = False
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            cancelled = True
            await asyncio.shield(task)
        except BaseException:
            async with self._close_lock:
                if self._close_task is task:
                    self._close_task = None
            raise
        if cancelled:
            raise asyncio.CancelledError

    async def _cleanup(self) -> None:
        if self._close_callback is not None:
            await self._close_callback()
        async with self._close_lock:
            self._closed = True


def _validate_memory_scope(value: "str | None") -> "str | None":
    if value is not None:
        validate_memory_scope(value)
    return value


__all__ = ["Runtime"]
