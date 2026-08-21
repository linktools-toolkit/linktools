#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime composition boundary."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from typing import Protocol

from linktools.core import environ
from pydantic import BaseModel

from ..agent import AgentBindingSnapshot, AgentCompiler, AgentDefinition, AgentDefinitionCatalog
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
from ..task import TaskGraph, TaskGraphLimits, TaskGraphRequest, TaskGraphResult, TaskNode
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
_AGENT_TASK_V1_FIELDS = frozenset({"type", "version", "agent_id", "user_prompt"})
_AGENT_TASK_V2_FIELDS = frozenset(
    {
        "type",
        "version",
        "agent_id",
        "binding_digest",
        "binding",
        "user_prompt",
        "planning",
        "thinking",
    }
)


class _LocalRuntimeCoordinatorPort(Protocol):
    async def run(self, binding_digest: str, request: ExecutionRequest) -> ExecutionHandle: ...

    async def resume(
        self,
        binding_digest: str,
        session_id: str,
        request: ResumeSessionRequest,
    ) -> ExecutionHandle: ...

    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]: ...


class Runtime:
    """Assemble and execute Agent specifications through one composed service graph."""

    def __init__(
        self,
        catalog: AgentDefinitionCatalog,
        compiler: AgentCompiler,
        assets: AssetRepository,
        execution: ExecutionService,
        session: SessionService,
        task: TaskService,
        evaluation: EvaluationService,
        approval: ApprovalService,
        event: EventService,
        artifact: ArtifactService,
        *,
        tenant_id: str = "default",
        close_callback: "Callable[[], Awaitable[None]] | None" = None,
        local_coordinator: "_LocalRuntimeCoordinatorPort | None" = None,
    ) -> None:
        if any(
            value is None
            for value in (
                catalog,
                compiler,
                assets,
                execution,
                session,
                task,
                evaluation,
                approval,
                event,
                artifact,
            )
        ):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._catalog = catalog
        self._compiler = compiler
        self._assets = assets
        self.execution = execution
        self.session = session
        self.task = task
        self.evaluation = evaluation
        self.approval = approval
        self.event = event
        self.artifact = artifact
        self._tenant_id = validate_tenant_id(tenant_id)
        self._default_principal = Principal(
            principal_id="runtime",
            tenant_id=self._tenant_id,
            kind=PrincipalKind.LOCAL_TRUSTED.value,
        )
        self._close_callback = close_callback
        self._local_coordinator = local_coordinator
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: "asyncio.Task[None] | None" = None

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def default_principal(self) -> Principal:
        return self._default_principal

    @property
    def assets(self) -> AssetRepository:
        return self._assets

    def agent(
        self,
        agent: "str | AgentSpec" = "default",
        *,
        capabilities: "Sequence[RuntimeCapability]" = (),
        output: "type[BaseModel] | None" = None,
        planning: bool = False,
        thinking: bool = False,
    ) -> AgentHandle:
        self._ensure_open()
        if not isinstance(planning, bool) or not isinstance(thinking, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if any(not isinstance(capability, RuntimeCapability) for capability in capabilities):
            raise TypeError("capabilities must contain RuntimeCapability values")
        if isinstance(agent, str):
            validate_agent_id(agent)
            base = self._catalog.root_definition(agent)
            if not capabilities and output is None:
                definition = base
            else:
                definition = self._compiler.compile(
                    base.spec,
                    capabilities=capabilities,
                    output=output,
                )
        elif isinstance(agent, AgentSpec):
            definition = self._compiler.compile(
                agent,
                capabilities=capabilities,
                output=output,
            )
        else:
            raise TypeError("agent must be an Agent id or AgentSpec")
        definition = self._catalog.register(definition)
        return AgentHandle(
            self,
            definition.spec.id,
            definition.digest,
            planning,
            thinking,
        )

    def _definition(self, binding_digest: str) -> AgentDefinition:
        self._ensure_open()
        return self._catalog.definition(binding_digest)

    def _restore_definition(self, snapshot: AgentBindingSnapshot) -> AgentDefinition:
        self._ensure_open()
        try:
            current = self._catalog.definition(snapshot.binding_digest)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
        else:
            if current.binding_snapshot != snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        restored = self._compiler.restore(snapshot)
        return self._catalog.register(restored)

    async def _compile_agent(self, agent_id: str) -> AgentDefinition:
        self._ensure_open()
        return self._catalog.root_definition(agent_id)

    async def start(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> ExecutionHandle:
        definition = self._catalog.root_definition("default")
        return await self._start_for_binding(
            definition.digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=False,
            thinking=False,
        )

    async def _start_for_binding(
        self,
        binding_digest: str,
        user_prompt: str,
        *,
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
        local_stream: bool = False,
    ) -> ExecutionHandle:
        self._ensure_open()
        principal = self._resolve_principal(principal)
        validate_user_prompt(user_prompt)
        requested = self._catalog.definition(binding_digest)
        request = ExecutionRequest(
            user_prompt=user_prompt,
            principal=principal,
            idempotency_key=idempotency_key or secrets.token_urlsafe(32),
            memory_scope=_validate_memory_scope(memory_scope),
            planning=planning,
            thinking=thinking,
        )
        if session_id is None:
            definition = requested
            if local_stream and self._local_coordinator is not None:
                handle = await self._local_coordinator.run(definition.digest, request)
            else:
                handle = await self.execution.run(definition.digest, request)
        else:
            if not session_id.strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            definition = await self._session_definition(
                session_id,
                principal,
                preferred=requested,
            )
            await self._ensure_session(definition, session_id, principal)
            resume_request = ResumeSessionRequest(
                principal=principal,
                user_prompt=user_prompt,
                idempotency_key=request.idempotency_key or "",
                memory_scope=request.memory_scope,
                planning=planning,
                thinking=thinking,
            )
            if local_stream and self._local_coordinator is not None:
                handle = await self._local_coordinator.resume(
                    definition.digest,
                    session_id,
                    resume_request,
                )
            else:
                handle = await self.session.resume(definition.digest, session_id, resume_request)
        _logger.info(
            "runtime execution admitted: execution=%s agent=%s session=%s planning=%s thinking=%s",
            handle.execution_id,
            definition.spec.id,
            session_id,
            planning,
            thinking,
        )
        return handle

    async def run(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        definition = self._catalog.root_definition("default")
        return await self._run_for_binding(
            definition.digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=False,
            thinking=False,
            timeout_seconds=timeout_seconds,
        )

    async def _run_for_binding(
        self,
        binding_digest: str,
        user_prompt: str,
        *,
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
        timeout_seconds: "float | None",
    ) -> ExecutionResult:
        principal = self._resolve_principal(principal)
        handle = await self._start_for_binding(
            binding_digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )
        return await self.execution.wait(
            handle.execution_id,
            principal=principal,
            timeout_seconds=timeout_seconds,
        )

    async def cancel(
        self,
        execution_id: str,
        *,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> CancelExecutionResult:
        self._ensure_open()
        principal = self._resolve_principal(principal)
        validate_resource_id(execution_id)
        result = await self.execution.cancel(
            execution_id,
            CancelExecutionRequest(
                principal,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                force,
            ),
        )
        _logger.info(
            "runtime execution cancellation requested: execution=%s cancelled=%s",
            execution_id,
            result.cancelled,
        )
        return result

    def stream(
        self,
        user_prompt: str,
        *,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        definition = self._catalog.root_definition("default")
        return self._stream_for_binding(
            definition.digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=False,
            thinking=False,
        )

    def _stream_for_binding(
        self,
        binding_digest: str,
        user_prompt: str,
        *,
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        principal = self._resolve_principal(principal)
        return self._stream(
            binding_digest,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            planning=planning,
            thinking=thinking,
        )

    async def _stream(
        self,
        binding_digest: str,
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        planning: bool,
        thinking: bool,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        handle = await self._start_for_binding(
            binding_digest,
            user_prompt,
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

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> SessionView:
        definition = self._catalog.root_definition("default")
        return await self._create_session_for_binding(
            definition.digest,
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
        )

    async def _create_session_for_binding(
        self,
        binding_digest: str,
        session_id: str,
        *,
        principal: "Principal | None",
        cwd: "str | None",
        metadata: "Mapping[str, JsonValue] | None",
    ) -> SessionView:
        principal = self._resolve_principal(principal)
        definition = self._catalog.definition(binding_digest)
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

    async def run_evaluation(self, request: RunEvaluationRequest) -> EvaluationHandle:
        definition = self._catalog.root_definition("default")
        return await self._run_evaluation_for_binding(definition.digest, request)

    async def _run_evaluation_for_binding(
        self,
        binding_digest: str,
        request: RunEvaluationRequest,
    ) -> EvaluationHandle:
        definition = self._catalog.definition(binding_digest)
        return await self.evaluation.run(
            definition.digest,
            definition.output_schema_fingerprint,
            request,
        )

    async def replay_evaluation(
        self,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
    ) -> ExecutionHandle:
        definition = self._catalog.root_definition("default")
        return await self._replay_evaluation_for_binding(definition.digest, snapshot_id, request)

    async def _replay_evaluation_for_binding(
        self,
        binding_digest: str,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
    ) -> ExecutionHandle:
        definition = self._catalog.definition(binding_digest)
        return await self.evaluation.replay(definition.digest, snapshot_id, request)

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
        return await self.task.run_graph_and_wait(request, timeout_seconds=timeout_seconds)

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
            if payload.get("type") != "linktools.ai.agent":
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            version = payload.get("version")
            if version == 1:
                if set(payload) != _AGENT_TASK_V1_FIELDS:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                agent_id = payload.get("agent_id")
                user_prompt = payload.get("user_prompt")
                if not isinstance(agent_id, str) or not isinstance(user_prompt, str):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                definition = self._catalog.root_definition(agent_id)
                planning = False
                thinking = False
            elif version == 2:
                if set(payload) != _AGENT_TASK_V2_FIELDS:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                agent_id = payload.get("agent_id")
                binding_digest = payload.get("binding_digest")
                binding = payload.get("binding")
                user_prompt = payload.get("user_prompt")
                planning = payload.get("planning")
                thinking = payload.get("thinking")
                if (
                    not isinstance(agent_id, str)
                    or not isinstance(binding_digest, str)
                    or not isinstance(user_prompt, str)
                    or not isinstance(planning, bool)
                    or not isinstance(thinking, bool)
                ):
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                try:
                    snapshot = AgentBindingSnapshot.from_payload(binding)
                except AIError as error:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
                if snapshot.binding_digest != binding_digest or snapshot.agent_spec.id != agent_id:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                definition = self._restore_definition(snapshot)
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            validate_agent_id(agent_id)
            validate_user_prompt(user_prompt)
            agent_ids.add(agent_id)
            admitted_nodes.append(
                TaskNode(
                    node.node_id,
                    node.dependencies,
                    input={
                        "type": "linktools.ai.agent",
                        "version": 2,
                        "agent_id": agent_id,
                        "binding_digest": definition.digest,
                        "binding": definition.binding_snapshot.to_payload(),
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
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.binding_digest != definition.digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)

    async def _session_definition(
        self,
        session_id: str,
        principal: Principal,
        *,
        preferred: AgentDefinition,
    ) -> AgentDefinition:
        try:
            session = await self.session.get(session_id, principal=principal)
        except AIError as error:
            if error.code not in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
                raise
            return preferred
        if session.binding_digest != preferred.digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
        return self._catalog.definition(session.binding_digest)

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
                self._close_task = asyncio.create_task(self._cleanup(), name="linktools-runtime-close")
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
