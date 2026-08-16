#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime composition boundary."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

from linktools.core import environ

from ..agent import AgentCompiler, AgentDefinition
from ..core import (
    JsonValue,
    Principal,
    SessionStatus,
    validate_agent_id,
    validate_idempotency_key,
    validate_memory_scope,
    validate_resource_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
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
    ExecutionEvent,
    ExecutionHandle,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ReplayEvaluationRequest,
    RunEvaluationRequest,
    SessionService,
    SessionView,
    TaskService,
)

_logger = environ.get_logger("ai.runtime")
_AGENT_TASK_FIELDS = frozenset({"type", "version", "agent_id", "user_prompt"})


class Runtime:
    """Execute named Agent definitions through one composed service graph."""

    def __init__(
        self,
        compiler: AgentCompiler,
        execution: ExecutionService,
        session: SessionService,
        task: TaskService,
        evaluation: EvaluationService,
        approval: ApprovalService,
        event: EventService,
        artifact: ArtifactService,
        *,
        definitions: "dict[str, AgentDefinition] | None" = None,
        close_callback: "Callable[[], Awaitable[None]] | None" = None,
    ) -> None:
        if any(value is None for value in (compiler, execution, session, task, evaluation, approval, event, artifact)):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._compiler = compiler
        self.execution = execution
        self.session = session
        self.task = task
        self.evaluation = evaluation
        self.approval = approval
        self.event = event
        self.artifact = artifact
        self._definitions = {} if definitions is None else definitions
        self._close_callback = close_callback
        self._closed = False
        self._close_lock = asyncio.Lock()
        self._close_task: "asyncio.Task[None] | None" = None

    def agent(self, agent_id: str) -> AgentHandle:
        self._ensure_open()
        validate_agent_id(agent_id)
        return AgentHandle(self, agent_id)

    async def _compile_agent(self, agent_id: str) -> AgentDefinition:
        self._ensure_open()
        definition = await self._compiler.compile(agent_id=agent_id)
        _logger.debug("agent definition compiled for handle: agent=%s digest=%s", agent_id, definition.digest)
        return definition

    def _register_definition(self, definition: AgentDefinition) -> None:
        self._definitions[definition.digest] = definition
        _logger.debug(
            "agent definition registered: agent=%s digest=%s",
            definition.spec.id,
            definition.digest,
        )

    async def _compile_and_register(self, agent_id: str) -> AgentDefinition:
        definition = await self._compile_agent(agent_id)
        self._register_definition(definition)
        return definition

    async def start(
        self,
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> ExecutionHandle:
        return await self._start_for_agent(
            None,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )

    async def _start_for_agent(
        self,
        agent_id: "str | None",
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
    ) -> ExecutionHandle:
        self._ensure_open()
        validate_user_prompt(user_prompt)
        request = ExecutionRequest(
            user_prompt,
            principal,
            idempotency_key or secrets.token_urlsafe(32),
            _validate_memory_scope(memory_scope),
        )
        if session_id is None:
            definition = await self._compile_and_register(
                "default" if agent_id is None else agent_id
            )
            handle = await self.execution.run(definition.digest, request)
        else:
            if not session_id.strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            definition = await self._compile_session(
                session_id,
                principal,
                preferred_agent_id=agent_id,
            )
            if agent_id is not None and definition.spec.id != agent_id:
                raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)
            self._register_definition(definition)
            await self._ensure_session(definition, session_id, principal)
            from .service_api import ResumeSessionRequest

            handle = await self.session.resume(
                definition.digest,
                session_id,
                ResumeSessionRequest(
                    principal,
                    user_prompt,
                    request.idempotency_key or "",
                    request.memory_scope,
                ),
            )
        _logger.info(
            "runtime execution admitted: execution=%s agent=%s session=%s",
            handle.execution_id,
            definition.spec.id,
            session_id,
        )
        return handle

    async def run(
        self,
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        return await self._run_for_agent(
            None,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
            timeout_seconds=timeout_seconds,
        )

    async def cancel(
        self,
        execution_id: str,
        *,
        principal: Principal,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> CancelExecutionResult:
        """Explicitly request cancellation of an admitted execution."""
        self._ensure_open()
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

    async def _run_for_agent(
        self,
        agent_id: "str | None",
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        timeout_seconds: "float | None",
    ) -> ExecutionResult:
        handle = await self._start_for_agent(
            agent_id,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )
        return await self.execution.wait(
            handle.execution_id,
            principal=principal,
            timeout_seconds=timeout_seconds,
        )

    def stream(
        self,
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> AsyncIterator[ExecutionEvent]:
        return self._stream_for_agent(
            None,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )

    def _stream_for_agent(
        self,
        agent_id: "str | None",
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
    ) -> AsyncIterator[ExecutionEvent]:
        return self._stream(
            agent_id,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )

    async def _stream(
        self,
        agent_id: "str | None",
        user_prompt: str,
        *,
        principal: Principal,
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
    ) -> AsyncIterator[ExecutionEvent]:
        handle = await self._start_for_agent(
            agent_id,
            user_prompt,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )
        async for event in self.event.stream(handle.execution_id, principal=principal):
            yield event

    async def create_session(
        self,
        session_id: str,
        *,
        principal: Principal,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
    ) -> SessionView:
        return await self._create_session_for_agent(
            "default",
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
        )

    async def _create_session_for_agent(
        self,
        agent_id: str,
        session_id: str,
        *,
        principal: Principal,
        cwd: "str | None",
        metadata: "Mapping[str, JsonValue] | None",
    ) -> SessionView:
        definition = await self._compile_and_register(agent_id)
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
        return await self._run_evaluation_for_agent("default", request)

    async def _run_evaluation_for_agent(
        self,
        agent_id: str,
        request: RunEvaluationRequest,
    ) -> EvaluationHandle:
        definition = await self._compile_and_register(agent_id)
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
        return await self._replay_evaluation_for_agent("default", snapshot_id, request)

    async def _replay_evaluation_for_agent(
        self,
        agent_id: str,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
    ) -> ExecutionHandle:
        definition = await self._compile_and_register(agent_id)
        return await self.evaluation.replay(definition.digest, snapshot_id, request)

    async def run_graph(
        self,
        graph: TaskGraph,
        *,
        principal: Principal,
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
        principal: Principal,
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
        principal: Principal,
        idempotency_key: str,
        limits: "TaskGraphLimits | None",
    ) -> TaskGraphRequest:
        self._ensure_open()
        selected_limits = limits or TaskGraphLimits()
        validate_idempotency_key(idempotency_key)
        graph.validate_limits(selected_limits)
        logical: list[tuple[TaskNode, str, str]] = []
        agent_ids: set[str] = set()
        for node in graph.nodes:
            payload = node.input
            if set(payload) != _AGENT_TASK_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if payload["type"] != "linktools.ai.agent" or payload["version"] != 1:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            agent_id = payload["agent_id"]
            user_prompt = payload["user_prompt"]
            if not isinstance(agent_id, str) or not isinstance(user_prompt, str):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            validate_agent_id(agent_id)
            validate_user_prompt(user_prompt)
            agent_ids.add(agent_id)
            logical.append((node, agent_id, user_prompt))
        compiled = {
            agent_id: await self._compile_agent(agent_id)
            for agent_id in sorted(agent_ids)
        }
        for definition in compiled.values():
            self._register_definition(definition)
        admitted_nodes = tuple(
            TaskNode(
                node.node_id,
                node.dependencies,
                input={
                    "type": "linktools.ai.agent",
                    "version": 1,
                    "agent_id": agent_id,
                    "binding_digest": compiled[agent_id].digest,
                    "user_prompt": user_prompt,
                },
                budget_cost=node.budget_cost,
            )
            for node, agent_id, user_prompt in logical
        )
        admitted = TaskGraph(graph.graph_id, admitted_nodes)
        _logger.info(
            "agent task graph admitted: graph=%s tenant=%s agents=%s nodes=%s",
            graph.graph_id,
            principal.tenant_id,
            tuple(sorted(agent_ids)),
            len(admitted.nodes),
        )
        return TaskGraphRequest(
            admitted,
            principal,
            idempotency_key,
            selected_limits,
        )

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

    async def _compile_session(
        self,
        session_id: str,
        principal: Principal,
        *,
        preferred_agent_id: "str | None",
    ) -> AgentDefinition:
        try:
            session = await self.session.get(session_id, principal=principal)
        except AIError as error:
            if error.code not in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
                raise
            if preferred_agent_id is None:
                return await self._compile_agent("default")
            return await self._compile_agent(preferred_agent_id)
        agent_id = session.metadata.get("linktools.ai.agent_id")
        if not isinstance(agent_id, str) or not agent_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self._compile_agent(agent_id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

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
