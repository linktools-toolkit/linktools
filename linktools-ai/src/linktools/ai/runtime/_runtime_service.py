#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime composition boundary and runtime-bound convenience behavior."""

import asyncio
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import TYPE_CHECKING, Generic, Protocol, TypeVar, overload

from linktools.core import environ
from pydantic import BaseModel

from ..agent import (
    AgentBinding,
    AgentBindingSnapshot,
    AgentCatalog,
    AgentCompiler,
    AgentDefinition,
)
from ..capability import CapabilityGroup
from ..core import (
    ExecutionMode,
    JsonValue,
    Page,
    Principal,
    PrincipalKind,
    SessionStatus,
    ThinkingValue,
    normalize_execution_mode,
    normalize_thinking,
    validate_agent_id,
    validate_idempotency_key,
    validate_memory_scope,
    validate_resource_id,
    validate_tenant_id,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..model import ModelRegistry
if TYPE_CHECKING:
    from ..observe import Middleware
from ..task import (
    TaskGraph,
    TaskGraphLimits,
    TaskGraphRequest,
    TaskGraphResult,
    TaskNode,
)
from ..workspace import Workspace
from ._agent import Agent, Execution, Session
from ._input import UserPromptTransport, user_prompt_transport
from .service_api import (
    ApprovalService,
    ArtifactService,
    CancelExecutionRequest,
    CancelExecutionResult,
    CloseSessionRequest,
    CreateSessionRequest,
    EvaluationHandle,
    EvaluationService,
    EventService,
    ExecutionHandle,
    ExecutionHistoryItem,
    ExecutionRequest,
    ExecutionResult,
    ExecutionService,
    ExecutionStreamEvent,
    ExecutionTraceItem,
    ForkExecutionRequest,
    ForkSessionRequest,
    ReplayEvaluationRequest,
    ResumeSessionRequest,
    RetryExecutionRequest,
    RunEvaluationRequest,
    SessionHistoryItem,
    SessionService,
    SessionView,
    TaskService,
    TranscriptItem,
    UpdateSessionRequest,
)
from .state import RuntimeState

_logger = environ.get_logger("ai.runtime")
AppT = TypeVar("AppT")
_AGENT_TASK_V1_FIELDS = frozenset(
    {
        "type",
        "version",
        "binding",
        "user_prompt",
        "user_prompt_codec",
        "mode",
        "planning",
        "thinking",
    }
)


class _LocalRuntimeCoordinatorPort(Protocol):
    def stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]: ...


class Runtime(Generic[AppT]):
    """Frozen Runtime composition and service graph."""

    def __init__(
        self,
        catalog: AgentCatalog,
        compiler: AgentCompiler,
        execution: ExecutionService,
        session: SessionService,
        task: TaskService,
        evaluation: EvaluationService,
        approval: ApprovalService,
        event: EventService,
        artifact: ArtifactService,
        *,
        workspace: Workspace,
        app: AppT,
        tenant_id: str = "default",
        close_callback: "Callable[[], Awaitable[None]] | None" = None,
        local_coordinator: "_LocalRuntimeCoordinatorPort | None" = None,
    ) -> None:
        if any(
            value is None
            for value in (
                catalog,
                compiler,
                execution,
                session,
                task,
                evaluation,
                approval,
                event,
                artifact,
                workspace,
            )
        ):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._catalog = catalog
        self._compiler = compiler
        self.execution = execution
        self.session = session
        self.task = task
        self.evaluation = evaluation
        self.approval = approval
        self.event = event
        self.artifact = artifact
        self._workspace = workspace
        self._app = app
        self._tenant_id = validate_tenant_id(tenant_id)
        self._default_principal = Principal(
            principal_id="runtime",
            tenant_id=self._tenant_id,
            kind=PrincipalKind.LOCAL_TRUSTED.value,
        )
        self._close_callback = close_callback
        self._local_coordinator = local_coordinator
        self._closed = False
        self._closing = False
        self._close_lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None

    @classmethod
    @overload
    def open(
        cls,
        workspace: Workspace,
        *,
        app: None = None,
        tenant_id: "str | None" = None,
        models: "ModelRegistry | None" = None,
        state: "RuntimeState | None" = None,
        capabilities: "Sequence[CapabilityGroup[None]]" = (),
        middleware: "Sequence[Middleware]" = (),
    ) -> "AbstractAsyncContextManager[Runtime[None]]": ...

    @classmethod
    @overload
    def open(
        cls,
        workspace: Workspace,
        *,
        app: AppT,
        tenant_id: "str | None" = None,
        models: "ModelRegistry | None" = None,
        state: "RuntimeState | None" = None,
        capabilities: "Sequence[CapabilityGroup[AppT]]" = (),
        middleware: "Sequence[Middleware]" = (),
    ) -> "AbstractAsyncContextManager[Runtime[AppT]]": ...

    @classmethod
    def open(
        cls,
        workspace: Workspace,
        *,
        app: object = None,
        tenant_id: "str | None" = None,
        models: "ModelRegistry | None" = None,
        state: "RuntimeState | None" = None,
        capabilities: "Sequence[CapabilityGroup[object]]" = (),
        middleware: "Sequence[Middleware]" = (),
    ) -> "AbstractAsyncContextManager[Runtime[object]]":
        return _open_runtime(
            workspace,
            app=app,
            tenant_id=tenant_id,
            models=models,
            state=state,
            capabilities=capabilities,
            middleware=middleware,
        )

    @property
    def tenant_id(self) -> str:
        return self._tenant_id

    @property
    def default_principal(self) -> Principal:
        return self._default_principal

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    @property
    def app(self) -> AppT:
        return self._app

    def agent(self, agent_id: str = "default") -> "Agent[AppT]":
        """Resolve one frozen root Agent by id."""
        self._ensure_open()
        validate_agent_id(agent_id)
        definition = self._catalog.root_definition(agent_id)
        return Agent(self, definition.spec.id, definition.digest)

    def _definition(self, agent_digest: str) -> AgentDefinition:
        self._ensure_open()
        return self._catalog.definition(agent_digest)

    def _bind_agent(
        self,
        agent_digest: str,
        *,
        output: "type[BaseModel] | None" = None,
    ) -> AgentBinding:
        self._ensure_open()
        definition = self._catalog.definition(agent_digest)
        return self._catalog.register_binding(
            self._compiler.bind(definition, output=output)
        )

    def _restore_binding(self, snapshot: AgentBindingSnapshot) -> AgentBinding:
        self._ensure_open()
        try:
            current = self._catalog.binding(snapshot.binding_digest)
        except AIError as error:
            if error.code is not ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                raise
        else:
            if current.snapshot != snapshot:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            return current
        restored = self._compiler.restore(snapshot)
        return self._catalog.register_binding(restored)

    async def _compile_agent(self, agent_id: str) -> AgentDefinition:
        self._ensure_open()
        return self._catalog.root_definition(agent_id)

    async def _start_for_agent(
        self,
        agent_digest: str,
        user_prompt: UserPromptTransport,
        *,
        output: "type[BaseModel] | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        mode: ExecutionMode,
        planning: "bool | None",
        thinking: "ThinkingValue | None",
    ) -> "Execution[AppT]":
        self._ensure_open()
        resolved_principal = self._resolve_principal(principal)
        validate_user_prompt(str(user_prompt))
        definition = self._catalog.definition(agent_digest)
        resolved_mode, resolved_planning, resolved_thinking = _execution_policy(
            definition,
            mode=mode,
            planning=planning,
            thinking=thinking,
        )
        binding = self._catalog.register_binding(
            self._compiler.bind(definition, output=output)
        )
        request = ExecutionRequest(
            user_prompt=str(user_prompt),
            user_prompt_codec=user_prompt.codec,
            principal=resolved_principal,
            idempotency_key=idempotency_key or secrets.token_urlsafe(32),
            memory_scope=_validate_memory_scope(memory_scope),
            mode=resolved_mode,
            planning=resolved_planning,
            thinking=resolved_thinking,
        )
        if session_id is None:
            handle = await self.execution.run(binding.digest, request)
        else:
            if not isinstance(session_id, str) or not session_id.strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            await self._ensure_session(definition, session_id, resolved_principal)
            resume_request = ResumeSessionRequest(
                principal=resolved_principal,
                user_prompt=request.user_prompt,
                user_prompt_codec=request.user_prompt_codec,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
                mode=request.mode,
                planning=request.planning,
                thinking=request.thinking,
            )
            handle = await self.session.resume(
                definition.spec.id,
                binding.digest,
                session_id,
                resume_request,
            )
        _logger.info(
            "runtime execution admitted: execution=%s agent=%s session=%s mode=%s planning=%s thinking=%s",
            handle.execution_id,
            definition.spec.id,
            session_id,
            resolved_mode,
            resolved_planning,
            resolved_thinking,
        )
        return Execution(self, handle.execution_id, binding.digest, resolved_principal)

    def _execution_stream(
        self,
        execution_id: str,
        *,
        principal: Principal,
        after_sequence: int = 0,
    ) -> AsyncIterator[ExecutionStreamEvent]:
        self._ensure_open()
        validate_resource_id(execution_id)
        if self._local_coordinator is not None:
            return self._local_coordinator.stream(
                execution_id,
                principal=principal,
                after_sequence=after_sequence,
            )
        return self.event.stream(
            execution_id,
            principal=principal,
            after_sequence=after_sequence,
        )

    async def _retry_execution(
        self,
        binding_digest: str,
        execution_id: str,
        user_prompt: UserPromptTransport,
        *,
        principal: Principal,
        idempotency_key: "str | None",
    ) -> "Execution[AppT]":
        self._ensure_open()
        request = RetryExecutionRequest(
            str(user_prompt),
            user_prompt.codec,
            principal,
            idempotency_key or secrets.token_urlsafe(32),
        )
        handle = await self.execution.retry(binding_digest, execution_id, request)
        return Execution(self, handle.execution_id, binding_digest, principal)

    async def _fork_execution(
        self,
        binding_digest: str,
        execution_id: str,
        user_prompt: UserPromptTransport,
        *,
        principal: Principal,
        idempotency_key: "str | None",
    ) -> "Execution[AppT]":
        self._ensure_open()
        request = ForkExecutionRequest(
            str(user_prompt),
            user_prompt.codec,
            principal,
            idempotency_key or secrets.token_urlsafe(32),
        )
        handle = await self.execution.fork(binding_digest, execution_id, request)
        return Execution(self, handle.execution_id, binding_digest, principal)

    async def cancel(
        self,
        execution_id: str,
        *,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        force: bool = False,
    ) -> CancelExecutionResult:
        self._ensure_open()
        resolved_principal = self._resolve_principal(principal)
        validate_resource_id(execution_id)
        return await self.execution.cancel(
            execution_id,
            CancelExecutionRequest(
                resolved_principal,
                idempotency_key or secrets.token_urlsafe(32),
                force,
            ),
        )

    async def _create_session_for_agent(
        self,
        agent_id: str,
        session_id: str,
        *,
        principal: "Principal | None",
        cwd: "str | None",
        metadata: "Mapping[str, JsonValue] | None",
        idempotency_key: "str | None",
    ) -> SessionView:
        resolved_principal = self._resolve_principal(principal)
        validate_agent_id(agent_id)
        values = dict(metadata or {})
        if any(key.startswith("linktools.ai.") for key in values):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return await self.session.create(
            agent_id,
            CreateSessionRequest(
                resolved_principal,
                session_id,
                idempotency_key or secrets.token_urlsafe(32),
                cwd,
                values,
            ),
        )

    async def _fork_session(
        self,
        agent_id: str,
        agent_digest: str,
        session_id: str,
        new_session_id: str,
        *,
        principal: "Principal | None",
        idempotency_key: "str | None",
        cwd: "str | None",
    ) -> "Session[AppT]":
        resolved_principal = self._resolve_principal(principal)
        await self.session.fork(
            agent_id,
            session_id,
            ForkSessionRequest(
                resolved_principal,
                new_session_id,
                idempotency_key or secrets.token_urlsafe(32),
                cwd,
            ),
        )
        return Session(self, agent_id, agent_digest, new_session_id, resolved_principal)

    async def _update_session(
        self,
        agent_id: str,
        session_id: str,
        *,
        expected_revision: int,
        metadata: Mapping[str, JsonValue],
        principal: "Principal | None",
        idempotency_key: "str | None",
        cwd: "str | None",
    ) -> SessionView:
        resolved_principal = self._resolve_principal(principal)
        return await self.session.update(
            agent_id,
            session_id,
            UpdateSessionRequest(
                resolved_principal,
                expected_revision,
                idempotency_key or secrets.token_urlsafe(32),
                metadata,
                cwd,
            ),
        )

    async def _close_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None",
        idempotency_key: "str | None",
        force: bool,
        wait_timeout_seconds: int,
    ) -> SessionView:
        resolved_principal = self._resolve_principal(principal)
        return await self.session.close(
            session_id,
            CloseSessionRequest(
                resolved_principal,
                idempotency_key or secrets.token_urlsafe(32),
                force,
                wait_timeout_seconds,
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
        return await self.evaluation.run(binding.digest, request)

    async def _replay_evaluation_for_agent(
        self,
        agent_digest: str,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
        *,
        output: "type[BaseModel] | None",
    ) -> "Execution[AppT]":
        binding = self._bind_agent(agent_digest, output=output)
        handle = await self.evaluation.replay(
            binding.digest,
            snapshot_id,
            request,
        )
        return Execution(self, handle.execution_id, binding.digest, request.principal)

    def _task_for_agent(
        self,
        agent_digest: str,
        node_id: str,
        user_prompt: UserPromptTransport,
        *,
        dependencies: tuple[str, ...],
        budget_cost: int,
        output: "type[BaseModel] | None",
        planning: "bool | None",
        thinking: "ThinkingValue | None",
    ) -> TaskNode:
        definition = self._definition(agent_digest)
        _mode, resolved_planning, resolved_thinking = _execution_policy(
            definition,
            mode="run",
            planning=planning,
            thinking=thinking,
        )
        binding = self._bind_agent(agent_digest, output=output)
        return TaskNode(
            node_id,
            dependencies,
            input={
                "type": "linktools.ai.agent",
                "version": 1,
                "binding": binding.snapshot.to_payload(),
                "user_prompt": str(user_prompt),
                "user_prompt_codec": user_prompt.codec,
                "mode": "run",
                "planning": resolved_planning,
                "thinking": resolved_thinking,
            },
            budget_cost=budget_cost,
        )

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
        resolved_principal = self._resolve_principal(principal)
        selected_limits = limits or TaskGraphLimits()
        validate_idempotency_key(idempotency_key)
        graph.validate_limits(selected_limits)
        admitted_nodes: list[TaskNode] = []
        agent_ids: set[str] = set()
        for node in graph.nodes:
            payload = node.input
            if set(payload) != _AGENT_TASK_V1_FIELDS:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if payload.get("type") != "linktools.ai.agent" or payload.get("version") != 1:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            user_prompt = payload.get("user_prompt")
            user_prompt_codec = payload.get("user_prompt_codec")
            mode = payload.get("mode")
            planning = payload.get("planning")
            thinking = payload.get("thinking")
            if not isinstance(user_prompt, str) or not isinstance(user_prompt_codec, str):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                transport = user_prompt_transport(user_prompt, user_prompt_codec)
                resolved_mode = normalize_execution_mode(mode)
                resolved_thinking = normalize_thinking(thinking)
            except (AIError, TypeError, ValueError) as error:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            if resolved_mode != "run" or not isinstance(planning, bool):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                snapshot = AgentBindingSnapshot.from_payload(payload.get("binding"))
                binding = self._restore_binding(snapshot)
            except AIError as error:
                if error.code is ErrorCode.AGENT_DEFINITION_UNAVAILABLE:
                    raise
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
            validate_user_prompt(str(transport))
            agent_ids.add(binding.definition.spec.id)
            admitted_nodes.append(
                TaskNode(
                    node.node_id,
                    node.dependencies,
                    input={
                        "type": "linktools.ai.agent",
                        "version": 1,
                        "binding": binding.snapshot.to_payload(),
                        "user_prompt": str(transport),
                        "user_prompt_codec": transport.codec,
                        "mode": "run",
                        "planning": planning,
                        "thinking": resolved_thinking,
                    },
                    budget_cost=node.budget_cost,
                )
            )
        admitted = TaskGraph(graph.graph_id, tuple(admitted_nodes))
        _logger.info(
            "agent task graph admitted: graph=%s tenant=%s agents=%s nodes=%s",
            graph.graph_id,
            resolved_principal.tenant_id,
            tuple(sorted(agent_ids)),
            len(admitted.nodes),
        )
        return TaskGraphRequest(
            admitted,
            resolved_principal,
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
            if error.code not in {
                ErrorCode.SESSION_NOT_FOUND,
                ErrorCode.AUTHORIZATION_DENIED,
            }:
                raise
            try:
                await self.session.create(
                    definition.spec.id,
                    CreateSessionRequest(
                        principal,
                        session_id,
                        secrets.token_urlsafe(32),
                        None,
                        {},
                    ),
                )
                return
            except AIError as create_error:
                if create_error.code is not ErrorCode.STORAGE_CONFLICT:
                    raise
            session = await self.session.get(session_id, principal=principal)
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.agent_id != definition.spec.id:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    def _resolve_principal(self, principal: "Principal | None") -> Principal:
        if principal is not None:
            return principal
        return self._default_principal

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            if not self._closing:
                self._closing = True
                _logger.info("runtime close started: tenant=%s", self._tenant_id)
            task = self._close_task
            retry = task is None
            if task is not None and task.done():
                try:
                    task.result()
                except (asyncio.CancelledError, Exception):
                    retry = True
                else:
                    if self._closed:
                        return
                    retry = True
            if retry:
                task = asyncio.create_task(
                    self._cleanup(),
                    name="linktools-runtime-close",
                )
                task.add_done_callback(self._consume_close_result)
                self._close_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                try:
                    task.result()
                except asyncio.CancelledError:
                    pass
                except BaseException:  # noqa: BLE001
                    _logger.exception("runtime close failed after caller cancellation")
            raise
        except BaseException:
            async with self._close_lock:
                if self._close_task is task:
                    self._close_task = None
            raise

    def _consume_close_result(self, task: "asyncio.Task[None]") -> None:
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("runtime close task failed")

    async def _cleanup(self) -> None:
        if self._close_callback is not None:
            await self._close_callback()
        async with self._close_lock:
            self._closed = True
            _logger.info("runtime close completed: tenant=%s", self._tenant_id)


def _execution_policy(
    definition: AgentDefinition,
    *,
    mode: ExecutionMode,
    planning: "bool | None",
    thinking: "ThinkingValue | None",
) -> "tuple[ExecutionMode, bool, ThinkingValue]":
    resolved_mode = normalize_execution_mode(mode)
    if planning is not None and not isinstance(planning, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    resolved_planning = definition.spec.planning if planning is None else planning
    if resolved_mode == "plan":
        resolved_planning = True
    resolved_thinking = (
        definition.spec.thinking
        if thinking is None
        else normalize_thinking(thinking)
    )
    return resolved_mode, resolved_planning, resolved_thinking


def _validate_memory_scope(value: "str | None") -> "str | None":
    if value is not None:
        validate_memory_scope(value)
    return value


@asynccontextmanager
async def _open_runtime(
    workspace: Workspace,
    *,
    app: object,
    tenant_id: "str | None",
    models: "ModelRegistry | None",
    state: "RuntimeState | None",
    capabilities: "Sequence[CapabilityGroup[object]]",
    middleware: "Sequence[Middleware]",
):
    from ._factory import compose_runtime_components

    components = await compose_runtime_components(
        workspace,
        app=app,
        tenant_id=tenant_id,
        models=models,
        state=state,
        capabilities=capabilities,
        middleware=middleware,
    )
    try:
        runtime = Runtime(
            components.catalog,
            components.compiler,
            components.execution,
            components.session,
            components.task,
            components.evaluation,
            components.approval,
            components.event,
            components.artifact,
            workspace=workspace,
            app=app,
            tenant_id=components.tenant_id,
            close_callback=components.close_callback,
            local_coordinator=components.local_coordinator,
        )
    except BaseException:
        await components.close_callback()
        raise
    try:
        yield runtime
    except BaseException as body_error:
        try:
            await runtime.close()
        except BaseException as close_error:
            raise close_error from body_error
        raise
    else:
        await runtime.close()


__all__ = ["Runtime"]
