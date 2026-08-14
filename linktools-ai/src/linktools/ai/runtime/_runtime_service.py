#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public Runtime execution root."""

import secrets
import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping

from linktools.core import environ

from ..agent import AgentCompiler, AgentDefinition
from ..core import JsonValue, Principal, SessionStatus, validate_memory_scope
from ..errors import AIError, ErrorCode
from .service_api import (
    ApprovalService,
    ArtifactService,
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


class Runtime:
    """Execute any compiled AgentDefinition through one composed service graph."""

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
        self.compiler = compiler
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

    async def compile_agent(
        self,
        agent_id: "str | None" = None,
        *,
        prompt_id: "str | None" = None,
    ) -> AgentDefinition:
        self._ensure_open()
        resolved_agent = _selection_id(agent_id)
        resolved_prompt = _selection_id(prompt_id)
        definition = await self.compiler.compile(agent_id=resolved_agent, prompt_id=resolved_prompt)
        self._definitions[definition.digest] = definition
        _logger.debug("agent definition registered: agent=%s prompt=%s digest=%s", resolved_agent, resolved_prompt, definition.digest)
        return definition

    async def start(
        self,
        prompt: str,
        *,
        principal: Principal,
        agent_id: "str | None" = None,
        prompt_id: "str | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> ExecutionHandle:
        request = ExecutionRequest(
            prompt,
            principal,
            idempotency_key or secrets.token_urlsafe(32),
            _validate_memory_scope(memory_scope),
        )
        if session_id is None:
            definition = await self.compile_agent(agent_id, prompt_id=prompt_id)
            handle = await self.execution.run(definition.digest, request)
        else:
            if agent_id is not None or prompt_id is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            if not session_id.strip():
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            definition = await self._compile_session(session_id, principal)
            await self._ensure_session(definition, session_id, principal)
            from .service_api import ResumeSessionRequest

            handle = await self.session.resume(
                definition.digest,
                session_id,
                ResumeSessionRequest(principal, prompt, request.idempotency_key or "", request.memory_scope),
            )
        _logger.debug("runtime execution admitted: execution=%s definition=%s session=%s", handle.execution_id, definition.digest, session_id)
        return handle

    async def run_evaluation(
        self,
        request: RunEvaluationRequest,
        *,
        agent_id: "str | None" = None,
        prompt_id: "str | None" = None,
    ) -> EvaluationHandle:
        definition = await self.compile_agent(agent_id, prompt_id=prompt_id)
        return await self.evaluation.run(definition.digest, definition.output_schema_fingerprint, request)

    async def replay_evaluation(
        self,
        snapshot_id: str,
        request: ReplayEvaluationRequest,
        *,
        agent_id: "str | None" = None,
        prompt_id: "str | None" = None,
    ) -> ExecutionHandle:
        definition = await self.compile_agent(agent_id, prompt_id=prompt_id)
        return await self.evaluation.replay(definition.digest, snapshot_id, request)

    async def run(
        self,
        prompt: str,
        *,
        principal: Principal,
        agent_id: "str | None" = None,
        prompt_id: "str | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
        timeout_seconds: "float | None" = None,
    ) -> ExecutionResult:
        handle = await self.start(
            prompt,
            principal=principal,
            agent_id=agent_id,
            prompt_id=prompt_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )
        return await self.execution.wait(handle.execution_id, principal=principal, timeout_seconds=timeout_seconds)

    def stream(
        self,
        prompt: str,
        *,
        principal: Principal,
        agent_id: "str | None" = None,
        prompt_id: "str | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        memory_scope: "str | None" = None,
    ) -> AsyncIterator[ExecutionEvent]:
        return self._stream(
            prompt,
            principal=principal,
            agent_id=agent_id,
            prompt_id=prompt_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: Principal,
        agent_id: "str | None" = None,
        prompt_id: "str | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue]" = {},
    ) -> SessionView:
        definition = await self.compile_agent(agent_id, prompt_id=prompt_id)
        values = dict(metadata)
        if any(key.startswith("linktools.ai.") for key in values):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        values.update({"linktools.ai.agent_id": _selection_id(agent_id), "linktools.ai.prompt_id": _selection_id(prompt_id)})
        request_id = secrets.token_urlsafe(32)
        return await self.session.create(
            definition.digest,
            CreateSessionRequest(principal, session_id, request_id, cwd, values),
        )

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

    async def _stream(
        self,
        prompt: str,
        *,
        principal: Principal,
        agent_id: "str | None",
        prompt_id: "str | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
    ) -> AsyncIterator[ExecutionEvent]:
        handle = await self.start(
            prompt,
            principal=principal,
            agent_id=agent_id,
            prompt_id=prompt_id,
            session_id=session_id,
            idempotency_key=idempotency_key,
            memory_scope=memory_scope,
        )
        async for event in self.event.stream(handle.execution_id, principal=principal):
            yield event

    async def _ensure_session(self, definition: AgentDefinition, session_id: str, principal: Principal) -> None:
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
                    {
                        "linktools.ai.agent_id": definition.spec.id,
                        "linktools.ai.prompt_id": definition.prompt.id,
                    },
                ),
            )
            return
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.binding_digest != definition.digest:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)

    async def _compile_session(self, session_id: str, principal: Principal) -> AgentDefinition:
        try:
            session = await self.session.get(session_id, principal=principal)
        except AIError as error:
            if error.code not in {ErrorCode.SESSION_NOT_FOUND, ErrorCode.AUTHORIZATION_DENIED}:
                raise
            return await self.compile_agent()
        agent_id = session.metadata.get("linktools.ai.agent_id")
        prompt_id = session.metadata.get("linktools.ai.prompt_id")
        if not isinstance(agent_id, str) or not agent_id.strip() or not isinstance(prompt_id, str) or not prompt_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self.compile_agent(agent_id, prompt_id=prompt_id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


def _validate_memory_scope(value: "str | None") -> "str | None":
    if value is not None:
        validate_memory_scope(value)
    return value


def _selection_id(value: "str | None") -> str:
    if value is None:
        return "default"
    if not isinstance(value, str) or not value.strip():
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return value


__all__ = ["Runtime"]
