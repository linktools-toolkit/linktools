#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent and Session behavior with independently owned Execution handles."""

import secrets
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel
from ..session import SessionContext
from ..core import ExecutionMode, RunBudget, JsonValue, Page, Principal, ThinkingValue
from ..observe import WaitResult
from ..errors import AIError, ErrorCode
from ..execution import Execution
from ..observe.stream import validate_wait
from ..execution import ExecutionInputContext
from ..model import ModelBinding, UserPromptInput, validate_user_input
from ..execution import ExecutionResult, ExecutionTreeEvent
from ..session import SessionHistoryItem, SessionTurn, SessionView

if TYPE_CHECKING:
    from .agent_execution import AgentExecutionPort
    from ..agent import CompiledAgent, AgentBinding, AgentBindingContract
    from ._runtime_service import Runtime
    from ._native_session import _NativeSession, _NativeSessionTurn, _NativeTurnResult
    from .turn_clients import _SessionOccupancy

AppT = TypeVar("AppT")


@dataclass(frozen=True, slots=True)
class Session(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    agent_id: str
    _agent_revision: int | None
    session_id: str
    _principal: "Principal | None" = None
    _compiled_agent: "CompiledAgent | None" = None
    _binding_contract: "AgentBindingContract | None" = None
    _derivations: tuple[Mapping[str, object], ...] = ()
    _observed_binding: "AgentBinding | None" = None

    @property
    def principal(self) -> Principal:
        return self._runtime.default_principal if self._principal is None else self._principal

    async def inspect(self, *, principal: Principal | None = None) -> "SessionView | dict[str, JsonValue]":
        return await self._runtime.sessions.inspect(
            self.session_id, principal=self.principal if principal is None else principal,
        )

    def watch(self, *, principal: Principal | None = None) -> "AsyncIterator[SessionView] | AsyncIterator[_SessionOccupancy]":
        """Observe occupancy changes made by this Runtime without polling."""
        actor = self.principal if principal is None else principal
        return self._runtime.sessions.watch(self.session_id, principal=actor)

    def _definition(self) -> "Agent[AppT]":
        return Agent(self._runtime, self.agent_id, self._agent_revision,
                     self._compiled_agent, self._binding_contract, self._derivations,
                     self._observed_binding)

    async def start(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT] | _NativeSessionTurn[AppT, _NativeTurnResult]":
        """Start run mode; planning enables the planning capability without selecting plan mode."""
        return await self._definition().start(
            user_prompt,
            files=files,
            session_context=session_context,
            budget=budget,
            output=output,
            principal=principal or self._principal,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
        )

    async def run(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_event_content: bool = False,
        close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
    ) -> "WaitResult[ExecutionResult] | WaitResult[_NativeTurnResult]":
        """Run and wait; planning enables a capability, while include_event_content only controls observation."""
        return await self._definition().run(
            user_prompt, files=files, session_context=session_context, budget=budget,
            output=output, principal=principal or self._principal, session_id=self.session_id,
            idempotency_key=idempotency_key, planning=planning, thinking=thinking,
            correlation=correlation, timeout_seconds=timeout_seconds, on_event=on_event,
            include_event_content=include_event_content, close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
        )

    async def plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_event_content: bool = False,
        close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
    ) -> "WaitResult[ExecutionResult] | WaitResult[_NativeTurnResult]":
        """Execute plan mode with plan-safe tools and wait. This can call models and tools; it is not a dry run."""
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        return await self._definition().plan(
            user_prompt,
            files=files,
            session_context=session_context,
            budget=budget,
            output=output,
            principal=principal or self._principal,
            session_id=self.session_id,
            idempotency_key=idempotency_key,
            thinking=thinking,
            correlation=correlation,
            timeout_seconds=timeout_seconds,
            on_event=on_event,
            include_event_content=include_event_content,
            close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
        )

    async def history(
        self,
        *,
        principal: "Principal | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionHistoryItem] | dict[str, JsonValue]":
        actor = self._runtime._resolve_principal(principal or self._principal)
        return await self._runtime.sessions.history(
            self.session_id,
            principal=actor,
            cursor=cursor,
            limit=limit,
        )

    async def timeline(
        self,
        *,
        principal: "Principal | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionTurn] | dict[str, JsonValue]":
        actor = self._runtime._resolve_principal(principal or self._principal)
        return await self._runtime.sessions.timeline(
            self.session_id,
            principal=actor,
            cursor=cursor,
            limit=limit,
        )

    async def fork(
        self,
        new_session_id: str,
        *,
        expected_revision: int | None = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        cwd: "str | None" = None,
    ) -> "Session[AppT] | None":
        """Fork the committed prefix at an explicit revision on SQL storage."""
        sessions = self._runtime.sessions
        actor = principal or self._principal
        agent = self._definition()
        if sessions._uses_turn_sessions:
            if expected_revision is None or idempotency_key is None or cwd is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={
                    "field": "expected_revision" if expected_revision is None else
                             "idempotency_key" if idempotency_key is None else "cwd",
                })
            forked = await sessions.fork_turn_session(
                self.session_id, new_session_id, principal=self._runtime._resolve_principal(actor),
                expected_revision=expected_revision, idempotency_key=idempotency_key,
            )
            return agent.session(new_session_id, principal=actor) if forked else None
        if expected_revision is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "expected_revision"})
        await self._runtime._fork_session(
            agent.id, self.session_id, new_session_id, principal=actor,
            idempotency_key=idempotency_key, cwd=cwd,
        )
        return agent.session(new_session_id, principal=actor)

    async def update(
        self,
        *,
        expected_revision: int,
        metadata: Mapping[str, JsonValue],
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        cwd: "str | None" = None,
        replace_cwd: bool = False,
    ) -> SessionView | bool:
        """Update session metadata; SQL uses the native revision check."""
        sessions = self._runtime.sessions
        actor = principal or self._principal
        if sessions._uses_turn_sessions:
            if idempotency_key is not None:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "idempotency_key"})
            return await sessions.update_turn_session(
                self.session_id, principal=self._runtime._resolve_principal(actor),
                expected_revision=expected_revision, metadata=metadata, cwd=cwd,
                replace_cwd=replace_cwd or cwd is not None,
            )
        if replace_cwd:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "replace_cwd"})
        return await self._runtime._update_session(
            self.agent_id, self.session_id, expected_revision=expected_revision,
            metadata=metadata, principal=actor, idempotency_key=idempotency_key, cwd=cwd,
        )

    async def close(
        self,
        *,
        expected_revision: int | None = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        force: bool = False,
        wait_timeout_seconds: int = 30,
    ) -> SessionView | bool:
        """Close an idle SQL session at an explicit revision."""
        sessions = self._runtime.sessions
        actor = principal or self._principal
        if sessions._uses_turn_sessions:
            if expected_revision is None or idempotency_key is not None or force or wait_timeout_seconds != 30:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={
                    "field": "expected_revision" if expected_revision is None else
                             "idempotency_key" if idempotency_key is not None else
                             "force" if force else "wait_timeout_seconds",
                })
            return await sessions.close_turn_session(
                self.session_id, principal=self._runtime._resolve_principal(actor),
                expected_revision=expected_revision,
            )
        if expected_revision is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "expected_revision"})
        return await self._runtime._close_session(
            self.session_id, principal=actor, idempotency_key=idempotency_key, force=force,
            wait_timeout_seconds=wait_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class Agent(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    id: str
    _agent_revision: int | None
    _compiled_agent: "CompiledAgent | None" = None
    _binding_contract: "AgentBindingContract | None" = None
    _derivations: tuple[Mapping[str, object], ...] = ()
    _observed_binding: "AgentBinding | None" = None

    @property
    def execution_port(self) -> "AgentExecutionPort[AppT]":
        return self._runtime._agent_execution_port(
            self.id, self.revision, self.compiled, self._binding_contract,
            derivations=self._derivations, observed_binding=self._observed_binding,
        )

    @property
    def runtime(self) -> "Runtime[AppT]":
        """Runtime that owns this Agent definition."""
        return self._runtime

    @property
    def revision(self) -> int | None:
        """Observed definition revision, or None before this handle is resolved."""
        return self._agent_revision

    @property
    def model(self) -> ModelBinding | None:
        """Observed model declaration, not a live source lookup or provider client."""
        if self._compiled_agent is None:
            return None
        compiled = self._runtime._compiled_agent(self.id, self.revision, self._compiled_agent)
        return compiled.model

    @property
    def compiled(self) -> "CompiledAgent | None":
        return self._compiled_agent

    async def resolve(self) -> "Agent[AppT]":
        """Inspect a current definition without pinning subsequent admissions."""
        binding = await self.execution_port.resolve_definition()
        return replace(
            self, _agent_revision=binding.compiled_agent.spec.revision,
            _compiled_agent=binding.compiled_agent, _observed_binding=binding,
        )

    def derive(
        self,
        *,
        model: str | None = None,
        system_prompt: str | None = None,
        instructions: Sequence[str] | None = None,
        allow_tools: Sequence[str] | None = None,
        allow_skills: Sequence[str] | None = None,
    ) -> "Agent[AppT]":
        return self._runtime._derive_agent(
            self,
            model=model,
            system_prompt=system_prompt,
            instructions=instructions,
            allow_tools=allow_tools,
            allow_skills=allow_skills,
        )

    async def start(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        input_context: ExecutionInputContext | None = None,
    ) -> "Execution[AppT] | _NativeSessionTurn[AppT, _NativeTurnResult]":
        """Start run mode; planning enables the planning capability without selecting plan mode."""
        return await self._start_on_owner(
            user_prompt, files=files, session_context=session_context, budget=budget, output=output,
            principal=principal, session_id=session_id, idempotency_key=idempotency_key, mode="run",
            planning=planning, thinking=thinking, correlation=correlation, input_context=input_context,
        )

    async def _start_on_owner(
        self, user_prompt: "UserPromptInput", *, files: Sequence[str], session_context: SessionContext,
        budget: RunBudget | None, output: "type[BaseModel] | None", principal: "Principal | None",
        session_id: str | None, idempotency_key: str | None, mode: ExecutionMode, planning: bool | None,
        thinking: "ThinkingValue | None", correlation: "Mapping[str, object] | None",
        input_context: ExecutionInputContext | None = None,
    ) -> "Execution[AppT] | _NativeSessionTurn[AppT, _NativeTurnResult]":
        if session_id is not None:
            sessions = self._runtime.sessions
            if sessions._uses_turn_sessions:
                if input_context is not None:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "input_context"})
                actor = self._runtime._resolve_principal(principal)
                session = sessions._bind_native_session(session_id, principal=actor, local_workspace=False)
                receipt = await sessions._start_turn_receipt(
                    session_id, user_prompt, agent=self, principal=actor,
                    idempotency_key=secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                    files=files, session_context=session_context, budget=budget, output=output,
                    mode=mode, planning=planning,
                    thinking=thinking, correlation=correlation,
                )
                return replace(session._turn(receipt.turn_id, actor), _replayed=receipt.replayed)
        return await self.execution_port.start(
            validate_user_input(user_prompt), files=files, session_context=session_context,
            budget=budget, output=output, principal=principal, session_id=session_id,
            idempotency_key=idempotency_key, mode=mode, planning=planning, thinking=thinking,
            correlation=correlation, **({"input_context": input_context} if mode == "run" else {}),
        )

    def _open_native_observer(self, session_id, principal, on_event, include_event_content):
        if session_id is None or on_event is None or not self._runtime.sessions._uses_turn_sessions:
            return nullcontext()
        if include_event_content:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"field": "include_event_content"})
        from ._native_session import _NativeMetadataObserver

        return _NativeMetadataObserver(
            self._runtime, session_id, self._runtime._resolve_principal(principal),
            local_workspace=False,
        )

    async def run(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        input_context: ExecutionInputContext | None = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_event_content: bool = False,
        close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
    ) -> "WaitResult[ExecutionResult] | WaitResult[_NativeTurnResult]":
        """Run and wait, optionally delivering a live callback.

        Native SQL sessions project metadata only. include_event_content has
        no effect without a callback and is rejected for native callbacks.
        """
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        from ._native_session import _NativeSessionTurn

        with self._open_native_observer(session_id, principal, on_event, include_event_content) as observer:
            execution = await self._start_on_owner(
                user_prompt, files=files, session_context=session_context, budget=budget, output=output,
                principal=principal, session_id=session_id, idempotency_key=idempotency_key,
                mode="run", planning=planning, thinking=thinking, correlation=correlation, input_context=input_context,
            )
            if isinstance(execution, _NativeSessionTurn):
                if observer is None:
                    return WaitResult(await execution.wait_final(timeout_seconds=timeout_seconds), None)
                assert on_event is not None
                return await execution._wait_observed(
                    observer, agent_id=self.id, on_event=on_event, timeout_seconds=timeout_seconds,
                    close_timeout_seconds=close_timeout_seconds, raise_on_stream_error=raise_on_stream_error,
                )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_event_content=include_event_content,
            close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
        )

    async def start_plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT] | _NativeSessionTurn[AppT, _NativeTurnResult]":
        """Start a plan on its existing session owner, without waiting for completion."""
        return await self._start_on_owner(
            user_prompt, files=files, session_context=session_context, budget=budget, output=output,
            principal=principal, session_id=session_id, idempotency_key=idempotency_key,
            mode="plan", planning=True, thinking=thinking, correlation=correlation,
        )

    async def plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        session_context: SessionContext = "all",
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        timeout_seconds: "float | None" = None,
        on_event: Callable[[ExecutionTreeEvent], Awaitable[None]] | None = None,
        include_event_content: bool = False,
        close_timeout_seconds: float = 5.0,
        raise_on_stream_error: bool = False,
    ) -> "WaitResult[ExecutionResult] | WaitResult[_NativeTurnResult]":
        """Execute plan mode with plan-safe tools and wait; this is not a dry run.

        Like run(), native SQL callbacks are metadata-only.
        """
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        from ._native_session import _NativeSessionTurn

        with self._open_native_observer(session_id, principal, on_event, include_event_content) as observer:
            execution = await self._start_on_owner(
                user_prompt, files=files, session_context=session_context, budget=budget, output=output,
                principal=principal, session_id=session_id, idempotency_key=idempotency_key,
                mode="plan", planning=True, thinking=thinking, correlation=correlation,
            )
            if isinstance(execution, _NativeSessionTurn):
                if observer is None:
                    return WaitResult(await execution.wait_final(timeout_seconds=timeout_seconds), None)
                assert on_event is not None
                return await execution._wait_observed(
                    observer, agent_id=self.id, on_event=on_event, timeout_seconds=timeout_seconds,
                    close_timeout_seconds=close_timeout_seconds, raise_on_stream_error=raise_on_stream_error,
                )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_event_content=include_event_content,
            close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
        )

    def session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
    ) -> "Session[AppT]":
        """Build a lazy definition-bound session handle without creating or checking a stored session."""
        return Session(
            self._runtime,
            self.id,
            self._agent_revision,
            session_id,
            principal,
            self._compiled_agent,
            self._binding_contract,
            self._derivations,
            self._observed_binding,
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
        idempotency_key: "str | None" = None,
    ) -> "Session[AppT] | _NativeSession[AppT, _NativeTurnResult]":
        sessions = self._runtime.sessions
        if sessions._uses_turn_sessions:
            from ..session import CreateSessionRequest

            return await sessions.create(CreateSessionRequest(
                self._runtime._resolve_principal(principal), session_id,
                secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key,
                cwd, {} if metadata is None else metadata,
            ))
        await self._runtime._create_session_for_agent(
            self.id,
            session_id,
            principal=principal,
            cwd=cwd,
            metadata=metadata,
            idempotency_key=idempotency_key,
        )
        return self.session(session_id, principal=principal)

__all__ = ["Agent", "Session"]
