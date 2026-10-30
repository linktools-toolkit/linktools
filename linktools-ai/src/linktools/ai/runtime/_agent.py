#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent and Session behavior with independently owned Execution handles."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Awaitable, Callable, Generic, TypeVar

from pydantic import BaseModel
from ..core import RunBudget, JsonValue, Page, Principal, ThinkingValue
from ..observe import WaitResult
from ..execution import Execution
from ..observe.stream import validate_wait
from ..execution import ExecutionInputContext
from ..model import ModelBinding, UserPromptInput, validate_user_input
from ..execution import ExecutionResult, ExecutionTreeEvent, SessionHistoryItem, SessionTurn, SessionView

if TYPE_CHECKING:
    from .agent_execution import AgentExecutionPort
    from ..agent import CompiledAgent, AgentBindingContract
    from ._runtime_service import Runtime

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

    def _definition(self) -> "Agent[AppT]":
        if self._agent_revision is None:
            return self._runtime.agents.get(self.agent_id)
        return Agent(self._runtime, self.agent_id, self._agent_revision,
                     self._compiled_agent, self._binding_contract)

    async def start(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
    ) -> "Execution[AppT]":
        """Start run mode; planning enables the planning capability without selecting plan mode."""
        return await self._definition().start(
            user_prompt,
            files=files,
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
    ) -> WaitResult[ExecutionResult]:
        """Run and wait; planning enables a capability, while include_event_content only controls observation."""
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        execution = await self.start(
            user_prompt,
            files=files,
            budget=budget,
            output=output,
            principal=principal,
            idempotency_key=idempotency_key,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
        )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_event_content=include_event_content,
            close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
        )

    async def plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
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
    ) -> WaitResult[ExecutionResult]:
        """Execute plan mode with plan-safe tools and wait. This can call models and tools; it is not a dry run."""
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        return await self._definition().plan(
            user_prompt,
            files=files,
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
    ) -> "Page[SessionHistoryItem]":
        return await self._runtime.sessions.history(
            self.session_id,
            principal=self._runtime._resolve_principal(principal or self._principal),
            cursor=cursor,
            limit=limit,
        )

    async def timeline(
        self,
        *,
        principal: "Principal | None" = None,
        cursor: "str | None" = None,
        limit: int = 100,
    ) -> "Page[SessionTurn]":
        return await self._runtime.sessions.timeline(
            self.session_id,
            principal=self._runtime._resolve_principal(principal or self._principal),
            cursor=cursor,
            limit=limit,
        )

    async def fork(
        self,
        new_session_id: str,
        *,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        cwd: "str | None" = None,
    ) -> "Session[AppT]":
        agent = self._definition()
        await self._runtime._fork_session(
            agent.id,
            self.session_id,
            new_session_id,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            cwd=cwd,
        )
        return agent.session(new_session_id, principal=principal or self._principal)

    async def update(
        self,
        *,
        expected_revision: int,
        metadata: Mapping[str, JsonValue],
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        cwd: "str | None" = None,
    ) -> SessionView:
        return await self._runtime._update_session(
            self.agent_id,
            self.session_id,
            expected_revision=expected_revision,
            metadata=metadata,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            cwd=cwd,
        )

    async def close(
        self,
        *,
        principal: "Principal | None" = None,
        idempotency_key: "str | None" = None,
        force: bool = False,
        wait_timeout_seconds: int = 30,
    ) -> SessionView:
        return await self._runtime._close_session(
            self.session_id,
            principal=principal or self._principal,
            idempotency_key=idempotency_key,
            force=force,
            wait_timeout_seconds=wait_timeout_seconds,
        )


@dataclass(frozen=True, slots=True)
class Agent(Generic[AppT]):
    _runtime: "Runtime[AppT]"
    id: str
    _agent_revision: int
    _compiled_agent: "CompiledAgent | None" = None
    _binding_contract: "AgentBindingContract | None" = None

    @property
    def execution_port(self) -> "AgentExecutionPort[AppT]":
        return self._runtime._agent_execution_port(self.id, self.revision, self.compiled, self._binding_contract)

    @property
    def runtime(self) -> "Runtime[AppT]":
        """Runtime that owns this Agent definition."""
        return self._runtime

    @property
    def revision(self) -> int:
        return self._agent_revision

    @property
    def model(self) -> ModelBinding:
        """Current model for inspection; no provider client is created."""
        compiled = self._runtime._compiled_agent(self.id, self.revision, self._compiled_agent)
        return self._runtime._compiler.bind(compiled).compiled_agent.model

    @property
    def compiled(self) -> "CompiledAgent | None":
        return self._compiled_agent

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
        budget: RunBudget | None = None,
        output: "type[BaseModel] | None" = None,
        principal: "Principal | None" = None,
        session_id: "str | None" = None,
        idempotency_key: "str | None" = None,
        planning: "bool | None" = None,
        thinking: "ThinkingValue | None" = None,
        correlation: "Mapping[str, object] | None" = None,
        input_context: ExecutionInputContext | None = None,
    ) -> "Execution[AppT]":
        """Start run mode; planning enables the planning capability without selecting plan mode."""
        prompt = validate_user_input(user_prompt)
        return await self.execution_port.start(
            prompt,
            files=files,
            budget=budget,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            mode="run",
            planning=planning,
            thinking=thinking,
            correlation=correlation,
            input_context=input_context,
        )

    async def run(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
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
    ) -> WaitResult[ExecutionResult]:
        """Run and wait; planning enables a capability, while include_event_content only controls observation."""
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        execution = await self.start(
            user_prompt,
            files=files,
            budget=budget,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            planning=planning,
            thinking=thinking,
            correlation=correlation,
            input_context=input_context,
        )
        return await execution.wait(
            timeout_seconds=timeout_seconds, on_event=on_event, include_event_content=include_event_content,
            close_timeout_seconds=close_timeout_seconds,
            raise_on_stream_error=raise_on_stream_error,
        )

    async def plan(
        self,
        user_prompt: "UserPromptInput",
        *,
        files: Sequence[str] = (),
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
    ) -> WaitResult[ExecutionResult]:
        """Execute plan mode with plan-safe tools and wait. This can call models and tools; it is not a dry run."""
        validate_wait(on_event, None, include_event_content, timeout_seconds, close_timeout_seconds,
                      raise_on_stream_error=raise_on_stream_error)
        prompt = validate_user_input(user_prompt)
        execution = await self.execution_port.start(
            prompt,
            files=files,
            budget=budget,
            output=output,
            principal=principal,
            session_id=session_id,
            idempotency_key=idempotency_key,
            mode="plan",
            planning=True,
            thinking=thinking,
            correlation=correlation,
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
        )

    async def create_session(
        self,
        session_id: str,
        *,
        principal: "Principal | None" = None,
        cwd: "str | None" = None,
        metadata: "Mapping[str, JsonValue] | None" = None,
        idempotency_key: "str | None" = None,
    ) -> "Session[AppT]":
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
