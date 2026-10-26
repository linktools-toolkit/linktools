#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Agent execution operations bound to one producer and captured behavior."""
from __future__ import annotations

import secrets
from collections.abc import Mapping, Sequence, Callable, Awaitable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Generic, TypeVar
from pydantic import BaseModel
from linktools.core import environ
from ..agent import AgentBinding, AgentBindingContract, AgentCompiler, CompiledAgent, OutputBinding, bind_output
from ..core import (OperationLedgerInput, ExecutionMode, RunBudget, Principal, SessionStatus, ThinkingValue, normalize_execution_mode,
                    normalize_thinking, overlay_correlation, validate_idempotency_key, validate_memory_scope)
from ..errors import AIError, ErrorCode
from ..execution import Execution, ExecutionInputContext, ExecutionRequest, ResumeSessionRequest, CreateSessionRequest
from ..execution.handle import ExecutionBinding, ExecutionTreeWatcher
from ..model import CanonicalUserInput, normalize_input_files
if TYPE_CHECKING:
    from ._execution import DefaultExecutionService
    from ..execution import ExecutionRecord, SessionService
    from ..inputs import InputCaptures
    from ..persistence import Persistence

AppT = TypeVar("AppT")
_logger = environ.get_logger("ai.agent")

@dataclass(frozen=True, slots=True)
class AgentExecutionPort(Generic[AppT]):
    persistence: Persistence
    compiled: CompiledAgent
    _binding_contract: AgentBindingContract | None
    compiler: AgentCompiler
    service: DefaultExecutionService
    sessions: SessionService
    operations: ExecutionBinding
    watch_tree: ExecutionTreeWatcher
    resolve_binding: Callable[[AgentBinding], Awaitable[AgentBinding]]
    default_principal: Principal
    correlation: Mapping[str, object]
    captures: InputCaptures | None
    captured: bool = False

    @property
    def binding_contract(self) -> AgentBindingContract:
        """Capture declaration semantics without resolving models for ordinary starts."""
        contract = self._binding_contract
        if contract is None:
            contract = self.compiler.bind(self.compiled).binding_contract
            object.__setattr__(self, "_binding_contract", contract)
        return contract

    async def canonicalize_files(self, files: Sequence[str]) -> tuple[str, ...]:
        self.operations.ensure_open()
        if self.captures is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self.captures.canonicalize_files(files)

    async def materialize_input(self, prompt: CanonicalUserInput, files: Sequence[str] = ()) -> CanonicalUserInput:
        self.operations.ensure_open()
        if self.captures is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self.captures.materialize_input(prompt, files)

    async def resolve_contract(self, contract: AgentBindingContract) -> AgentBindingContract:
        self.operations.ensure_open()
        restored = self.compiler.restore(contract, model=self.compiled.model)
        return (await self.resolve_binding(restored)).binding_contract

    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool:
        return self.service.runtime_backend().owns_execution(execution_id, tenant_id=tenant_id)

    async def recover_bound_execution(
        self, execution: ExecutionRecord, *, recovery_operation: OperationLedgerInput,
    ) -> ExecutionRecord:
        """Resume the producer after its task owner validates and authorizes the binding."""
        self.operations.ensure_open()
        if (
            not execution.requires_task_invocation_capture
            or not isinstance(execution.binding, AgentBindingContract)
            or execution.binding.agent_spec.id != self.compiled.spec.id
            or execution.binding.agent_spec.revision != self.compiled.spec.revision
            or recovery_operation.tenant_id != self.persistence.tenant_id
            or recovery_operation.execution_id != execution.execution_id
            or recovery_operation.resource_id != execution.execution_id
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return await self.service.runtime_backend().recover_execution(
            execution.execution_id, tenant_id=recovery_operation.tenant_id,
            expected_revision=execution.revision, recovery_operation=recovery_operation,
        )

    async def get(self, execution_id: str, principal: Principal) -> Execution[AppT]:
        self.operations.ensure_open()
        return Execution(self.operations, execution_id, principal, self.watch_tree)

    async def acquire_hold(self, execution_id: str, principal: Principal, hold_id: str) -> None:
        await self.service.acquire_dependency_hold(execution_id, tenant_id=principal.tenant_id, hold_id=hold_id)

    async def release_hold(self, execution_id: str, principal: Principal, hold_id: str) -> None:
        await self.service.release_dependency_hold(execution_id, tenant_id=principal.tenant_id, hold_id=hold_id)

    async def start(
        self,
        user_prompt: CanonicalUserInput,
        *,
        files: Sequence[str],
        output: "type[BaseModel] | OutputBinding | None",
        principal: "Principal | None",
        session_id: "str | None",
        idempotency_key: "str | None",
        memory_scope: "str | None",
        mode: ExecutionMode,
        planning: "bool | None",
        thinking: "ThinkingValue | None",
        correlation: "Mapping[str, object] | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        input_context: ExecutionInputContext | None = None,
        budget: RunBudget | None = None,
        budget_scope_id: str | None = None,
    ) -> "Execution[AppT]":
        self.operations.ensure_open()
        key = secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key
        validate_idempotency_key(key)
        if input_context is not None and session_id is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "imported_context_requires_new_execution"})
        resolved_principal = self.default_principal if principal is None else principal
        effective_correlation = overlay_correlation(self.correlation, correlation)
        resolved_files = normalize_input_files(files)
        compiled_agent = self.compiled
        binding_contract = self.binding_contract if self.captured else None
        resolved_mode, resolved_planning, resolved_thinking = _execution_policy(
            compiled_agent,
            mode=mode,
            planning=planning,
            thinking=thinking,
        )
        request = ExecutionRequest(
            user_prompt=user_prompt,
            principal=resolved_principal,
            idempotency_key=key,
            memory_scope=_validate_memory_scope(memory_scope),
            mode=resolved_mode,
            planning=resolved_planning,
            thinking=resolved_thinking,
            correlation=effective_correlation,
            files=resolved_files,
            input_context=input_context,
            budget=budget,
        )
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if binding_contract is None:
            binding = self.compiler.bind(compiled_agent, output=output)
        else:
            binding = self.compiler.restore(binding_contract, model=compiled_agent.model)
            if output is not None:
                output_binding = output if isinstance(output, OutputBinding) else bind_output(output)
                binding = replace(
                    binding,
                    output_binding=output_binding,
                    binding_contract=replace(
                        binding.binding_contract,
                        output_mode=output_binding.mode,
                        output_schema=output_binding.schema_definition,
                    ),
                )
        binding = await self.resolve_binding(binding)
        if session_id is None:
            handle = await self.service.start(
                binding.binding_digest,
                request,
                dependency_hold_id=dependency_hold_id,
                budget_scope_id=budget_scope_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
                binding_contract=binding.binding_contract,
                resolved_binding=binding,
            )
        else:
            await self._ensure_session(compiled_agent, session_id, resolved_principal)
            resume_request = ResumeSessionRequest(
                principal=resolved_principal,
                user_prompt=request.user_prompt,
                idempotency_key=request.idempotency_key,
                memory_scope=request.memory_scope,
                mode=request.mode,
                planning=request.planning,
                thinking=request.thinking,
                correlation=effective_correlation,
                files=request.files,
                budget=request.budget,
            )
            handle = await self.sessions.resume(
                compiled_agent.spec.id,
                binding.binding_digest,
                session_id,
                resume_request,
                binding_contract=binding.binding_contract,
                resolved_binding=binding,
                dependency_hold_id=dependency_hold_id,
                budget_scope_id=budget_scope_id,
                requires_task_invocation_capture=requires_task_invocation_capture,
            )
        _logger.info(
            "runtime execution admitted: execution=%s agent=%s session=%s "
            "mode=%s planning=%s thinking=%s",
            handle.execution_id,
            compiled_agent.spec.id,
            session_id,
            resolved_mode,
            resolved_planning,
            resolved_thinking,
        )
        return Execution(self.operations,
            handle.execution_id,
            resolved_principal,
            self.watch_tree,
        )


    async def _ensure_session(
        self,
        compiled_agent: CompiledAgent,
        session_id: str,
        principal: Principal,
    ) -> None:
        try:
            session = await self.sessions.get(
                session_id,
                principal=principal,
            )
        except AIError as error:
            if error.code not in {
                ErrorCode.SESSION_NOT_FOUND,
                ErrorCode.AUTHORIZATION_DENIED,
            }:
                raise
            try:
                await self.sessions.create(
                    compiled_agent.spec.id,
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
            session = await self.sessions.get(
                session_id,
                principal=principal,
            )
        if session.status is not SessionStatus.OPEN:
            raise AIError(ErrorCode.SESSION_CONFLICT)
        if session.agent_id != compiled_agent.spec.id:
            raise AIError(ErrorCode.SESSION_BINDING_MISMATCH)


def _execution_policy(
    compiled_agent: CompiledAgent,
    *,
    mode: ExecutionMode,
    planning: "bool | None",
    thinking: "ThinkingValue | None",
) -> "tuple[ExecutionMode, bool, ThinkingValue]":
    resolved_mode = normalize_execution_mode(mode)
    if planning is not None and not isinstance(planning, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    resolved_planning = (
        compiled_agent.spec.planning
        if planning is None
        else planning
    )
    if resolved_mode == "plan":
        resolved_planning = True
    resolved_thinking = (
        compiled_agent.spec.thinking if thinking is None else normalize_thinking(thinking)
    )
    return resolved_mode, resolved_planning, resolved_thinking


def _validate_memory_scope(value: "str | None") -> "str | None":
    if value is not None:
        validate_memory_scope(value)
    return value


__all__ = ["AgentExecutionPort"]
