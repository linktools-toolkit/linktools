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
from ..capability import MemoryCapability
from ._capabilities import runtime_feature_leaves
from ..core import (OperationLedgerInput, ExecutionMode, RunBudget, Principal, ThinkingValue, ToolEffects, normalize_execution_mode,
                    normalize_thinking, overlay_correlation, validate_idempotency_key)
from ..errors import AIError, ErrorCode
from ..execution import Execution, ExecutionInputContext, ExecutionRequest, ResumeSessionRequest
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
    compiled: CompiledAgent | None
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
    memory_scope: str | None = None
    captured: bool = False
    agent_id: str | None = None
    derivations: tuple[Mapping[str, object], ...] = ()
    resolve_current: Callable[
        [str, Sequence[Mapping[str, object]], type[BaseModel] | OutputBinding | None],
        Awaitable[AgentBinding],
    ] | None = None
    restore_binding: Callable[[AgentBindingContract], Awaitable[AgentBinding]] | None = None
    observed_binding: AgentBinding | None = None

    @property
    def binding_contract(self) -> AgentBindingContract:
        """Capture declaration semantics without resolving models for ordinary starts."""
        contract = self._binding_contract
        if contract is None and self.observed_binding is not None:
            contract = self.observed_binding.binding_contract
        if contract is None:
            if self.compiled is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY,
                              safe_details={"reason": "agent_definition_unresolved"})
            contract = self.compiler.bind(self.compiled).binding_contract
            object.__setattr__(self, "_binding_contract", contract)
        return contract

    async def resolve_definition(self) -> AgentBinding:
        """Read the current declaration for inspection, without changing this port."""
        self.operations.ensure_open()
        if self.captured or self.resolve_current is None:
            if self.compiled is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if self.captured:
                return (
                    await self.restore_binding(self.binding_contract)
                    if self.restore_binding is not None
                    else self.compiler.restore(self.binding_contract, model=self.compiled.model)
                )
            return self.compiler.bind(self.compiled)
        agent_id = self.agent_id if self.compiled is None else self.compiled.spec.id
        if agent_id is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return await self.resolve_current(agent_id, self.derivations, None)

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
        if self.restore_binding is not None:
            return (await self.restore_binding(contract)).binding_contract
        restored = self.compiler.restore(
            contract, model=None if self.compiled is None else self.compiled.model,
        )
        return (await self.resolve_binding(restored)).binding_contract

    def owns_execution(self, execution_id: str, *, tenant_id: str) -> bool:
        return self.service.runtime_backend().owns_execution(execution_id, tenant_id=tenant_id)

    async def recover_bound_execution(
        self, execution: ExecutionRecord, *, recovery_operation: OperationLedgerInput,
    ) -> ExecutionRecord:
        """Resume the producer after its task owner validates and authorizes the binding."""
        self.operations.ensure_open()
        if (
            self.compiled is None
            or not execution.requires_task_invocation_capture
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
        mode: ExecutionMode,
        planning: "bool | None",
        thinking: "ThinkingValue | None",
        correlation: "Mapping[str, object] | None" = None,
        dependency_hold_id: "str | None" = None,
        requires_task_invocation_capture: bool = False,
        input_context: ExecutionInputContext | None = None,
        budget: RunBudget | None = None,
        budget_scope_id: str | None = None,
        tool_effects: ToolEffects = "live",
    ) -> "Execution[AppT]":
        self.operations.ensure_open()
        key = secrets.token_urlsafe(32) if idempotency_key is None else idempotency_key
        validate_idempotency_key(key)
        if input_context is not None and session_id is not None:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, safe_details={"reason": "imported_context_requires_new_execution"})
        resolved_principal = self.default_principal if principal is None else principal
        effective_correlation = overlay_correlation(self.correlation, correlation)
        resolved_files = normalize_input_files(files)
        captured = self.captured or requires_task_invocation_capture
        if not captured and self.resolve_current is not None:
            agent_id = self.agent_id if self.compiled is None else self.compiled.spec.id
            if agent_id is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            binding = await self.resolve_current(agent_id, self.derivations, output)
        else:
            compiled_agent = self.compiled
            if compiled_agent is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            if not captured:
                binding = self.compiler.bind(compiled_agent, output=output)
            else:
                binding = (
                    await self.restore_binding(self.binding_contract)
                    if self.restore_binding is not None
                    else self.compiler.restore(self.binding_contract, model=compiled_agent.model)
                )
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
            if not captured or self.restore_binding is None:
                binding = await self.resolve_binding(binding)
        compiled_agent = binding.compiled_agent
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
            memory_scope=_memory_scope(compiled_agent) if input_context is None else None,
            mode=resolved_mode,
            planning=resolved_planning,
            thinking=resolved_thinking,
            correlation=effective_correlation,
            files=resolved_files,
            input_context=input_context,
            budget=budget,
            tool_effects=tool_effects,
        )
        if session_id is not None and (not isinstance(session_id, str) or not session_id.strip()):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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
                tool_effects=request.tool_effects,
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




def _memory_scope(compiled_agent: CompiledAgent) -> str | None:
    scopes = [
        capability.scope
        for contribution in compiled_agent.runtime_contributions
        for capability in runtime_feature_leaves(contribution.value)
        if isinstance(capability, MemoryCapability)
    ]
    if len(scopes) > 1:
        raise AIError(ErrorCode.CAPABILITY_CONFLICT)
    return scopes[0] if scopes else None


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


__all__ = ["AgentExecutionPort"]
