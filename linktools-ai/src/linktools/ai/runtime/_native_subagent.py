#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient authority for an exact claimed native delegation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import ExecutionLineageKind, Principal
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from ..execution import ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation, ExecutionStartReservationResult
    from ..persistence.commands import StateCommands
    from ..persistence.contracts import ToolOperationRecord
    from ..persistence.repositories import ExecutionRepositoryImpl, ToolRepositoryImpl
    from ..persistence.state import StateStore, StateTransaction
    from ..persistence.turns import ExecutionAuthority, SessionTurnKey, SqlSessionTurns
    from ..spec import RepositoryInstructions
    from ._tool import ToolOperationDecision


@dataclass(frozen=True, slots=True)
class NativeSubagentAdmission:
    turns: SqlSessionTurns
    key: SessionTurnKey
    generation: int
    tools: ToolRepositoryImpl
    decision: ToolOperationDecision
    invocation_id: str
    principal: Principal
    original: ToolOperationRecord | None = None
    instructions: RepositoryInstructions | None = None

    def require_child(self, child: ExecutionRecord, principal: Principal, generation: int | None) -> None:
        if ((child.parent_execution_id, child.root_execution_id, child.parent_invocation_id)
                != (self.key.turn_id, self.key.turn_id, self.invocation_id)
                or (child.principal_id, child.principal_kind) != (principal.principal_id, principal.kind)
                or self.principal != principal or self.generation != generation):
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)

    async def reserve(
        self, repository: ExecutionRepositoryImpl, reservation: ExecutionStartReservation,
        *, authority: ExecutionAuthority | None,
    ) -> ExecutionStartReservationResult:
        if self.original is not None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        self.require_child(reservation.execution, self.principal, reservation.parent_producer_generation)
        return await repository.reserve_native_subagent(
            reservation, turns=self.turns, key=self.key, tools=self.tools, operation_id=self.decision.operation_id,
            owner=self.decision.owner, fence=self.decision.fence, invocation_id=self.invocation_id,
            principal=self.principal, authority=authority,
        )

    async def commit_start(self, commands: StateCommands, claim: ExecutionStartClaim) -> ExecutionRecord:
        if self.original is not None:
            return await commands.commit_recovered_native_subagent_start(
                self.turns, self.key, generation=self.generation, principal=self.principal,
                original=self.original, claim=claim,
            )
        return await commands.commit_native_subagent_start(
            claim, turns=self.turns, key=self.key, generation=self.generation, tools=self.tools,
            operation_id=self.decision.operation_id, owner=self.decision.owner, fence=self.decision.fence,
            invocation_id=self.invocation_id, principal=self.principal,
        )


    def require_deferred_child(self, child: ExecutionRecord) -> None:
        if child.lineage_kind is not ExecutionLineageKind.SUBAGENT or child.session_id is not None:
            raise AIError(ErrorCode.STORAGE_OWNER_MISMATCH)
        self.require_child(child, self.principal, self.generation)

    async def guard_deferred_resume(
        self, child_store: StateStore, write: Callable[[StateTransaction], Awaitable[ExecutionRecord]],
    ) -> ExecutionRecord:
        if self.original is not None:
            raise AIError(ErrorCode.STORAGE_RECOVERY_REQUIRED)
        return await self.turns.with_claimed_subagent_parent(
            self.key, generation=self.generation, tools=self.tools, operation_id=self.decision.operation_id,
            owner=self.decision.owner, fence=self.decision.fence, invocation_id=self.invocation_id,
            child_store=child_store, principal=self.principal, write=write,
        )
