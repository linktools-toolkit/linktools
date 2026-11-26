#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transient authority for an exact claimed native delegation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import Principal
from ..errors import AIError, ErrorCode

if TYPE_CHECKING:
    from ..execution import ExecutionRecord, ExecutionStartClaim, ExecutionStartReservation, ExecutionStartReservationResult
    from ..persistence.commands import StateCommands
    from ..persistence.repositories import ExecutionRepositoryImpl, ToolRepositoryImpl
    from ..persistence.turns import ExecutionAuthority, SessionTurnKey, SqlSessionTurns
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
        self.require_child(reservation.execution, self.principal, reservation.parent_producer_generation)
        return await repository.reserve_native_subagent(
            reservation, turns=self.turns, key=self.key, tools=self.tools, operation_id=self.decision.operation_id,
            owner=self.decision.owner, fence=self.decision.fence, invocation_id=self.invocation_id,
            principal=self.principal, authority=authority,
        )

    async def commit_start(self, commands: StateCommands, claim: ExecutionStartClaim) -> ExecutionRecord:
        return await commands.commit_native_subagent_start(
            claim, turns=self.turns, key=self.key, generation=self.generation, tools=self.tools,
            operation_id=self.decision.operation_id, owner=self.decision.owner, fence=self.decision.fence,
            invocation_id=self.invocation_id, principal=self.principal,
        )
