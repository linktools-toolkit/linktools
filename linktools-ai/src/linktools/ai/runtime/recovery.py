#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Public execution recovery value contracts."""

from dataclasses import dataclass
from typing import Any, TypeAlias

from ..core import Principal, ToolOperationStatus, validate_idempotency_key


@dataclass(frozen=True, slots=True)
class ToolEffectApplied:
    result: Any


@dataclass(frozen=True, slots=True)
class ToolEffectNotApplied:
    pass


@dataclass(frozen=True, slots=True)
class ToolEffectFailed:
    pass


ToolEffectResolution: TypeAlias = (
    ToolEffectApplied | ToolEffectNotApplied | ToolEffectFailed
)


@dataclass(frozen=True, slots=True)
class ResolveToolEffectRequest:
    principal: Principal
    operation_id: str
    expected_fence: int
    resolution: ToolEffectResolution
    idempotency_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.principal, Principal):
            raise TypeError("principal must be Principal")
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        if (
            not isinstance(self.expected_fence, int)
            or isinstance(self.expected_fence, bool)
            or self.expected_fence < 1
        ):
            raise ValueError("expected_fence must be a positive integer")
        if not isinstance(
            self.resolution,
            (ToolEffectApplied, ToolEffectNotApplied, ToolEffectFailed),
        ):
            raise TypeError("resolution is invalid")
        validate_idempotency_key(self.idempotency_key)


@dataclass(frozen=True, slots=True)
class ExecutionRecoveryEffect:
    operation_id: str
    execution_id: str
    step_run_id: str
    tool_call_id: str
    tool_name: str
    fence: int
    idempotency_key_digest: str
    replay_safe: bool
    error_code: "str | None"

    def __post_init__(self) -> None:
        identities = (
            self.operation_id,
            self.execution_id,
            self.step_run_id,
            self.tool_call_id,
            self.tool_name,
            self.idempotency_key_digest,
        )
        if any(not isinstance(value, str) or not value.strip() for value in identities):
            raise ValueError("recovery effect identity is invalid")
        if (
            not isinstance(self.fence, int)
            or isinstance(self.fence, bool)
            or self.fence < 1
        ):
            raise ValueError("recovery effect fence is invalid")
        if not isinstance(self.replay_safe, bool):
            raise TypeError("replay_safe must be bool")
        if self.error_code is not None and (
            not isinstance(self.error_code, str) or not self.error_code.strip()
        ):
            raise ValueError("recovery effect error code is invalid")


@dataclass(frozen=True, slots=True)
class ToolEffectResolutionResult:
    operation_id: str
    execution_id: str
    status: ToolOperationStatus
    fence: int

    def __post_init__(self) -> None:
        if not isinstance(self.operation_id, str) or not self.operation_id.strip():
            raise ValueError("operation_id must be a non-empty string")
        if not isinstance(self.execution_id, str) or not self.execution_id.strip():
            raise ValueError("execution_id must be a non-empty string")
        if not isinstance(self.status, ToolOperationStatus):
            raise TypeError("status must be ToolOperationStatus")
        if self.status is ToolOperationStatus.EFFECT_UNKNOWN:
            raise ValueError("resolved tool effect cannot remain EFFECT_UNKNOWN")
        if not isinstance(self.fence, int) or isinstance(self.fence, bool) or self.fence < 1:
            raise ValueError("fence must be a positive integer")


__all__ = [
    "ExecutionRecoveryEffect",
    "ResolveToolEffectRequest",
    "ToolEffectApplied",
    "ToolEffectFailed",
    "ToolEffectNotApplied",
    "ToolEffectResolution",
    "ToolEffectResolutionResult",
]
