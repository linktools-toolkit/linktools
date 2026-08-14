#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned tool authorization and durable operation contracts."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Protocol

from ..core import (
    Principal,
    ResourceRef,
    ToolOperationStatus,
    canonical_sha256,
    validate_lease_owner,
    validate_tenant_id,
)
from ..errors import AIError
from ..storage import ObjectRef


@dataclass(frozen=True, slots=True)
class ToolOperationRecord:
    tool_operation_id: str
    tenant_id: str
    step_run_id: str
    tool_call_id: str
    idempotency_key_hash: str
    tool_name: str
    arguments_hash: str
    binding_fingerprint: str
    replay_safe: bool
    status: ToolOperationStatus
    owner: "str | None"
    fence: int
    lease_expires_at: "datetime | None"
    result_object_ref: "ObjectRef | None"
    error_code: "str | None"
    created_at: datetime
    updated_at: datetime

    def __post_init__(self) -> None:
        try:
            validate_tenant_id(self.tenant_id)
            if self.owner is not None:
                validate_lease_owner(self.owner)
        except AIError as error:
            raise ValueError("tool operation lease identity is invalid") from error


class ToolStateRepository(Protocol):
    async def reserve(self, record: ToolOperationRecord) -> ToolOperationRecord: ...
    async def get_operation(self, tool_operation_id: str, *, tenant_id: str) -> "ToolOperationRecord | None": ...
    async def claim(self, tool_operation_id: str, *, tenant_id: str, owner: str, lease_seconds: int) -> ToolOperationRecord: ...
    async def renew(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, lease_seconds: int) -> ToolOperationRecord: ...
    async def complete(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, result_object_ref: "ObjectRef | None") -> ToolOperationRecord: ...
    async def fail(self, tool_operation_id: str, *, tenant_id: str, owner: str, fence: int, error_code: str) -> ToolOperationRecord: ...


@dataclass(frozen=True, slots=True)
class ToolDescriptor:
    name: str
    replay_safe: bool = False


class ToolAuthorization(StrEnum):
    ALLOW = "ALLOW"
    DENY = "DENY"
    REQUIRE_APPROVAL = "REQUIRE_APPROVAL"


class AllowAllToolPolicy:
    @property
    def fingerprint(self) -> str:
        return canonical_sha256({"tool_policy": "allow_all", "version": 1})

    async def authorize_tool(
        self,
        principal: Principal,
        execution: ResourceRef,
        tool: ToolDescriptor,
        arguments_digest: str,
    ) -> ToolAuthorization:
        del principal, execution, tool, arguments_digest
        return ToolAuthorization.ALLOW


class ToolPolicy(Protocol):
    @property
    def fingerprint(self) -> str: ...

    async def authorize_tool(
        self,
        principal: Principal,
        execution: ResourceRef,
        tool: ToolDescriptor,
        arguments_digest: str,
    ) -> ToolAuthorization: ...


__all__ = ["AllowAllToolPolicy", "ToolAuthorization", "ToolDescriptor", "ToolOperationRecord", "ToolPolicy", "ToolStateRepository"]
