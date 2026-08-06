#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deferred call and approval invariants."""

from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class ApprovalValue(StrEnum):
    """Allowed approval decisions."""

    APPROVED = "approved"
    DENIED = "denied"


class ApprovalDecision(BaseModel):
    """One immutable decision identified independently from its request."""

    model_config = ConfigDict(frozen=True)

    decision_id: str
    value: ApprovalValue
    reason: "str | None" = None


class PendingDeferredCall(BaseModel):
    """Frozen model-generated deferred call."""

    model_config = ConfigDict(frozen=True)

    call_id: str
    execution_id: str
    tool_call_id: str
    kind: str
    tool_name: str
    parameters: "dict[str, object]"
    parameter_digest: str
    schema_digest: str
    policy_digest: str
    approval_id: "str | None" = None
    pre_dispatched: bool = False
    external_task_id: "str | None" = None
    external_result: "object | None" = None
    status: str = "PENDING"

    def matches_frozen_contract(
        self, parameter_digest: str, schema_digest: str, policy_digest: str
    ) -> bool:
        """Return whether a decision still targets the frozen call."""
        return (
            self.parameter_digest == parameter_digest
            and self.schema_digest == schema_digest
            and self.policy_digest == policy_digest
        )

    def mark_approved(self) -> "PendingDeferredCall":
        """Return the call after one approval decision."""
        if self.status != "PENDING":
            raise ValueError("deferred call already decided")
        return self.model_copy(update={"status": "APPROVED"})

    def mark_denied(self) -> "PendingDeferredCall":
        """Return the call after one denial decision."""
        if self.status != "PENDING":
            raise ValueError("deferred call already decided")
        return self.model_copy(update={"status": "DENIED"})

    def mark_external_result(self, result: object) -> "PendingDeferredCall":
        """Return a completed external call without performing the effect."""
        if self.status not in {"APPROVED", "EXTERNAL"}:
            raise ValueError("deferred call is not approved")
        return self.model_copy(update={"status": "EXTERNAL", "external_result": result})
