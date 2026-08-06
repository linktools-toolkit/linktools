#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Deferred Tool output and exactly-once external effect identity."""

from pydantic import BaseModel, ConfigDict

from ..foundation.digest import hmac_digest


class DeferredToolRequests(BaseModel):
    """Separate approval and external call collections."""

    model_config = ConfigDict(frozen=True)

    approvals: "tuple[str, ...]" = ()
    calls: "tuple[str, ...]" = ()


class DeferredToolResults(BaseModel):
    """Results rebuilt in original model call order."""

    model_config = ConfigDict(frozen=True)

    results_by_tool_call: "dict[str, object]"
    original_order: "tuple[str, ...]"

    def ordered(self) -> 'tuple[object, ...]':
        """Return results in the model's original call order."""
        return tuple(self.results_by_tool_call[call_id] for call_id in self.original_order)


class ToolEffect(BaseModel):
    """Exactly-once external effect ledger value."""

    model_config = ConfigDict(frozen=True)

    effect_id: str
    state: str = "RESERVED"
    result: "object | None" = None

    def start(self) -> "ToolEffect":
        """Move a reserved effect into execution exactly once."""
        if self.state != "RESERVED":
            raise ValueError("tool effect is not reserved")
        return self.model_copy(update={"state": "STARTED"})

    @staticmethod
    def stable_id(execution_id: str, tool_call_id: str, approval_id: str) -> str:
        """Derive an effect ID without accepting caller-selected IDs."""
        return hmac_digest(execution_id.encode("utf-8"), f"{tool_call_id}:{approval_id}".encode("utf-8"))

    def complete(self, result: object) -> "ToolEffect":
        """Record a completed effect."""
        if self.state != "STARTED":
            raise ValueError("tool effect is not started")
        return self.model_copy(update={"state": "COMPLETED", "result": result})

    def mark_unknown(self) -> "ToolEffect":
        """Retain unknown remote side effects for reconciliation."""
        return self.model_copy(update={"state": "UNKNOWN"})
