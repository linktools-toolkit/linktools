#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Semantic execution trace values."""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..foundation.digest import sha256_digest
from ..foundation.json import canonical_json_bytes


class StopReason(StrEnum):
    END_TURN = "END_TURN"
    CANCELLED = "CANCELLED"
    MAX_TOKENS = "MAX_TOKENS"
    REFUSAL = "REFUSAL"
    TOOL_USE = "TOOL_USE"
    ERROR = "ERROR"


class TraceKind(StrEnum):
    MODEL = "model"
    TOOL = "tool"
    USAGE = "usage"
    FAILURE = "failure"
    CANCEL = "cancel"
    TERMINAL = "terminal"


class TraceEvent(BaseModel):
    model_config = ConfigDict(frozen=True)

    execution_id: str
    sequence: int = Field(ge=1)
    run_id: str
    kind: TraceKind
    attempt: int = Field(default=1, ge=1)
    timestamp: datetime
    message_digest: "str | None" = None
    settings_digest: "str | None" = None
    tool_schema_digest: "str | None" = None
    parts_digest: "str | None" = None
    finish_reason: "str | None" = None
    provider_response_id: "str | None" = None
    operation_id: "str | None" = None
    status: "str | None" = None
    replayed: bool = False
    usage: "dict[str, int] | None" = None
    payload_ref: "str | None" = None
    result_digest: "str | None" = None


class ModelTrace(TraceEvent):
    kind: TraceKind = TraceKind.MODEL


class ToolTrace(TraceEvent):
    kind: TraceKind = TraceKind.TOOL


class RunSnapshot(BaseModel):
    model_config = ConfigDict(frozen=True)

    snapshot_id: str
    execution_id: str
    run_id: str
    input_digest: str
    release_digest: str
    bundle_digest: str
    model_plan_digest: str
    prompt_digest: str
    dataset_digest: "str | None" = None
    evaluator_digest: "str | None" = None
    target_digest: "str | None" = None
    artifact_digests: "tuple[str, ...]" = ()
    trace_start: int = Field(ge=1)
    trace_end: int = Field(ge=1)
    result_digest: "str | None" = None
    checkpoint_ref: "str | None" = None
    usage: "dict[str, int]" = Field(default_factory=dict)
    stop_reason: "StopReason | None" = None
    digest: str

    def verify(self) -> bool:
        values = self.model_dump(mode="json")
        values.pop("digest", None)
        optional_values = {
            "dataset_digest": self.dataset_digest,
            "evaluator_digest": self.evaluator_digest,
            "target_digest": self.target_digest,
            "artifact_digests": self.artifact_digests,
        }
        for key, value in optional_values.items():
            if not value:
                values.pop(key, None)
        return self.digest == sha256_digest(canonical_json_bytes(values))

    def verify_artifacts(self, available_digests: "tuple[str, ...]") -> bool:
        """Verify every immutable artifact reference is still available."""
        return set(self.artifact_digests) <= set(available_digests)


__all__ = ["ModelTrace", "RunSnapshot", "StopReason", "ToolTrace", "TraceEvent", "TraceKind"]
