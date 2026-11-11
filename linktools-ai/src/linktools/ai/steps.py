#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native step records with explicitly owned Runtime lifecycle semantics."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic_ai_harness.step_persistence import StepEvent

from .core import validate_resource_id


def validate_native_step_ids(
    *, run_id: str, conversation_id: str | None = None, parent_run_id: str | None = None,
) -> None:
    """Apply the resource ID contract to native run identity and lineage."""
    validate_resource_id(run_id)
    for value in (conversation_id, parent_run_id):
        if value is not None:
            validate_resource_id(value)


@dataclass(frozen=True, slots=True)
class NativeStepSource:
    """Physical native checkpoint owner admitted with an execution."""

    group_id: str
    store_digest: str

    def __post_init__(self) -> None:
        for value, size in ((self.group_id, 32), (self.store_digest, 64)):
            if (not isinstance(value, str) or len(value) != size
                    or any(character not in "0123456789abcdef" for character in value)):
                raise ValueError("native checkpoint source identity is invalid")


def rekey_native_step_sources(value: object, groups: Mapping[str, str]) -> object:
    """Relocate only the explicit durable native source wire contract."""
    if isinstance(value, dict):
        result = {key: rekey_native_step_sources(item, groups) for key, item in value.items()}
        if value.get("$dataclass") == "native_step_source":
            fields = result.get("fields")
            if not isinstance(fields, dict):
                raise ValueError("native checkpoint source fields are invalid")
            source = NativeStepSource(**fields)
            if source.group_id not in groups:
                raise ValueError("native checkpoint source group is missing")
            fields["group_id"] = groups[source.group_id]
        return result
    if isinstance(value, list):
        return [rekey_native_step_sources(item, groups) for item in value]
    if isinstance(value, tuple):
        return tuple(rekey_native_step_sources(item, groups) for item in value)
    return value


LifecycleOutcome = Literal["succeeded", "interrupted", "failed", "cancelled"]


@dataclass(kw_only=True)
class NativeLifecycleEvent(StepEvent):
    """One native terminal body, without modifying shared run metadata."""

    outcome: LifecycleOutcome

    def __post_init__(self) -> None:
        validate_native_step_ids(run_id=self.run_id, conversation_id=self.conversation_id,
                                 parent_run_id=self.parent_run_id)
        kind = {"succeeded": "run_completed", "interrupted": "run_completed",
                "failed": "run_failed", "cancelled": "run_failed"}.get(self.outcome)
        if (kind is None or self.kind != kind or self.idempotency_key is None
                or self.tool_call_id is not None or self.tool_name is not None
                or (kind == "run_completed" and self.error is not None)):
            raise ValueError("native lifecycle outcome does not match its event")

    @classmethod
    def source_key(cls) -> str:
        """The first shared terminal observation of this single-shot native run."""
        return "terminal"

    @property
    def event_type(self) -> Literal["AGENT_RUN_SUCCEEDED", "AGENT_RUN_INTERRUPTED", "AGENT_RUN_FAILED"]:
        if self.outcome == "succeeded":
            return "AGENT_RUN_SUCCEEDED"
        if self.outcome == "interrupted":
            return "AGENT_RUN_INTERRUPTED"
        return "AGENT_RUN_FAILED"


__all__ = ["validate_native_step_ids", "LifecycleOutcome", "NativeLifecycleEvent", "NativeStepSource", "rekey_native_step_sources"]
