#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The persisted, resumable execution snapshot and the engine-return snapshot data."""


from dataclasses import dataclass
from typing import Literal
from ..json import JsonValue

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from datetime import datetime
    from .domain import RunStatus, RunUsage

@dataclass(frozen=True, slots=True)
class AgentSnapshotData:
    resume_messages: "tuple[JsonValue, ...]"
    final_output: "JsonValue | None"
    usage: "RunUsage"
    trace_end_sequence: int


@dataclass(frozen=True, slots=True)
class RunSnapshot:
    schema: "Literal['run-snapshot.v1']"
    run_id: str
    revision: int
    resume_messages: "tuple[JsonValue, ...]"
    final_output: "JsonValue | None"
    status: "RunStatus"
    usage: "RunUsage"
    trace_end_sequence: int
    created_at: "datetime"


__all__: "list[str]" = ["AgentSnapshotData", "RunSnapshot"]
