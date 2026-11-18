#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Internal semantic query extensions above the shared state primitives."""

from collections.abc import Mapping

from ..core import JsonValue
from .state import RecordQuery
from .codec import execution_prepared_handoff_candidate
from ._record_owner import record_scan_kind


class ExecutionRecoveryCandidateQuery(RecordQuery):
    """Internal indexed candidate selection; decoded recovery remains authoritative."""

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.kind != "execution" or self.states is not None:
            raise ValueError("execution recovery candidates require execution kind without a status filter")


def record_scan_projection(kind: str, state: str | None, data: Mapping[str, JsonValue]) -> str:
    """Project canonical control for storage without making it a second authority."""
    projection = record_scan_kind(kind, state)
    if projection == "e:0" and execution_prepared_handoff_candidate(data):
        return record_scan_kind(kind, state, prepared=True)
    return projection


__all__ = ["ExecutionRecoveryCandidateQuery", "record_scan_projection"]
