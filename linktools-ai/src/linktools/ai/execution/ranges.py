#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded model metadata read ranges."""

from ..core import validate_page_limit
from ..errors import AIError, ErrorCode
from ._service_api import ModelInteractionReadBoundary, UsageReadCutoff

def model_interaction_update_ranges(
    execution_id: str, boundary: ModelInteractionReadBoundary,
    after_cutoffs: tuple[UsageReadCutoff, ...], limit: int, *, allow_unavailable: bool = False,
) -> tuple[tuple[int, int, int], ...]:
    remaining = validate_page_limit(limit)
    afters: dict[int, int] = {}
    for cutoff in after_cutoffs:
        validate_model_metadata_range(cutoff.agent_run_seq, 0, cutoff.model_request_seq, limit)
        if cutoff.execution_id != execution_id or cutoff.agent_run_seq in afters:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        afters[cutoff.agent_run_seq] = cutoff.model_request_seq
    if (allow_unavailable and not boundary.cutoffs and not boundary.local_staging_available
            and not boundary.durable_history_available):
        return ()
    current = {value.agent_run_seq: value.model_request_seq for value in boundary.cutoffs}
    if any(run not in current or after > current[run] for run, after in afters.items()):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    ranges = []
    for cutoff in boundary.cutoffs:
        if cutoff.execution_id != execution_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        after = afters.get(cutoff.agent_run_seq, 0)
        count = min(remaining, cutoff.model_request_seq - after)
        if count > 0:
            ranges.append((cutoff.agent_run_seq, after, after + count))
            remaining -= count
    return tuple(ranges)


def validate_model_metadata_range(
    agent_run_seq: int, after_model_request_seq: int, through_model_request_seq: int, limit: int,
) -> int:
    limit = validate_page_limit(limit)
    if (isinstance(agent_run_seq, bool) or not isinstance(agent_run_seq, int)
            or agent_run_seq < 1
            or any(isinstance(value, bool) or not isinstance(value, int) or value < 0
                   for value in (after_model_request_seq, through_model_request_seq))
            or after_model_request_seq > through_model_request_seq):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return limit


__all__ = ["model_interaction_update_ranges", "validate_model_metadata_range"]
