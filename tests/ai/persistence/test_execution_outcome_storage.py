#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A published terminal result owns the shared recovery outcome body."""

from copy import deepcopy
from dataclasses import replace

import pytest

from linktools.ai.core import ExecutionEventType, ExecutionStatus, StopReason, UsageMetrics, canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.persistence import PersistenceDomain
from linktools.ai.persistence.codec import (
    _decode_enveloped_domain, _encode_persisted_domain, decode_domain, encode_domain,
    encode_envelope, iter_runtime_object_refs,
)
from linktools.ai.persistence.contracts import (
    ExecutionRecord, ExecutionRecoveryDetail, RecoveryHandoffPhase,
    RecoveryTerminalHandoff, RecoveryTerminalOutcome, ResultRecord,
)
from linktools.ai.storage import ObjectRef, StoredPayload

from tests.ai.persistence.test_repository_instruction_persistence import _execution


def _terminal(status: ExecutionStatus = ExecutionStatus.SUCCEEDED) -> ExecutionRecord:
    execution = _execution(None)
    successful = status is ExecutionStatus.SUCCEEDED
    payload = StoredPayload.inline_json({"text": "shared-output-sentinel:" + "output" * 1000}) if successful else None
    details = {} if successful else {"reason": "shared-error-sentinel:" + "detail" * 1000}
    error_code = None if successful else "failed"
    reason = StopReason.END_TURN if successful else StopReason.ERROR
    event_type = (ExecutionEventType.EXECUTION_SUCCEEDED if successful else
                  ExecutionEventType.EXECUTION_FAILED if status is ExecutionStatus.FAILED else
                  ExecutionEventType.EXECUTION_CANCELLED)
    result = ResultRecord(payload, reason, UsageMetrics(model_requests=2, output_tokens=17), execution.updated_at)
    outcome = RecoveryTerminalOutcome(
        status, error_code, details, reason, payload, None, result.usage,
        event_type, {"agent_run_id": "run"}, result.created_at,
    )
    return replace(execution, status=status, error_code=error_code, safe_error_details=details,
                   result=result, recovery=ExecutionRecoveryDetail(
                       agent_run_id="run", handoff_phase=RecoveryHandoffPhase.PREPARED,
                       terminal_handoff=RecoveryTerminalHandoff(outcome, "run", None),
                   ))


def _outcome(wire: dict) -> dict:
    return wire["fields"]["recovery"]["fields"]["terminal_handoff"]["fields"]["outcome"]["fields"]


def _restore(wire: dict) -> ExecutionRecord:
    return _decode_enveloped_domain(
        encode_envelope({"type": "execution_record", "payload": wire}), ExecutionRecord,
    )


@pytest.mark.parametrize("status", (ExecutionStatus.SUCCEEDED, ExecutionStatus.FAILED, ExecutionStatus.CANCELLED))
@pytest.mark.parametrize("phase", (RecoveryHandoffPhase.PREPARED, RecoveryHandoffPhase.COMPLETED))
def test_published_result_has_one_body_and_restores_recovery_outcome(status, phase) -> None:
    execution = _terminal(status)
    execution = replace(execution, recovery=replace(execution.recovery, handoff_phase=phase))
    wire = _encode_persisted_domain(execution)
    stored = canonical_json_bytes(wire)
    assert stored.count(b"shared-output-sentinel:") + stored.count(b"shared-error-sentinel:") == 1
    assert _restore(wire) == execution
    assert _encode_persisted_domain(_restore(wire)) == wire
    assert decode_domain(encode_domain(execution), ExecutionRecord) == execution
    assert execution.recovery.terminal_handoff.outcome.output == execution.result.output


@pytest.mark.parametrize("difference", ("unpublished", "output", "usage", "error_details"))
def test_unpublished_or_distinct_outcomes_retain_their_own_body(difference) -> None:
    execution = _terminal()
    if difference == "unpublished":
        execution = replace(execution, result=None, status=ExecutionStatus.STARTED)
    else:
        outcome = execution.recovery.terminal_handoff.outcome
        if difference == "output":
            outcome = replace(outcome, output=StoredPayload.inline_json({"text": "independent"}))
        elif difference == "usage":
            outcome = replace(outcome, usage=UsageMetrics(model_requests=3))
        else:
            outcome = replace(outcome, safe_error_details={"observation": "different"})
        execution = replace(execution, recovery=replace(execution.recovery,
                            terminal_handoff=replace(execution.recovery.terminal_handoff, outcome=outcome)))
    wire = _encode_persisted_domain(execution)
    assert "execution_result" not in _outcome(wire)
    assert _restore(wire) == execution


@pytest.mark.parametrize("source", (PersistenceDomain.EXECUTION, PersistenceDomain.CHECKPOINT))
def test_object_retention_preserves_the_owning_domain(source) -> None:
    execution = _terminal()
    payload = StoredPayload.object(ObjectRef("runtime", "result", "b" * 64, 4096))
    outcome = replace(execution.recovery.terminal_handoff.outcome, output=payload, object_source_domain=source)
    execution = replace(execution, result=replace(execution.result, output=payload),
                        recovery=replace(execution.recovery, terminal_handoff=replace(
                            execution.recovery.terminal_handoff, outcome=outcome)))
    wire = _encode_persisted_domain(execution)
    refs = set(iter_runtime_object_refs(wire, default_domain=PersistenceDomain.EXECUTION))
    assert (PersistenceDomain.EXECUTION, payload.ref) in refs
    assert (source, payload.ref) in refs
    assert ("execution_result" in _outcome(wire)) is (source is PersistenceDomain.EXECUTION)
    assert _restore(wire) == execution


@pytest.mark.parametrize("damage", ("missing_result", "invalid_reference", "duplicate_output", "foreign_source"))
def test_invalid_result_references_fail_closed(damage) -> None:
    wire = deepcopy(_encode_persisted_domain(_terminal()))
    outcome = _outcome(wire)
    if damage == "missing_result":
        wire["fields"]["result"] = None
    elif damage == "invalid_reference":
        outcome["execution_result"] = 1
    elif damage == "duplicate_output":
        outcome["output"] = wire["fields"]["result"]["fields"]["output"]
    else:
        outcome["object_source_domain"] = encode_domain(PersistenceDomain.CHECKPOINT)
    with pytest.raises(AIError) as error:
        _restore(wire)
    assert error.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
