#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Frozen Runtime v1 persistence protocol fixtures."""

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

import pytest
from linktools.ai.core import BudgetUsage, IdempotencyStatus, OperationStatus, RunBudget, UsageMetrics
from linktools.ai.evaluation import CaseRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._codec import (
    _V1_ENUM_WIRE_TYPES,
    _V1_WIRE_TYPES,
    _CURRENT_CODEC,
    _decode_domain,
    _decode_step_envelope,
    _encode_step_envelope,
    _encode_persisted_domain,
    CURRENT_DATA_VERSION,
    decode_alias,
    decode_domain,
    decode_envelope,
    decode_fact,
    decode_operation,
    decode_record,
    encode_domain,
    encode_record,
    parse_envelope,
    wire_type_id,
)
from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ExecutionHistoryState,
    IdempotencyTerminalUpdate,
    OperationTerminalUpdate,
    StoredAgentRunCheckpoint,
    TranscriptMessageRef,
    TranscriptHeadRecord,
    TranscriptOwnerDomain,
    HistoryQuality,
)
from linktools.ai.runtime.state._step_contracts import AgentRunRecord, StepEvent
from linktools.ai.task import TaskBindingContract


def _fixture() -> dict[str, object]:
    path = Path(__file__).with_name("fixtures") / "runtime_state_v1_golden.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_current_wire_registry_matches_golden_manifest() -> None:
    fixture = _fixture()
    assert fixture["version"] == CURRENT_DATA_VERSION
    assert fixture["wire_type_ids"] == [wire_id for wire_id, _ in _V1_WIRE_TYPES]
    assert fixture["enum_wire_type_ids"] == [
        wire_id for wire_id, _ in _V1_ENUM_WIRE_TYPES
    ]
    assert len(fixture["wire_type_ids"]) == len(set(fixture["wire_type_ids"]))

    for wire_id, target in _V1_WIRE_TYPES:
        assert wire_type_id(target) == wire_id


def test_golden_current_envelopes_and_storage_primitives_decode() -> None:
    fixture = _fixture()
    envelopes = fixture["envelopes"]
    assert isinstance(envelopes, list)
    for value in envelopes:
        parsed = parse_envelope(value)
        assert parsed.version == CURRENT_DATA_VERSION
        assert decode_envelope(value) == parsed

    cursor_payload = envelopes[0]["value"]["payload"]
    cursor = decode_domain(cursor_payload, ConversationCursor)
    assert cursor == ConversationCursor("run", None, 0)

    state_payload = envelopes[1]["value"]["payload"]
    assert (
        decode_domain(state_payload, ExecutionHistoryState)
        is ExecutionHistoryState.OPEN
    )

    ref_payload = envelopes[2]["value"]["payload"]
    ref = decode_domain(ref_payload, TranscriptMessageRef)
    assert ref.owner_id == "run"
    assert ref.message_index == 0

    primitives = fixture["stored_primitives"]
    record = decode_record(primitives["record"])
    assert record.kind == "golden"
    assert decode_fact(primitives["fact"]).sequence == 1
    assert decode_operation(primitives["operation"]).sequence == 1
    assert decode_alias(primitives["alias"]).record_key_digest == bytes.fromhex(
        "77" * 32
    )


@pytest.mark.parametrize("wire_id,target", _V1_ENUM_WIRE_TYPES)
def test_registered_enum_wire_values_round_trip(
    wire_id: str, target: type[Enum],
) -> None:
    for member in target:
        payload = {"$enum": wire_id, "value": member.value}
        assert encode_domain(member) == payload
        assert decode_domain(payload, target) is member

    for raw, expected in (
        ("future-enum-value", ErrorCode.STORAGE_VERSION_UNSUPPORTED),
        ([], ErrorCode.STORAGE_INTEGRITY_ERROR),
    ):
        with pytest.raises(AIError) as raised:
            decode_domain({"$enum": wire_id, "value": raw}, target)
        assert raised.value.code is expected


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("update,expected", (
    (
        IdempotencyTerminalUpdate(
            "scope", "a" * 64, IdempotencyStatus.STARTED,
            IdempotencyStatus.COMPLETED, "b" * 64, None, None,
        ),
        {
            "scope": "scope", "idempotency_key_digest": "a" * 64,
            "expected_status": {"$enum": "idempotency_status", "value": "STARTED"},
            "next_status": {"$enum": "idempotency_status", "value": "COMPLETED"},
            "request_digest": "b" * 64, "result_digest": None, "error_code": None,
        },
    ),
    (
        OperationTerminalUpdate(
            "operation", OperationStatus.RUNNING, OperationStatus.SUCCEEDED,
            "result", "c" * 64, None,
        ),
        {
            "operation_id": "operation",
            "expected_status": {"$enum": "operation_status", "value": "RUNNING"},
            "next_status": {"$enum": "operation_status", "value": "SUCCEEDED"},
            "result_ref": "result", "result_digest": "c" * 64, "error_code": None,
        },
    ),
))
def test_terminal_updates_preserve_plain_mapping_wire(
    update: IdempotencyTerminalUpdate | OperationTerminalUpdate,
    expected: dict[str, object],
    persisted: bool,
) -> None:
    encode = _encode_persisted_domain if persisted else encode_domain
    payload = encode(update)
    assert payload == expected
    assert _decode_domain(payload, type(update), _CURRENT_CODEC, persisted=persisted) == update
    assert _decode_domain(
        {**payload, "future_note": True}, type(update), _CURRENT_CODEC, persisted=persisted,
    ) == update
    for name in payload:
        with pytest.raises(AIError) as raised:
            _decode_domain(
                {key: value for key, value in payload.items() if key != name},
                type(update), _CURRENT_CODEC, persisted=persisted,
            )
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_task_binding_contract_uses_wire_version_and_behavior_reference() -> None:
    contract = TaskBindingContract(
        "example.task",
        3,
        "replay_safe",
        {"kind": "json"},
        None,
        2,
        0.5,
    )
    encoded = encode_domain(contract)
    fields = encoded["fields"]

    assert fields["version"] == 1
    assert fields["id"] == "example.task"
    assert fields["revision"] == 3
    assert "task_id" not in fields
    assert "task_revision" not in fields
    assert decode_domain(encoded, TaskBindingContract) == contract


@pytest.mark.parametrize("values,field_name", (
    ((ConversationCursor("first", None, 1), ConversationCursor("second", "history", 2)), "message_count"),
    ((CaseRef("dataset", "first", 1), CaseRef("dataset", "second", 2)), "revision"),
))
def test_repeated_domain_decodes_validate_each_payload(
    values: tuple[ConversationCursor, ConversationCursor] | tuple[CaseRef, CaseRef],
    field_name: str,
) -> None:
    for value in values:
        payload = encode_domain(value)
        assert decode_domain(payload, type(value)) == value
        malformed = {**payload, "fields": {**payload["fields"], field_name: True}}
        with pytest.raises(AIError) as raised:
            decode_domain(malformed, type(value))
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
        assert decode_domain(payload, type(value)) == value


def test_record_reader_ignores_additive_fields() -> None:
    value = dict(_fixture()["stored_primitives"]["record"])
    value["partition"] = "11" * 32
    value["lease"] = {**value["lease"], "future_note": {"source": "remote"}}
    assert decode_record(value) == decode_record(
        _fixture()["stored_primitives"]["record"]
    )


def test_future_envelope_version_is_parseable_but_not_decoded_without_registry() -> None:
    future = {"v": CURRENT_DATA_VERSION + 1, "value": {"type": "future"}}
    assert parse_envelope(future).version == CURRENT_DATA_VERSION + 1
    with pytest.raises(AIError) as raised:
        decode_envelope(future)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_checkpoint_frontier_always_writes_pending_request_index() -> None:
    checkpoint = StoredAgentRunCheckpoint(
        "run",
        1,
        datetime(2026, 9, 20, tzinfo=timezone.utc),
        "complete",
        "projection",
        0,
        True,
    )

    encoded = _encode_step_envelope(checkpoint)
    fields = encoded["value"]["payload"]["fields"]  # type: ignore[index]

    assert fields["pending_request_index"] is None
    assert _decode_step_envelope(encoded) == checkpoint


def test_checkpoint_frontier_preserves_pending_request_index() -> None:
    checkpoint = StoredAgentRunCheckpoint(
        "run",
        1,
        datetime(2026, 9, 20, tzinfo=timezone.utc),
        "complete",
        "projection",
        0,
        True,
        3,
    )

    encoded = _encode_step_envelope(checkpoint)
    fields = encoded["value"]["payload"]["fields"]  # type: ignore[index]

    assert fields["pending_request_index"] == 3
    assert _decode_step_envelope(encoded) == checkpoint


def test_golden_step_event_uses_current_event_type_wire() -> None:
    event = StepEvent(
        "run",
        "AGENT_RUN_SUCCEEDED",
        1,
        timestamp=datetime(2026, 9, 20, tzinfo=timezone.utc),
    )
    encoded = _fixture()["step_event"]
    assert _encode_step_envelope(event) == encoded
    assert _decode_step_envelope(encoded) == event


@pytest.mark.parametrize("persisted", (False, True))
def test_declared_wire_defaults_restore_omitted_domain_fields(persisted: bool) -> None:
    head = TranscriptHeadRecord(TranscriptOwnerDomain.CONVERSATION, "history", 2, 1, HistoryQuality.COMPLETE)
    encode = _encode_persisted_domain if persisted else encode_domain
    payload = encode(head)
    del payload["fields"]["pending"]
    del payload["fields"]["pending_part_count"]
    assert _decode_domain(payload, TranscriptHeadRecord, _CURRENT_CODEC, persisted=persisted) == head


@pytest.mark.parametrize("value,field_name", (
    (ConversationCursor("run"), "message_count"),
    (UsageMetrics(input_tokens=19), "input_tokens"),
    (BudgetUsage("scope", RunBudget(), total_tokens=21), "total_tokens"),
))
def test_constructor_defaults_do_not_replace_required_durable_facts(value: object, field_name: str) -> None:
    payload = _encode_persisted_domain(value)
    del payload["fields"][field_name]
    with pytest.raises(AIError) as raised:
        _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("field_name", (
    "owner_domain", "owner_id", "message_count", "chunk_count", "quality",
))
def test_transcript_head_still_requires_authoritative_fields(field_name: str) -> None:
    head = TranscriptHeadRecord(TranscriptOwnerDomain.CONVERSATION, "history", 2, 1, HistoryQuality.COMPLETE)
    payload = _encode_persisted_domain(head)
    del payload["fields"][field_name]
    with pytest.raises(AIError) as raised:
        _decode_domain(payload, TranscriptHeadRecord, _CURRENT_CODEC, persisted=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("field_name,value", (
    ("pending", 0), ("pending", False),
    ("pending_part_count", None), ("pending_part_count", False),
    ("pending_part_count", "0"), ("pending_part_count", -1),
    ("pending_part_count", 1),
))
def test_explicit_transcript_defaults_are_validated(field_name: str, value: object) -> None:
    head = TranscriptHeadRecord(TranscriptOwnerDomain.CONVERSATION, "history", 2, 1, HistoryQuality.COMPLETE)
    payload = _encode_persisted_domain(head)
    payload["fields"][field_name] = value
    with pytest.raises(AIError) as raised:
        _decode_domain(payload, TranscriptHeadRecord, _CURRENT_CODEC, persisted=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("value,field_name", (
    (AgentRunRecord("run"), "started_at"),
    (StepEvent("run", "AGENT_RUN_STARTED", 0), "timestamp"),
    (AgentRunRecord("run"), "metadata"),
))
def test_domain_decode_does_not_invoke_missing_field_factories(
    value: AgentRunRecord | StepEvent, field_name: str,
) -> None:
    payload = _encode_persisted_domain(value)
    del payload["fields"][field_name]
    with pytest.raises(AIError) as raised:
        _decode_domain(payload, type(value), _CURRENT_CODEC, persisted=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("value", (True, -1, "0"))
def test_checkpoint_boundary_rejects_invalid_explicit_values(value: object) -> None:
    checkpoint = StoredAgentRunCheckpoint(
        "run", 1, datetime(2026, 9, 20, tzinfo=timezone.utc), "complete", "projection", 2,
    )
    payload = _encode_persisted_domain(checkpoint)
    payload["fields"]["transcript_message_count"] = value
    with pytest.raises(AIError) as raised:
        _decode_domain(payload, StoredAgentRunCheckpoint, _CURRENT_CODEC, persisted=True)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
