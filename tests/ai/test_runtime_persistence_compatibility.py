#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Systemic compatibility contract for Runtime persisted values."""

import copy
from dataclasses import replace
from datetime import datetime, timezone
from typing import cast

import pytest
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_persisted_domain,
    decode_domain,
    decode_record,
    encode_domain,
    wire_type_id,
)
from linktools.ai.runtime.state._maintenance import _validate_enveloped_value
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ContextProjectionItem,
    SessionRecord,
)
from linktools.ai.spec import AgentSpec
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskNode
from pydantic_ai_harness.step_persistence import RunRecord


def _session() -> SessionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        resource_generation=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        history_id="history",
    )


def _envelope(
    payload: object,
    *,
    wire_id: str = "session_record",
) -> dict[str, object]:
    return {
        "v": 1,
        "value": {
            "type": wire_id,
            "payload": payload,
        },
    }


def _binding_snapshot_payload() -> dict[str, object]:
    output = bind_output()
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
        global_runtime_capability_descriptors=(),
    )
    return snapshot.to_payload()


def test_current_generic_dataclass_round_trip() -> None:
    cursor = _session()
    encoded = encode_domain(cursor)
    assert encoded["$dataclass"] == "session_record"
    assert "schema" not in encoded
    assert decode_domain(encoded, SessionRecord) == cursor


def test_persisted_generic_writer_keeps_schema_one() -> None:
    session = _session()
    payload = _encode_persisted_domain(session)
    assert payload["schema"] == 1
    assert _decode_enveloped_domain(
        _envelope(payload),
        SessionRecord,
    ) == session


def test_context_projection_persisted_writer_round_trips() -> None:
    projection = ContextProjection((), "d" * 64)
    payload = _encode_persisted_domain(projection)

    assert _decode_enveloped_domain(
        _envelope(payload, wire_id="context_projection"),
        ContextProjection,
    ) == projection


def test_context_projection_rejects_runtime_type_mismatch_at_construction() -> None:
    with pytest.raises(TypeError):
        ContextProjection(
            cast("tuple[ContextProjectionItem, ...]", ("invalid",)),
            "d" * 64,
        )


def test_persisted_writer_rejects_mutated_stored_payload() -> None:
    payload = StoredPayload.inline_json({"value": 1})
    value = payload.value
    assert isinstance(value, dict)
    value["value"] = 2

    with pytest.raises(AIError) as raised:
        _encode_persisted_domain(payload)
    assert raised.value.code is ErrorCode.INTERNAL_ERROR
    assert raised.value.retryable is False
    assert raised.value.safe_details == {
        "phase": "persistence_encode",
        "domain_type": "StoredPayload",
    }


def test_missing_defaulted_session_field_uses_constructor_default() -> None:
    session = _session()
    payload = copy.deepcopy(_encode_persisted_domain(session))
    payload["fields"].pop("history_id")
    assert _decode_enveloped_domain(
        _envelope(payload),
        SessionRecord,
    ) == replace(session, history_id=None)


def test_missing_required_session_field_is_integrity_error() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["fields"].pop("status")
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_unknown_historical_field_is_ignored() -> None:
    session = _session()
    payload = copy.deepcopy(_encode_persisted_domain(session))
    payload["fields"]["removed_field"] = {"not": "decoded"}
    assert _decode_enveloped_domain(
        _envelope(payload),
        SessionRecord,
    ) == session


def test_malformed_known_field_is_integrity_error() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["fields"]["revision"] = "not-an-int"
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_generic_payload_without_or_with_schema_one_is_readable() -> None:
    session = _session()
    with_schema = _envelope(_encode_persisted_domain(session))
    without_schema = copy.deepcopy(with_schema)
    without_schema["value"]["payload"].pop("schema")
    assert _decode_enveloped_domain(with_schema, SessionRecord) == session
    assert _decode_enveloped_domain(without_schema, SessionRecord) == session


@pytest.mark.parametrize("schema", [True, 0, -1, "1"])
def test_malformed_generic_schema_is_integrity_error(schema: object) -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["schema"] = schema
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_positive_unknown_generic_schema_is_unsupported() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["schema"] = 2
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_unknown_outer_version_is_unsupported() -> None:
    value = _envelope(_encode_persisted_domain(_session()))
    value["v"] = 2
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(value, SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_unknown_wire_type_is_unsupported() -> None:
    value = _envelope(_encode_persisted_domain(_session()), wire_id="future_record")
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(value, SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_known_wire_type_with_wrong_target_is_integrity_error() -> None:
    value = _envelope(_encode_persisted_domain(_session()))
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(value, RunRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_reserved_dataclass_tag_cannot_be_combined() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["$mapping"] = {"items": []}
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_envelope_metadata_is_not_accepted() -> None:
    value = _envelope(_encode_persisted_domain(_session()))
    value["trace_id"] = "old"
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(value, SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    reserved = copy.deepcopy(value)
    reserved["$tuple"] = []
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(reserved, SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_typed_envelope_rejects_extra_framing_field() -> None:
    value = _envelope(_encode_persisted_domain(_session()))
    typed = cast("dict[str, object]", value["value"])
    typed["future_metadata"] = {"$future_v2": ["must", "not", "decode"]}
    with pytest.raises(AIError) as raised:
        _validate_enveloped_value(value)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_current_enum_value_round_trip_and_unknown_value_boundary() -> None:
    session = _session()
    payload = copy.deepcopy(_encode_persisted_domain(session))
    assert _decode_enveloped_domain(_envelope(payload), SessionRecord) == session

    payload["fields"]["status"]["value"] = "FUTURE"
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_explicit_custom_task_node_tolerates_additive_field() -> None:
    node = TaskNode("node", (), input={"key": "value"}, budget_cost=1)
    payload = encode_domain(node)
    payload["fields"]["extra"] = {"must": "not be decoded"}
    assert decode_domain(payload, TaskNode) == node


def test_agent_binding_snapshot_ignores_unknown_ordinary_field() -> None:
    payload = _binding_snapshot_payload()
    payload["future_metadata"] = {"$future_v2": ["must", "not", "be", "interpreted"]}
    decoded = AgentBindingSnapshot.from_payload(payload)
    assert decoded.to_payload() == _binding_snapshot_payload()


@pytest.mark.parametrize(
    "missing",
    (
        "version",
        "agent_spec",
        "agent_digest",
        "output_schema_id",
        "output_schema_revision",
        "output_schema_fingerprint",
        "local_runtime_capability_descriptors",
        "binding_digest",
        "global_runtime_capability_descriptors",
    ),
)
def test_agent_binding_snapshot_requires_current_fields(missing: str) -> None:
    payload = _binding_snapshot_payload()
    payload.pop(missing)
    with pytest.raises(AIError) as raised:
        AgentBindingSnapshot.from_payload(payload)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_low_level_record_shape_remains_strict() -> None:
    record = StoredPayload.inline_bytes(b"payload")
    stored = {
        "key": "6b" * 32,
        "partition": "70" * 32,
        "scope": None,
        "parent": None,
        "kind": "payload",
        "sort": "payload",
        "state": None,
        "storage_version": 0,
        "lease": {"owner": None, "fence": 0, "expires_at": None},
        "data": {"value": record.to_json()},
    }
    stored.pop("kind")
    with pytest.raises(AIError) as raised:
        decode_record(stored)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_step_persistence_keeps_schema_one_and_reads_unversioned_payload() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = RunRecord(
        run_id="run",
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="agent",
        metadata={},
        started_at=now,
    )
    current = _decode_step_envelope(
        {
            "v": 1,
            "value": {
                "type": wire_type_id(run),
                "payload": _encode_persisted_domain(run),
            },
        }
    )
    assert current == run
