#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime persistence compatibility contracts."""

import copy
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from linktools.ai.core import SessionStatus
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._codec import (
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_persisted_domain,
    wire_type_id,
)
from linktools.ai.runtime.state import RuntimeDomain
from linktools.ai.runtime.state._contracts import (
    ContextProjection,
    ModelInteractionRecord,
    RuntimePayloadRef,
    SessionRecord,
)
from linktools.ai.storage import StoredPayload
from linktools.ai.runtime.state._step_contracts import RunRecord
from linktools.ai.task import TaskNode


def _session() -> SessionRecord:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return SessionRecord(
        session_id="session",
        tenant_id="tenant",
        owner_principal_id="owner",
        agent_id="agent",
        status=SessionStatus.OPEN,
        revision=0,
        cwd=None,
        metadata={},
        created_at=now,
        updated_at=now,
        closed_at=None,
        active_execution_id=None,
        history_id="history",
    )


def _envelope(payload: object, *, wire_id: str = "session_record") -> dict[str, object]:
    return {"v": 1, "value": {"type": wire_id, "payload": payload}}


def test_persisted_session_round_trips() -> None:
    session = _session()
    payload = _encode_persisted_domain(session)

    assert payload["schema"] == 1
    assert _decode_enveloped_domain(_envelope(payload), SessionRecord) == session


def test_persisted_session_allows_additive_and_defaulted_fields() -> None:
    session = _session()
    additive = copy.deepcopy(_encode_persisted_domain(session))
    additive["fields"]["future_metadata"] = {"future": True}
    assert _decode_enveloped_domain(_envelope(additive), SessionRecord) == session

    defaulted = copy.deepcopy(_encode_persisted_domain(session))
    defaulted["fields"].pop("history_id")
    assert _decode_enveloped_domain(_envelope(defaulted), SessionRecord) == replace(
        session,
        history_id=None,
    )


def test_persisted_session_rejects_missing_required_field() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["fields"].pop("status")

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_persisted_session_rejects_malformed_known_field() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["fields"]["revision"] = "not-an-int"

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize(
    ("schema", "expected"),
    (
        (0, ErrorCode.STORAGE_INTEGRITY_ERROR),
        (2, ErrorCode.STORAGE_VERSION_UNSUPPORTED),
    ),
)
def test_persisted_schema_version_boundaries(schema: object, expected: ErrorCode) -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["schema"] = schema

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)

    assert raised.value.code is expected


def test_persisted_payload_requires_schema() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload.pop("schema")

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_unknown_outer_version_and_wire_type_are_unsupported() -> None:
    payload = _encode_persisted_domain(_session())
    future_version = _envelope(payload)
    future_version["v"] = 2
    with pytest.raises(AIError) as version_error:
        _decode_enveloped_domain(future_version, SessionRecord)
    assert version_error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

    with pytest.raises(AIError) as type_error:
        _decode_enveloped_domain(
            _envelope(payload, wire_id="future_record"),
            SessionRecord,
        )
    assert type_error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_known_wire_type_with_wrong_target_is_integrity_error() -> None:
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(
            _envelope(_encode_persisted_domain(_session())),
            RunRecord,
        )

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_unknown_enum_value_is_unsupported() -> None:
    payload = copy.deepcopy(_encode_persisted_domain(_session()))
    payload["fields"]["status"]["value"] = "FUTURE"

    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(_envelope(payload), SessionRecord)

    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED


def test_persisted_custom_dataclass_allows_additive_field() -> None:
    node = TaskNode("node", (), input={"key": "value"}, budget_cost=1)
    payload = copy.deepcopy(_encode_persisted_domain(node))
    payload["fields"]["future_metadata"] = {"future": True}

    assert _decode_enveloped_domain(
        _envelope(payload, wire_id="task_node"),
        TaskNode,
    ) == node


def test_persisted_model_interaction_defaults_legacy_attachments() -> None:
    interaction = ModelInteractionRecord(
        run_id="run",
        step_index=1,
        request_sequence=1,
        purpose="agent",
        output_retry_index=None,
        model={"route_id": "default"},
        request_context=ContextProjection(()),
        request_envelope=RuntimePayloadRef(
            StoredPayload.inline_bytes(b"{}"),
            RuntimeDomain.EXECUTION,
        ),
        response_context=None,
        status="CANCELLED",
        error_code=None,
        duration_ns=1,
        usage=None,
        attachments=(
            {
                "fact": "included_in_request",
                "attachment_id": "a" * 64,
                "source": "binary",
                "media_type": "image/png",
                "size": 4,
                "digest": "b" * 64,
                "content_key": "b" * 64,
                "position": 0,
                "call_id": None,
            },
        ),
    )
    payload = copy.deepcopy(_encode_persisted_domain(interaction))
    payload["fields"].pop("attachments")

    decoded = _decode_enveloped_domain(
        _envelope(payload, wire_id="model_interaction"),
        ModelInteractionRecord,
    )

    assert decoded == replace(interaction, attachments=())


def test_step_persistence_reads_current_payload() -> None:
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
