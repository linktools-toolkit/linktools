"""Frozen V1 persistence protocol fixtures."""

import json
from pathlib import Path

import pytest
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._codec import (
    _V1_ENUM_WIRE_TYPES,
    _V1_WIRE_TYPES,
    CURRENT_DATA_VERSION,
    decode_alias,
    decode_domain,
    decode_envelope,
    decode_fact,
    decode_operation,
    decode_record,
    parse_envelope,
    wire_type_id,
)
from linktools.ai.runtime.state._contracts import (
    ConversationCursor,
    ExecutionHistoryState,
    TranscriptMessageRef,
)


def _fixture() -> dict[str, object]:
    path = Path(__file__).with_name("fixtures") / "runtime_state_v1_golden.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_v1_wire_registry_matches_golden_manifest() -> None:
    fixture = _fixture()
    assert fixture["version"] == CURRENT_DATA_VERSION
    assert fixture["wire_type_ids"] == [wire_id for wire_id, _ in _V1_WIRE_TYPES]
    assert fixture["enum_wire_type_ids"] == [
        wire_id for wire_id, _ in _V1_ENUM_WIRE_TYPES
    ]
    assert len(fixture["wire_type_ids"]) == len(set(fixture["wire_type_ids"]))

    for wire_id, target in _V1_WIRE_TYPES:
        assert wire_type_id(target) == wire_id


def test_golden_v1_envelopes_and_storage_primitives_decode() -> None:
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
    assert decode_record(primitives["record"]).kind == "golden"
    assert decode_fact(primitives["fact"]).sequence == 1
    assert decode_operation(primitives["operation"]).sequence == 1
    assert decode_alias(primitives["alias"]).record_key_digest == bytes.fromhex(
        "77" * 32
    )


def test_future_envelope_version_is_parseable_but_not_decoded_without_registry() -> None:
    future = {"v": CURRENT_DATA_VERSION + 1, "value": {"type": "future"}}
    assert parse_envelope(future).version == CURRENT_DATA_VERSION + 1
    with pytest.raises(AIError) as raised:
        decode_envelope(future)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
