#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Systemic compatibility contract for Runtime persisted dataclasses."""

import copy
import json
from pathlib import Path

import pytest

from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state._codec import (
    _CURRENT_CODEC,
    _DataclassPersistenceContract,
    _apply_persisted_upgrades,
    _decode_enveloped_domain,
    _decode_step_envelope,
    _encode_step_envelope,
    _runtime_persistence_manifest,
    decode_domain,
    encode_domain,
    encode_envelope,
    wire_type_id,
)
from linktools.ai.runtime.state._contracts import ConversationCursor
from linktools.ai.agent import AgentBindingSnapshot
from linktools.ai.agent._output import bind_output
from linktools.ai.spec import AgentSpec
from pydantic_ai_harness.step_persistence import RunRecord
from datetime import datetime, timezone


def _legacy_cursor_envelope() -> dict[str, object]:
    return {
        "v": 1,
        "value": {
            "type": "conversation_cursor",
            "payload": {
                "$dataclass": "conversation_cursor",
                "fields": {
                    "step_run_id": "run",
                    "history_id": None,
                    "message_count": 0,
                },
            },
        },
    }


def test_semantic_encoding_remains_unversioned_and_byte_stable() -> None:
    cursor = ConversationCursor("run", None, 0)
    assert encode_domain(cursor) == {
        "$dataclass": "conversation_cursor",
        "fields": {
            "step_run_id": "run",
            "history_id": None,
            "message_count": 0,
        },
    }
    assert decode_domain(encode_domain(cursor), ConversationCursor) == cursor


def test_persistence_envelope_tags_dataclass_schema_without_changing_wire_major() -> None:
    cursor = ConversationCursor("run", None, 0)
    envelope = encode_envelope(
        {"type": wire_type_id(cursor), "payload": encode_domain(cursor)}
    )
    assert envelope["v"] == 1
    payload = envelope["value"]["payload"]
    assert payload["$dataclass"] == "conversation_cursor"
    assert payload["schema"] == 1
    assert set(payload) == {"$dataclass", "schema", "fields"}
    assert _decode_enveloped_domain(envelope, ConversationCursor) == cursor


def test_legacy_unversioned_v1_remains_readable() -> None:
    envelope = _legacy_cursor_envelope()
    assert _decode_enveloped_domain(envelope, ConversationCursor) == ConversationCursor(
        "run", None, 0
    )


def test_unknown_schema_is_unsupported_but_malformed_known_schema_is_integrity() -> None:
    envelope = encode_envelope(
        {
            "type": "conversation_cursor",
            "payload": encode_domain(ConversationCursor("run", None, 0)),
        }
    )
    future = copy.deepcopy(envelope)
    future["value"]["payload"]["schema"] = 99
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(future, ConversationCursor)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED

    malformed = copy.deepcopy(envelope)
    malformed["value"]["payload"]["fields"].pop("history_id")
    with pytest.raises(AIError) as raised:
        _decode_enveloped_domain(malformed, ConversationCursor)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR

    for invalid in (True, 0, -1, "1"):
        malformed_revision = copy.deepcopy(envelope)
        malformed_revision["value"]["payload"]["schema"] = invalid
        with pytest.raises(AIError) as raised:
            _decode_enveloped_domain(malformed_revision, ConversationCursor)
        assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_adjacent_upgrade_chain_is_explicit_and_pure() -> None:
    def one_to_two(raw, _codec):
        assert set(raw) == {"old"}
        return {"new": raw["old"]}

    contract = _DataclassPersistenceContract(
        fingerprints={1: "a", 2: "b"},
        upgrades={1: one_to_two},
        legacy_revision=1,
    )
    source = {"old": "value"}
    assert _apply_persisted_upgrades(source, 1, contract, _CURRENT_CODEC) == {
        "new": "value"
    }
    assert source == {"old": "value"}


def test_step_persistence_writes_schema_and_reads_legacy_unversioned() -> None:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    run = RunRecord(
        run_id="run",
        conversation_id="conversation",
        parent_run_id=None,
        agent_name="agent",
        metadata={},
        started_at=now,
    )
    current = _encode_step_envelope(run)
    assert current["value"]["payload"]["schema"] == 1
    assert _decode_step_envelope(current) == run

    legacy = copy.deepcopy(current)
    legacy["value"]["payload"].pop("schema")
    assert _decode_step_envelope(legacy) == run


def _binding_snapshot_payload() -> dict[str, object]:
    output = bind_output()
    snapshot = AgentBindingSnapshot(
        version=1,
        agent_spec=AgentSpec("agent", 1, "default"),
        agent_digest="b" * 64,
        output_type_module=output.value_type.__module__,
        output_type_qualname=output.value_type.__qualname__,
        output_schema_id=output.schema_id,
        output_schema_revision=output.schema_revision,
        output_schema_fingerprint=output.schema_fingerprint,
        local_runtime_capability_descriptors=(),
        binding_digest="a" * 64,
    )
    return snapshot.to_payload()


def test_agent_binding_snapshot_future_version_is_unsupported() -> None:
    payload = _binding_snapshot_payload()
    assert AgentBindingSnapshot.from_payload(payload).version == 1
    future = dict(payload)
    future["version"] = 2
    with pytest.raises(AIError) as raised:
        AgentBindingSnapshot.from_payload(future)
    assert raised.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    malformed = dict(payload)
    malformed["version"] = True
    with pytest.raises(AIError) as raised:
        AgentBindingSnapshot.from_payload(malformed)
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_checked_in_runtime_persistence_manifest_matches_codec_contract() -> None:
    path = (
        Path(__file__).resolve().parents[2]
        / "linktools-ai/scripts/build/matrix/runtime-persistence-v1.json"
    )
    assert json.loads(path.read_text(encoding="utf-8")) == _runtime_persistence_manifest()
