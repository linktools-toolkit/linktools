#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Record reads validate raw boundaries without re-encoding decoded bodies."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import datetime, timezone
from types import MappingProxyType

import pytest

from linktools.ai.core import ImmutableJsonMapping, JsonValue
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime.state import RuntimeDomain, _model_interaction_store, _store
from linktools.ai.runtime.state._codec import decode_record, encode_record
from linktools.ai.runtime.state._memory import InMemoryStateStore
from linktools.ai.runtime.state._model_interaction_store import ModelInteractionStateStepArchive
from linktools.ai.runtime.state._sql import _record_from_row, _record_values
from linktools.ai.runtime.state._store import StoredRecord, validate_record_identity

from .test_model_interaction_records import _prepared, _running, _terminal


def _record(data: Mapping[str, JsonValue]) -> StoredRecord:
    return StoredRecord(
        b"k" * 32, None, None, "record", "r:1", "ACTIVE", 0, None, 0, None, data,
    )


@pytest.mark.parametrize("source", ("constructor", "replacement", "filesystem", "sql"))
def test_record_identity_reuses_canonical_immutable_data(
    source: str, monkeypatch: pytest.MonkeyPatch,
) -> None:
    data = {"value": [{"nested": [1, True, None, "text"]}]}
    record = _record(data)
    if source == "replacement":
        record = replace(record, storage_version=1)
    elif source == "filesystem":
        record = decode_record(encode_record(record))
    elif source == "sql":
        record = _record_from_row(_record_values(record))
    expected = encode_record(record)
    data["value"][0]["nested"].append("source mutation")
    record.data["value"][0]["nested"].append("read mutation")
    materializations = 0
    serializations = 0
    getitem = ImmutableJsonMapping.__getitem__
    canonical = _store.canonical_json_bytes

    def read_value(self, key):
        nonlocal materializations
        materializations += 1
        return getitem(self, key)

    def serialize(value):
        nonlocal serializations
        serializations += 1
        return canonical(value)

    with monkeypatch.context() as patch:
        patch.setattr(ImmutableJsonMapping, "__getitem__", read_value)
        patch.setattr(_store, "canonical_json_bytes", serialize)
        validate_record_identity(record)
    assert (materializations, serializations) == (0, 0)
    assert encode_record(record) == expected


@pytest.mark.parametrize("source", ("constructor", "filesystem", "sql"))
@pytest.mark.parametrize("data", (
    {"value": float("nan")},
    {"value": float("inf")},
    {"value": b"bytes"},
    {"value": (1, 2)},
    {"value": {1: "non-string key"}},
    {"value": MappingProxyType({"nested": 1})},
    {"": "empty root key"},
))
def test_record_construction_rejects_malformed_data(
    source: str, data: Mapping[str, JsonValue],
) -> None:
    if source == "constructor":
        with pytest.raises((TypeError, ValueError)):
            _record(data)
        return
    record = _record({"value": None})
    with pytest.raises(AIError) as raised:
        if source == "filesystem":
            decode_record({**encode_record(record), "data": data})
        else:
            _record_from_row({**_record_values(record), "payload_json": data})
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_record_identity_still_validates_custom_mapping_behavior() -> None:
    class CustomMapping(ImmutableJsonMapping):
        def __getitem__(self, key):
            return float("nan")

    record = _record(CustomMapping({"value": None}))
    with pytest.raises(ValueError, match="record data is not canonical JSON"):
        validate_record_identity(record)


@pytest.fixture
def interaction_archive() -> ModelInteractionStateStepArchive:
    return ModelInteractionStateStepArchive(
        InMemoryStateStore(), object_store=None, namespace="reads", tenant_id="tenant",
        runtime_domain=RuntimeDomain.RECOVERY,
    )


def test_interaction_read_does_not_encode_a_second_body(
    interaction_archive: ModelInteractionStateStepArchive, monkeypatch: pytest.MonkeyPatch,
) -> None:
    value = _terminal(_prepared(_running()))
    record = interaction_archive._stored_interaction(value)
    encodes = 0
    encode = _model_interaction_store._encode_step

    def encode_step(value):
        nonlocal encodes
        encodes += 1
        return encode(value)

    monkeypatch.setattr(_model_interaction_store, "_encode_step", encode_step)
    assert interaction_archive._decode_interaction_record(record) == value
    assert encodes == 0


@pytest.mark.parametrize("field,value", (
    ("key_digest", b"x" * 32),
    ("scope_digest", b"x" * 32),
    ("parent_digest", b"x" * 32),
    ("kind", "other"),
    ("sort_key", "m:2"),
    ("state", "FAILED"),
    ("lease_owner", "worker"),
    ("lease_fence", 1),
    ("lease_expires_at", datetime(2026, 1, 1, tzinfo=timezone.utc)),
))
def test_interaction_read_rejects_metadata_drift(
    interaction_archive: ModelInteractionStateStepArchive, field: str, value: object,
) -> None:
    record = interaction_archive._stored_interaction(_running())
    with pytest.raises(AIError) as raised:
        interaction_archive._decode_interaction_record(replace(record, **{field: value}))
    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("corruption", ("schema", "required", "scalar", "payload"))
def test_interaction_read_still_validates_the_complete_body(
    interaction_archive: ModelInteractionStateStepArchive, corruption: str,
) -> None:
    record = interaction_archive._stored_interaction(_prepared(_running()))
    data = dict(record.data)
    payload = data["value"]["payload"]
    fields = payload["fields"]
    expected = ErrorCode.STORAGE_INTEGRITY_ERROR
    if corruption == "schema":
        payload["schema"] += 1
        expected = ErrorCode.STORAGE_VERSION_UNSUPPORTED
    elif corruption == "required":
        del fields["model"]
    elif corruption == "scalar":
        fields["model_request_seq"] = False
    else:
        fields["request_envelope"]["fields"]["payload"]["fields"]["digest"] = "0" * 64
    with pytest.raises(AIError) as raised:
        interaction_archive._decode_interaction_record(replace(record, data=data))
    assert raised.value.code is expected
