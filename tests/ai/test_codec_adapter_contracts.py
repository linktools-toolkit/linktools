#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compact wire adapters preserve their independent reader/writer contracts."""

import pytest

from linktools.ai.errors import AIError, ErrorCode, ErrorDiagnostics
from linktools.ai.persistence import codec
from linktools.ai.persistence.contracts import StoredUserInput
from linktools.ai.storage import StoredPayload
from linktools.ai.task import TaskResultRecord


def _encode(value: object, persisted: bool):
    return codec._encode_domain(value, codec._CURRENT_CODEC, persisted=persisted)


def _decode(value, target: type, persisted: bool):
    return codec._decode_domain(value, target, codec._CURRENT_CODEC, persisted=persisted)


@pytest.mark.parametrize("registry", ("_V1_DATACLASS_ENCODERS", "_V1_DATACLASS_DECODERS"))
def test_wire_registry_accepts_independent_reader_and_writer_adapters(monkeypatch, registry: str) -> None:
    monkeypatch.setattr(codec, registry, {})
    codec._validate_v1_codec_definition()


@pytest.mark.parametrize("registry", ("_V1_DATACLASS_ENCODERS", "_V1_DATACLASS_DECODERS"))
def test_wire_registry_rejects_unknown_adapter_types(monkeypatch, registry: str) -> None:
    adapters = dict(getattr(codec, registry))
    adapters["unregistered_type"] = next(iter(adapters.values()))
    monkeypatch.setattr(codec, registry, adapters)
    with pytest.raises(RuntimeError, match="unknown type"):
        codec._validate_v1_codec_definition()


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("collected", (False, True))
def test_error_diagnostics_keep_compact_wire_and_optional_defaults(persisted: bool, collected: bool) -> None:
    diagnostics = ErrorDiagnostics(
        "ValueError", "invalid input", "a" * 64,
        exception_chain=({"exception_type": "ValueError", "exception_message": "invalid input", "relation": "root"},)
        if collected else None,
        traceback="trace" if collected else None,
        content_omission_reason="body omitted" if collected else None,
    )
    expected_fields = diagnostics.to_payload()
    if collected:
        expected_fields["exception_chain"] = _encode(diagnostics.exception_chain, persisted)
    expected = {"$dataclass": "error_diagnostics", "fields": expected_fields}
    if persisted:
        expected["schema"] = 1
    wire = _encode(diagnostics, persisted)
    assert wire == expected
    assert _decode(wire, ErrorDiagnostics, persisted) == diagnostics
    if not collected:
        wire["fields"].update(exception_chain=None, traceback=None, content_omission_reason=None)
        assert _decode(wire, ErrorDiagnostics, persisted) == diagnostics
    wire["fields"]["future_field"] = "ignored"
    assert _decode(wire, ErrorDiagnostics, persisted) == diagnostics


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("field,value", (
    ("exception_type", ""), ("cause_digest", "not-a-digest"),
    ("exception_chain", {"$tuple": ["invalid"]}), ("traceback", 1),
    ("content_omission_reason", ""),
))
def test_error_diagnostics_reject_explicit_invalid_fields(
    persisted: bool, field: str, value: object,
) -> None:
    wire = _encode(ErrorDiagnostics("ValueError", "message", "a" * 64), persisted)
    wire["fields"][field] = value
    with pytest.raises(AIError) as caught:
        _decode(wire, ErrorDiagnostics, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
def test_error_diagnostics_report_missing_required_fields(persisted: bool) -> None:
    wire = _encode(ErrorDiagnostics("ValueError", "message", "a" * 64), persisted)
    del wire["fields"]["cause_digest"]
    with pytest.raises(AIError) as caught:
        _decode(wire, ErrorDiagnostics, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    assert caught.value.safe_details == {
        "wire_type": "error_diagnostics", "missing_fields": ["cause_digest"],
    }


@pytest.mark.parametrize("persisted", (False, True))
def test_task_result_writer_keeps_wire_and_reader_rejects_extra_fields(persisted: bool) -> None:
    record = TaskResultRecord("graph", "node", "b" * 64, "execution")
    expected = {
        "$dataclass": "task_result",
        "fields": {"graph_id": "graph", "node_id": "node", "result_digest": "b" * 64, "execution_id": "execution"},
    }
    if persisted:
        expected["schema"] = 1
    wire = _encode(record, persisted)
    assert wire == expected
    assert _decode(wire, TaskResultRecord, persisted) == record
    wire["fields"]["future_field"] = "not permitted"
    with pytest.raises(AIError) as caught:
        _decode(wire, TaskResultRecord, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    del wire["fields"]["future_field"]
    del wire["fields"]["execution_id"]
    with pytest.raises(AIError) as caught:
        _decode(wire, TaskResultRecord, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
@pytest.mark.parametrize("view", (None, {"version": 1, "label": "value"}))
def test_stored_user_input_view_is_optional_but_explicit_values_are_validated(persisted: bool, view) -> None:
    stored = StoredUserInput("text", StoredPayload.inline_text("prompt"), view)
    wire = _encode(stored, persisted)
    assert _decode(wire, StoredUserInput, persisted) == stored
    wire["fields"]["view"] = _encode(view, persisted)
    assert _decode(wire, StoredUserInput, persisted) == stored
    del wire["fields"]["view"]
    assert _decode(wire, StoredUserInput, persisted) == StoredUserInput("text", stored.payload)
    wire["fields"]["view"] = ["not a mapping"]
    with pytest.raises(AIError) as caught:
        _decode(wire, StoredUserInput, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("persisted", (False, True))
def test_stored_user_input_checks_required_fields_before_codec_and_codec_before_view(persisted: bool) -> None:
    wire = _encode(StoredUserInput("text", StoredPayload.inline_text("prompt")), persisted)
    wire["fields"]["codec"] = "future-codec"
    wire["fields"]["view"] = ["invalid"]
    with pytest.raises(AIError) as caught:
        _decode(wire, StoredUserInput, persisted)
    assert caught.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    del wire["fields"]["payload"]
    with pytest.raises(AIError) as caught:
        _decode(wire, StoredUserInput, persisted)
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
