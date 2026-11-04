#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Native message fields and host provenance survive durable round trips."""

import hashlib
import json
from dataclasses import fields, replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from pathlib import Path

import pytest
from pydantic_ai import PydanticAIDeprecationWarning, RequestUsage, messages as sdk

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.core import canonical_json_bytes
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.model.messages import (
    RetiredReferencedContent, SummaryPromptPart, decode_model_messages,
    encode_model_messages,
)
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

_STAMP = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _response() -> sdk.ModelResponse:
    return sdk.ModelResponse(parts=[sdk.TextPart("answer")], timestamp=_STAMP)


def test_native_response_fields_and_provider_usage_are_preserved() -> None:
    identity = {key: value for key, value in {
        "model_id": "configured:secondary", "canonical_system": "azure",
    }.items() if key in {field.name for field in fields(sdk.ModelResponse)}}
    response = replace(
        _response(), **identity, model_name="provider-model", history_digest="d" * 64,
        workspace_ref=sdk.WorkspaceRef(provider="test", id="workspace-fixed"),
        provider_response_id="response-fixed",
        usage=RequestUsage(input_tokens=3, provider_extra=7),
        provider_details={"opaque": {"sequence": [1, None, False]}},
    )
    restored, = decode_model_messages(encode_model_messages((response,)))
    assert restored == response
    assert restored.usage.provider_extra == 7
    for key, value in identity.items():
        assert getattr(restored, key) == value
    assert restored.model_name == "provider-model"


def test_metadata_does_not_supply_model_identity_or_workspace_authority() -> None:
    response = replace(_response(), metadata={
        "model_id": "spoof", "canonical_system": "spoof",
        "workspace_ref": {"provider": "spoof", "id": "spoof"},
    })
    restored, = decode_model_messages(encode_model_messages((response,)))
    assert getattr(restored, "model_id", None) is None
    assert getattr(restored, "canonical_system", None) is None
    assert restored.workspace_ref is None
    assert restored.metadata == response.metadata


def test_provider_continuation_still_detects_changed_history_after_restore() -> None:
    # The native decision has no public offline inspection API; no provider is called.
    from pydantic_ai.models.openai import OpenAIResponsesModel, _history_hasher, _with_history_digest

    original = [sdk.ModelRequest(parts=[sdk.UserPromptPart("original", timestamp=_STAMP)])]
    altered = [sdk.ModelRequest(parts=[sdk.UserPromptPart("redacted", timestamp=_STAMP)])]
    import inspect
    options = {"with_instructions": False} if "with_instructions" in inspect.signature(_history_hasher).parameters else {}
    response = _with_history_digest(
        replace(_response(), provider_response_id="response-fixed"),
        _history_hasher(original, None, **options),
    )
    owner = SimpleNamespace(_history_digest_key=None)
    restored, = decode_model_messages(encode_model_messages((response,)))
    assert not OpenAIResponsesModel._history_changed(owner, restored, original)
    assert OpenAIResponsesModel._history_changed(owner, restored, altered)
    assert restored.provider_response_id == response.provider_response_id


def test_summary_reference_retirement_and_tool_values_keep_host_provenance() -> None:
    reference = ReferencedContent(
        AssetVersionRef(AssetKey("document", "fixture"), "configured", StorageEntryRevision(1),
                        hashlib.sha256(b"fixture").hexdigest(), 7),
        media_type="application/pdf", source=CapabilityRef("capability", "files", 1),
        source_parameters={"version": "fixed"},
    )
    retired = RetiredReferencedContent(reference, "retired attachment")
    binary = sdk.BinaryContent(b"\x00\xff", media_type="image/png")
    with pytest.warns(PydanticAIDeprecationWarning, match=r"`RetryPromptPart` is deprecated"):
        retry = sdk.RetryPromptPart("correct output", tool_call_id="retry-fixed", timestamp=_STAMP)
    request = sdk.ModelRequest(parts=[
        SummaryPromptPart("trusted summary", timestamp=_STAMP),
        sdk.UserPromptPart([reference, retired, binary], timestamp=_STAMP),
        sdk.ToolReturnPart("read", [reference, {"ordinary": "value"}],
                           tool_call_id="call-fixed", timestamp=_STAMP),
        retry,
    ], timestamp=_STAMP)
    with pytest.warns(PydanticAIDeprecationWarning, match=r"`RetryPromptPart` is deprecated"):
        restored, = decode_model_messages(encode_model_messages((request,)))
    assert type(restored.parts[0]) is SummaryPromptPart
    assert restored.parts[0] == request.parts[0]
    content = restored.parts[1].content
    assert type(content[0]) is ReferencedContent and content[0] == reference
    assert type(content[1]) is RetiredReferencedContent and content[1] == retired
    assert (content[2].data, content[2].media_type, content[2].identifier) == (
        binary.data, binary.media_type, binary.identifier,
    )
    assert restored.parts[2] == request.parts[2]
    assert type(restored.parts[3]) is sdk.RetryPromptPart
    assert restored.parts[3] == request.parts[3]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_writer_rejects_nonfinite_metadata_before_json_normalization(bad: float) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((replace(_response(), provider_details={"value": bad}),))


@pytest.mark.parametrize("bad", [-1, True])
def test_standard_usage_counters_are_nonnegative_integers_on_write_and_read(bad: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((replace(_response(), usage=RequestUsage(input_tokens=bad)),))
    wire = json.loads(encode_model_messages((_response(),)))
    wire[0]["usage"]["input_tokens"] = bad
    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("invalid", ["duration", "timestamp"])
def test_attempt_accounting_rejects_negative_duration_and_naive_timestamp(invalid: str) -> None:
    attempt = sdk.ModelRequestAttempt(
        model_name="model", outcome="error",
        timestamp=_STAMP.replace(tzinfo=None) if invalid == "timestamp" else _STAMP,
        duration=timedelta(seconds=-1) if invalid == "duration" else timedelta(),
    )
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((replace(_response(), failed_attempts=[attempt]),))
    valid = replace(attempt, timestamp=_STAMP, duration=timedelta())
    wire = json.loads(encode_model_messages((replace(_response(), failed_attempts=[valid]),)))
    wire[0]["failed_attempts"][0][invalid] = (
        "-PT1S" if invalid == "duration" else "2026-01-01T00:00:00"
    )
    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_instruction_digest_rejects_invalid_hash_on_write_and_read() -> None:
    bad = sdk.InstructionDigest(content_hash="not-a-sha256")
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((sdk.ModelRequest(parts=[sdk.InstructionBaselinePart(instructions=[bad])]),))
    valid = replace(bad, content_hash="a" * 64)
    wire = json.loads(encode_model_messages((sdk.ModelRequest(parts=[
        sdk.InstructionBaselinePart(instructions=[valid]),
    ]),)))
    wire[0]["parts"][0]["instructions"][0]["content_hash"] = bad.content_hash
    with pytest.raises(AIError):
        decode_model_messages(canonical_json_bytes(wire))


def test_instruction_source_is_not_invented_or_erased() -> None:
    invalid = sdk.InstructionPart("instruction", id="agent")
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((sdk.ModelRequest(parts=[sdk.InstructionBaselinePart(instructions=[invalid])]),))
    wire = json.loads(encode_model_messages((sdk.ModelRequest(parts=[
        sdk.InstructionDeltaPart(id=None, content="instruction"),
    ]),)))
    wire[0]["parts"][0]["id"] = "future:unknown"
    with pytest.raises(AIError):
        decode_model_messages(canonical_json_bytes(wire))


def test_retry_budget_boolean_is_not_converted_to_an_integer() -> None:
    bad = sdk.RetryFeedbackPart("retry", retry=sdk.RetryInfo(cause="model_retry", attempt=True))
    with pytest.raises((TypeError, ValueError)):
        encode_model_messages((sdk.ModelRequest(parts=[bad]),))
    good = replace(bad, retry=sdk.RetryInfo(cause="model_retry", attempt=1))
    wire = json.loads(encode_model_messages((sdk.ModelRequest(parts=[good]),)))
    wire[0]["parts"][0]["retry"]["attempt"] = True
    with pytest.raises(AIError):
        decode_model_messages(canonical_json_bytes(wire))


@pytest.mark.parametrize("location", ["message", "part"])
def test_unknown_native_fields_are_not_silently_discarded(location: str) -> None:
    wire = json.loads(encode_model_messages((_response(),)))
    target = wire[0] if location == "message" else wire[0]["parts"][0]
    target["future_semantic_field"] = {"value": "preserve or reject"}
    with pytest.raises(AIError):
        decode_model_messages(canonical_json_bytes(wire))


def test_native_authoring_default_does_not_replace_missing_durable_tool_identity() -> None:
    response = replace(_response(), parts=[sdk.ToolCallPart("tool", tool_call_id="call-fixed")])
    wire = json.loads(encode_model_messages((response,)))
    del wire[0]["parts"][0]["tool_call_id"]
    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("container", [list, tuple])
def test_reference_sequences_do_not_degrade_to_display_text(container: type) -> None:
    reference = ReferencedContent(
        AssetVersionRef(AssetKey("document", "sequence"), "fixed", StorageEntryRevision(1),
                        hashlib.sha256(b"x").hexdigest(), 1),
        media_type="text/plain", source=CapabilityRef("capability", "files", 1),
    )
    message = sdk.ModelRequest([sdk.UserPromptPart(container([reference]))])
    restored, = decode_model_messages(encode_model_messages([message]))
    assert restored.parts[0].content[0] == reference
    assert type(restored.parts[0].content[0]) is ReferencedContent


@pytest.mark.parametrize("cost", ["NaN", "Infinity", "-Infinity"])
def test_nonfinite_usage_cost_is_rejected_before_a_durable_write(cost: str) -> None:
    from decimal import Decimal
    with pytest.raises(ValueError):
        encode_model_messages([replace(_response(), usage=RequestUsage(cost=Decimal(cost)))])


@pytest.mark.parametrize("content,expected", [({1: "one"}, {"1": "one"}), (b"abc", "YWJj")])
def test_message_tool_results_keep_the_host_snapshot_contract(content: object, expected: object) -> None:
    request = sdk.ModelRequest([sdk.ToolReturnPart("tool", content, tool_call_id="call")])
    restored, = decode_model_messages(encode_model_messages([request]))
    assert restored.parts[0].content == expected
    assert restored.parts[0].tool_call_id == "call"


def test_native_codec_validation_does_not_echo_message_values() -> None:
    import traceback

    wire = json.loads(encode_model_messages([_response()]))
    wire[0]["model_id"] = {"private": "sensitive-codec-value"}
    with pytest.raises(AIError) as error:
        decode_model_messages(canonical_json_bytes(wire))
    assert "sensitive-codec-value" not in "".join(traceback.format_exception(error.value))



def test_retry_round_trip_does_not_require_optional_recorded_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    # The declared 2.61 lower bound has retry feedback but no recorded-settings part.
    import builtins
    import importlib.util
    import sys
    from linktools.ai.model import messages as codec

    original_import = builtins.__import__
    def without_recorded_settings(name, globals=None, locals=None, fromlist=(), level=0):
        if name == "pydantic_ai.messages" and "ModelSettingsDeltaPart" in fromlist:
            raise ImportError("optional recorded settings are unavailable")
        return original_import(name, globals, locals, fromlist, level)

    module_name = "linktools.ai.model.retry_without_settings_codec"
    spec = importlib.util.spec_from_file_location(module_name, codec.__file__)
    isolated = importlib.util.module_from_spec(spec)
    with monkeypatch.context() as imports:
        imports.setitem(sys.modules, module_name, isolated)
        imports.setattr(builtins, "__import__", without_recorded_settings)
        spec.loader.exec_module(isolated)
        request = sdk.ModelRequest([sdk.RetryFeedbackPart(
            "retry", retry=sdk.RetryInfo(cause="model_retry", attempt=1, max_retries=2), timestamp=_STAMP,
        )], timestamp=_STAMP)
        assert isolated.decode_model_messages(isolated.encode_model_messages([request])) == (request,)



def test_unaddressable_instruction_source_is_rejected_before_writing() -> None:
    source_type = getattr(sdk, "RunInstructionSource", None)
    if source_type is None:
        pytest.skip("installed SDK has no run instruction source")
    # This native serializer emits bare 'run', but its reader only addresses 'run:name'.
    identity = sdk.InstructionId(source_type())
    request = sdk.ModelRequest([sdk.InstructionDeltaPart(id=identity, content="instruction")])
    with pytest.raises(AIError) as error:
        encode_model_messages([request])
    assert error.value.code is ErrorCode.STORAGE_VERSION_UNSUPPORTED
    named = replace(request, parts=[replace(request.parts[0], id=replace(identity, name="step"))])
    assert decode_model_messages(encode_model_messages([named])) == (named,)


def test_namespaced_tool_kind_survives_registered_and_fresh_reader(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys
    from typing_extensions import TypedDict

    if not hasattr(sdk, "ToolKind"):
        pytest.skip("installed native SDK predates namespaced tool kinds")

    class LookupArgs(TypedDict):
        key: str

    class LookupResult(TypedDict):
        count: int

    kind = sdk.ToolKind("codec_contract", "lookup", args=LookupArgs, content=LookupResult)

    class LookupCall(sdk.ToolCallPart[LookupArgs], kind=kind):
        pass

    class LookupReturn(sdk.ToolReturnPart[LookupResult], kind=kind):
        pass

    messages = (
        sdk.ModelResponse([LookupCall("renamed_lookup", {"key": "fixture"}, tool_call_id="call-fixed")],
                          timestamp=_STAMP),
        sdk.ModelRequest([LookupReturn("renamed_lookup", {"count": 3}, tool_call_id="call-fixed",
                                      timestamp=_STAMP)], timestamp=_STAMP),
    )
    wire = encode_model_messages(messages)
    restored = decode_model_messages(wire)
    assert isinstance(restored[0].parts[0], LookupCall)
    assert isinstance(restored[1].parts[0], LookupReturn)
    assert encode_model_messages(restored) == wire
    path = tmp_path / "tool-kind.json"
    path.write_bytes(wire)
    script = '''
import sys
from pathlib import Path
from typing_extensions import TypedDict
from pydantic_ai import messages as sdk
from linktools.ai.model.messages import decode_model_messages, encode_model_messages
wire = Path(sys.argv[1]).read_bytes()
assert sdk.parse_tool_kind("codec_contract.lookup") is None
messages = decode_model_messages(wire)
assert type(messages[0].parts[0]) is sdk.ToolCallPart
assert type(messages[1].parts[0]) is sdk.ToolReturnPart
assert [message.parts[0].tool_kind for message in messages] == ["codec_contract.lookup"] * 2
assert [message.parts[0].tool_call_id for message in messages] == ["call-fixed"] * 2
assert messages[0].parts[0].args == {"key": "fixture"}
assert messages[1].parts[0].content == {"count": 3}
assert encode_model_messages(messages) == wire
class LookupArgs(TypedDict):
    key: str
class LookupResult(TypedDict):
    count: int
kind = sdk.ToolKind("codec_contract", "lookup", args=LookupArgs, content=LookupResult)
class LookupCall(sdk.ToolCallPart[LookupArgs], kind=kind):
    pass
class LookupReturn(sdk.ToolReturnPart[LookupResult], kind=kind):
    pass
promoted = decode_model_messages(wire)
assert isinstance(promoted[0].parts[0], LookupCall)
assert isinstance(promoted[1].parts[0], LookupReturn)
assert encode_model_messages(promoted) == wire
print("registered writer -> unregistered reader -> registered typed reader: preserved")
'''
    result = subprocess.run([sys.executable, "-c", script, str(path)], env=os.environ.copy(),
                            text=True, capture_output=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("native", (False, True))
@pytest.mark.parametrize("registration", ("none", "unregistered", "registered"))
def test_tool_kind_preserves_owned_multimodal_content(native: bool, registration: str) -> None:
    if registration == "registered" and not hasattr(sdk, "ToolKind"):
        pytest.skip("installed native SDK predates namespaced tool kinds")
    owner = sdk.NativeToolReturnPart if native else sdk.ToolReturnPart
    kind_id = None if registration == "none" else f"codec_content.{registration}"
    typed = None
    if registration == "registered":
        kind = sdk.ToolKind("codec_content", "registered", content=list[object])
        if native:
            class TypedReturn(sdk.NativeToolReturnPart[list[object]], kind=kind):
                pass
        else:
            class TypedReturn(sdk.ToolReturnPart[list[object]], kind=kind):
                pass
        typed = TypedReturn
        owner = typed
    reference = ReferencedContent(
        AssetVersionRef(AssetKey("document", "fixture"), "configured", StorageEntryRevision(1),
                        hashlib.sha256(b"fixture").hexdigest(), 7),
        media_type="application/pdf", source=CapabilityRef("capability", "files", 1),
        source_parameters={"version": "fixed"},
    )
    retired = RetiredReferencedContent(reference, "retired attachment")
    binary = sdk.BinaryContent(b"\x00\xff", media_type="image/png")
    part = owner("read", [reference, retired, binary, {1: "snapshot"}], tool_call_id="call-fixed",
                 tool_kind=kind_id, timestamp=_STAMP)
    message = (sdk.ModelResponse if native else sdk.ModelRequest)([part], timestamp=_STAMP)
    restored, = decode_model_messages(encode_model_messages((message,)))
    result = restored.parts[0]
    assert result.tool_kind == kind_id and result.tool_call_id == "call-fixed"
    if typed is not None:
        assert isinstance(result, typed)
    assert type(result.content[0]) is ReferencedContent
    assert result.content[0] == reference
    assert result.content[0].ref == reference.ref
    assert type(result.content[1]) is RetiredReferencedContent and result.content[1] == retired
    assert type(result.content[2]) is sdk.BinaryContent and result.content[2] == binary
    assert result.content[3] == {"1": "snapshot"}
    again, = decode_model_messages(encode_model_messages((restored,)))
    assert again.parts[0] == result


def test_core_tool_kind_still_rejects_unsubstantiated_content() -> None:
    assert sdk.ToolSearchReturnPart is not None
    part = sdk.ToolReturnPart("load", "invalid structured result", tool_call_id="call-fixed",
                              tool_kind="tool-search", timestamp=_STAMP)
    message = sdk.ModelRequest([replace(part, tool_kind=None)], timestamp=_STAMP)
    message.parts[0].tool_kind = "tool-search"
    with pytest.raises(ValueError, match="does not substantiate"):
        encode_model_messages((message,))
    ordinary = replace(part, tool_kind=None)
    wire = json.loads(encode_model_messages((sdk.ModelRequest([ordinary], timestamp=_STAMP),)))
    wire[0]["parts"][0]["tool_kind"] = "tool-search"
    with pytest.raises(AIError) as caught:
        decode_model_messages(canonical_json_bytes(wire))
    assert caught.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("location", ["prompt", "file", "speech"])
def test_native_binary_content_keeps_durable_media_and_kind_requirements(location: str) -> None:
    binary = sdk.BinaryContent(b"content", media_type="audio/wav" if location == "speech" else "image/png")
    if location == "prompt":
        message = sdk.ModelRequest([sdk.UserPromptPart([binary])])
    elif location == "file":
        message = sdk.ModelResponse([sdk.FilePart(binary)])
    else:
        message = sdk.ModelRequest([sdk.SpeechPart(speaker="user", audio=binary)])
    wire = json.loads(encode_model_messages([message]))
    part = wire[0]["parts"][0]
    content = part["content"][0] if location == "prompt" else part["content"] if location == "file" else part["audio"]
    del content["kind"]
    with pytest.raises(AIError) as missing:
        decode_model_messages(canonical_json_bytes(wire))
    assert missing.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    content["kind"] = "binary"
    content["media_type"] = ""
    with pytest.raises(AIError) as malformed:
        decode_model_messages(canonical_json_bytes(wire))
    assert malformed.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
    binary.media_type = ""
    with pytest.raises(ValueError, match="media type"):
        encode_model_messages([message])
