#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""File result projection retains validated attachment occurrences, not bodies."""

import hashlib

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, ModelResponse, TextPart, ToolReturn, ToolReturnPart, UserPromptPart

from linktools.ai.asset import AssetKey, AssetVersionRef
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.runtime._attachment import bind_tool_return_attachments, request_attachment_facts
from linktools.ai.runtime._message import decode_model_messages, encode_model_messages, project_transient_binary_content
from linktools.ai.runtime._user_content_codec import decode_user_content_item, encode_user_content_item
from linktools.ai.spec import CapabilityRef
from linktools.ai.storage import StorageEntryRevision

_BODY = b"recoverable file body"
_SOURCE = CapabilityRef("capability", "file-source", 1)


def _reference(*, representation: str = "binary", identifier: str | None = None) -> ReferencedContent:
    return ReferencedContent(
        AssetVersionRef(AssetKey("file", "document"), "source", StorageEntryRevision(1),
                        hashlib.sha256(_BODY).hexdigest(), len(_BODY)),
        media_type="text/plain", source=_SOURCE, representation=representation,
        identifier=identifier, source_parameters={"part": "document"},
    )


def _file(index: int = 0) -> dict[str, object]:
    return {"path": f"document-{index}.txt", "input_index": index, "mime_type": "text/plain",
            "size": len(_BODY), "sha256": hashlib.sha256(_BODY).hexdigest()}


def _original(tool: str = "read_file", count: int = 1) -> ToolReturn:
    return bind_tool_return_attachments(tool, "read", ToolReturn(
        return_value={"files": [_file(i) for i in range(count)], "texts": []},
        content=[BinaryContent(_BODY, media_type="text/plain") for _ in range(count)],
    ))


@pytest.mark.parametrize("tool", ("read_file", "attach_files"))
@pytest.mark.parametrize("representation", ("binary", "text"))
def test_referenced_files_keep_distinct_occurrences_and_request_facts(tool: str, representation: str) -> None:
    reference = _reference(representation=representation)
    files = [_file(0), _file(1)]
    original = _original(tool, 2)
    result = bind_tool_return_attachments(tool, "read", ToolReturn(
        return_value=original.return_value,
        content=[_reference(representation=representation, identifier=item.identifier) for item in original.content],
    ))
    first, second = result.content
    assert first.identifier and second.identifier and first.identifier != second.identifier
    assert reference.identifier is None
    assert first.ref == second.ref == reference.ref
    assert first.source_parameters == second.source_parameters == reference.source_parameters
    for content in result.content:
        assert decode_user_content_item(encode_user_content_item(content)) == content
    if tool == "read_file":
        assert result.return_value["files"] == [
            {**file, "attachment": content.identifier} for file, content in zip(files, result.content)
        ]
    accepted = set()
    request = ModelRequest(parts=[
        ToolReturnPart(tool, result.return_value, "read"), UserPromptPart(list(reversed(result.content))),
    ])
    facts = request_attachment_facts([request], (), accepted_attachment_ids=accepted)
    assert [(item["fact"], item["attachment_id"]) for item in facts] == [
        ("accepted", first.identifier), ("accepted", second.identifier),
        ("included_in_request", second.identifier), ("included_in_request", first.identifier),
    ]
    assert all(item["source"] == tool and item["call_id"] == "read" for item in facts)
    again = request_attachment_facts([request], (), accepted_attachment_ids=accepted)
    assert [item["fact"] for item in again] == ["included_in_request", "included_in_request"]
    restored = decode_model_messages(encode_model_messages([request]))
    assert request_attachment_facts(restored, (), accepted_attachment_ids=set()) == facts
    consumed = project_transient_binary_content([*restored, ModelResponse(parts=[TextPart("done")])])
    summaries = consumed[0].parts[1].content
    assert [encode_user_content_item(item)["metadata"]["referenced_content"]["identifier"] for item in summaries] == [second.identifier, first.identifier]
    assert not request_attachment_facts(consumed, (), accepted_attachment_ids=accepted)


@pytest.mark.parametrize("field,value", (("mime_type", "image/png"), ("size", 0), ("sha256", "0" * 64)))
def test_referenced_files_do_not_relax_original_byte_fact_validation(field: str, value: object) -> None:
    file = {**_file(), field: value}
    original = _original()
    with pytest.raises(AIError) as failure:
        bind_tool_return_attachments("read_file", "read", ToolReturn(
            return_value={"files": [file]}, content=[_reference(identifier=original.content[0].identifier)],
        ))
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


def test_plain_text_reference_can_coexist_with_a_file_occurrence() -> None:
    # The ordinary text is deliberately byte-identical to the file. It must not
    # consume the file's occurrence merely because its descriptor matches.
    text = _reference(representation="text")
    original = _original()
    result = bind_tool_return_attachments("read_file", "read", ToolReturn(
        return_value={**original.return_value, "texts": [{"path": "text.txt", "summary": "text"}]},
        content=[text, _reference(identifier=original.content[0].identifier)],
    ))
    assert result.content[0] is text and text.identifier is None
    assert result.content[1].identifier == result.return_value["files"][0]["attachment"]
    plain = bind_tool_return_attachments("read_file", "text", ToolReturn(
        return_value={"files": [], "texts": [{"summary": "text"}]}, content=[text],
    ))
    assert plain.content == [text]


def test_mixed_binary_and_referenced_files_preserve_order_and_reject_extras() -> None:
    original = _original(count=2)
    result = bind_tool_return_attachments("read_file", "read", ToolReturn(
        return_value=original.return_value,
        content=[original.content[0], _reference(identifier=original.content[1].identifier)],
    ))
    assert [item.identifier for item in result.content] == [file["attachment"] for file in result.return_value["files"]]
    with pytest.raises(AIError):
        bind_tool_return_attachments("read_file", "read", ToolReturn(
            return_value={"files": [_file()]},
            content=[_reference(identifier=original.content[0].identifier)] * 2,
        ))


@pytest.mark.parametrize("identifiers", ((None, None), ("wrong", "wrong"), ("reverse", "reverse")))
def test_missing_or_forged_reference_occurrences_never_match_by_content(
    identifiers: tuple[str | None, str | None],
) -> None:
    original = _original(count=2)
    if identifiers == ("reverse", "reverse"):
        identifiers = tuple(item.identifier for item in reversed(original.content))
    with pytest.raises(AIError) as failure:
        bind_tool_return_attachments("read_file", "read", ToolReturn(
            return_value=original.return_value,
            content=[_reference(identifier=value) for value in identifiers],
        ))
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.parametrize("value", ({"summary": "file"}, {"files": []}))
def test_file_occurrence_references_require_the_original_file_receipt(value: object) -> None:
    original = _original()
    with pytest.raises(AIError) as failure:
        bind_tool_return_attachments("read_file", "read", ToolReturn(
            return_value=value, content=[_reference(identifier=original.content[0].identifier)],
        ))
    assert failure.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
