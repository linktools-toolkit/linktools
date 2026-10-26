#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Input attachment display metadata preserves admission and capture provenance."""

import json

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, TextContent, UserPromptPart

from linktools.ai.core import PromptLimits, WorkspaceFileInput
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model._attachment import request_attachment_facts
from linktools.ai.inputs import (
    ExecutionInputMaterializer,
    captured_input_files,
    captured_input_prompt,
    decode_task_prompt_draft,
    input_intent,
    merge_captured_input_files,
    task_prompt_draft,
)
from linktools.ai.model import MaterializedUserContent


class _Files:
    def __init__(self) -> None:
        self.values = {"inline.txt": b"INLINE", "suffix.txt": b"SUFFIX"}
        self.reads: list[str] = []

    async def canonicalize_path(self, path: str) -> str:
        return path.removeprefix("./")

    async def read_bytes(self, path: str, *, max_bytes: int | None = None) -> bytes:
        self.reads.append(path)
        value = self.values[path]
        if max_bytes is not None and len(value) > max_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return value

    async def close(self) -> None:
        return None


def _metadata(text: object, binary: BinaryContent, path: str | None = None) -> None:
    assert isinstance(text, TextContent)
    expected = {
        "attachment": binary.identifier,
        "size": len(binary.data),
        "mime_type": binary.media_type,
    }
    if path is not None:
        expected["path"] = path
    assert json.loads(text.content) == expected


@pytest.mark.asyncio
async def test_direct_binary_metadata_does_not_change_accepted_input_identity() -> None:
    binary = BinaryContent(
        b"DIRECT", media_type="text/plain", identifier="direct",
        vendor_metadata={"cache": "keep"},
    )
    author_text = TextContent("Author supplied text", metadata={"author": "user"})
    prompt = ("Inspect", author_text, binary)
    materializer = ExecutionInputMaterializer(None, PromptLimits(max_binary_input_parts=1))
    intent = input_intent(prompt, ())
    original = await materializer.store(prompt, tenant_id="tenant")

    accepted = await materializer.materialize(prompt, ())
    stored = await materializer.store(accepted, tenant_id="tenant")

    assert accepted[:2] == prompt[:2]
    _metadata(accepted[2], binary)
    assert accepted[3] is binary
    assert binary.vendor_metadata == {"cache": "keep"}
    assert input_intent(prompt, ()).digest == intent.digest
    assert stored.view == original.view
    assert await materializer.restore(stored) == accepted
    assert await materializer.materialize(accepted, ()) is accepted
    assert await materializer.materialize(tuple(accepted), ()) == accepted


@pytest.mark.asyncio
async def test_arbitrary_path_labels_do_not_become_attachment_provenance() -> None:
    binary = BinaryContent(b"DIRECT", media_type="text/plain")
    label = 'Workspace file path: "private.txt"'
    author_text = TextContent('{"path":"private.txt"}')
    materializer = ExecutionInputMaterializer(None, PromptLimits())

    accepted = await materializer.materialize((label, author_text, binary), ())

    assert accepted[:2] == (label, author_text)
    _metadata(accepted[2], binary)
    assert accepted[3] is binary


@pytest.mark.asyncio
@pytest.mark.parametrize("prompt", ("Inspect the files", ("",)))
async def test_independent_files_precede_prompt_in_declared_order(prompt: str | tuple[str, ...]) -> None:
    files = _Files()
    materializer = ExecutionInputMaterializer(files, PromptLimits())

    accepted = await materializer.materialize(prompt, ("suffix.txt", "inline.txt"))

    _metadata(accepted[0], accepted[1], "suffix.txt")
    _metadata(accepted[2], accepted[3], "inline.txt")
    assert accepted[1].data == b"SUFFIX"
    assert accepted[3].data == b"INLINE"
    assert accepted[4:] == ((prompt,) if isinstance(prompt, str) else prompt)
    assert captured_input_files(accepted, 2) == accepted[:4]
    assert merge_captured_input_files(prompt, captured_input_files(accepted, 2)) == accepted
    stored = await materializer.store(accepted, tenant_id="tenant")
    restored = await materializer.restore(stored)
    assert restored == accepted
    assert await materializer.materialize(restored, ()) is restored
    assert files.reads == ["suffix.txt", "inline.txt"]


@pytest.mark.asyncio
async def test_mixed_capture_tracks_workspace_files_after_expanded_binary_parts() -> None:
    files = _Files()
    materializer = ExecutionInputMaterializer(files, PromptLimits())
    binary = BinaryContent(b"DIRECT", media_type="text/plain", identifier="direct")
    original = (
        "Before", binary, WorkspaceFileInput("./inline.txt", identifier="inline"),
        "Between", binary, WorkspaceFileInput("inline.txt", identifier="second"), "After",
    )
    canonical = await materializer.canonicalize_input(original)
    accepted = await materializer.materialize(canonical, ("suffix.txt",))
    captured = captured_input_prompt(original, accepted)
    prefix = captured_input_files(accepted, 1)

    assert captured[:2] == original[:2]
    assert captured[3].data == b"INLINE"
    _metadata(captured[2], captured[3], "inline.txt")
    assert captured[4:6] == ("Between", binary)
    assert captured[7].identifier == "second"
    _metadata(captured[6], captured[7], "inline.txt")
    assert captured[8] == "After"
    _metadata(prefix[0], prefix[1], "suffix.txt")
    assert prefix[1].data == b"SUFFIX"
    assert prefix == accepted[:2]
    assert accepted[2] == "Before"
    assert accepted[-1] == "After"
    facts = request_attachment_facts(
        (ModelRequest(parts=[UserPromptPart(content=accepted)]),),
        accepted.view["attachments"],
        accepted_attachment_ids=set(),
    )
    occurrences = {item["position"]: item["attachment_id"] for item in accepted.view["attachments"]}
    assert {fact["attachment_id"]: fact["position"] for fact in facts} == {
        occurrences[0]: 1,
        occurrences[1]: 2,
        occurrences[2]: 3,
        occurrences[3]: 4,
        occurrences[4]: 0,
    }

    files.values.clear()
    replay = decode_task_prompt_draft(task_prompt_draft(merge_captured_input_files(captured, prefix)))
    replayed = await ExecutionInputMaterializer(None, PromptLimits()).materialize(replay, ())
    assert tuple(replayed) == tuple(accepted)
    assert files.reads == ["inline.txt", "inline.txt", "suffix.txt"]


@pytest.mark.asyncio
async def test_equal_bytes_keep_distinct_occurrences_after_files_move_before_prompt() -> None:
    files = _Files()
    binary = BinaryContent(files.values["inline.txt"], media_type="text/plain")
    prompt = (binary, WorkspaceFileInput("inline.txt"), binary)
    materializer = ExecutionInputMaterializer(files, PromptLimits())
    accepted = await materializer.materialize(prompt, ("inline.txt", "inline.txt"))
    occurrences = accepted.view["attachments"]

    assert [item["position"] for item in occurrences] == [3, 4, 0, 1, 2]
    assert len({item["attachment_id"] for item in occurrences}) == 5
    facts = request_attachment_facts(
        (ModelRequest(parts=[UserPromptPart(content=accepted)]),), occurrences,
        accepted_attachment_ids=set(),
    )
    assert [(item["attachment_id"], item["position"]) for item in facts] == [
        (occurrence["attachment_id"], index) for index, occurrence in enumerate(occurrences)
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("field", ("path", "attachment", "size", "mime_type"))
async def test_capture_rejects_metadata_that_disagrees_with_accepted_file(field: str) -> None:
    materializer = ExecutionInputMaterializer(_Files(), PromptLimits())
    accepted = await materializer.materialize("Inspect", ("suffix.txt",))
    text = accepted[0]
    assert isinstance(text, TextContent)
    altered = json.loads(text.content)
    altered[field] = 999 if field == "size" else "different"
    corrupt = MaterializedUserContent((
        TextContent(json.dumps(altered), metadata=text.metadata), accepted[1], accepted[2],
    ), accepted.view)

    with pytest.raises(AIError) as raised:
        captured_input_files(corrupt, 1)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR


@pytest.mark.asyncio
@pytest.mark.parametrize("malformation", ("untyped_label", "misplaced_files"))
async def test_capture_requires_typed_metadata_at_the_file_prefix(malformation: str) -> None:
    materializer = ExecutionInputMaterializer(_Files(), PromptLimits())
    accepted = await materializer.materialize("Inspect", ("suffix.txt",))
    if malformation == "untyped_label":
        parts = (accepted[0].content, accepted[1], accepted[2])
    else:
        parts = (accepted[2], accepted[0], accepted[1])
    corrupt = MaterializedUserContent(parts, accepted.view)

    with pytest.raises(AIError) as raised:
        captured_input_files(corrupt, 1)

    assert raised.value.code is ErrorCode.STORAGE_INTEGRITY_ERROR
