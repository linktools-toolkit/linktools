#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pydantic_ai.messages import ModelRequest, TextContent, UserPromptPart

from linktools.ai.runtime._attachment_projection import (
    _prepare_current_messages,
    initial_attachment_prompt,
)
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentPresentation,
    ContentRef,
    InputAttachmentPart,
    InputTextPart,
    InputV2,
    ModelExposureEntry,
)
from linktools.ai.storage import ObjectRef


def _entry(owner: str, slot: int, digest: str) -> AttachmentEntry:
    return AttachmentEntry(
        f"virtual:attachments/p.{owner}.{slot}",
        f"evidence-{slot}.png",
        "image/png",
        AttachmentPresentation(None, None),
        ContentRef(
            "execution",
            f"input-prepare:{owner}",
            ObjectRef("memory", f"body-{slot}", digest, 4),
        ),
    )


def test_initial_projection_activates_only_direct_parts() -> None:
    owner = "c" * 64
    manifest = (
        _entry(owner, 0, "a" * 64),
        _entry(owner, 1, "b" * 64),
    )
    prompt = InputV2(
        2,
        (
            InputTextPart("text", "inspect"),
            InputAttachmentPart("attachment", 0),
        ),
        (1,),
        (),
    )

    projected, active = initial_attachment_prompt(
        prompt,
        manifest,
        execution_id="execution",
        execution_record_key="d" * 64,
    )

    assert isinstance(projected, tuple)
    assert projected[0] == "inspect"
    assert isinstance(projected[1], TextContent)
    assert len(active) == 1
    assert active[0].slot == 0
    assert active[0].entry == manifest[0]
    assert manifest[1].path not in projected[1].content


def test_current_message_projection_keeps_old_slot_as_plain_description_and_restores_active() -> None:
    owner = "c" * 64
    prompt = InputV2(
        2,
        (InputAttachmentPart("attachment", 0),),
        (),
        (),
    )
    lightweight, active = initial_attachment_prompt(
        prompt,
        (_entry(owner, 0, "a" * 64),),
        execution_id="execution",
        execution_record_key="d" * 64,
    )
    assert isinstance(lightweight, tuple)
    current = active[0]
    assert isinstance(current, ModelExposureEntry)
    current_placeholder = lightweight[0]
    assert isinstance(current_placeholder, TextContent)

    old = TextContent(
        "old attachment",
        metadata={
            "linktools.attachment-slot.v1": {
                "activation_id": "f" * 64,
                "source": {"resource": "state:execution", "space": "records", "key": "e" * 64},
                "slot": 0,
                "entry_digest": "1" * 64,
            }
        },
    )
    messages = [ModelRequest(parts=[UserPromptPart(content=(old,))])]

    normalized = _prepare_current_messages(messages, (current,))

    request = normalized[0]
    assert isinstance(request, ModelRequest)
    original_part = request.parts[0]
    assert isinstance(original_part, UserPromptPart)
    assert not isinstance(original_part.content, str)
    normalized_old = original_part.content[0]
    assert isinstance(normalized_old, TextContent)
    assert normalized_old.content == "old attachment"
    assert normalized_old.metadata is None
    restored_part = request.parts[-1]
    assert isinstance(restored_part, UserPromptPart)
    assert not isinstance(restored_part.content, str)
    restored = restored_part.content[0]
    assert isinstance(restored, TextContent)
    assert restored.metadata == current_placeholder.metadata
