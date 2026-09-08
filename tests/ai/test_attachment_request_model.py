#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json

import pytest
from pydantic_ai.messages import BinaryContent, ModelRequest, TextContent, UserPromptPart
from pydantic_ai.models.test import TestModel

from linktools.ai.core import canonical_sha256
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.runtime._attachment_adapter import (
    AttachmentRequestModel,
    attachment_placeholder,
)
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentPresentation,
    ContentRef,
    Locator,
    ModelExposure,
    ModelExposureEntry,
    PathOrigin,
    model_exposure_activation_digest,
)
from linktools.ai.storage import ObjectRef

_BODY_DIGEST = "230d8358dc8e8890b4c58deeb62912ee2f20357ae92a5cc861b98e68fe31acb5"


def _entry(path: str, *, name: str) -> AttachmentEntry:
    return AttachmentEntry(
        path,
        name,
        "image/png",
        AttachmentPresentation("image", {"detail": "high"}),
        ContentRef(
            "execution",
            "input-prepare:" + "c" * 64,
            ObjectRef("memory", "body", _BODY_DIGEST, 4),
        ),
    )


def _activation(activation_id: str, path: str, *, name: str, source: str) -> ModelExposureEntry:
    return ModelExposureEntry(
        activation_id,
        Locator("state:execution", "records", source),
        0,
        _entry(path, name=name),
    )


def _model(
    activations: tuple[ModelExposureEntry, ...],
    *,
    reads: list[str],
) -> AttachmentRequestModel:
    origin = PathOrigin(1, "attachments", "posix", "/workspace")

    async def check_execution(run_step: int) -> None:
        assert run_step == 0

    async def authorize_entries(
        run_step: int,
        entries: tuple[ModelExposureEntry, ...],
    ) -> None:
        assert run_step == 0
        assert all(entry in activations for entry in entries)

    async def commit_exposure(
        run_step: int,
        entries: tuple[ModelExposureEntry, ...],
    ) -> ModelExposure:
        return ModelExposure(
            1,
            "f" * 64,
            "execution",
            "run",
            run_step,
            origin,
            entries,
            model_exposure_activation_digest(entries),
        )

    async def read_content(content: ContentRef) -> bytes:
        reads.append(content.object.key)
        return b"body"

    return AttachmentRequestModel(
        TestModel(custom_output_text="ok"),
        execution_id="execution",
        step_run_id="run",
        run_step=0,
        path_origin=origin,
        activations=activations,
        check_execution=check_execution,
        authorize_entries=authorize_entries,
        commit_exposure=commit_exposure,
        read_content=read_content,
    )


def test_attachment_placeholder_uses_exact_trusted_metadata_shape() -> None:
    activation = _activation(
        "1" * 64,
        "virtual:attachments/p." + "c" * 64 + ".0",
        name="screen.png",
        source="b" * 64,
    )

    placeholder = attachment_placeholder(activation)

    assert isinstance(placeholder, TextContent)
    assert json.loads(placeholder.content) == {
        "path": activation.entry.path,
        "name": "screen.png",
        "type": "image/png",
        "size": 4,
    }
    assert placeholder.metadata is not None
    assert set(placeholder.metadata) == {"linktools.attachment-slot.v1"}
    marker = placeholder.metadata["linktools.attachment-slot.v1"]
    assert isinstance(marker, dict)
    assert marker["activation_id"] == activation.activation_id
    assert marker["source"] == activation.source.to_json()
    assert marker["slot"] == 0
    assert isinstance(marker["entry_digest"], str)
    assert len(marker["entry_digest"]) == 64


@pytest.mark.asyncio
async def test_first_generation_requires_every_active_slot() -> None:
    first = _activation(
        "1" * 64,
        "virtual:attachments/p." + "c" * 64 + ".0",
        name="a.png",
        source="b" * 64,
    )
    second = _activation(
        "2" * 64,
        "virtual:attachments/p." + "d" * 64 + ".0",
        name="b.png",
        source="e" * 64,
    )
    model = _model((first, second), reads=[])
    messages = [
        ModelRequest(parts=[UserPromptPart(content=(attachment_placeholder(first),))])
    ]

    with pytest.raises(AIError) as raised:
        await model._prepare_call(messages, generation=True)

    assert raised.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT


@pytest.mark.asyncio
async def test_same_body_and_presentation_is_read_and_expanded_once() -> None:
    first = _activation(
        "1" * 64,
        "virtual:attachments/p." + "c" * 64 + ".0",
        name="first.png",
        source="b" * 64,
    )
    second = _activation(
        "2" * 64,
        "virtual:attachments/p." + "d" * 64 + ".0",
        name="second.png",
        source="e" * 64,
    )
    reads: list[str] = []
    model = _model((first, second), reads=reads)
    messages = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=(
                        attachment_placeholder(first),
                        attachment_placeholder(second),
                    )
                )
            ]
        )
    ]

    projected, _ticket = await model._prepare_call(messages, generation=True)

    assert reads == ["body"]
    request = projected[0]
    assert isinstance(request, ModelRequest)
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert not isinstance(part.content, str)
    assert isinstance(part.content[0], BinaryContent)
    assert isinstance(part.content[1], TextContent)
    assert request.metadata == {"linktools.ai.exposure_id": "f" * 64}


@pytest.mark.asyncio
async def test_tampered_slot_metadata_is_rejected_before_content_read() -> None:
    activation = _activation(
        "1" * 64,
        "virtual:attachments/p." + "c" * 64 + ".0",
        name="screen.png",
        source="b" * 64,
    )
    placeholder = attachment_placeholder(activation)
    assert placeholder.metadata is not None
    marker = dict(placeholder.metadata["linktools.attachment-slot.v1"])
    marker["entry_digest"] = canonical_sha256("tampered")
    tampered = TextContent(
        placeholder.content,
        metadata={"linktools.attachment-slot.v1": marker},
    )
    reads: list[str] = []
    model = _model((activation,), reads=reads)

    with pytest.raises(AIError) as raised:
        await model._prepare_call(
            [ModelRequest(parts=[UserPromptPart(content=(tampered,))])],
            generation=True,
        )

    assert raised.value.code is ErrorCode.CAPABILITY_POLICY_CONFLICT
    assert reads == []
