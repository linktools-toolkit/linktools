#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import hashlib

import pytest

from linktools.ai.core import Principal
from linktools.ai.runtime.state import (
    AttachmentEntry,
    AttachmentPresentation,
    AttachmentSemanticEntry,
    AttachmentUploadRecord,
    ContentRef,
    InputAttachmentPart,
    InputPrepareRecord,
    InputSource,
    InputTextPart,
    InputV2,
    Locator,
    ModelExposureEntry,
    PathOrigin,
    PreparedInput,
    RuntimeDomain,
    input_v2_digest,
    iter_runtime_object_refs,
    managed_attachment_locator,
    managed_attachment_path,
    model_exposure_activation_digest,
    wire_type_id,
)
from linktools.ai.runtime.state._codec import decode_domain, encode_domain
from linktools.ai.storage import ObjectRef


def _entry(*, path: str, data: bytes = b"payload") -> AttachmentEntry:
    digest = hashlib.sha256(data).hexdigest()
    return AttachmentEntry(
        path,
        "sample.png",
        "image/png",
        AttachmentPresentation(None, None),
        ContentRef(
            "execution",
            None,
            ObjectRef("objects", f"objects/{digest}", digest, len(data)),
        ),
    )


def test_managed_attachment_path_is_strict_and_round_trips() -> None:
    owner = "a" * 64
    path = managed_attachment_path("p", owner, 7)
    assert path == f"virtual:attachments/p.{owner}.7"
    assert managed_attachment_locator(path) == ("p", owner, 7)
    with pytest.raises(ValueError):
        managed_attachment_locator(f"virtual:attachments/u.{owner}.1")
    with pytest.raises(ValueError):
        managed_attachment_locator(f"virtual:attachments/p.{owner}.01")


def test_prepared_input_digest_ignores_physical_object_location() -> None:
    owner = "b" * 64
    path = managed_attachment_path("p", owner, 0)
    first = _entry(path=path)
    moved = AttachmentEntry(
        first.path,
        first.name,
        first.media_type,
        first.presentation,
        ContentRef(
            "recovery",
            "archive",
            ObjectRef(
                "archive-store",
                "different-key",
                first.content.object.digest,
                first.content.object.size,
            ),
        ),
    )
    prompt = InputV2(
        2,
        (InputTextPart("text", "inspect"), InputAttachmentPart("attachment", 0)),
        (),
        (InputSource("evidence/sample.png", 0),),
    )
    assert input_v2_digest(prompt, (first,)) == input_v2_digest(prompt, (moved,))


def test_prepared_input_requires_every_manifest_entry_to_be_granted() -> None:
    owner = "c" * 64
    first = _entry(path=managed_attachment_path("p", owner, 0))
    second = _entry(path=managed_attachment_path("p", owner, 1), data=b"second")
    prompt = InputV2(2, (InputAttachmentPart("attachment", 0),), (), ())
    digest = input_v2_digest(prompt, (first, second))
    with pytest.raises(ValueError):
        PreparedInput(
            1,
            "linktools-input-v2",
            prompt,
            (first, second),
            "d" * 64,
            digest,
            PathOrigin(1, "workspace", "posix", "/workspace"),
        )


def test_released_upload_does_not_retain_content() -> None:
    owner = "e" * 64
    entry = _entry(path=managed_attachment_path("u", owner, 0))
    semantic = AttachmentSemanticEntry(
        entry.path,
        entry.name,
        entry.media_type,
        entry.presentation,
        entry.content.object.digest,
        entry.content.object.size,
    )
    principal = Principal("user", "tenant", "local_trusted")
    with pytest.raises(ValueError):
        AttachmentUploadRecord(1, principal, "f" * 64, semantic, entry.content, "RELEASED")


def test_model_exposure_digest_is_semantic_not_physical() -> None:
    owner = "1" * 64
    entry = _entry(path=managed_attachment_path("p", owner, 0))
    source = Locator("state:execution", "records", "2" * 64)
    activation = "3" * 64
    first = ModelExposureEntry(activation, source, 0, entry)
    moved_entry = AttachmentEntry(
        entry.path,
        entry.name,
        entry.media_type,
        entry.presentation,
        ContentRef(
            "recovery",
            None,
            ObjectRef(
                "other-store",
                "other-key",
                entry.content.object.digest,
                entry.content.object.size,
            ),
        ),
    )
    second = ModelExposureEntry(activation, source, 0, moved_entry)
    assert model_exposure_activation_digest((first,)) == model_exposure_activation_digest((second,))


def test_attachment_upload_wire_round_trips_and_visits_held_body() -> None:
    owner = "4" * 64
    entry = _entry(path=managed_attachment_path("u", owner, 0))
    semantic = AttachmentSemanticEntry(
        entry.path,
        entry.name,
        entry.media_type,
        entry.presentation,
        entry.content.object.digest,
        entry.content.object.size,
    )
    record = AttachmentUploadRecord(
        1,
        Principal("user", "tenant", "local_trusted"),
        "5" * 64,
        semantic,
        entry.content,
        "HELD",
    )
    wire = encode_domain(record)
    assert wire_type_id(record) == "attachment_upload_v1"
    assert decode_domain(wire, AttachmentUploadRecord) == record
    refs = tuple(iter_runtime_object_refs(wire, default_domain=RuntimeDomain.EXECUTION))
    assert refs == ((RuntimeDomain.EXECUTION, entry.content.object),)


def test_input_prepare_wire_visits_ready_manifest_only_once() -> None:
    owner = "6" * 64
    entry = _entry(path=managed_attachment_path("p", owner, 0))
    prompt = InputV2(2, (InputAttachmentPart("attachment", 0),), (), ())
    prepared = PreparedInput(
        1,
        "linktools-input-v2",
        prompt,
        (entry,),
        "7" * 64,
        input_v2_digest(prompt, (entry,)),
        PathOrigin(1, "workspace", "posix", "/workspace"),
    )
    record = InputPrepareRecord(
        1,
        prepared.intent_digest,
        prepared.path_origin,
        "READY",
        (),
        prepared,
        None,
        None,
    )
    wire = encode_domain(record)
    assert wire_type_id(record) == "input_prepare_v1"
    assert decode_domain(wire, InputPrepareRecord) == record
    refs = tuple(iter_runtime_object_refs(wire, default_domain=RuntimeDomain.EXECUTION))
    assert refs == ((RuntimeDomain.EXECUTION, entry.content.object),)
