#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Portable attachment persistence values."""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

from ...core import (
    ImmutableJsonMapping,
    JsonValue,
    Principal,
    canonical_sha256,
    normalize_json_value,
)
from ...storage import ObjectRef
from ._plan import RuntimeDomain
from ._relocation import Locator, LogicalPath, PathOrigin

_SHA256 = re.compile(r"[0-9a-f]{64}")
_ATTACHMENT_PATH = re.compile(
    r"virtual:attachments/(?P<kind>[upe])\.(?P<owner>[0-9a-f]{64})\.(?P<slot>0|[1-9][0-9]*)"
)

AttachmentUploadStatus: TypeAlias = Literal["HELD", "RELEASED"]
InputPrepareStatus: TypeAlias = Literal["PREPARING", "READY", "ADOPTED", "ABORTED"]


def _require_string(value: object, field: str, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or (nonempty and not value):
        raise ValueError(f"{field} must be a{' non-empty' if nonempty else ''} string")
    return value


def _require_int(value: object, field: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{field} must be an integer >= {minimum}")
    return value


def _require_sha256(value: object, field: str) -> str:
    text = _require_string(value, field)
    if _SHA256.fullmatch(text) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return text


def _immutable_object(
    value: Mapping[str, JsonValue],
    field: str,
) -> Mapping[str, JsonValue]:
    normalized = normalize_json_value(value)
    if not isinstance(normalized, dict):
        raise ValueError(f"{field} must be an object")
    return ImmutableJsonMapping(normalized)


def _validate_managed_path(path: str) -> tuple[str, str, int]:
    match = _ATTACHMENT_PATH.fullmatch(path)
    if match is None:
        raise ValueError("attachment path must use the managed virtual namespace")
    kind = match.group("kind")
    slot = int(match.group("slot"))
    if kind in {"u", "e"} and slot != 0:
        raise ValueError("upload/source attachment paths require slot 0")
    return kind, match.group("owner"), slot


def managed_attachment_path(kind: Literal["u", "p", "e"], owner_key: str, slot: int) -> str:
    """Build one canonical managed attachment path from an owner locator."""
    if kind not in {"u", "p", "e"}:
        raise ValueError("managed attachment kind is invalid")
    if _SHA256.fullmatch(owner_key) is None:
        raise ValueError("managed attachment owner key must be lowercase 64-hex")
    _require_int(slot, "attachment slot")
    if kind in {"u", "e"} and slot != 0:
        raise ValueError("upload/source attachment paths require slot 0")
    return f"virtual:attachments/{kind}.{owner_key}.{slot}"


def managed_attachment_locator(path: str) -> tuple[str, str, int]:
    """Parse one canonical managed attachment path without granting authorization."""
    _require_string(path, "attachment path")
    return _validate_managed_path(path)


@dataclass(frozen=True, slots=True)
class AttachmentPresentation:
    identifier: str | None
    vendor_metadata: Mapping[str, JsonValue] | None

    def __post_init__(self) -> None:
        if self.identifier is not None:
            _require_string(self.identifier, "attachment presentation identifier")
        if self.vendor_metadata is not None:
            object.__setattr__(
                self,
                "vendor_metadata",
                _immutable_object(
                    self.vendor_metadata,
                    "attachment presentation vendor_metadata",
                ),
            )


@dataclass(frozen=True, slots=True)
class ContentRef:
    domain: str
    owner_scope: str | None
    object: ObjectRef

    def __post_init__(self) -> None:
        if self.domain not in {item.value for item in RuntimeDomain}:
            raise ValueError("attachment content domain is invalid")
        if self.owner_scope is not None:
            _require_string(self.owner_scope, "attachment content owner_scope")
        if not isinstance(self.object, ObjectRef):
            raise TypeError("attachment content object must be ObjectRef")
        if self.object.size <= 0:
            raise ValueError("attachment content size must be positive")


@dataclass(frozen=True, slots=True)
class AttachmentEntry:
    path: str
    name: str | None
    media_type: str
    presentation: AttachmentPresentation
    content: ContentRef

    def __post_init__(self) -> None:
        _require_string(self.path, "attachment path")
        _validate_managed_path(self.path)
        if self.name is not None:
            _require_string(self.name, "attachment name")
        _require_string(self.media_type, "attachment media_type")
        if not isinstance(self.presentation, AttachmentPresentation):
            raise TypeError("attachment presentation must be AttachmentPresentation")
        if not isinstance(self.content, ContentRef):
            raise TypeError("attachment content must be ContentRef")


@dataclass(frozen=True, slots=True)
class AttachmentSemanticEntry:
    path: str
    name: str | None
    media_type: str
    presentation: AttachmentPresentation
    digest: str
    size: int

    def __post_init__(self) -> None:
        _require_string(self.path, "attachment semantic path")
        _validate_managed_path(self.path)
        if self.name is not None:
            _require_string(self.name, "attachment semantic name")
        _require_string(self.media_type, "attachment semantic media_type")
        if not isinstance(self.presentation, AttachmentPresentation):
            raise TypeError("attachment semantic presentation must be AttachmentPresentation")
        _require_sha256(self.digest, "attachment semantic digest")
        _require_int(self.size, "attachment semantic size", minimum=1)


@dataclass(frozen=True, slots=True)
class InputTextPart:
    kind: Literal["text"]
    text: str

    def __post_init__(self) -> None:
        if self.kind != "text":
            raise ValueError("text input part kind must be text")
        if not isinstance(self.text, str):
            raise TypeError("text input part text must be a string")


@dataclass(frozen=True, slots=True)
class InputNativePart:
    kind: Literal["native"]
    codec: Literal["pydantic-user-content-v1"]
    value: Mapping[str, JsonValue]

    def __post_init__(self) -> None:
        if self.kind != "native" or self.codec != "pydantic-user-content-v1":
            raise ValueError("native input part codec is invalid")
        object.__setattr__(
            self,
            "value",
            _immutable_object(self.value, "native input value"),
        )


@dataclass(frozen=True, slots=True)
class InputAttachmentPart:
    kind: Literal["attachment"]
    index: int

    def __post_init__(self) -> None:
        if self.kind != "attachment":
            raise ValueError("attachment input part kind must be attachment")
        _require_int(self.index, "attachment input index")


InputPart: TypeAlias = InputTextPart | InputNativePart | InputAttachmentPart


@dataclass(frozen=True, slots=True)
class InputSource:
    relative: str
    index: int

    def __post_init__(self) -> None:
        _require_string(self.relative, "input source relative")
        if self.relative.startswith("/") or "\\" in self.relative or any(
            part in {"", ".", ".."} for part in self.relative.split("/")
        ):
            raise ValueError("input source relative must be a normalized relative path")
        _require_int(self.index, "input source index")


@dataclass(frozen=True, slots=True)
class InputV2:
    version: int
    parts: tuple[InputPart, ...]
    available: tuple[int, ...]
    sources: tuple[InputSource, ...]

    def __post_init__(self) -> None:
        if self.version != 2 or isinstance(self.version, bool):
            raise ValueError("InputV2 version must be 2")
        parts = tuple(self.parts)
        if any(
            not isinstance(item, (InputTextPart, InputNativePart, InputAttachmentPart))
            for item in parts
        ):
            raise TypeError("InputV2 parts contain an unsupported value")
        available = tuple(_require_int(item, "available attachment index") for item in self.available)
        if len(available) != len(set(available)):
            raise ValueError("InputV2 available indices must be first-order unique")
        sources = tuple(self.sources)
        if any(not isinstance(item, InputSource) for item in sources):
            raise TypeError("InputV2 sources must contain InputSource")
        if tuple(sorted(sources, key=lambda item: item.relative)) != sources:
            raise ValueError("InputV2 sources must be sorted by relative path")
        if len({item.relative for item in sources}) != len(sources):
            raise ValueError("InputV2 sources contain duplicate relative paths")
        object.__setattr__(self, "parts", parts)
        object.__setattr__(self, "available", available)
        object.__setattr__(self, "sources", sources)


@dataclass(frozen=True, slots=True)
class PreparedInput:
    version: int
    user_prompt_codec: Literal["linktools-input-v2"]
    user_prompt: InputV2
    attachment_manifest: tuple[AttachmentEntry, ...]
    intent_digest: str
    input_digest: str
    path_origin: PathOrigin

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("PreparedInput version must be 1")
        if self.user_prompt_codec != "linktools-input-v2":
            raise ValueError("PreparedInput user_prompt_codec is invalid")
        if not isinstance(self.user_prompt, InputV2):
            raise TypeError("PreparedInput user_prompt must be InputV2")
        manifest = tuple(self.attachment_manifest)
        if any(not isinstance(item, AttachmentEntry) for item in manifest):
            raise TypeError("PreparedInput attachment_manifest contains invalid values")
        paths = [item.path for item in manifest]
        if len(paths) != len(set(paths)):
            raise ValueError("PreparedInput attachment_manifest contains duplicate paths")
        referenced = {
            item.index
            for item in self.user_prompt.parts
            if isinstance(item, InputAttachmentPart)
        }
        referenced.update(self.user_prompt.available)
        indexes = referenced.union(item.index for item in self.user_prompt.sources)
        if any(index >= len(manifest) for index in indexes):
            raise ValueError("PreparedInput references an invalid attachment index")
        if referenced != set(range(len(manifest))):
            raise ValueError("every manifest entry must be referenced by parts or available")
        _require_sha256(self.intent_digest, "PreparedInput intent_digest")
        expected = input_v2_digest(self.user_prompt, manifest)
        if _require_sha256(self.input_digest, "PreparedInput input_digest") != expected:
            raise ValueError("PreparedInput input_digest does not match the frozen input")
        if not isinstance(self.path_origin, PathOrigin):
            raise TypeError("PreparedInput path_origin must be PathOrigin")
        object.__setattr__(self, "attachment_manifest", manifest)


@dataclass(frozen=True, slots=True)
class AttachmentUploadRecord:
    version: int
    owner_principal: Principal
    intent_digest: str
    descriptor: AttachmentSemanticEntry
    held_content: ContentRef | None
    status: AttachmentUploadStatus

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("AttachmentUploadRecord version must be 1")
        if not isinstance(self.owner_principal, Principal):
            raise TypeError("attachment upload owner_principal must be Principal")
        _require_sha256(self.intent_digest, "attachment upload intent_digest")
        if not isinstance(self.descriptor, AttachmentSemanticEntry):
            raise TypeError("attachment upload descriptor must be AttachmentSemanticEntry")
        if self.status == "HELD":
            if not isinstance(self.held_content, ContentRef):
                raise ValueError("HELD attachment upload must retain content")
            if (
                self.held_content.object.digest != self.descriptor.digest
                or self.held_content.object.size != self.descriptor.size
            ):
                raise ValueError("attachment upload descriptor does not match retained content")
        elif self.status == "RELEASED":
            if self.held_content is not None:
                raise ValueError("RELEASED attachment upload cannot retain content")
        else:
            raise ValueError("attachment upload status is invalid")


@dataclass(frozen=True, slots=True)
class InputPrepareSlot:
    slot: int
    relative: str | None
    entry: AttachmentEntry

    def __post_init__(self) -> None:
        _require_int(self.slot, "input prepare slot")
        if self.relative is not None:
            LogicalPath("attachment-input", self.relative)
        if not isinstance(self.entry, AttachmentEntry):
            raise TypeError("input prepare slot entry must be AttachmentEntry")


@dataclass(frozen=True, slots=True)
class InputTarget:
    at: Locator
    node_id: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.at, Locator):
            raise TypeError("input target locator must be Locator")
        if self.at.space != "records" or self.at.resource not in {
            "state:execution",
            "state:task",
        }:
            raise ValueError("input target must identify an execution or task record")
        if self.at.resource == "state:execution":
            if self.node_id is not None:
                raise ValueError("execution input target node_id must be null")
        else:
            _require_string(self.node_id, "task input target node_id")


@dataclass(frozen=True, slots=True)
class InputPrepareRecord:
    version: int
    intent_digest: str
    path_origin: PathOrigin
    status: InputPrepareStatus
    slots: tuple[InputPrepareSlot, ...]
    input: PreparedInput | None
    target: InputTarget | None
    error_code: str | None

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("InputPrepareRecord version must be 1")
        _require_sha256(self.intent_digest, "input prepare intent_digest")
        if not isinstance(self.path_origin, PathOrigin):
            raise TypeError("input prepare path_origin must be PathOrigin")
        slots = tuple(self.slots)
        if any(not isinstance(item, InputPrepareSlot) for item in slots):
            raise TypeError("input prepare slots contain invalid values")
        slot_ids = [item.slot for item in slots]
        if slot_ids != sorted(slot_ids) or len(slot_ids) != len(set(slot_ids)):
            raise ValueError("input prepare slots must be unique and ordered")
        if self.status == "PREPARING":
            if self.input is not None or self.target is not None or self.error_code is not None:
                raise ValueError("PREPARING input prepare has invalid terminal fields")
        elif self.status == "READY":
            if slots or not isinstance(self.input, PreparedInput) or self.target is not None or self.error_code is not None:
                raise ValueError("READY input prepare state is invalid")
            if self.input.intent_digest != self.intent_digest or self.input.path_origin != self.path_origin:
                raise ValueError("READY input prepare does not match its frozen intent/origin")
        elif self.status == "ADOPTED":
            if slots or self.input is not None or not isinstance(self.target, InputTarget) or self.error_code is not None:
                raise ValueError("ADOPTED input prepare state is invalid")
        elif self.status == "ABORTED":
            if slots or self.input is not None or self.target is not None:
                raise ValueError("ABORTED input prepare state is invalid")
            _require_string(self.error_code, "aborted input prepare error_code")
        else:
            raise ValueError("input prepare status is invalid")
        object.__setattr__(self, "slots", slots)


@dataclass(frozen=True, slots=True)
class AttachmentSourceRecord:
    version: int
    execution_id: str
    relative: str
    entry: AttachmentEntry

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("AttachmentSourceRecord version must be 1")
        _require_string(self.execution_id, "attachment source execution_id")
        LogicalPath("attachment-source", self.relative)
        if not isinstance(self.entry, AttachmentEntry):
            raise TypeError("attachment source entry must be AttachmentEntry")
        kind, _owner, slot = _validate_managed_path(self.entry.path)
        if kind != "e" or slot != 0:
            raise ValueError("attachment source entry must use an e attachment path")


@dataclass(frozen=True, slots=True)
class AttachmentResult:
    version: int
    entry: AttachmentEntry

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("AttachmentResult version must be 1")
        if not isinstance(self.entry, AttachmentEntry):
            raise TypeError("AttachmentResult entry must be AttachmentEntry")


@dataclass(frozen=True, slots=True)
class ModelExposureEntry:
    activation_id: str
    source: Locator
    slot: int
    entry: AttachmentEntry

    def __post_init__(self) -> None:
        _require_sha256(self.activation_id, "model exposure activation_id")
        if not isinstance(self.source, Locator):
            raise TypeError("model exposure source must be Locator")
        _require_int(self.slot, "model exposure slot")
        if not isinstance(self.entry, AttachmentEntry):
            raise TypeError("model exposure entry must be AttachmentEntry")


@dataclass(frozen=True, slots=True)
class ModelExposure:
    version: int
    exposure_id: str
    execution_id: str
    step_run_id: str
    run_step: int
    path_origin: PathOrigin
    entries: tuple[ModelExposureEntry, ...]
    activation_digest: str

    def __post_init__(self) -> None:
        if self.version != 1 or isinstance(self.version, bool):
            raise ValueError("ModelExposure version must be 1")
        _require_sha256(self.exposure_id, "model exposure exposure_id")
        _require_string(self.execution_id, "model exposure execution_id")
        _require_string(self.step_run_id, "model exposure step_run_id")
        _require_int(self.run_step, "model exposure run_step")
        if not isinstance(self.path_origin, PathOrigin):
            raise TypeError("model exposure path_origin must be PathOrigin")
        entries = tuple(self.entries)
        if any(not isinstance(item, ModelExposureEntry) for item in entries):
            raise TypeError("model exposure entries contain invalid values")
        if len({item.activation_id for item in entries}) != len(entries):
            raise ValueError("model exposure entries contain duplicate activation ids")
        expected = model_exposure_activation_digest(entries)
        if _require_sha256(self.activation_digest, "model exposure activation_digest") != expected:
            raise ValueError("model exposure activation_digest is invalid")
        object.__setattr__(self, "entries", entries)


def semantic_attachment_entry(entry: AttachmentEntry) -> AttachmentSemanticEntry:
    """Project an attachment onto its portable semantic identity."""
    if not isinstance(entry, AttachmentEntry):
        raise TypeError("entry must be AttachmentEntry")
    return AttachmentSemanticEntry(
        entry.path,
        entry.name,
        entry.media_type,
        entry.presentation,
        entry.content.object.digest,
        entry.content.object.size,
    )


def input_v2_digest(
    prompt: InputV2,
    manifest: Sequence[AttachmentEntry],
) -> str:
    """Return the semantic digest for one frozen managed input."""
    if not isinstance(prompt, InputV2):
        raise TypeError("prompt must be InputV2")
    values = tuple(manifest)
    if any(not isinstance(item, AttachmentEntry) for item in values):
        raise TypeError("manifest must contain AttachmentEntry values")
    return canonical_sha256(
        {
            "codec": "linktools-input-v2",
            "prompt": _input_v2_semantic_json(prompt),
            "manifest": [
                _semantic_entry_json(semantic_attachment_entry(item)) for item in values
            ],
        }
    )


def model_exposure_activation_digest(
    entries: Sequence[ModelExposureEntry],
) -> str:
    """Return the ordered semantic activation digest for a model exposure."""
    values = tuple(entries)
    if any(not isinstance(item, ModelExposureEntry) for item in values):
        raise TypeError("entries must contain ModelExposureEntry values")
    return canonical_sha256(
        [
            {
                "activation_id": item.activation_id,
                "source": item.source.to_json(),
                "slot": item.slot,
                "entry": _semantic_entry_json(semantic_attachment_entry(item.entry)),
            }
            for item in values
        ]
    )


def _input_v2_semantic_json(prompt: InputV2) -> dict[str, JsonValue]:
    parts: list[JsonValue] = []
    for part in prompt.parts:
        if isinstance(part, InputTextPart):
            parts.append({"kind": "text", "text": part.text})
        elif isinstance(part, InputNativePart):
            parts.append(
                {
                    "kind": "native",
                    "codec": "pydantic-user-content-v1",
                    "value": dict(part.value),
                }
            )
        else:
            parts.append({"kind": "attachment", "index": part.index})
    return {
        "version": 2,
        "parts": parts,
        "available": list(prompt.available),
        "sources": [
            {"relative": item.relative, "index": item.index} for item in prompt.sources
        ],
    }


def _semantic_entry_json(entry: AttachmentSemanticEntry) -> dict[str, JsonValue]:
    return {
        "path": entry.path,
        "name": entry.name,
        "media_type": entry.media_type,
        "presentation": {
            "identifier": entry.presentation.identifier,
            "vendor_metadata": None
            if entry.presentation.vendor_metadata is None
            else dict(entry.presentation.vendor_metadata),
        },
        "digest": entry.digest,
        "size": entry.size,
    }


__all__ = [
    "AttachmentEntry",
    "AttachmentPresentation",
    "AttachmentResult",
    "AttachmentSemanticEntry",
    "AttachmentSourceRecord",
    "AttachmentUploadRecord",
    "AttachmentUploadStatus",
    "ContentRef",
    "InputAttachmentPart",
    "InputNativePart",
    "InputPart",
    "InputPrepareRecord",
    "InputPrepareSlot",
    "InputPrepareStatus",
    "InputSource",
    "InputTarget",
    "InputTextPart",
    "InputV2",
    "ModelExposure",
    "ModelExposureEntry",
    "PreparedInput",
    "input_v2_digest",
    "managed_attachment_locator",
    "managed_attachment_path",
    "model_exposure_activation_digest",
    "semantic_attachment_entry",
]
