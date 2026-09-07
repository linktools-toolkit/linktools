#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical wire support for portable attachment owner records."""

import sys
from collections.abc import Iterator, Mapping
from types import MappingProxyType
from typing import Any, cast

from ...core import JsonValue, Principal
from ...errors import AIError, ErrorCode
from ...storage import ObjectRef
from ._attachments import (
    AttachmentEntry,
    AttachmentPresentation,
    AttachmentSemanticEntry,
    AttachmentSourceRecord,
    AttachmentUploadRecord,
    ContentRef,
    InputAttachmentPart,
    InputNativePart,
    InputPrepareRecord,
    InputPrepareSlot,
    InputSource,
    InputTarget,
    InputTextPart,
    InputV2,
    ModelExposure,
    ModelExposureEntry,
    PreparedInput,
)
from ._codec import (
    _V1_DATACLASS_DECODERS,
    _V1_DATACLASS_ENCODERS,
    _V1_DOMAIN_TYPES,
    _V1_ENUM_TYPES,
    _V1_ENUM_WIRE_IDS,
    _V1_EXTERNAL_SCHEMA_TYPES,
    _V1_WIRE_IDS,
    _V1_WIRE_TYPES,
    _VersionCodec,
    _decode_domain,
    _encode_domain,
    _iter_runtime_object_refs,
)
from ._plan import RuntimeDomain
from ._relocation import Locator, PathOrigin

_ATTACHMENT_WIRE_TYPES = (
    ("attachment_upload_v1", AttachmentUploadRecord),
    ("input_prepare_v1", InputPrepareRecord),
    ("attachment_source_v1", AttachmentSourceRecord),
    ("model_exposure_v1", ModelExposure),
)
_ATTACHMENT_WIRE_IDS = frozenset(item[0] for item in _ATTACHMENT_WIRE_TYPES)
_CODEC_MODULE = cast(Any, sys.modules[__package__ + "._codec"])


def install_attachment_codec() -> None:
    """Extend the Runtime v1 codec with attachment owner records."""
    current = dict(_CODEC_MODULE._V1_WIRE_IDS)
    if all(
        current.get(target) == wire_id
        for wire_id, target in _ATTACHMENT_WIRE_TYPES
    ):
        return
    if any(wire_id in _V1_DOMAIN_TYPES for wire_id in _ATTACHMENT_WIRE_IDS):
        raise RuntimeError("attachment wire id conflicts with Runtime codec")
    if any(target in _V1_WIRE_IDS for _wire_id, target in _ATTACHMENT_WIRE_TYPES):
        raise RuntimeError("attachment type conflicts with Runtime codec")

    wire_types = (*_V1_WIRE_TYPES, *_ATTACHMENT_WIRE_TYPES)
    wire_ids = MappingProxyType({target: wire_id for wire_id, target in wire_types})
    domain_types = MappingProxyType({wire_id: target for wire_id, target in wire_types})
    encoders = MappingProxyType(
        {
            **dict(_V1_DATACLASS_ENCODERS),
            "attachment_upload_v1": _encode_upload,
            "input_prepare_v1": _encode_prepare,
            "attachment_source_v1": _encode_source,
            "model_exposure_v1": _encode_exposure,
        }
    )
    decoders = MappingProxyType(
        {
            **dict(_V1_DATACLASS_DECODERS),
            "attachment_upload_v1": _decode_upload,
            "input_prepare_v1": _decode_prepare,
            "attachment_source_v1": _decode_source,
            "model_exposure_v1": _decode_exposure,
        }
    )
    codec = _VersionCodec(
        version=1,
        wire_ids=wire_ids,
        domain_types=domain_types,
        enum_wire_ids=_V1_ENUM_WIRE_IDS,
        enum_types=_V1_ENUM_TYPES,
        dataclass_encoders=encoders,
        dataclass_decoders=decoders,
        external_schema_types=_V1_EXTERNAL_SCHEMA_TYPES,
    )
    original_iter = _iter_runtime_object_refs

    def iter_object_refs(
        value: object,
        domain: RuntimeDomain,
        current_codec: Any,
    ) -> Iterator[tuple[RuntimeDomain, ObjectRef]]:
        if isinstance(value, Mapping):
            wire_id = value.get("$dataclass")
            target = domain_types.get(wire_id) if isinstance(wire_id, str) else None
            if wire_id in _ATTACHMENT_WIRE_IDS and target is not None:
                decoded = _decode_domain(
                    value,
                    target,
                    current_codec,
                    persisted=True,
                )
                yield from _record_refs(decoded, cast(str, wire_id))
                return
        yield from original_iter(value, domain, current_codec)

    _CODEC_MODULE._V1_WIRE_TYPES = wire_types
    _CODEC_MODULE._V1_WIRE_IDS = wire_ids
    _CODEC_MODULE._V1_DOMAIN_TYPES = domain_types
    _CODEC_MODULE._V1_DATACLASS_ENCODERS = encoders
    _CODEC_MODULE._V1_DATACLASS_DECODERS = decoders
    _CODEC_MODULE._V1_CODEC = codec
    _CODEC_MODULE._VERSION_CODECS = MappingProxyType({1: codec})
    _CODEC_MODULE._CURRENT_CODEC = codec
    _CODEC_MODULE._iter_runtime_object_refs = iter_object_refs


def _encode_upload(
    value: object,
    codec: Any,
    persisted: bool,
) -> Mapping[str, JsonValue]:
    if not isinstance(value, AttachmentUploadRecord):
        raise TypeError("attachment_upload_v1 received the wrong type")
    return {
        "version": 1,
        "owner_principal": _encode_domain(
            value.owner_principal,
            codec,
            persisted=persisted,
        ),
        "intent_digest": value.intent_digest,
        "descriptor": _semantic_json(value.descriptor),
        "held_content": (
            None if value.held_content is None else _content_json(value.held_content)
        ),
        "status": value.status,
    }


def _decode_upload(
    raw: Mapping[str, object],
    codec: Any,
    persisted: bool,
) -> AttachmentUploadRecord:
    _fields(
        raw,
        {
            "version",
            "owner_principal",
            "intent_digest",
            "descriptor",
            "held_content",
            "status",
        },
    )
    _version(raw["version"], 1)
    owner = _decode_domain(
        raw["owner_principal"],
        Principal,
        codec,
        persisted=persisted,
    )
    held = raw["held_content"]
    if not isinstance(owner, Principal):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AttachmentUploadRecord(
        1,
        owner,
        _string(raw["intent_digest"]),
        _semantic(raw["descriptor"]),
        None if held is None else _content(held),
        cast(Any, _string(raw["status"])),
    )


def _encode_prepare(
    value: object,
    codec: Any,
    persisted: bool,
) -> Mapping[str, JsonValue]:
    del codec, persisted
    if not isinstance(value, InputPrepareRecord):
        raise TypeError("input_prepare_v1 received the wrong type")
    return {
        "version": 1,
        "intent_digest": value.intent_digest,
        "path_origin": value.path_origin.to_json(),
        "status": value.status,
        "slots": [_slot_json(item) for item in value.slots],
        "input": None if value.input is None else _prepared_json(value.input),
        "target": None if value.target is None else _target_json(value.target),
        "error_code": value.error_code,
    }


def _decode_prepare(
    raw: Mapping[str, object],
    codec: Any,
    persisted: bool,
) -> InputPrepareRecord:
    del codec, persisted
    _fields(
        raw,
        {
            "version",
            "intent_digest",
            "path_origin",
            "status",
            "slots",
            "input",
            "target",
            "error_code",
        },
    )
    _version(raw["version"], 1)
    error_code = raw["error_code"]
    if error_code is not None and not isinstance(error_code, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return InputPrepareRecord(
        1,
        _string(raw["intent_digest"]),
        PathOrigin.from_json(raw["path_origin"]),
        cast(Any, _string(raw["status"])),
        tuple(_slot(item) for item in _array(raw["slots"])),
        None if raw["input"] is None else _prepared(raw["input"]),
        None if raw["target"] is None else _target(raw["target"]),
        error_code,
    )


def _encode_source(
    value: object,
    codec: Any,
    persisted: bool,
) -> Mapping[str, JsonValue]:
    del codec, persisted
    if not isinstance(value, AttachmentSourceRecord):
        raise TypeError("attachment_source_v1 received the wrong type")
    return {
        "version": 1,
        "execution_id": value.execution_id,
        "relative": value.relative,
        "entry": _entry_json(value.entry),
    }


def _decode_source(
    raw: Mapping[str, object],
    codec: Any,
    persisted: bool,
) -> AttachmentSourceRecord:
    del codec, persisted
    _fields(raw, {"version", "execution_id", "relative", "entry"})
    _version(raw["version"], 1)
    return AttachmentSourceRecord(
        1,
        _string(raw["execution_id"]),
        _string(raw["relative"]),
        _entry(raw["entry"]),
    )


def _encode_exposure(
    value: object,
    codec: Any,
    persisted: bool,
) -> Mapping[str, JsonValue]:
    del codec, persisted
    if not isinstance(value, ModelExposure):
        raise TypeError("model_exposure_v1 received the wrong type")
    return {
        "version": 1,
        "exposure_id": value.exposure_id,
        "execution_id": value.execution_id,
        "step_run_id": value.step_run_id,
        "run_step": value.run_step,
        "path_origin": value.path_origin.to_json(),
        "entries": [_exposure_entry_json(item) for item in value.entries],
        "activation_digest": value.activation_digest,
    }


def _decode_exposure(
    raw: Mapping[str, object],
    codec: Any,
    persisted: bool,
) -> ModelExposure:
    del codec, persisted
    _fields(
        raw,
        {
            "version",
            "exposure_id",
            "execution_id",
            "step_run_id",
            "run_step",
            "path_origin",
            "entries",
            "activation_digest",
        },
    )
    _version(raw["version"], 1)
    return ModelExposure(
        1,
        _string(raw["exposure_id"]),
        _string(raw["execution_id"]),
        _string(raw["step_run_id"]),
        _integer(raw["run_step"]),
        PathOrigin.from_json(raw["path_origin"]),
        tuple(_exposure_entry(item) for item in _array(raw["entries"])),
        _string(raw["activation_digest"]),
    )


def _record_refs(
    value: object,
    wire_id: str,
) -> tuple[tuple[RuntimeDomain, ObjectRef], ...]:
    if wire_id == "attachment_upload_v1":
        record = cast(AttachmentUploadRecord, value)
        return () if record.held_content is None else (_content_ref(record.held_content),)
    if wire_id == "input_prepare_v1":
        record = cast(InputPrepareRecord, value)
        refs = [_content_ref(item.entry.content) for item in record.slots]
        if record.input is not None:
            refs.extend(
                _content_ref(item.content)
                for item in record.input.attachment_manifest
            )
        return tuple(refs)
    if wire_id == "attachment_source_v1":
        return (_content_ref(cast(AttachmentSourceRecord, value).entry.content),)
    if wire_id == "model_exposure_v1":
        record = cast(ModelExposure, value)
        return tuple(_content_ref(item.entry.content) for item in record.entries)
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _content_ref(value: ContentRef) -> tuple[RuntimeDomain, ObjectRef]:
    try:
        return RuntimeDomain(value.domain), value.object
    except ValueError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _presentation_json(value: AttachmentPresentation) -> dict[str, JsonValue]:
    return {
        "identifier": value.identifier,
        "vendor_metadata": (
            None if value.vendor_metadata is None else dict(value.vendor_metadata)
        ),
    }


def _presentation(value: object) -> AttachmentPresentation:
    raw = _object(value)
    _fields(raw, {"identifier", "vendor_metadata"})
    identifier = raw["identifier"]
    metadata = raw["vendor_metadata"]
    if identifier is not None and not isinstance(identifier, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if metadata is not None and not isinstance(metadata, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AttachmentPresentation(
        identifier,
        None if metadata is None else cast(Mapping[str, JsonValue], metadata),
    )


def _content_json(value: ContentRef) -> dict[str, JsonValue]:
    return {
        "domain": value.domain,
        "owner_scope": value.owner_scope,
        "object": {
            "store_id": value.object.store_id,
            "key": value.object.key,
            "digest": value.object.digest,
            "size": value.object.size,
        },
    }


def _content(value: object) -> ContentRef:
    raw = _object(value)
    _fields(raw, {"domain", "owner_scope", "object"})
    owner_scope = raw["owner_scope"]
    if owner_scope is not None and not isinstance(owner_scope, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    obj = _object(raw["object"])
    _fields(obj, {"store_id", "key", "digest", "size"})
    return ContentRef(
        _string(raw["domain"]),
        owner_scope,
        ObjectRef(
            _string(obj["store_id"]),
            _string(obj["key"]),
            _string(obj["digest"]),
            _integer(obj["size"]),
        ),
    )


def _entry_json(value: AttachmentEntry) -> dict[str, JsonValue]:
    return {
        "path": value.path,
        "name": value.name,
        "media_type": value.media_type,
        "presentation": _presentation_json(value.presentation),
        "content": _content_json(value.content),
    }


def _entry(value: object) -> AttachmentEntry:
    raw = _object(value)
    _fields(raw, {"path", "name", "media_type", "presentation", "content"})
    name = raw["name"]
    if name is not None and not isinstance(name, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AttachmentEntry(
        _string(raw["path"]),
        name,
        _string(raw["media_type"]),
        _presentation(raw["presentation"]),
        _content(raw["content"]),
    )


def _semantic_json(value: AttachmentSemanticEntry) -> dict[str, JsonValue]:
    return {
        "path": value.path,
        "name": value.name,
        "media_type": value.media_type,
        "presentation": _presentation_json(value.presentation),
        "digest": value.digest,
        "size": value.size,
    }


def _semantic(value: object) -> AttachmentSemanticEntry:
    raw = _object(value)
    _fields(raw, {"path", "name", "media_type", "presentation", "digest", "size"})
    name = raw["name"]
    if name is not None and not isinstance(name, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return AttachmentSemanticEntry(
        _string(raw["path"]),
        name,
        _string(raw["media_type"]),
        _presentation(raw["presentation"]),
        _string(raw["digest"]),
        _integer(raw["size"]),
    )


def _input_json(value: InputV2) -> dict[str, JsonValue]:
    parts: list[JsonValue] = []
    for part in value.parts:
        if isinstance(part, InputTextPart):
            parts.append({"kind": "text", "text": part.text})
        elif isinstance(part, InputNativePart):
            parts.append(
                {"kind": "native", "codec": part.codec, "value": dict(part.value)}
            )
        elif isinstance(part, InputAttachmentPart):
            parts.append({"kind": "attachment", "index": part.index})
        else:
            raise TypeError("unsupported InputV2 part")
    return {
        "version": 2,
        "parts": parts,
        "available": list(value.available),
        "sources": [
            {"relative": item.relative, "index": item.index}
            for item in value.sources
        ],
    }


def _input(value: object) -> InputV2:
    raw = _object(value)
    _fields(raw, {"version", "parts", "available", "sources"})
    _version(raw["version"], 2)
    parts: list[InputTextPart | InputNativePart | InputAttachmentPart] = []
    for item in _array(raw["parts"]):
        part = _object(item)
        kind = part.get("kind")
        if kind == "text":
            _fields(part, {"kind", "text"})
            parts.append(
                InputTextPart("text", _string(part["text"], nonempty=False))
            )
        elif kind == "native":
            _fields(part, {"kind", "codec", "value"})
            parts.append(
                InputNativePart(
                    "native",
                    cast(Any, _string(part["codec"])),
                    cast(Mapping[str, JsonValue], _object(part["value"])),
                )
            )
        elif kind == "attachment":
            _fields(part, {"kind", "index"})
            parts.append(
                InputAttachmentPart("attachment", _integer(part["index"]))
            )
        else:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return InputV2(
        2,
        tuple(parts),
        tuple(_integer(item) for item in _array(raw["available"])),
        tuple(_source(item) for item in _array(raw["sources"])),
    )


def _source(value: object) -> InputSource:
    raw = _object(value)
    _fields(raw, {"relative", "index"})
    return InputSource(_string(raw["relative"]), _integer(raw["index"]))


def _prepared_json(value: PreparedInput) -> dict[str, JsonValue]:
    return {
        "version": 1,
        "user_prompt_codec": value.user_prompt_codec,
        "user_prompt": _input_json(value.user_prompt),
        "attachment_manifest": [
            _entry_json(item) for item in value.attachment_manifest
        ],
        "intent_digest": value.intent_digest,
        "input_digest": value.input_digest,
        "path_origin": value.path_origin.to_json(),
    }


def _prepared(value: object) -> PreparedInput:
    raw = _object(value)
    _fields(
        raw,
        {
            "version",
            "user_prompt_codec",
            "user_prompt",
            "attachment_manifest",
            "intent_digest",
            "input_digest",
            "path_origin",
        },
    )
    _version(raw["version"], 1)
    return PreparedInput(
        1,
        cast(Any, _string(raw["user_prompt_codec"])),
        _input(raw["user_prompt"]),
        tuple(_entry(item) for item in _array(raw["attachment_manifest"])),
        _string(raw["intent_digest"]),
        _string(raw["input_digest"]),
        PathOrigin.from_json(raw["path_origin"]),
    )


def _slot_json(value: InputPrepareSlot) -> dict[str, JsonValue]:
    return {
        "slot": value.slot,
        "relative": value.relative,
        "entry": _entry_json(value.entry),
    }


def _slot(value: object) -> InputPrepareSlot:
    raw = _object(value)
    _fields(raw, {"slot", "relative", "entry"})
    relative = raw["relative"]
    if relative is not None and not isinstance(relative, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return InputPrepareSlot(
        _integer(raw["slot"]),
        relative,
        _entry(raw["entry"]),
    )


def _target_json(value: InputTarget) -> dict[str, JsonValue]:
    return {"at": value.at.to_json(), "node_id": value.node_id}


def _target(value: object) -> InputTarget:
    raw = _object(value)
    _fields(raw, {"at", "node_id"})
    node_id = raw["node_id"]
    if node_id is not None and not isinstance(node_id, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return InputTarget(Locator.from_json(raw["at"]), node_id)


def _exposure_entry_json(value: ModelExposureEntry) -> dict[str, JsonValue]:
    return {
        "activation_id": value.activation_id,
        "source": value.source.to_json(),
        "slot": value.slot,
        "entry": _entry_json(value.entry),
    }


def _exposure_entry(value: object) -> ModelExposureEntry:
    raw = _object(value)
    _fields(raw, {"activation_id", "source", "slot", "entry"})
    return ModelExposureEntry(
        _string(raw["activation_id"]),
        Locator.from_json(raw["source"]),
        _integer(raw["slot"]),
        _entry(raw["entry"]),
    )


def _object(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return cast(Mapping[str, object], value)


def _array(value: object) -> list[object]:
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return cast(list[object], value)


def _fields(value: Mapping[str, object], expected: set[str]) -> None:
    if set(value) != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _string(value: object, *, nonempty: bool = True) -> str:
    if not isinstance(value, str) or nonempty and not value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _version(value: object, expected: int) -> None:
    if isinstance(value, bool) or value != expected:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


__all__: tuple[str, ...] = ()
