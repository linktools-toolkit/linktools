#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned canonical transport for Pydantic AI user content."""

import base64
import hashlib
import json
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import TypeAlias, cast

from pydantic_ai import ModelMessagesTypeAdapter
from pydantic_ai.messages import (
    BinaryContent,
    ModelRequest,
    UploadedFile,
    UserContent,
    UserPromptPart,
)

from ..core import JsonValue, canonical_json_bytes, normalize_json_value, validate_user_prompt
from ..errors import AIError, ErrorCode
from .state import (
    InputAttachmentPart,
    InputNativePart,
    InputSource,
    InputTextPart,
    InputV2,
    PreparedInput,
)

_TEXT_CODEC = "text"
_USER_CONTENT_CODEC = "pydantic-user-content-v1"
_PORTABLE_INPUT_CODEC = "linktools-input-v2"
_MANAGED_DRAFT_CODEC = "linktools-managed-draft"
_WIRE_TIMESTAMP = datetime(1970, 1, 1, tzinfo=timezone.utc)

_UserPromptInput: TypeAlias = str | Sequence[UserContent]
_RuntimeUserPrompt: TypeAlias = str
DraftPrompt: TypeAlias = JsonValue
TaskPrompt: TypeAlias = JsonValue


class UserPromptTransport(str):
    """Internal prompt value carrying a durable codec or an in-memory draft."""

    __slots__ = ("codec", "draft")

    def __new__(
        cls,
        value: str,
        codec: str,
        draft: "_UserPromptInput | None" = None,
    ) -> "UserPromptTransport":
        if not isinstance(value, str) or not isinstance(codec, str) or not codec:
            raise TypeError("user prompt transport is invalid")
        if draft is not None and codec != _MANAGED_DRAFT_CODEC:
            raise TypeError("only managed draft transport can retain raw prompt content")
        instance = str.__new__(cls, value)
        instance.codec = codec
        instance.draft = draft
        return instance

    def __add__(self, other: object) -> "UserPromptTransport":
        if not isinstance(other, str):
            return NotImplemented
        if self.draft is None:
            return UserPromptTransport(str.__add__(self, other), self.codec)
        if isinstance(self.draft, str):
            draft: _UserPromptInput = self.draft + other
        else:
            draft = (*tuple(self.draft), other)
        return UserPromptTransport(
            str.__add__(self, other),
            self.codec,
            draft,
        )


def prepare_user_prompt(value: _UserPromptInput) -> UserPromptTransport:
    """Prepare legacy durable transport or retain a binary-bearing in-memory draft."""
    if isinstance(value, str):
        validate_user_prompt(value)
        return UserPromptTransport(value, _TEXT_CODEC)
    content = _require_user_content_sequence(value)
    if any(isinstance(item, UploadedFile) for item in content):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={
                "field": "user_prompt",
                "reason": "uploaded_file_not_durable",
            },
        )
    if any(isinstance(item, BinaryContent) for item in content):
        _draft_prompt(content)
        return UserPromptTransport(
            "managed-input",
            _MANAGED_DRAFT_CODEC,
            content,
        )
    payload = _encode_user_content(content)
    wire = canonical_json_bytes(payload).decode("utf-8")
    validate_user_prompt(wire)
    return UserPromptTransport(wire, _USER_CONTENT_CODEC)


def managed_user_prompt_draft(
    value: UserPromptTransport,
) -> "_UserPromptInput | None":
    """Return binary-bearing raw content before any durable request is constructed."""
    if not isinstance(value, UserPromptTransport):
        raise TypeError("value must be UserPromptTransport")
    if value.codec == _MANAGED_DRAFT_CODEC:
        if value.draft is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value.draft
    if value.draft is not None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return None


def input_intent_digest(
    value: _UserPromptInput,
    attachments: Sequence[str] = (),
) -> str:
    """Return the pre-import semantic intent digest without embedding binary bodies."""
    raw_attachments = _require_attachment_paths(attachments)
    draft = _draft_prompt(value)
    return hashlib.sha256(
        canonical_json_bytes(
            {
                "version": 1,
                "prompt": draft,
                "attachments": list(raw_attachments),
            }
        )
    ).hexdigest()


def task_prompt_draft(value: _UserPromptInput) -> TaskPrompt:
    """Encode the in-memory Agent Task v2 prompt without persistence semantics."""
    if isinstance(value, str):
        validate_user_prompt(value)
        return {"kind": "text", "text": value}
    content = _require_user_content_sequence(value)
    result: list[JsonValue] = []
    for item in content:
        if isinstance(item, UploadedFile):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "field": "user_prompt",
                    "reason": "uploaded_file_not_durable",
                },
            )
        if isinstance(item, str):
            result.append({"kind": "text", "text": item})
            continue
        if isinstance(item, BinaryContent):
            if not item.data or not isinstance(item.media_type, str) or not item.media_type:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            result.append(
                {
                    "kind": "binary",
                    "data_b64": base64.b64encode(item.data).decode("ascii"),
                    "media_type": item.media_type,
                    "identifier": item.identifier,
                    "vendor_metadata": _json_object_or_none(item.vendor_metadata),
                }
            )
            continue
        result.append(
            {
                "kind": "native",
                "codec": _USER_CONTENT_CODEC,
                "value": _encode_user_content((item,)),
            }
        )
    return result


def prepared_user_prompt_transport(value: PreparedInput) -> UserPromptTransport:
    """Encode a frozen managed input without serializing any attachment body."""
    if not isinstance(value, PreparedInput):
        raise TypeError("value must be PreparedInput")
    wire = canonical_json_bytes(_input_v2_to_json(value.user_prompt)).decode("utf-8")
    validate_user_prompt(wire)
    return UserPromptTransport(wire, _PORTABLE_INPUT_CODEC)


def user_prompt_transport(value: str, codec: str = _TEXT_CODEC) -> UserPromptTransport:
    """Restore an internal prompt transport after a durable boundary."""
    validate_user_prompt(value)
    if codec not in {_TEXT_CODEC, _USER_CONTENT_CODEC, _PORTABLE_INPUT_CODEC}:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if codec == _PORTABLE_INPUT_CODEC:
        _decode_input_v2_wire(value)
    return UserPromptTransport(value, codec)


def _restore_user_prompt(value: str) -> str | tuple[UserContent, ...]:
    """Restore legacy durable text transport or an in-memory managed draft."""
    if isinstance(value, UserPromptTransport) and value.codec == _MANAGED_DRAFT_CODEC:
        draft = managed_user_prompt_draft(value)
        if isinstance(draft, str):
            return draft
        if draft is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content = tuple(draft)
        if len(content) == 1 and isinstance(content[0], str):
            return content[0]
        return content
    validate_user_prompt(value)
    codec = value.codec if isinstance(value, UserPromptTransport) else _TEXT_CODEC
    if codec == _TEXT_CODEC:
        return str(value)
    if codec == _USER_CONTENT_CODEC:
        return _decode_user_content_wire(str(value))
    if codec == _PORTABLE_INPUT_CODEC:
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _restore_input_v2(value: str) -> InputV2:
    """Restore one canonical managed-input transport for the attachment pipeline."""
    validate_user_prompt(value)
    return _decode_input_v2_wire(value)


def _draft_prompt(value: _UserPromptInput) -> DraftPrompt:
    if isinstance(value, str):
        validate_user_prompt(value)
        return {"kind": "text", "text": value}
    content = _require_user_content_sequence(value)
    draft: list[JsonValue] = []
    for item in content:
        if isinstance(item, UploadedFile):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "field": "user_prompt",
                    "reason": "uploaded_file_not_durable",
                },
            )
        if isinstance(item, str):
            draft.append({"kind": "text", "text": item})
            continue
        if isinstance(item, BinaryContent):
            if not item.data or not isinstance(item.media_type, str) or not item.media_type:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            vendor_metadata = _json_object_or_none(item.vendor_metadata)
            draft.append(
                {
                    "kind": "binary",
                    "digest": hashlib.sha256(item.data).hexdigest(),
                    "size": len(item.data),
                    "media_type": item.media_type,
                    "identifier": item.identifier,
                    "vendor_metadata": vendor_metadata,
                }
            )
            continue
        draft.append(
            {
                "kind": "native",
                "codec": _USER_CONTENT_CODEC,
                "value": _encode_user_content((item,)),
            }
        )
    return draft


def _input_v2_to_json(value: InputV2) -> dict[str, JsonValue]:
    if not isinstance(value, InputV2):
        raise TypeError("value must be InputV2")
    parts: list[JsonValue] = []
    for part in value.parts:
        if isinstance(part, InputTextPart):
            parts.append({"kind": "text", "text": part.text})
        elif isinstance(part, InputNativePart):
            parts.append(
                {
                    "kind": "native",
                    "codec": _USER_CONTENT_CODEC,
                    "value": dict(part.value),
                }
            )
        elif isinstance(part, InputAttachmentPart):
            parts.append({"kind": "attachment", "index": part.index})
        else:
            raise TypeError("InputV2 contains an unsupported part")
    return {
        "version": 2,
        "parts": parts,
        "available": list(value.available),
        "sources": [
            {"relative": source.relative, "index": source.index}
            for source in value.sources
        ],
    }


def _decode_input_v2_wire(value: str) -> InputV2:
    try:
        payload = json.loads(value, object_pairs_hook=_reject_duplicate_keys)
        normalized = normalize_json_value(payload)
    except (json.JSONDecodeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict) or canonical_json_bytes(normalized) != value.encode("utf-8"):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if set(normalized) != {"version", "parts", "available", "sources"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    version = normalized["version"]
    parts_value = normalized["parts"]
    available_value = normalized["available"]
    sources_value = normalized["sources"]
    if version != 2 or isinstance(version, bool):
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if (
        not isinstance(parts_value, list)
        or not isinstance(available_value, list)
        or not isinstance(sources_value, list)
    ):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        parts = tuple(_decode_input_part(item) for item in parts_value)
        available = tuple(_strict_nonnegative_int(item) for item in available_value)
        sources = tuple(_decode_input_source(item) for item in sources_value)
        result = InputV2(2, parts, available, sources)
    except AIError:
        raise
    except (TypeError, ValueError, KeyError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if _input_v2_to_json(result) != normalized:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


def _decode_input_part(
    value: JsonValue,
) -> InputTextPart | InputNativePart | InputAttachmentPart:
    if not isinstance(value, Mapping):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    kind = value.get("kind")
    if kind == "text":
        if set(value) != {"kind", "text"} or not isinstance(value.get("text"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return InputTextPart("text", cast(str, value["text"]))
    if kind == "attachment":
        if set(value) != {"kind", "index"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return InputAttachmentPart(
            "attachment", _strict_nonnegative_int(value["index"])
        )
    if kind == "native":
        if (
            set(value) != {"kind", "codec", "value"}
            or value.get("codec") != _USER_CONTENT_CODEC
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        native = value.get("value")
        if not isinstance(native, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return InputNativePart(
            "native",
            _USER_CONTENT_CODEC,
            cast(Mapping[str, JsonValue], native),
        )
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _decode_input_source(value: JsonValue) -> InputSource:
    if not isinstance(value, Mapping) or set(value) != {"relative", "index"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    relative = value.get("relative")
    if not isinstance(relative, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return InputSource(relative, _strict_nonnegative_int(value["index"]))


def _strict_nonnegative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _reject_duplicate_keys(
    pairs: list[tuple[str, JsonValue]],
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {}
    for key, item in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = item
    return result


def _json_object_or_none(value: object) -> JsonValue:
    if value is None:
        return None
    try:
        normalized = normalize_json_value(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return normalized


def _require_attachment_paths(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return result


def _require_user_content_sequence(value: _UserPromptInput) -> tuple[UserContent, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    content = tuple(value)
    if not content:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return content


def _decode_user_content_wire(value: str) -> tuple[UserContent, ...]:
    try:
        payload, end = json.JSONDecoder().raw_decode(value)
    except json.JSONDecodeError as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    payload_text = value[:end]
    try:
        normalized = normalize_json_value(payload)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if canonical_json_bytes(normalized) != payload_text.encode("utf-8"):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = _decode_user_content(normalized)
    suffix = value[end:]
    return (*content, suffix) if suffix else content


def _encode_user_content(content: Sequence[UserContent]) -> dict[str, JsonValue]:
    request = ModelRequest(
        parts=[
            UserPromptPart(
                content=tuple(content),
                timestamp=_WIRE_TIMESTAMP,
            )
        ]
    )
    try:
        encoded = ModelMessagesTypeAdapter.dump_python([request], mode="json")
        normalized = normalize_json_value(encoded)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if (
        not isinstance(normalized, list)
        or len(normalized) != 1
        or not isinstance(normalized[0], dict)
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return {"message": normalized[0]}


def _decode_user_content(payload: dict[str, JsonValue]) -> tuple[UserContent, ...]:
    if set(payload) != {"message"}:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    message = payload["message"]
    if not isinstance(message, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        raw = canonical_json_bytes([message])
        messages = ModelMessagesTypeAdapter.validate_json(raw)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if len(messages) != 1 or not isinstance(messages[0], ModelRequest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    request = messages[0]
    if len(request.parts) != 1 or not isinstance(request.parts[0], UserPromptPart):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    part = request.parts[0]
    if part.timestamp != _WIRE_TIMESTAMP or isinstance(part.content, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    content = tuple(cast(Sequence[UserContent], part.content))
    if not content or any(isinstance(item, UploadedFile) for item in content):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return content


__all__: tuple[str, ...] = ()
