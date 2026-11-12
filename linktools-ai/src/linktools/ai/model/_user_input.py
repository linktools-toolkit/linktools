#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure validation for user input accepted by runtime request contracts."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TypeAlias

from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    TextContent,
    UploadedFile,
    UserContent,
    VideoUrl,
)

from ..core import (
    ImmutableJsonMapping,
    JsonValue,
    WorkspaceFileInput,
    canonical_sha256,
    canonical_json_bytes,
    normalize_json_value,
    validate_user_prompt,
)
from ..errors import AIError, ErrorCode
from ..storage import StoredPayload

UserPromptInput: TypeAlias = str | Sequence[UserContent | WorkspaceFileInput]
CanonicalUserInput: TypeAlias = str | tuple[UserContent | WorkspaceFileInput, ...]
_USER_CONTENT_TYPES = (
    str,
    TextContent,
    ImageUrl,
    AudioUrl,
    DocumentUrl,
    VideoUrl,
    BinaryContent,
    UploadedFile,
    CachePoint,
    WorkspaceFileInput,
)


def normalize_input_files(value: Sequence[str]) -> tuple[str, ...]:
    """Freeze request file names without applying Workspace path semantics."""
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    files = tuple(value)
    if any(not isinstance(item, str) or not item for item in files):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return files


class MaterializedUserContent(tuple):
    """Validated prompt bytes with their accepted input provenance."""

    view: Mapping[str, JsonValue]

    def __new__(
        cls,
        items: Sequence[UserContent],
        view: Mapping[str, JsonValue],
    ) -> "MaterializedUserContent":
        value = super().__new__(cls, items)
        value.view = ImmutableJsonMapping(view, allow_empty_keys=True)
        return value

    def __getnewargs__(self) -> tuple[tuple[UserContent, ...], Mapping[str, JsonValue]]:
        # The upstream checkpoint capability deep-copies native messages.
        # Preserve accepted provenance as well as the tuple payload.
        return tuple(self), self.view

    def append_text(self, text: str) -> "MaterializedUserContent":
        prompt = self.view.get("prompt")
        if isinstance(prompt, str):
            items: list[JsonValue] = [{"kind": "text", "text": prompt}]
        elif isinstance(prompt, Mapping) and isinstance(prompt.get("items"), list):
            items = list(prompt["items"])
        else:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return type(self)(
            (*self, text),
            {
                **self.view,
                "prompt": {
                    "kind": "items",
                    "items": [*items, {"kind": "text", "text": text}],
                },
            },
        )


def append_user_input_text(value: CanonicalUserInput, text: str) -> CanonicalUserInput:
    if isinstance(value, MaterializedUserContent):
        return value.append_text(text)
    return value + text if isinstance(value, str) else (*value, text)


def validate_user_input(value: UserPromptInput) -> CanonicalUserInput:
    if isinstance(value, str):
        validate_user_prompt(value)
        return value
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    content = tuple(value)
    if not content:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    validate_user_content(content)
    return value if isinstance(value, MaterializedUserContent) else content


def validate_user_content(
    content: Sequence[UserContent | WorkspaceFileInput],
) -> None:
    for item in content:
        if not isinstance(item, _USER_CONTENT_TYPES):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if isinstance(item, BinaryContent) and (
            not isinstance(item.data, bytes)
            or not isinstance(item.media_type, str)
            or not item.media_type
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if isinstance(
            item,
            (
                TextContent,
                ImageUrl,
                AudioUrl,
                DocumentUrl,
                VideoUrl,
                BinaryContent,
                UploadedFile,
            ),
        ):
            metadata = (
                item.metadata
                if isinstance(item, TextContent)
                else item.vendor_metadata
            )
            try:
                normalize_json_value(metadata)
            except (TypeError, ValueError) as error:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error


__all__ = [
    "capture_plain_text_input",
    "plain_text_input_content",
    "CanonicalUserInput",
    "MaterializedUserContent",
    "append_user_input_text",
    "UserPromptInput",
    "validate_user_content",
    "validate_user_input",
    "normalize_input_files",
]


@dataclass(frozen=True, slots=True, weakref_slot=True)
class StoredUserInput:
    codec: str
    payload: StoredPayload
    view: Mapping[str, JsonValue] | None = None

    def __post_init__(self) -> None:
        if self.codec not in {"text", "user-content-v1", "task-input-v1"}:
            raise ValueError("stored user input codec is invalid")
        if not isinstance(self.payload, StoredPayload):
            raise TypeError("stored user input payload is invalid")
        if self.view is not None:
            if not isinstance(self.view, Mapping) or self.view.get("version") != 1:
                raise ValueError("stored user input view is invalid")
            try:
                view = ImmutableJsonMapping(self.view, allow_empty_keys=True)
            except (TypeError, ValueError) as error:
                raise ValueError("stored user input view is invalid") from error
            object.__setattr__(self, "view", view)

    @property
    def digest(self) -> str:
        return canonical_sha256(
            {
                "codec": self.codec,
                "payload_digest": self.payload.digest,
                "payload_size": self.payload.size,
            }
        )


def capture_plain_text_input(content: str | Sequence[str]) -> StoredUserInput:
    """Capture only text or an ordered text-only prompt with its derived view."""
    if isinstance(content, str):
        return StoredUserInput("text", StoredPayload.inline_text(content))
    if (not isinstance(content, (list, tuple)) or not content
            or any(not isinstance(item, str) for item in content)):
        raise ValueError("plain text input requires a nonempty ordered text sequence")
    items = [{"kind": "text", "text": item} for item in content]
    return StoredUserInput("user-content-v1", StoredPayload.inline_json({"items": items}),
                           {"version": 1, "prompt": {"kind": "items", "items": items},
                            "files": [], "attachments": []})


def plain_text_input_content(value: StoredUserInput) -> str | list[str] | None:
    """Select only a complete canonical capture, never infer attachment provenance."""
    if not isinstance(value, StoredUserInput) or value.payload.kind != "inline":
        return None
    decoded = value.payload.decode()
    if value.codec == "text":
        content = decoded if isinstance(decoded, str) else None
    elif (value.codec == "user-content-v1" and isinstance(decoded, Mapping)
          and set(decoded) == {"items"} and isinstance(decoded["items"], (list, tuple))):
        items = decoded["items"]
        if not items or any(not isinstance(item, Mapping) or set(item) != {"kind", "text"}
                            or item["kind"] != "text" or not isinstance(item["text"], str) for item in items):
            return None
        content = [item["text"] for item in items]
    else:
        return None
    if content is None:
        return None
    expected = capture_plain_text_input(content)
    def wire(item: StoredUserInput) -> bytes:
        return canonical_json_bytes(normalize_json_value(
            {"codec": item.codec, "payload": item.payload.to_json(), "view": None if item.view is None else dict(item.view)}))
    return content if wire(value) == wire(expected) else None
