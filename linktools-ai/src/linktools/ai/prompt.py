#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Protocol-neutral user prompt content."""

import base64
import binascii
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias

if TYPE_CHECKING:
    from pydantic_ai.messages import UserContent

MAX_PROMPT_PARTS = 128
MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_BINARY_PART_BYTES = 10 * 1024 * 1024
MAX_BINARY_PROMPT_BYTES = 25 * 1024 * 1024
MAX_URI_LENGTH = 8 * 1024


class PromptValidationError(ValueError):
    """Raised when a prompt violates the domain limits."""


@dataclass(frozen=True, slots=True)
class TextPromptPart:
    text: str


@dataclass(frozen=True, slots=True)
class ImagePromptPart:
    data: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class AudioPromptPart:
    data: bytes
    mime_type: str


@dataclass(frozen=True, slots=True)
class EmbeddedResourcePromptPart:
    mime_type: "str | None"
    text: "str | None" = None
    data: "bytes | None" = None


@dataclass(frozen=True, slots=True)
class ResourceLinkPromptPart:
    uri: str
    name: str = ""
    mime_type: "str | None" = None


UserPromptPart: TypeAlias = TextPromptPart | ImagePromptPart | AudioPromptPart | EmbeddedResourcePromptPart | ResourceLinkPromptPart


@dataclass(frozen=True, slots=True)
class UserPrompt:
    parts: "tuple[UserPromptPart, ...]"

    def __post_init__(self) -> None:
        validate_user_prompt(self)

    @classmethod
    def from_text(cls, text: str) -> "UserPrompt":
        return cls((TextPromptPart(text),))

    def text_fallback(self) -> str:
        return "".join(
            part.text
            for part in self.parts
            if isinstance(part, (TextPromptPart, EmbeddedResourcePromptPart))
            and part.text is not None
        )

    def to_json(self) -> dict:
        result = []
        for part in self.parts:
            if isinstance(part, TextPromptPart):
                result.append({"type": "text", "text": part.text})
            elif isinstance(part, ImagePromptPart):
                result.append({"type": "image", "mime_type": part.mime_type, "data": _encode(part.data)})
            elif isinstance(part, AudioPromptPart):
                result.append({"type": "audio", "mime_type": part.mime_type, "data": _encode(part.data)})
            elif isinstance(part, EmbeddedResourcePromptPart):
                value = {"type": "resource", "mime_type": part.mime_type}
                if part.text is not None:
                    value["text"] = part.text
                if part.data is not None:
                    value["data"] = _encode(part.data)
                result.append(value)
            else:
                result.append({"type": "resource_link", "uri": part.uri, "name": part.name, "mime_type": part.mime_type})
        return {"type": "user_prompt", "parts": result}

    def model_content(self) -> "list[UserContent]":
        from pydantic_ai.messages import AudioUrl, BinaryContent, DocumentUrl, ImageUrl

        result = []
        for part in self.parts:
            if isinstance(part, TextPromptPart):
                result.append(part.text)
            elif isinstance(part, (ImagePromptPart, AudioPromptPart)):
                result.append(BinaryContent(data=part.data, media_type=part.mime_type))
            elif isinstance(part, EmbeddedResourcePromptPart):
                if part.text is not None:
                    result.append(part.text)
                elif part.data is not None:
                    result.append(BinaryContent(data=part.data, media_type=part.mime_type or "application/octet-stream"))
            elif part.mime_type and part.mime_type.startswith("image/"):
                result.append(ImageUrl(url=part.uri, media_type=part.mime_type, force_download=False))
            elif part.mime_type and part.mime_type.startswith("audio/"):
                result.append(AudioUrl(url=part.uri, media_type=part.mime_type, force_download=False))
            else:
                result.append(DocumentUrl(url=part.uri, media_type=part.mime_type, force_download=False))
        return result


def decode_base64(value: str) -> bytes:
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PromptValidationError("invalid base64 content") from exc


def validate_user_prompt(prompt: UserPrompt) -> None:
    if not 1 <= len(prompt.parts) <= MAX_PROMPT_PARTS:
        raise PromptValidationError("prompt must contain 1 to 128 content blocks")
    text_size = binary_size = 0
    for part in prompt.parts:
        if isinstance(part, TextPromptPart):
            text_size += len(part.text.encode("utf-8"))
        elif isinstance(part, (ImagePromptPart, AudioPromptPart)):
            _validate_media(part.mime_type, part.data)
            binary_size += len(part.data)
        elif isinstance(part, EmbeddedResourcePromptPart):
            if part.text is None and part.data is None:
                raise PromptValidationError("embedded resource must contain text or data")
            if part.text is not None:
                text_size += len(part.text.encode("utf-8"))
            if part.data is not None:
                _validate_media(part.mime_type or "application/octet-stream", part.data)
                binary_size += len(part.data)
        elif isinstance(part, ResourceLinkPromptPart):
            if not part.uri or len(part.uri) > MAX_URI_LENGTH:
                raise PromptValidationError("resource URI is too long or empty")
        else:
            raise PromptValidationError("unsupported prompt part")
    if text_size > MAX_TEXT_BYTES or binary_size > MAX_BINARY_PROMPT_BYTES:
        raise PromptValidationError("prompt content exceeds its size limit")


def _validate_media(mime_type: str, data: bytes) -> None:
    if not mime_type or "/" not in mime_type:
        raise PromptValidationError("invalid media MIME type")
    if len(data) > MAX_BINARY_PART_BYTES:
        raise PromptValidationError("binary content exceeds 10 MiB")


def _encode(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


__all__ = ["AudioPromptPart", "EmbeddedResourcePromptPart", "ImagePromptPart", "PromptValidationError", "ResourceLinkPromptPart", "TextPromptPart", "UserPrompt", "UserPromptPart", "decode_base64", "validate_user_prompt"]
