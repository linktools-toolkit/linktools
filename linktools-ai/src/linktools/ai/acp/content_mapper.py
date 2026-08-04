#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""ACP content blocks to the protocol-neutral prompt domain."""

from typing import Any

from ..prompt import (
    AudioPromptPart,
    EmbeddedResourcePromptPart,
    ImagePromptPart,
    ResourceLinkPromptPart,
    TextPromptPart,
    UserPrompt,
    decode_base64,
    PromptValidationError,
)
from .errors import request_error


class AcpContentMapper:
    def __init__(self, *, image: bool = False, audio: bool = False, embedded: bool = False) -> None:
        self.image = image
        self.audio = audio
        self.embedded = embedded

    def map(self, blocks: "list[Any]") -> UserPrompt:
        if not blocks:
            raise request_error("empty_prompt")
        parts = []
        for block in blocks:
            kind = getattr(block, "type", None)
            if kind == "text":
                parts.append(TextPromptPart(block.text))
            elif kind == "image":
                if not self.image:
                    raise request_error("unsupported_content_type")
                parts.append(ImagePromptPart(_decode(block.data), block.mime_type))
            elif kind == "audio":
                if not self.audio:
                    raise request_error("unsupported_content_type")
                parts.append(AudioPromptPart(_decode(block.data), block.mime_type))
            elif kind == "resource_link":
                parts.append(
                    ResourceLinkPromptPart(
                        uri=block.uri,
                        name=block.name,
                        mime_type=block.mime_type,
                    )
                )
            elif kind == "resource":
                if not self.embedded:
                    raise request_error("unsupported_content_type")
                resource = block.resource
                if getattr(resource, "text", None) is not None:
                    parts.append(
                        EmbeddedResourcePromptPart(
                            mime_type=resource.mime_type,
                            text=resource.text,
                        )
                    )
                else:
                    parts.append(
                        EmbeddedResourcePromptPart(
                            mime_type=resource.mime_type,
                            data=_decode(resource.blob),
                        )
                    )
            else:
                raise request_error("unsupported_content_type")
        try:
            return UserPrompt(tuple(parts))
        except ValueError as exc:
            raise request_error("invalid_prompt") from exc


__all__ = ["AcpContentMapper"]


def _decode(value: str) -> bytes:
    try:
        return decode_base64(value)
    except PromptValidationError as exc:
        raise request_error("invalid_prompt") from exc
