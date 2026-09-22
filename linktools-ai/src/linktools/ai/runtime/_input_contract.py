#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure validation for user input accepted by runtime request contracts."""

from collections.abc import Mapping, Sequence
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

from ..core import JsonValue, WorkspaceFileInput, normalize_json_value, validate_user_prompt
from ..errors import AIError, ErrorCode

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


class MaterializedUserContent(tuple):
    """Validated prompt bytes with their accepted input provenance."""

    view: Mapping[str, JsonValue]

    def __new__(
        cls,
        items: Sequence[UserContent],
        view: Mapping[str, JsonValue],
    ) -> "MaterializedUserContent":
        value = super().__new__(cls, items)
        value.view = dict(view)
        return value

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
    "CanonicalUserInput",
    "MaterializedUserContent",
    "append_user_input_text",
    "UserPromptInput",
    "validate_user_content",
    "validate_user_input",
]
