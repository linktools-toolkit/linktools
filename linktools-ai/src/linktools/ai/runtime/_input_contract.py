#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure validation for user input accepted by runtime request contracts."""

from collections.abc import Sequence
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

from ..core import normalize_json_value, validate_user_prompt
from ..errors import AIError, ErrorCode

UserPromptInput: TypeAlias = str | Sequence[UserContent]
CanonicalUserInput: TypeAlias = str | tuple[UserContent, ...]
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
)


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
    return content


def validate_user_content(content: Sequence[UserContent]) -> None:
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
    "UserPromptInput",
    "validate_user_content",
    "validate_user_input",
]
