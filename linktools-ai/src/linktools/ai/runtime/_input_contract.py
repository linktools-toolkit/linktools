#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pure validation for user input accepted by runtime request contracts."""

from collections.abc import Sequence
from typing import TypeAlias

from pydantic_ai.messages import BinaryContent, UploadedFile, UserContent

from ..core import validate_user_prompt
from ..errors import AIError, ErrorCode

UserPromptInput: TypeAlias = str | Sequence[UserContent]
CanonicalUserInput: TypeAlias = str | tuple[UserContent, ...]


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
    _validate_content(content)
    return content


def _validate_content(content: Sequence[UserContent]) -> None:
    for item in content:
        if isinstance(item, UploadedFile):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "field": "user_prompt",
                    "reason": "uploaded_file_not_durable",
                },
            )
        if isinstance(item, BinaryContent) and (
            not isinstance(item.data, bytes)
            or not isinstance(item.media_type, str)
            or not item.media_type
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


__all__ = ["CanonicalUserInput", "UserPromptInput", "validate_user_input"]
