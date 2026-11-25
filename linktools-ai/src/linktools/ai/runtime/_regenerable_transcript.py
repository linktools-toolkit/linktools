#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store the new model messages of a plain turn without copying its input."""

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserContent,
    UserPromptPart,
)

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..model import decode_ordinary_transcript, encode_ordinary_transcript


def encode_regenerable_turn(
    prompt: str | Sequence[UserContent], history: Sequence[ModelMessage],
    messages: Sequence[ModelMessage], output: JsonValue, *,
    instructions: tuple[str, ...] = (), system_prompt: str = "",
) -> JsonValue:
    """Keep the message envelope, substituting the admitted prompt only on read."""
    prefix = len(history)
    if tuple(messages[:prefix]) != tuple(history):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable run replaced its context")
    suffix = tuple(messages[prefix:])
    if not suffix or reply_from_turn(suffix) != output:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Terminal output cannot be projected from the transcript")
    return encode_ordinary_transcript(
        (_substitute_input(suffix[0], prompt, ""), *suffix[1:]),
        instructions=instructions, system_prompt=system_prompt,
    )


def decode_regenerable_turn(
    prompt: str | Sequence[UserContent], payload: JsonValue, *,
    instructions: list[str] | None = None, system_prompt: str | None = None,
) -> tuple[ModelMessage, ...]:
    messages = decode_ordinary_transcript(payload, instructions=instructions, system_prompt=system_prompt)
    if not messages:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn messages are missing")
    restored = prompt if isinstance(prompt, str) else list(prompt)
    return (_substitute_input(messages[0], "", restored), *messages[1:])


def reply_from_turn(messages: Sequence[ModelMessage]) -> JsonValue:
    response = messages[-1] if messages else None
    if not isinstance(response, ModelResponse) or response.text is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no final text")
    return {"text": response.text}


def _substitute_input(
    message: ModelMessage, expected: str | Sequence[UserContent], value: str | Sequence[UserContent],
) -> ModelRequest:
    if not isinstance(message, ModelRequest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no matching admitted input")
    prompts = [part for part in message.parts if isinstance(part, UserPromptPart)]
    if len(prompts) != 1:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no matching admitted input")
    [prompt] = prompts
    content = prompt.content
    same_input = (
        content == expected
        if isinstance(content, str) or isinstance(expected, str)
        else tuple(content) == tuple(expected)
    )
    if not same_input:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no matching admitted input")
    return replace(message, parts=[
        replace(part, content=value) if part is prompt else part for part in message.parts
    ])
