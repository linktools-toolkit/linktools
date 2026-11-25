#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store the new model messages of a plain turn without copying its input."""

import json
from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    UserPromptPart,
)

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ..model.messages import decode_model_messages, encode_model_messages


def encode_regenerable_turn(
    prompt: str, history: Sequence[ModelMessage], messages: Sequence[ModelMessage], output: JsonValue,
) -> JsonValue:
    """Keep the message envelope, substituting the admitted prompt only on read."""
    prefix = len(history)
    if tuple(messages[:prefix]) != tuple(history):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable run replaced its context")
    suffix = tuple(messages[prefix:])
    if not suffix or reply_from_turn(suffix) != output:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Terminal output cannot be projected from the transcript")
    return json.loads(encode_model_messages((_substitute_input(suffix[0], prompt, ""), *suffix[1:])))


def decode_regenerable_turn(prompt: str, payload: JsonValue) -> tuple[ModelMessage, ...]:
    messages = decode_model_messages(canonical_json_bytes(payload))
    if not messages:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn messages are missing")
    return (_substitute_input(messages[0], "", prompt), *messages[1:])


def reply_from_turn(messages: Sequence[ModelMessage]) -> JsonValue:
    response = messages[-1] if messages else None
    if not isinstance(response, ModelResponse) or response.text is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no final text")
    return {"text": response.text}


def _substitute_input(message: ModelMessage, expected: str, value: str) -> ModelRequest:
    if not isinstance(message, ModelRequest):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no matching admitted input")
    prompts = [part for part in message.parts if isinstance(part, UserPromptPart)]
    if len(prompts) != 1 or prompts[0].content != expected:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no matching admitted input")
    return replace(message, parts=[replace(part, content=value) if part is prompts[0] else part
                                   for part in message.parts])
