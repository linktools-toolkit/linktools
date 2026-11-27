#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Store the new model messages of a plain turn without copying its input."""

from collections.abc import Sequence
from dataclasses import replace

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    UserContent,
    repair_messages,
)

from ..model.messages import canonical_structured_output, canonical_structured_output_reference, restore_turn_context, substitute_turn_input
from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..model.ordinary_transcript import decode_ordinary_transcript, encode_ordinary_transcript


def encode_regenerable_turn(
    prompt: str | Sequence[UserContent], history: Sequence[ModelMessage],
    messages: Sequence[ModelMessage], output: JsonValue, *,
    instructions: tuple[str, ...] = (), system_prompt: str = "",
) -> JsonValue:
    """Keep the message envelope, substituting the admitted prompt only on read."""
    return encode_ordinary_transcript(
        _canonical_delta(prompt, history, messages, output),
        instructions=instructions, system_prompt=system_prompt,
    )


def prepare_native_turn_transcript(
    prompt: str | Sequence[UserContent], history: Sequence[ModelMessage],
    messages: Sequence[ModelMessage], output: JsonValue, *,
    instructions: tuple[str, ...], system_prompt: str, structured: bool = False,
    merge_context_requests: bool = False,
) -> tuple[tuple[ModelMessage, ...], tuple[ModelMessage, ...], tuple[tuple[int, int | None], ...]]:
    """Use typed placeholders so immutable native messages are valid model values."""
    delta = _canonical_delta(
        prompt, history, messages, output, structured=structured,
        merge_context_requests=merge_context_requests,
    )
    static_instructions = "\n".join(instructions)
    omit_instructions = len(static_instructions.encode("utf-8")) >= 128
    omit_system = len(system_prompt.encode("utf-8")) >= 128
    refs: list[tuple[int, int | None]] = []
    normalized: list[ModelMessage] = []
    for index, message in enumerate(delta):
        if isinstance(message, ModelRequest):
            if omit_instructions and message.instructions == static_instructions:
                message = replace(message, instructions=None)
                refs.append((index, None))
            if omit_system:
                parts = None
                for position, part in enumerate(message.parts):
                    if isinstance(part, SystemPromptPart) and part.dynamic_ref is None and part.content == system_prompt:
                        if parts is None:
                            parts = list(message.parts)
                        parts[position] = replace(part, content="")
                        refs.append((index, position))
                if parts is not None:
                    message = replace(message, parts=parts)
        normalized.append(message)
    return delta, tuple(normalized), tuple(refs)


def _canonical_delta(
    prompt: str | Sequence[UserContent], history: Sequence[ModelMessage],
    messages: Sequence[ModelMessage], output: JsonValue, *, structured: bool = False,
    merge_context_requests: bool = False,
) -> tuple[ModelMessage, ...]:
    frozen = tuple(history)
    prefix = len(frozen)
    if tuple(messages[:prefix]) != frozen:
        if not merge_context_requests:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable run replaced its context")
        # The SDK joins adjacent provider requests, including the last output
        # receipt of one turn and the stored user prompt of the next. Compare the
        # full repaired envelope, but never accept dropped or synthesized parts.
        repaired = tuple(repair_messages(frozen, repair_last_response=False))
        prefix = len(repaired)
        if (prefix >= len(frozen) or tuple(messages[:prefix]) != repaired
                or _request_merge_contents(frozen) != _request_merge_contents(repaired)):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable run replaced its context")
    suffix = tuple(messages[prefix:])
    if not suffix or reply_from_turn(suffix, structured=structured) != output:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Terminal output cannot be projected from the transcript")
    return (substitute_turn_input(suffix[0], prompt, ""), *suffix[1:])


def _request_merge_contents(messages: Sequence[ModelMessage]) -> tuple[object, ...]:
    return tuple(
        item for message in messages
        for item in (message.parts if isinstance(message, ModelRequest) else (message,))
    )


def decode_regenerable_turn(
    prompt: str | Sequence[UserContent], payload: JsonValue, *,
    instructions: list[str] | None = None, system_prompt: str | None = None,
) -> tuple[ModelMessage, ...]:
    messages = decode_ordinary_transcript(payload, instructions=instructions, system_prompt=system_prompt)
    return restore_turn_context(prompt, messages, interrupted=False)


def reply_from_turn(messages: Sequence[ModelMessage], *, structured: bool = False) -> JsonValue:
    if structured:
        return canonical_structured_output(messages, canonical_structured_output_reference(messages))
    response = messages[-1] if messages else None
    if not isinstance(response, ModelResponse) or response.text is None:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Regenerable turn has no final text")
    return {"text": response.text}
