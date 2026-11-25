#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ordinary turn transcript references to exact, same-turn frozen Agent instructions."""

from __future__ import annotations

import json
from collections.abc import Sequence

from pydantic_ai.messages import ModelMessage

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from .messages import decode_model_messages, encode_model_messages

_REF = {"source": "agent_spec"}
_INSTRUCTIONS = "instructions_from_binding"
_SYSTEM_PROMPT = "system_prompt_from_binding"


def encode_ordinary_transcript(
    messages: Sequence[ModelMessage], *, instructions: tuple[str, ...], system_prompt: str,
) -> JsonValue:
    """Replace only exact static content; restore from the same admitted binding."""
    wire = json.loads(encode_model_messages(messages))
    if not instructions and not system_prompt:
        return wire
    static_instructions = "\n".join(instructions)
    # Tiny strings cost less inline than the reference and its envelope.
    reference_instructions = len(static_instructions.encode("utf-8")) >= 128
    reference_system_prompt = len(system_prompt.encode("utf-8")) >= 128
    flags = {}
    for message in wire:
        if message.get("kind") != "request":
            continue
        if reference_instructions and message.get("instructions") == static_instructions:
            message["instructions"] = _REF
            flags[_INSTRUCTIONS] = True
        if reference_system_prompt:
            for part in message["parts"]:
                if (part.get("part_kind") == "system-prompt" and part.get("dynamic_ref") is None
                        and part.get("content") == system_prompt):
                    part["content"] = _REF
                    flags[_SYSTEM_PROMPT] = True
    return {"messages": wire, **flags} if flags else wire


def _expanded_wire(
    payload: JsonValue, *, instructions: list[str] | None = None,
    system_prompt: str | None = None, display: bool = False,
) -> bytes:
    if isinstance(payload, list):
        return canonical_json_bytes(payload)

    allowed = {_INSTRUCTIONS, _SYSTEM_PROMPT}
    if not isinstance(payload, dict) or not isinstance(payload.get("messages"), list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    declared = payload.keys() - {"messages"}
    if not declared or not declared <= allowed or any(payload[flag] is not True for flag in declared):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    static_instructions = None
    if _INSTRUCTIONS in declared and not display:
        if not isinstance(instructions, list) or not all(isinstance(line, str) for line in instructions):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Ordinary turn is missing frozen Agent instructions")
        static_instructions = "\n".join(instructions)
        if not static_instructions:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Ordinary turn is missing frozen Agent instructions")
    if _SYSTEM_PROMPT in declared and not display and (not isinstance(system_prompt, str) or not system_prompt):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Ordinary turn is missing its frozen system prompt")

    seen = set()
    expanded = []
    for message in payload["messages"]:
        if not isinstance(message, dict) or message.get("kind") != "request":
            expanded.append(message)
            continue
        request = dict(message)
        if request.get("instructions") == _REF:
            seen.add(_INSTRUCTIONS)
            request["instructions"] = static_instructions
        parts = request.get("parts")
        if not isinstance(parts, list):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if _SYSTEM_PROMPT in declared:
            restored = []
            for part in parts:
                if (isinstance(part, dict) and part.get("part_kind") == "system-prompt"
                        and part.get("dynamic_ref") is None and part.get("content") == _REF):
                    seen.add(_SYSTEM_PROMPT)
                    restored.append({**part, "content": "" if display else system_prompt})
                else:
                    restored.append(part)
            request["parts"] = restored
        expanded.append(request)
    if seen != declared:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR, "Ordinary turn has inconsistent Agent instruction references")
    return canonical_json_bytes(expanded)


def decode_ordinary_transcript(
    payload: JsonValue, *, instructions: list[str] | None = None,
    system_prompt: str | None = None, display: bool = False,
) -> tuple[ModelMessage, ...]:
    """Decode exact frozen messages, or omit only static source text for display."""
    return decode_model_messages(_expanded_wire(
        payload, instructions=instructions, system_prompt=system_prompt, display=display,
    ))


def ordinary_transcript_size(
    payload: JsonValue, *, instructions: list[str] | None = None, system_prompt: str | None = None,
) -> int:
    """Charge original expanded canonical bytes against model context limits."""
    if not isinstance(payload, dict) or not payload.keys() & {_INSTRUCTIONS, _SYSTEM_PROMPT}:
        return len(canonical_json_bytes(payload))
    return len(_expanded_wire(payload, instructions=instructions, system_prompt=system_prompt))


__all__ = ["decode_ordinary_transcript", "encode_ordinary_transcript", "ordinary_transcript_size"]
