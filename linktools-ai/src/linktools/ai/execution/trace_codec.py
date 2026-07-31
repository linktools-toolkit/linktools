#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""One canonical codec for model messages and semantic trace payloads.

Resume-message (de)serialization delegates to pydantic-ai's
``ModelMessagesTypeAdapter`` so format upgrades are transparent and no
part type is silently dropped. Trace-step payloads use a hand-rolled
shape for human-readable inspection but do not affect resume fidelity.
"""


from datetime import datetime, timezone
from typing import Any
from pydantic_ai.messages import ModelRequest, ModelResponse, SystemPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart, ModelMessagesTypeAdapter
from ..json import normalize_json

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.messages import ModelMessage
    from ..json import JsonValue

def encode_model_messages(messages: "tuple[ModelMessage, ...] | list[ModelMessage]") -> "tuple[JsonValue, ...]":
    return tuple(ModelMessagesTypeAdapter.dump_python(list(messages), mode="json"))


def decode_model_messages(values: "tuple[JsonValue, ...] | list[JsonValue]") -> "tuple[ModelMessage, ...]":
    return tuple(ModelMessagesTypeAdapter.validate_python(list(values)))


def _part(value: object) -> "JsonValue":
    if isinstance(value, SystemPromptPart):
        return {"type": "system_prompt", "content": value.content}
    if isinstance(value, UserPromptPart):
        return {"type": "user_prompt", "content": normalize_json(value.content)}
    if isinstance(value, TextPart):
        return {"type": "text", "content": value.content}
    if isinstance(value, ToolCallPart):
        return {"type": "tool_call", "call_id": value.tool_call_id, "tool_name": value.tool_name, "arguments": normalize_json(value.args)}
    if isinstance(value, ToolReturnPart):
        return {"type": "tool_result", "call_id": value.tool_call_id, "tool_name": value.tool_name, "result": normalize_json(value.content), "status": value.outcome}
    return {"type": "unsupported", "source_type": type(value).__name__, "safe_summary": type(value).__name__}


def _message(value: "ModelMessage") -> "JsonValue":
    if isinstance(value, ModelRequest):
        return {"kind": "request", "parts": [_part(item) for item in value.parts], "timestamp": (value.timestamp or datetime.now(timezone.utc)).isoformat()}
    if isinstance(value, ModelResponse):
        usage = value.usage
        return {"kind": "response", "parts": [_part(item) for item in value.parts], "model_name": value.model_name, "finish_reason": value.finish_reason, "provider_response_id": value.provider_response_id, "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.input_tokens + usage.output_tokens}}
    return {"kind": "unsupported", "source_type": type(value).__name__, "safe_summary": type(value).__name__}


def encode_model_request(messages: "tuple[ModelMessage, ...]", settings: object, tools: object) -> "JsonValue":
    return {"messages": list(encode_model_messages(messages)), "settings": normalize_json(settings), "tools": normalize_json(tools)}


def encode_model_response(response: ModelResponse) -> "JsonValue":
    return _message(response)


__all__ = ["decode_model_messages", "encode_model_messages", "encode_model_request", "encode_model_response"]
