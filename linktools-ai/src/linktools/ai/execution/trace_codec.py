"""One canonical codec for model messages and semantic trace payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RequestUsage,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..storage.json import JsonValue, normalize_json


def _part(value: object) -> JsonValue:
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


def _message(value: ModelMessage) -> JsonValue:
    if isinstance(value, ModelRequest):
        return {"kind": "request", "parts": [_part(item) for item in value.parts], "timestamp": (value.timestamp or datetime.now(timezone.utc)).isoformat()}
    if isinstance(value, ModelResponse):
        usage = value.usage
        return {"kind": "response", "parts": [_part(item) for item in value.parts], "model_name": value.model_name, "finish_reason": value.finish_reason, "provider_response_id": value.provider_response_id, "usage": {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens, "total_tokens": usage.input_tokens + usage.output_tokens}}
    return {"kind": "unsupported", "source_type": type(value).__name__, "safe_summary": type(value).__name__}


def encode_model_messages(messages: tuple[ModelMessage, ...] | list[ModelMessage]) -> tuple[JsonValue, ...]:
    return tuple(_message(message) for message in messages)


def _decode_part(value: dict[str, Any]) -> object:
    kind = value.get("type")
    if kind == "system_prompt":
        return SystemPromptPart(value["content"])
    if kind == "user_prompt":
        return UserPromptPart(value["content"])
    if kind == "text":
        return TextPart(value["content"])
    if kind == "tool_call":
        return ToolCallPart(tool_name=value["tool_name"], args=value.get("arguments"), tool_call_id=value["call_id"])
    if kind == "tool_result":
        return ToolReturnPart(tool_name=value["tool_name"], content=value.get("result"), tool_call_id=value["call_id"], outcome=value.get("status", "success"))
    raise ValueError(f"unsupported canonical part: {kind!r}")


def decode_model_messages(values: tuple[JsonValue, ...] | list[JsonValue]) -> tuple[ModelMessage, ...]:
    result: list[ModelMessage] = []
    for value in values:
        item = dict(value)
        if item["kind"] == "request":
            result.append(ModelRequest([_decode_part(part) for part in item["parts"]]))
        elif item["kind"] == "response":
            usage = item.get("usage", {})
            result.append(ModelResponse([_decode_part(part) for part in item["parts"]], model_name=item.get("model_name"), finish_reason=item.get("finish_reason"), provider_response_id=item.get("provider_response_id"), usage=RequestUsage(input_tokens=usage.get("input_tokens", 0), output_tokens=usage.get("output_tokens", 0))))
        else:
            raise ValueError(f"unsupported canonical message: {item.get('kind')!r}")
    return tuple(result)


def encode_model_request(messages: tuple[ModelMessage, ...], settings: object, tools: object) -> JsonValue:
    return {"messages": list(encode_model_messages(messages)), "settings": normalize_json(settings), "tools": normalize_json(tools)}


def encode_model_response(response: ModelResponse) -> JsonValue:
    return _message(response)


__all__ = ["decode_model_messages", "encode_model_messages", "encode_model_request", "encode_model_response"]
