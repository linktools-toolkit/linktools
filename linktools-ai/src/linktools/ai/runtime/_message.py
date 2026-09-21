#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical persistence conversion and active model-context projection."""

import binascii
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import datetime
from decimal import Decimal, InvalidOperation
from typing import cast

from pydantic_ai import RequestUsage
from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    FilePart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SpeechPart,
    SystemPromptPart,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from ._input import _decode_user_content_item, _encode_user_content_item
from ._tool_return_codec import decode_tool_return_content, encode_tool_return_content

_CONSUMED_BINARY_MARKER = "[binary content already consumed]"
_MESSAGE_VERSION = 1
_TOOL_KINDS = frozenset({"tool-search", "capability-load"})
_REQUEST_STATES = frozenset({"complete", "interrupted"})
_RESPONSE_STATES = frozenset({"complete", "incomplete", "suspended", "interrupted"})
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_call", "error"})
_TOOL_OUTCOMES = frozenset({"success", "failed", "denied", "interrupted"})


def _portable_json(value: object) -> JsonValue:
    if value is None or isinstance(value, (bool, int, str)):
        return cast(JsonValue, value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("durable message JSON requires finite floats")
        return value
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("durable message mappings require string keys")
        return {
            key: _portable_json(value[key])
            for key in sorted(value)
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_portable_json(item) for item in value]
    raise TypeError("durable message value is not JSON portable")


def project_transient_binary_content(
    messages: Sequence[ModelMessage],
) -> tuple[ModelMessage, ...]:
    """Remove binary bodies already consumed by a complete model response."""
    values = tuple(messages)
    seen_complete_response = False
    changed = False
    projected_reversed: list[ModelMessage] = []

    for message in reversed(values):
        if isinstance(message, ModelResponse):
            if message.state == "complete":
                seen_complete_response = True
            projected_reversed.append(message)
            continue
        if not seen_complete_response or not isinstance(message, ModelRequest):
            projected_reversed.append(message)
            continue

        message_changed = False
        parts = []
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                parts.append(part)
                continue
            content = []
            part_changed = False
            for item in part.content:
                if isinstance(item, BinaryContent):
                    content.append(_CONSUMED_BINARY_MARKER)
                    part_changed = True
                else:
                    content.append(item)
            if part_changed:
                parts.append(replace(part, content=content))
                message_changed = True
            else:
                parts.append(part)
        if message_changed:
            projected_reversed.append(replace(message, parts=parts))
            changed = True
        else:
            projected_reversed.append(message)

    if not changed:
        return values
    projected_reversed.reverse()
    return tuple(projected_reversed)


def binary_content_usage(messages: Sequence[ModelMessage]) -> tuple[int, int]:
    """Return BinaryContent part count and bytes in user-prompt content."""
    count = 0
    total_bytes = 0
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            for item in part.content:
                if isinstance(item, BinaryContent):
                    count += 1
                    total_bytes += len(item.data)
    return count, total_bytes


def encode_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    """Encode model history using the LinkTools-owned v1 durable contract."""
    return canonical_json_bytes([_encode_message(message) for message in messages])


def freeze_model_messages(
    messages: Sequence[ModelMessage],
) -> tuple[ModelMessage, ...]:
    """Freeze model messages through the canonical persistence codec."""
    return decode_model_messages(encode_model_messages(messages))


def decode_model_messages(raw: bytes) -> tuple[ModelMessage, ...]:
    try:
        value = json.loads(raw.decode("utf-8"))
        if canonical_json_bytes(cast(JsonValue, value)) != raw:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    except AIError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(value, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        return tuple(_decode_message(item) for item in value)
    except AIError:
        raise
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _encode_message(message: ModelMessage) -> dict[str, JsonValue]:
    if isinstance(message, ModelRequest):
        return {
            "version": _MESSAGE_VERSION,
            "kind": "request",
            "parts": [_encode_request_part(part) for part in message.parts],
            "timestamp": _encode_datetime(message.timestamp),
            "instructions": message.instructions,
            "run_id": message.run_id,
            "conversation_id": message.conversation_id,
            "metadata": _encode_optional_mapping(message.metadata),
            "state": message.state,
        }
    if isinstance(message, ModelResponse):
        return {
            "version": _MESSAGE_VERSION,
            "kind": "response",
            "parts": [_encode_response_part(part) for part in message.parts],
            "usage": _encode_usage(message.usage),
            "model_name": message.model_name,
            "timestamp": _encode_datetime(message.timestamp),
            "provider_name": message.provider_name,
            "provider_url": message.provider_url,
            "provider_details": _encode_optional_mapping(message.provider_details),
            "provider_response_id": message.provider_response_id,
            "finish_reason": message.finish_reason,
            "run_id": message.run_id,
            "conversation_id": message.conversation_id,
            "metadata": _encode_optional_mapping(message.metadata),
            "state": message.state,
        }
    raise TypeError(f"unsupported model message type: {type(message).__name__}")


def _decode_message(value: object) -> ModelMessage:
    if not isinstance(value, Mapping):
        raise ValueError("model message must be an object")
    version = value.get("version")
    if version != _MESSAGE_VERSION:
        if isinstance(version, int) and not isinstance(version, bool):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        raise ValueError("model message version is invalid")
    kind = value.get("kind")
    if kind == "request":
        _require_keys(
            value,
            {
                "version",
                "kind",
                "parts",
                "timestamp",
                "instructions",
                "run_id",
                "conversation_id",
                "metadata",
                "state",
            },
        )
        parts = value["parts"]
        if not isinstance(parts, list):
            raise ValueError("model request parts are invalid")
        state = _required_string(value["state"])
        if state not in _REQUEST_STATES:
            raise ValueError("model request state is invalid")
        return ModelRequest(
            parts=[_decode_request_part(item) for item in parts],
            timestamp=_decode_optional_datetime(value["timestamp"]),
            instructions=_optional_string(value["instructions"]),
            run_id=_optional_string(value["run_id"]),
            conversation_id=_optional_string(value["conversation_id"]),
            metadata=_decode_optional_mapping(value["metadata"]),
            state=cast(object, state),
        )
    if kind == "response":
        _require_keys(
            value,
            {
                "version",
                "kind",
                "parts",
                "usage",
                "model_name",
                "timestamp",
                "provider_name",
                "provider_url",
                "provider_details",
                "provider_response_id",
                "finish_reason",
                "run_id",
                "conversation_id",
                "metadata",
                "state",
            },
        )
        parts = value["parts"]
        if not isinstance(parts, list):
            raise ValueError("model response parts are invalid")
        state = _required_string(value["state"])
        if state not in _RESPONSE_STATES:
            raise ValueError("model response state is invalid")
        finish_reason = value["finish_reason"]
        if finish_reason is not None and finish_reason not in _FINISH_REASONS:
            raise ValueError("model response finish reason is invalid")
        timestamp = _decode_optional_datetime(value["timestamp"])
        if timestamp is None:
            raise ValueError("model response timestamp is required")
        return ModelResponse(
            parts=[_decode_response_part(item) for item in parts],
            usage=_decode_usage(value["usage"]),
            model_name=_optional_string(value["model_name"]),
            timestamp=timestamp,
            provider_name=_optional_string(value["provider_name"]),
            provider_url=_optional_string(value["provider_url"]),
            provider_details=_decode_optional_mapping(value["provider_details"]),
            provider_response_id=_optional_string(value["provider_response_id"]),
            finish_reason=cast(object, finish_reason),
            run_id=_optional_string(value["run_id"]),
            conversation_id=_optional_string(value["conversation_id"]),
            metadata=_decode_optional_mapping(value["metadata"]),
            state=cast(object, state),
        )
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _encode_request_part(part: object) -> dict[str, JsonValue]:
    if isinstance(part, SystemPromptPart):
        return {
            "part_kind": "system-prompt",
            "content": part.content,
            "timestamp": _encode_datetime(part.timestamp),
            "dynamic_ref": part.dynamic_ref,
        }
    if isinstance(part, UserPromptPart):
        content: dict[str, JsonValue]
        if isinstance(part.content, str):
            content = {"type": "text", "value": part.content}
        else:
            content = {
                "type": "items",
                "items": [_encode_user_content_item(item) for item in part.content],
            }
        return {
            "part_kind": "user-prompt",
            "content": content,
            "timestamp": _encode_datetime(part.timestamp),
        }
    if isinstance(part, ToolReturnPart):
        return _encode_tool_return_part(part, native=False)
    if isinstance(part, RetryPromptPart):
        return {
            "part_kind": "retry-prompt",
            "content": _portable_json(part.content),
            "tool_name": part.tool_name,
            "tool_call_id": part.tool_call_id,
            "timestamp": _encode_datetime(part.timestamp),
        }
    if isinstance(part, ToolAvailabilityDeltaPart):
        return {
            "part_kind": "tool-availability-delta",
            "tools_added": list(part.tools_added),
            "tool_call_id": part.tool_call_id,
        }
    if isinstance(part, SpeechPart):
        return _encode_speech_part(part)
    raise TypeError(f"unsupported model request part: {type(part).__name__}")


def _decode_request_part(value: object) -> object:
    part = _part_mapping(value)
    kind = part.get("part_kind")
    if kind == "system-prompt":
        _require_keys(part, {"part_kind", "content", "timestamp", "dynamic_ref"})
        return SystemPromptPart(
            _required_string(part["content"]),
            timestamp=_required_datetime(part["timestamp"]),
            dynamic_ref=_optional_string(part["dynamic_ref"]),
        )
    if kind == "user-prompt":
        _require_keys(part, {"part_kind", "content", "timestamp"})
        raw_content = part["content"]
        if not isinstance(raw_content, Mapping):
            raise ValueError("user prompt content is invalid")
        content_type = raw_content.get("type")
        if content_type == "text":
            _require_keys(raw_content, {"type", "value"})
            content: object = _required_string(raw_content["value"])
        elif content_type == "items":
            _require_keys(raw_content, {"type", "items"})
            items = raw_content["items"]
            if not isinstance(items, list):
                raise ValueError("user prompt items are invalid")
            content = [_decode_user_content_item(item) for item in items]
        else:
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return UserPromptPart(
            cast(object, content),
            timestamp=_required_datetime(part["timestamp"]),
        )
    if kind == "tool-return":
        return _decode_tool_return_part(part, native=False)
    if kind == "retry-prompt":
        _require_keys(
            part,
            {"part_kind", "content", "tool_name", "tool_call_id", "timestamp"},
        )
        content = part["content"]
        if not isinstance(content, (str, list)):
            raise ValueError("retry content is invalid")
        return RetryPromptPart(
            cast(object, content),
            tool_name=_optional_string(part["tool_name"]),
            tool_call_id=_required_string(part["tool_call_id"]),
            timestamp=_required_datetime(part["timestamp"]),
        )
    if kind == "tool-availability-delta":
        _require_keys(part, {"part_kind", "tools_added", "tool_call_id"})
        tools = part["tools_added"]
        if not isinstance(tools, list) or any(not isinstance(item, str) for item in tools):
            raise ValueError("tool availability is invalid")
        return ToolAvailabilityDeltaPart(
            tools_added=cast(list[str], tools),
            tool_call_id=_optional_string(part["tool_call_id"]),
        )
    if kind == "speech":
        speech = _decode_speech_part(part)
        if speech.speaker != "user":
            raise ValueError("request speech speaker is invalid")
        return speech
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _encode_response_part(part: object) -> dict[str, JsonValue]:
    if isinstance(part, TextPart):
        return {
            "part_kind": "text",
            "content": part.content,
            "id": part.id,
            "provider_name": part.provider_name,
            "provider_details": _encode_optional_mapping(part.provider_details),
        }
    if isinstance(part, NativeToolCallPart):
        return _encode_tool_call_part(part, native=True)
    if isinstance(part, ToolCallPart):
        return _encode_tool_call_part(part, native=False)
    if isinstance(part, NativeToolReturnPart):
        return _encode_tool_return_part(part, native=True)
    if isinstance(part, ThinkingPart):
        return {
            "part_kind": "thinking",
            "content": part.content,
            "id": part.id,
            "signature": part.signature,
            "provider_name": part.provider_name,
            "provider_details": _encode_optional_mapping(part.provider_details),
        }
    if isinstance(part, CompactionPart):
        return {
            "part_kind": "compaction",
            "content": part.content,
            "id": part.id,
            "provider_name": part.provider_name,
            "provider_details": _encode_optional_mapping(part.provider_details),
        }
    if isinstance(part, FilePart):
        return {
            "part_kind": "file",
            "content": _encode_user_content_item(part.content),
            "id": part.id,
            "provider_name": part.provider_name,
            "provider_details": _encode_optional_mapping(part.provider_details),
        }
    if isinstance(part, SpeechPart):
        return _encode_speech_part(part)
    raise TypeError(f"unsupported model response part: {type(part).__name__}")


def _decode_response_part(value: object) -> object:
    part = _part_mapping(value)
    kind = part.get("part_kind")
    if kind == "text":
        _require_keys(
            part,
            {"part_kind", "content", "id", "provider_name", "provider_details"},
        )
        return TextPart(
            _required_string(part["content"]),
            id=_optional_string(part["id"]),
            provider_name=_optional_string(part["provider_name"]),
            provider_details=_decode_optional_mapping(part["provider_details"]),
        )
    if kind == "tool-call":
        return _decode_tool_call_part(part, native=False)
    if kind == "builtin-tool-call":
        return _decode_tool_call_part(part, native=True)
    if kind == "builtin-tool-return":
        return _decode_tool_return_part(part, native=True)
    if kind == "thinking":
        _require_keys(
            part,
            {
                "part_kind",
                "content",
                "id",
                "signature",
                "provider_name",
                "provider_details",
            },
        )
        return ThinkingPart(
            _required_string(part["content"]),
            id=_optional_string(part["id"]),
            signature=_optional_string(part["signature"]),
            provider_name=_optional_string(part["provider_name"]),
            provider_details=_decode_optional_mapping(part["provider_details"]),
        )
    if kind == "compaction":
        _require_keys(
            part,
            {"part_kind", "content", "id", "provider_name", "provider_details"},
        )
        content = part["content"]
        if content is not None and not isinstance(content, str):
            raise ValueError("compaction content is invalid")
        return CompactionPart(
            cast(str | None, content),
            id=_optional_string(part["id"]),
            provider_name=_optional_string(part["provider_name"]),
            provider_details=_decode_optional_mapping(part["provider_details"]),
        )
    if kind == "file":
        _require_keys(
            part,
            {"part_kind", "content", "id", "provider_name", "provider_details"},
        )
        content = _decode_user_content_item(part["content"])
        if not isinstance(content, BinaryContent):
            raise ValueError("file part content is invalid")
        return FilePart(
            content,
            id=_optional_string(part["id"]),
            provider_name=_optional_string(part["provider_name"]),
            provider_details=_decode_optional_mapping(part["provider_details"]),
        )
    if kind == "speech":
        speech = _decode_speech_part(part)
        if speech.speaker != "assistant":
            raise ValueError("response speech speaker is invalid")
        return speech
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _encode_tool_call_part(
    part: ToolCallPart | NativeToolCallPart,
    *,
    native: bool,
) -> dict[str, JsonValue]:
    tool_kind = _encode_tool_kind(part.tool_kind)
    result: dict[str, JsonValue] = {
        "part_kind": "builtin-tool-call" if native else "tool-call",
        "tool_name": part.tool_name,
        "args": _portable_json(part.args),
        "tool_call_id": part.tool_call_id,
        "tool_kind": tool_kind,
        "id": part.id,
        "provider_name": part.provider_name,
        "provider_details": _encode_optional_mapping(part.provider_details),
    }
    return result


def _decode_tool_call_part(
    part: Mapping[str, object],
    *,
    native: bool,
) -> object:
    _require_keys(
        part,
        {
            "part_kind",
            "tool_name",
            "args",
            "tool_call_id",
            "tool_kind",
            "id",
            "provider_name",
            "provider_details",
        },
    )
    tool_kind = _decode_tool_kind(part["tool_kind"])
    args = part["args"]
    if args is not None and not isinstance(args, (str, Mapping)):
        raise ValueError("tool call args are invalid")
    kwargs = {
        "tool_name": _required_string(part["tool_name"]),
        "args": cast(object, dict(args) if isinstance(args, Mapping) else args),
        "tool_call_id": _required_string(part["tool_call_id"]),
        "tool_kind": cast(object, tool_kind),
        "id": _optional_string(part["id"]),
        "provider_name": _optional_string(part["provider_name"]),
        "provider_details": _decode_optional_mapping(part["provider_details"]),
    }
    if native:
        value = NativeToolCallPart(**kwargs)
        return NativeToolCallPart.narrow_type(value)
    value = ToolCallPart(**kwargs)
    return ToolCallPart.narrow_type(value)


def _encode_tool_return_part(
    part: ToolReturnPart | NativeToolReturnPart,
    *,
    native: bool,
) -> dict[str, JsonValue]:
    result: dict[str, JsonValue] = {
        "part_kind": "builtin-tool-return" if native else "tool-return",
        "tool_name": part.tool_name,
        "content": encode_tool_return_content(part.content),
        "tool_call_id": part.tool_call_id,
        "tool_kind": _encode_tool_kind(part.tool_kind),
        "metadata": _portable_json(part.metadata),
        "timestamp": _encode_datetime(part.timestamp),
        "outcome": part.outcome,
    }
    if native:
        native_part = cast(NativeToolReturnPart, part)
        result["provider_name"] = native_part.provider_name
        result["provider_details"] = _encode_optional_mapping(
            native_part.provider_details
        )
    return result


def _decode_tool_return_part(
    part: Mapping[str, object],
    *,
    native: bool,
) -> object:
    expected = {
        "part_kind",
        "tool_name",
        "content",
        "tool_call_id",
        "tool_kind",
        "metadata",
        "timestamp",
        "outcome",
    }
    if native:
        expected |= {"provider_name", "provider_details"}
    _require_keys(part, expected)
    outcome = _required_string(part["outcome"])
    if outcome not in _TOOL_OUTCOMES:
        raise ValueError("tool outcome is invalid")
    kwargs = {
        "tool_name": _required_string(part["tool_name"]),
        "content": decode_tool_return_content(cast(JsonValue, part["content"])),
        "tool_call_id": _required_string(part["tool_call_id"]),
        "tool_kind": cast(object, _decode_tool_kind(part["tool_kind"])),
        "metadata": _portable_json(part["metadata"]),
        "timestamp": _required_datetime(part["timestamp"]),
        "outcome": cast(object, outcome),
    }
    if native:
        value = NativeToolReturnPart(
            **kwargs,
            provider_name=_optional_string(part["provider_name"]),
            provider_details=_decode_optional_mapping(part["provider_details"]),
        )
        return NativeToolReturnPart.narrow_type(value)
    value = ToolReturnPart(**kwargs)
    return ToolReturnPart.narrow_type(value)


def _encode_speech_part(part: SpeechPart) -> dict[str, JsonValue]:
    return {
        "part_kind": "speech",
        "speaker": part.speaker,
        "transcript": part.transcript,
        "audio": (
            None
            if part.audio is None
            else _encode_user_content_item(part.audio)
        ),
        "interrupted_at_ms": part.interrupted_at_ms,
        "id": part.id,
        "provider_name": part.provider_name,
        "provider_details": _encode_optional_mapping(part.provider_details),
    }


def _decode_speech_part(part: Mapping[str, object]) -> SpeechPart:
    _require_keys(
        part,
        {
            "part_kind",
            "speaker",
            "transcript",
            "audio",
            "interrupted_at_ms",
            "id",
            "provider_name",
            "provider_details",
        },
    )
    speaker = _required_string(part["speaker"])
    if speaker not in {"user", "assistant"}:
        raise ValueError("speech speaker is invalid")
    audio = part["audio"]
    decoded_audio = None
    if audio is not None:
        decoded_audio = _decode_user_content_item(audio)
        if not isinstance(decoded_audio, BinaryContent):
            raise ValueError("speech audio is invalid")
    interrupted = part["interrupted_at_ms"]
    if interrupted is not None and (
        isinstance(interrupted, bool)
        or not isinstance(interrupted, int)
        or interrupted < 0
    ):
        raise ValueError("speech interruption offset is invalid")
    return SpeechPart(
        speaker=cast(object, speaker),
        transcript=_optional_string(part["transcript"]),
        audio=decoded_audio,
        interrupted_at_ms=cast(int | None, interrupted),
        id=_optional_string(part["id"]),
        provider_name=_optional_string(part["provider_name"]),
        provider_details=_decode_optional_mapping(part["provider_details"]),
    )


def _encode_usage(usage: RequestUsage) -> dict[str, JsonValue]:
    known = {
        "input_tokens": usage.input_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "output_tokens": usage.output_tokens,
        "input_audio_tokens": usage.input_audio_tokens,
        "cache_audio_read_tokens": usage.cache_audio_read_tokens,
        "output_audio_tokens": usage.output_audio_tokens,
        "details": _portable_json(usage.details),
        "cost": None if usage.cost is None else str(usage.cost),
    }
    extensions = {
        key: _portable_json(value)
        for key, value in sorted(usage.__dict__.items())
        if key not in known
    }
    known["extensions"] = extensions
    return cast(dict[str, JsonValue], known)


def _decode_usage(value: object) -> RequestUsage:
    if not isinstance(value, Mapping):
        raise ValueError("usage must be an object")
    _require_keys(
        value,
        {
            "input_tokens",
            "cache_write_tokens",
            "cache_read_tokens",
            "output_tokens",
            "input_audio_tokens",
            "cache_audio_read_tokens",
            "output_audio_tokens",
            "details",
            "cost",
            "extensions",
        },
    )
    numeric_names = (
        "input_tokens",
        "cache_write_tokens",
        "cache_read_tokens",
        "output_tokens",
        "input_audio_tokens",
        "cache_audio_read_tokens",
        "output_audio_tokens",
    )
    numbers: dict[str, int] = {}
    for name in numeric_names:
        item = value[name]
        if isinstance(item, bool) or not isinstance(item, int) or item < 0:
            raise ValueError("usage token count is invalid")
        numbers[name] = item
    details = value["details"]
    extensions = value["extensions"]
    if not isinstance(details, Mapping) or not isinstance(extensions, Mapping):
        raise ValueError("usage details are invalid")
    if any(
        not isinstance(key, str)
        for key in set(details) | set(extensions)
    ):
        raise ValueError("usage extension key is invalid")
    cost = value["cost"]
    if cost is not None and not isinstance(cost, str):
        raise ValueError("usage cost is invalid")
    parsed_cost = None
    if cost is not None:
        try:
            parsed_cost = Decimal(cost)
        except (InvalidOperation, ValueError) as error:
            raise ValueError("usage cost is invalid") from error
        if not parsed_cost.is_finite():
            raise ValueError("usage cost is invalid")
    return RequestUsage(
        **numbers,
        details=dict(details),
        cost=parsed_cost,
        **dict(extensions),
    )


def _encode_optional_mapping(value: object) -> JsonValue:
    if value is None:
        return None
    encoded = _portable_json(value)
    if not isinstance(encoded, dict):
        raise TypeError("durable message metadata must be an object")
    return encoded


def _decode_optional_mapping(value: object) -> dict[str, object] | None:
    if value is None:
        return None
    encoded = _portable_json(value)
    if not isinstance(encoded, dict):
        raise ValueError("durable message metadata is invalid")
    return cast(dict[str, object], encoded)


def _encode_tool_kind(value: object) -> JsonValue:
    if value is None:
        return None
    if not isinstance(value, str) or value not in _TOOL_KINDS:
        raise TypeError("unsupported tool part kind")
    return value


def _decode_tool_kind(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("tool part kind is invalid")
    if value not in _TOOL_KINDS:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return value


def _encode_datetime(value: datetime | None) -> JsonValue:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError("message timestamp must be datetime")
    encoded = value.isoformat()
    return encoded[:-6] + "Z" if encoded.endswith("+00:00") else encoded


def _decode_optional_datetime(value: object) -> datetime | None:
    if value is None:
        return None
    return _required_datetime(value)


def _required_datetime(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("message timestamp is invalid")
    try:
        return datetime.fromisoformat(
            value[:-1] + "+00:00" if value.endswith("Z") else value
        )
    except ValueError as error:
        raise ValueError("message timestamp is invalid") from error


def _optional_string(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("optional string is invalid")
    return cast(str | None, value)


def _required_string(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("required string is invalid")
    return value


def _part_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("message part must be an object")
    return cast(Mapping[str, object], value)


def _require_keys(
    value: Mapping[object, object],
    expected: set[str],
) -> None:
    if set(value) != expected:
        raise ValueError("durable message shape is invalid")


__all__ = [
    "binary_content_usage",
    "decode_model_messages",
    "encode_model_messages",
    "freeze_model_messages",
    "project_transient_binary_content",
]
