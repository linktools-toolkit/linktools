#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical persistence conversion and active model-context projection."""

import binascii
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import cast

from pydantic import TypeAdapter
from pydantic_core import ErrorDetails
from pydantic_ai import RequestUsage
from pydantic_ai.settings import ThinkingEffort
from pydantic_ai.messages import (
    BinaryContent,
    CompactionPart,
    FilePart,
    InstructionBaselinePart,
    InstructionDeltaPart,
    InstructionDigest,
    InstructionId,
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelRequestAttempt,
    ModelResponse,
    NativeToolCallPart,
    NativeToolReturnPart,
    RetryPromptPart,
    SerializedInstructionId,
    SpeechPart,
    SystemPromptPart,
    TextContent,
    TextPart,
    ThinkingPart,
    ToolAvailabilityDeltaPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

try:
    from pydantic_ai.messages import RetryFeedbackPart, RetryInfo
except ImportError:
    RetryFeedbackPart = RetryInfo = None

try:
    from pydantic_ai.messages import ModelSettingsDeltaPart
except ImportError:
    ModelSettingsDeltaPart = None

from ..core import JsonValue, canonical_json_bytes
from ..errors import AIError, ErrorCode
from . import model_input_content, ReferencedContent, project_model_attempts, project_model_response_attempts
from ._attachment import binary_attachment_text, has_binary_attachment_metadata
from ._user_content_codec import RetiredReferencedContent, decode_user_content_item, encode_user_content_item
from ._wire_fields import decode_fields
from ._json_snapshot import stable_json_snapshot
from ._tool_return_codec import decode_tool_return_content, encode_tool_return_content

_MESSAGE_VERSION = 1
_TOOL_KINDS = frozenset({"tool-search", "capability-load"})
_REQUEST_STATES = frozenset({"complete", "interrupted"})
_RESPONSE_STATES = frozenset({"complete", "incomplete", "suspended", "interrupted"})
_FINISH_REASONS = frozenset({"stop", "length", "content_filter", "tool_call", "error"})
_TOOL_OUTCOMES = frozenset({"success", "failed", "denied", "interrupted", "retried"})
_TOOL_RETRY_CAUSES = frozenset({"validation_error", "model_retry", "unknown_tool", "unavailable_tool"})
_FEEDBACK_RETRY_CAUSES = frozenset({"validation_error", "model_retry", "no_output"})
_ERROR_DETAILS_ADAPTER = TypeAdapter(list[ErrorDetails])
_INSTRUCTION_ID_ADAPTER = TypeAdapter(SerializedInstructionId)


@dataclass
class SummaryPromptPart(SystemPromptPart):
    """Runtime-owned compaction provenance, serialized outside the SDK union."""


def without_history_system_prompts(
    messages: Sequence[ModelMessage],
) -> list[ModelMessage]:
    """Project current context without replacing trusted summaries or source data."""
    values: list[ModelMessage] = []
    for message in messages:
        if isinstance(message, ModelRequest):
            parts = [
                part for part in message.parts
                if not isinstance(part, SystemPromptPart)
                or isinstance(part, SummaryPromptPart)
            ]
            if not parts:
                continue
            message = replace(message, parts=parts)
        values.append(message)
    return values


def project_provider_messages(
    messages: Sequence[ModelMessage],
) -> list[ModelMessage]:
    """Lower Runtime summaries only in the outbound provider view."""
    return [
        replace(message, parts=[
            SystemPromptPart(part.content, timestamp=part.timestamp, dynamic_ref=part.dynamic_ref)
            if isinstance(part, SummaryPromptPart) else part
            for part in message.parts
        ]) if isinstance(message, ModelRequest) else message
        for message in messages
    ]


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
    try:
        return stable_json_snapshot(value)
    except (TypeError, ValueError) as error:
        raise TypeError("durable message value is not JSON portable") from error


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
            if not isinstance(part, (UserPromptPart, ToolReturnPart)):
                parts.append(part)
                continue
            content = []
            part_changed = False
            for item in model_input_content(part):
                if isinstance(item, ReferencedContent):
                    descriptor = encode_user_content_item(item)
                    canonical = decode_user_content_item(descriptor)
                    content.append(RetiredReferencedContent(canonical, canonical.content))
                    part_changed = True
                elif isinstance(item, BinaryContent):
                    if not has_binary_attachment_metadata(item, message):
                        content.append(binary_attachment_text(item))
                    part_changed = True
                else:
                    content.append(item)
            if part_changed:
                if isinstance(part, ToolReturnPart):
                    # Empty results still acknowledge their original tool call.
                    value = content if isinstance(part.content, list) else (content[0] if content else "")
                    parts.append(replace(part, content=value))
                elif content:
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
    """Count pending input bytes in user prompts and native tool results."""
    count = 0
    total_bytes = 0
    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            for item in model_input_content(part):
                if isinstance(item, ReferencedContent):
                    count += 1
                    total_bytes += item.ref.size
                elif isinstance(item, BinaryContent):
                    count += 1
                    total_bytes += len(item.data)
    return count, total_bytes


def encode_model_messages(messages: Sequence[ModelMessage]) -> bytes:
    """Encode model history using the LinkTools-owned v1 durable contract."""
    return canonical_json_bytes([_encode_message(message) for message in messages])


def model_attempts_details(
    attempts: Sequence[object], *, content_omission_reason: str | None = None,
) -> JsonValue:
    """Reuse the durable attempt shape for terminal failure diagnostics."""
    return _encode_model_attempts(project_model_attempts(
        attempts, content_omission_reason=content_omission_reason,
    ))


def model_attempts_usage(attempts: Sequence[object] | None) -> RequestUsage | None:
    """Sum only usage the SDK reported; an entirely unmeasured set stays unknown.

    This subtotal is not another provider request and must not be repriced.
    Non-streaming SDK accounting already consumes it; only the streaming
    failure boundary bridges the SDK's otherwise unaccounted exception group.
    """
    result = None
    for value in attempts or ():
        attempt = _model_request_attempt(value)
        if attempt.usage is not None:
            if result is None:
                result = RequestUsage()
            result.incr(attempt.usage)
    return result


def model_response_usage(response: ModelResponse) -> RequestUsage:
    """Project known billed usage without mutating or recounting SDK run usage."""
    result = RequestUsage()
    result.incr(response.usage)
    attempts = model_attempts_usage(response.failed_attempts)
    if attempts is not None:
        result.incr(attempts)
    return result


def _model_request_attempt(attempt: object) -> ModelRequestAttempt:
    if not isinstance(attempt, ModelRequestAttempt):
        raise TypeError("model request attempt is invalid")
    return attempt


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
            "agent_conversation_id": message.conversation_id,
            "metadata": _encode_optional_mapping(message.metadata),
            "state": message.state,
        }
    if isinstance(message, ModelResponse):
        message = project_model_response_attempts(message)
        result: dict[str, JsonValue] = {
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
            "agent_conversation_id": message.conversation_id,
            "metadata": _encode_optional_mapping(message.metadata),
            "state": message.state,
        }
        attempts = message.failed_attempts
        if attempts is not None:
            result["failed_attempts"] = _encode_model_attempts(attempts)
        return result
    raise TypeError(f"unsupported model message type: {type(message).__name__}")


def _decode_message(value: object) -> ModelMessage:
    if not isinstance(value, Mapping):
        raise ValueError("model message must be an object")
    version = value.get("version")
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("model message version is invalid")
    if version != _MESSAGE_VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    kind = value.get("kind")
    if kind == "request":
        return ModelRequest(**decode_fields(value, ModelRequest, {
            "state": lambda raw: _choice(raw, _REQUEST_STATES),
            "parts": lambda raw: _decode_list(raw, _decode_request_part),
            "timestamp": _decode_optional_datetime,
            "instructions": _optional_string,
            "run_id": _optional_string,
            "agent_conversation_id": _optional_string,
            "metadata": _decode_optional_mapping,
        }, required={"version", "kind"}, aliases={"agent_conversation_id": "conversation_id"}))
    if kind == "response":
        return ModelResponse(**decode_fields(value, ModelResponse, {
            "state": lambda raw: _choice(raw, _RESPONSE_STATES),
            "finish_reason": lambda raw: None if raw is None else _choice(raw, _FINISH_REASONS),
            "timestamp": _required_datetime,
            "failed_attempts": _decode_model_attempts,
            "parts": lambda raw: _decode_list(raw, _decode_response_part),
            "usage": _decode_usage,
            "model_name": _optional_string,
            "provider_name": _optional_string,
            "provider_url": _optional_string,
            "provider_details": _decode_optional_mapping,
            "provider_response_id": _optional_string,
            "run_id": _optional_string,
            "agent_conversation_id": _optional_string,
            "metadata": _decode_optional_mapping,
        }, required={"version", "kind"}, aliases={"agent_conversation_id": "conversation_id"}))
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

def _encode_request_part(part: object) -> dict[str, JsonValue]:
    if isinstance(part, SystemPromptPart):
        return {
            "part_kind": ("runtime-summary-prompt" if isinstance(part, SummaryPromptPart)
                          else "system-prompt"),
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
                "items": [encode_user_content_item(item) for item in part.content],
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
    if RetryFeedbackPart is not None and isinstance(part, RetryFeedbackPart):
        return {
            "part_kind": "retry-feedback",
            "content": _portable_json(_retry_content(part.content)),
            "retry": _encode_retry_info(part.retry, feedback=True),
            "timestamp": _encode_datetime(part.timestamp),
        }
    if isinstance(part, ToolAvailabilityDeltaPart):
        return {
            "part_kind": "tool-availability-delta",
            "tools_added": list(part.tools_added),
            "tool_call_id": part.tool_call_id,
        }
    if ModelSettingsDeltaPart is not None and isinstance(part, ModelSettingsDeltaPart):
        return {
            "part_kind": "model-settings-delta",
            "settings": _recorded_model_settings(part.settings),
            "top_level": _settings_top_level(part.top_level),
        }
    if isinstance(part, InstructionBaselinePart):
        if not isinstance(part.instructions, list):
            raise TypeError("instruction baseline must contain a list")
        return {
            "part_kind": "instruction-baseline",
            "instructions": [_encode_instruction_block(block) for block in part.instructions],
        }
    if isinstance(part, InstructionDeltaPart):
        return {
            "part_kind": "instruction-delta",
            "id": _encode_instruction_id(part.id),
            "content": _optional_string(part.content),
            "base_hash": None if part.base_hash is None else _instruction_hash(part.base_hash),
        }
    if isinstance(part, CompactionPart):
        if not hasattr(CompactionPart, "metadata"):
            raise TypeError("SDK does not support request-side compaction")
        return _encode_response_part(part)
    if isinstance(part, SpeechPart):
        return _encode_speech_part(part)
    raise TypeError(f"unsupported model request part: {type(part).__name__}")


def _decode_request_part(value: object) -> object:
    part = _part_mapping(value)
    kind = part.get("part_kind")
    if kind in {"system-prompt", "runtime-summary-prompt"}:
        part_type = SummaryPromptPart if kind == "runtime-summary-prompt" else SystemPromptPart
        return part_type(**decode_fields(part, part_type, {
            "content": _required_string,
            "timestamp": _required_datetime,
            "dynamic_ref": _optional_string,
        }, required={"part_kind"}))
    if kind == "user-prompt":
        return UserPromptPart(**decode_fields(part, UserPromptPart, {
            "content": _decode_prompt_content,
            "timestamp": _required_datetime,
        }, required={"part_kind"}))
    if kind == "tool-return":
        return _decode_tool_return_part(part, native=False)
    if kind == "retry-prompt":
        return RetryPromptPart(**decode_fields(part, RetryPromptPart, {
            "content": _retry_content,
            "tool_name": _optional_string,
            "tool_call_id": _required_string,
            "timestamp": _required_datetime,
        }, required={"part_kind", "tool_call_id"}))
    if kind == "retry-feedback":
        if RetryFeedbackPart is None:
            _require_keys(part, {"part_kind", "content", "retry", "timestamp"})
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return RetryFeedbackPart(**decode_fields(part, RetryFeedbackPart, {
            "content": _retry_content,
            "retry": lambda raw: _decode_retry_info(raw, feedback=True),
            "timestamp": _required_datetime,
        }, required={"part_kind"}))
    if kind == "tool-availability-delta":
        return ToolAvailabilityDeltaPart(**decode_fields(part, ToolAvailabilityDeltaPart, {
            "tools_added": lambda raw: _decode_list(raw, _required_string),
            "tool_call_id": _optional_string,
        }, required={"part_kind"}))
    if kind == "model-settings-delta":
        if ModelSettingsDeltaPart is None:
            _require_keys(part, {"part_kind", "settings", "top_level"})
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return ModelSettingsDeltaPart(**decode_fields(part, ModelSettingsDeltaPart, {
            "settings": _recorded_model_settings,
            "top_level": _settings_top_level,
        }, required={"part_kind"}))
    if kind == "instruction-baseline":
        return InstructionBaselinePart(**decode_fields(part, InstructionBaselinePart, {
            "instructions": lambda raw: _decode_list(raw, _decode_instruction_block),
        }, required={"part_kind"}))
    if kind == "instruction-delta":
        return InstructionDeltaPart(**decode_fields(part, InstructionDeltaPart, {
            "id": _decode_instruction_id,
            "content": _optional_string,
            "base_hash": lambda raw: None if raw is None else _instruction_hash(raw),
        }, required={"part_kind"}))
    if kind == "compaction":
        if not hasattr(CompactionPart, "metadata"):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return _decode_response_part(part)
    if kind == "speech":
        speech = _decode_speech_part(part)
        if speech.speaker != "user":
            raise ValueError("request speech speaker is invalid")
        return speech
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _decode_prompt_content(value: object) -> object:
    if not isinstance(value, Mapping):
        raise ValueError("user prompt content is invalid")
    content_type = value.get("type")
    if content_type == "text":
        _require_keys(value, {"type", "value"})
        return _required_string(value["value"])
    if content_type == "items":
        _require_keys(value, {"type", "items"})
        return _decode_list(value["items"], decode_user_content_item)
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

def _recorded_model_settings(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise ValueError("recorded model settings must be a mapping")
    if value.get("thinking") is not None:
        TypeAdapter(ThinkingEffort).validate_python(value["thinking"], strict=True)
    return cast(dict[str, JsonValue], _portable_json(value))


def _settings_top_level(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("model settings top_level must be a boolean")
    return value


def _encode_instruction_block(block: object) -> dict[str, JsonValue]:
    if isinstance(block, InstructionPart):
        return {
            "part_kind": "instruction",
            "id": _encode_instruction_id(block.id),
            "content": _required_string(block.content),
            "dynamic": _instruction_dynamic(block.dynamic),
            "on_change": _instruction_policy(block.on_change),
            "name": _optional_string(block.name),
        }
    if isinstance(block, InstructionDigest):
        return {
            "part_kind": "instruction-digest",
            "id": _encode_instruction_id(block.id),
            "dynamic": _instruction_dynamic(block.dynamic),
            "content_hash": _instruction_hash(block.content_hash),
        }
    raise TypeError(f"unsupported instruction baseline block: {type(block).__name__}")


def _decode_instruction_block(value: object) -> object:
    block = _part_mapping(value)
    kind = block.get("part_kind")
    if kind == "instruction":
        return InstructionPart(**decode_fields(block, InstructionPart, {
            "content": _required_string,
            "id": _decode_instruction_id,
            "dynamic": _instruction_dynamic,
            "on_change": _instruction_policy,
            "name": _optional_string,
        }, required={"part_kind"}))
    if kind == "instruction-digest":
        return InstructionDigest(**decode_fields(block, InstructionDigest, {
            "id": _decode_instruction_id,
            "dynamic": _instruction_dynamic,
            "content_hash": _instruction_hash,
        }, required={"part_kind"}))
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)

def _encode_instruction_id(value: InstructionId | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, InstructionId):
        raise TypeError("instruction id must be an InstructionId")
    encoded = str(value)
    if _decode_instruction_id(encoded) != value:
        raise ValueError("instruction id is invalid")
    return encoded


def _decode_instruction_id(value: object) -> InstructionId | None:
    if value is None:
        return None
    encoded = _required_string(value)
    result = _INSTRUCTION_ID_ADAPTER.validate_python(encoded)
    # SDK readers replace unknown source namespaces with None; durable history must not lose them.
    if result is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return result


def _instruction_dynamic(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("instruction dynamic flag is invalid")
    return value


def _instruction_policy(value: object) -> str:
    if not isinstance(value, str) or value not in {"rewrite", "append"}:
        raise ValueError("instruction change policy is invalid")
    return value


def _instruction_hash(value: object) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("instruction content hash is invalid")
    return value


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
            **({"metadata": _encode_optional_mapping(part.metadata)}
               if hasattr(part, "metadata") and part.metadata is not None else {}),
        }
    if isinstance(part, FilePart):
        return {
            "part_kind": "file",
            "content": encode_user_content_item(part.content),
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
        return TextPart(**decode_fields(part, TextPart, {
            "content": _required_string,
            "id": _optional_string,
            "provider_name": _optional_string,
            "provider_details": _decode_optional_mapping,
        }, required={"part_kind"}))
    if kind == "tool-call":
        return _decode_tool_call_part(part, native=False)
    if kind == "builtin-tool-call":
        return _decode_tool_call_part(part, native=True)
    if kind == "builtin-tool-return":
        return _decode_tool_return_part(part, native=True)
    if kind == "thinking":
        return ThinkingPart(**decode_fields(part, ThinkingPart, {
            "content": _required_string,
            "id": _optional_string,
            "signature": _optional_string,
            "provider_name": _optional_string,
            "provider_details": _decode_optional_mapping,
        }, required={"part_kind"}))
    if kind == "compaction":
        if "metadata" in part and not hasattr(CompactionPart, "metadata"):
            _require_keys(part, {"part_kind", "content", "id", "provider_name", "provider_details", "metadata"})
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        decoders = {
            "content": _optional_string,
            "id": _optional_string,
            "provider_name": _optional_string,
            "provider_details": _decode_optional_mapping,
        }
        if hasattr(CompactionPart, "metadata"):
            decoders["metadata"] = _decode_optional_mapping
        return CompactionPart(**decode_fields(part, CompactionPart, decoders, required={"part_kind"}))
    if kind == "file":
        return FilePart(**decode_fields(part, FilePart, {
            "content": _decode_binary_content,
            "id": _optional_string,
            "provider_name": _optional_string,
            "provider_details": _decode_optional_mapping,
        }, required={"part_kind"}))
    if kind == "speech":
        speech = _decode_speech_part(part)
        if speech.speaker != "assistant":
            raise ValueError("response speech speaker is invalid")
        return speech
    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)


def _decode_binary_content(value: object) -> BinaryContent:
    content = decode_user_content_item(value)
    if not isinstance(content, BinaryContent):
        raise ValueError("binary message content is invalid")
    return content

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
    part_type = NativeToolCallPart if native else ToolCallPart
    value = part_type(**decode_fields(part, part_type, {
        "tool_kind": _decode_tool_kind,
        "args": _decode_tool_args,
        "tool_name": _required_string,
        "tool_call_id": _required_string,
        "id": _optional_string,
        "provider_name": _optional_string,
        "provider_details": _decode_optional_mapping,
    }, required={"part_kind", "tool_call_id"}))
    return part_type.narrow_type(value)


def _decode_tool_args(value: object) -> object:
    if value is not None and not isinstance(value, (str, Mapping)):
        raise ValueError("tool call args are invalid")
    return dict(value) if isinstance(value, Mapping) else value

def _encode_tool_return_part(
    part: ToolReturnPart | NativeToolReturnPart,
    *,
    native: bool,
) -> dict[str, JsonValue]:
    if part.outcome not in _TOOL_OUTCOMES:
        raise ValueError("tool outcome is invalid")
    if part.outcome == "retried" and RetryInfo is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
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
    elif RetryInfo is not None:
        tool_part = cast(ToolReturnPart, part)
        if tool_part.retry is not None:
            result["retry"] = _encode_retry_info(tool_part.retry, feedback=False)
    return result


def _decode_tool_return_part(
    part: Mapping[str, object],
    *,
    native: bool,
) -> object:
    part_type = NativeToolReturnPart if native else ToolReturnPart
    decoders: dict[str, Callable[[object], object]] = {
        "outcome": _decode_tool_outcome,
        "tool_name": _required_string,
        "content": decode_tool_return_content,
        "tool_call_id": _required_string,
        "tool_kind": _decode_tool_kind,
        "metadata": _portable_json,
        "timestamp": _required_datetime,
    }
    if native:
        decoders.update(provider_name=_optional_string, provider_details=_decode_optional_mapping)
    elif "retry" in part or RetryInfo is not None:
        decoders["retry"] = lambda raw: None if raw is None else _decode_retry_info(raw, feedback=False)
    kwargs = decode_fields(part, part_type, decoders, required={"part_kind", "tool_call_id"})
    if kwargs.get("retry") is None and not any(field.name == "retry" for field in fields(part_type)):
        kwargs.pop("retry", None)
    value = part_type(**kwargs)
    return part_type.narrow_type(value)


def _decode_tool_outcome(value: object) -> str:
    outcome = _choice(value, _TOOL_OUTCOMES)
    if outcome == "retried" and RetryInfo is None:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    return outcome

def _retry_content(value: object) -> str | list[ErrorDetails]:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return _ERROR_DETAILS_ADAPTER.validate_python(value)
    raise ValueError("retry content is invalid")


def _encode_retry_info(value: object, *, feedback: bool) -> dict[str, JsonValue]:
    if RetryInfo is None or not isinstance(value, RetryInfo):
        raise TypeError("retry information is invalid")
    causes = _FEEDBACK_RETRY_CAUSES if feedback else _TOOL_RETRY_CAUSES
    if not isinstance(value.cause, str) or value.cause not in causes:
        raise ValueError("retry cause is invalid")
    for budget in (value.attempt, value.max_retries):
        if budget is not None and (isinstance(budget, bool) or not isinstance(budget, int)):
            raise ValueError("retry budget must be an integer or null")
    return {"cause": value.cause, "attempt": value.attempt, "max_retries": value.max_retries}


def _decode_retry_info(value: object, *, feedback: bool) -> "RetryInfo":
    raw = _part_mapping(value)
    if RetryInfo is None:
        _require_keys(raw, {"cause", "attempt", "max_retries"})
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    retry = RetryInfo(**decode_fields(raw, RetryInfo, {
        "cause": lambda item: item,
        "attempt": lambda item: item,
        "max_retries": lambda item: item,
    }))
    # The SDK dataclass constructor does not validate its field annotations.
    _encode_retry_info(retry, feedback=feedback)
    return retry

def _encode_speech_part(part: SpeechPart) -> dict[str, JsonValue]:
    return {
        "part_kind": "speech",
        "speaker": part.speaker,
        "transcript": part.transcript,
        "audio": (
            None
            if part.audio is None
            else encode_user_content_item(part.audio)
        ),
        "interrupted_at_ms": part.interrupted_at_ms,
        "id": part.id,
        "provider_name": part.provider_name,
        "provider_details": _encode_optional_mapping(part.provider_details),
    }


def _decode_speech_part(part: Mapping[str, object]) -> SpeechPart:
    return SpeechPart(**decode_fields(part, SpeechPart, {
        "speaker": lambda raw: _choice(raw, frozenset({"user", "assistant"})),
        "transcript": _optional_string,
        "audio": lambda raw: None if raw is None else _decode_binary_content(raw),
        "interrupted_at_ms": lambda raw: None if raw is None else _nonnegative_integer(raw),
        "id": _optional_string,
        "provider_name": _optional_string,
        "provider_details": _decode_optional_mapping,
    }, required={"part_kind"}))

def _encode_model_attempts(attempts: Sequence[object]) -> JsonValue:
    values: list[JsonValue] = []
    for value in attempts:
        attempt = _model_request_attempt(value)
        duration = attempt.duration
        if not isinstance(duration, timedelta) or duration < timedelta():
            raise ValueError("model request attempt duration is invalid")
        if not isinstance(attempt.timestamp, datetime) or attempt.timestamp.tzinfo is None:
            raise ValueError("model request attempt timestamp must include a timezone")
        if attempt.outcome not in {"error", "rejected"}:
            raise ValueError("model request attempt outcome is invalid")
        values.append({
            "model_name": attempt.model_name,
            "provider_name": attempt.provider_name,
            "outcome": attempt.outcome,
            "error": attempt.error,
            "timestamp": _encode_datetime(attempt.timestamp),
            "duration_us": ((duration.days * 86400 + duration.seconds) * 1000000
                            + duration.microseconds),
            "usage": None if attempt.usage is None else _encode_usage(attempt.usage),
        })
    return values


def _decode_model_attempts(value: object) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("model request attempts must be a list")
    attempts: list[object] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ValueError("model request attempt must be an object")
        attempts.append(ModelRequestAttempt(**decode_fields(item, ModelRequestAttempt, {
            "model_name": _required_string,
            "provider_name": _optional_string,
            "outcome": lambda raw: _choice(raw, frozenset({"error", "rejected"})),
            "error": _optional_string,
            "timestamp": _attempt_timestamp,
            "duration_us": _attempt_duration,
            "usage": lambda raw: None if raw is None else _decode_usage(raw),
        }, aliases={"duration_us": "duration"})))
    return attempts


def _attempt_timestamp(value: object) -> datetime:
    timestamp = _required_datetime(value)
    if timestamp.tzinfo is None:
        raise ValueError("model request attempt timestamp must include a timezone")
    return timestamp


def _attempt_duration(value: object) -> timedelta:
    if type(value) is not int or value < 0:
        raise ValueError("model request attempt duration is invalid")
    try:
        return timedelta(microseconds=value)
    except OverflowError as error:
        raise ValueError("model request attempt duration is invalid") from error

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
    numeric_names = (
        "input_tokens", "cache_write_tokens", "cache_read_tokens", "output_tokens",
        "input_audio_tokens", "cache_audio_read_tokens", "output_audio_tokens",
    )
    decoders: dict[str, Callable[[object], object]] = {
        name: _nonnegative_integer for name in numeric_names
    }
    decoders.update(details=_usage_details, cost=_usage_cost)
    kwargs = decode_fields(value, RequestUsage, decoders, required={"extensions"})
    extensions = value["extensions"]
    if not isinstance(extensions, Mapping) or any(not isinstance(key, str) for key in extensions):
        raise ValueError("usage extensions are invalid")
    if set(extensions) & (set(decoders) | {"extensions"}):
        raise ValueError("usage extensions cannot override standard fields")
    return RequestUsage(**kwargs, **dict(extensions))


def _usage_details(value: object) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError("usage details are invalid")
    return dict(value)


def _usage_cost(value: object) -> Decimal | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("usage cost is invalid")
    try:
        result = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ValueError("usage cost is invalid") from error
    if not result.is_finite():
        raise ValueError("usage cost is invalid")
    return result


def _nonnegative_integer(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("nonnegative integer is required")
    return value

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



def _decode_list(value: object, decode: Callable[[object], object]) -> list[object]:
    if not isinstance(value, list):
        raise ValueError("message field must be a list")
    return [decode(item) for item in value]


def _choice(value: object, choices: frozenset[str]) -> str:
    result = _required_string(value)
    if result not in choices:
        raise ValueError("message field has an invalid choice")
    return result


def _require_keys(
    value: Mapping[object, object],
    expected: set[str],
) -> None:
    if set(value) != expected:
        raise ValueError("durable message shape is invalid")


__all__ = ['RetiredReferencedContent', 'SummaryPromptPart', 'UnsupportedUserContentKind', 'binary_attachment_text', 'binary_content_usage', 'bind_tool_return_attachments', 'decode_model_messages', 'decode_tool_return_content', 'decode_user_content_item', 'encode_model_messages', 'encode_tool_return_content', 'encode_user_content_item', 'file_tool_result_metadata', 'freeze_model_messages', 'has_binary_attachment_metadata', 'input_attachment_views', 'is_binary_attachment_text', 'model_attempts_details', 'model_attempts_usage', 'model_input_content', 'model_response_usage', 'project_model_response_attempts', 'project_provider_messages', 'project_transient_binary_content', 'rehydrate_deferred_tool_results', 'request_attachment_facts', 'stable_json_snapshot', 'tool_return_content_digest', 'without_history_system_prompts']

from ._attachment import (binary_attachment_text, bind_tool_return_attachments, file_tool_result_metadata, has_binary_attachment_metadata, input_attachment_views, is_binary_attachment_text, request_attachment_facts)

from ._user_content_codec import (RetiredReferencedContent, UnsupportedUserContentKind, decode_user_content_item, encode_user_content_item)

from ._json_snapshot import (stable_json_snapshot)

from ._tool_return_codec import (decode_tool_return_content, encode_tool_return_content, rehydrate_deferred_tool_results, tool_return_content_digest)
