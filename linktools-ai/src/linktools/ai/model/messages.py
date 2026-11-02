#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical persistence conversion and active model-context projection."""

import binascii
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, fields, replace
from datetime import datetime, timedelta
from decimal import Decimal
from typing import cast

from pydantic import TypeAdapter, ValidationError
from pydantic_core import PydanticSerializationError
from pydantic_ai import RequestUsage
from pydantic_ai.settings import ThinkingEffort
from typing import get_args
from pydantic_ai.messages import (
    BinaryContent, FilePart, InstructionBaselinePart, InstructionDeltaPart, InstructionDigest,
    InstructionId, InstructionPart, SerializedInstructionId, ModelMessage, ModelMessagesTypeAdapter, ModelRequest,
    ModelRequestAttempt, ModelResponse, NativeToolReturnPart, RetryPromptPart,
    SpeechPart, SystemPromptPart, TextContent, ToolReturnPart, UserContent, UserPromptPart,
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
from ._json_snapshot import stable_json_snapshot
from ._tool_return_codec import decode_tool_return_content, encode_tool_return_content

_MESSAGE_VERSION = 1
_HOST_CONTENT = "linktools-content"
_SUMMARY_KIND = "runtime-summary-prompt"
_RETRY_ADAPTER = TypeAdapter(RetryPromptPart)
_USAGE_ADAPTER = TypeAdapter(RequestUsage)
_USER_CONTENT_ADAPTER = TypeAdapter(UserContent)
_DURATION_ADAPTER = TypeAdapter(timedelta)
_INSTRUCTION_ID_ADAPTER = TypeAdapter(SerializedInstructionId)
_USAGE_TOKEN_FIELDS = tuple(field.name for field in fields(RequestUsage) if field.name.endswith("_tokens"))
_CORRELATED_PARTS = frozenset({
    "tool-call", "builtin-tool-call", "tool-return", "builtin-tool-return", "retry-prompt",
})


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
            _system_prompt(part)
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
    """Encode native message fields with the Runtime-owned durable extensions."""
    try:
        return canonical_json_bytes(_encode_messages(messages))
    except PydanticSerializationError:
        # Native diagnostics can embed message bodies; do not expose them as codec errors.
        raise ValueError("durable native message value is invalid") from None


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


def _parse_model_message_payload(raw: bytes) -> list[JsonValue]:
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
    return cast(list[JsonValue], value)


def _read_model_message_payloads(
    payloads: Iterable[bytes], *, decode: bool,
) -> tuple[bytes, tuple[ModelMessage, ...]]:
    fragments: list[bytes | memoryview] = [b"["]
    messages: list[ModelMessage] = []
    try:
        for raw in payloads:
            leaf = _parse_model_message_payload(raw)
            if len(leaf) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if len(fragments) > 1:
                fragments.append(b",")
            fragments.append(memoryview(raw)[1:-1])
            if decode:
                messages.append(_decode_message(leaf[0]))
    except AIError:
        raise
    except ValidationError:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from None
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    fragments.append(b"]")
    return b"".join(fragments), tuple(messages)


def combine_model_message_payloads(payloads: Iterable[bytes]) -> bytes:
    """Combine validated singleton arrays without materializing SDK defaults.

    Message semantics are checked by the normal decoder when consumed. Original
    canonical fragments are framed once, without copying each array's contents.
    """
    return _read_model_message_payloads(payloads, decode=False)[0]


def decode_model_message_payloads(
    payloads: Iterable[bytes],
) -> tuple[bytes, tuple[ModelMessage, ...]]:
    """Read singleton message arrays and retain their original canonical bytes."""
    return _read_model_message_payloads(payloads, decode=True)


def decode_model_messages(raw: bytes) -> tuple[ModelMessage, ...]:
    value = _parse_model_message_payload(raw)
    try:
        return tuple(_decode_message(item) for item in value)
    except AIError:
        raise
    except ValidationError:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from None
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error


def _system_prompt(part: SystemPromptPart) -> SystemPromptPart:
    return SystemPromptPart(**{
        field.name: getattr(part, field.name) for field in fields(SystemPromptPart) if field.init
    })


def _normalize_native_wire(source: object, wire: object) -> object:
    """Check pre-JSON values and order open metadata sets in one linear pass."""
    if isinstance(source, float) and not math.isfinite(source):
        raise ValueError("durable message JSON requires finite floats")
    if isinstance(source, Mapping):
        for key, item in source.items():
            if not isinstance(key, str):
                raise TypeError("durable message mappings require string keys")
            if isinstance(wire, dict) and key in wire:
                wire[key] = _normalize_native_wire(item, wire[key])
            else:
                _normalize_native_wire(item, None)
    elif isinstance(source, (list, tuple)):
        for index, item in enumerate(source):
            if isinstance(wire, list) and index < len(wire):
                wire[index] = _normalize_native_wire(item, wire[index])
            else:
                _normalize_native_wire(item, None)
    elif isinstance(source, (set, frozenset)):
        return stable_json_snapshot(source)
    return wire


def _dump_native(adapter: TypeAdapter, value: object) -> object:
    # Keep the pre-JSON snapshot: native JSON serialization can erase NaN or key types.
    source = adapter.dump_python(value, mode="python", warnings="error")
    wire = adapter.dump_python(value, mode="json", warnings="error")
    return _normalize_native_wire(source, wire)


def _recorded_settings(value: object) -> dict[str, JsonValue]:
    # Recorded windows are open JSON, unlike the SDK's model settings TypedDict.
    result = _portable_json(value)
    if not isinstance(result, dict):
        raise ValueError("recorded model settings must be an object")
    thinking = result.get("thinking")
    if thinking is not None and thinking not in get_args(ThinkingEffort):
        raise ValueError("recorded thinking setting is invalid")
    return result


def _tool_return_with_content(
    part: ToolReturnPart | NativeToolReturnPart, content: object, tool_kind: str | None,
) -> ToolReturnPart | NativeToolReturnPart:
    if tool_kind is not None and not isinstance(tool_kind, str):
        raise TypeError("tool kind must be a string")
    owner = NativeToolReturnPart if isinstance(part, NativeToolReturnPart) else ToolReturnPart
    restored = owner(**{
        field.name: (content if field.name == "content" else tool_kind if field.name == "tool_kind"
                     else getattr(part, field.name))
        for field in fields(owner) if field.init
    })
    restored = owner.narrow_type(restored)
    if restored.tool_kind != tool_kind:
        raise ValueError("tool content does not substantiate its native kind")
    return restored


def _validate_message_facts(message: ModelMessage) -> None:
    """Enforce durable invariants absent from the SDK's authoring dataclasses."""
    usages = []
    if isinstance(message, ModelResponse):
        usages.append(message.usage)
        for attempt in message.failed_attempts or ():
            if not isinstance(attempt.duration, timedelta) or attempt.duration < timedelta():
                raise ValueError("model request attempt duration is invalid")
            if not isinstance(attempt.timestamp, datetime) or attempt.timestamp.tzinfo is None:
                raise ValueError("model request attempt timestamp must include a timezone")
            if attempt.usage is not None:
                usages.append(attempt.usage)
    for usage in usages:
        if usage.cost is not None and (not isinstance(usage.cost, Decimal) or not usage.cost.is_finite()):
            raise ValueError("usage cost is invalid")
        for name in _USAGE_TOKEN_FIELDS:
            count = getattr(usage, name)
            if type(count) is not int or count < 0:
                raise ValueError("usage token count must be a nonnegative integer")
    for part in message.parts:
        content = (part.content if isinstance(part, UserPromptPart) else
                   [part.content] if isinstance(part, FilePart) else
                   [part.audio] if isinstance(part, SpeechPart) else ())
        if not isinstance(content, str):
            for item in content:
                if isinstance(item, BinaryContent) and (not isinstance(item.media_type, str) or not item.media_type):
                    raise ValueError("binary content media type is invalid")
        if isinstance(part, SpeechPart) and part.interrupted_at_ms is not None:
            if type(part.interrupted_at_ms) is not int or part.interrupted_at_ms < 0:
                raise ValueError("speech interruption position is invalid")
        retry = getattr(part, "retry", None)
        if retry is not None:
            if RetryInfo is None or not isinstance(retry, RetryInfo):
                raise TypeError("retry information is invalid")
            causes = ({"validation_error", "model_retry", "no_output"}
                      if RetryFeedbackPart is not None and isinstance(part, RetryFeedbackPart)
                      else {"validation_error", "model_retry", "unknown_tool", "unavailable_tool"})
            if retry.cause not in causes:
                raise ValueError("retry cause is invalid")
            for budget in (retry.attempt, retry.max_retries):
                if budget is not None and type(budget) is not int:
                    raise ValueError("retry budget must be an integer or null")
        instructions = (part.instructions if isinstance(part, InstructionBaselinePart)
                        else [part] if isinstance(part, InstructionDeltaPart) else ())
        for instruction in instructions:
            if not isinstance(instruction, (InstructionPart, InstructionDigest, InstructionDeltaPart)):
                raise TypeError("instruction value is invalid")
            if instruction.id is not None and not isinstance(instruction.id, InstructionId):
                raise TypeError("instruction identity is invalid")
            if instruction.id is not None:
                identity = _INSTRUCTION_ID_ADAPTER.validate_python(str(instruction.id))
                if identity is None:
                    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
                if identity != instruction.id:
                    raise ValueError("instruction identity is not durably representable")
            digest = (instruction.content_hash if isinstance(instruction, InstructionDigest)
                      else instruction.base_hash if isinstance(instruction, InstructionDeltaPart) else None)
            if digest is not None and (not isinstance(digest, str) or len(digest) != 64
                                       or any(char not in "0123456789abcdef" for char in digest)):
                raise ValueError("instruction digest is invalid")


def _require_preserved_fields(supplied: object, restored: object) -> None:
    """Allow declared missing defaults, not ignored or coerced explicit facts."""
    if supplied is restored:
        return
    if isinstance(supplied, dict):
        if not isinstance(restored, dict) or supplied.keys() - restored.keys():
            raise ValueError("native message decoder discarded fields")
        for key, value in supplied.items():
            _require_preserved_fields(value, restored[key])
    elif isinstance(supplied, list):
        if not isinstance(restored, list) or len(supplied) != len(restored):
            raise ValueError("native message decoder changed a sequence")
        for left, right in zip(supplied, restored):
            _require_preserved_fields(left, right)
    elif isinstance(supplied, str) and isinstance(restored, str) and supplied.endswith("+00:00") and restored == supplied[:-6] + "Z":
        # Native datetime JSON uses Z for UTC; source archive bytes remain untouched.
        return
    elif type(supplied) is not type(restored) or supplied != restored:
        raise ValueError("native message decoder changed an explicit value")


def _encode_messages(messages: Sequence[ModelMessage]) -> list[JsonValue]:
    lowered = []
    patches = []
    for mi, message in enumerate(messages):
        _validate_message_facts(message)
        if isinstance(message, ModelResponse):
            message = project_model_response_attempts(message)
        parts = []
        for pi, part in enumerate(message.parts):
            if isinstance(part, RetryPromptPart):
                # The native union upgrades this supported authoring type to retry feedback,
                # losing its tool correlation. Its own adapter retains the recorded fact.
                parts.append(UserPromptPart("", timestamp=part.timestamp))
                patches.append((mi, pi, "retry", _dump_native(_RETRY_ADAPTER, part)))
            elif ModelSettingsDeltaPart is not None and isinstance(part, ModelSettingsDeltaPart):
                settings = _recorded_settings(part.settings)
                parts.append(replace(part, settings={}))
                patches.append((mi, pi, "settings", settings))
            elif isinstance(part, SummaryPromptPart):
                parts.append(_system_prompt(part))
                patches.append((mi, pi, "summary", None))
            elif isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                content = []
                for ci, item in enumerate(part.content):
                    if isinstance(item, (ReferencedContent, RetiredReferencedContent)):
                        content.append(TextContent(""))
                        patches.append((mi, pi, ci, {
                            "kind": _HOST_CONTENT, "value": encode_user_content_item(item),
                        }))
                    else:
                        content.append(item)
                parts.append(replace(part, content=content))
            elif isinstance(part, (ToolReturnPart, NativeToolReturnPart)):
                patches.append((mi, pi, "tool", (encode_tool_return_content(part.content), part.tool_kind)))
                # The content owner preserves host references and snapshots. Native kind
                # promotion happens against restored content, never its JSON placeholder.
                parts.append(_tool_return_with_content(part, "", None))
            else:
                parts.append(part)
        lowered.append(replace(message, parts=parts))
    wire = _dump_native(ModelMessagesTypeAdapter, lowered)
    for mi, pi, target, value in patches:
        part = wire[mi]["parts"][pi]
        if target == "summary":
            part["part_kind"] = _SUMMARY_KIND
        elif target == "retry":
            wire[mi]["parts"][pi] = value
        elif target == "settings":
            part["settings"] = value
        elif target == "tool":
            original = messages[mi].parts[pi]
            _tool_return_with_content(original, original.content, original.tool_kind)
            part["content"], part["tool_kind"] = value
        else:
            part["content"][target] = value
    for message, original in zip(wire, lowered):
        message["version"] = _MESSAGE_VERSION
        if isinstance(original, ModelResponse):
            # The nested native attempt schema omits provider usage extensions.
            # Delegate this value to its own native adapter, not another field map.
            for index, attempt in enumerate(original.failed_attempts or ()):
                if attempt.usage is not None:
                    message["failed_attempts"][index]["usage"] = _dump_native(_USAGE_ADAPTER, attempt.usage)
    return wire


def _legacy_message_wire(value: dict[str, object]) -> dict[str, object]:
    """Lower the frozen filesystem message format without altering its stored bytes."""
    if "conversation_id" in value:
        raise ValueError("message contains conflicting conversation identities")
    wire = dict(value)
    wire["conversation_id"] = wire.pop("agent_conversation_id")

    def content(raw: object) -> object:
        item = decode_user_content_item(raw)
        _require_preserved_fields(raw, encode_user_content_item(item))
        if isinstance(item, (ReferencedContent, RetiredReferencedContent)):
            return {"kind": _HOST_CONTENT, "value": raw}
        return _dump_native(_USER_CONTENT_ADAPTER, item)

    def usage(raw: object) -> object:
        if not isinstance(raw, dict):
            raise ValueError("legacy usage must be an object")
        result = dict(raw)
        extensions = result.pop("extensions")
        if not isinstance(extensions, dict) or extensions.keys() & (result.keys() | {"extensions"}):
            raise ValueError("legacy usage extensions conflict with recorded fields")
        result.update(extensions)
        return result

    parts = []
    for original in wire["parts"]:
        if not isinstance(original, dict):
            raise ValueError("legacy message part must be an object")
        part = dict(original)
        kind = part.get("part_kind")
        if kind == "user-prompt":
            raw = part["content"]
            if not isinstance(raw, dict):
                raise ValueError("legacy prompt envelope is invalid")
            if raw.get("type") == "text" and set(raw) == {"type", "value"}:
                if not isinstance(raw["value"], str):
                    raise ValueError("legacy prompt text is invalid")
                part["content"] = raw["value"]
            elif raw.get("type") == "items" and set(raw) == {"type", "items"} and isinstance(raw["items"], list):
                part["content"] = [content(item) for item in raw["items"]]
            else:
                raise ValueError("legacy prompt envelope is invalid")
        elif kind == "file":
            part["content"] = content(part["content"])
        elif kind == "speech" and part.get("audio") is not None:
            part["audio"] = content(part["audio"])
        parts.append(part)
    wire["parts"] = parts
    if wire.get("kind") == "response":
        wire["usage"] = usage(wire["usage"])
        if wire.get("failed_attempts") is not None:
            if not isinstance(wire["failed_attempts"], list):
                raise ValueError("legacy attempts must be a list")
            attempts = []
            for original in wire["failed_attempts"]:
                if not isinstance(original, dict):
                    raise ValueError("legacy attempt must be an object")
                attempt = dict(original)
                duration = attempt.pop("duration_us")
                if type(duration) is not int or duration < 0 or "duration" in attempt:
                    raise ValueError("legacy attempt duration is invalid")
                try:
                    elapsed = timedelta(microseconds=duration)
                except OverflowError as error:
                    raise ValueError("legacy attempt duration is invalid") from error
                attempt["duration"] = _dump_native(_DURATION_ADAPTER, elapsed)
                if attempt.get("usage") is not None:
                    attempt["usage"] = usage(attempt["usage"])
                attempts.append(attempt)
            wire["failed_attempts"] = attempts
    return wire


def _decode_message(value: object) -> ModelMessage:
    if not isinstance(value, dict):
        raise ValueError("message must be an object")
    version = value.get("version")
    if type(version) is not int:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if version != _MESSAGE_VERSION:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    if "kind" not in value or "parts" not in value or not isinstance(value["parts"], list):
        raise ValueError("durable message discriminators are required")
    if "agent_conversation_id" in value:
        value = _legacy_message_wire(value)
    wire = {key: item for key, item in value.items() if key != "version"}
    wire["parts"] = [dict(part) if isinstance(part, dict) else part for part in wire["parts"]]
    patches = []
    attempt_usages = []
    if isinstance(wire.get("failed_attempts"), list):
        wire["failed_attempts"] = [dict(attempt) if isinstance(attempt, dict) else attempt
                                   for attempt in wire["failed_attempts"]]
        for index, attempt in enumerate(wire["failed_attempts"]):
            if isinstance(attempt, dict) and attempt.get("usage") is not None:
                usage = _USAGE_ADAPTER.validate_python(attempt["usage"])
                usage_wire = _USAGE_ADAPTER.dump_python(usage, mode="json", warnings="error")
                _require_preserved_fields(attempt["usage"], usage_wire)
                attempt_usages.append((index, usage))
                attempt["usage"] = None
    for pi, part in enumerate(wire["parts"]):
        if not isinstance(part, dict) or "part_kind" not in part:
            raise ValueError("durable part discriminator is required")
        kind = part["part_kind"]
        content = (part.get("content") if kind == "user-prompt" else
                   [part.get("content")] if kind == "file" else
                   [part.get("audio")] if kind == "speech" else ())
        if isinstance(content, (list, tuple)):
            for item in content:
                if isinstance(item, dict) and "kind" not in item:
                    raise ValueError("durable content discriminator is required")
        instructions = (part.get("instructions", ()) if kind == "instruction-baseline"
                        else [part] if kind == "instruction-delta" else ())
        if isinstance(instructions, list):
            for instruction in instructions:
                if not isinstance(instruction, dict):
                    continue
                if instruction.get("part_kind") not in {"instruction", "instruction-digest", "instruction-delta"}:
                    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
                identity = instruction.get("id")
                if isinstance(identity, str) and _INSTRUCTION_ID_ADAPTER.validate_python(identity) is None:
                    raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        if kind in _CORRELATED_PARTS and "tool_call_id" not in part:
            raise ValueError("durable tool correlation is required")
        if kind == "retry-prompt":
            restored = _RETRY_ADAPTER.validate_python(part)
            _require_preserved_fields(part, _RETRY_ADAPTER.dump_python(restored, mode="json", warnings="error"))
            patches.append((pi, "retry", restored))
            wire["parts"][pi] = {"part_kind": "user-prompt", "content": ""}
        elif kind == "model-settings-delta":
            patches.append((pi, "settings", _recorded_settings(part["settings"])))
            part["settings"] = {}
        elif kind == _SUMMARY_KIND:
            part["part_kind"] = "system-prompt"
            patches.append((pi, "summary", None))
        elif kind == "user-prompt" and isinstance(part.get("content"), list):
            part["content"] = list(part["content"])
            for ci, item in enumerate(part["content"]):
                if isinstance(item, dict) and item.get("kind") == _HOST_CONTENT:
                    if set(item) != {"kind", "value"}:
                        raise ValueError("host content envelope is invalid")
                    restored = decode_user_content_item(item["value"])
                    if not isinstance(restored, (ReferencedContent, RetiredReferencedContent)):
                        raise ValueError("host content must be a reference descriptor")
                    patches.append((pi, ci, restored))
                    part["content"][ci] = {"kind": "text-content", "content": ""}
        elif kind in {"tool-return", "builtin-tool-return"}:
            content = decode_tool_return_content(part["content"])
            patches.append((pi, "tool", (content, part.get("tool_kind"))))
            part["content"] = ""
            part["tool_kind"] = None
    try:
        message, = ModelMessagesTypeAdapter.validate_python([wire])
    except ValidationError as error:
        code = (ErrorCode.STORAGE_VERSION_UNSUPPORTED
                if all(item["type"] == "union_tag_invalid" for item in error.errors(include_input=False, include_context=False))
                else ErrorCode.STORAGE_INTEGRITY_ERROR)
        raise AIError(code) from None
    # Global extra=forbid would reject supported provider usage extensions.
    _require_preserved_fields(wire, ModelMessagesTypeAdapter.dump_python([message], mode="json", warnings="error")[0])
    for pi, target, restored in patches:
        part = message.parts[pi]
        if target == "summary":
            message.parts[pi] = SummaryPromptPart(**{
                field.name: getattr(part, field.name) for field in fields(SystemPromptPart) if field.init
            })
        elif target == "retry":
            message.parts[pi] = restored
        elif target == "settings":
            part.settings = restored
        elif target == "tool":
            message.parts[pi] = _tool_return_with_content(part, *restored)
        else:
            part.content[target] = restored
    for index, usage in attempt_usages:
        message.failed_attempts[index] = replace(message.failed_attempts[index], usage=usage)
    _validate_message_facts(message)
    return message


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


def _encode_datetime(value: datetime | None) -> JsonValue:
    if value is None:
        return None
    if not isinstance(value, datetime):
        raise TypeError("message timestamp must be datetime")
    encoded = value.isoformat()
    return encoded[:-6] + "Z" if encoded.endswith("+00:00") else encoded




__all__ = ['RetiredReferencedContent', 'SummaryPromptPart', 'UnsupportedUserContentKind', 'binary_attachment_text', 'binary_content_usage', 'bind_tool_return_attachments', 'combine_model_message_payloads', 'decode_model_message_payloads', 'decode_model_messages', 'decode_tool_return_content', 'decode_user_content_item', 'encode_model_messages', 'encode_tool_return_content', 'encode_user_content_item', 'file_tool_result_metadata', 'freeze_model_messages', 'has_binary_attachment_metadata', 'input_attachment_views', 'is_binary_attachment_text', 'model_attempts_details', 'model_attempts_usage', 'model_input_content', 'model_response_usage', 'project_model_response_attempts', 'project_provider_messages', 'project_transient_binary_content', 'rehydrate_deferred_tool_results', 'request_attachment_facts', 'stable_json_snapshot', 'tool_return_content_digest', 'without_history_system_prompts']

from ._attachment import (binary_attachment_text, bind_tool_return_attachments, file_tool_result_metadata, has_binary_attachment_metadata, input_attachment_views, is_binary_attachment_text, request_attachment_facts)

from ._user_content_codec import (RetiredReferencedContent, UnsupportedUserContentKind, decode_user_content_item, encode_user_content_item)

from ._json_snapshot import (stable_json_snapshot)

from ._tool_return_codec import (decode_tool_return_content, encode_tool_return_content, rehydrate_deferred_tool_results, tool_return_content_digest)
