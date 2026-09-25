#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small in-process contracts for model interaction observation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields, is_dataclass
from typing import cast

from pydantic import TypeAdapter
from pydantic_ai.messages import BinaryContent, ModelMessage, ModelResponse
from pydantic_ai.models import ModelRequestParameters
from pydantic_ai.settings import ModelSettings

from ..core import (
    JsonValue,
    UsageMetrics,
    canonical_json_bytes,
    normalize_json_value,
)
from ..storage import StoredPayload
from ._message import encode_model_messages
from .state._contracts import (
    ContextProjection,
    InlineContextBlock,
    RuntimePayloadRef,
    TranscriptMessageRef,
    TranscriptSpanRef,
)
from .state._plan import RuntimeDomain

_ENVELOPE_VERSION = 1


@dataclass(frozen=True, slots=True)
class StagedContextSpan:
    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError("staged context span is invalid")


@dataclass(frozen=True, slots=True)
class StagedContextInline:
    payload_digest: str
    size: int

    def __post_init__(self) -> None:
        if (
            len(self.payload_digest) != 64
            or any(value not in "0123456789abcdef" for value in self.payload_digest)
            or self.size < 0
        ):
            raise ValueError("staged context inline is invalid")


StagedContextItem = StagedContextSpan | StagedContextInline | TranscriptSpanRef
StagedContextSource = TranscriptMessageRef | int | None


@dataclass(frozen=True, slots=True)
class StagedContextProjection:
    items: tuple[StagedContextItem, ...]

    def __post_init__(self) -> None:
        if any(
            not isinstance(
                item,
                (StagedContextSpan, StagedContextInline, TranscriptSpanRef),
            )
            for item in self.items
        ):
            raise ValueError("staged context projection is invalid")


@dataclass(frozen=True, slots=True)
class StagedModelInteraction:
    agent_run_id: str
    step_index: int
    request_sequence: int
    purpose: str
    output_retry_index: int | None
    model: Mapping[str, str]
    request_context: StagedContextProjection
    request_envelope_digest: str
    response_context: StagedContextProjection | None
    status: str
    error_code: str | None
    duration_ns: int
    usage: UsageMetrics | None
    attachments: tuple[Mapping[str, JsonValue], ...] = ()

    def __post_init__(self) -> None:
        if (
            not self.agent_run_id
            or self.step_index < 0
            or self.request_sequence < 1
            or self.purpose not in {"agent", "compaction"}
            or self.status not in {"SUCCEEDED", "FAILED", "CANCELLED"}
            or self.duration_ns < 0
            or len(self.request_envelope_digest) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.request_envelope_digest
            )
            or self.status == "SUCCEEDED"
            and self.response_context is None
            or self.status != "SUCCEEDED"
            and self.response_context is not None
            or self.status == "FAILED"
            and not self.error_code
        ):
            raise ValueError("staged model interaction is invalid")
        if not isinstance(self.request_context, StagedContextProjection):
            raise TypeError("staged request context is invalid")
        if self.response_context is not None and not isinstance(
            self.response_context,
            StagedContextProjection,
        ):
            raise TypeError("staged response context is invalid")
        object.__setattr__(self, "model", dict(self.model))
        normalized_attachments: list[Mapping[str, JsonValue]] = []
        for attachment in self.attachments:
            if not isinstance(attachment, Mapping):
                raise TypeError("staged model attachment fact is invalid")
            try:
                normalized = normalize_json_value(dict(attachment))
            except (TypeError, ValueError) as error:
                raise ValueError("staged model attachment fact is invalid") from error
            if not isinstance(normalized, dict):
                raise ValueError("staged model attachment fact is invalid")
            normalized_attachments.append(normalized)
        object.__setattr__(self, "attachments", tuple(normalized_attachments))


PayloadIntern = Callable[[bytes], tuple[str, int]]


def build_context_projection(
    source: Sequence[ModelMessage],
    projected: Sequence[ModelMessage],
    intern_payload: PayloadIntern,
    *,
    source_refs: Sequence[StagedContextSource] | None = None,
    source_keys: Sequence[bytes] | None = None,
) -> StagedContextProjection:
    source_values = tuple(source)
    projected_values = tuple(projected)
    refs = (
        tuple(range(len(source_values)))
        if source_refs is None
        else tuple(source_refs)
    )
    if len(refs) != len(source_values):
        raise ValueError("context source references do not match source messages")
    keys = (
        tuple(_message_key(message) for message in source_values)
        if source_keys is None
        else tuple(source_keys)
    )
    if len(keys) != len(source_values):
        raise ValueError("context source keys do not match source messages")

    items: list[StagedContextItem] = []

    def append_source(ref: StagedContextSource) -> bool:
        if isinstance(ref, int) and not isinstance(ref, bool):
            if ref < 0:
                raise ValueError("local transcript source index cannot be negative")
            if items and isinstance(items[-1], StagedContextSpan):
                previous = items[-1]
                if previous.end == ref:
                    items[-1] = StagedContextSpan(previous.start, ref + 1)
                    return True
            items.append(StagedContextSpan(ref, ref + 1))
            return True
        if isinstance(ref, TranscriptMessageRef):
            if items and isinstance(items[-1], TranscriptSpanRef):
                previous = items[-1]
                if (
                    previous.source_domain is ref.source_domain
                    and previous.owner_id == ref.owner_id
                    and previous.end == ref.message_index
                ):
                    items[-1] = TranscriptSpanRef(
                        previous.source_domain,
                        previous.owner_id,
                        previous.start,
                        ref.message_index + 1,
                    )
                    return True
            items.append(
                TranscriptSpanRef(
                    ref.source_domain,
                    ref.owner_id,
                    ref.message_index,
                    ref.message_index + 1,
                )
            )
            return True
        if ref is not None:
            raise TypeError("context source reference is invalid")
        return False

    if projected_values == source_values:
        for message, ref in zip(projected_values, refs, strict=True):
            if append_source(ref):
                continue
            items.append(
                StagedContextInline(
                    *intern_payload(encode_model_messages((message,)))
                )
            )
        return StagedContextProjection(tuple(items))

    signatures: dict[bytes, list[int]] = {}
    for index, key in enumerate(keys):
        signatures.setdefault(key, []).append(index)
    projected_keys = tuple(
        _message_key(message) for message in projected_values
    )
    for message, key in zip(
        projected_values,
        projected_keys,
        strict=True,
    ):
        candidates = signatures.get(key, ())
        source_index = candidates[0] if len(candidates) == 1 else None
        if source_index is not None and append_source(refs[source_index]):
            continue
        items.append(
            StagedContextInline(*intern_payload(encode_model_messages((message,))))
        )
    return StagedContextProjection(tuple(items))


def build_inline_context_projection(
    messages: Sequence[ModelMessage],
    intern_payload: PayloadIntern,
) -> StagedContextProjection:
    return StagedContextProjection(
        tuple(
            StagedContextInline(*intern_payload(encode_model_messages((message,))))
            for message in messages
        )
    )

def context_projection_to_durable(
    projection: StagedContextProjection,
    *,
    owner_id: str,
    source_domain: RuntimeDomain,
    payload: Callable[[str], bytes],
    local_message_base: int = 0,
) -> ContextProjection:
    if local_message_base < 0:
        raise ValueError("local message base cannot be negative")
    items = []
    for item in projection.items:
        if isinstance(item, StagedContextSpan):
            items.append(
                TranscriptSpanRef(
                    source_domain,
                    owner_id,
                    local_message_base + item.start,
                    local_message_base + item.end,
                )
            )
        elif isinstance(item, TranscriptSpanRef):
            items.append(item)
        else:
            items.append(
                InlineContextBlock(
                    RuntimePayloadRef(
                        StoredPayload.inline_bytes(payload(item.payload_digest)),
                        source_domain,
                    )
                )
            )
    return ContextProjection(tuple(items))


def request_envelope(
    *,
    model_settings: ModelSettings | None,
    parameters: ModelRequestParameters,
    streaming: bool,
) -> tuple[JsonValue, bytes]:
    value: dict[str, JsonValue] = {
        "version": _ENVELOPE_VERSION,
        "streaming": streaming,
        "settings": _json_snapshot({} if model_settings is None else model_settings),
        "parameters": _parameters_snapshot(parameters),
    }
    return value, canonical_json_bytes(value)


def project_public_messages(messages: Sequence[ModelMessage]) -> list[JsonValue]:
    values: list[JsonValue] = []
    for message in messages:
        value = _json_snapshot(message)
        _sanitize_binary(message, value)
        values.append(value)
    return values


def model_identity(model: object, *, route_id: str | None = None) -> dict[str, str]:
    return {
        "route_id": str(route_id or getattr(model, "model_id", "")),
        "system": str(getattr(model, "system", "")),
        "model_name": str(getattr(model, "model_name", "")),
    }


def model_response_projection(response: ModelResponse) -> JsonValue:
    value = _json_snapshot(response)
    _sanitize_binary(response, value)
    return value


def _message_key(message: ModelMessage) -> bytes:
    return encode_model_messages((message,))


def _json_snapshot(value: object) -> JsonValue:
    return cast(JsonValue, TypeAdapter(object).dump_python(value, mode="json"))


def _parameters_snapshot(parameters: ModelRequestParameters) -> JsonValue:
    value: dict[str, JsonValue] = {}
    for name in (
        "function_tools",
        "tool_visibility",
        "revealed_tool_names",
        "deferred_capability_ids",
        "output_mode",
        "output_object",
        "output_tools",
        "prompted_output_template",
        "allow_text_output",
        "allow_image_output",
        "instruction_parts",
        "thinking",
    ):
        raw = getattr(parameters, name)
        if isinstance(raw, (set, frozenset)):
            raw = sorted(raw)
        value[name] = _json_snapshot(raw)
    native_tools: list[JsonValue] = []
    for tool in parameters.native_tools:
        snapshot = _json_snapshot(tool)
        if not isinstance(snapshot, Mapping):
            raise TypeError("native tool observation is not a mapping")
        native_tools.append(
            {
                "kind": str(getattr(tool, "kind", "native")),
                "name": str(getattr(tool, "name", type(tool).__name__)),
                **dict(snapshot),
            }
        )
    value["native_tools"] = native_tools
    return value


def _sanitize_binary(source: object, value: JsonValue) -> None:
    if isinstance(source, BinaryContent):
        if not isinstance(value, dict):
            raise TypeError("binary content observation is not a mapping")
        value.pop("data", None)
        value["size"] = len(source.data)
        value["digest"] = hashlib.sha256(source.data).hexdigest()
        return
    if isinstance(source, Mapping) and isinstance(value, dict):
        for key, item in source.items():
            if isinstance(key, str) and key in value:
                _sanitize_binary(item, value[key])
        return
    if isinstance(source, (list, tuple)) and isinstance(value, list):
        for item, projected in zip(source, value, strict=False):
            _sanitize_binary(item, projected)
        return
    if is_dataclass(source) and isinstance(value, dict):
        for field in fields(source):
            if field.name in value:
                _sanitize_binary(getattr(source, field.name), value[field.name])


__all__ = [
    "StagedContextInline",
    "StagedContextItem",
    "StagedContextProjection",
    "StagedContextSpan",
    "StagedContextSource",
    "StagedModelInteraction",
    "build_context_projection",
    "build_inline_context_projection",
    "model_identity",
    "model_response_projection",
    "project_public_messages",
    "request_envelope",
]
