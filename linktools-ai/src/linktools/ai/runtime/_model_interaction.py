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

from ..core import JsonValue, UsageMetrics, canonical_json_bytes
from ..storage import StoredPayload
from ._message import encode_model_messages
from .state._contracts import (
    ContextProjection,
    InlineContextBlock,
    RuntimePayloadRef,
    TranscriptSpanRef,
)
from .state._plan import RuntimeDomain

_PREFIX_SEED = hashlib.sha256(b"linktools.model-context-prefix.v1").digest()
_ENVELOPE_VERSION = 1
_INLINE_SOURCE_DIGEST = "0" * 64


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


StagedContextItem = StagedContextSpan | StagedContextInline


@dataclass(frozen=True, slots=True)
class StagedContextProjection:
    source_message_count: int
    source_prefix_digest: str
    items: tuple[StagedContextItem, ...]

    def __post_init__(self) -> None:
        if (
            self.source_message_count < 0
            or len(self.source_prefix_digest) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.source_prefix_digest
            )
            or any(
                not isinstance(item, (StagedContextSpan, StagedContextInline))
                for item in self.items
            )
        ):
            raise ValueError("staged context projection is invalid")


@dataclass(frozen=True, slots=True)
class StagedModelInteraction:
    run_id: str
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

    def __post_init__(self) -> None:
        if (
            not self.run_id
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


PayloadIntern = Callable[[bytes], tuple[str, int]]


def message_prefix_digest(messages: Sequence[ModelMessage]) -> str:
    value = _PREFIX_SEED
    for message in messages:
        value = hashlib.sha256(
            value + hashlib.sha256(encode_model_messages((message,))).digest()
        ).digest()
    return value.hex()


def extend_prefix_digest(prefix_digest: str, message: ModelMessage) -> str:
    try:
        prefix = bytes.fromhex(prefix_digest)
    except ValueError as error:
        raise ValueError("message prefix digest is invalid") from error
    return hashlib.sha256(
        prefix + hashlib.sha256(encode_model_messages((message,))).digest()
    ).hexdigest()


def build_context_projection(
    source: Sequence[ModelMessage],
    projected: Sequence[ModelMessage],
    intern_payload: PayloadIntern,
    *,
    source_prefix_digest: str | None = None,
) -> StagedContextProjection:
    source_values = tuple(source)
    signatures: dict[bytes, list[int]] = {}
    for index, message in enumerate(source_values):
        signatures.setdefault(_message_signature(message), []).append(index)
    next_candidate: dict[bytes, int] = {}
    items: list[StagedContextItem] = []
    span_start: int | None = None
    span_end = 0

    def flush_span() -> None:
        nonlocal span_start
        if span_start is not None:
            items.append(StagedContextSpan(span_start, span_end))
            span_start = None

    for message in projected:
        signature = _message_signature(message)
        candidates = signatures.get(signature, ())
        candidate_index = next_candidate.get(signature, 0)
        source_index = (
            candidates[candidate_index]
            if candidate_index < len(candidates)
            else None
        )
        if source_index is None:
            flush_span()
            items.append(
                StagedContextInline(*intern_payload(encode_model_messages((message,))))
            )
            continue
        next_candidate[signature] = candidate_index + 1
        if span_start is not None and source_index == span_end:
            span_end += 1
        else:
            flush_span()
            span_start = source_index
            span_end = source_index + 1
    flush_span()
    return StagedContextProjection(
        len(source_values),
        message_prefix_digest(source_values)
        if source_prefix_digest is None
        else source_prefix_digest,
        tuple(items),
    )


def build_inline_context_projection(
    messages: Sequence[ModelMessage],
    intern_payload: PayloadIntern,
) -> StagedContextProjection:
    return StagedContextProjection(
        0,
        _INLINE_SOURCE_DIGEST,
        tuple(
            StagedContextInline(*intern_payload(encode_model_messages((message,))))
            for message in messages
        ),
    )


def context_projection_to_durable(
    projection: StagedContextProjection,
    *,
    owner_id: str,
    source_domain: RuntimeDomain,
    payload: Callable[[str], bytes],
) -> ContextProjection:
    items = []
    for item in projection.items:
        if isinstance(item, StagedContextSpan):
            items.append(
                TranscriptSpanRef(source_domain, owner_id, item.start, item.end)
            )
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


def _message_signature(message: ModelMessage) -> bytes:
    return hashlib.sha256(encode_model_messages((message,))).digest()


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
    "StagedModelInteraction",
    "build_context_projection",
    "build_inline_context_projection",
    "extend_prefix_digest",
    "message_prefix_digest",
    "model_identity",
    "model_response_projection",
    "project_public_messages",
    "request_envelope",
]
