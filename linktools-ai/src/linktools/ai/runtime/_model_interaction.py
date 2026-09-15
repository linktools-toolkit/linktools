#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Small in-process contracts for model interaction observation."""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

from pydantic import TypeAdapter
from pydantic_ai.messages import (
    ModelMessage,
    ModelResponse,
)
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
    """Return the chained digest for a logical message prefix."""
    value = _PREFIX_SEED
    for message in messages:
        value = hashlib.sha256(
            value + hashlib.sha256(encode_model_messages((message,))).digest()
        ).digest()
    return value.hex()


def extend_prefix_digest(prefix_digest: str, message: ModelMessage) -> str:
    """Extend a previously computed message prefix digest by one message."""
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
    """Reduce one model context to source spans and unique inline deltas."""
    source_values = tuple(source)
    projected_values = tuple(projected)
    signatures: dict[bytes, list[int]] = {}
    for index, message in enumerate(source_values):
        signatures.setdefault(_message_signature(message), []).append(index)
    used: set[int] = set()
    items: list[StagedContextItem] = []
    span_start: int | None = None
    span_end = 0

    def flush_span() -> None:
        nonlocal span_start
        if span_start is not None:
            items.append(StagedContextSpan(span_start, span_end))
            span_start = None

    for message in projected_values:
        candidates = signatures.get(_message_signature(message), ())
        source_index = next((i for i in candidates if i not in used), None)
        if source_index is None:
            flush_span()
            digest, size = intern_payload(encode_model_messages((message,)))
            items.append(StagedContextInline(digest, size))
            continue
        used.add(source_index)
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


def context_projection_to_durable(
    projection: StagedContextProjection,
    *,
    owner_id: str,
    source_domain: RuntimeDomain,
    payload: Callable[[str], bytes],
) -> ContextProjection:
    """Convert staging references after the durable owner is known."""
    items = []
    for item in projection.items:
        if isinstance(item, StagedContextSpan):
            items.append(
                TranscriptSpanRef(
                    source_domain,
                    owner_id,
                    item.start,
                    item.end,
                )
            )
            continue
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
    """Create the versioned, provider-object-free request envelope."""
    settings = _json_snapshot({} if model_settings is None else model_settings)
    parameter_value = _parameters_snapshot(parameters)
    value: dict[str, JsonValue] = {
        "version": _ENVELOPE_VERSION,
        "streaming": streaming,
        "settings": settings,
        "parameters": parameter_value,
    }
    return value, canonical_json_bytes(value)


def project_public_messages(messages: Sequence[ModelMessage]) -> list[JsonValue]:
    """Project model messages without exposing binary bodies."""
    values: list[JsonValue] = []
    for message in messages:
        value = _json_snapshot(message)
        _sanitize_binary(value)
        values.append(value)
    return values


def model_identity(model: object) -> dict[str, str]:
    return {
        "route_id": str(getattr(model, "model_id", "")),
        "system": str(getattr(model, "system", "")),
        "model_name": str(getattr(model, "model_name", "")),
    }


def model_response_projection(response: ModelResponse) -> JsonValue:
    value = _json_snapshot(response)
    _sanitize_binary(value)
    return value


def _message_signature(message: ModelMessage) -> bytes:
    return hashlib.sha256(encode_model_messages((message,))).digest()


def _json_snapshot(value: object, *, exclude: set[str] | None = None) -> JsonValue:
    try:
        dumped = TypeAdapter(object).dump_python(value, mode="json")
    except (TypeError, ValueError):
        dumped = {}
    if exclude and isinstance(dumped, dict):
        dumped = {key: item for key, item in dumped.items() if key not in exclude}
    return cast(JsonValue, dumped)


def _parameters_snapshot(parameters: ModelRequestParameters) -> JsonValue:
    value: dict[str, JsonValue] = {}
    for name in (
        "function_tools",
        "tool_visibility",
        "revealed_tool_names",
        "output_mode",
        "output_object",
        "output_tools",
        "prompted_output_template",
        "allow_text_output",
        "allow_image_output",
        "instruction_parts",
        "thinking",
    ):
        value[name] = _json_snapshot(getattr(parameters, name))
    native_tools: list[JsonValue] = []
    for tool in parameters.native_tools:
        snapshot = _json_snapshot(tool)
        if not isinstance(snapshot, Mapping):
            snapshot = {}
        native_tools.append(
            {
                "kind": str(getattr(tool, "kind", "native")),
                "name": str(getattr(tool, "name", type(tool).__name__)),
                **dict(snapshot),
            }
        )
    value["native_tools"] = native_tools
    return value


def _sanitize_binary(value: JsonValue) -> None:
    if isinstance(value, list):
        for item in value:
            _sanitize_binary(item)
        return
    if not isinstance(value, dict):
        return
    if value.get("kind") == "binary":
        data = value.pop("data", None)
        if isinstance(data, str):
            try:
                raw = base64.b64decode(data, validate=True)
            except ValueError:
                raw = data.encode("utf-8")
            value["size"] = len(raw)
            value["digest"] = hashlib.sha256(raw).hexdigest()
        return
    for item in value.values():
        _sanitize_binary(item)


__all__ = [
    "StagedContextInline",
    "StagedContextItem",
    "StagedContextProjection",
    "StagedContextSpan",
    "StagedModelInteraction",
    "build_context_projection",
    "extend_prefix_digest",
    "message_prefix_digest",
    "model_identity",
    "model_response_projection",
    "project_public_messages",
    "request_envelope",
]
