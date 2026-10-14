#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured file-input support and validation shared by admission and requests."""

import codecs
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Literal

from pydantic_ai.messages import (
    AudioUrl, BinaryContent, DocumentUrl, ImageUrl, ModelRequest,
    ToolReturnPart, UploadedFile, UserPromptPart, VideoUrl,
)

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ._openai_input import openai_input_transport

if TYPE_CHECKING:
    from ._contract import ModelBinding

ModelInputStatus = Literal["supported", "unsupported", "unknown"]
ModelInputTransport = Literal["text-inline", "file-block", "image", "audio"]
ModelInputKind = Literal["binary", "image-url", "audio-url", "document-url", "video-url", "uploaded"]
UnknownModelInputs = Literal["reject", "passthrough"]

_STATUSES = frozenset({"supported", "unsupported", "unknown"})
_TRANSPORTS = frozenset({"text-inline", "file-block", "image", "audio"})
_INPUT_KINDS = frozenset({"binary", "image-url", "audio-url", "document-url", "video-url", "uploaded"})


@dataclass(frozen=True, slots=True)
class ModelInputCapability:
    """Declare MIME support; the adapter resolves an omitted transport at capture.

    An explicit transport distinguishes known support for an alternate encoding,
    such as uploaded text files. Unsupported MIME declarations deny every encoding.
    """

    media_type: str
    status: ModelInputStatus = "supported"
    max_bytes: int | None = None
    transport: ModelInputTransport | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        object.__setattr__(self, "media_type", _media_type(self.media_type))
        if (self.transport is not None and self.transport not in _TRANSPORTS) or self.status not in _STATUSES:
            raise ValueError("model input transport or status is invalid")
        if self.max_bytes is not None and (type(self.max_bytes) is not int or self.max_bytes <= 0):
            raise ValueError("model input max_bytes must be a positive integer")

    def to_payload(self) -> dict[str, JsonValue]:
        if self.transport is None and self.status != "unsupported":
            raise ValueError("model input transport must be resolved before serialization")
        result: dict[str, JsonValue] = {
            "media_type": self.media_type, "status": self.status,
        }
        if self.transport is not None:
            result["transport"] = self.transport
        if self.max_bytes is not None:
            result["max_bytes"] = self.max_bytes
        return result

    @classmethod
    def from_payload(cls, value: object) -> "ModelInputCapability":
        if not isinstance(value, Mapping) or not {"media_type", "status"}.issubset(value):
            raise ValueError("model input capability is invalid")
        transport = value.get("transport")
        if transport is None and value["status"] != "unsupported":
            raise ValueError("captured model input transport must be resolved")
        return cls(value["media_type"], value["status"], value.get("max_bytes"), transport=transport)


@dataclass(frozen=True, slots=True)
class ModelInputDecision:
    """A declaration decision; allowed unknown inputs remain explicitly unverified."""

    media_type: str
    input_kind: ModelInputKind
    transport: ModelInputTransport | None
    status: ModelInputStatus
    allowed: bool
    verified: bool
    reason: str | None = None

    def require_allowed(self) -> "ModelInputDecision":
        if not self.allowed:
            details: dict[str, JsonValue] = {"reason": self.reason or "model_input_not_supported"}
            if self.reason != "image_input_not_supported":
                details.update(media_type=self.media_type, input_kind=self.input_kind, transport=self.transport, status=self.status)
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False, safe_details=details)
        return self


def normalize_input_capabilities(
    values: Sequence[ModelInputCapability], *,
    resolve_transport: Callable[[str], str | None] | None = None,
) -> tuple[ModelInputCapability, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("model input capabilities must be a sequence")
    if any(not isinstance(item, ModelInputCapability) for item in values):
        raise ValueError("model input capability declaration is invalid")
    resolved: list[ModelInputCapability] = []
    for item in values:
        if item.status == "unsupported":
            resolved.append(replace(item, transport=None))
            continue
        transport = item.transport
        if transport is None and resolve_transport is not None:
            transport = resolve_transport(item.media_type)
        if transport is None:
            raise ValueError("MIME has no default input transport; an explicit transport is required")
        resolved.append(replace(item, transport=transport))
    result = tuple(sorted(resolved, key=lambda item: (item.media_type, item.transport or "")))
    if len({(item.media_type, item.transport) for item in result}) != len(result):
        raise ValueError("model input capabilities contain duplicate MIME/transport declarations")
    return result


def decide_model_input(
    binding: "ModelBinding", media_type: str, *, input_kind: ModelInputKind = "binary", size: int | None = None,
) -> ModelInputDecision:
    """Inspect captured support without reading a file, fetching a URL or contacting a provider."""
    media_type = _media_type(media_type)
    if input_kind not in _INPUT_KINDS:
        raise ValueError("model input kind is invalid")
    if size is not None and (type(size) is not int or size < 0):
        raise ValueError("model input size must be a non-negative integer")
    contract = binding.contract
    declaration = contract.get("inputs")
    if "inputs" not in contract:
        if not binding.vision and (media_type.startswith("image/") or input_kind == "image-url"):
            return ModelInputDecision(media_type, input_kind, None, "unsupported", False, True, "image_input_not_supported")
        rules, unknown = (), "passthrough"
    else:
        try:
            if not isinstance(declaration, Mapping) or not {"endpoint", "unknown", "capabilities"}.issubset(declaration):
                raise ValueError("model input declaration is invalid")
            if declaration["endpoint"] != "openai-chat" or binding.provider != "openai":
                raise ValueError("model input endpoint is unsupported")
            unknown = declaration["unknown"]
            if unknown not in {"reject", "passthrough"}:
                raise ValueError("model unknown input policy is invalid")
            items = declaration["capabilities"]
            if not isinstance(items, (tuple, list)):
                raise ValueError("model input capabilities are invalid")
            rules = normalize_input_capabilities(tuple(ModelInputCapability.from_payload(item) for item in items))
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.MODEL_CONFIG_INVALID, retryable=False, safe_details={"reason": "model_input_contract_invalid"}) from error
    transport = None
    if binding.provider == "openai":
        model = binding.model_identity.removeprefix("openai:")
        transport = openai_input_transport(media_type, input_kind, model)
        if transport is None:
            return ModelInputDecision(media_type, input_kind, None, "unsupported", False, True, "input_encoding_not_supported")
    rule = next((item for item in rules if item.media_type == media_type and item.status == "unsupported"), None)
    if rule is not None:
        return ModelInputDecision(media_type, input_kind, transport, "unsupported", False, True, "model_input_not_supported")
    rule = next((item for item in rules if item.media_type == media_type and item.transport == transport), None)
    status = rule.status if rule is not None else "unknown"
    if rule is not None and rule.max_bytes is not None:
        if size is None or size > rule.max_bytes:
            reason = "model_input_size_unknown" if size is None else "model_input_size_exceeded"
            return ModelInputDecision(media_type, input_kind, transport, status, False, False, reason)
    allowed = status == "supported" or (status == "unknown" and unknown == "passthrough")
    reason = None if allowed else ("model_input_unknown" if status == "unknown" else "model_input_not_supported")
    return ModelInputDecision(media_type, input_kind, transport, status, allowed, status == "supported", reason)


def validate_model_input(
    binding: "ModelBinding", media_type: str, *, input_kind: ModelInputKind = "binary", size: int | None = None,
) -> ModelInputDecision:
    """Validate one file's facts; SDK byte encoding is checked by validate_model_inputs."""
    return decide_model_input(binding, media_type, input_kind=input_kind, size=size).require_allowed()


def validate_model_inputs(binding: "ModelBinding", value: object) -> tuple[ModelInputDecision, ...]:
    """Validate all concrete files in input/history, including URLs and uploaded handles.

    This does not fetch remote inputs or replace the caller's aggregate resource budgets.
    A configured byte limit rejects references whose byte size cannot be established.
    """
    decisions: list[ModelInputDecision] = []
    for item in _file_inputs(value):
        if isinstance(item, BinaryContent):
            kind, size = "binary", len(item.data)
        elif isinstance(item, ImageUrl):
            kind, size = "image-url", None
        elif isinstance(item, AudioUrl):
            kind, size = "audio-url", None
        elif isinstance(item, DocumentUrl):
            kind, size = "document-url", None
        elif isinstance(item, VideoUrl):
            kind, size = "video-url", None
        else:
            kind, size = "uploaded", None
        try:
            media_type = item.media_type
        except (KeyError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False, safe_details={"reason": "model_input_media_type_unknown"}) from error
        decision = validate_model_input(binding, media_type, input_kind=kind, size=size)
        if binding.provider == "openai":
            if isinstance(item, UploadedFile) and item.provider_name != "openai":
                _encoding_error(decision, "uploaded_file_provider_mismatch")
            if isinstance(item, (BinaryContent, AudioUrl, DocumentUrl)):
                if openai_input_transport(item.media_type, kind, binding.model_identity.removeprefix("openai:")) is None:
                    _encoding_error(decision, "input_encoding_not_supported")
            if isinstance(item, BinaryContent):
                if decision.transport == "text-inline":
                    try:
                        decoder = codecs.getincrementaldecoder("utf-8")()
                        for offset in range(0, len(item.data), 65536):
                            decoder.decode(item.data[offset:offset + 65536])
                        decoder.decode(b"", final=True)
                    except UnicodeDecodeError:
                        _encoding_error(decision, "text_input_not_utf8")
            if isinstance(item, ImageUrl) and item.force_download:
                try:
                    item.format
                except (KeyError, ValueError):
                    _encoding_error(decision, "input_encoding_not_supported")
        decisions.append(decision)
    return tuple(decisions)


def _encoding_error(decision: ModelInputDecision, reason: str) -> None:
    ModelInputDecision(decision.media_type, decision.input_kind, decision.transport, "unsupported", False, True, reason).require_allowed()


def _file_inputs(value: object) -> Iterator[BinaryContent | ImageUrl | AudioUrl | DocumentUrl | VideoUrl | UploadedFile]:
    if isinstance(value, (BinaryContent, ImageUrl, AudioUrl, DocumentUrl, VideoUrl, UploadedFile)):
        yield value
    elif isinstance(value, ModelRequest):
        for part in value.parts:
            if isinstance(part, (UserPromptPart, ToolReturnPart)):
                yield from _file_inputs(part.content)
    elif isinstance(value, Mapping):
        for item in value.values():
            yield from _file_inputs(item)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for item in value:
            yield from _file_inputs(item)


def _media_type(value: str) -> str:
    if not isinstance(value, str):
        raise ValueError("model input media_type must be a MIME type")
    value = value.strip().lower()
    if value.count("/") != 1 or any(not part for part in value.split("/")) or "*" in value or any(c.isspace() for c in value) or ";" in value:
        raise ValueError("model input media_type must be an exact MIME type")
    return value


__all__ = [
    "ModelInputCapability", "ModelInputDecision", "ModelInputStatus", "ModelInputTransport", "ModelInputKind", "UnknownModelInputs",
    "decide_model_input", "validate_model_input", "validate_model_inputs", "normalize_input_capabilities",
]
