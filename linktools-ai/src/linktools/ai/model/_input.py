#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Captured MIME allowlists and SDK input validation."""

import codecs
from collections.abc import Iterator, Mapping, Sequence
from typing import TYPE_CHECKING, get_args

from pydantic_ai.messages import (
    AudioMediaType, DocumentMediaType, ImageMediaType, VideoMediaType,
    AudioUrl, BinaryContent, DocumentUrl, ImageUrl, ModelRequest,
    ToolReturnPart, UploadedFile, UserPromptPart, VideoUrl,
)

from ..errors import AIError, ErrorCode
from ._openai_input import openai_input_transport

if TYPE_CHECKING:
    from ._contract import ModelBinding

_INPUT_KINDS = frozenset({"binary", "image-url", "audio-url", "document-url", "video-url", "uploaded"})


def expand_input_types(values: Sequence[str]) -> tuple[str, ...]:
    """Resolve authoring categories once; captured contracts contain only MIME patterns."""
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("input_types must be a sequence of strings")
    categories = {
        "image": ImageMediaType, "audio": AudioMediaType,
        "document": DocumentMediaType, "video": VideoMediaType,
    }
    mime_types: list[str] = []
    for value in values:
        if not isinstance(value, str):
            raise ValueError("input_types must contain only strings")
        if value in categories:
            mime_types.extend(get_args(categories[value]))
        elif value.startswith("mime:"):
            mime_types.append(value[len("mime:"):])
        else:
            raise ValueError("input_types entries must be image, audio, document, video, or mime:<MIME type>")
    return normalize_input_mime_types(mime_types)


def normalize_input_mime_types(values: Sequence[str]) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise ValueError("input_mime_types must be a sequence of MIME strings")
    return tuple(sorted({_mime_type(value, allow_family=True) for value in values}))


def validate_model_input(
    binding: "ModelBinding", mime_type: str, *, input_kind: str = "binary", size: int | None = None,
) -> None:
    """Validate MIME permission and encoding without reading or fetching input bytes."""
    mime_type = _mime_type(mime_type)
    if input_kind not in _INPUT_KINDS:
        raise ValueError("model input kind is invalid")
    if size is not None and (type(size) is not int or size < 0):
        raise ValueError("model input size must be a non-negative integer")
    contract = binding.contract
    try:
        rules = normalize_input_mime_types(contract["input_mime_types"])
    except (KeyError, TypeError, ValueError) as error:
        raise AIError(ErrorCode.MODEL_CONFIG_INVALID, retryable=False,
                      safe_details={"reason": "model_input_contract_invalid"}) from error
    if mime_type not in rules and mime_type.split("/", 1)[0] + "/*" not in rules:
        _input_error(mime_type, input_kind, "model_input_not_supported")
    if _input_transport(binding, mime_type, input_kind) is None:
        _input_error(mime_type, input_kind, "input_encoding_not_supported")


def validate_model_inputs(binding: "ModelBinding", value: object) -> None:
    """Validate concrete inputs, including SDK encoding, without fetching remote files."""
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
            mime_type = item.media_type
        except (KeyError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False,
                          safe_details={"reason": "model_input_mime_type_unknown"}) from error
        validate_model_input(binding, mime_type, input_kind=kind, size=size)
        if binding.provider in {"openai", "anthropic"}:
            if isinstance(item, UploadedFile) and item.provider_name != binding.provider:
                _input_error(mime_type, kind, "uploaded_file_provider_mismatch")
            transport = _input_transport(binding, mime_type, kind)
            if isinstance(item, (BinaryContent, AudioUrl, DocumentUrl)) and transport is None:
                _input_error(mime_type, kind, "input_encoding_not_supported")
            if isinstance(item, BinaryContent) and transport == "text-inline":
                try:
                    decoder = codecs.getincrementaldecoder("utf-8")()
                    for offset in range(0, len(item.data), 65536):
                        decoder.decode(item.data[offset:offset + 65536])
                    decoder.decode(b"", final=True)
                except UnicodeDecodeError:
                    _input_error(mime_type, kind, "text_input_not_utf8")
            if isinstance(item, ImageUrl) and item.force_download:
                try:
                    item.format
                except (KeyError, ValueError):
                    _input_error(mime_type, kind, "input_encoding_not_supported")



def _input_transport(binding: "ModelBinding", mime_type: str, kind: str) -> str | None:
    contract = binding.contract
    api = contract.get("api", "openai-chat" if binding.provider == "openai" else None)
    if api == "openai-chat":
        return openai_input_transport(mime_type, kind, binding.model_identity.removeprefix("openai:"),
                                      contract.get("profile"))
    if api == "openai-responses":
        if kind in {"uploaded", "image-url"}:
            return "file-block"
        if kind in {"audio-url", "document-url"}:
            return "file-block"
        probe = BinaryContent(b"", media_type=mime_type)
        if kind == "binary" and (probe.is_image or probe.is_document):
            try:
                probe.format
            except (KeyError, ValueError):
                return None
            return "file-block"
        return None
    if api == "anthropic-messages":
        if kind == "uploaded":
            return "file-block" if mime_type.startswith(("image/", "text/", "application/")) else None
        if kind in {"binary", "image-url"} and mime_type in {"image/jpeg", "image/png", "image/gif", "image/webp"}:
            return "image"
        if kind in {"binary", "document-url"} and mime_type in {"application/pdf", "text/plain"}:
            return "text-inline" if mime_type == "text/plain" else "file-block"
        return None
    # Custom bindings own their SDK encoding constraints.
    return "custom"

def _input_error(mime_type: str, input_kind: str, reason: str) -> None:
    raise AIError(ErrorCode.REQUEST_FIELD_INVALID, retryable=False,
                  safe_details={"reason": reason, "mime_type": mime_type, "input_kind": input_kind})


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


def _mime_type(value: str, *, allow_family: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError("model input mime_type must be a MIME type")
    value = value.strip().lower()
    parts = value.split("/")
    if len(parts) != 2 or any(not part for part in parts) or any(c.isspace() for c in value) or ";" in value or ":" in value:
        raise ValueError("model input mime_type must be a MIME type")
    if "*" in value and not (allow_family and parts[1] == "*" and "*" not in parts[0]):
        raise ValueError("model input mime_type must be an exact MIME type or a type/* declaration")
    return value


__all__ = ["validate_model_input", "validate_model_inputs"]
