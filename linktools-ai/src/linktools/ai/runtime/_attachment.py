#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned attachment occurrence identity and request facts."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import cast

from pydantic_ai.messages import (
    AudioUrl,
    BaseToolReturnPart,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    ToolReturn,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

from ..core import JsonValue, canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode

_URL_TYPES = (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)


def input_attachment_views(
    value: str | Sequence[UserContent],
    files: Sequence[Mapping[str, JsonValue]] = (),
) -> tuple[dict[str, JsonValue], ...]:
    """Project accepted input occurrences without retaining raw external references."""
    result: list[dict[str, JsonValue]] = []
    position = 0
    if not isinstance(value, str):
        for item in value:
            descriptor = _content_descriptor(item)
            if descriptor is None:
                continue
            result.append(
                _accepted_occurrence(
                    descriptor,
                    source=cast(str, descriptor["source"]),
                    position=position,
                )
            )
            position += 1

    for raw in files:
        media_type = raw.get("media_type")
        size = raw.get("size")
        digest = raw.get("digest")
        if (
            not isinstance(media_type, str)
            or not media_type
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _is_digest(digest)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        descriptor: dict[str, JsonValue] = {
            "source": "workspace",
            "media_type": media_type,
            "size": size,
            "digest": cast(str, digest),
            "content_key": cast(str, digest),
        }
        result.append(
            _accepted_occurrence(
                descriptor,
                source="workspace",
                position=position,
            )
        )
        position += 1
    return tuple(result)


def bind_tool_return_attachments(
    tool_name: str,
    call_id: str,
    result: object,
) -> object:
    """Bind attach_files content to stable occurrence ids before it enters history."""
    if tool_name != "attach_files" or not isinstance(result, ToolReturn):
        return result
    occurrences = _attach_file_occurrences(result.return_value, call_id=call_id)
    if not occurrences:
        return result
    content = result.content
    if content is None or isinstance(content, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values = tuple(content)
    binaries = tuple(item for item in values if isinstance(item, BinaryContent))
    if len(binaries) != len(occurrences):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    occurrence_index = 0
    bound: list[UserContent] = []
    for item in values:
        if not isinstance(item, BinaryContent):
            bound.append(item)
            continue
        occurrence = occurrences[occurrence_index]
        occurrence_index += 1
        descriptor = _content_descriptor(item)
        if descriptor is None or not _same_attachment(occurrence, descriptor):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        bound.append(
            BinaryContent.narrow_type(
                BinaryContent(
                    item.data,
                    media_type=item.media_type,
                    identifier=_require_string(occurrence.get("attachment_id")),
                    vendor_metadata=item.vendor_metadata,
                )
            )
        )
    return ToolReturn(
        return_value=result.return_value,
        content=bound,
        metadata=result.metadata,
        tools=result.tools,
    )


def request_attachment_facts(
    messages: Sequence[ModelMessage],
    initial: Sequence[Mapping[str, JsonValue]],
    *,
    accepted_attachment_ids: set[str],
) -> tuple[dict[str, JsonValue], ...]:
    """Record attachment facts from the exact request handed to the model adapter."""
    tool_occurrences: dict[str, dict[str, JsonValue]] = {}
    tool_facts: list[dict[str, JsonValue]] = []
    user_candidates: list[dict[str, JsonValue]] = []
    request_position = 0

    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if (
                isinstance(part, BaseToolReturnPart)
                and part.tool_name == "attach_files"
                and isinstance(part.tool_call_id, str)
                and part.tool_call_id
            ):
                for occurrence in _attach_file_occurrences(
                    part.content,
                    call_id=part.tool_call_id,
                ):
                    attachment_id = _require_string(
                        occurrence.get("attachment_id")
                    )
                    previous = tool_occurrences.get(attachment_id)
                    if previous is not None and previous != occurrence:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                    tool_occurrences[attachment_id] = occurrence
                    if attachment_id in accepted_attachment_ids:
                        continue
                    tool_facts.append(
                        _request_fact(
                            occurrence,
                            fact="accepted",
                            request_position=_require_non_negative_int(
                                occurrence.get("call_position")
                            ),
                            call_id=part.tool_call_id,
                        )
                    )
                    accepted_attachment_ids.add(attachment_id)
                continue

            if not isinstance(part, UserPromptPart) or isinstance(part.content, str):
                continue
            for item in part.content:
                descriptor = _content_descriptor(item)
                if descriptor is None:
                    continue
                occurrence = (
                    tool_occurrences.get(item.identifier)
                    if isinstance(item, BinaryContent)
                    else None
                )
                if occurrence is not None and _same_attachment(
                    occurrence,
                    descriptor,
                ):
                    tool_facts.append(
                        _request_fact(
                            occurrence,
                            fact="included_in_request",
                            request_position=request_position,
                            call_id=_require_string(occurrence.get("call_id")),
                        )
                    )
                else:
                    candidate = dict(descriptor)
                    candidate["request_position"] = request_position
                    user_candidates.append(candidate)
                request_position += 1

    initial_facts: list[dict[str, JsonValue]] = []
    for expected, candidate in _match_initial_occurrences(initial, user_candidates):
        initial_facts.append(
            _request_fact(
                expected,
                fact="included_in_request",
                request_position=_require_non_negative_int(
                    candidate.get("request_position")
                ),
                call_id=None,
            )
        )
    return tuple((*initial_facts, *tool_facts))


def _attach_file_occurrences(
    value: object,
    *,
    call_id: str,
) -> tuple[dict[str, JsonValue], ...]:
    if not isinstance(value, Mapping):
        return ()
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        return ()
    result: list[dict[str, JsonValue]] = []
    for position, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        media_type = raw.get("media_type")
        size = raw.get("size")
        digest = raw.get("sha256")
        if (
            not isinstance(media_type, str)
            or not media_type
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or not _is_digest(digest)
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        content_key = cast(str, digest)
        result.append(
            {
                "attachment_id": _attachment_id(
                    "attach_files",
                    position,
                    content_key,
                    call_id=call_id,
                ),
                "source": "attach_files",
                "media_type": media_type,
                "size": size,
                "digest": content_key,
                "content_key": content_key,
                "call_id": call_id,
                "call_position": position,
            }
        )
    return tuple(result)


def _accepted_occurrence(
    descriptor: Mapping[str, JsonValue],
    *,
    source: str,
    position: int,
) -> dict[str, JsonValue]:
    content_key = _require_string(descriptor.get("content_key"))
    return {
        "fact": "accepted",
        "attachment_id": _attachment_id(source, position, content_key),
        "source": source,
        "media_type": descriptor.get("media_type"),
        "size": descriptor.get("size"),
        "digest": descriptor.get("digest"),
        "content_key": content_key,
        "position": position,
        "call_id": None,
    }


def _request_fact(
    occurrence: Mapping[str, JsonValue],
    *,
    fact: str,
    request_position: int,
    call_id: str | None,
) -> dict[str, JsonValue]:
    value = {
        "fact": fact,
        "attachment_id": occurrence.get("attachment_id"),
        "source": occurrence.get("source"),
        "media_type": occurrence.get("media_type"),
        "size": occurrence.get("size"),
        "digest": occurrence.get("digest"),
        "content_key": occurrence.get("content_key"),
        "position": request_position,
        "call_id": call_id,
    }
    try:
        normalized = normalize_json_value(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return normalized


def _match_initial_occurrences(
    initial: Sequence[Mapping[str, JsonValue]],
    candidates: Sequence[Mapping[str, JsonValue]],
) -> tuple[
    tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]],
    ...,
]:
    expected = tuple(initial)
    if not expected:
        return ()
    matches: list[tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]]] = []
    candidate_index = len(candidates) - 1
    for value in reversed(expected):
        while candidate_index >= 0 and not _same_attachment(
            value,
            candidates[candidate_index],
        ):
            candidate_index -= 1
        if candidate_index < 0:
            continue
        matches.append((value, candidates[candidate_index]))
        candidate_index -= 1
    matches.reverse()
    return tuple(matches)


def _same_attachment(
    expected: Mapping[str, JsonValue],
    candidate: Mapping[str, JsonValue],
) -> bool:
    source = expected.get("source")
    candidate_source = candidate.get("source")
    if source in {"workspace", "attach_files"}:
        if candidate_source != "binary":
            return False
    elif source != candidate_source:
        return False
    return (
        expected.get("content_key") == candidate.get("content_key")
        and expected.get("media_type") == candidate.get("media_type")
        and expected.get("size") == candidate.get("size")
        and expected.get("digest") == candidate.get("digest")
    )


def _content_descriptor(item: object) -> dict[str, JsonValue] | None:
    if isinstance(item, BinaryContent):
        digest = hashlib.sha256(item.data).hexdigest()
        return {
            "source": "binary",
            "media_type": item.media_type,
            "size": len(item.data),
            "digest": digest,
            "content_key": digest,
        }
    if isinstance(item, _URL_TYPES):
        return {
            "source": "url",
            "media_type": item.media_type,
            "size": None,
            "digest": None,
            "content_key": canonical_sha256(
                {
                    "kind": item.kind,
                    "url": item.url,
                    "identifier": item.identifier,
                }
            ),
        }
    if isinstance(item, UploadedFile):
        return {
            "source": "uploaded_file",
            "media_type": item.media_type,
            "size": None,
            "digest": None,
            "content_key": canonical_sha256(
                {
                    "provider_name": item.provider_name,
                    "file_id": item.file_id,
                    "identifier": item.identifier,
                }
            ),
        }
    return None


def _iter_multimodal(value: object) -> tuple[object, ...]:
    if _content_descriptor(value) is not None:
        return (value,)
    if isinstance(value, Mapping):
        result: list[object] = []
        for item in value.values():
            result.extend(_iter_multimodal(item))
        return tuple(result)
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        result = []
        for item in value:
            result.extend(_iter_multimodal(item))
        return tuple(result)
    return ()


def _attachment_id(
    source: str,
    position: int,
    content_key: str,
    *,
    call_id: str | None = None,
) -> str:
    value: dict[str, JsonValue] = {
        "contract": "attachment-occurrence-v1",
        "source": source,
        "position": position,
        "content_key": content_key,
    }
    if call_id is not None:
        value["call_id"] = call_id
    return canonical_sha256(value)


def _require_string(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _require_non_negative_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return value


def _is_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


__all__ = [
    "bind_tool_return_attachments",
    "input_attachment_views",
    "request_attachment_facts",
]
