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
                    source=descriptor["source"],
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


def request_attachment_facts(
    messages: Sequence[ModelMessage],
    initial: Sequence[Mapping[str, JsonValue]],
    *,
    accepted_attachment_ids: set[str],
) -> tuple[dict[str, JsonValue], ...]:
    """Record attachment facts from the exact request handed to the model adapter."""
    user_candidates: list[dict[str, JsonValue]] = []
    tool_candidates: list[dict[str, JsonValue]] = []
    request_position = 0

    for message in messages:
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                for item in part.content:
                    descriptor = _content_descriptor(item)
                    if descriptor is None:
                        continue
                    candidate = dict(descriptor)
                    candidate["request_position"] = request_position
                    user_candidates.append(candidate)
                    request_position += 1
                continue
            if (
                isinstance(part, BaseToolReturnPart)
                and part.tool_name == "attach_files"
                and isinstance(part.tool_call_id, str)
                and part.tool_call_id
            ):
                call_position = 0
                for item in _iter_multimodal(part.content):
                    descriptor = _content_descriptor(item)
                    if descriptor is None:
                        continue
                    candidate = dict(descriptor)
                    candidate["source"] = "attach_files"
                    candidate["call_id"] = part.tool_call_id
                    candidate["call_position"] = call_position
                    candidate["request_position"] = request_position
                    tool_candidates.append(candidate)
                    call_position += 1
                    request_position += 1

    facts: list[dict[str, JsonValue]] = []
    matched = _match_initial_occurrences(initial, user_candidates)
    for expected, candidate in matched:
        facts.append(
            _request_fact(
                expected,
                fact="included_in_request",
                request_position=_require_non_negative_int(
                    candidate.get("request_position")
                ),
                call_id=None,
            )
        )

    for candidate in tool_candidates:
        call_id = candidate.get("call_id")
        if not isinstance(call_id, str) or not call_id:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        call_position = _require_non_negative_int(candidate.get("call_position"))
        attachment_id = _attachment_id(
            "attach_files",
            call_position,
            _require_string(candidate.get("content_key")),
            call_id=call_id,
        )
        occurrence = {
            "attachment_id": attachment_id,
            "source": "attach_files",
            "media_type": candidate.get("media_type"),
            "size": candidate.get("size"),
            "digest": candidate.get("digest"),
            "content_key": candidate.get("content_key"),
        }
        if attachment_id not in accepted_attachment_ids:
            facts.append(
                _request_fact(
                    occurrence,
                    fact="accepted",
                    request_position=_require_non_negative_int(
                        candidate.get("request_position")
                    ),
                    call_id=call_id,
                )
            )
            accepted_attachment_ids.add(attachment_id)
        facts.append(
            _request_fact(
                occurrence,
                fact="included_in_request",
                request_position=_require_non_negative_int(
                    candidate.get("request_position")
                ),
                call_id=call_id,
            )
        )

    return tuple(facts)


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
    if source == "workspace":
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


__all__ = ["input_attachment_views", "request_attachment_facts"]
