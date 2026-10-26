#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned attachment occurrence identity and request facts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import PureWindowsPath
from typing import cast
from urllib.parse import urlsplit

from pydantic_ai.messages import (
    AudioUrl,
    BaseToolReturnPart,
    BinaryContent,
    DocumentUrl,
    ImageUrl,
    ModelMessage,
    ModelRequest,
    TextContent,
    ToolReturn,
    UploadedFile,
    UserContent,
    UserPromptPart,
    VideoUrl,
)

from linktools.ai.core import (
    JsonValue,
    WorkspaceFileInput,
    canonical_sha256,
    normalize_json_value,
)
from linktools.ai.errors import AIError, ErrorCode
from linktools.ai.model import ReferencedContent
from linktools.ai.workspace import validate_workspace_path

_URL_TYPES = (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)
_FILE_TOOLS = frozenset({"attach_files", "read_file"})


def binary_attachment_text(
    item: BinaryContent,
    *,
    path: str | None = None,
) -> TextContent:
    """Present existing content identity and known Workspace metadata as JSON.

    The identifier denotes a Runtime occurrence only after tool-return binding.
    The metadata tag associates generated text with its binary without treating
    arbitrary neighboring user text as attachment provenance.
    """
    value: dict[str, JsonValue] = {
        "attachment": None if _is_external_reference(item.identifier) else item.identifier,
        "size": len(item.data),
        "mime_type": item.media_type,
    }
    if path is not None:
        value["path"] = validate_workspace_path(path)
    return TextContent(
        json.dumps(value, ensure_ascii=True, separators=(",", ":")),
        metadata={"source": "linktools.attachment", "attachment": item.identifier},
    )


def is_binary_attachment_text(
    value: object,
    item: BinaryContent,
    *,
    path: str | None = None,
) -> bool:
    """Match generated attachment text, optionally requiring its known path."""
    if not isinstance(value, TextContent) or value.metadata != {
        "source": "linktools.attachment", "attachment": item.identifier,
    }:
        return False
    try:
        metadata = json.loads(value.content)
        if not isinstance(metadata, dict) or type(metadata.get("size")) is not int:
            return False
        known_path = metadata.get("path")
        if "path" in metadata and not isinstance(known_path, str):
            return False
        if path is not None and known_path != path:
            return False
        return metadata == json.loads(binary_attachment_text(item, path=known_path).content)
    except (TypeError, ValueError):
        return False


def has_binary_attachment_metadata(item: BinaryContent, message: ModelRequest) -> bool:
    """Check retained metadata in this request without guessing across messages."""
    descriptor = None
    matched_without_path = False
    paths: set[str] = set()
    for part in message.parts:
        if isinstance(part, UserPromptPart) and not isinstance(part.content, str):
            if any(is_binary_attachment_text(value, item) for value in part.content):
                return True
        if (not isinstance(part, BaseToolReturnPart)
            or part.tool_name not in _FILE_TOOLS
            or not isinstance(part.tool_call_id, str) or not part.tool_call_id):
            continue
        value = file_tool_result_metadata(part.content)
        if value is None:
            continue
        try:
            occurrences = _tool_file_occurrences(
                part.content, source=part.tool_name, call_id=part.tool_call_id,
            )
        except AIError:
            continue
        for occurrence in occurrences:
            if occurrence["attachment_id"] != item.identifier:
                continue
            if descriptor is None:
                descriptor = _content_descriptor(item)
            if descriptor is None or not _same_attachment(occurrence, descriptor):
                continue
            files = cast(list[Mapping[str, JsonValue]], value["files"])
            file = files[cast(int, occurrence["call_position"])]
            if "attachment" in file and file["attachment"] != item.identifier:
                continue
            if "path" not in file:
                matched_without_path = True
                continue
            path = file.get("path")
            if not isinstance(path, str):
                continue
            try:
                paths.add(validate_workspace_path(path))
            except ValueError:
                continue
    return len(paths) == 1 or (matched_without_path and not paths)


def _is_external_reference(value: str) -> bool:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return True
    return bool(
        parsed.scheme or parsed.netloc or value.startswith(("/", "\\"))
        or PureWindowsPath(value).is_absolute()
    )


def input_attachment_views(
    value: str | Sequence[UserContent | WorkspaceFileInput],
    files: Sequence[Mapping[str, JsonValue]] = (),
) -> tuple[dict[str, JsonValue], ...]:
    """Project accepted input occurrences without retaining raw external references."""
    result: list[dict[str, JsonValue]] = []
    position = 0
    workspace_files = iter(
        raw for raw in files if raw.get("_prompt_occurrence") is True
    )
    if not isinstance(value, str):
        for item in value:
            if isinstance(item, WorkspaceFileInput):
                try:
                    raw = next(workspace_files)
                except StopIteration as error:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
                descriptor = _workspace_descriptor(raw)
                result.append(
                    _accepted_occurrence(
                        descriptor,
                        source="workspace",
                        position=position,
                    )
                )
                position += 1
                continue
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
        if raw.get("_prompt_occurrence") is True:
            continue
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
            "input_identifier": raw.get("input_identifier"),
        }
        result.append(
            _accepted_occurrence(
                descriptor,
                source="workspace",
                position=position,
            )
        )
        position += 1
    try:
        next(workspace_files)
    except StopIteration:
        return tuple(result)
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def bind_tool_return_attachments(
    tool_name: str,
    call_id: str,
    result: object,
) -> object:
    """Bind file-input content to stable occurrence ids before it enters history."""
    if tool_name not in _FILE_TOOLS or not isinstance(result, ToolReturn):
        return result
    native = isinstance(result.return_value, list)
    value = file_tool_result_metadata(result.return_value)
    content = result.return_value[1:] if native else result.content
    occurrences = _tool_file_occurrences(
        value, source=tool_name, call_id=call_id,
    )
    if not occurrences:
        if tool_name == "read_file" and content is not None and any(
            isinstance(item, BinaryContent)
            or isinstance(item, ReferencedContent) and item.identifier is not None
            for item in content
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return result
    if content is None or isinstance(content, str):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    values = tuple(content)
    occurrence_index = 0
    bound: list[UserContent] = []
    for item in values:
        if (not isinstance(item, (BinaryContent, ReferencedContent))
            or isinstance(item, ReferencedContent) and item.identifier is None):
            bound.append(item)
            continue
        descriptor = _content_descriptor(item)
        occurrence = occurrences[occurrence_index] if occurrence_index < len(occurrences) else None
        if occurrence is None:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        occurrence_index += 1
        if descriptor is None or not _same_attachment(occurrence, descriptor):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        identifier = _require_string(occurrence.get("attachment_id"))
        if isinstance(item, ReferencedContent):
            if item.identifier != identifier:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            bound.append(item)
            continue
        bound.append(
            BinaryContent.narrow_type(
                BinaryContent(
                    item.data,
                    media_type=item.media_type,
                    identifier=identifier,
                    vendor_metadata=item.vendor_metadata,
                )
            )
        )
    if occurrence_index != len(occurrences):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if tool_name == "read_file":
        value = dict(cast(Mapping[str, JsonValue], value))
        files = cast(list[Mapping[str, JsonValue]], value["files"])
        value["files"] = [
            {**file, "attachment": occurrence["attachment_id"]}
            for file, occurrence in zip(files, occurrences)
        ]
    return ToolReturn(
        return_value=[value, *bound] if native else value,
        content=result.content if native else bound,
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
                and part.tool_name in _FILE_TOOLS
                and isinstance(part.tool_call_id, str)
                and part.tool_call_id
            ):
                for occurrence in _tool_file_occurrences(
                    part.content,
                    source=part.tool_name,
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
                content = part.content[1:] if isinstance(part.content, list) else ()
            elif isinstance(part, UserPromptPart) and not isinstance(part.content, str):
                content = part.content
            else:
                continue
            for item in content:
                descriptor = _content_descriptor(item)
                if descriptor is None:
                    continue
                occurrence = (
                    tool_occurrences.get(item.identifier)
                    if isinstance(item, (BinaryContent, ReferencedContent))
                    else None
                )
                if (isinstance(part, BaseToolReturnPart) and occurrence is not None
                    and occurrence.get("call_id") != part.tool_call_id):
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
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


def file_tool_result_metadata(value: object) -> Mapping[str, object] | None:
    """Read the structured receipt from a file tool's native or metadata-only result."""
    if isinstance(value, list) and value:
        value = value[0]
    return value if isinstance(value, Mapping) else None


def _tool_file_occurrences(
    value: object,
    *,
    source: str,
    call_id: str,
) -> tuple[dict[str, JsonValue], ...]:
    value = file_tool_result_metadata(value)
    if value is None:
        return ()
    raw_files = value.get("files")
    if not isinstance(raw_files, list):
        if source == "read_file" and "files" in value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return ()
    result: list[dict[str, JsonValue]] = []
    previous_input_index = -1
    for position, raw in enumerate(raw_files):
        if not isinstance(raw, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if source == "read_file":
            input_index = _require_non_negative_int(raw.get("input_index"))
            if input_index <= previous_input_index:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            previous_input_index = input_index
        media_type = raw.get("media_type")
        if "mime_type" in raw:
            if "media_type" in raw and media_type != raw["mime_type"]:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            media_type = raw["mime_type"]
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
                    source,
                    position,
                    content_key,
                    call_id=call_id,
                ),
                "source": source,
                "media_type": media_type,
                "size": size,
                "digest": content_key,
                "content_key": content_key,
                "input_identifier": None,
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
        "input_identifier": descriptor.get("input_identifier"),
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
        "input_identifier": occurrence.get("input_identifier"),
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
    if source in {"attach_files", "read_file"}:
        if candidate_source not in {"binary", "referenced"}:
            return False
    elif source == "workspace":
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
    if isinstance(item, ReferencedContent):
        return {
            "source": "referenced",
            "media_type": item.media_type,
            "size": item.ref.size,
            "digest": item.ref.digest,
            "content_key": item.ref.digest,
            "input_identifier": item.identifier,
        }
    if isinstance(item, BinaryContent):
        digest = hashlib.sha256(item.data).hexdigest()
        return {
            "source": "binary",
            "media_type": item.media_type,
            "size": len(item.data),
            "digest": digest,
            "content_key": digest,
            "input_identifier": item.identifier,
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
            "input_identifier": item.identifier,
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
            "input_identifier": item.identifier,
        }
    return None


def _workspace_descriptor(
    raw: Mapping[str, JsonValue],
) -> dict[str, JsonValue]:
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
    return {
        "source": "workspace",
        "media_type": media_type,
        "size": size,
        "digest": cast(str, digest),
        "content_key": cast(str, digest),
        "input_identifier": raw.get("input_identifier"),
    }


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
    "file_tool_result_metadata",
    "binary_attachment_text",
    "has_binary_attachment_metadata",
    "is_binary_attachment_text",
    "input_attachment_views",
    "request_attachment_facts",
]
