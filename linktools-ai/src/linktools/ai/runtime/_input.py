#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical execution input and workspace file materialization."""

import base64
import binascii
import hashlib
import json
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeAlias, cast

from linktools.core import environ
from pydantic_ai.messages import (
    AudioUrl,
    BinaryContent,
    CachePoint,
    DocumentUrl,
    ImageUrl,
    TextContent,
    UserContent,
    UploadedFile,
    VideoUrl,
)

from ..capability import WorkspaceAccess
from ..core import JsonValue, PromptLimits, canonical_json_bytes, normalize_json_value
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline
from ..workspace import normalize_workspace_path
from ._input_contract import (
    CanonicalUserInput,
    UserPromptInput,
    validate_user_content,
    validate_user_input,
)

if TYPE_CHECKING:
    from ._object import RuntimeObjectKeyFactory
    from .state._contracts import StoredUserInput

_TEXT_CODEC = "text"
_USER_CONTENT_CODEC = "user-content-v1"
_UserPromptInput = UserPromptInput
DraftPrompt: TypeAlias = JsonValue
TaskPrompt: TypeAlias = JsonValue
_logger = environ.get_logger("ai.runtime.input")


@dataclass(frozen=True, slots=True)
class InputIntent:
    """The replay identity of an input before any file body is read."""

    prompt: DraftPrompt
    files: tuple[str, ...]

    @property
    def digest(self) -> str:
        return hashlib.sha256(
            canonical_json_bytes(
                {
                    "version": 1,
                    "prompt": self.prompt,
                    "files": list(self.files),
                }
            )
        ).hexdigest()


class _MaterializedUserContent(tuple):
    view: Mapping[str, JsonValue]

    def __new__(
        cls,
        items: Sequence[UserContent],
        view: Mapping[str, JsonValue],
    ) -> "_MaterializedUserContent":
        value = super().__new__(cls, items)
        value.view = dict(view)
        return value


def input_intent(value: _UserPromptInput, files: Sequence[str]) -> InputIntent:
    canonical = validate_user_input(value)
    normalized_files = _require_files(files)
    return InputIntent(_draft_prompt(canonical), normalized_files)


def task_prompt_draft(value: _UserPromptInput) -> TaskPrompt:
    """Encode generic task input while rejecting body-persistent binaries."""
    canonical = validate_user_input(value)
    if isinstance(canonical, str):
        return {"kind": "text", "text": canonical}
    if any(isinstance(item, BinaryContent) for item in canonical):
        raise AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            safe_details={
                "field": "user_prompt",
                "reason": "binary_content_not_supported_for_task",
            },
        )
    return {
        "kind": "pydantic-user-content-v1",
        "value": _encode_user_content(canonical),
    }


def decode_user_content_payload(value: Mapping[str, JsonValue]) -> tuple[UserContent, ...]:
    return _decode_user_content(cast(dict[str, JsonValue], value))


class ExecutionInputMaterializer:
    """Own every Workspace file read used to admit an execution."""

    def __init__(
        self,
        access: "WorkspaceAccess | None",
        limits: PromptLimits,
        *,
        object_store: ObjectStore | None = None,
        object_key_factory: "RuntimeObjectKeyFactory | None" = None,
        payload_policy: PayloadPolicy | None = None,
    ) -> None:
        if access is not None and not isinstance(access, WorkspaceAccess):
            raise TypeError("access must be WorkspaceAccess or None")
        if not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        self._access = access
        self._limits = limits
        self._object_store = object_store
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy or PayloadPolicy()
        self._mime = mimetypes.MimeTypes(filenames=())

    async def close(self) -> None:
        if self._access is not None:
            await self._access.close()

    @property
    def access(self) -> "WorkspaceAccess | None":
        return self._access

    async def canonicalize_files(self, files: Sequence[str]) -> tuple[str, ...]:
        raw_files = _require_files(files)
        if raw_files and self._access is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "files", "reason": "workspace_required"},
            )
        result: list[str] = []
        seen: set[str] = set()
        for path in raw_files:
            try:
                access = self._access
                if access is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                canonical = normalize_workspace_path(
                    await access.canonicalize_path(path)
                )
            except AIError as error:
                mapped = _file_request_error(error, request_invalid_reason="path_invalid")
                if mapped is None:
                    raise
                raise mapped from error
            except (TypeError, ValueError) as error:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={"field": "files", "reason": "path_invalid"},
                ) from error
            if canonical in seen:
                continue
            seen.add(canonical)
            result.append(canonical)
        return tuple(result)

    def intent(
        self,
        value: _UserPromptInput,
        canonical_files: Sequence[str],
    ) -> InputIntent:
        files = _require_canonical_files(canonical_files)
        return input_intent(value, files)

    async def materialize(
        self,
        value: _UserPromptInput,
        canonical_files: Sequence[str],
    ) -> CanonicalUserInput:
        canonical = validate_user_input(value)
        files = _require_canonical_files(canonical_files)
        direct_binary = _binary_parts(canonical)
        if len(direct_binary) > self._limits.max_binary_input_parts:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        total_bytes = sum(len(item.data) for item in direct_binary)
        if total_bytes > self._limits.max_binary_input_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if len(direct_binary) + len(files) > self._limits.max_binary_input_parts:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not files:
            return canonical
        access = self._access
        if access is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "files", "reason": "workspace_required"},
            )

        additions: list[UserContent] = []
        file_views: list[dict[str, JsonValue]] = []
        for path in files:
            media_type = self._media_type(path)
            remaining = self._limits.max_binary_input_bytes - total_bytes
            if remaining < 0:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            try:
                body = await access.read_bytes(path, max_bytes=remaining)
            except AIError as error:
                mapped = _file_request_error(error, request_invalid_reason="file_invalid")
                if mapped is None:
                    raise
                raise mapped from error
            total_bytes += len(body)
            if total_bytes > self._limits.max_binary_input_bytes:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            file_views.append(
                {
                    "path": path,
                    "media_type": media_type,
                    "size": len(body),
                    "digest": hashlib.sha256(body).hexdigest(),
                }
            )
            additions.extend(
                (
                    f"Workspace file path: {json.dumps(path)}",
                    BinaryContent(
                        data=body,
                        media_type=media_type,
                    ),
                )
            )
        if isinstance(canonical, str):
            materialized: CanonicalUserInput = (canonical, *additions)
        else:
            materialized = (*canonical, *additions)
        validate_user_content(materialized)
        _logger.info(
            "execution input materialized: files=%s binary_bytes=%s",
            len(files),
            total_bytes,
        )
        return cast(
            CanonicalUserInput,
            _MaterializedUserContent(
                cast(Sequence[UserContent], materialized),
                _input_view(canonical, file_views),
            ),
        )

    async def store(
        self,
        value: CanonicalUserInput,
        *,
        tenant_id: str,
    ) -> "StoredUserInput":
        from .state._contracts import StoredUserInput

        canonical = validate_user_input(value)
        if isinstance(canonical, str):
            return StoredUserInput(
                _TEXT_CODEC,
                StoredPayload.inline_text(canonical),
            )
        view = (
            dict(value.view)
            if isinstance(value, _MaterializedUserContent)
            else _input_view(value)
        )
        payload = StoredPayload.inline_json(_encode_user_content(canonical))
        if not payload_fits_inline(payload, self._payload_policy):
            if self._object_store is None or self._object_key_factory is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            body = canonical_json_bytes(cast(JsonValue, payload.value))
            from ._object import RuntimeObjectKeyFactory, put_runtime_object
            from .state import RuntimeDomain

            if not isinstance(self._object_key_factory, RuntimeObjectKeyFactory):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            reference = await put_runtime_object(
                self._object_store,
                self._object_key_factory,
                RuntimeDomain.EXECUTION,
                tenant_id,
                body,
            )
            payload = StoredPayload.object(reference)
        return StoredUserInput(_USER_CONTENT_CODEC, payload, view)

    async def restore(self, value: "StoredUserInput") -> CanonicalUserInput:
        from .state._contracts import StoredUserInput

        if not isinstance(value, StoredUserInput):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        payload = value.payload
        if payload.kind == "object":
            if self._object_store is None or payload.ref is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            from ._object import read_runtime_object

            raw = await read_runtime_object(self._object_store, payload.ref)
            try:
                decoded = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
        else:
            decoded = payload.decode()
        if value.codec == _TEXT_CODEC:
            if not isinstance(decoded, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            validate_user_input(decoded)
            return decoded
        if value.codec != _USER_CONTENT_CODEC or not isinstance(decoded, Mapping):
            raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
        return _decode_user_content(cast(dict[str, JsonValue], decoded))

    def _media_type(self, path: str) -> str:
        media_type, _ = self._mime.guess_type(path, strict=False)
        if not media_type:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "field": "files",
                    "reason": "media_type_unknown",
                },
            )
        return media_type


def _file_request_error(
    error: AIError,
    *,
    request_invalid_reason: str,
) -> AIError | None:
    if error.code is ErrorCode.REQUEST_FIELD_INVALID:
        return AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={
                "field": "files",
                "reason": request_invalid_reason,
            },
        )
    if error.code is ErrorCode.STORAGE_NOT_FOUND:
        return AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={"field": "files", "reason": "file_not_found"},
        )
    if error.code is ErrorCode.AUTHORIZATION_DENIED:
        return AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={"field": "files", "reason": "path_not_allowed"},
        )
    return None


def _require_files(value: Sequence[str]) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(
        value,
        (str, bytes, bytearray),
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    result = tuple(value)
    if any(not isinstance(item, str) or not item for item in result):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return result


def _require_canonical_files(value: Sequence[str]) -> tuple[str, ...]:
    files = _require_files(value)
    for path in files:
        try:
            canonical = normalize_workspace_path(path)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if canonical != path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return files


def _binary_parts(content: Sequence[UserContent]) -> tuple[BinaryContent, ...]:
    return tuple(item for item in content if isinstance(item, BinaryContent))


def _draft_prompt(value: _UserPromptInput) -> DraftPrompt:
    canonical = validate_user_input(value)
    if isinstance(canonical, str):
        return {"kind": "text", "text": canonical}
    result: list[JsonValue] = []
    for item in canonical:
        if isinstance(item, str):
            result.append({"kind": "text", "text": item})
        elif isinstance(item, BinaryContent):
            result.append(
                {
                    "kind": "binary",
                    "digest": hashlib.sha256(item.data).hexdigest(),
                    "size": len(item.data),
                    "media_type": item.media_type,
                    "identifier": item.identifier,
                    "vendor_metadata": _json_object_or_none(item.vendor_metadata),
                }
            )
        else:
            result.append(
                {
                    "kind": "native",
                    "codec": _USER_CONTENT_CODEC,
                    "value": _encode_user_content((item,)),
                }
            )
    return result


def _input_view(
    value: _UserPromptInput,
    files: Sequence[Mapping[str, JsonValue]] = (),
) -> dict[str, JsonValue]:
    canonical = validate_user_input(value)
    prompt = _draft_prompt(canonical)
    if not isinstance(canonical, str):
        prompt = {"kind": "items", "items": prompt}
    try:
        normalized = normalize_json_value(
            {
                "version": 1,
                "prompt": prompt,
                "files": [dict(item) for item in files],
            }
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return normalized


def stored_user_input_view(value: "StoredUserInput") -> dict[str, JsonValue]:
    from .state._contracts import StoredUserInput

    if not isinstance(value, StoredUserInput):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    if value.view is not None:
        return dict(value.view)
    if value.payload.kind != "inline":
        raise AIError(ErrorCode.SESSION_HISTORY_UNAVAILABLE)
    decoded = value.payload.decode()
    if value.codec == _TEXT_CODEC:
        if not isinstance(decoded, str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return _input_view(decoded)
    if value.codec != _USER_CONTENT_CODEC or not isinstance(decoded, Mapping):
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    content = _decode_user_content(cast(dict[str, JsonValue], decoded))
    return _input_view(content)


def _json_object_or_none(value: object) -> JsonValue:
    if value is None:
        return None
    try:
        normalized = normalize_json_value(value)
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return normalized


def _encode_user_content(content: Sequence[UserContent]) -> dict[str, JsonValue]:
    return {"items": [_encode_user_content_item(item) for item in content]}


def _decode_user_content(payload: dict[str, JsonValue]) -> tuple[UserContent, ...]:
    if set(payload) != {"items"} or not isinstance(payload.get("items"), list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        content = tuple(
            _decode_user_content_item(item)
            for item in cast(list[object], payload["items"])
        )
    except AIError:
        raise
    except _UnsupportedUserContentKind as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    validate_user_content(content)
    return content


def _encode_user_content_item(item: UserContent) -> JsonValue:
    try:
        if isinstance(item, str):
            return {"kind": "text", "text": item}
        if isinstance(item, TextContent):
            return {
                "kind": "text-content",
                "content": item.content,
                "metadata": normalize_json_value(item.metadata),
            }
        if isinstance(item, BinaryContent):
            return {
                "kind": "binary",
                "data": base64.b64encode(item.data).decode("ascii"),
                "media_type": item.media_type,
                "identifier": item.identifier,
                "vendor_metadata": normalize_json_value(item.vendor_metadata),
            }
        if isinstance(item, (ImageUrl, AudioUrl, DocumentUrl, VideoUrl)):
            return {
                "kind": item.kind,
                "url": item.url,
                "media_type": item.media_type,
                "identifier": item.identifier,
                "force_download": item.force_download,
                "vendor_metadata": normalize_json_value(item.vendor_metadata),
            }
        if isinstance(item, UploadedFile):
            return {
                "kind": "uploaded-file",
                "file_id": item.file_id,
                "provider_name": item.provider_name,
                "media_type": item.media_type,
                "identifier": item.identifier,
                "vendor_metadata": normalize_json_value(item.vendor_metadata),
            }
        if isinstance(item, CachePoint):
            return {"kind": "cache-point", "ttl": item.ttl}
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _decode_user_content_item(value: object) -> UserContent:
    if not isinstance(value, Mapping) or not isinstance(value.get("kind"), str):
        raise ValueError("user content item is invalid")
    kind = value["kind"]
    if kind == "text":
        text = value.get("text")
        if not isinstance(text, str):
            raise ValueError("text content is invalid")
        return text
    if kind == "text-content":
        content = value.get("content")
        if not isinstance(content, str) or "metadata" not in value:
            raise ValueError("text content is invalid")
        return TextContent(content, metadata=value["metadata"])
    if kind == "binary":
        data = value.get("data")
        media_type = value.get("media_type")
        if (
            not isinstance(data, str)
            or not isinstance(media_type, str)
            or not media_type
            or "identifier" not in value
            or "vendor_metadata" not in value
        ):
            raise ValueError("binary content is invalid")
        return BinaryContent(
            base64.b64decode(data, validate=True),
            media_type=media_type,
            identifier=_optional_string(value["identifier"]),
            vendor_metadata=value["vendor_metadata"],
        )
    url_types = {
        "image-url": ImageUrl,
        "audio-url": AudioUrl,
        "document-url": DocumentUrl,
        "video-url": VideoUrl,
    }
    url_type = url_types.get(kind)
    if url_type is not None:
        url = value.get("url")
        media_type = value.get("media_type")
        force_download = value.get("force_download")
        if (
            not isinstance(url, str)
            or media_type is not None
            and not isinstance(media_type, str)
            or not isinstance(force_download, bool)
            or "identifier" not in value
            or "vendor_metadata" not in value
        ):
            raise ValueError("URL content is invalid")
        return url_type(
            url,
            media_type=media_type,
            identifier=_optional_string(value["identifier"]),
            force_download=force_download,
            vendor_metadata=value["vendor_metadata"],
        )
    if kind == "uploaded-file":
        file_id = value.get("file_id")
        provider_name = value.get("provider_name")
        media_type = value.get("media_type")
        if (
            not isinstance(file_id, str)
            or not isinstance(provider_name, str)
            or media_type is not None
            and not isinstance(media_type, str)
            or "identifier" not in value
            or "vendor_metadata" not in value
        ):
            raise ValueError("uploaded file is invalid")
        return UploadedFile(
            file_id,
            provider_name,
            media_type=media_type,
            identifier=_optional_string(value["identifier"]),
            vendor_metadata=value["vendor_metadata"],
        )
    if kind == "cache-point":
        ttl = value.get("ttl")
        if ttl not in {"5m", "1h"}:
            raise ValueError("cache point is invalid")
        return CachePoint(ttl=cast(str, ttl))
    raise _UnsupportedUserContentKind("unknown user content kind")


class _UnsupportedUserContentKind(ValueError):
    pass


def _optional_string(value: object) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError("user content identifier is invalid")
    return cast(str | None, value)


__all__ = [
    "CanonicalUserInput",
    "ExecutionInputMaterializer",
    "InputIntent",
    "decode_user_content_payload",
    "input_intent",
    "stored_user_input_view",
    "task_prompt_draft",
    "validate_user_input",
]
