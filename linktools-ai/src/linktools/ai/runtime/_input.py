#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Canonical execution input and workspace file materialization."""

import binascii
import hashlib
import json
import mimetypes
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, TypeAlias, cast

from linktools.core import environ
from pydantic_ai.messages import BinaryContent, UserContent

from ..core import (
    JsonValue,
    PromptLimits,
    WorkspaceFileInput,
    canonical_json_bytes,
    normalize_json_value,
)
from ..errors import AIError, ErrorCode
from ..storage import ObjectStore, PayloadPolicy, StoredPayload, payload_fits_inline
from ..workspace import normalize_workspace_input_path
from ._attachment import input_attachment_views
from .state._plan import RuntimeDomain
from ._input_contract import (
    CanonicalUserInput,
    MaterializedUserContent,
    UserPromptInput,
    validate_user_content,
    validate_user_input,
)
from ._user_content_codec import (
    UnsupportedUserContentKind,
    decode_user_content_item,
    encode_user_content_item,
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


def input_intent(value: _UserPromptInput, files: Sequence[str]) -> InputIntent:
    canonical = validate_user_input(value)
    normalized_files = _require_files(files)
    return InputIntent(_draft_prompt(canonical), normalized_files)


def task_prompt_draft(value: _UserPromptInput) -> TaskPrompt:
    """Encode a construction draft; attachment bodies freeze at admission."""
    canonical = validate_user_input(value)
    if isinstance(canonical, str):
        return {"kind": "text", "text": canonical}
    if any(isinstance(item, (BinaryContent, WorkspaceFileInput)) for item in canonical):
        return {
            "kind": "task-user-content-v1",
            "intent": _draft_prompt(canonical),
            "items": [_encode_task_prompt_item(item) for item in canonical],
        }
    return {
        "kind": "pydantic-user-content-v1",
        "value": _encode_user_content(canonical),
    }


def decode_user_content_payload(value: Mapping[str, JsonValue]) -> tuple[UserContent, ...]:
    return _decode_user_content(cast(dict[str, JsonValue], value))


def decode_task_prompt_draft(value: Mapping[str, JsonValue]) -> CanonicalUserInput:
    """Decode the construction-state prompt used by an Agent task node."""
    kind = value.get("kind")
    if kind == "text":
        if set(value) != {"kind", "text"} or not isinstance(value.get("text"), str):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return cast(str, value["text"])
    if kind == "pydantic-user-content-v1":
        if set(value) != {"kind", "value"} or not isinstance(value.get("value"), Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return decode_user_content_payload(
            cast(Mapping[str, JsonValue], value["value"])
        )
    if kind != "task-user-content-v1":
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    items = value.get("items")
    if set(value) != {"kind", "items", "intent"} or not isinstance(items, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    decoded: list[UserContent | WorkspaceFileInput] = []
    for item in items:
        if not isinstance(item, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        item_kind = item.get("kind")
        if item_kind == "workspace-file":
            if set(item) != {"kind", "path", "media_type", "identifier"}:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            path = item.get("path")
            media_type = item.get("media_type")
            identifier = item.get("identifier")
            if not isinstance(path, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if media_type is not None and not isinstance(media_type, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if identifier is not None and not isinstance(identifier, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decoded.append(WorkspaceFileInput(path, media_type, identifier))
            continue
        if item_kind == "text":
            if set(item) != {"kind", "text"} or not isinstance(item.get("text"), str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decoded.append(cast(str, item["text"]))
            continue
        if item_kind == "native":
            if (
                set(item) != {"kind", "codec", "value"}
                or item.get("codec") != _USER_CONTENT_CODEC
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            native = item.get("value")
            if not isinstance(native, Mapping):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            content = decode_user_content_payload(
                cast(Mapping[str, JsonValue], native)
            )
            if len(content) != 1:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            decoded.append(content[0])
            continue
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED)
    result = validate_user_input(tuple(decoded))
    if value["intent"] != _draft_prompt(result):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return result


class _InputFileSource(Protocol):
    async def canonicalize_path(self, path: str) -> str: ...

    async def read_bytes(
        self,
        path: str,
        *,
        max_bytes: "int | None" = None,
    ) -> bytes: ...

    async def close(self) -> None: ...


class ExecutionInputMaterializer:
    """Own input materialization and optional file-source reads."""

    def __init__(
        self,
        access: "_InputFileSource | None",
        limits: PromptLimits,
        *,
        object_store: ObjectStore | None = None,
        object_key_factory: "RuntimeObjectKeyFactory | None" = None,
        payload_policy: PayloadPolicy | None = None,
        object_domain: RuntimeDomain = RuntimeDomain.EXECUTION,
    ) -> None:
        if not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        self._access = access
        self._limits = limits
        self._object_store = object_store
        self._object_key_factory = object_key_factory
        self._payload_policy = payload_policy or PayloadPolicy()
        self._object_domain = object_domain
        self._mime = mimetypes.MimeTypes(filenames=())

    async def close(self) -> None:
        if self._access is not None:
            await self._access.close()

    @property
    def access(self) -> "_InputFileSource | None":
        return self._access

    async def canonicalize_files(self, files: Sequence[str]) -> tuple[str, ...]:
        raw_files = _require_files(files)
        if raw_files and self._access is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "files", "reason": "workspace_required"},
            )
        result: list[str] = []
        for path in raw_files:
            try:
                access = self._access
                if access is None:
                    raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
                canonical = normalize_workspace_input_path(
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
            result.append(canonical)
        return tuple(result)

    async def canonicalize_input(
        self,
        value: _UserPromptInput,
    ) -> CanonicalUserInput:
        canonical = validate_user_input(value)
        if isinstance(canonical, str):
            return canonical
        if not any(isinstance(item, WorkspaceFileInput) for item in canonical):
            return canonical
        if self._access is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "user_prompt", "reason": "workspace_required"},
            )
        values: list[UserContent | WorkspaceFileInput] = []
        for item in canonical:
            if not isinstance(item, WorkspaceFileInput):
                values.append(item)
                continue
            try:
                # Keep request intent independent of the file's current
                # existence; the actual read below is the authorization and
                # freeze boundary.
                path = normalize_workspace_input_path(item.path)
            except (AIError, TypeError, ValueError) as error:
                mapped = _file_request_error(
                    error
                    if isinstance(error, AIError)
                    else AIError(ErrorCode.REQUEST_FIELD_INVALID),
                    request_invalid_reason="path_invalid",
                    field="user_prompt",
                )
                if mapped is None:
                    raise
                raise mapped from error
            values.append(
                WorkspaceFileInput(
                    path,
                    media_type=item.media_type,
                    identifier=item.identifier,
                )
            )
        return tuple(values)

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
        workspace_inputs = (
            ()
            if isinstance(canonical, str)
            else tuple(
                item
                for item in canonical
                if isinstance(item, WorkspaceFileInput)
            )
        )
        direct_binary = _binary_parts(canonical)
        if len(direct_binary) + len(workspace_inputs) > self._limits.max_binary_input_parts:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        total_bytes = sum(len(item.data) for item in direct_binary)
        if total_bytes > self._limits.max_binary_input_bytes:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (
            len(direct_binary) + len(workspace_inputs) + len(files)
            > self._limits.max_binary_input_parts
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not files and not workspace_inputs:
            return canonical
        access = self._access
        if access is None:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "files", "reason": "workspace_required"},
            )

        additions: list[UserContent] = []
        file_views: list[dict[str, JsonValue]] = []
        materialized_items: list[UserContent] = []
        if isinstance(canonical, str):
            materialized_items.append(canonical)
        else:
            for item in canonical:
                if not isinstance(item, WorkspaceFileInput):
                    materialized_items.append(item)
                    continue
                remaining = self._limits.max_binary_input_bytes - total_bytes
                if remaining < 0:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                addition, view = await self._materialize_workspace_file(
                    item,
                    max_bytes=remaining,
                )
                total_bytes += int(view["size"])
                if total_bytes > self._limits.max_binary_input_bytes:
                    raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
                materialized_items.extend(addition)
                file_views.append(view)
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
            file_addition = (
                (
                    f"Workspace file path: {json.dumps(path)}",
                    BinaryContent(
                        data=body,
                        media_type=media_type,
                    ),
                )
            )
            additions.extend(file_addition)
        materialized_items.extend(additions)
        materialized: CanonicalUserInput = tuple(materialized_items)
        validate_user_content(materialized)
        _logger.info(
            "execution input materialized: files=%s binary_bytes=%s",
            len(files) + len(workspace_inputs),
            total_bytes,
        )
        return cast(
            CanonicalUserInput,
            MaterializedUserContent(
                cast(Sequence[UserContent], materialized),
                _input_view(canonical, file_views),
            ),
        )

    async def _materialize_workspace_file(
        self,
        item: WorkspaceFileInput,
        *,
        max_bytes: int,
    ) -> tuple[tuple[UserContent, ...], dict[str, JsonValue]]:
        access = self._access
        if access is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        media_type = item.media_type or self._media_type(
            item.path,
            field="user_prompt",
        )
        try:
            body = await access.read_bytes(
                item.path,
                max_bytes=max_bytes,
            )
        except AIError as error:
            mapped = _file_request_error(
                error,
                request_invalid_reason="file_invalid",
                field="user_prompt",
            )
            if mapped is None:
                raise
            raise mapped from error
        view: dict[str, JsonValue] = {
            "path": item.path,
            "media_type": media_type,
            "size": len(body),
            "digest": hashlib.sha256(body).hexdigest(),
            "identifier": item.identifier,
            "input_identifier": item.identifier,
            "_prompt_occurrence": True,
        }
        return (
            (
                f"Workspace file path: {json.dumps(item.path)}",
                BinaryContent(
                    data=body,
                    media_type=media_type,
                    identifier=item.identifier,
                ),
            ),
            view,
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
            if isinstance(value, MaterializedUserContent)
            else _input_view(value)
        )
        payload = StoredPayload.inline_json(_encode_user_content(canonical))
        if not payload_fits_inline(payload, self._payload_policy):
            if self._object_store is None or self._object_key_factory is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            body = canonical_json_bytes(cast(JsonValue, payload.value))
            from ._object import RuntimeObjectKeyFactory, put_runtime_object

            if not isinstance(self._object_key_factory, RuntimeObjectKeyFactory):
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            reference = await put_runtime_object(
                self._object_store,
                self._object_key_factory,
                self._object_domain,
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
        content = _decode_user_content(cast(dict[str, JsonValue], decoded))
        return (
            MaterializedUserContent(content, value.view)
            if value.view is not None else content
        )

    def _media_type(self, path: str, *, field: str = "files") -> str:
        media_type, _ = self._mime.guess_type(path, strict=False)
        if not media_type:
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={
                    "field": field,
                    "reason": "media_type_unknown",
                },
            )
        return media_type


def _file_request_error(
    error: AIError,
    *,
    request_invalid_reason: str,
    field: str = "files",
) -> AIError | None:
    if error.code is ErrorCode.REQUEST_FIELD_INVALID:
        return AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={
                "field": field,
                "reason": request_invalid_reason,
            },
        )
    if error.code is ErrorCode.STORAGE_NOT_FOUND:
        return AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={"field": field, "reason": "file_not_found"},
        )
    if error.code is ErrorCode.AUTHORIZATION_DENIED:
        return AIError(
            ErrorCode.REQUEST_FIELD_INVALID,
            retryable=False,
            safe_details={"field": field, "reason": "path_not_allowed"},
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
            canonical = normalize_workspace_input_path(path)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID) from error
        if canonical != path:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return files


def _binary_parts(
    content: Sequence[UserContent | WorkspaceFileInput],
) -> tuple[BinaryContent, ...]:
    return tuple(item for item in content if isinstance(item, BinaryContent))


def _draft_prompt(value: _UserPromptInput) -> DraftPrompt:
    canonical = validate_user_input(value)
    if isinstance(canonical, str):
        return {"kind": "text", "text": canonical}
    result: list[JsonValue] = []
    for item in canonical:
        if isinstance(item, str):
            result.append({"kind": "text", "text": item})
        elif isinstance(item, WorkspaceFileInput):
            result.append(
                {
                    "kind": "workspace-file",
                    "path": item.path,
                    "media_type": item.media_type,
                    "identifier": item.identifier,
                }
            )
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
                "files": [
                    {
                        key: value
                        for key, value in item.items()
                        if key != "_prompt_occurrence"
                    }
                    for item in files
                ],
                "attachments": list(input_attachment_views(canonical, files)),
            }
        )
    except (TypeError, ValueError) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    if not isinstance(normalized, dict):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return normalized


def stored_input_attachment_views(
    value: "StoredUserInput",
) -> tuple[Mapping[str, JsonValue], ...]:
    from .state._contracts import StoredUserInput

    if not isinstance(value, StoredUserInput):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    view = value.view
    if view is None:
        return ()
    raw = view.get("attachments")
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    result: list[Mapping[str, JsonValue]] = []
    for item in raw:
        if not isinstance(item, Mapping):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        result.append(dict(item))
    return tuple(result)


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
    return {"items": [encode_user_content_item(item) for item in content]}


def _encode_task_prompt_item(item: UserContent | WorkspaceFileInput) -> JsonValue:
    if isinstance(item, str):
        return {"kind": "text", "text": item}
    if isinstance(item, WorkspaceFileInput):
        return {
            "kind": "workspace-file",
            "path": item.path,
            "media_type": item.media_type,
            "identifier": item.identifier,
        }
    return {
        "kind": "native",
        "codec": _USER_CONTENT_CODEC,
        "value": _encode_user_content((item,)),
    }


def _decode_user_content(payload: dict[str, JsonValue]) -> tuple[UserContent, ...]:
    if set(payload) != {"items"} or not isinstance(payload.get("items"), list):
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    try:
        content = tuple(
            decode_user_content_item(item)
            for item in cast(list[object], payload["items"])
        )
    except AIError:
        raise
    except UnsupportedUserContentKind as error:
        raise AIError(ErrorCode.STORAGE_VERSION_UNSUPPORTED) from error
    except (TypeError, ValueError, KeyError, binascii.Error) as error:
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
    validate_user_content(content)
    return content



__all__ = [
    "CanonicalUserInput",
    "ExecutionInputMaterializer",
    "InputIntent",
    "decode_task_prompt_draft",
    "decode_user_content_payload",
    "stored_input_attachment_views",
    "input_intent",
    "stored_user_input_view",
    "task_prompt_draft",
    "validate_user_input",
]
