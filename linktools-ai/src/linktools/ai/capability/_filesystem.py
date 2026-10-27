#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt one opened workspace backend to Pydantic AI tools."""

import asyncio
import hashlib
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from copy import copy
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar, cast

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, InstructionPart, ToolReturn, UserContent

from pydantic_ai.toolsets import AbstractToolset, FunctionToolset

from ..core import PromptLimits
from ..model import ModelBinding, validate_model_input, validate_model_inputs
from ..errors import AIError, ErrorCode
from ..workspace import (
    WorkspaceOperationRejected,
    SupportsFileTools,
    Workspace,
    WorkspacePolicy,
)
from ._context import AgentContext
from ._workspace_policy import workspace_policy_toolset
from ._tool_signal import ToolCallFailed, ToolCallRetry
from ._tool_metadata import tool_effect_policy_from_metadata, tool_metadata

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class FileSystemToolDeclaration:
    """Immutable metadata declaration for one Workspace tool."""

    name: str
    metadata: Mapping[str, object]
    revision: int = 1

    def __post_init__(self) -> None:
        immutable_metadata = _immutable_tool_metadata(self.metadata)
        object.__setattr__(self, "metadata", MappingProxyType(immutable_metadata))


def _immutable_tool_metadata(
    metadata: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: tuple(value) if isinstance(value, list) else value
        for key, value in metadata.items()
    }


def _runtime_tool_metadata(
    declaration: FileSystemToolDeclaration,
) -> dict[str, object]:
    metadata = dict(declaration.metadata)
    path_fields = metadata.get("linktools.ai.path_fields")
    if isinstance(path_fields, tuple):
        metadata["linktools.ai.path_fields"] = list(path_fields)
    return metadata


_FILESYSTEM_TOOL_DECLARATIONS = (
    FileSystemToolDeclaration("list_files", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    )),
    FileSystemToolDeclaration("read_file", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
        context_dedupe="workspace_file_read_v1",
    )),
    FileSystemToolDeclaration("search_files", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    ), revision=1),
    FileSystemToolDeclaration("write_file", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    ), revision=1),

)
_FILESYSTEM_TOOL_DECLARATIONS_BY_NAME = {
    declaration.name: declaration
    for declaration in _FILESYSTEM_TOOL_DECLARATIONS
}
_EFFECTFUL_WORKSPACE_TOOLS = frozenset(
    {
        declaration.name
        for declaration in _FILESYSTEM_TOOL_DECLARATIONS
        if tool_effect_policy_from_metadata(
            _runtime_tool_metadata(declaration),
            require=True,
        )
        != "none"
    }
)
_FILESYSTEM_CAPABILITY_ID = "filesystem"
_WORKSPACE_INSTRUCTIONS = InstructionPart(
    content=(
        "Workspace file-tool paths are relative to the logical Workspace root unless "
        "the tool contract says otherwise. Prefer workspace-relative paths and do not "
        "infer host absolute paths. A visible Workspace tool is not automatically "
        "authorized: follow approval requirements and do not switch tools to bypass a "
        "denied or restricted operation."
    ),
    name="workspace",
    dynamic=False,
)
_logger = environ.get_logger("ai.capability.workspace")
_WORKSPACE_REJECTED_CODES = frozenset({
    ErrorCode.REQUEST_FIELD_INVALID,
    ErrorCode.STORAGE_NOT_FOUND,
    ErrorCode.AUTHORIZATION_DENIED,
    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
})
_INVALID_WORKSPACE_REQUEST = (
    "The workspace tool arguments or target are invalid. Correct them and retry."
)
_MISSING_WORKSPACE_TARGET = (
    "The requested workspace path does not exist, or its parent directory is missing. "
    "Correct the path and retry."
)
_WORKSPACE_CONFLICT = (
    "The workspace changed since it was read. Read the target again and retry with "
    "the current hash."
)
_DISALLOWED_WORKSPACE_TARGET = (
    "The requested workspace path or command is not allowed. Choose an allowed target "
    "or command and retry."
)
_LARGE_WORKSPACE_REQUEST = (
    "The workspace tool arguments are too large. Use a smaller request and retry."
)
_TOO_MANY_WORKSPACE_COMMANDS = (
    "Too many workspace operations are pending. Reuse, inspect, or stop existing "
    "operations before starting another one."
)
_UNSUPPORTED_IMAGE_INPUT = (
    "The current model does not support image attachments. "
    "Use a non-image input or another approach."
)
_UNRECOGNIZED_ATTACHMENT_TYPE = (
    "A requested attachment has no recognized media type. Use a workspace file with "
    "a recognized file extension or inspect it with another tool."
)
_INVALID_UTF8_WORKSPACE_CONTENT = (
    "The requested workspace content is not valid UTF-8 text. Use a binary-capable "
    "approach or choose a UTF-8 text target."
)
_INVALID_WORKSPACE_REQUESTS = {
    "read_file": (
        "The read_file arguments are invalid. Use a non-negative offset and a positive "
        "limit; if the offset is past the file, retry from an earlier offset."
    ),
    "write_file": (
        "The write_file arguments are invalid. Use a writable workspace path and valid "
        "UTF-8 text; if expected_hash is supplied, use the current file hash."
    ),
    "search_files": (
        "The search_files arguments are invalid. Use a valid regular expression, a "
        "valid workspace path, and a relative include_glob when provided."
    ),
    "list_files": "Use a valid workspace directory and a relative include_glob.",
}


class _FileSystemToolSurface:
    def __init__(
        self,
        backend: SupportsFileTools | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        model_binding: ModelBinding | None = None,
    ) -> None:
        self._backend = backend
        self._policy = policy
        self._limits = limits
        self._model_binding = model_binding

    def _require_backend(self) -> SupportsFileTools:
        if self._backend is None:
            raise RuntimeError("workspace sandbox backend is not open")
        return self._backend

    @staticmethod
    async def _call(
        operation: Awaitable[_ResultT],
        *,
        name: str,
        rejected_codes: frozenset[ErrorCode] = _WORKSPACE_REJECTED_CODES,
        rejected_message: str = _INVALID_WORKSPACE_REQUEST,
    ) -> _ResultT:
        try:
            return await operation
        except WorkspaceOperationRejected as error:
            if error.code not in rejected_codes:
                raise
            raise _filesystem_tool_rejected(name, error, rejected_message) from error
        except AIError as error:
            if name in _EFFECTFUL_WORKSPACE_TOOLS or error.code not in rejected_codes:
                raise
            raise _filesystem_tool_rejected(name, error, rejected_message) from error



    async def read_file(
        self,
        path: str | list[str],
        *,
        as_file: bool = False,
        offset: int | None = None,
        limit: int | None = None,
        cursor: str | None = None,
        full_metadata: bool = False,
    ) -> ToolReturn:
        """Read text or supported media; as_file sends whole files. Lists are atomic batches.

        Ranges/metadata require one text preview: zero-based offset, default 200 lines.
        cursor excludes offset; full_metadata scans for SHA-256 and line count.
        """
        return await self._call(
            self._read_files_bounded(path, as_file=as_file, offset=offset,
                                     limit=limit, cursor=cursor,
                                     full_metadata=full_metadata),
            name="read_file",
            rejected_codes=_WORKSPACE_REJECTED_CODES | {ErrorCode.STORAGE_CONFLICT},
        )

    async def _read_files_bounded(
        self, path: str | list[str], *, as_file: bool, offset: int | None,
        limit: int | None, cursor: str | None, full_metadata: bool,
    ) -> ToolReturn:
        try:
            return await asyncio.wait_for(
                self._read_files(path, as_file=as_file, offset=offset, limit=limit,
                                 cursor=cursor, full_metadata=full_metadata),
                timeout=30.0,
            )
        except asyncio.TimeoutError as error:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "file_batch_timeout"}) from error

    async def _read_files(
        self, path: str | list[str], *, as_file: bool, offset: int | None,
        limit: int | None, cursor: str | None, full_metadata: bool,
    ) -> ToolReturn:
        paths = [path] if isinstance(path, str) else path
        if (not isinstance(paths, list) or not paths
                or any(not isinstance(item, str) or not item for item in paths)
                or len(paths) > self._limits.max_binary_input_parts):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "file_batch_invalid"})
        ranged = offset is not None or limit is not None or cursor is not None or full_metadata
        if (type(as_file) is not bool or type(full_metadata) is not bool
                or ranged and (len(paths) != 1 or as_file)
                or cursor is not None and offset is not None):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "text_range_invalid"})
        backend = self._require_backend()
        facts = []
        probe_bytes = 0
        source_budget = self._limits.max_binary_input_bytes + 8192 * len(paths)
        for item in paths:
            remaining = source_budget - probe_bytes
            if remaining <= 0:
                raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                              safe_details={"reason": "file_batch_bytes_limit"})
            fact = await backend.inspect_file(item, max_bytes=remaining)
            probe_bytes += cast(int, fact["bytes_read"])
            facts.append(fact)
        if ranged and not facts[0]["is_text"]:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                          safe_details={"reason": "text_range_invalid"})
        # Reserve metadata before reading bodies so a large batch cannot silently
        # omit an entry after exhausting its shared output budget.
        output_limit = self._limits.max_file_output_bytes
        reserve = sum(len(item.encode("utf-8")) + 2048 for item in paths)
        if reserve >= output_limit:
            raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                          safe_details={"reason": "file_batch_output_limit"})
        total_bytes = 0
        source_bytes = probe_bytes
        output_bytes = reserve
        texts: list[dict[str, object]] = []
        files: list[dict[str, object]] = []
        content: list[UserContent] = []
        for index, (item, fact) in enumerate(zip(paths, facts)):
            if fact["is_text"] and not as_file:
                available = min(output_limit - output_bytes,
                                self._limits.max_binary_input_bytes - total_bytes)
                if available < 4:
                    raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                                  safe_details={"reason": "file_batch_output_limit"})
                text = dict(await backend.read_text(
                    item, offset=offset, limit=limit, cursor=cursor,
                    full_metadata=full_metadata, max_bytes=available,
                    max_read_bytes=source_budget - source_bytes,
                ))
                if text.pop("state") != fact["state"]:
                    raise AIError(ErrorCode.STORAGE_CONFLICT,
                                  safe_details={"reason": "file_changed"})
                source_bytes += cast(int, text.pop("bytes_read"))
                text.update(input_index=index, mime_type=fact["mime_type"],
                            type_source=fact["type_source"])
                if "detection_reason" in fact:
                    text["detection_reason"] = fact["detection_reason"]
                text_bytes = len(str(text["content"]).encode("utf-8"))
                total_bytes += text_bytes
                output_bytes += text_bytes
                texts.append(text)
                continue
            media_type = fact["mime_type"]
            if not isinstance(media_type, str) or not media_type:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                              safe_details={"reason": "media_type_unknown"})
            size = fact["size"]
            if not isinstance(size, int):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            remaining = self._limits.max_binary_input_bytes - total_bytes
            if size > remaining or size > source_budget - source_bytes:
                raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                              safe_details={"reason": "file_batch_bytes_limit"})
            if self._model_binding is not None:
                try:
                    validate_model_input(self._model_binding, media_type)
                except AIError as error:
                    raise _file_input_error(error, item, media_type, size) from error
            body = await backend.read_bytes(item, max_bytes=remaining,
                                            expected_state=cast(str, fact["state"]))
            binary = BinaryContent(data=body, media_type=media_type)
            if self._model_binding is not None:
                try:
                    validate_model_inputs(self._model_binding, (binary,))
                except AIError as error:
                    raise _file_input_error(error, item, media_type, size) from error
            total_bytes += len(body)
            source_bytes += len(body)
            files.append({"path": item, "input_index": index,
                          "mime_type": media_type, "type_source": fact["type_source"],
                          "size": len(body), "sha256": hashlib.sha256(body).hexdigest()})
            if "detection_reason" in fact:
                files[-1]["detection_reason"] = fact["detection_reason"]
            content.append(binary)
        value = {"files": files, "texts": texts}
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > output_limit:
            raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                          safe_details={"reason": "file_batch_output_limit"})
        return ToolReturn(return_value=[value, *content] if content else value)

    async def write_file(
        self, path: str, content: str, *, old_text: str | None = None,
        expected_hash: str | None = None,
    ) -> str:
        """Create/overwrite with content, or replace one unique nonempty old_text.

        Empty content is valid. Optional expected_hash checks current raw-byte SHA-256.
        """
        if not isinstance(content, str) or old_text is not None and (
            not isinstance(old_text, str) or not old_text
        ):
            raise ToolCallRetry("content must be text; old_text must be nonempty or omitted.")
        backend = self._require_backend()
        operation = backend.write_file(path, content, old_text=old_text, expected_hash=expected_hash)
        rejected_codes = _WORKSPACE_REJECTED_CODES
        if expected_hash is not None:
            rejected_codes |= {ErrorCode.STORAGE_CONFLICT}
        return await self._call(operation, name="write_file", rejected_codes=rejected_codes)

    async def list_files(self, path: str = ".", include_glob: str = "*") -> str:
        """List names and type hints; only ** globs recurse below path."""
        return await self._call(self._require_backend().list_files(
            path, include_glob=include_glob), name="list_files")



    async def search_files(
        self,
        pattern: str,
        *,
        path: str = ".",
        include_glob: str | None = None,
    ) -> str:
        """Search text recursively by regex, returning matching lines and coverage limits.

        include_glob is relative to path: * stays in one level; ** spans zero or more.
        """
        return await self._call(
            self._require_backend().search_text(
                pattern,
                path=path,
                include_glob=include_glob,
            ),
            name="search_files",
        )






class _FileSystemToolset(FunctionToolset[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        backend: SupportsFileTools | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        model_binding: ModelBinding | None = None,
    ) -> None:
        super().__init__(id=_FILESYSTEM_CAPABILITY_ID)
        surface = _FileSystemToolSurface(
            backend,
            policy,
            limits,
            model_binding=model_binding,
        )
        for name in selected_tool_names:
            self.add_tool(
                _filesystem_tool(
                    surface,
                    name,
                    _runtime_tool_metadata(_declaration_for(name)),
                )
            )




class FileSystem(AbstractCapability[AgentContext[object]]):
    """Expose selected tools through a caller-owned sandbox backend.

    Native bindings enforce the Workspace tool policy. Runtime supplies its own
    admitted execution boundary; both borrow, rather than close, the backend.
    """

    id: str | None = "runtime.filesystem"

    def __init__(self, workspace: Workspace) -> None:
        if not isinstance(workspace, Workspace):
            raise TypeError("workspace must be Workspace")
        self._workspace = workspace
        self._bound = False
        self._toolset: AbstractToolset[AgentContext[object]] | None = None

    @property
    def workspace(self) -> Workspace:
        return self._workspace

    def tool_definitions(self) -> tuple[Tool[Any], ...]:
        """Return stable declarations for Agent selection and durable bindings."""
        return _filesystem_tool_definitions(self.workspace)

    def bind(
        self,
        selected_tool_names: Sequence[str],
        *,
        backend: SupportsFileTools | None,
        limits: PromptLimits | None = None,
        model_binding: ModelBinding | None = None,
        wrap_toolset: Callable[
            [AbstractToolset[AgentContext[object]]],
            AbstractToolset[AgentContext[object]],
        ] | None = None,
    ) -> "FileSystem":
        """Build tools without modifying this declaration or owning the backend.

        A supplied wrap_toolset owns the execution boundary, including policy
        enforcement. Without one, the native SDK handles this Workspace policy.
        """
        selected_limits = PromptLimits() if limits is None else limits
        if not isinstance(selected_limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        selected = frozenset(selected_tool_names)
        unknown = selected.difference(_FILESYSTEM_TOOL_DECLARATIONS_BY_NAME)
        if unknown:
            raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
        if selected and backend is None:
            raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
        bound = copy(self)
        bound._bound = True
        bound._toolset = None
        if selected:
            toolset = _FileSystemToolset(
                tuple(item.name for item in filesystem_tool_declarations() if item.name in selected),
                backend, self.workspace.policy, selected_limits, model_binding=model_binding,
            )
            policy = self.workspace.policy.tool_permissions
            if wrap_toolset is not None:
                bound._toolset = wrap_toolset(toolset)
            else:
                bound._toolset = workspace_policy_toolset(toolset, policy)
        return bound

    def get_instructions(self) -> InstructionPart | None:
        return _WORKSPACE_INSTRUCTIONS if self._bound and self._toolset is not None else None

    def get_toolset(self) -> AbstractToolset[AgentContext[object]] | None:
        if not self._bound:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self._toolset


def _filesystem_tool_definitions(workspace: Workspace) -> tuple[Tool[Any], ...]:
    """Return stable Workspace tool definitions before execution materialization."""
    surface = _FileSystemToolSurface(None, workspace.policy, PromptLimits())
    return tuple(
        _filesystem_tool(
            surface,
            declaration.name,
            _runtime_tool_metadata(declaration),
        )
        for declaration in filesystem_tool_declarations()
    )


def filesystem_tool_declarations() -> tuple[FileSystemToolDeclaration, ...]:
    """Return the canonical Workspace tool declarations in name order."""
    return tuple(
        sorted(_FILESYSTEM_TOOL_DECLARATIONS, key=lambda item: item.name)
    )


def _declaration_for(name: str) -> FileSystemToolDeclaration:
    try:
        return _FILESYSTEM_TOOL_DECLARATIONS_BY_NAME[name]
    except KeyError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error


def filesystem_capabilities(
    workspace: Workspace,
    selected_tool_names: Sequence[str],
    *,
    limits: "PromptLimits | None" = None,
    backend: SupportsFileTools | None = None,
    model_binding: ModelBinding | None = None,
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Adapt selected tools to a caller-owned, already-opened backend."""
    bound = FileSystem(workspace).bind(
        selected_tool_names, backend=backend, limits=limits, model_binding=model_binding,
    )
    return () if bound.get_toolset() is None else (bound,)


def _filesystem_tool(
    surface: _FileSystemToolSurface,
    name: str,
    metadata: Mapping[str, object],
) -> Tool[Any]:
    return Tool(
        cast(Any, getattr(surface, name)),
        takes_ctx=False,
        name=name,
        metadata=dict(metadata),
    )


_FILE_REASON_MESSAGES = {
    'model_input_not_supported': 'This model route explicitly does not support this file type. Choose a compatible route or an authorized conversion.',
    'input_encoding_not_supported': 'The current provider endpoint cannot encode this input. Use a supported transport or an authorized conversion.',
    'model_input_mime_type_unknown': 'The input has no known media type. Provide a recognized file type or explicitly convert it with an authorized tool.',
    'uploaded_file_provider_mismatch': 'The uploaded file belongs to a different provider. Use a file reference for this provider or an authorized local copy.',
    'text_input_not_utf8': 'Whole-file text input must be valid UTF-8. Use a bounded text preview to inspect decoding loss, or explicitly convert the source.',
    'file_batch_timeout': 'The file batch exceeded its time budget. Retry with fewer files or a smaller text range.',
    'file_batch_invalid': 'Supply one path or a nonempty path array within the batch item limit; split larger batches.',
    'file_batch_bytes_limit': 'The file batch exceeds its byte budget. Split the batch or use smaller supported files.',
    'file_batch_output_limit': 'The file batch cannot fit its output budget. Use fewer files or a smaller text range.',
    'text_range_invalid': 'offset, limit, cursor and full_metadata require exactly one text preview; cursor cannot be combined with offset.',
    'file_byte_limit': 'The file exceeds the remaining byte budget. Use a smaller file or split the batch.',
    'stale_cursor': 'The source changed after this cursor was issued. Read the current file again to obtain a new cursor.',
    'file_changed': 'The source changed during the operation. Read the current file again before retrying.',
    'invalid_cursor': 'The cursor is malformed. Use the next_cursor returned by a successful read of this file.',
    'cursor_path_mismatch': 'This cursor belongs to another path. Continue the original path or start a fresh read.',
    'invalid_cursor_position': 'The cursor position is outside this file. Start a fresh read.',
    'invalid_cursor_boundary': 'The cursor does not start at a complete text character boundary. Use a returned cursor or start a fresh read.',
    'offset_out_of_range': 'The requested line offset is past the file. Start at an earlier offset.',
    'metadata_byte_limit': 'Full metadata exceeds the scan budget. Use a bounded preview without full_metadata or an authorized external hash operation.',
    'scan_byte_limit': 'Finding the requested range exceeds the scan budget. Continue from an existing cursor or choose an earlier offset.',
    'scan_time_limit': 'The text scan exceeded its time budget. Use a smaller range or continue from an existing cursor.',
    'edit_scan_byte_limit': 'The edit exceeds the source scan budget. Use an authorized large-file editing workflow.',
    'edit_scan_time_limit': 'The edit exceeded its scan time budget. Read the current source before choosing a smaller authorized editing workflow.',
    'edit_staging_byte_limit': 'The edited file exceeds the temporary output budget. Use a smaller replacement or an authorized large-file editing workflow.',
    'no_match': 'old_text does not occur in the current file. Read the target and supply old_text that matches exactly once.',
    'multiple_matches': 'old_text occurs more than once. Include enough surrounding text to identify exactly one match.',
    'invalid_text': 'The source contains binary content that cannot be safely edited as text. Use an authorized format-specific editor.',
}



def _file_input_error(error: AIError, path: str, mime_type: str, size: int) -> AIError:
    details = dict(error.safe_details)
    details.update(mime_type=mime_type, size=size)
    # Only logical workspace-relative paths may enter model-visible failures.
    if path and not path.startswith("/") and "\\" not in path and ":" not in path and ".." not in path.split("/"):
        details["path"] = path
    return AIError(error.code, category=error.category, retryable=error.retryable,
                   operation_id=error.operation_id, safe_details=details,
                   diagnostics=error.diagnostics)


def _filesystem_tool_rejected(
    name: str,
    error: AIError,
    default_message: str,
) -> ToolCallRetry:
    reason = error.safe_details.get("reason")
    if name == "read_file" and reason == "image_input_not_supported":
        message = _UNSUPPORTED_IMAGE_INPUT
    elif name == "read_file" and reason == "media_type_unknown":
        message = _UNRECOGNIZED_ATTACHMENT_TYPE
    elif isinstance(reason, str) and reason in _FILE_REASON_MESSAGES:
        message = _FILE_REASON_MESSAGES[reason]
    elif reason == "invalid_utf8":
        message = _INVALID_UTF8_WORKSPACE_CONTENT
    elif error.code is ErrorCode.STORAGE_NOT_FOUND:
        message = _MISSING_WORKSPACE_TARGET
    elif error.code is ErrorCode.STORAGE_CONFLICT:
        message = _WORKSPACE_CONFLICT
    elif error.code is ErrorCode.AUTHORIZATION_DENIED:
        message = _DISALLOWED_WORKSPACE_TARGET
    elif error.code is ErrorCode.TOOL_ARGUMENTS_TOO_LARGE:
        message = _LARGE_WORKSPACE_REQUEST
    elif error.code is ErrorCode.TOO_MANY_PENDING_OPERATIONS:
        message = _TOO_MANY_WORKSPACE_COMMANDS
    else:
        message = _INVALID_WORKSPACE_REQUESTS.get(name, default_message)
    _logger.debug(
        "workspace tool call retry: operation=%s code=%s",
        name,
        error.code.value,
    )
    if name == "read_file" and "size" in error.safe_details and "mime_type" in error.safe_details:
        facts = {key: value[:160] if isinstance(value, str) else value
                 for key, value in error.safe_details.items()
                 if key in {"path", "size", "mime_type", "reason"}}
        message += " File: " + json.dumps(facts, ensure_ascii=False)
    return ToolCallRetry(message)


__all__ = [
    "FileSystem",
    "FileSystemToolDeclaration",
    "filesystem_tool_declarations",
    "filesystem_capabilities",
]
