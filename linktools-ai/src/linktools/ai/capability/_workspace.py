#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Adapt one opened workspace session to Pydantic AI tools."""

import asyncio
import hashlib
import json
import mimetypes
from collections.abc import Awaitable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, TypeVar, cast

from linktools.core import environ
from pydantic_ai import Tool
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.messages import BinaryContent, InstructionPart, ToolReturn, UserContent
from pydantic_ai.toolsets import FunctionToolset

from ..core import PromptLimits
from ..model import ModelBinding, validate_model_input, validate_model_inputs
from ..errors import AIError, ErrorCode
from ..workspace import (
    Sandbox,
    SandboxOperationRejected,
    SandboxSession,
    Workspace,
    WorkspacePolicy,
)
from ._context import AgentContext
from ._tool_signal import ToolCallRetry
from ._tool_metadata import tool_effect_policy_from_metadata, tool_metadata

_ResultT = TypeVar("_ResultT")


@dataclass(frozen=True, slots=True)
class WorkspaceToolDeclaration:
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
    declaration: WorkspaceToolDeclaration,
) -> dict[str, object]:
    metadata = dict(declaration.metadata)
    path_fields = metadata.get("linktools.ai.path_fields")
    if isinstance(path_fields, tuple):
        metadata["linktools.ai.path_fields"] = list(path_fields)
    return metadata


_WORKSPACE_TOOL_DECLARATIONS = (
    WorkspaceToolDeclaration("attach_files", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("paths",),
    )),
    WorkspaceToolDeclaration("create_directory", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    )),
    WorkspaceToolDeclaration("edit_file", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    )),
    WorkspaceToolDeclaration("file_info", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    )),
    WorkspaceToolDeclaration("find_files", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    )),
    WorkspaceToolDeclaration("list_directory", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    )),
    WorkspaceToolDeclaration("list_files", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    )),
    WorkspaceToolDeclaration("read_file", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
        context_dedupe="workspace_file_read_v2",
    ), revision=2),
    WorkspaceToolDeclaration("search_files", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="filesystem.read",
        path_fields=("path",),
    ), revision=2),
    WorkspaceToolDeclaration("write_file", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="filesystem.write",
        path_fields=("path",),
    ), revision=2),
    WorkspaceToolDeclaration("check_command", tool_metadata(
        effect_policy="none",
        plan_safe=True,
        tool_class="shell",
    )),
    WorkspaceToolDeclaration("run_command", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="shell",
    )),
    WorkspaceToolDeclaration("start_command", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="shell",
    )),
    WorkspaceToolDeclaration("stop_command", tool_metadata(
        effect_policy="non_replay_safe",
        tool_class="shell",
    )),
)
_WORKSPACE_TOOL_DECLARATIONS_BY_NAME = {
    declaration.name: declaration
    for declaration in _WORKSPACE_TOOL_DECLARATIONS
}
_EFFECTFUL_WORKSPACE_TOOLS = frozenset(
    {
        declaration.name
        for declaration in _WORKSPACE_TOOL_DECLARATIONS
        if tool_effect_policy_from_metadata(
            _runtime_tool_metadata(declaration),
            require=True,
        )
        != "none"
    }
)
_WORKSPACE_SANDBOX_CAPABILITY_ID = "workspace-sandbox"
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
    "attach_files": (
        "The attachment request is invalid. Provide at least one existing workspace "
        "file with a recognized type and reduce the number or total size of files "
        "before retrying."
    ),
    "read_file": (
        "The read_file arguments are invalid. Use a non-negative offset and a positive "
        "limit; if the offset is past the file, retry from an earlier offset."
    ),
    "write_file": (
        "The write_file arguments are invalid. Use a writable workspace path and valid "
        "UTF-8 text; if expected_hash is supplied, use the current file hash."
    ),
    "edit_file": (
        "The edit_file arguments are invalid. Read the target and retry with non-empty "
        "old_text that matches exactly once, valid UTF-8 replacement text, and the "
        "current hash if supplied."
    ),
    "list_directory": (
        "The list_directory target is invalid. Use a valid workspace directory path "
        "and retry."
    ),
    "search_files": (
        "The search_files arguments are invalid. Use a valid regular expression, a "
        "valid workspace path, and a relative include_glob when provided."
    ),
    "find_files": (
        "The find_files arguments are invalid. Use a relative glob pattern and a valid "
        "workspace directory path."
    ),
    "create_directory": (
        "The create_directory target is invalid. Use an allowed writable workspace "
        "path and retry."
    ),
    "file_info": (
        "The file_info target is invalid. Use a workspace path that resolves to a "
        "regular file or directory."
    ),
    "run_command": (
        "The run_command arguments are invalid. Use a permitted non-interactive "
        "command and a valid timeout, then retry."
    ),
    "start_command": (
        "The start_command arguments are invalid. Use a permitted non-interactive "
        "command and retry."
    ),
    "check_command": (
        "The command id is invalid or no longer active. Use an active command id "
        "returned by start_command."
    ),
    "stop_command": (
        "The command id is invalid or no longer active. Use an active command id "
        "returned by start_command."
    ),
}


class _WorkspaceToolSurface:
    def __init__(
        self,
        session: SandboxSession | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        vision: bool = True,
        model_binding: ModelBinding | None = None,
    ) -> None:
        self._session = session
        self._policy = policy
        self._limits = limits
        self._vision = vision
        self._model_binding = model_binding
        self._mime = mimetypes.MimeTypes(filenames=())

    def _require_session(self) -> SandboxSession:
        if self._session is None:
            raise RuntimeError("workspace sandbox session is not open")
        return self._session

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
        except SandboxOperationRejected as error:
            if error.code not in rejected_codes:
                raise
            raise _workspace_tool_rejected(name, error, rejected_message) from error
        except AIError as error:
            if name in _EFFECTFUL_WORKSPACE_TOOLS or error.code not in rejected_codes:
                raise
            raise _workspace_tool_rejected(name, error, rejected_message) from error

    async def attach_files(self, paths: list[str]) -> ToolReturn:
        """Attach Workspace files to the next model request.

        Args:
            paths: File paths relative to the root directory.

        Returns:
            Lightweight file metadata plus the file content for the next model request.
        """
        return await self._call(
            self._attach_files(paths),
            name="attach_files",
        )

    async def _attach_files(self, paths: list[str]) -> ToolReturn:
        if not paths:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        attachment_paths = tuple(paths)
        if len(attachment_paths) > self._limits.max_binary_input_parts:
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)

        session = self._require_session()
        total_bytes = 0
        metadata: list[dict[str, object]] = []
        content: list[UserContent] = []
        for path in attachment_paths:
            media_type, _ = self._mime.guess_type(path, strict=False)
            if not media_type:
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={
                        "field": "paths",
                        "reason": "media_type_unknown",
                    },
                )
            if self._model_binding is None and not self._vision and media_type.lower().startswith("image/"):
                raise AIError(
                    ErrorCode.REQUEST_FIELD_INVALID,
                    safe_details={
                        "field": "paths",
                        "reason": "image_input_not_supported",
                    },
                )
            remaining = self._limits.max_binary_input_bytes - total_bytes
            if remaining < 0:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            body = await session.read_bytes(path, max_bytes=remaining)
            if self._model_binding is not None:
                validate_model_inputs(self._model_binding, BinaryContent(body, media_type=media_type))
            total_bytes += len(body)
            if total_bytes > self._limits.max_binary_input_bytes:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
            metadata.append(
                {
                    "path": path,
                    "media_type": media_type,
                    "size": len(body),
                    "sha256": hashlib.sha256(body).hexdigest(),
                }
            )
            content.append(
                BinaryContent(
                    data=body,
                    media_type=media_type,
                )
            )
        return ToolReturn(
            return_value={"files": metadata},
            content=content,
        )

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
        session = self._require_session()
        facts = []
        probe_bytes = 0
        source_budget = self._limits.max_binary_input_bytes + 8192 * len(paths)
        for item in paths:
            remaining = source_budget - probe_bytes
            if remaining <= 0:
                raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                              safe_details={"reason": "file_batch_bytes_limit"})
            fact = await session.inspect_file(item, max_bytes=remaining)
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
                text = dict(await session.read_text(
                    item, offset=offset, limit=limit, cursor=cursor,
                    full_metadata=full_metadata, max_bytes=available,
                    max_read_bytes=source_budget - source_bytes,
                ))
                if text.pop("state") != fact["state"]:
                    raise AIError(ErrorCode.STORAGE_CONFLICT,
                                  safe_details={"reason": "file_changed"})
                source_bytes += cast(int, text.pop("bytes_read"))
                text.update(input_index=index, media_type=fact["media_type"],
                            type_source=fact["type_source"])
                if "detection_reason" in fact:
                    text["detection_reason"] = fact["detection_reason"]
                text_bytes = len(str(text["content"]).encode("utf-8"))
                total_bytes += text_bytes
                output_bytes += text_bytes
                texts.append(text)
                continue
            media_type = fact["media_type"]
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
            decision = None
            if self._model_binding is not None:
                decision = validate_model_input(self._model_binding, media_type, size=size)
            elif not self._vision and media_type.startswith("image/"):
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                              safe_details={"reason": "image_input_not_supported"})
            body = await session.read_bytes(item, max_bytes=remaining,
                                            expected_state=cast(str, fact["state"]))
            binary = BinaryContent(data=body, media_type=media_type)
            if self._model_binding is not None:
                validate_model_inputs(self._model_binding, (binary,))
            total_bytes += len(body)
            source_bytes += len(body)
            files.append({"path": item, "input_index": index,
                          "media_type": media_type, "type_source": fact["type_source"],
                          "size": len(body), "sha256": hashlib.sha256(body).hexdigest(),
                          "transport": decision.transport if decision else "unverified",
                          "verified": decision.verified if decision else False})
            if "detection_reason" in fact:
                files[-1]["detection_reason"] = fact["detection_reason"]
            content.append(binary)
        value = {"files": files, "texts": texts}
        if len(json.dumps(value, ensure_ascii=False).encode("utf-8")) > output_limit:
            raise AIError(ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                          safe_details={"reason": "file_batch_output_limit"})
        return ToolReturn(return_value=value, content=content or None)

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
        session = self._require_session()
        operation = (session.write_file(path, content, expected_hash=expected_hash)
                     if old_text is None else session.edit_file(
                         path, old_text, content, expected_hash=expected_hash))
        rejected_codes = _WORKSPACE_REJECTED_CODES
        if expected_hash is not None:
            rejected_codes |= {ErrorCode.STORAGE_CONFLICT}
        return await self._call(operation, name="write_file", rejected_codes=rejected_codes)

    async def list_files(self, path: str = ".", include_glob: str = "*") -> str:
        """List names and type hints; only ** globs recurse below path."""
        return await self._call(self._require_session().list_files(
            path, include_glob=include_glob), name="list_files")

    async def edit_file(
        self,
        path: str,
        old_text: str,
        new_text: str,
        *,
        expected_hash: str | None = None,
    ) -> str:
        """Edit a file by exact string replacement with conflict detection.

        The old_text must appear exactly once in the file. Include surrounding
        context lines to ensure uniqueness.

        Args:
            path: File path relative to the root directory.
            old_text: The exact text to find (must appear exactly once).
            new_text: The replacement text.
            expected_hash: If provided, rejects the edit when the file's
                current hash doesn't match (optimistic concurrency).

        Returns:
            Summary with new hash for subsequent operations.
        """
        rejected_codes = _WORKSPACE_REJECTED_CODES
        if expected_hash is not None:
            rejected_codes |= {ErrorCode.STORAGE_CONFLICT}
        return await self._call(
            self._require_session().edit_file(
                path,
                old_text,
                new_text,
                expected_hash=expected_hash,
            ),
            name="edit_file",
            rejected_codes=rejected_codes,
        )

    async def list_directory(self, path: str = ".") -> str:
        """List the contents of a directory.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            A newline-separated listing with type indicators and sizes.
        """
        return await self._call(
            self._require_session().list_directory(path),
            name="list_directory",
        )

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
            self._require_session().search_text(
                pattern,
                path=path,
                include_glob=include_glob,
            ),
            name="search_files",
        )

    async def find_files(self, pattern: str, *, path: str = ".") -> str:
        """Find files by glob pattern (name matching, not content search).

        Args:
            pattern: Glob pattern to match, relative to `path` (e.g. '*.py',
                '**/*.json'). Absolute patterns are rejected.
            path: Directory to search in, relative to the root directory.

        Returns:
            Newline-separated list of matching file paths relative to root.
        """
        return await self._call(
            self._require_session().find_files(pattern, path=path),
            name="find_files",
        )

    async def create_directory(self, path: str) -> str:
        """Create a directory and any missing parents.

        Args:
            path: Directory path relative to the root directory.

        Returns:
            Confirmation message.
        """
        return await self._call(
            self._require_session().create_directory(path),
            name="create_directory",
        )

    async def file_info(self, path: str) -> str:
        """Get metadata about a file or directory.

        Args:
            path: File or directory path relative to the root directory.

        Returns:
            Formatted metadata including size, type, and permissions.
        """
        return await self._call(
            self._require_session().file_info(path),
            name="file_info",
        )

    async def run_command(
        self,
        command: str,
        *,
        timeout_seconds: float | None = None,
    ) -> str:
        """Execute a shell command and return its output.

        Args:
            command: The shell command to run.
            timeout_seconds: Maximum seconds to wait (default: 30).

        Returns:
            Labeled stdout/stderr output with exit code on non-zero exit.
        """
        return await self._call(
            self._require_session().run_command(
                command,
                timeout_seconds=timeout_seconds,
            ),
            name="run_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                }
            ),
        )

    async def start_command(self, command: str) -> str:
        """Start a long-running command in the background (e.g. a server or watcher).

        Callers MUST call `stop_command(command_id)` when done to terminate the
        process and clean up temporary output files.

        Args:
            command: The shell command to run in the background.

        Returns:
            A message containing the unique command ID for later check/stop calls.
        """
        return await self._call(
            self._require_session().start_command(command),
            name="start_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.AUTHORIZATION_DENIED,
                    ErrorCode.TOOL_ARGUMENTS_TOO_LARGE,
                    ErrorCode.TOO_MANY_PENDING_OPERATIONS,
                }
            ),
        )

    async def check_command(self, command_id: str) -> str:
        """Check the status and recent output of a background command.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Status and recent output of the background command.
        """
        return await self._call(
            self._require_session().check_command(command_id),
            name="check_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                }
            ),
        )

    async def stop_command(self, command_id: str) -> str:
        """Stop a background command and return its final output.

        Args:
            command_id: The ID returned by start_command.

        Returns:
            Final output and exit status of the stopped command.
        """
        return await self._call(
            self._require_session().stop_command(command_id),
            name="stop_command",
            rejected_codes=frozenset(
                {
                    ErrorCode.REQUEST_FIELD_INVALID,
                    ErrorCode.STORAGE_NOT_FOUND,
                    ErrorCode.AUTHORIZATION_DENIED,
                }
            ),
        )


class _WorkspaceSandboxToolset(FunctionToolset[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        session: SandboxSession | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        vision: bool,
        model_binding: ModelBinding | None = None,
    ) -> None:
        super().__init__(id=_WORKSPACE_SANDBOX_CAPABILITY_ID)
        surface = _WorkspaceToolSurface(
            session,
            policy,
            limits,
            vision=vision,
            model_binding=model_binding,
        )
        for name in selected_tool_names:
            self.add_tool(
                _workspace_tool(
                    surface,
                    name,
                    _runtime_tool_metadata(_declaration_for(name)),
                )
            )


class _WorkspaceCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        selected_tool_names: tuple[str, ...],
        session: SandboxSession | None,
        policy: WorkspacePolicy,
        limits: PromptLimits,
        *,
        vision: bool,
        model_binding: ModelBinding | None = None,
    ) -> None:
        self.id = _WORKSPACE_SANDBOX_CAPABILITY_ID
        self._selected_tool_names = selected_tool_names
        self._session = session
        self._policy = policy
        self._limits = limits
        self._vision = vision
        self._model_binding = model_binding

    def get_instructions(self) -> InstructionPart:
        return _WORKSPACE_INSTRUCTIONS

    def get_toolset(self) -> _WorkspaceSandboxToolset:
        return _WorkspaceSandboxToolset(
            self._selected_tool_names,
            self._session,
            self._policy,
            self._limits,
            vision=self._vision,
            model_binding=self._model_binding,
        )


def _workspace_tool_definitions(workspace: Workspace) -> tuple[Tool[Any], ...]:
    """Return stable Workspace tool definitions before execution materialization."""
    surface = _WorkspaceToolSurface(None, workspace.policy, PromptLimits())
    return tuple(
        _workspace_tool(
            surface,
            declaration.name,
            _runtime_tool_metadata(declaration),
        )
        for declaration in workspace_tool_declarations()
    )


def workspace_tool_declarations() -> tuple[WorkspaceToolDeclaration, ...]:
    """Return the canonical Workspace tool declarations in name order."""
    return tuple(
        sorted(_WORKSPACE_TOOL_DECLARATIONS, key=lambda item: item.name)
    )


def _declaration_for(name: str) -> WorkspaceToolDeclaration:
    try:
        return _WORKSPACE_TOOL_DECLARATIONS_BY_NAME[name]
    except KeyError as error:
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID) from error


def workspace_capabilities(
    workspace: Workspace,
    selected_tool_names: Sequence[str],
    *,
    limits: "PromptLimits | None" = None,
    session: SandboxSession | None = None,
    vision: bool = True,
    model_binding: ModelBinding | None = None,
) -> tuple[AbstractCapability[AgentContext[object]], ...]:
    """Adapt selected tools to a caller-owned, already-opened session."""
    selected_limits = PromptLimits() if limits is None else limits
    if not isinstance(selected_limits, PromptLimits):
        raise TypeError("limits must be PromptLimits")
    selected = frozenset(selected_tool_names)
    unknown = selected.difference(_WORKSPACE_TOOL_DECLARATIONS_BY_NAME)
    if unknown:
        raise ValueError(f"unknown workspace tools: {tuple(sorted(unknown))}")
    if not selected:
        return ()
    if session is None:
        raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
    ordered = tuple(
        item.name for item in workspace_tool_declarations() if item.name in selected
    )
    _logger.debug(
        "workspace capability materialized: tools=%s session_open=%s",
        ordered,
        session is not None,
    )
    return (
        _WorkspaceCapability(
            ordered,
            session,
            workspace.policy,
            selected_limits,
            vision=vision,
            model_binding=model_binding,
        ),
    )


def _workspace_tool(
    surface: _WorkspaceToolSurface,
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
    'model_input_unknown': 'This model route has no verified support declaration for this file type and transport. Choose a configured compatible route or an authorized conversion.',
    'model_input_not_supported': 'This model route explicitly does not support this file type. Choose a compatible route or an authorized conversion.',
    'input_encoding_not_supported': 'The current provider endpoint cannot encode this input. Use a supported transport or an authorized conversion.',
    'model_input_media_type_unknown': 'The input has no known media type. Provide a recognized file type or explicitly convert it with an authorized tool.',
    'model_input_size_unknown': 'This route requires a known input size. Use a bounded local file instead of an unmeasured reference.',
    'model_input_size_exceeded': 'The file exceeds this model route’s byte limit. Use a smaller supported file or an authorized split/conversion.',
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


def _workspace_tool_rejected(
    name: str,
    error: AIError,
    default_message: str,
) -> ToolCallRetry:
    reason = error.safe_details.get("reason")
    if name in {"attach_files", "read_file"} and reason == "image_input_not_supported":
        message = _UNSUPPORTED_IMAGE_INPUT
    elif name in {"attach_files", "read_file"} and reason == "media_type_unknown":
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
    return ToolCallRetry(message)


__all__ = [
    "WorkspaceToolDeclaration",
    "workspace_tool_declarations",
    "workspace_capabilities",
]
