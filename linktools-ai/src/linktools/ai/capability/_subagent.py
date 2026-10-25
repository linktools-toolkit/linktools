#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Subagent discovery and delegation capability."""

import json
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import JsonValue as PydanticJsonValue
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..core import JsonValue, validate_user_prompt
from ..errors import AIError, ErrorCode
from ..spec import SubagentRef
from ._context import AgentContext
from ._tool_signal import ToolCallFailed, ToolCallRetry
from ._tool_metadata import tool_metadata

SUBAGENT_CAPABILITY_ID = "linktools.ai.subagents"


class _DelegatedTaskPromptError(AIError):
    pass


class SubagentDelegate(Protocol):
    async def __call__(
        self,
        ref: "SubagentRef",
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> "dict[str, JsonValue]": ...


class SubagentCapability(AbstractCapability[AgentContext[object]]):
    def __init__(
        self,
        refs: "Sequence[SubagentRef]",
        delegate: SubagentDelegate,
        descriptions: "Mapping[str, str | None] | None" = None,
    ) -> None:
        self.id = SUBAGENT_CAPABILITY_ID
        ordered = tuple(sorted(refs, key=lambda item: item.id))
        ids = tuple(item.id for item in ordered)
        if len(ids) != len(set(ids)):
            raise AIError(ErrorCode.CAPABILITY_CONFLICT)
        metadata = {} if descriptions is None else dict(descriptions)
        if any(
            key not in ids
            or value is not None
            and (not isinstance(value, str) or not 1 <= len(value) <= 1024)
            for key, value in metadata.items()
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._refs = ordered
        self._by_id = {item.id: item for item in ordered}
        self._descriptions = metadata
        self._delegate = delegate

    def get_toolset(self) -> FunctionToolset[AgentContext[object]]:
        toolset = FunctionToolset[AgentContext[object]](id=self.id)

        @toolset.tool(
            metadata=tool_metadata(
                plan_safe=True,
                compaction_keep_result=True,
            )
        )
        async def list_subagents(
            _ctx: PydanticRunContext[AgentContext[object]],
        ) -> list[dict[str, str]]:
            """List subagents available for this agent run."""
            return await self.list_subagents()

        @toolset.tool(
            metadata=tool_metadata(compaction_keep_result=True)
        )
        async def delegate_task(
            ctx: PydanticRunContext[AgentContext[object]],
            subagent_id: str,
            task: str,
            files: tuple[str, ...] = (),
        ) -> dict[str, PydanticJsonValue]:
            """Delegate one task and an explicit file subset to a selected subagent."""
            if not ctx.tool_call_id:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            try:
                return await self.delegate_task(
                    subagent_id,
                    task,
                    files=files,
                    invocation_id=ctx.tool_call_id,
                )
            except _DelegatedTaskPromptError as error:
                raise ToolCallRetry(
                    "The delegated task exceeds the allowed prompt size. Shorten the "
                    "task and retry."
                ) from error
            except AIError as error:
                if error.code is ErrorCode.TOOL_EXECUTION_FAILED:
                    raise ToolCallFailed(self._failure_message(error, subagent_id=subagent_id)) from error
                if error.code is ErrorCode.CAPABILITY_RESOLUTION_INVALID:
                    raise ToolCallRetry(
                        "The requested subagent id is not available. Call "
                        "list_subagents and retry with one of the returned ids."
                    ) from error
                if error.code is ErrorCode.REQUEST_FIELD_INVALID:
                    field = error.safe_details.get("field")
                    if field == "subagent_id":
                        message = (
                            "The subagent id is invalid. Call list_subagents and retry "
                            "with one of the returned ids."
                        )
                    elif field == "task":
                        message = (
                            "The delegated task is invalid. Provide a non-empty task "
                            "and retry."
                        )
                    elif field == "files":
                        message = (
                            "The delegated files are invalid. Provide non-empty "
                            "workspace file paths or omit files and retry."
                        )
                    else:
                        raise ToolCallFailed(self._failure_message(error, subagent_id=subagent_id)) from error
                    raise ToolCallRetry(message) from error
                raise

        return toolset

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def get_instructions(self) -> str | None:
        if not self._refs:
            return None
        lines = [
            "The following subagents are available for delegated tasks.",
            "Use `delegate_task` when a listed subagent is better suited to the task.",
        ]
        lines.extend(
            f"- {ref.id}: {self._description(ref.id)}"
            for ref in self._refs
        )
        return "\n".join(lines)

    async def list_subagents(self) -> "list[dict[str, str]]":
        return [
            {
                "id": ref.id,
                "description": self._description(ref.id),
            }
            for ref in self._refs
        ]

    def _description(self, subagent_id: str) -> str:
        if subagent_id not in self._descriptions:
            return f"Available subagent {subagent_id}"
        description = self._descriptions[subagent_id]
        if description is None:
            return f"Subagent {subagent_id} is currently unavailable"
        return description

    async def delegate_task(
        self,
        subagent_id: str,
        task: str,
        *,
        files: Sequence[str] = (),
        invocation_id: str,
    ) -> "dict[str, JsonValue]":
        if not isinstance(subagent_id, str) or not subagent_id.strip():
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "subagent_id"},
            )
        if not isinstance(task, str) or not task.strip():
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "task"},
            )
        try:
            validate_user_prompt(task)
        except AIError as error:
            if error.code is ErrorCode.PROMPT_TOO_LARGE:
                raise _DelegatedTaskPromptError(ErrorCode.PROMPT_TOO_LARGE) from error
            raise
        if not isinstance(invocation_id, str) or not invocation_id.strip():
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not isinstance(files, Sequence) or isinstance(
            files,
            (str, bytes, bytearray),
        ):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "files"},
            )
        paths = tuple(files)
        if any(not isinstance(path, str) or not path for path in paths):
            raise AIError(
                ErrorCode.REQUEST_FIELD_INVALID,
                safe_details={"field": "files"},
            )
        ref = self._by_id.get(subagent_id)
        if ref is None:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                safe_details={"subagent_id": subagent_id},
            )
        result = await self._delegate(
            ref,
            task.strip(),
            files=paths,
            invocation_id=invocation_id,
        )
        if not isinstance(result, dict):
            raise AIError(ErrorCode.INTERNAL_ERROR)
        return result


    @classmethod
    def _failure_message(cls, error: AIError, *, subagent_id: str) -> str:
        def diagnostic_value(value: object) -> str | None:
            if not isinstance(value, str) or not value:
                return None
            if not value.isprintable() or "://" in value or len(json.dumps(value, ensure_ascii=False)) > 130:
                return "[omitted]"
            return value

        outer = error.safe_details
        identity = diagnostic_value(outer.get("subagent_id", subagent_id))
        subject = f"Subagent {identity!r}" if identity is not None else "The delegated subagent"
        cancelled = outer.get("status") == "CANCELLED"
        raw_code = outer.get("error_code", error.code.value)
        try:
            code = ErrorCode(raw_code).value
        except (TypeError, ValueError):
            code = ErrorCode.TOOL_EXECUTION_FAILED.value
        details = outer.get("safe_error_details", outer)
        details = details if isinstance(details, Mapping) else {}
        projected = {}
        execution_id = diagnostic_value(outer.get("execution_id"))
        if execution_id is not None:
            projected["execution_id"] = execution_id
        # These fields are owner-authored diagnostics, not arbitrary exception
        # text. Even this projection must not forward URLs or unbounded values.
        for key in ("reason", "mime_type", "input_kind", "field", "phase",
                    "provider", "api", "model_identity", "model_name"):
            value = diagnostic_value(details.get(key))
            if value is not None:
                projected[key] = value
        description = "was cancelled" if cancelled else f"failed with {code}"
        message = f"{subject} {description} and produced no result."
        if projected:
            message += " Details: " + json.dumps(projected, ensure_ascii=False, separators=(",", ":")) + "."
        return (message + " Do not infer findings or claim completion from this missing result. "
                "If the task depends on this work, resolve or report the failure.")


__all__ = ["SUBAGENT_CAPABILITY_ID", "SubagentCapability", "SubagentDelegate"]
