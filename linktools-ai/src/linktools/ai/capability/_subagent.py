#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Subagent discovery and delegation capability."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import JsonValue as PydanticJsonValue
from pydantic_ai.capabilities import AbstractCapability
from pydantic_ai.exceptions import ModelRetry, ToolFailed
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.toolsets import FunctionToolset

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import SubagentRef
from ._context import AgentContext
from ._tool_semantic import tool_semantic_metadata

SUBAGENT_CAPABILITY_ID = "linktools.ai.subagents"


class SubagentDelegate(Protocol):
    async def __call__(
        self,
        ref: "SubagentRef",
        task: str,
        *,
        files: tuple[str, ...],
        invocation_id: str,
    ) -> "dict[str, JsonValue]": ...


class LinkToolsSubagents(AbstractCapability[AgentContext[object]]):
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

    def get_instructions(self) -> str | None:
        return self.instructions()

    def get_toolset(self) -> FunctionToolset[AgentContext[object]]:
        toolset = FunctionToolset[AgentContext[object]](id=self.id)

        @toolset.tool(
            metadata=tool_semantic_metadata(
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
            metadata=tool_semantic_metadata(compaction_keep_result=True)
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
            except AIError as error:
                if error.code is ErrorCode.TOOL_EXECUTION_FAILED:
                    raise ToolFailed("subagent execution failed; adapt and continue") from error
                if error.code in {
                    ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                    ErrorCode.REQUEST_FIELD_INVALID,
                }:
                    raise ModelRetry("requested subagent, task, or files are invalid") from error
                raise

        return toolset

    @classmethod
    def get_serialization_name(cls) -> str | None:
        return None

    def instructions(self) -> "str | None":
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
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(task, str) or not task.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(invocation_id, str) or not invocation_id.strip():
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if not isinstance(files, Sequence) or isinstance(
            files,
            (str, bytes, bytearray),
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        paths = tuple(files)
        if any(not isinstance(path, str) or not path for path in paths):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
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


__all__ = ["LinkToolsSubagents", "SUBAGENT_CAPABILITY_ID", "SubagentDelegate"]
