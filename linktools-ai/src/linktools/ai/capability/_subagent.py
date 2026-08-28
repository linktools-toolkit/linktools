#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Vendor-neutral Subagent discovery and delegation capability."""

from collections.abc import Mapping, Sequence
from typing import Protocol

from ..core import JsonValue
from ..errors import AIError, ErrorCode
from ..spec import SubagentRef


class SubagentDelegate(Protocol):
    async def __call__(
        self,
        ref: "SubagentRef",
        task: str,
        *,
        invocation_id: str,
    ) -> "dict[str, JsonValue]": ...


class SubagentCapability:
    def __init__(
        self,
        refs: "Sequence[SubagentRef]",
        delegate: SubagentDelegate,
        descriptions: "Mapping[str, str | None] | None" = None,
    ) -> None:
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
        invocation_id: str,
    ) -> "dict[str, JsonValue]":
        if not isinstance(subagent_id, str) or not subagent_id.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(task, str) or not task.strip():
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(invocation_id, str) or not invocation_id.strip():
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        ref = self._by_id.get(subagent_id)
        if ref is None:
            raise AIError(
                ErrorCode.CAPABILITY_RESOLUTION_INVALID,
                safe_details={"subagent_id": subagent_id},
            )
        result = await self._delegate(ref, task.strip(), invocation_id=invocation_id)
        if not isinstance(result, dict):
            raise AIError(ErrorCode.INTERNAL_ERROR)
        return result


__all__ = ["SubagentCapability", "SubagentDelegate"]
