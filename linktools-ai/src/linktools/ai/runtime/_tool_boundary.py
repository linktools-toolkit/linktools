#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-owned leaf boundary for LinkTools toolsets."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from contextlib import AsyncExitStack
from dataclasses import dataclass
from typing import Any, Literal, Protocol

from pydantic import ValidationError
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext as PydanticRunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset, ToolsetTool

from ..capability import AgentContext
from ..core import canonical_sha256, normalize_json_value
from ..errors import AIError, ErrorCode
from ..workspace import SandboxSession, WorkspaceToolPermissionPolicy
from ._tool import ToolOperationBridge
from ._tool_metrics import _ToolMetricContext

_WORKSPACE_PATH_FIELDS_KEY = "linktools.ai.workspace_path_fields"


class RepositoryInstructionBoundary(Protocol):
    def render(self) -> str: ...

    async def check(
        self,
        *,
        tool_name: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        path_fields: tuple[str, ...],
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class ManagedToolDescriptor:
    effect_owner: Literal["none", "tool_operation"]
    effect: Literal["none", "replay_safe", "non_replay_safe"]
    tool_class: Literal[
        "business",
        "filesystem.read",
        "filesystem.write",
        "shell",
        "mcp",
    ]
    workspace_path_fields: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.effect_owner not in {"none", "tool_operation"}:
            raise ValueError("effect owner is invalid")
        if self.effect not in {"none", "replay_safe", "non_replay_safe"}:
            raise ValueError("effect is invalid")
        if self.tool_class not in {
            "business",
            "filesystem.read",
            "filesystem.write",
            "shell",
            "mcp",
        }:
            raise ValueError("tool class is invalid")
        if any(
            not isinstance(field, str) or not field
            for field in self.workspace_path_fields
        ):
            raise ValueError("workspace path fields must be non-empty strings")
        if len(self.workspace_path_fields) != len(set(self.workspace_path_fields)):
            raise ValueError("workspace path fields must be unique")
        if self.effect_owner == "none" and self.effect != "none":
            raise ValueError("effect-free tools must use effect=none")
        if self.effect_owner == "tool_operation" and self.effect == "none":
            raise ValueError("tool-operation tools require an effect")


class RuntimeToolBoundaryToolset(AbstractToolset[AgentContext[object]]):
    """Apply workspace policy and effect durability at the final leaf call."""

    def __init__(
        self,
        toolsets: Sequence[AbstractToolset[AgentContext[object]]],
        descriptors: Mapping[str, ManagedToolDescriptor],
        *,
        id: str,
        default_descriptor: ManagedToolDescriptor | None = None,
        workspace_policy: WorkspaceToolPermissionPolicy | None = None,
        sandbox_session: SandboxSession | None = None,
        tool_operations: ToolOperationBridge | None = None,
        tool_metrics: _ToolMetricContext | None = None,
        repository_boundary: RepositoryInstructionBoundary | None = None,
        background_tasks: set[asyncio.Task[object]] | None = None,
    ) -> None:
        if not isinstance(id, str) or not id:
            raise ValueError("toolset id must be non-empty")
        self._toolsets = tuple(toolsets)
        self._descriptors = dict(descriptors)
        self._id = id
        self._default_descriptor = default_descriptor
        self._workspace_policy = workspace_policy
        self._sandbox_session = sandbox_session
        self._tool_operations = tool_operations
        self._tool_metrics = tool_metrics
        self._repository_boundary = repository_boundary
        self._background_tasks = (
            background_tasks if background_tasks is not None else set()
        )
        self._raw_tools: dict[
            str,
            tuple[
                AbstractToolset[AgentContext[object]],
                ToolsetTool[AgentContext[object]],
            ],
        ] = {}
        self._exit_stack: AsyncExitStack | None = None

    @property
    def id(self) -> str:
        return self._id

    async def __aenter__(self) -> "RuntimeToolBoundaryToolset":
        async with AsyncExitStack() as stack:
            for toolset in self._toolsets:
                await stack.enter_async_context(toolset)
            self._exit_stack = stack.pop_all()
        return self

    async def __aexit__(self, *args: Any) -> bool | None:
        if self._exit_stack is None:
            return None
        exit_stack = self._exit_stack
        self._exit_stack = None
        return await exit_stack.aclose()

    async def get_instructions(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
    ) -> Sequence[object] | None:
        values: list[object] = []
        for toolset in self._toolsets:
            value = await toolset.get_instructions(ctx)
            if value is not None:
                values.append(value)
        return values or None

    async def get_tools(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
    ) -> dict[str, ToolsetTool[AgentContext[object]]]:
        result: dict[str, ToolsetTool[AgentContext[object]]] = {}
        for toolset in self._toolsets:
            raw_tools = await toolset.get_tools(ctx)
            for name, raw_tool in raw_tools.items():
                if name in result:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                if name not in self._descriptors and self._default_descriptor is None:
                    raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
                result[name] = ToolsetTool(
                    toolset=self,
                    tool_def=raw_tool.tool_def,
                    max_retries=raw_tool.max_retries,
                    args_validator=raw_tool.args_validator,
                    args_validator_func=raw_tool.args_validator_func,
                )
                self._raw_tools[name] = (toolset, raw_tool)
        return result

    async def call_tool(
        self,
        name: str,
        tool_args: dict[str, Any],
        ctx: PydanticRunContext[AgentContext[object]],
        tool: ToolsetTool[AgentContext[object]],
    ) -> Any:
        if not isinstance(tool_args, dict):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        descriptor = self._descriptors.get(name, self._default_descriptor)
        if descriptor is None or tool.toolset is not self:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        raw_toolset, raw_tool = await self._raw_tool(name, ctx)
        path_fields = _workspace_path_fields(tool.tool_def, descriptor)
        final_args = await self._canonicalize_args(tool_args, path_fields)
        call_id = ctx.tool_call_id
        if not isinstance(call_id, str) or not call_id:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        call = ToolCallPart(name, args=final_args, tool_call_id=call_id)
        if self._repository_boundary is not None:
            await self._repository_boundary.check(
                tool_name=name,
                tool_call_id=call_id,
                arguments=final_args,
                path_fields=path_fields,
            )
        await self._authorize(
            name,
            descriptor,
            final_args,
            approved=ctx.tool_call_approved,
        )
        if descriptor.effect_owner == "none":
            return await self._invoke(
                call,
                tool.tool_def,
                final_args,
                lambda args: raw_toolset.call_tool(name, args, ctx, raw_tool),
            )
        bridge = self._tool_operations
        if bridge is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        replay_safe = descriptor.effect == "replay_safe"
        decision = await bridge.begin(
            ctx,
            call,
            tool.tool_def,
            final_args,
            replay_safe,
        )
        if decision.cached_error is not None:
            raise decision.cached_error
        if decision.has_cached_result:
            return decision.cached_result

        async def invoke(args: dict[str, Any]) -> Any:
            return await raw_toolset.call_tool(name, args, ctx, raw_tool)

        async def unknown_after_leaf(error: BaseException) -> None:
            await bridge.unknown(decision, error)
            if replay_safe:
                raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN) from error
            raise ToolFailed(
                "TOOL_EFFECT_UNKNOWN: verify side effects before retry"
            ) from error

        try:
            result = await self._invoke(
                call,
                tool.tool_def,
                final_args,
                invoke,
            )
        except (
            ApprovalRequired,
            CallDeferred,
        ) as error:
            if not replay_safe:
                await unknown_after_leaf(error)
            cancelled = await bridge.defer(decision)
            if cancelled:
                raise asyncio.CancelledError
            raise
        except (
            ValidationError,
            ModelRetry,
            ToolRetryError,
            ToolFailed,
            ToolFailedError,
        ) as error:
            if (
                not replay_safe
                and not _is_workspace_pre_effect_retry(error, descriptor)
            ):
                await unknown_after_leaf(error)
            cancelled = await bridge.fail(decision, error)
            if cancelled:
                raise asyncio.CancelledError
            raise
        except SkipToolExecution as error:
            if not replay_safe:
                await unknown_after_leaf(error)
            cancelled = await bridge.complete(decision, error.result)
            if cancelled:
                raise asyncio.CancelledError
            raise
        except asyncio.CancelledError as error:
            await bridge.unknown(decision, error)
            raise
        except BaseException as error:
            await unknown_after_leaf(error)
            raise AssertionError("unreachable")
        cancelled = await bridge.complete(decision, result)
        if cancelled:
            raise asyncio.CancelledError
        return result

    async def _raw_tool(
        self,
        name: str,
        ctx: PydanticRunContext[AgentContext[object]],
    ) -> tuple[AbstractToolset[AgentContext[object]], ToolsetTool[AgentContext[object]]]:
        raw_tool = self._raw_tools.get(name)
        if raw_tool is not None:
            return raw_tool
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)

    async def _invoke(
        self,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Any,
    ) -> Any:
        if self._tool_metrics is None:
            return await handler(args)
        return await self._tool_metrics.execute(
            call=call,
            tool_def=tool_def,
            args=args,
            handler=handler,
            suppress_cancel=lambda: False,
        )

    async def _authorize(
        self,
        name: str,
        descriptor: ManagedToolDescriptor,
        args: dict[str, Any],
        *,
        approved: bool,
    ) -> None:
        policy = self._workspace_policy
        if policy is None or (
            not descriptor.tool_class.startswith("filesystem")
            and descriptor.tool_class != "shell"
        ):
            return
        try:
            decision = policy.decide(tool_name=name, tool_class=descriptor.tool_class)
        except (TypeError, ValueError) as error:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT) from error
        if decision == "deny":
            raise ToolFailed("workspace permission denied")
        if decision == "ask" and not approved:
            raise ApprovalRequired(
                metadata={
                    "linktools": {
                        "kind": "workspace_approval",
                        "version": 1,
                        "canonical_args": normalize_json_value(args),
                        "arguments_digest": canonical_sha256(
                            normalize_json_value(args)
                        ),
                    }
                }
            )

    async def _canonicalize_args(
        self,
        args: dict[str, Any],
        path_fields: tuple[str, ...],
    ) -> dict[str, Any]:
        if not path_fields:
            return dict(args)
        session = self._sandbox_session
        if session is None:
            raise AIError(ErrorCode.SANDBOX_SESSION_CLOSED)
        result = dict(args)
        for field in path_fields:
            if field not in result:
                continue
            value = result[field]
            if isinstance(value, str):
                result[field] = await session.canonicalize_path(value)
            elif isinstance(value, tuple):
                canonical_values = []
                for item in value:
                    canonical_values.append(await session.canonicalize_path(item))
                result[field] = tuple(canonical_values)
            elif isinstance(value, list):
                canonical_values = []
                for item in value:
                    canonical_values.append(await session.canonicalize_path(item))
                result[field] = canonical_values
            else:
                raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        return result


def _workspace_path_fields(
    tool_def: ToolDefinition,
    descriptor: ManagedToolDescriptor,
) -> tuple[str, ...]:
    metadata = tool_def.metadata or {}
    value = metadata.get(_WORKSPACE_PATH_FIELDS_KEY)
    if value is None:
        return descriptor.workspace_path_fields
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(field, str) or not field for field in value
    ):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    fields = tuple(value)
    if len(fields) != len(set(fields)):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    return fields


def _is_workspace_pre_effect_retry(
    error: BaseException,
    descriptor: ManagedToolDescriptor,
) -> bool:
    return (
        isinstance(error, ModelRetry)
        and descriptor.tool_class in {
            "filesystem.read",
            "filesystem.write",
            "shell",
        }
        and isinstance(error.__cause__, AIError)
    )


__all__ = [
    "ManagedToolDescriptor",
    "RepositoryInstructionBoundary",
    "RuntimeToolBoundaryToolset",
]
