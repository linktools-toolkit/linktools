#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped Pydantic AI infrastructure capabilities."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from linktools.core import environ
from pydantic_ai.capabilities import AgentCapability as PydanticAgentCapability
from pydantic_ai.capabilities import WrapToolExecuteHandler
from pydantic_ai.exceptions import ModelRetry, ToolRetryError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.models import Model
from pydantic_ai.tools import RunContext, ToolDefinition
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.memory import Memory, SearchableMemoryStore
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import StepPersistence, StepStore

from ..core import canonical_sha256
from ..errors import AIError, ErrorCode

_logger = environ.get_logger("ai.agent.capabilities")


class _RetryAwareStepPersistence(StepPersistence[None]):
    async def wrap_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        try:
            return await super().wrap_tool_execute(ctx, call=call, tool_def=tool_def, args=args, handler=handler)
        except (ModelRetry, ToolRetryError) as error:
            _logger.debug(
                "tool effect marked failed: run=%s tool=%s call=%s",
                self.run_id or ctx.run_id,
                tool_def.name,
                call.tool_call_id,
            )
            await super().on_tool_execute_error(ctx, call=call, tool_def=tool_def, args=args, error=error)
            raise


@dataclass(frozen=True, slots=True)
class AgentRunScope:
    root: Path
    agent_name: str
    conversation_id: "str | None"
    step_run_id: str
    segment_sequence: "int | None"
    memory_namespace: "str | None"
    step_store: StepStore
    memory_store: "SearchableMemoryStore | None"
    allow_tools: bool = True
    context_target_tokens: "int | None" = None
    parent_step_run_id: "str | None" = None


async def compose_platform_capabilities(
    scope: AgentRunScope,
    *,
    model_factory: Callable[["str | Model | None"], "str | Model"],
    parent_model: "str | Model",
) -> "tuple[PydanticAgentCapability[None], ...]":
    del model_factory, parent_model
    _validate_compaction_target(scope.context_target_tokens)
    capabilities: list[PydanticAgentCapability[None]] = [
        _RetryAwareStepPersistence(
            store=scope.step_store,
            agent_name=scope.agent_name,
            run_id=scope.step_run_id,
            parent_run_id=scope.parent_step_run_id,
            metadata={
                "capability_scope": "parent",
                "agent_name": scope.agent_name,
                **({} if scope.segment_sequence is None else {"segment_sequence": str(scope.segment_sequence)}),
            },
        )
    ]
    if scope.allow_tools:
        if scope.memory_namespace is not None:
            if scope.memory_store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            capabilities.append(Memory(store=scope.memory_store, namespace=scope.memory_namespace, agent_name="memory", inject_memory=False))
        capabilities.append(Planning())
    capabilities.append(_build_compaction(scope.context_target_tokens))
    _logger.debug(
        "platform capabilities composed: agent=%s step=%s tools=%s count=%s namespace_digest=%s",
        scope.agent_name,
        scope.step_run_id,
        scope.allow_tools,
        len(capabilities),
        None if scope.memory_namespace is None else canonical_sha256(scope.memory_namespace),
    )
    return tuple(capabilities)


def _build_compaction(context_target_tokens: "int | None") -> PydanticAgentCapability[None]:
    deduplicate = DeduplicateFileReads(file_key=_workspace_file_key)
    if context_target_tokens is None:
        return deduplicate
    return TieredCompaction(
        tiers=[deduplicate, ClearToolResults(max_tokens=1, keep_pairs=3), SummarizingCompaction(max_messages=1, keep_messages=20)],
        target_tokens=context_target_tokens,
    )


def _validate_compaction_target(context_target_tokens: "int | None") -> None:
    if context_target_tokens is not None and (
        not isinstance(context_target_tokens, int)
        or isinstance(context_target_tokens, bool)
        or context_target_tokens <= 0
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _workspace_file_key(part: ToolCallPart) -> "str | None":
    if part.tool_name != "read_file":
        return None
    try:
        arguments = part.args_as_dict()
    except (TypeError, ValueError):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) else None


__all__ = ["AgentRunScope", "compose_platform_capabilities"]
