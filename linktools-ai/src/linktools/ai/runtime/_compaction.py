#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness-backed context compaction with Runtime projection tracking."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.tools import RunContext
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)

from ..errors import AIError, ErrorCode
from ..workspace import normalize_workspace_path
from ._journal import ModelRequestFact, ModelRequestJournal

_KEEP_COMPLETED_PAIRS = 3
_SUMMARY_TAIL_MESSAGES = 20
_CONTROL_TOOL_NAMES = frozenset(
    {
        "delegate_task",
        "list_skills",
        "list_subagents",
        "load_skill",
        "load_subagent",
        "memory",
        "read_memory",
        "search_memory",
        "write_memory",
        "delete_memory",
        "search_tools",
        "subagent",
        "tool_search",
        "write_plan",
    }
)


class ExternalModelRequestObserver(Protocol):
    async def __call__(
        self,
        ctx: RunContext[Any],
        fact: ModelRequestFact,
        phase: str,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None: ...


class _ContextProjectionSink(Protocol):
    def __call__(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None: ...


class _ObservedCompactionModel(WrapperModel):
    """Observe one Harness summary request without changing model semantics."""

    def __init__(
        self,
        wrapped: Model,
        *,
        ctx: RunContext[Any],
        journal: ModelRequestJournal,
        observer: ExternalModelRequestObserver,
    ) -> None:
        super().__init__(wrapped)
        self._ctx = ctx
        self._journal = journal
        self._observer = observer

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        step_index = self._ctx.run_step
        fact = self._journal.begin(step_index, purpose="compaction")
        await self._observer(self._ctx, fact, "started", None, None)
        try:
            response = await self.wrapped.request(
                messages,
                model_settings,
                model_request_parameters,
            )
        except asyncio.CancelledError as error:
            fact = self._journal.finish(step_index, status="CANCELLED")
            self._journal.consume(step_index)
            await self._observer(
                self._ctx,
                fact,
                "cancelled",
                None,
                error,
            )
            raise
        except BaseException as error:
            fact = self._journal.finish(step_index, status="FAILED")
            self._journal.consume(step_index)
            await self._observer(
                self._ctx,
                fact,
                "failed",
                None,
                error,
            )
            raise
        fact = self._journal.finish(step_index, status="SUCCEEDED")
        self._journal.consume(step_index)
        await self._observer(
            self._ctx,
            fact,
            "completed",
            response,
            None,
        )
        return response


class RuntimeCompaction:
    """Adapt Harness compaction to Runtime context projection ownership."""

    def __init__(
        self,
        target_tokens: int | None,
        *,
        journal: ModelRequestJournal | None = None,
        observer: ExternalModelRequestObserver | None = None,
        projection_sink: _ContextProjectionSink | None = None,
        trusted_workspace_read: bool = False,
    ) -> None:
        if target_tokens is not None and (
            not isinstance(target_tokens, int)
            or isinstance(target_tokens, bool)
            or target_tokens <= 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(trusted_workspace_read, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if (journal is None) != (observer is None):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        self._target_tokens = target_tokens
        self._journal = journal
        self._observer = observer
        self._projection_sink = projection_sink
        self._deduplicate = DeduplicateFileReads(
            file_key=(
                _workspace_file_key
                if trusted_workspace_read
                else lambda _call: None
            )
        )

    async def before_model_request(
        self,
        ctx: RunContext[Any],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        source = tuple(request_context.messages)
        request_context = await self._deduplicate.before_model_request(
            ctx,
            request_context,
        )
        if self._target_tokens is not None:
            summary_model: Model | None = None
            if self._journal is not None and self._observer is not None:
                summary_model = _ObservedCompactionModel(
                    request_context.model,
                    ctx=ctx,
                    journal=self._journal,
                    observer=self._observer,
                )
            tiered = TieredCompaction(
                tiers=(
                    ClearToolResults(
                        max_tokens=1,
                        keep_pairs=_KEEP_COMPLETED_PAIRS,
                        exclude_tools=_CONTROL_TOOL_NAMES,
                    ),
                    SummarizingCompaction(
                        model=summary_model,
                        max_messages=1,
                        keep_messages=_SUMMARY_TAIL_MESSAGES,
                    ),
                ),
                target_tokens=self._target_tokens,
            )
            request_context = await tiered.before_model_request(
                ctx,
                request_context,
            )
        projected = tuple(request_context.messages)
        if self._projection_sink is not None:
            self._projection_sink(
                source,
                None if projected == source else projected,
            )
        return request_context


def _workspace_file_key(call: ToolCallPart) -> str | None:
    if call.tool_name != "read_file":
        return None
    try:
        arguments = call.args_as_dict()
    except (TypeError, ValueError):
        return None
    if set(arguments).difference({"path", "offset", "limit"}):
        return None
    path = arguments.get("path")
    if not isinstance(path, str):
        return None
    try:
        normalized = normalize_workspace_path(path)
    except (TypeError, ValueError, AIError):
        return None
    offset = arguments.get("offset", 0)
    limit = arguments.get("limit")
    if (
        not isinstance(offset, int)
        or isinstance(offset, bool)
        or offset < 0
        or limit is not None
        and (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
        )
    ):
        return None
    return json.dumps(
        {"path": normalized, "offset": offset, "limit": limit},
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = ["ExternalModelRequestObserver", "RuntimeCompaction"]
