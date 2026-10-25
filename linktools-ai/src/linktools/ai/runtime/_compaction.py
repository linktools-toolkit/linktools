#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Harness-backed context compaction with Runtime projection tracking."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Protocol

from linktools.core import environ
from pydantic_ai.capabilities import AbstractCapability, WrapModelRequestHandler
from pydantic_ai.exceptions import RunCancelled
from pydantic_ai.messages import (
    InstructionPart,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    SystemPromptPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)
from pydantic_ai.models import Model, ModelRequestContext, ModelRequestParameters
from pydantic_ai.models.wrapper import WrapperModel
from pydantic_ai.settings import ModelSettings, merge_model_settings
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    SummarizingCompaction,
    TieredCompaction,
    estimate_context_tokens,
    estimate_token_count,
    is_pinned,
)

from ._trace_capture import record_http_bodies_enabled
from ._attachment import file_tool_result_metadata
from ..core import PromptLimits
from ..errors import AIError, ErrorCode
from ..model import ModelClientRequest, ModelClientRequestRecorder, ReferencedContent
from ..workspace import validate_workspace_path
from ._budget import RunBudgetContext
from ._journal import ModelRequestFact, ModelRequestJournal, await_request_handoff
from ._harness import SummaryProviderModel, prepare_input_retention
from ._metric_capability import model_request_error
from ._message import (
    SummaryPromptPart, model_input_content, binary_content_usage, project_transient_binary_content, project_model_response_attempts,
)

_KEEP_COMPLETED_PAIRS = 3
_SUMMARY_TAIL_MESSAGES = 20
_UNIFIED_READ_POLICY = "workspace_file_read_v1"
_SUPERSEDED_FILE_READ = "[superseded file read]"
_logger = environ.get_logger("ai.runtime.compaction")


@dataclass(slots=True)
class CompactionPolicy:
    """Ephemeral compaction rules derived for one agent run."""

    keep_result_tools: frozenset[str] = field(default_factory=frozenset)
    context_dedupe_by_tool: Mapping[str, str] = field(default_factory=dict)


class ExternalModelRequestObserver(Protocol):
    async def __call__(
        self,
        ctx: PydanticRunContext[Any],
        fact: ModelRequestFact,
        phase: str,
        model: Model,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None: ...


class ExternalModelRequestRecorder(Protocol):
    async def __call__(
        self,
        ctx: PydanticRunContext[Any],
        fact: ModelRequestFact,
        phase: str,
        model: Model,
        response: ModelResponse | None,
        error: BaseException | None,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        streaming: bool,
        source_messages: Sequence[ModelMessage] | None,
        client_requests: tuple[ModelClientRequest, ...] | None = None,
    ) -> None: ...


class _ContextProjectionSink(Protocol):
    def __call__(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None: ...


class _WorkspaceFileReadDeduplication(AbstractCapability[Any]):
    """Preserve file-input metadata while compacting only proven text repeats."""

    def __init__(self, policies: Callable[[], Mapping[str, str]]) -> None:
        self._policies = policies

    async def before_model_request(
        self, ctx: PydanticRunContext[Any], request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return replace(request_context, messages=await self.compact(request_context.messages, ctx))

    async def compact(
        self, messages: list[ModelMessage], ctx: PydanticRunContext[Any],
    ) -> list[ModelMessage]:
        return _deduplicate_text_entries(messages, self._policies())


def _dependency_safe_tail(messages: Sequence[ModelMessage], keep: int) -> int:
    start = max(0, len(messages) - max(1, keep))
    calls: dict[str, int] = {}
    dependencies: list[int] = []
    orphans: set[int] = set()
    for index, message in enumerate(messages):
        dependency = index
        for part in message.parts:
            if isinstance(part, ToolCallPart):
                calls[part.tool_call_id] = index
            elif isinstance(part, ToolReturnPart) or (
                isinstance(part, RetryPromptPart) and part.tool_name is not None
            ):
                if part.tool_call_id not in calls:
                    orphans.add(index)
                else:
                    dependency = min(dependency, calls[part.tool_call_id])
        dependencies.append(dependency)
    for index in range(len(messages) - 1, -1, -1):
        if index < start:
            break
        start = min(start, dependencies[index])
    if any(index >= start for index in orphans):
        raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE,
                      safe_details={"reason": "orphan_tool_context"})
    return start


class RuntimeSummarizingCompaction(SummarizingCompaction):
    """Adapt public Harness compaction while preserving Runtime context identity."""

    async def compact(
        self, messages: list[ModelMessage], ctx: PydanticRunContext[Any],
    ) -> list[ModelMessage]:
        if len(messages) <= max(1, self.keep_messages):
            return messages
        start = _dependency_safe_tail(messages, self.keep_messages)
        if start == 0:
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE,
                          safe_details={"reason": "unsplittable_tool_context"})
        retained = len(messages) - start
        summary_input = messages
        suffix: list[ModelMessage] = []
        if self.keep_tokens is None and not (
            self.keep_user_messages or self.receipts or self.bridge_prefix
        ):
            # The first retained message is the real boundary. Later reuse of a
            # call ID must not link an older completed call to a new retry.
            summary_input = messages[:start + 1]
            suffix = messages[start + 1:]
            retained = 1
        adapter = replace(self, keep_messages=retained, preserve_first_user_message=False)
        result = [*await SummarizingCompaction.compact(adapter, summary_input, ctx), *suffix]
        if result == messages:
            return messages
        original = {id(part) for message in messages for part in message.parts}
        created = [
            part for message in result for part in message.parts
            if isinstance(part, SystemPromptPart) and id(part) not in original
        ]
        tail = len(messages)
        result_tail = len(result)
        while tail and result_tail and messages[tail - 1] is result[result_tail - 1]:
            tail -= 1
            result_tail -= 1
        if (
            len(created) != 1 or tail > start
            or any(isinstance(part, SummaryPromptPart)
                   for message in result for part in message.parts)
        ):
            raise AIError(ErrorCode.CONTEXT_PAYLOAD_TOO_LARGE,
                          safe_details={"reason": "unsupported_compaction_projection"})
        summary = created[0]
        identified: list[ModelMessage] = [
            replace(message, parts=[
                SummaryPromptPart(part.content, timestamp=part.timestamp, dynamic_ref=part.dynamic_ref)
                if part is summary else part for part in message.parts
            ]) if isinstance(message, ModelRequest) and any(
                part is summary for part in message.parts
            ) else message
            for message in result
        ]
        # Preserve user occurrences, not the first mixed request or a set of part IDs.
        first = next((
            (index, message) for index, message in enumerate(messages)
            if isinstance(message, ModelRequest)
            and any(isinstance(part, UserPromptPart) and not is_pinned(part)
                    for part in message.parts)
        ), None)
        if first is not None and first[0] < tail:
            request = first[1]
            parts = [part for part in request.parts
                     if isinstance(part, UserPromptPart) and not is_pinned(part)]
            summary_index = next(
                index for index, message in enumerate(identified)
                if any(isinstance(part, SummaryPromptPart) for part in message.parts)
            )
            identified.insert(summary_index + 1, replace(request, parts=parts))
        return identified


class _ObservedCompactionModel(WrapperModel):
    """Observe one Harness summary request without changing model semantics."""

    def __init__(
        self,
        wrapped: Model,
        *,
        ctx: PydanticRunContext[Any],
        journal: ModelRequestJournal,
        observer: ExternalModelRequestObserver | None,
        recorder: ExternalModelRequestRecorder | None,
        source_messages: Sequence[ModelMessage],
        budget: RunBudgetContext | None = None,
    ) -> None:
        super().__init__(wrapped)
        self._ctx = ctx
        self._journal = journal
        self._observer = observer
        self._recorder = recorder
        self._source_messages = tuple(source_messages)
        self._budget = budget
        self._last_request_fact: ModelRequestFact | None = None
        self._content_omission_reason = SummaryProviderModel.content_omission_reason(source_messages)

    def project_error(self, error: Exception) -> Exception:
        """Project only after the summary Agent has accounted for native failures."""
        return model_request_error(
            error, content_omission_reason=self._content_omission_reason, fact=self._last_request_fact,
        )

    async def request(
        self,
        messages: list[ModelMessage],
        model_settings: ModelSettings | None,
        model_request_parameters: ModelRequestParameters,
    ) -> ModelResponse:
        fact = self._journal.begin(self._ctx.run_step, purpose="compaction")
        model_request_seq = fact.model_request_seq
        with ModelClientRequestRecorder(
            record_http_bodies=record_http_bodies_enabled(self._ctx.capabilities.values()),
        ) as client_recorder:
            try:
                # Harness formats a summary prompt from the original context;
                # retention still follows that context's referenced inputs.
                prepare_input_retention(list(self._source_messages))
                prepare_input_retention(messages)
                await self._notify(
                    fact,
                    phase="started",
                    response=None,
                    error=None,
                    messages=messages,
                    model_settings=model_settings,
                    parameters=model_request_parameters,
                    client_requests=client_recorder.requests,
                )
                try:
                    async def request() -> ModelResponse:
                        return await self.wrapped.request(messages, model_settings, model_request_parameters)

                    response = (await request() if self._budget is None else
                                await self._budget.run_model(fact.observation_id, request))
                except asyncio.CancelledError as error:
                    await self._finish(
                        model_request_seq,
                        status="CANCELLED",
                        phase="cancelled",
                        response=None,
                        error=error,
                        messages=messages,
                        model_settings=model_settings,
                        parameters=model_request_parameters,
                        client_requests=client_recorder.requests,
                    )
                    raise
                except RunCancelled as error:
                    await self._finish(
                        model_request_seq,
                        status="CANCELLED",
                        phase="cancelled",
                        response=None,
                        error=error,
                        messages=messages,
                        model_settings=model_settings,
                        parameters=model_request_parameters,
                        client_requests=client_recorder.requests,
                    )
                    raise
                except Exception as error:
                    await self._finish(
                        model_request_seq,
                        status="FAILED",
                        phase="failed",
                        response=None,
                        error=error,
                        messages=messages,
                        model_settings=model_settings,
                        parameters=model_request_parameters,
                        client_requests=client_recorder.requests,
                    )
                    raise
                response = project_model_response_attempts(
                    response, content_omission_reason=client_recorder.body_omission_reason,
                )
                finished = self._journal.finish(model_request_seq, status="SUCCEEDED")
                await self._notify(
                    finished,
                    phase="completed",
                    response=response,
                    error=None,
                    messages=messages,
                    model_settings=model_settings,
                    parameters=model_request_parameters,
                    client_requests=client_recorder.requests,
                )
                return response
            finally:
                self._content_omission_reason = client_recorder.body_omission_reason
                self._last_request_fact = self._journal.consume(model_request_seq)

    async def _finish(
        self,
        model_request_seq: int,
        *,
        status: str,
        phase: str,
        response: ModelResponse | None,
        error: BaseException | None,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        client_requests: tuple[ModelClientRequest, ...],
    ) -> None:
        finished = self._journal.finish(model_request_seq, status=status)
        await self._notify(
            finished,
            phase=phase,
            response=response,
            error=error,
            messages=messages,
            model_settings=model_settings,
            parameters=parameters,
            client_requests=client_requests,
        )

    async def _notify(
        self,
        fact: ModelRequestFact,
        *,
        phase: str,
        response: ModelResponse | None,
        error: BaseException | None,
        messages: Sequence[ModelMessage],
        model_settings: ModelSettings | None,
        parameters: ModelRequestParameters,
        client_requests: tuple[ModelClientRequest, ...],
    ) -> None:
        interrupted = False
        if self._observer is not None:
            observation = self._observer(
                self._ctx,
                fact,
                phase,
                self.wrapped,
                response,
                error,
            )
            if phase == "started":
                await observation
            else:
                interrupted = await await_request_handoff(observation)
        if self._recorder is not None:
            await self._recorder(
                self._ctx,
                fact,
                phase,
                self.wrapped,
                response,
                error,
                messages,
                model_settings,
                parameters,
                False,
                self._source_messages,
                client_requests,
            )
        if interrupted:
            raise asyncio.CancelledError


class CompactionCapability(AbstractCapability[None]):
    """Project the provider context without rewriting the raw run transcript."""

    def __init__(
        self,
        target_tokens: int | None,
        *,
        limits: PromptLimits,
        journal: ModelRequestJournal | None = None,
        observer: ExternalModelRequestObserver | None = None,
        request_recorder: ExternalModelRequestRecorder | None = None,
        projection_sink: _ContextProjectionSink | None = None,
        policy: CompactionPolicy | None = None,
        budget: RunBudgetContext | None = None,
    ) -> None:
        self.id = "linktools.ai.compaction"
        if target_tokens is not None and (
            not isinstance(target_tokens, int)
            or isinstance(target_tokens, bool)
            or target_tokens <= 0
        ):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        if not isinstance(limits, PromptLimits):
            raise TypeError("limits must be PromptLimits")
        self._target_tokens = target_tokens
        self._limits = limits
        self._journal = journal
        self._observer = observer
        self._request_recorder = request_recorder
        self._projection_sink = projection_sink
        self._policy = policy
        self._budget = budget
        self._keep_result_tools: frozenset[str] = frozenset()
        self._context_dedupe_by_tool: dict[str, str] = {}
        self._refresh_policy()
        self._deduplicate = _WorkspaceFileReadDeduplication(
            lambda: self._context_dedupe_by_tool,
        )

    async def wrap_model_request(
        self,
        ctx: PydanticRunContext[Any],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        self._refresh_policy()
        source = tuple(request_context.messages)
        binary_projected = project_transient_binary_content(source)
        projected_context = replace(
            request_context,
            messages=list(binary_projected),
        )
        _validate_pending_binary_content(
            projected_context.messages,
            self._limits,
        )
        # Harness may rewrite its context history; only the provider view is mutable.
        projected_run_context = replace(ctx, messages=projected_context.messages)
        target_tokens = self._request_target(projected_context)
        if target_tokens is None:
            projected_context = await self._deduplicate.before_model_request(
                projected_run_context,
                projected_context,
            )
        else:
            summary_model: _ObservedCompactionModel | None = None
            if self._journal is not None and (
                self._observer is not None or self._request_recorder is not None
            ):
                summary_model = _ObservedCompactionModel(
                    projected_context.model,
                    ctx=ctx,
                    journal=self._journal,
                    observer=self._observer,
                    recorder=self._request_recorder,
                    source_messages=source,
                    budget=self._budget,
                )
            tiered = TieredCompaction(
                tiers=(
                    self._deduplicate,
                    ClearToolResults(
                        max_tokens=1,
                        keep_pairs=_KEEP_COMPLETED_PAIRS,
                        exclude_tools=self._keep_result_tools | frozenset(
                            name for name, policy in self._context_dedupe_by_tool.items()
                            if policy == _UNIFIED_READ_POLICY
                        ),
                    ),
                    RuntimeSummarizingCompaction(
                        model=summary_model,
                        max_messages=1,
                        keep_messages=_SUMMARY_TAIL_MESSAGES,
                    ),
                ),
                target_tokens=target_tokens,
            )
            try:
                projected_context = await tiered.before_model_request(
                    projected_run_context,
                    projected_context,
                )
            except Exception as error:
                # The Harness summary Agent shares parent usage and must see
                # native fallback errors before terminal diagnostics replace them.
                safe_error = (
                    summary_model.project_error(error) if summary_model is not None else model_request_error(
                        error, content_omission_reason=SummaryProviderModel.content_omission_reason(source),
                    )
                )
                if safe_error is not error:
                    raise safe_error from None
                raise
        projected = tuple(projected_context.messages)
        if self._projection_sink is not None:
            self._projection_sink(
                source,
                None if projected == source else projected,
            )
        return await handler(projected_context)

    def _request_target(self, request: ModelRequestContext) -> int | None:
        target = self._target_tokens
        window = request.model.context_window
        if window is not None and window > 0:
            settings = merge_model_settings(request.model.settings, request.model_settings)
            output_tokens = (settings or {}).get("max_tokens") or 0
            available = max(1, window - output_tokens)
            target = available if target is None else min(target, available)
        if target is None:
            return None

        parameters = request.model_request_parameters
        messages = list(request.messages)
        if parameters.instruction_parts is not None:
            for index in range(len(messages) - 1, -1, -1):
                if isinstance(message := messages[index], ModelRequest):
                    messages[index] = replace(
                        message,
                        instructions=InstructionPart.join(parameters.instruction_parts),
                    )
                    break
        schema_text: list[str] = []
        for tool in parameters.function_tools:
            if (
                parameters.visibility_of(tool.name) == "withheld"
                and tool.name not in parameters.revealed_tool_names
            ):
                continue
            schema_text.extend((
                tool.name,
                tool.description or "",
                json.dumps(tool.parameters_json_schema, sort_keys=True),
            ))
        output_text = "".join(
            tool.name + (tool.description or "")
            + json.dumps(tool.parameters_json_schema, sort_keys=True)
            for tool in parameters.output_tools
        )
        if output := parameters.output_object:
            object_text = parameters.prompted_output_instructions or "".join((
                output.name or "",
                output.description or "",
                json.dumps(output.json_schema, sort_keys=True),
            ))
            # Auto output carries both alternatives until the provider selects a mode.
            output_text = max(output_text, object_text, key=len)
        schema_text.append(output_text)
        # Provider usage already includes schemas and instructions. Use the current
        # request's text/schema estimate as a floor, not an extra charge on that anchor.
        current_estimate = estimate_token_count([
            *messages,
            ModelRequest(parts=[UserPromptPart("".join(schema_text))]),
        ])
        anchored_estimate = estimate_context_tokens(
            request.messages,
            model_request_parameters=parameters,
        )
        pending_text_growth = 0
        for message in reversed(request.messages):
            if isinstance(message, ModelResponse) and message.usage.input_tokens:
                break
            if not isinstance(message, ModelRequest):
                continue
            for part in message.parts:
                for item in model_input_content(part):
                    if isinstance(item, ReferencedContent) and item.representation == "text":
                        pending_text_growth += max(0, item.ref.size - len(item.content))
        # Unconsumed text bodies are absent from the SDK's descriptor-only
        # heuristic. Approximate their tail growth at four UTF-8 bytes/token,
        # without loading bytes or charging content already covered by usage.
        return max(1, target - max(0, current_estimate - anchored_estimate) - pending_text_growth // 4)

    def _refresh_policy(self) -> None:
        if self._policy is None:
            return
        keep_result_tools = frozenset(self._policy.keep_result_tools)
        context_dedupe_by_tool = dict(self._policy.context_dedupe_by_tool)
        if any(
            value != _UNIFIED_READ_POLICY
            for value in context_dedupe_by_tool.values()
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        if (
            keep_result_tools == self._keep_result_tools
            and context_dedupe_by_tool == self._context_dedupe_by_tool
        ):
            return
        self._keep_result_tools = keep_result_tools
        self._context_dedupe_by_tool = context_dedupe_by_tool
        _logger.debug(
            "runtime compaction policy refreshed: keep_results=%s dedupe_tools=%s",
            tuple(sorted(keep_result_tools)),
            tuple(sorted(context_dedupe_by_tool)),
        )


def _validate_pending_binary_content(
    messages: Sequence[ModelMessage],
    limits: PromptLimits,
) -> None:
    count, total_bytes = binary_content_usage(messages)
    output_bytes = 0
    # Only newly returned previews belong to the pending parallel batch. Older
    # text remains ordinary context, governed by the existing compaction budget.
    for message in reversed(messages):
        if isinstance(message, ModelResponse) and message.state == "complete":
            break
        if not isinstance(message, ModelRequest):
            continue
        for part in message.parts:
            if not isinstance(part, ToolReturnPart) or part.tool_name != "read_file":
                continue
            value = file_tool_result_metadata(part.content)
            if not isinstance(value, Mapping) or not isinstance(value.get("texts"), list):
                continue
            output_bytes += len(json.dumps(value, ensure_ascii=False).encode("utf-8"))
            for text in value["texts"]:
                if isinstance(text, Mapping) and isinstance(text.get("content"), str):
                    count += 1
                    total_bytes += len(text["content"].encode("utf-8"))
    if (
        count > limits.max_binary_input_parts
        or total_bytes > limits.max_binary_input_bytes
        or output_bytes > limits.max_file_output_bytes
    ):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID,
                      safe_details={"reason": "file_request_budget_exceeded"})


def _deduplicate_text_entries(
    messages: list[ModelMessage],
    policies: Mapping[str, str],
) -> list[ModelMessage]:
    seen: set[str] = set()
    projected: list[ModelMessage] = []
    for message in reversed(messages):
        if not isinstance(message, ModelRequest):
            projected.append(message)
            continue
        parts = []
        changed = False
        for part in reversed(message.parts):
            value = file_tool_result_metadata(part.content) if isinstance(part, ToolReturnPart) else None
            if (
                not isinstance(part, ToolReturnPart)
                or policies.get(part.tool_name) != _UNIFIED_READ_POLICY
                or value is None
                or not isinstance(value.get("texts"), list)
            ):
                parts.append(part)
                continue
            texts = []
            keys: set[str] = set()
            text_changed = False
            for entry in value["texts"]:
                key = _text_read_key(entry)
                if key is not None:
                    keys.add(key)
                if key is not None and key in seen:
                    texts.append({**entry, "content": _SUPERSEDED_FILE_READ})
                    text_changed = True
                else:
                    texts.append(entry)
            seen.update(keys)
            if text_changed:
                receipt = {**value, "texts": texts}
                content = [receipt, *part.content[1:]] if isinstance(part.content, list) else receipt
                parts.append(replace(part, content=content))
                changed = True
            else:
                parts.append(part)
        projected.append(
            replace(message, parts=list(reversed(parts))) if changed else message
        )
    projected.reverse()
    return projected


def _text_read_key(value: object) -> str | None:
    if not isinstance(value, Mapping) or not isinstance(value.get("content"), str):
        return None
    if value["content"] == _SUPERSEDED_FILE_READ:
        return None
    digest = value.get("hash")
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    path = value.get("path")
    if not isinstance(path, str):
        return None
    try:
        path = validate_workspace_path(path)
    except (TypeError, ValueError, AIError):
        return None
    coverage = {}
    for name in (
        "offset", "end_offset", "byte_offset", "end_byte_offset",
        "line_offset", "end_line_offset",
    ):
        position = value.get(name)
        if isinstance(position, bool) or not isinstance(position, int) or position < 0:
            return None
        coverage[name] = position
    if (
        coverage["end_offset"] < coverage["offset"]
        or coverage["end_byte_offset"] < coverage["byte_offset"]
    ):
        return None
    lossy = value.get("lossy")
    if not isinstance(lossy, bool):
        return None
    return json.dumps(
        {"path": path, "hash": digest, "coverage": coverage, "lossy": lossy},
        sort_keys=True,
        separators=(",", ":"),
    )


__all__ = [
    "RuntimeSummarizingCompaction",
    "CompactionCapability",
    "CompactionPolicy",
    "ExternalModelRequestObserver",
    "ExternalModelRequestRecorder",
]
