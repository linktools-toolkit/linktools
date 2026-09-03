#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Execution-scoped Pydantic AI infrastructure capabilities."""

from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NoReturn, Protocol

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    CapabilityOrdering,
    NodeResult,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    RetryPromptPart,
    ToolCallPart,
    ToolReturnPart,
)
from pydantic_ai.models import Model, ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext, ToolDefinition
from pydantic_ai.toolsets import AbstractToolset
from pydantic_ai_harness.compaction import (
    ClearToolResults,
    DeduplicateFileReads,
    SummarizingCompaction,
    TieredCompaction,
)
from pydantic_ai_harness.memory import Memory, SearchableMemoryStore
from pydantic_ai_harness.planning import PlanStore, Planning
from pydantic_ai_harness.step_persistence import StepEvent, StepPersistence, StepStore

from ..capability import (
    SKILL_TOOL_NAMES,
    SUBAGENT_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_READ_TOOL_NAMES,
    WORKSPACE_FILESYSTEM_TOOL_NAMES,
    WORKSPACE_SHELL_TOOL_NAMES,
)
from ..core import JsonValue, canonical_json_bytes, normalize_json_value
from ..errors import AIError, ErrorCode
from ..observe import MiddlewarePipeline, RunContext as ObservedRunContext
from ..workspace import (
    RepositoryInstructionDocument,
    RepositoryInstructionResolver,
    RepositoryInstructions,
    WorkspacePolicy,
)

_logger = environ.get_logger("ai.runtime.capabilities")

MEMORY_TOOL_NAMES = ("delete_memory", "read_memory", "search_memory", "write_memory")
MEMORY_READ_TOOL_NAMES = ("read_memory", "search_memory")
PLANNING_TOOL_NAMES = ("write_plan",)
PLAN_SAFE_METADATA_KEY = "linktools.ai.plan_safe"
_REPLAY_SAFE_METADATA_KEY = "linktools.ai.replay_safe"
_MODEL_USAGE_INPUT_METADATA_KEY = "linktools.ai.model_usage.input_tokens"
_MODEL_USAGE_OUTPUT_METADATA_KEY = "linktools.ai.model_usage.output_tokens"
_MODEL_USAGE_CACHE_READ_METADATA_KEY = "linktools.ai.model_usage.cache_read_tokens"
_MODEL_USAGE_CACHE_WRITE_METADATA_KEY = "linktools.ai.model_usage.cache_write_tokens"
_MODEL_TOOL_ERROR_MAX_CHARS = 4096
_MODEL_TOOL_ERROR_HEAD_CHARS = 1024
_MODEL_TOOL_ERROR_TRUNCATION_MARKER = "...[truncated]..."
_MODEL_EFFECT_UNKNOWN_MESSAGE = "TOOL_EFFECT_UNKNOWN: verify side effects before retry"
_MODEL_EXECUTION_FAILED_MESSAGE = "tool execution failed; adapt and continue"
_MODEL_RETRY_PREFIX = "TOOL_RETRY_REQUIRED"
_MODEL_FAILED_PREFIX = "TOOL_EXECUTION_FAILED"
_REPOSITORY_MARKER_HEADER = "[linktools.repository-instructions.v1]"
_REPOSITORY_MARKER_ACTION = (
    "Apply the payload as newly applicable repository instructions for this failed "
    "filesystem target, then reconsider the failed tool call."
)
_WORKSPACE_SCOPED_TOOL_NAMES = frozenset(
    {
        "read_file",
        "write_file",
        "edit_file",
        "file_info",
        "create_directory",
        "list_directory",
        "search_files",
        "find_files",
    }
)
_TRUSTED_TOOL_CLASSES = frozenset(
    {
        "control",
        "filesystem.read",
        "filesystem.write",
        "shell",
        "memory.read",
        "memory.write",
    }
)
_WORKSPACE_FILESYSTEM_CAPABILITY_ID = "workspace-filesystem"
_WORKSPACE_SHELL_CAPABILITY_ID = "workspace-shell"
_SKILL_CAPABILITY_ID = "linktools-skill"
_MEMORY_CAPABILITY_ID = "linktools-memory"
_PLANNING_CAPABILITY_ID = "linktools-planning"
_SUBAGENT_CAPABILITY_ID = "linktools-subagent"


@dataclass(frozen=True, slots=True)
class ToolOperationDecision:
    operation_id: str
    owner: str
    fence: int
    replay_safe: bool
    cached_result: JsonValue = None
    has_cached_result: bool = False
    cached_error: "BaseException | None" = None


@dataclass(frozen=True, slots=True)
class _ToolEffectPolicy:
    replay_safe: bool
    effect_free: bool


@dataclass
class _ToolCallState:
    decision: ToolOperationDecision
    policy: _ToolEffectPolicy
    handler_entered: bool = False
    handler_observed: bool = False
    heartbeat_observed: bool = False
    operation_terminalized: bool = False
    preserve_started: bool = False
    cached_failure: bool = False
    effect_terminalized: bool = False
    heartbeat_task: "asyncio.Task[None] | None" = None


class ToolOperationBridge(Protocol):
    async def begin(
        self,
        ctx: "RunContext[None]",
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision: ...

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision: ...

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool: ...

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool: ...

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None: ...

    async def existing_call_ids(
        self, tool_call_ids: Sequence[str]
    ) -> frozenset[str]: ...


class _MissingToolOperationBridge:
    async def begin(
        self,
        ctx: "RunContext[None]",
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        replay_safe: bool,
    ) -> ToolOperationDecision:
        del ctx, call, tool_def, args, replay_safe
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def renew(self, decision: ToolOperationDecision) -> ToolOperationDecision:
        del decision
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def complete(self, decision: ToolOperationDecision, result: Any) -> bool:
        del decision, result
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def fail(self, decision: ToolOperationDecision, error: BaseException) -> bool:
        del decision, error
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def unknown(self, decision: ToolOperationDecision, error: BaseException) -> None:
        del decision, error
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

    async def existing_call_ids(
        self, tool_call_ids: Sequence[str]
    ) -> frozenset[str]:
        del tool_call_ids
        raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)


@dataclass(kw_only=True, eq=False)
class _RuntimeStepPersistence(StepPersistence[None]):
    tool_operations: ToolOperationBridge = field(repr=False, compare=False)
    plan_mode: bool = False
    trusted_tool_classes: "tuple[tuple[str, str], ...]" = ()
    trusted_mcp_selectors: "tuple[str, ...]" = ()
    background_tasks: "set[asyncio.Task[Any]]" = field(
        default_factory=set,
        repr=False,
        compare=False,
    )
    deferred_pause_sink: "Callable[[int], None] | None" = field(
        default=None,
        repr=False,
        compare=False,
    )
    _calls: "dict[tuple[str, str], _ToolCallState]" = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _last_observed_step_index: "int | None" = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan_mode, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _validate_trusted_tool_classes(self.trusted_tool_classes)
        _validate_trusted_mcp_selectors(self.trusted_mcp_selectors)

    async def after_node_run(
        self,
        ctx: "RunContext[None]",
        *,
        node: AgentNode[None],
        result: NodeResult[None],
    ) -> NodeResult[None]:
        observed = await super().after_node_run(ctx, node=node, result=result)
        self._last_observed_step_index = ctx.run_step
        return observed

    async def after_run(
        self,
        ctx: "RunContext[None]",
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        output = result.output
        if isinstance(output, DeferredToolRequests):
            if not output.approvals or output.calls:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if self._last_observed_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self.deferred_pause_sink is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            self.deferred_pause_sink(self._last_observed_step_index)
        return await super().after_run(ctx, result=result)

    async def after_model_request(
        self,
        ctx: "RunContext[None]",
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del request_context
        run_id = self.run_id or ctx.run_id
        if not run_id:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        metadata = dict(self.metadata)
        metadata.update(_model_usage_metadata(response))
        await self.store.append_event(
            StepEvent(
                run_id=run_id,
                kind="model_request_completed",
                step_index=ctx.run_step,
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
                metadata=metadata,
            )
        )
        return response

    async def before_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del ctx, call
        if self.plan_mode and not tool_allowed_in_planning(
            tool_def,
            trusted_tool_classes=self.trusted_tool_classes,
            trusted_mcp_selectors=self.trusted_mcp_selectors,
        ):
            raise AIError(
                ErrorCode.CAPABILITY_POLICY_CONFLICT,
                safe_details={"tool_name": tool_def.name, "mode": "plan"},
            )
        # Effect admission starts only after every before-hook has settled.
        return args

    async def wrap_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        policy = _tool_effect_policy(
            tool_def,
            trusted_tool_classes=self.trusted_tool_classes,
        )
        try:
            decision = await self.tool_operations.begin(
                ctx,
                call,
                tool_def,
                args,
                policy.replay_safe,
            )
        except AIError as error:
            if error.code is ErrorCode.TOOL_EFFECT_UNKNOWN:
                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error
            raise
        if decision.replay_safe is not policy.replay_safe:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        key = self._decision_key(ctx, call)
        state = _ToolCallState(
            decision=decision,
            policy=policy,
            operation_terminalized=decision.has_cached_result or decision.cached_error is not None,
            cached_failure=decision.cached_error is not None,
        )
        self._calls[key] = state
        try:
            await super().before_tool_execute(ctx, call=call, tool_def=tool_def, args=args)
        except BaseException:
            self._calls.pop(key, None)
            raise
        if state.decision.cached_error is not None:
            error = state.decision.cached_error
            try:
                await self._record_failed_effect(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=error,
                    state=state,
                )
            except BaseException as raised:
                if _bypasses_tool_error_hook(raised):
                    self._calls.pop(key, None)
                raise
            raise AssertionError("cached failure effect hook must raise")
        if state.decision.has_cached_result:
            return state.decision.cached_result

        async def tracked_handler(validated_args: dict[str, Any]) -> Any:
            state.handler_entered = True
            return await handler(validated_args)

        handler_task = asyncio.create_task(
            super().wrap_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                handler=tracked_handler,
            ),
            name=f"tool-handler-{call.tool_call_id}",
        )
        heartbeat_task = asyncio.create_task(
            self._heartbeat(state, handler_task),
            name=f"tool-heartbeat-{call.tool_call_id}",
        )
        state.heartbeat_task = heartbeat_task
        handler_detached = False
        keep_call_state = True
        try:
            done, _ = await asyncio.wait(
                (handler_task, heartbeat_task),
                return_when=asyncio.FIRST_COMPLETED,
            )
            if heartbeat_task in done:
                state.heartbeat_observed = True
                heartbeat_error = heartbeat_task.exception()
                if heartbeat_error is not None:
                    handler_task.cancel()
                    self._detach_task(handler_task, "tool handler after heartbeat loss")
                    handler_detached = True
                    if state.handler_entered and not state.policy.replay_safe:
                        await self._mark_unknown(state, heartbeat_error)
                        keep_call_state = False
                        raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from heartbeat_error
                    state.preserve_started = True
                    raise heartbeat_error
            try:
                result = await handler_task
            finally:
                state.handler_observed = True
            try:
                cancelled = await self.tool_operations.complete(
                    state.decision,
                    result,
                )
            except BaseException:
                state.preserve_started = True
                raise
            state.operation_terminalized = True
            if cancelled:
                state.preserve_started = False
                keep_call_state = False
                self._calls.pop(key, None)
                raise asyncio.CancelledError
            return result
        except SkipToolExecution as signal:
            state.preserve_started = True
            cancelled = await self.tool_operations.complete(
                state.decision,
                signal.result,
            )
            state.operation_terminalized = True
            state.preserve_started = False
            if cancelled:
                keep_call_state = False
                self._calls.pop(key, None)
                raise asyncio.CancelledError
            await self._record_completed_effect(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=signal.result,
                state=state,
            )
            keep_call_state = False
            self._calls.pop(key, None)
            raise
        except (
            ValidationError,
            ModelRetry,
            ToolRetryError,
            ToolFailed,
            ToolFailedError,
        ) as error:
            if (
                isinstance(error, ValidationError)
                and state.handler_entered
                and not state.policy.effect_free
                and not state.policy.replay_safe
            ):
                await self._mark_unknown(state, error)
                keep_call_state = False
                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error
            try:
                await self._fail_known_effect(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=error,
                    state=state,
                )
            except BaseException as raised:
                if state.operation_terminalized and _bypasses_tool_error_hook(raised):
                    keep_call_state = False
                    self._calls.pop(key, None)
                raise
            raise AssertionError("known failure effect hook must raise")
        except (CallDeferred, ApprovalRequired) as signal:
            if state.handler_entered and not state.policy.replay_safe and not state.policy.effect_free:
                await self._mark_unknown(state, signal)
                keep_call_state = False
                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from signal
            unsupported = AIError(
                ErrorCode.CAPABILITY_POLICY_CONFLICT,
                safe_details={
                    "tool_name": tool_def.name,
                    "reason": "dynamic_deferred_unsupported",
                },
            )
            await self._fail_known_effect(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=unsupported,
                state=state,
            )
            raise AssertionError("dynamic deferred failure hook must raise")
        except asyncio.CancelledError as error:
            if state.operation_terminalized:
                state.preserve_started = False
                keep_call_state = False
                self._calls.pop(key, None)
                raise
            state.preserve_started = True
            if state.handler_entered and not state.policy.replay_safe:
                await self._mark_unknown(state, error)
            keep_call_state = False
            self._calls.pop(key, None)
            raise
        except Exception as error:
            if not state.handler_entered:
                raise
            if isinstance(error, AIError):
                if state.policy.effect_free:
                    try:
                        await self._fail_known_effect(
                            ctx,
                            call=call,
                            tool_def=tool_def,
                            args=args,
                            error=error,
                            state=state,
                        )
                    except BaseException as raised:
                        if state.operation_terminalized and _bypasses_tool_error_hook(raised):
                            keep_call_state = False
                            self._calls.pop(key, None)
                        raise
                    raise AssertionError("runtime failure effect hook must raise")
                if state.policy.replay_safe:
                    state.preserve_started = True
                    raise
                await self._mark_unknown(state, error)
                raise
            if state.policy.effect_free:
                try:
                    await self._fail_unclassified_effect(
                        ctx,
                        call=call,
                        tool_def=tool_def,
                        args=args,
                        error=error,
                        state=state,
                    )
                except BaseException as raised:
                    if state.operation_terminalized and _bypasses_tool_error_hook(raised):
                        keep_call_state = False
                        self._calls.pop(key, None)
                    raise
                raise AssertionError("tool failure effect hook must raise")
            if state.policy.replay_safe:
                state.preserve_started = True
                raise AIError(
                    ErrorCode.TOOL_EFFECT_UNKNOWN,
                    safe_details={"phase": "tool_effect_replay"},
                ) from error
            await self._mark_unknown(state, error)
            keep_call_state = False
            raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error
        finally:
            await self._stop_heartbeat(state)
            if not handler_detached:
                if not handler_task.done():
                    handler_task.cancel()
                    self._detach_task(handler_task, "tool handler cleanup")
                elif not state.handler_observed:
                    self._consume_task(handler_task, "tool handler cleanup")
            if not keep_call_state:
                self._calls.pop(key, None)

    async def after_tool_execute(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        key = self._decision_key(ctx, call)
        state = self._calls.get(key)
        if state is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        if state.preserve_started or not state.operation_terminalized or state.cached_failure:
            self._calls.pop(key, None)
            raise AIError(ErrorCode.TOOL_EFFECT_UNKNOWN)
        try:
            return await super().after_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=result,
            )
        finally:
            self._calls.pop(key, None)

    async def on_tool_execute_error(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
    ) -> Any:
        key = self._decision_key(ctx, call)
        state = self._calls.get(key)
        if state is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY) from error
        try:
            if state.operation_terminalized or state.cached_failure or state.preserve_started:
                raise error
            if isinstance(
                error,
                (ValidationError, ModelRetry, ToolRetryError, ToolFailed, ToolFailedError),
            ):
                if (
                    isinstance(error, ValidationError)
                    and state.handler_entered
                    and not state.policy.effect_free
                    and not state.policy.replay_safe
                ):
                    await self._mark_unknown(state, error)
                    raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error
                return await self._fail_known_effect(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=error,
                    state=state,
                )
            if (
                isinstance(error, AIError)
                and state.handler_entered
                and not state.policy.effect_free
            ):
                if state.policy.replay_safe:
                    state.preserve_started = True
                else:
                    await self._mark_unknown(state, error)
                raise error
            if state.handler_entered and not state.policy.effect_free:
                if state.policy.replay_safe:
                    state.preserve_started = True
                    raise AIError(
                        ErrorCode.TOOL_EFFECT_UNKNOWN,
                        safe_details={"phase": "tool_effect_replay"},
                    ) from error
                await self._mark_unknown(state, error)
                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error
            return await self._fail_unclassified_effect(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=error,
                state=state,
            )
        finally:
            await self._stop_heartbeat(state)
            self._calls.pop(key, None)

    async def _heartbeat(
        self,
        state: _ToolCallState,
        handler_task: "asyncio.Task[Any]",
    ) -> None:
        while not handler_task.done():
            await asyncio.sleep(15)
            if handler_task.done():
                return
            state.decision = await self.tool_operations.renew(state.decision)

    async def _stop_heartbeat(self, state: _ToolCallState) -> None:
        task = state.heartbeat_task
        if task is None:
            return
        if not task.done():
            task.cancel()
            self._detach_task(task, "tool heartbeat cleanup")
        elif not state.heartbeat_observed:
            self._consume_task(task, "tool heartbeat cleanup")
        state.heartbeat_task = None

    def _detach_task(self, task: asyncio.Task[Any], label: str) -> None:
        if task.done():
            self._consume_task(task, label)
            return
        if task in self.background_tasks:
            return
        self.background_tasks.add(task)

        def consume(done: asyncio.Task[Any]) -> None:
            try:
                self._consume_task(done, label)
            finally:
                self.background_tasks.discard(done)

        task.add_done_callback(consume)

    @staticmethod
    def _consume_task(task: asyncio.Task[Any], label: str) -> None:
        if not task.done():
            return
        try:
            task.result()
        except asyncio.CancelledError:
            pass
        except BaseException:  # noqa: BLE001
            _logger.exception("detached %s failed", label)

    async def _fail_known_effect(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
        state: _ToolCallState,
    ) -> Any:
        state.preserve_started = True
        durable_error = _durable_failure_error(error, call=call, tool_def=tool_def)
        cancelled = await self.tool_operations.fail(state.decision, durable_error)
        state.operation_terminalized = True
        state.preserve_started = False
        if cancelled:
            raise asyncio.CancelledError
        _logger.debug(
            "tool effect marked failed: run=%s tool=%s call=%s",
            self.run_id or ctx.run_id,
            tool_def.name,
            call.tool_call_id,
        )
        return await self._record_failed_effect(
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            error=error,
            state=state,
        )

    async def _fail_unclassified_effect(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
        state: _ToolCallState,
    ) -> Any:
        state.preserve_started = True
        cancelled = await self.tool_operations.fail(state.decision, error)
        state.operation_terminalized = True
        state.preserve_started = False
        if cancelled:
            raise asyncio.CancelledError
        _logger.debug(
            "tool execution marked failed: run=%s tool=%s call=%s",
            self.run_id or ctx.run_id,
            tool_def.name,
            call.tool_call_id,
        )
        return await self._record_failed_effect(
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            error=error,
            state=state,
        )

    async def _record_failed_effect(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: BaseException,
        state: _ToolCallState,
    ) -> Any:
        if state.effect_terminalized:
            raise error
        try:
            result = await super().on_tool_execute_error(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=error,
            )
        except BaseException as raised:
            state.effect_terminalized = True
            if raised is error:
                model_error = _model_tool_error(
                    error,
                    call=call,
                    tool_def=tool_def,
                )
                if model_error is not error:
                    raise model_error from error
            raise
        state.effect_terminalized = True
        return result

    async def _record_completed_effect(
        self,
        ctx: "RunContext[None]",
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
        state: _ToolCallState,
    ) -> Any:
        if state.effect_terminalized:
            return result
        try:
            value = await super().after_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=result,
            )
        except BaseException:
            state.effect_terminalized = True
            raise
        state.effect_terminalized = True
        return value

    async def _mark_unknown(self, state: _ToolCallState, error: BaseException) -> None:
        state.preserve_started = True
        state.operation_terminalized = True
        await self.tool_operations.unknown(state.decision, error)

    def _decision_key(self, ctx: "RunContext[None]", call: ToolCallPart) -> tuple[str, str]:
        return self._effective_run_id(ctx), call.tool_call_id


class _ObservationalMiddlewareCapability(AbstractCapability[None]):
    def __init__(
        self,
        pipeline: MiddlewarePipeline,
        context: ObservedRunContext,
    ) -> None:
        if not isinstance(pipeline, MiddlewarePipeline):
            raise TypeError("pipeline must be MiddlewarePipeline")
        if not isinstance(context, ObservedRunContext):
            raise TypeError("context must be observe.RunContext")
        self._pipeline = pipeline
        self._context = context

    async def before_run(self, ctx: RunContext[None]) -> None:
        del ctx
        await self._pipeline.before_run(self._context)

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        await self._pipeline.before_model(self._context)
        return request_context

    async def after_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        del ctx, request_context
        await self._pipeline.after_model(self._context)
        return response

    async def before_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del ctx, call, tool_def
        await self._pipeline.before_tool(self._context)
        return args

    async def after_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
    ) -> Any:
        del ctx, call, tool_def, args
        await self._pipeline.after_tool(self._context)
        return result

    async def on_run_error(
        self,
        ctx: RunContext[None],
        *,
        error: BaseException,
    ) -> NoReturn:
        del ctx
        await self._pipeline.on_error(self._context, error)
        raise error

    async def after_run(
        self,
        ctx: RunContext[None],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        del ctx
        await self._pipeline.after_run(self._context)
        return result


class _WorkspaceToolGate(AbstractCapability[None]):
    def __init__(
        self,
        *,
        execution_id: str,
        workspace_root: Path,
        repository_instruction_history: tuple[ModelMessage, ...],
        repository_instruction_marker_authority: frozenset[tuple[str, str]],
        repository_instructions: RepositoryInstructions | None,
        instruction_resolver: RepositoryInstructionResolver,
        policy: WorkspacePolicy,
        trusted_tool_classes: tuple[tuple[str, str], ...],
    ) -> None:
        if not isinstance(execution_id, str) or not execution_id.strip():
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(workspace_root, Path):
            raise TypeError("workspace_root must be Path")
        if not isinstance(policy, WorkspacePolicy):
            raise TypeError("policy must be WorkspacePolicy")
        if not isinstance(repository_instruction_history, tuple) or any(
            not isinstance(message, (ModelRequest, ModelResponse))
            for message in repository_instruction_history
        ):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        if not isinstance(repository_instruction_marker_authority, frozenset):
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        for value in repository_instruction_marker_authority:
            if (
                not isinstance(value, tuple)
                or len(value) != 2
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        _validate_trusted_tool_classes(trusted_tool_classes)
        trusted = dict(trusted_tool_classes)
        if len(trusted) != len(trusted_tool_classes):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        self._execution_id = execution_id
        self._workspace_root = workspace_root
        self._instruction_resolver = instruction_resolver
        self._policy = policy
        self._trusted_tool_class_by_name = trusted
        self._repository_instructions_enabled = repository_instructions is not None
        self._exposure_map: dict[str, RepositoryInstructionDocument] = {}
        if repository_instructions is not None:
            self._exposure_map.update(
                (document.source, document) for document in repository_instructions.documents
            )
        self._marker_authority = repository_instruction_marker_authority
        self._refresh_required = False
        if self._repository_instructions_enabled:
            self._restore_exposure_map(repository_instruction_history)
            self._validate_active_limits()

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(wraps=(_ObservationalMiddlewareCapability,))

    def get_instructions(self) -> Callable[[RunContext[None]], str]:
        def active_repository_instructions(_ctx: RunContext[None]) -> str:
            if not self._repository_instructions_enabled:
                return ""
            return RepositoryInstructions(tuple(self._exposure_map.values())).render()

        return active_repository_instructions

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        del ctx
        self._refresh_required = False
        return request_context

    async def before_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del call
        if self._refresh_required:
            raise ToolFailed(
                "Repository instructions changed; reconsider this tool call on the next model step."
            )
        tool_class = self._trusted_tool_class_by_name.get(tool_def.name)
        decision = self._policy.tool_permissions.decide(
            tool_name=tool_def.name,
            tool_class=tool_class,
        )
        if decision == "deny":
            raise ToolFailed("Tool execution is denied by the current workspace policy.")
        if (
            self._repository_instructions_enabled
            and tool_def.name in _WORKSPACE_SCOPED_TOOL_NAMES
            and tool_class in {"filesystem.read", "filesystem.write"}
        ):
            target = args.get("path")
            if not isinstance(target, str):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            subset = await self._instruction_resolver.resolve(
                _repository_instruction_lookup_target(self._workspace_root, target),
                exclude_sources=frozenset(self._exposure_map),
            )
            if any(document.source in self._exposure_map for document in subset.documents):
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if subset.documents:
                candidate = RepositoryInstructions(
                    (*tuple(self._exposure_map.values()), *subset.documents)
                )
                self._validate_bundle_limits(candidate)
                marker = _repository_instruction_marker(self._execution_id, subset)
                if len(marker.encode("utf-8")) > self._policy.max_repository_instruction_bytes:
                    raise AIError(ErrorCode.PROMPT_TOO_LARGE)
                self._exposure_map = {
                    document.source: document for document in candidate.documents
                }
                self._refresh_required = True
                raise ToolFailed(marker)
        if decision == "ask" and not ctx.tool_call_approved:
            raise ApprovalRequired()
        return args

    def _restore_exposure_map(self, messages: tuple[ModelMessage, ...]) -> None:
        calls: dict[tuple[str, str], list[ToolCallPart]] = {}
        for message in messages:
            run_id = message.run_id
            if isinstance(message, ModelResponse):
                if not isinstance(run_id, str) or not run_id:
                    continue
                for part in message.parts:
                    if isinstance(part, ToolCallPart):
                        calls.setdefault((run_id, part.tool_call_id), []).append(part)
                continue
            if not isinstance(message, ModelRequest):
                continue
            if not isinstance(run_id, str) or not run_id:
                continue
            for part in message.parts:
                if not isinstance(part, ToolReturnPart) or part.outcome != "failed":
                    continue
                authority = (run_id, part.tool_call_id)
                if authority not in self._marker_authority:
                    continue
                content = part.content
                if not isinstance(content, str) or not content.startswith(_REPOSITORY_MARKER_HEADER):
                    continue
                current_execution = _marker_execution_id(content)
                if current_execution != self._execution_id:
                    continue
                paired = calls.get(authority, ())
                if len(paired) != 1:
                    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
                subset = self._parse_repository_instruction_marker(run_id, paired[0], part)
                if subset is None:
                    continue
                for document in subset.documents:
                    existing = self._exposure_map.get(document.source)
                    if existing is None:
                        self._exposure_map[document.source] = document
                    elif existing != document:
                        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)

    def _parse_repository_instruction_marker(
        self,
        run_id: str,
        call: ToolCallPart,
        result: ToolReturnPart,
    ) -> RepositoryInstructions | None:
        if (run_id, result.tool_call_id) not in self._marker_authority:
            return None
        content = result.content
        if result.outcome != "failed" or not isinstance(content, str):
            return None
        if not content.startswith(_REPOSITORY_MARKER_HEADER):
            return None
        execution_id = _marker_execution_id(content)
        if execution_id != self._execution_id:
            return None
        try:
            lines = content.split("\n")
            if len(lines) != 5:
                raise ValueError("repository marker line count is invalid")
            if lines[0] != _REPOSITORY_MARKER_HEADER:
                raise ValueError("repository marker header is invalid")
            if lines[1] != f"execution_id={self._execution_id}":
                raise ValueError("repository marker execution is invalid")
            digest_line = lines[2]
            if not digest_line.startswith("digest="):
                raise ValueError("repository marker digest is missing")
            digest = digest_line.removeprefix("digest=")
            if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
                raise ValueError("repository marker digest is invalid")
            if lines[3] != f"action={_REPOSITORY_MARKER_ACTION}":
                raise ValueError("repository marker action is invalid")
            if not lines[4].startswith("payload="):
                raise ValueError("repository marker payload is missing")
            payload_text = lines[4].removeprefix("payload=")
            raw = json.loads(payload_text)
            normalized = normalize_json_value(raw)
            if not isinstance(normalized, Mapping):
                raise ValueError("repository marker payload is not a mapping")
            if canonical_json_bytes(normalized).decode("utf-8") != payload_text:
                raise ValueError("repository marker payload is not canonical")
            instructions = RepositoryInstructions.from_payload(normalized)
            if instructions.digest != digest:
                raise ValueError("repository marker digest mismatch")
            tool_class = self._trusted_tool_class_by_name.get(call.tool_name)
            if (
                call.tool_name not in _WORKSPACE_SCOPED_TOOL_NAMES
                or tool_class not in {"filesystem.read", "filesystem.write"}
                or call.tool_call_id != result.tool_call_id
                or call.tool_name != result.tool_name
            ):
                raise ValueError("repository marker tool provenance is invalid")
            arguments = call.args_as_dict()
            target = arguments.get("path")
            if not isinstance(target, str):
                raise ValueError("repository marker target is invalid")
            target_scope = _logical_target_scope(
                self._workspace_root,
                _repository_instruction_target(target),
            )
            if any(
                not _scope_applies_to_target(document.scope, target_scope)
                for document in instructions.documents
            ):
                raise ValueError("repository marker scope is invalid")
            return instructions
        except AIError as error:
            if error.code in {
                ErrorCode.OUTPUT_CONTRACT_INVALID,
                ErrorCode.STORAGE_VERSION_UNSUPPORTED,
                ErrorCode.AGENT_INSTRUCTIONS_OUTSIDE_ROOT,
            }:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error
            raise
        except (json.JSONDecodeError, TypeError, ValueError, UnicodeError, OSError) as error:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR) from error

    def _validate_active_limits(self) -> None:
        self._validate_bundle_limits(
            RepositoryInstructions(tuple(self._exposure_map.values()))
        )

    def _validate_bundle_limits(self, instructions: RepositoryInstructions) -> None:
        if len(instructions.documents) > self._policy.max_repository_instruction_documents:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)
        if len(instructions.render().encode("utf-8")) > self._policy.max_repository_instruction_bytes:
            raise AIError(ErrorCode.PROMPT_TOO_LARGE)


def _repository_instruction_marker(
    execution_id: str,
    subset: RepositoryInstructions,
) -> str:
    if not isinstance(execution_id, str) or not execution_id.strip():
        raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
    return (
        _REPOSITORY_MARKER_HEADER
        + "\n"
        + f"execution_id={execution_id}\n"
        + f"digest={subset.digest}\n"
        + f"action={_REPOSITORY_MARKER_ACTION}\n"
        + "payload="
        + canonical_json_bytes(subset.to_payload()).decode("utf-8")
    )


def _marker_execution_id(content: str) -> str | None:
    if not isinstance(content, str):
        return None
    lines = content.split("\n", 2)
    if len(lines) < 2 or lines[0] != _REPOSITORY_MARKER_HEADER:
        return None
    execution_line = lines[1]
    prefix = "execution_id="
    if not execution_line.startswith(prefix):
        return None
    execution_id = execution_line.removeprefix(prefix)
    if not execution_id or execution_id != execution_id.strip():
        return None
    return execution_id


def _repository_instruction_target(target: str) -> str:
    return "." if target == "" else target


def _repository_instruction_lookup_target(root: Path, target: str) -> str:
    lookup_target = _repository_instruction_target(target)
    try:
        _logical_target_scope(root, lookup_target)
    except (OSError, ValueError) as error:
        raise ModelRetry(
            _format_model_tool_error(
                _MODEL_RETRY_PREFIX,
                "workspace path is invalid or outside the workspace; use a path within the workspace root and retry",
            )
        ) from error
    return lookup_target


def _logical_target_scope(root: Path, target: str) -> str:
    raw = os.fspath(target)
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise ValueError("repository marker target is invalid")
    root_value = os.fspath(root)
    if os.path.isabs(raw):
        normalized = Path(os.path.abspath(os.path.normpath(raw)))
    else:
        normalized = Path(
            os.path.abspath(os.path.normpath(os.path.join(root_value, raw)))
        )
    try:
        relative = normalized.relative_to(root)
    except (ValueError, OSError) as error:
        raise ValueError("repository marker target is outside workspace") from error
    scope = relative.as_posix()
    if scope in {"", "."}:
        return "."
    if (
        "\\" in scope
        or "\x00" in scope
        or any(character in "\r\n|[]" for character in scope)
    ):
        raise ValueError("repository marker target scope is invalid")
    parts = scope.split("/")
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError("repository marker target scope is invalid")
    return scope


def _scope_applies_to_target(scope: str, target: str) -> bool:
    if scope == ".":
        return True
    scope_parts = scope.split("/")
    target_parts = [] if target == "." else target.split("/")
    return (
        len(target_parts) >= len(scope_parts)
        and target_parts[: len(scope_parts)] == scope_parts
    )


async def compose_platform_capabilities(
    *,
    agent_name: str,
    conversation_id: "str | None",
    step_run_id: str,
    segment_sequence: "int | None",
    history_id: "str | None",
    memory_scope: "str | None",
    step_store: StepStore,
    memory_store: "SearchableMemoryStore | None",
    runtime_tool_names: "tuple[str, ...]",
    plan_mode: bool,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
    trusted_mcp_selectors: "tuple[str, ...]",
    context_target_tokens: "int | None",
    parent_step_run_id: "str | None",
    tool_operations: "ToolOperationBridge | None",
    background_tasks: "set[asyncio.Task[object]]",
    plan_store_resolver: "Callable[[RunContext[None]], PlanStore] | None",
    deferred_pause_sink: "Callable[[int], None] | None" = None,
) -> "tuple[AbstractCapability[None], ...]":
    _validate_compaction_target(context_target_tokens)
    _validate_trusted_tool_classes(trusted_tool_classes)
    _validate_trusted_mcp_selectors(trusted_mcp_selectors)
    capabilities: list[AbstractCapability[None]] = []
    capabilities.append(
        _RuntimeStepPersistence(
            store=step_store,
            agent_name=agent_name,
            run_id=step_run_id,
            parent_run_id=parent_step_run_id,
            metadata={
                "capability_scope": "parent",
                "agent_name": agent_name,
                **({} if history_id is None else {"history_id": history_id}),
                **({} if segment_sequence is None else {"segment_sequence": str(segment_sequence)}),
            },
            tool_operations=tool_operations or _MissingToolOperationBridge(),
            plan_mode=plan_mode,
            trusted_tool_classes=trusted_tool_classes,
            trusted_mcp_selectors=trusted_mcp_selectors,
            background_tasks=background_tasks,
            deferred_pause_sink=deferred_pause_sink,
        )
    )
    selected = frozenset(runtime_tool_names)
    selected_memory = tuple(name for name in MEMORY_TOOL_NAMES if name in selected)
    if selected_memory:
        if memory_store is None or memory_scope is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(
            _SelectedMemory(
                store=memory_store,
                namespace=memory_scope,
                agent_name="memory",
                inject_memory=False,
                guidance=_memory_guidance(selected_memory),
                selected_tool_names=selected_memory,
                id=_MEMORY_CAPABILITY_ID,
            )
        )
    if any(name in selected for name in PLANNING_TOOL_NAMES):
        if plan_store_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(
            Planning(
                id=_PLANNING_CAPABILITY_ID,
                tools=PLANNING_TOOL_NAMES,
                store_resolver=plan_store_resolver,
            )
        )
    capabilities.append(
        _build_compaction(None)
        if context_target_tokens is None
        else _CompactionCapability(
            context_target_tokens,
            step_store=step_store,
            conversation_id=conversation_id,
            step_run_id=step_run_id,
        )
    )
    return tuple(capabilities)


@dataclass
class _SelectedMemory(Memory[None]):
    selected_tool_names: "tuple[str, ...]" = ()

    def get_toolset(self) -> "AbstractToolset[None] | None":
        toolset = super().get_toolset()
        return None if toolset is None else toolset.filtered(
            lambda _ctx, definition: definition.name in self.selected_tool_names
        )


class _CompactionCapability(AbstractCapability[None]):
    def __init__(
        self,
        target_tokens: int,
        *,
        step_store: StepStore,
        conversation_id: "str | None",
        step_run_id: str,
    ) -> None:
        self._target_tokens = target_tokens
        self._step_store = step_store
        self._conversation_id = conversation_id
        self._step_run_id = step_run_id
        self._compaction = _build_compaction(target_tokens)

    async def before_run(self, ctx: RunContext[None]) -> None:
        if ctx.message_history:
            return
        if self._conversation_id is None:
            return
        try:
            run = await self._step_store.get_run(run_id=self._step_run_id)
            if run is None or not run.messages:
                return
            ctx.message_history[:] = list(run.messages)
        except BaseException:
            ctx.message_history[:] = []
            raise

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return await self._compaction.before_model_request(ctx, request_context)


def _memory_guidance(selected_tools: "tuple[str, ...]") -> str:
    actions = ", ".join(f"`{name}`" for name in selected_tools)
    return f"Use only these memory tools when needed: {actions}."


def _build_compaction(
    context_target_tokens: "int | None",
) -> AbstractCapability[None]:
    deduplicate = DeduplicateFileReads(file_key=_workspace_file_key)
    if context_target_tokens is None:
        return deduplicate
    return TieredCompaction(
        tiers=[
            deduplicate,
            ClearToolResults(max_tokens=1, keep_pairs=3),
            SummarizingCompaction(max_messages=1, keep_messages=20),
        ],
        target_tokens=context_target_tokens,
    )


def _workspace_file_key(part: ToolCallPart) -> "str | None":
    if part.tool_name != "read_file":
        return None
    try:
        arguments = part.args_as_dict()
    except (TypeError, ValueError):
        return None
    path = arguments.get("path")
    return path if isinstance(path, str) else None


def _model_usage_metadata(response: ModelResponse) -> dict[str, str]:
    usage = response.usage
    return {
        _MODEL_USAGE_INPUT_METADATA_KEY: _model_usage_token(usage.input_tokens),
        _MODEL_USAGE_OUTPUT_METADATA_KEY: _model_usage_token(usage.output_tokens),
        _MODEL_USAGE_CACHE_READ_METADATA_KEY: _model_usage_token(usage.cache_read_tokens),
        _MODEL_USAGE_CACHE_WRITE_METADATA_KEY: _model_usage_token(usage.cache_write_tokens),
    }


def _model_usage_token(value: object) -> str:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise AIError(ErrorCode.MODEL_RESPONSE_INVALID)
    return str(value)


def _validate_compaction_target(value: "int | None") -> None:
    if value is None:
        return
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)


def _validate_trusted_tool_classes(value: tuple[tuple[str, str], ...]) -> None:
    seen: set[str] = set()
    for item in value:
        if (
            not isinstance(item, tuple)
            or len(item) != 2
            or not isinstance(item[0], str)
            or not isinstance(item[1], str)
            or not item[0]
            or item[0] in seen
            or item[1] not in _TRUSTED_TOOL_CLASSES
        ):
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
        seen.add(item[0])


def _validate_trusted_mcp_selectors(value: tuple[str, ...]) -> None:
    if len(set(value)) != len(value):
        raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)
    for selector in value:
        if not selector.startswith("mcp__") or selector.endswith("__") or "*" in selector:
            raise AIError(ErrorCode.CAPABILITY_RESOLUTION_INVALID)


def tool_name_allowed(name: str, allow_tools: "tuple[str, ...]") -> bool:
    return "*" in allow_tools or name in allow_tools


def _trusted_tool_capability(name: str, tool_class: str) -> "str | None":
    if tool_class == "control":
        if name in SKILL_TOOL_NAMES:
            return _SKILL_CAPABILITY_ID
        if name in PLANNING_TOOL_NAMES:
            return _PLANNING_CAPABILITY_ID
        if name in SUBAGENT_TOOL_NAMES:
            return _SUBAGENT_CAPABILITY_ID
        return None
    if tool_class in {"filesystem.read", "filesystem.write"}:
        if name not in WORKSPACE_FILESYSTEM_TOOL_NAMES:
            return None
        is_read = name in WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
        if is_read != (tool_class == "filesystem.read"):
            return None
        return _WORKSPACE_FILESYSTEM_CAPABILITY_ID
    if tool_class == "shell":
        return _WORKSPACE_SHELL_CAPABILITY_ID if name in WORKSPACE_SHELL_TOOL_NAMES else None
    if tool_class in {"memory.read", "memory.write"}:
        if name not in MEMORY_TOOL_NAMES:
            return None
        is_read = name in MEMORY_READ_TOOL_NAMES
        if is_read != (tool_class == "memory.read"):
            return None
        return _MEMORY_CAPABILITY_ID
    return None


def _tool_effect_policy(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
) -> _ToolEffectPolicy:
    _validate_trusted_tool_classes(trusted_tool_classes)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class is not None:
        expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
        if expected_capability is None or tool_def.capability_id != expected_capability:
            raise AIError(
                ErrorCode.CAPABILITY_POLICY_CONFLICT,
                safe_details={"tool_name": tool_def.name},
            )
        if tool_class in {"filesystem.read", "memory.read"}:
            return _ToolEffectPolicy(True, True)
        if tool_class == "memory.write":
            return _ToolEffectPolicy(True, False)
        if tool_class == "filesystem.write":
            return _ToolEffectPolicy(False, False)
        if tool_class == "shell":
            return _ToolEffectPolicy(
                tool_def.name == "check_command",
                tool_def.name == "check_command",
            )
        if tool_class == "control":
            if tool_def.name in SKILL_TOOL_NAMES:
                return _ToolEffectPolicy(True, True)
            if tool_def.name in PLANNING_TOOL_NAMES or tool_def.name == "delegate_task":
                return _ToolEffectPolicy(True, False)
            if tool_def.name == "list_subagents":
                return _ToolEffectPolicy(True, True)
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
    metadata = (tool_def.metadata or {}).get(_REPLAY_SAFE_METADATA_KEY, False)
    if not isinstance(metadata, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return _ToolEffectPolicy(metadata, False)


def _durable_failure_error(
    error: Exception,
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
) -> Exception:
    if isinstance(error, (ToolRetryError, ToolFailedError, AIError)):
        return error
    if isinstance(error, (ValidationError, ModelRetry)):
        return ToolRetryError(
            RetryPromptPart.from_error(
                error,
                tool_name=tool_def.name,
                tool_call_id=call.tool_call_id,
            )
        )
    if isinstance(error, ToolFailed):
        return ToolFailedError(
            ToolReturnPart(
                tool_def.name,
                error.message,
                tool_call_id=call.tool_call_id,
                outcome="failed",
            )
        )
    raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)


def _model_tool_error(
    error: BaseException,
    *,
    call: ToolCallPart,
    tool_def: ToolDefinition,
) -> BaseException:
    if isinstance(error, ValidationError):
        content = RetryPromptPart.from_error(
            error,
            tool_name=tool_def.name,
            tool_call_id=call.tool_call_id,
        ).content
        return ModelRetry(
            _format_model_tool_error(
                _MODEL_RETRY_PREFIX,
                _model_tool_error_content(content, "correct the call and retry"),
            )
        )
    if isinstance(error, ModelRetry):
        return ModelRetry(_format_model_tool_error(_MODEL_RETRY_PREFIX, error.message))
    if isinstance(error, ToolRetryError):
        return ModelRetry(
            _format_model_tool_error(
                _MODEL_RETRY_PREFIX,
                _model_tool_error_content(
                    error.tool_retry.content,
                    "correct the call and retry",
                ),
            )
        )
    if isinstance(error, ToolFailed):
        return ToolFailed(_format_model_tool_error(_MODEL_FAILED_PREFIX, error.message))
    if isinstance(error, ToolFailedError):
        return ToolFailed(
            _format_model_tool_error(
                _MODEL_FAILED_PREFIX,
                _model_tool_error_content(
                    error.tool_failed.content,
                    "adapt and continue",
                ),
            )
        )
    if isinstance(error, AIError):
        if error.code is ErrorCode.TOOL_EXECUTION_FAILED:
            return ToolFailed(
                _format_model_tool_error(
                    _MODEL_FAILED_PREFIX,
                    _MODEL_EXECUTION_FAILED_MESSAGE,
                )
            )
        return error
    if isinstance(error, Exception):
        return ToolFailed(
            _format_model_tool_error(
                _MODEL_FAILED_PREFIX,
                _MODEL_EXECUTION_FAILED_MESSAGE,
            )
        )
    return error


def _model_tool_error_content(content: object, fallback: str) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(
            content,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError):
        return fallback


def _format_model_tool_error(prefix: str, message: str) -> str:
    value = f"{prefix}: {message}"
    if len(value) <= _MODEL_TOOL_ERROR_MAX_CHARS:
        return value
    tail_chars = (
        _MODEL_TOOL_ERROR_MAX_CHARS
        - _MODEL_TOOL_ERROR_HEAD_CHARS
        - len(_MODEL_TOOL_ERROR_TRUNCATION_MARKER)
    )
    return (
        value[:_MODEL_TOOL_ERROR_HEAD_CHARS]
        + _MODEL_TOOL_ERROR_TRUNCATION_MARKER
        + value[-tail_chars:]
    )


def _bypasses_tool_error_hook(error: BaseException) -> bool:
    return isinstance(
        error,
        (ModelRetry, ToolRetryError, ToolFailed, ToolFailedError, SkipToolExecution, CallDeferred, ApprovalRequired),
    )


def tool_is_control(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
) -> bool:
    _validate_trusted_tool_classes(trusted_tool_classes)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class != "control":
        return False
    expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
    return expected_capability is not None and tool_def.capability_id == expected_capability


def tool_allowed_in_planning(
    tool_def: ToolDefinition,
    *,
    trusted_tool_classes: "tuple[tuple[str, str], ...]",
    trusted_mcp_selectors: "tuple[str, ...]",
) -> bool:
    _validate_trusted_tool_classes(trusted_tool_classes)
    _validate_trusted_mcp_selectors(trusted_mcp_selectors)
    tool_class = dict(trusted_tool_classes).get(tool_def.name)
    if tool_class is not None:
        expected_capability = _trusted_tool_capability(tool_def.name, tool_class)
        if expected_capability is None:
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        if tool_def.capability_id == expected_capability:
            return tool_class in {"control", "filesystem.read", "memory.read"}
    if tool_def.capability_id in trusted_mcp_selectors:
        if not tool_def.name.startswith(f"{tool_def.capability_id}__"):
            raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
        return False
    if any(tool_def.name.startswith(f"{selector}__") for selector in trusted_mcp_selectors):
        return False
    metadata = (tool_def.metadata or {}).get(PLAN_SAFE_METADATA_KEY)
    if metadata is None:
        return False
    if not isinstance(metadata, bool):
        raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
    return metadata


def select_runtime_tool_names(
    *,
    ordinary_tool_policy: "tuple[str, ...]",
    memory_scope: "str | None",
    subagent_available: bool = False,
    planning: bool = False,
) -> "tuple[str, ...]":
    names: set[str] = set()
    if memory_scope is not None:
        names.update(
            name
            for name in MEMORY_TOOL_NAMES
            if tool_name_allowed(name, ordinary_tool_policy)
        )
    if planning:
        names.update(PLANNING_TOOL_NAMES)
    if subagent_available:
        names.update(SUBAGENT_TOOL_NAMES)
    return tuple(sorted(names))


__all__ = [
    "MEMORY_READ_TOOL_NAMES",
    "MEMORY_TOOL_NAMES",
    "PLANNING_TOOL_NAMES",
    "PLAN_SAFE_METADATA_KEY",
    "SUBAGENT_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_READ_TOOL_NAMES",
    "WORKSPACE_FILESYSTEM_TOOL_NAMES",
    "WORKSPACE_SHELL_TOOL_NAMES",
    "ToolOperationBridge",
    "ToolOperationDecision",
    "compose_platform_capabilities",
    "select_runtime_tool_names",
    "tool_allowed_in_planning",
    "tool_is_control",
    "tool_name_allowed",
]
