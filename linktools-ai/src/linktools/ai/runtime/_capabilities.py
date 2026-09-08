#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Pydantic AI capability composition with Harness-owned generic capabilities."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, cast

from pydantic import ValidationError
from pydantic_ai.capabilities import (
    AbstractCapability,
    AgentNode,
    NodeResult,
    WrapModelRequestHandler,
    WrapToolExecuteHandler,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    ModelRetry,
    RunCancelled,
    SkipToolExecution,
    ToolFailed,
    ToolFailedError,
    ToolRetryError,
)
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart
from pydantic_ai.models import ModelRequestContext
from pydantic_ai.run import AgentRunResult
from pydantic_ai.tools import DeferredToolRequests, RunContext, ToolDefinition
from pydantic_ai_harness.planning import Planning
from pydantic_ai_harness.step_persistence import StepPersistence

import linktools.ai.runtime._capabilities_native as _native
from ._compaction import ExternalModelRequestObserver
from ._harness import (
    HarnessPlanStoreAdapter,
    HarnessStepStoreAdapter,
    bind_tool_operation_id,
    reset_tool_operation_id,
)
from ._harness_memory import build_harness_memory
from ._journal import ModelRequestFact, ModelRequestJournal
from ._memory import MemoryStore
from ._plan import RuntimePlanStore
from ._tool_metrics import _ToolMetricContext
from .state import StepStore
from ..errors import AIError, ErrorCode

MEMORY_TOOL_NAMES = _native.MEMORY_TOOL_NAMES
MEMORY_READ_TOOL_NAMES = _native.MEMORY_READ_TOOL_NAMES
PLANNING_TOOL_NAMES = _native.PLANNING_TOOL_NAMES
PLAN_SAFE_METADATA_KEY = _native.PLAN_SAFE_METADATA_KEY
SUBAGENT_TOOL_NAMES = _native.SUBAGENT_TOOL_NAMES
WORKSPACE_FILESYSTEM_READ_TOOL_NAMES = _native.WORKSPACE_FILESYSTEM_READ_TOOL_NAMES
WORKSPACE_FILESYSTEM_TOOL_NAMES = _native.WORKSPACE_FILESYSTEM_TOOL_NAMES
WORKSPACE_SHELL_TOOL_NAMES = _native.WORKSPACE_SHELL_TOOL_NAMES
ToolOperationBridge = _native.ToolOperationBridge
ToolOperationDecision = _native.ToolOperationDecision
_CompactionCapability = _native._CompactionCapability
_MissingToolOperationBridge = _native._MissingToolOperationBridge
_WorkspaceToolGate = _native._WorkspaceToolGate
_model_usage_metadata = _native._model_usage_metadata
_repository_instruction_marker = _native._repository_instruction_marker
_tool_execution_policy = _native._tool_execution_policy

select_runtime_tool_names = _native.select_runtime_tool_names
tool_allowed_in_planning = _native.tool_allowed_in_planning
tool_is_control = _native.tool_is_control
tool_name_allowed = _native.tool_name_allowed

_MEMORY_CAPABILITY_ID = _native._MEMORY_CAPABILITY_ID
_PLANNING_CAPABILITY_ID = _native._PLANNING_CAPABILITY_ID
_MODEL_EFFECT_UNKNOWN_MESSAGE = _native._MODEL_EFFECT_UNKNOWN_MESSAGE

@dataclass(kw_only=True, eq=False)
class _RuntimeStepPersistence(StepPersistence[None]):
    """Use Harness for graph persistence while LinkTools owns durable tool effects."""

    execution_id: str | None = field(default=None, repr=False, compare=False)
    tool_operations: ToolOperationBridge = field(repr=False, compare=False)
    plan_mode: bool = False
    trusted_tool_classes: tuple[tuple[str, str], ...] = ()
    trusted_mcp_selectors: tuple[str, ...] = ()
    background_tasks: set[asyncio.Task[Any]] = field(
        default_factory=set,
        repr=False,
        compare=False,
    )
    deferred_pause_sink: Callable[[int], None] | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    tool_metrics: _ToolMetricContext | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    model_journal: ModelRequestJournal | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    model_request_observer: ExternalModelRequestObserver | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    _calls: dict[tuple[str, str], _native._ToolCallState] = field(
        default_factory=dict,
        init=False,
        repr=False,
        compare=False,
    )
    _last_observed_step_index: int | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def __post_init__(self) -> None:
        if not isinstance(self.plan_mode, bool):
            raise AIError(ErrorCode.REQUEST_FIELD_INVALID)
        _native._validate_trusted_tool_classes(self.trusted_tool_classes)
        _native._validate_trusted_mcp_selectors(self.trusted_mcp_selectors)
        if not isinstance(self.store, HarnessStepStoreAdapter):
            self.store = HarnessStepStoreAdapter(
                cast(StepStore, self.store),
                execution_id=self.execution_id,
            )
        if self.model_journal is None and self.tool_metrics is not None:
            self.model_journal = ModelRequestJournal(
                source_namespace=self.tool_metrics.source_namespace,
                tenant_id=self.tool_metrics.tenant_id,
                execution_id=self.tool_metrics.execution_id,
                step_run_id=self.tool_metrics.step_run_id,
            )

    @property
    def _runtime_store(self) -> HarnessStepStoreAdapter:
        if not isinstance(self.store, HarnessStepStoreAdapter):
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        return self.store

    def _runtime_run_id(self, ctx: RunContext[Any]) -> str:
        value = self.run_id or ctx.run_id
        if not isinstance(value, str) or not value:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        return value

    async def after_node_run(
        self,
        ctx: RunContext[None],
        *,
        node: AgentNode[None],
        result: NodeResult[None],
    ) -> NodeResult[None]:
        observed = await super().after_node_run(ctx, node=node, result=result)
        self._last_observed_step_index = ctx.run_step
        return observed

    async def after_run(
        self,
        ctx: RunContext[None],
        *,
        result: AgentRunResult[Any],
    ) -> AgentRunResult[Any]:
        output = result.output
        interrupted = isinstance(output, DeferredToolRequests)
        if interrupted:
            if not output.approvals or output.calls:
                raise AIError(ErrorCode.CAPABILITY_POLICY_CONFLICT)
            if self._last_observed_step_index is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            if self.deferred_pause_sink is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            self.deferred_pause_sink(self._last_observed_step_index)
            self._runtime_store.mark_interrupted(self._runtime_run_id(ctx))
        observed = await super().after_run(ctx, result=result)
        if interrupted:
            await self._runtime_store.save_interrupted_snapshot(
                run_id=self._runtime_run_id(ctx),
                step_index=cast(int, self._last_observed_step_index),
                messages=result.all_messages(),
                conversation_id=ctx.conversation_id,
                parent_run_id=self.parent_run_id,
                agent_name=self.agent_name,
            )
        return observed

    async def on_run_error(
        self,
        ctx: RunContext[None],
        *,
        error: BaseException,
    ) -> AgentRunResult[Any]:
        return await super().on_run_error(ctx, error=error)

    def remember_context_projection(
        self,
        source: Sequence[ModelMessage],
        projected: Sequence[ModelMessage] | None,
    ) -> None:
        self._runtime_store.remember_context_projection(source, projected)

    async def before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
    ) -> ModelRequestContext:
        return request_context

    async def wrap_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        handler: WrapModelRequestHandler,
    ) -> ModelResponse:
        fact = (
            None
            if self.model_journal is None
            else self.model_journal.begin(ctx.run_step, purpose="agent")
        )
        start_metadata = (
            {}
            if fact is None
            else fact.metadata(include_observation=self.tool_metrics is not None)
        )
        await self._harness_before_model_request(ctx, request_context, start_metadata)
        try:
            response = await handler(request_context)
        except asyncio.CancelledError as error:
            if self.model_journal is None:
                raise
            self.model_journal.finish(ctx.run_step, status="CANCELLED")
            fact = self.model_journal.consume(ctx.run_step)
            await self._harness_cancelled_model_request(
                ctx,
                request_context,
                error,
                fact,
            )
            raise AssertionError("Harness cancellation hook must re-raise")
        except RunCancelled:
            if self.model_journal is not None:
                self.model_journal.finish(ctx.run_step, status="CANCELLED")
            raise
        except BaseException:
            if self.model_journal is not None:
                self.model_journal.finish(ctx.run_step, status="FAILED")
            raise
        if self.model_journal is not None:
            self.model_journal.finish(ctx.run_step, status="SUCCEEDED")
        return response

    async def _harness_before_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
        metadata: Mapping[str, str],
    ) -> None:
        previous = self.metadata
        self.metadata = {**previous, **metadata}
        try:
            await super().before_model_request(ctx, request_context)
        finally:
            self.metadata = previous

    async def _harness_cancelled_model_request(
        self,
        ctx: RunContext[None],
        request_context: ModelRequestContext,
        error: asyncio.CancelledError,
        fact: ModelRequestFact,
    ) -> None:
        metadata = fact.metadata(include_observation=self.tool_metrics is not None)
        previous = self.metadata
        self.metadata = {**previous, **metadata}
        try:
            try:
                await super().on_model_request_error(
                    ctx,
                    request_context=request_context,
                    error=cast(Exception, error),
                )
            except BaseException:
                if self.model_request_observer is not None:
                    await self.model_request_observer(
                        ctx,
                        fact,
                        "cancelled",
                        None,
                        error,
                    )
                raise
        finally:
            self.metadata = previous

    async def after_model_request(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        response: ModelResponse,
    ) -> ModelResponse:
        self._runtime_store.capture_model_context(request_context.messages)
        fact = self._consume_model_fact(ctx, status="SUCCEEDED")
        metadata = _model_usage_metadata(response)
        if fact is not None:
            metadata.update(
                fact.metadata(include_observation=self.tool_metrics is not None)
            )
        previous = self.metadata
        self.metadata = {**previous, **metadata}
        try:
            observed = await super().after_model_request(
                ctx,
                request_context=request_context,
                response=response,
            )
        finally:
            self.metadata = previous
        if fact is not None and self.model_request_observer is not None:
            await self.model_request_observer(
                ctx,
                fact,
                "completed",
                response,
                None,
            )
        return observed

    async def on_model_request_error(
        self,
        ctx: RunContext[None],
        *,
        request_context: ModelRequestContext,
        error: Exception,
    ) -> ModelResponse:
        self._runtime_store.capture_model_context(request_context.messages)
        fact = self._consume_model_fact(
            ctx,
            status=(
                "CANCELLED"
                if isinstance(error, (asyncio.CancelledError, RunCancelled))
                else "FAILED"
            ),
        )
        metadata = (
            {}
            if fact is None
            else fact.metadata(include_observation=self.tool_metrics is not None)
        )
        previous = self.metadata
        self.metadata = {**previous, **metadata}
        try:
            try:
                await super().on_model_request_error(
                    ctx,
                    request_context=request_context,
                    error=error,
                )
            except BaseException:
                if fact is not None and self.model_request_observer is not None:
                    await self.model_request_observer(
                        ctx,
                        fact,
                        "cancelled"
                        if isinstance(error, (asyncio.CancelledError, RunCancelled))
                        else "failed",
                        None,
                        error,
                    )
                raise
        finally:
            self.metadata = previous
        raise AssertionError("Harness model error hook must re-raise")

    def _consume_model_fact(
        self,
        ctx: RunContext[None],
        *,
        status: str,
    ) -> ModelRequestFact | None:
        if self.model_journal is None:
            return None
        try:
            fact = self.model_journal.current(ctx.run_step)
        except RuntimeError:
            return None
        if fact.duration_ns is None:
            self.model_journal.finish(ctx.run_step, status=status)
        return self.model_journal.consume(ctx.run_step)

    async def record_external_model_request(
        self,
        ctx: RunContext[Any],
        fact: ModelRequestFact,
        phase: str,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None:
        if phase not in {"started", "completed", "failed", "cancelled"}:
            raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
        metadata = fact.metadata(include_observation=self.tool_metrics is not None)
        if phase == "completed":
            if response is None:
                raise AIError(ErrorCode.STORAGE_INTEGRITY_ERROR)
            metadata.update(_model_usage_metadata(response))
        await self._runtime_store.append_runtime_event(
            run_id=self._runtime_run_id(ctx),
            kind=(
                "model_request_started"
                if phase == "started"
                else "model_request_completed"
                if phase == "completed"
                else "model_request_failed"
            ),
            step_index=ctx.run_step,
            conversation_id=ctx.conversation_id,
            parent_run_id=self.parent_run_id,
            agent_name=self.agent_name,
            metadata=metadata,
            error=None if error is None else repr(error),
        )

    async def before_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
    ) -> dict[str, Any]:
        del ctx, call
        _tool_execution_policy(
            tool_def,
            trusted_tool_classes=self.trusted_tool_classes,
        )
        if self.plan_mode and not tool_allowed_in_planning(
            tool_def,
            trusted_tool_classes=self.trusted_tool_classes,
            trusted_mcp_selectors=self.trusted_mcp_selectors,
        ):
            raise AIError(
                ErrorCode.CAPABILITY_POLICY_CONFLICT,
                safe_details={"tool_name": tool_def.name, "mode": "plan"},
            )
        return args

    async def wrap_tool_execute(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: WrapToolExecuteHandler,
    ) -> Any:
        policy = _tool_execution_policy(
            tool_def,
            trusted_tool_classes=self.trusted_tool_classes,
        )
        effective_args_method = getattr(self.tool_operations, "effective_args", None)
        effective_args = (
            args
            if effective_args_method is None
            else await effective_args_method(ctx, call, tool_def, args)
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
        state = _native._ToolCallState(
            decision=decision,
            policy=policy,
            operation_terminalized=(
                decision.has_cached_result or decision.cached_error is not None
            ),
            cached_failure=decision.cached_error is not None,
        )
        self._calls[key] = state
        try:
            await super().before_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
            )
        except BaseException:
            self._calls.pop(key, None)
            raise
        if state.decision.cached_error is not None:
            error = state.decision.cached_error
            try:
                await self._record_failed_tool(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=error,
                    state=state,
                )
            except BaseException as raised:
                if _native._bypasses_tool_error_hook(raised):
                    self._calls.pop(key, None)
                raise
            raise AssertionError("cached failure tool hook must raise")
        if state.decision.has_cached_result:
            return state.decision.cached_result

        async def tracked_handler(validated_args: dict[str, Any]) -> Any:
            state.handler_entered = True
            if self.tool_metrics is not None:
                state.metric_started_ns = _native.monotonic_ns()
            token = bind_tool_operation_id(state.decision.operation_id)
            try:
                if self.tool_metrics is None:
                    return await handler(validated_args)
                return await self.tool_metrics.execute(
                    call=call,
                    tool_def=tool_def,
                    args=validated_args,
                    handler=handler,
                    suppress_cancel=lambda: state.suppress_cancel_metric,
                )
            finally:
                reset_tool_operation_id(token)

        handler_task = asyncio.create_task(
            tracked_handler(effective_args),
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
                    state.suppress_cancel_metric = True
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
            await self._record_completed_tool(
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
                await self._fail_known_tool(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=error,
                    state=state,
                )
            except BaseException as raised:
                if state.operation_terminalized and _native._bypasses_tool_error_hook(raised):
                    keep_call_state = False
                    self._calls.pop(key, None)
                raise
            raise AssertionError("known failure tool hook must raise")
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
            await self._fail_known_tool(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=unsupported,
                state=state,
            )
            raise AssertionError("dynamic deferred failure hook must raise")
        except asyncio.CancelledError as error:
            if not state.handler_observed:
                state.suppress_cancel_metric = True
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
            if not state.handler_entered or state.policy.effect_free:
                raise
            if isinstance(error, AIError):
                if state.policy.replay_safe:
                    state.preserve_started = True
                else:
                    await self._mark_unknown(state, error)
                raise
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
                    state.suppress_cancel_metric = True
                    handler_task.cancel()
                    self._detach_task(handler_task, "tool handler cleanup")
                elif not state.handler_observed:
                    self._consume_task(handler_task, "tool handler cleanup")
            if not keep_call_state:
                self._calls.pop(key, None)

    def _tool_metric_metadata(
        self,
        call: ToolCallPart,
        state: _native._ToolCallState,
    ) -> dict[str, str]:
        metadata = dict(self.metadata)
        metrics = self.tool_metrics
        if metrics is None or not state.handler_entered or state.metric_started_ns is None:
            return metadata
        metadata[_native._OBSERVATION_ID_METADATA_KEY] = _native._tool_observation_id(
            metrics.source_namespace,
            metrics.tenant_id,
            metrics.execution_id,
            metrics.step_run_id,
            call.tool_call_id,
        )
        metadata[_native._DURATION_NS_METADATA_KEY] = str(
            _native.monotonic_ns() - state.metric_started_ns
        )
        return metadata

    async def _persist_tool_completed(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
        state: _native._ToolCallState,
    ) -> Any:
        previous = self.metadata
        self.metadata = self._tool_metric_metadata(call, state)
        try:
            return await super().after_tool_execute(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=result,
            )
        finally:
            self.metadata = previous

    async def _persist_tool_failed(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: BaseException,
        state: _native._ToolCallState,
    ) -> Any:
        previous = self.metadata
        self.metadata = self._tool_metric_metadata(call, state)
        try:
            return await super().on_tool_execute_error(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=cast(Exception, error),
            )
        finally:
            self.metadata = previous

    async def after_tool_execute(
        self,
        ctx: RunContext[None],
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
            return await self._persist_tool_completed(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=result,
                state=state,
            )
        finally:
            self._calls.pop(key, None)

    async def on_tool_execute_error(
        self,
        ctx: RunContext[None],
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
                return await self._fail_known_tool(
                    ctx,
                    call=call,
                    tool_def=tool_def,
                    args=args,
                    error=error,
                    state=state,
                )
            if state.handler_entered and not state.policy.effect_free:
                if state.policy.replay_safe:
                    state.preserve_started = True
                    raise AIError(
                        ErrorCode.TOOL_EFFECT_UNKNOWN,
                        safe_details={"phase": "tool_effect_replay"},
                    ) from error
                await self._mark_unknown(state, error)
                raise ToolFailed(_MODEL_EFFECT_UNKNOWN_MESSAGE) from error
            state.preserve_started = True
            cancelled = await self.tool_operations.fail(state.decision, error)
            state.operation_terminalized = True
            state.preserve_started = False
            if cancelled:
                self._calls.pop(key, None)
                raise asyncio.CancelledError
            return await self._record_failed_tool(
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
        state: _native._ToolCallState,
        handler_task: asyncio.Task[Any],
    ) -> None:
        while not handler_task.done():
            await asyncio.sleep(15)
            if handler_task.done():
                return
            state.decision = await self.tool_operations.renew(state.decision)

    async def _stop_heartbeat(self, state: _native._ToolCallState) -> None:
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
            _native._logger.exception("detached %s failed", label)

    async def _fail_known_tool(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: Exception,
        state: _native._ToolCallState,
    ) -> Any:
        state.preserve_started = True
        durable_error = _native._durable_failure_error(
            error,
            call=call,
            tool_def=tool_def,
        )
        cancelled = await self.tool_operations.fail(state.decision, durable_error)
        state.operation_terminalized = True
        state.preserve_started = False
        if cancelled:
            raise asyncio.CancelledError
        return await self._record_failed_tool(
            ctx,
            call=call,
            tool_def=tool_def,
            args=args,
            error=error,
            state=state,
        )

    async def _record_failed_tool(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        error: BaseException,
        state: _native._ToolCallState,
    ) -> Any:
        if state.terminal_event_recorded:
            raise error
        try:
            result = await self._persist_tool_failed(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                error=error,
                state=state,
            )
        except BaseException as raised:
            state.terminal_event_recorded = True
            if raised is error:
                model_error = _native._model_tool_error(
                    error,
                    call=call,
                    tool_def=tool_def,
                )
                if model_error is not error:
                    raise model_error from error
            raise
        state.terminal_event_recorded = True
        return result

    async def _record_completed_tool(
        self,
        ctx: RunContext[None],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        result: Any,
        state: _native._ToolCallState,
    ) -> Any:
        if state.terminal_event_recorded:
            return result
        try:
            value = await self._persist_tool_completed(
                ctx,
                call=call,
                tool_def=tool_def,
                args=args,
                result=result,
                state=state,
            )
        except BaseException:
            state.terminal_event_recorded = True
            raise
        state.terminal_event_recorded = True
        return value

    async def _mark_unknown(
        self,
        state: _native._ToolCallState,
        error: BaseException,
    ) -> None:
        state.preserve_started = True
        state.operation_terminalized = True
        await self.tool_operations.unknown(state.decision, error)

    def _decision_key(
        self,
        ctx: RunContext[None],
        call: ToolCallPart,
    ) -> tuple[str, str]:
        return self._runtime_run_id(ctx), call.tool_call_id


def _external_observer(
    persistence: _RuntimeStepPersistence,
    observer: ExternalModelRequestObserver | None,
) -> ExternalModelRequestObserver:
    async def record(
        ctx: RunContext[Any],
        fact: ModelRequestFact,
        phase: str,
        response: ModelResponse | None,
        error: BaseException | None,
    ) -> None:
        await persistence.record_external_model_request(
            ctx,
            fact,
            phase,
            response,
            error,
        )
        if observer is not None:
            await observer(ctx, fact, phase, response, error)

    return record


async def compose_platform_capabilities(
    *,
    agent_name: str,
    conversation_id: str | None,
    step_run_id: str,
    execution_id: str | None = None,
    segment_sequence: int | None,
    history_id: str | None,
    memory_scope: str | None,
    step_store: StepStore,
    memory_store: MemoryStore | None,
    runtime_tool_names: tuple[str, ...],
    plan_mode: bool,
    trusted_tool_classes: tuple[tuple[str, str], ...],
    trusted_mcp_selectors: tuple[str, ...],
    context_target_tokens: int | None,
    parent_step_run_id: str | None,
    tool_operations: ToolOperationBridge | None,
    background_tasks: set[asyncio.Task[object]],
    plan_store_resolver: Callable[[RunContext[None]], RuntimePlanStore] | None,
    deferred_pause_sink: Callable[[int], None] | None = None,
    tool_metrics: _ToolMetricContext | None = None,
    model_journal: ModelRequestJournal | None = None,
    external_model_request_observer: ExternalModelRequestObserver | None = None,
) -> tuple[AbstractCapability[None], ...]:
    _native._validate_compaction_target(context_target_tokens)
    _native._validate_trusted_tool_classes(trusted_tool_classes)
    _native._validate_trusted_mcp_selectors(trusted_mcp_selectors)
    capabilities: list[AbstractCapability[None]] = []
    persistence = _RuntimeStepPersistence(
        store=cast(Any, step_store),
        execution_id=execution_id,
        agent_name=agent_name,
        run_id=step_run_id,
        parent_run_id=parent_step_run_id,
        metadata={
            "capability_scope": "parent",
            "agent_name": agent_name,
            **({} if history_id is None else {"history_id": history_id}),
            **(
                {}
                if segment_sequence is None
                else {"segment_sequence": str(segment_sequence)}
            ),
        },
        tool_operations=tool_operations or _MissingToolOperationBridge(),
        plan_mode=plan_mode,
        trusted_tool_classes=trusted_tool_classes,
        trusted_mcp_selectors=trusted_mcp_selectors,
        background_tasks=background_tasks,
        deferred_pause_sink=deferred_pause_sink,
        tool_metrics=tool_metrics,
        model_journal=model_journal,
        model_request_observer=external_model_request_observer,
    )
    capabilities.append(persistence)
    selected = frozenset(runtime_tool_names)
    selected_memory = tuple(name for name in MEMORY_TOOL_NAMES if name in selected)
    if selected_memory:
        if memory_store is None or memory_scope is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
        capabilities.append(
            build_harness_memory(
                memory_store,
                selected_tool_names=selected_memory,
                capability_id=_MEMORY_CAPABILITY_ID,
            )
        )
    if any(name in selected for name in PLANNING_TOOL_NAMES):
        if plan_store_resolver is None:
            raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)

        def resolve_plan_store(ctx: RunContext[None]) -> HarnessPlanStoreAdapter:
            store = plan_store_resolver(ctx)
            if store is None:
                raise AIError(ErrorCode.RUNTIME_DEPENDENCY_NOT_READY)
            return HarnessPlanStoreAdapter(store)

        capabilities.append(
            Planning(
                id=_PLANNING_CAPABILITY_ID,
                tools=PLANNING_TOOL_NAMES,
                store_resolver=resolve_plan_store,
            )
        )
    capabilities.append(
        _CompactionCapability(
            context_target_tokens,
            trusted_workspace_read=(
                dict(trusted_tool_classes).get("read_file") == "filesystem.read"
            ),
            journal=model_journal,
            observer=_external_observer(
                persistence,
                external_model_request_observer,
            ),
            projection_sink=persistence.remember_context_projection,
        )
    )
    return tuple(capabilities)


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
