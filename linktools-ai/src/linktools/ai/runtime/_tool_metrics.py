#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-private observation of actual tool handler attempts."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any

from linktools.core import environ
from pydantic_ai.capabilities import (
    AbstractCapability,
    CapabilityOrdering,
    ValidatedToolArgs,
)
from pydantic_ai.exceptions import (
    ApprovalRequired,
    CallDeferred,
    SkipToolExecution,
)
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import RunContext as PydanticRunContext
from pydantic_ai.tools import ToolDefinition

from ..capability import AgentContext, ToolCallFailed, ToolCallRetry
from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Observation
from ._metric_id import _tool_observation_id

_logger = environ.get_logger("ai.runtime.tool_metrics")
TOOL_METRICS_MANAGED_METADATA_KEY = "linktools.tool_metrics_managed"


@dataclass(frozen=True, slots=True)
class _ToolMetricContext:
    recorder: MetricRecorder
    source_namespace: str
    tenant_id: str
    execution_id: str
    session_id: str | None
    step_run_id: str
    agent_id: str

    def record_error(
        self,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        error: Exception,
    ) -> None:
        """Record a model-visible failure before Pydantic handles it."""
        self._record(
            _tool_observation_id(
                self.source_namespace,
                self.tenant_id,
                self.execution_id,
                self.step_run_id,
                call.tool_call_id,
            ),
            call=call,
            tool_def=tool_def,
            started=monotonic_ns(),
            status="FAILED",
            error_code=_tool_error_code(error),
        )

    async def execute(
        self,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        suppress_cancel: Callable[[], bool],
    ) -> Any:
        attempt_id = _tool_observation_id(
            self.source_namespace,
            self.tenant_id,
            self.execution_id,
            self.step_run_id,
            call.tool_call_id,
        )
        started = monotonic_ns()
        try:
            result = await handler(args)
        except asyncio.CancelledError:
            if not suppress_cancel():
                self._record(
                    attempt_id,
                    call=call,
                    tool_def=tool_def,
                    started=started,
                    status="CANCELLED",
                    error_code=None,
                )
            raise
        except SkipToolExecution:
            self._record(
                attempt_id,
                call=call,
                tool_def=tool_def,
                started=started,
                status="SUCCEEDED",
                error_code=None,
            )
            raise
        except (ApprovalRequired, CallDeferred):
            self._record(
                attempt_id,
                call=call,
                tool_def=tool_def,
                started=started,
                status="DEFERRED",
                error_code=None,
            )
            raise
        except Exception as error:
            self._record(
                attempt_id,
                call=call,
                tool_def=tool_def,
                started=started,
                status="FAILED",
                error_code=_tool_error_code(error),
            )
            raise
        self._record(
            attempt_id,
            call=call,
            tool_def=tool_def,
            started=started,
            status="SUCCEEDED",
            error_code=None,
        )
        return result

    def _record(
        self,
        observation_id: str,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        started: int,
        status: str,
        error_code: str | None,
    ) -> None:
        correlation: dict[str, str | int] = {
            "execution_id": self.execution_id,
            "step_run_id": self.step_run_id,
            "tool_call_id": call.tool_call_id,
        }
        if self.session_id is not None:
            correlation["session_id"] = self.session_id
        try:
            observation = Observation(
                version=1,
                observation_id=observation_id,
                kind="linktools.tool.execution",
                occurred_at=datetime.now(timezone.utc),
                source_namespace=self.source_namespace,
                tenant_id=self.tenant_id,
                status=status,
                error_code=error_code,
                correlation=correlation,
                dimensions={
                    "agent_id": self.agent_id,
                    "tool_name": tool_def.name,
                },
                measurements=(
                    MetricMeasurement("latency_ns", 1, monotonic_ns() - started),
                ),
            )
            self.recorder.try_record(observation)
        except Exception:
            _logger.exception("tool metric observation rejected")


class RuntimeToolMetricsCapability(
    AbstractCapability[AgentContext[object]]
):
    """Observe capability tools before the outer Pydantic control boundary."""

    def __init__(self, context: _ToolMetricContext) -> None:
        self.id = "linktools.ai.tool-metrics"
        self._context = context

    def get_ordering(self) -> CapabilityOrdering:
        return CapabilityOrdering(position="innermost")

    async def wrap_tool_execute(
        self,
        ctx: PydanticRunContext[AgentContext[object]],
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: ValidatedToolArgs,
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
    ) -> Any:
        del ctx
        if (
            tool_def.metadata is not None
            and tool_def.metadata.get(TOOL_METRICS_MANAGED_METADATA_KEY) is True
        ):
            return await handler(args)
        return await self._context.execute(
            call=call,
            tool_def=tool_def,
            args=args,
            handler=handler,
            suppress_cancel=lambda: False,
        )


def _tool_error_code(error: Exception) -> str:
    if isinstance(error, AIError):
        return error.code.value
    if isinstance(error, ToolCallRetry):
        return ErrorCode.TOOL_RETRY_REQUIRED.value
    if isinstance(error, ToolCallFailed):
        return ErrorCode.TOOL_EXECUTION_FAILED.value
    return ErrorCode.TOOL_EXECUTION_FAILED.value


__all__ = [
    "RuntimeToolMetricsCapability",
    "TOOL_METRICS_MANAGED_METADATA_KEY",
]
