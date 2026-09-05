#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Runtime-private observation of actual tool handler attempts."""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime, timezone
from time import monotonic_ns
from typing import Any

from linktools.core import environ
from pydantic import ValidationError
from pydantic_ai.exceptions import ModelRetry, ToolFailed, ToolFailedError, ToolRetryError
from pydantic_ai.messages import ToolCallPart
from pydantic_ai.tools import ToolDefinition

from ..errors import AIError, ErrorCode
from ..observe import MetricMeasurement, MetricRecorder, Observation

_logger = environ.get_logger("ai.runtime.tool_metrics")


@dataclass(frozen=True, slots=True)
class _ToolMetricContext:
    recorder: MetricRecorder
    source_namespace: str
    tenant_id: str
    execution_id: str
    session_id: str | None
    step_run_id: str
    agent_id: str

    async def execute(
        self,
        *,
        call: ToolCallPart,
        tool_def: ToolDefinition,
        args: dict[str, Any],
        handler: Callable[[dict[str, Any]], Awaitable[Any]],
        suppress_cancel: Callable[[], bool],
    ) -> Any:
        attempt_id = uuid.uuid4().hex
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


def _tool_error_code(error: Exception) -> str:
    if isinstance(error, AIError):
        return error.code.value
    if isinstance(error, (ValidationError, ModelRetry, ToolRetryError)):
        return ErrorCode.TOOL_RETRY_REQUIRED.value
    if isinstance(error, (ToolFailed, ToolFailedError)):
        return ErrorCode.TOOL_EXECUTION_FAILED.value
    return ErrorCode.TOOL_EXECUTION_FAILED.value


__all__: list[str] = []
