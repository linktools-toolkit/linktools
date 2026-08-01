#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Semantic trace collection with persisted sequence continuity."""


import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timezone
from ..json import normalize_json
from .snapshots import AgentSnapshotData
from .trace_models import NewRunTraceStep

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..json import JsonValue
    from .domain import RunStatus, RunUsage
    from .store import ExecutionStore

@dataclass(slots=True)
class SemanticTraceCollector:
    run_id: str
    trace_port: "ExecutionStore"
    next_sequence: int = 0
    _pending: "list[NewRunTraceStep]" = field(default_factory=list)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _add(self, kind: str, payload: "JsonValue") -> None:
        self._pending.append(
            NewRunTraceStep(
                kind=kind,
                payload=normalize_json(payload),
                created_at=datetime.now(timezone.utc),
            )
        )

    async def model_request_succeeded(self, payload: "JsonValue") -> None:
        async with self._lock:
            self._add("model_interaction", payload)
            await self._flush_locked()

    async def model_request_failed(self, payload: "JsonValue") -> None:
        async with self._lock:
            self._add("model_interaction", payload)
            await self._flush_locked()

    async def tool_result(self, payload: "JsonValue") -> None:
        async with self._lock:
            self._add("tool_result", payload)
            await self._flush_locked()

    async def _flush_locked(self) -> int:
        if not self._pending:
            return self.next_sequence
        steps = tuple(self._pending)
        self.next_sequence = await self.trace_port.append_trace_steps(
            self.run_id, expected_sequence=self.next_sequence, steps=steps
        )
        self._pending.clear()
        return self.next_sequence

    async def flush(self) -> int:
        async with self._lock:
            return await self._flush_locked()

    async def build_snapshot(self, *, resume_messages: "tuple[JsonValue, ...]", final_output: "JsonValue | str | None", status: "RunStatus", usage: "RunUsage") -> AgentSnapshotData:
        await self.flush()
        return AgentSnapshotData(
            tuple(normalize_json(message) for message in resume_messages),
            normalize_json(final_output) if final_output is not None else None,
            usage,
            self.next_sequence,
        )
