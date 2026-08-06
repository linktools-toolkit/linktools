#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Durable event append coordination."""

from datetime import datetime

from ...domain.execution import ExecutionEvent
from ...foundation.digest import hmac_digest
from ...foundation.json import canonical_json_bytes


class EventService:
    def __init__(self, repository: object) -> None:
        self._repository = repository

    async def append(
        self,
        execution_id: str,
        event_type: str,
        sequence: int,
        occurred_at: datetime,
        payload: object,
        *,
        source_id: str = "",
        source_phase: str = "",
    ) -> ExecutionEvent:
        event_id = hmac_digest(
            execution_id.encode("utf-8"),
            canonical_json_bytes((event_type, source_id, source_phase)),
        )
        event = ExecutionEvent(event_id=event_id, execution_id=execution_id, sequence=sequence, event_type=event_type, source_id=source_id, source_phase=source_phase, occurred_at=occurred_at, recorded_at=occurred_at, payload=payload)
        return await self._repository.append(event)

    async def list_after(self, execution_id: str, after_sequence: int, limit: int) -> object:
        return await self._repository.list_after(execution_id, after_sequence, min(limit, 200))


__all__ = ["EventService"]
