#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyEventStore: DB-backed EventStore. The unique (run_id, sequence)
constraint on ai_events makes append's duplicate-sequence rejection a real
DB-level guarantee that fires unconditionally (i.e. even when the caller does
not pass expected_sequence), not just an application-level check gated behind
an optional argument."""

import json
from dataclasses import asdict
from typing import Callable

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import EventRow
from ...errors import EventSequenceConflictError
from ...events import payloads as _payloads_module
from ...events.envelope import EventEnvelope
from ...events.store import EventPage


class SqlAlchemyEventStore:
    def __init__(self, *, session_factory: "Callable[[], AsyncSession]") -> None:
        self._session_factory = session_factory

    async def append(self, event: EventEnvelope, *, expected_sequence: "int | None" = None) -> EventEnvelope:
        async with self._session_factory() as session:
            async with session.begin():
                if expected_sequence is not None:
                    existing = await session.execute(
                        select(EventRow).where(EventRow.run_id == event.run_id, EventRow.sequence == event.sequence)
                    )
                    if existing.scalar_one_or_none() is not None:
                        raise EventSequenceConflictError(
                            f"event already exists at sequence {event.sequence} for run {event.run_id}"
                        )
                row = EventRow(
                    event_id=event.event_id, run_id=event.run_id, sequence=event.sequence,
                    occurred_at=event.occurred_at, root_run_id=event.root_run_id, parent_run_id=event.parent_run_id,
                    session_id=event.session_id, runnable_id=event.runnable_id,
                    payload_type=type(event.payload).__name__, payload_json=json.dumps(asdict(event.payload)),
                )
                session.add(row)
                # This flush is unconditional (not nested inside the
                # expected_sequence is not None branch above) so that the
                # unique (run_id, sequence) DB constraint's IntegrityError is
                # converted to EventSequenceConflictError for every duplicate
                # append attempt, regardless of whether the caller supplied
                # expected_sequence.
                try:
                    await session.flush()
                except IntegrityError as exc:
                    raise EventSequenceConflictError(
                        f"event already exists at sequence {event.sequence} for run {event.run_id}"
                    ) from exc
        return event

    def _row_to_envelope(self, row: EventRow) -> EventEnvelope:
        payload_cls = getattr(_payloads_module, row.payload_type)
        payload = payload_cls(**json.loads(row.payload_json))
        return EventEnvelope(
            event_id=row.event_id, sequence=row.sequence, occurred_at=row.occurred_at,
            run_id=row.run_id, root_run_id=row.root_run_id, parent_run_id=row.parent_run_id,
            session_id=row.session_id, runnable_id=row.runnable_id, payload=payload,
        )

    async def list(self, run_id: str, *, after_sequence: int = 0, limit: int = 100) -> EventPage:
        async with self._session_factory() as session:
            result = await session.execute(
                select(EventRow)
                .where(EventRow.run_id == run_id, EventRow.sequence > after_sequence)
                .order_by(EventRow.sequence.asc())
                .limit(limit)
            )
            items = tuple(self._row_to_envelope(row) for row in result.scalars())
            return EventPage(items=items, cursor=None)
