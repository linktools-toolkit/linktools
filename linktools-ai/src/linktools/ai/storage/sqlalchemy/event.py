#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyEventStore: DB-backed EventStore. The unique (run_id, sequence)
constraint on ai_events is the backstop for sequence uniqueness; the store
additionally reserves the next sequence itself (review doc §8.1/§8.4) by
reading MAX(sequence)+1 for the stream inside the same transaction that
inserts the row. On the rare race where two concurrent transactions both
computed the same next_seq, the unique constraint's IntegrityError is caught
and the whole append retried (re-reading MAX under a fresh transaction)."""

import json
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Callable

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from .models import EventRow
from ...errors import EventSequenceConflictError
from ...events import payloads as _payloads_module
from ...events.envelope import EventEnvelope
from ...events.payloads import EventPayload
from ...events.store import EventPage


def _as_utc(dt: "datetime | None") -> "datetime | None":
    # SQLite (aiosqlite) round-trips datetimes as naive values regardless of the
    # tzinfo they were stored with, so reattach UTC on read to match the
    # timezone-aware datetimes EventEnvelope is constructed with everywhere else.
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


class SqlAlchemyEventStore:
    def __init__(self, *, session_factory: "Callable[[], AsyncSession]") -> None:
        self._session_factory = session_factory

    async def append(
        self,
        *,
        stream_id: str,
        run_id: str,
        root_run_id: str,
        parent_run_id: "str | None",
        session_id: str,
        runnable_id: str,
        payload: EventPayload,
    ) -> EventEnvelope:
        # Reserve the next sequence inside the inserting transaction: read
        # MAX(sequence) for the stream, add 1, insert. Retry the whole
        # transaction when the unique (run_id, sequence) constraint fires
        # (two concurrent reservations computed the same next_seq -- the loser
        # rolls back and re-reads MAX, which now reflects the winner's row).
        last_exc: "BaseException | None" = None
        for _ in range(8):
            try:
                async with self._session_factory() as session:
                    async with session.begin():
                        result = await session.execute(
                            select(func.max(EventRow.sequence)).where(EventRow.run_id == stream_id)
                        )
                        current = result.scalar()
                        next_seq = (current or 0) + 1
                        event_id = str(uuid.uuid4())
                        occurred_at = datetime.now(timezone.utc)
                        row = EventRow(
                            event_id=event_id, run_id=run_id, sequence=next_seq,
                            occurred_at=occurred_at, root_run_id=root_run_id, parent_run_id=parent_run_id,
                            session_id=session_id, runnable_id=runnable_id,
                            payload_type=type(payload).__name__,
                            payload_json=json.dumps(asdict(payload)),
                        )
                        session.add(row)
                        await session.flush()
                        return EventEnvelope(
                            event_id=event_id, sequence=next_seq, occurred_at=occurred_at,
                            run_id=run_id, root_run_id=root_run_id, parent_run_id=parent_run_id,
                            session_id=session_id, runnable_id=runnable_id, payload=payload,
                        )
            except IntegrityError as exc:
                # Unique (run_id, sequence) collision -- a concurrent append
                # reserved the same sequence first. Retry to re-read MAX.
                last_exc = exc
                continue
        raise EventSequenceConflictError(
            f"could not reserve a unique event sequence for stream {stream_id!r} "
            f"after repeated conflicts"
        ) from last_exc

    def _row_to_envelope(self, row: EventRow) -> EventEnvelope:
        payload_cls = getattr(_payloads_module, row.payload_type)
        payload = payload_cls(**json.loads(row.payload_json))
        return EventEnvelope(
            event_id=row.event_id, sequence=row.sequence, occurred_at=_as_utc(row.occurred_at),
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
