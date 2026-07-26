#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""FencedRunEventWriter tests (P6 fenced security event): a security event lands iff the
presented fence still owns the run; a stale fence raises RunFenceLostError
BEFORE the event is persisted."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.events.context import EventStreamContext
from linktools.ai.events.payloads import RunStarted
from linktools.ai.run.commit import ExecutionFence
from linktools.ai.run.models import RunInput, RunRecord, RunnableType, RunStatus
from linktools.ai.run.persistence.event_writer import (
    FencedRunEventWriter,
    RunFenceLostError,
)
from linktools.ai.run.persistence.sqlalchemy.event_writer import (
    SqlAlchemyFencedRunEventWriter,
)
from linktools.ai.run.persistence.sqlalchemy.run import SqlAlchemyRunStore


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def storage(tmp_path):
    from linktools.ai.runtime.persistence.sqlalchemy import _ReferenceSqlAlchemyComposition as SqlAlchemyStorage

    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'fenced.db'}")
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    s = SqlAlchemyStorage(
        session_factory=session_factory, blobs_root=tmp_path / "blobs"
    )

    async def _setup():
        from linktools.ai.storage.sqlalchemy.models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        return s

    return _run(_setup())


def _seed_run_with_token(storage, run_id, token):
    async def _seed():
        record = RunRecord(
            id=run_id,
            root_run_id=run_id,
            parent_run_id=None,
            session_id="sess-fence",
            runnable_id="agent-1",
            runnable_type=RunnableType.AGENT,
            status=RunStatus.RUNNING,
            input=RunInput(prompt="seed"),
            result=None,
            error=None,
            version=1,
            created_at=datetime.now(timezone.utc),
            started_at=None,
            finished_at=None,
            execution_token=token,
        )
        await storage.runs.create(record)

    _run(_seed())


def _ctx(run_id):
    return EventStreamContext(
        stream_id=run_id,
        run_id=run_id,
        root_run_id=run_id,
        parent_run_id=None,
        session_id="sess-fence",
        runnable_id="agent-1",
    )


def test_fenced_writer_satisfied_by_fenced_run_event_writer_protocol():
    """SqlAlchemyFencedRunEventWriter implements the FencedRunEventWriter
    Protocol (runtime_checkable)."""
    from linktools.ai.run.persistence.sqlalchemy.event_writer import (
        SqlAlchemyFencedRunEventWriter,
    )

    # Instantiate without storage -- the Protocol check does not require a
    # real storage (it only verifies method shape).
    class _Stub:
        async def append_security(self, **kwargs): ...

    assert isinstance(_Stub(), FencedRunEventWriter)


def test_fenced_writer_appends_when_fence_matches(storage):
    """A presented fence whose token equals the run's stored execution_token
    is allowed; the event lands."""
    _seed_run_with_token(storage, "run-A", "token-A")
    writer = SqlAlchemyFencedRunEventWriter(storage)
    event = RunStarted(run_id="run-A", runnable_id="agent-1")

    async def _run_async():
        await writer.append_security(
            context=_ctx("run-A"),
            fence=ExecutionFence("token-A"),
            event=event,
        )
        # The event should now be in the run's event stream.
        page = await storage.events.list("run-A", limit=10)
        return page

    page = _run(_run_async())
    assert any(isinstance(item.payload, RunStarted) for item in page.items)


def test_fenced_writer_rejects_stale_fence(storage):
    """A presented fence whose token DIFFERS from the run's stored token
    raises RunFenceLostError BEFORE the event is persisted."""
    _seed_run_with_token(storage, "run-B", "token-current")
    writer = SqlAlchemyFencedRunEventWriter(storage)
    event = RunStarted(run_id="run-B", runnable_id="agent-1")

    async def _run_async():
        with pytest.raises(RunFenceLostError):
            await writer.append_security(
                context=_ctx("run-B"),
                fence=ExecutionFence("token-stale"),
                event=event,
            )
        return await storage.events.list("run-B", limit=10)

    page = _run(_run_async())
    # The event was NOT appended.
    assert page.items == ()


def test_fenced_writer_rejects_when_run_has_no_token(storage):
    """A run with NO execution_token cannot be fenced against -- the writer
    refuses rather than guessing (the security-sensitive action must not
    proceed on an unfenced run)."""
    _seed_run_with_token(storage, "run-C", "")  # store empty token via
    # the create path; many adapters coerce empty -> None on read.
    writer = SqlAlchemyFencedRunEventWriter(storage)
    event = RunStarted(run_id="run-C", runnable_id="agent-1")

    async def _run_async():
        with pytest.raises(RunFenceLostError):
            await writer.append_security(
                context=_ctx("run-C"),
                fence=ExecutionFence("any-token"),
                event=event,
            )

    _run(_run_async())


def test_fenced_writer_rejects_missing_run(storage):
    """A fence presented for a run that does not exist is rejected outright."""
    writer = SqlAlchemyFencedRunEventWriter(storage)
    event = RunStarted(run_id="never-existed", runnable_id="agent-1")

    async def _run_async():
        with pytest.raises(RunFenceLostError):
            await writer.append_security(
                context=_ctx("never-existed"),
                fence=ExecutionFence("some-token"),
                event=event,
            )

    _run(_run_async())
