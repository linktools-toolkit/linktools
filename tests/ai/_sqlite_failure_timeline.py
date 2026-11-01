#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Bounded, parameter-free SQL lifecycle evidence for a shared-source test."""

import asyncio
import gc
from collections import deque
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from functools import wraps
from time import monotonic, process_time, thread_time

import pytest
from sqlalchemy import event
from sqlalchemy.dialects.sqlite.aiosqlite import AsyncAdapt_aiosqlite_connection, SQLiteDialect_aiosqlite
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from linktools.ai.storage import SqlStorageContext


class _SqliteTimeline:
    def __init__(self, capacity: int = 16384) -> None:
        self.events = deque(maxlen=capacity)
        self.dropped = 0
        self.started = monotonic()
        self.origin = ContextVar("sqlite_timeline_origin", default=None)
        self.listeners = []
        self.loop_handle = None
        self.gc_callback = None

    def emit(self, kind: str, **fields: object) -> None:
        try:
            task = asyncio.current_task()
        except RuntimeError:
            task = None
        if len(self.events) == self.events.maxlen:
            self.dropped += 1
        self.events.append((round(monotonic() - self.started, 6), kind,
                            id(task) if task is not None else None, self.origin.get(), fields))

    def format(self) -> str:
        return "SQLite lifecycle timeline (bounded; scheduling may be perturbed; dropped=" + str(self.dropped) + ")\n" + "\n".join(
            str(row) for row in self.events
        )

    def listen(self, target: object, name: str, listener: Callable) -> None:
        event.listen(target, name, listener)
        self.listeners.append((target, name, listener))

    def _loop_tick(self, expected: float) -> None:
        now = monotonic()
        lag = now - expected
        if lag >= 0.02:
            self.emit("loop.lag", seconds=round(lag, 6),
                      process_cpu=process_time(), thread_cpu=thread_time())
        self.loop_handle = asyncio.get_running_loop().call_later(0.02, self._loop_tick, now + 0.02)

    def install(self, patches: pytest.MonkeyPatch) -> None:
        def connection_id(connection):
            return id(connection.driver_connection)

        def before(connection, cursor, statement, parameters, context, executemany):
            if connection.dialect.name == "sqlite":
                words = statement.split(None, 1)
                verb = words[0].upper() if words else ""
                context._sqlite_failure_connection_id = connection_id(connection.connection)
                self.emit("sql.begin", connection=context._sqlite_failure_connection_id,
                          verb=verb if verb in {"SELECT", "INSERT", "UPDATE", "DELETE", "BEGIN", "COMMIT",
                                               "ROLLBACK", "PRAGMA", "CREATE", "DROP", "SAVEPOINT", "RELEASE"}
                          else "OTHER")

        def after(connection, cursor, statement, parameters, context, executemany):
            if connection.dialect.name == "sqlite":
                self.emit("sql.end", connection=connection_id(connection.connection))

        def failed(context):
            if context.dialect.name == "sqlite":
                self.emit("sql.error", connection=getattr(context.execution_context,
                                                          "_sqlite_failure_connection_id", None),
                          error=type(context.original_exception).__name__,
                          code=getattr(context.original_exception, "sqlite_errorcode", None))

        def begin(session, transaction, connection):
            if connection.dialect.name == "sqlite":
                self.emit("session.begin", session=id(session), connection=connection_id(connection.connection))

        def checkout(connection, record, proxy):
            if isinstance(connection, AsyncAdapt_aiosqlite_connection):
                self.emit("pool.out", connection=connection_id(connection))

        def checkin(connection, record):
            if isinstance(connection, AsyncAdapt_aiosqlite_connection):
                self.emit("pool.in", connection=connection_id(connection))

        for name, listener in (("before_cursor_execute", before), ("after_cursor_execute", after),
                               ("handle_error", failed), ("checkout", checkout), ("checkin", checkin)):
            self.listen(Engine, name, listener)
        self.listen(Session, "after_begin", begin)

        def wrap_driver(name):
            original = getattr(SQLiteDialect_aiosqlite, name)

            @wraps(original)
            def call(dialect, connection):
                identity = connection_id(connection)
                self.emit(name + ".begin", connection=identity)
                try:
                    result = original(dialect, connection)
                except BaseException as error:
                    self.emit(name + ".error", connection=identity, error=type(error).__name__,
                              code=getattr(error, "sqlite_errorcode", None))
                    raise
                self.emit(name + ".end", connection=identity)
                return result

            patches.setattr(SQLiteDialect_aiosqlite, name, call)

        wrap_driver("do_commit")
        wrap_driver("do_rollback")
        original_statement = SqlStorageContext.run_statement

        @wraps(original_statement)
        async def statement(context, operation):
            token = self.origin.set(id(asyncio.current_task()))
            try:
                return await original_statement(context, operation)
            finally:
                self.origin.reset(token)

        patches.setattr(SqlStorageContext, "run_statement", statement)
        original_mutation = SqlStorageContext.run_mutation

        @wraps(original_mutation)
        async def mutation(context, callback, **kwargs):
            self.emit("mutation.begin", context=id(context), engine=id(context.engine), domain=kwargs.get("domain"))
            try:
                result = await original_mutation(context, callback, **kwargs)
            except BaseException as error:
                native = getattr(error, "orig", error)
                self.emit("mutation.error", context=id(context), engine=id(context.engine),
                          domain=kwargs.get("domain"), error=type(error).__name__,
                          code=getattr(native, "sqlite_errorcode", None),
                          invalidated=getattr(error, "connection_invalidated", None))
                raise
            self.emit("mutation.end", context=id(context), engine=id(context.engine), domain=kwargs.get("domain"))
            return result

        patches.setattr(SqlStorageContext, "run_mutation", mutation)
        original_close = AsyncSession.close

        @wraps(original_close)
        async def close(session):
            selected = session.bind is not None and session.bind.dialect.name == "sqlite"
            if selected:
                self.emit("session.close.begin", session=id(session.sync_session))
            try:
                result = await original_close(session)
            except BaseException as error:
                if selected:
                    self.emit("session.close.error", session=id(session.sync_session), error=type(error).__name__)
                raise
            if selected:
                self.emit("session.close.end", session=id(session.sync_session))
            return result

        patches.setattr(AsyncSession, "close", close)

        def gc_event(phase, info):
            self.emit("gc." + phase, generation=info["generation"], collected=info["collected"],
                      process_cpu=process_time(), thread_cpu=thread_time())

        self.gc_callback = gc_event
        gc.callbacks.append(gc_event)
        self.loop_handle = asyncio.get_running_loop().call_later(0.02, self._loop_tick, monotonic() + 0.02)

    def remove(self) -> None:
        if self.loop_handle is not None:
            self.loop_handle.cancel()
        if self.gc_callback is not None:
            gc.callbacks.remove(self.gc_callback)
        for target, name, listener in reversed(self.listeners):
            event.remove(target, name, listener)


_active_timeline: ContextVar[_SqliteTimeline | None] = ContextVar("sqlite_failure_timeline", default=None)


def sqlite_failure_phase(name: str, *, producer: int | None = None) -> None:
    timeline = _active_timeline.get()
    if timeline is not None:
        timeline.emit("phase." + name, producer=producer)


def sqlite_failure_evidence(test: Callable[..., Awaitable[object]]) -> Callable[..., Awaitable[object]]:
    """Attach bounded lifecycle evidence only when the decorated test fails."""
    @wraps(test)
    async def run(*args, **kwargs):
        timeline = _SqliteTimeline()
        token = _active_timeline.set(timeline)
        with pytest.MonkeyPatch.context() as patches:
            try:
                try:
                    timeline.install(patches)
                    return await test(*args, **kwargs)
                finally:
                    timeline.remove()
                    _active_timeline.reset(token)
            except BaseException as error:
                error.add_note(timeline.format())
                raise

    return run


def sqlite_failure_timeline(test: Callable[..., Awaitable[object]]) -> Callable[..., Awaitable[object]]:
    """Select only the two-producer SQLite live-observation case."""
    selected = sqlite_failure_evidence(test)

    @wraps(test)
    async def run(*args, **kwargs):
        if kwargs.get("backend") != "sqlite" or kwargs.get("producer_count") != 2:
            return await test(*args, **kwargs)
        return await selected(*args, **kwargs)

    return run
