#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Attribute actual SQLAlchemy cursor activity without logging values."""

import asyncio
import logging

import pytest
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import create_async_engine

from linktools.ai.storage import current_sql_observation, sql_observation_scope
from linktools.ai.storage._database import create_sql_storage_context, create_sqlite_storage_context
from linktools.ai.storage._sql_observation import _logger

pytestmark = pytest.mark.asyncio


async def test_owned_engine_attributes_real_statements_without_sql_or_parameters(tmp_path, caplog):
    state = create_sqlite_storage_context(tmp_path / "debug.db")
    await state.initialize()
    await state.initialize()
    caplog.set_level(logging.DEBUG, logger=_logger.name)
    try:
        async with state.engine.connect() as connection:
            with sql_observation_scope("qa.send"):
                assert current_sql_observation() == ("qa.send",)
                await connection.exec_driver_sql("SELECT 'secret_literal'")
                with sql_observation_scope("runtime.write"):
                    await connection.exec_driver_sql("SELECT ?", ("secret_parameter",))
                    with pytest.raises(DBAPIError):
                        await connection.exec_driver_sql("SELECT secret_failed_identifier")
            assert current_sql_observation() == ()
            await connection.exec_driver_sql("SELECT 'outside_scope'")
    finally:
        await state.close()

    logs = [record.getMessage() for record in caplog.records if record.name == _logger.name]
    cursors = [line for line in logs if line.startswith("SQL cursor")]
    assert len(cursors) == 3
    assert all("verb=SELECT " in line and "elapsed_ms=" in line for line in cursors)
    assert sum("operation=qa.send/runtime.write" in line for line in cursors) == 2
    assert sum("error=OperationalError" in line for line in cursors) == 1
    assert any("operation=qa.send statements=3 by_verb={'SELECT': 3}" in line for line in logs)
    assert any("operation=qa.send/runtime.write statements=2" in line and "failures=1" in line for line in logs)
    assert all("secret" not in line and "outside_scope" not in line for line in logs)


async def test_borrowed_engine_uses_passive_context_without_duplicate_logs(tmp_path, caplog):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'borrowed.db'}")
    state = create_sql_storage_context(engine)
    await state.initialize()
    caplog.set_level(logging.DEBUG, logger=_logger.name)
    try:
        with sql_observation_scope("borrower.inspect"):
            assert current_sql_observation() == ("borrower.inspect",)
            async with state.engine.connect() as connection:
                await connection.exec_driver_sql("SELECT 1")
    finally:
        await state.close()
        await engine.dispose()
    assert not [record for record in caplog.records if record.name == _logger.name]


async def test_nested_labels_are_task_local_and_closed_parents_do_not_leak():
    child_started = asyncio.Event()
    resume = asyncio.Event()

    async def child():
        assert current_sql_observation() == ("caller",)
        child_started.set()
        await resume.wait()
        assert current_sql_observation() == ()
        with sql_observation_scope("worker"):
            await asyncio.sleep(0)
            assert current_sql_observation() == ("worker",)

    with sql_observation_scope("caller"):
        task = asyncio.create_task(child())
        await child_started.wait()
        assert current_sql_observation() == ("caller",)
    with sql_observation_scope("unrelated"):
        resume.set()
        await task
        assert current_sql_observation() == ("unrelated",)
    assert current_sql_observation() == ()


async def test_labels_reject_free_form_text_and_control_characters():
    with pytest.raises(ValueError):
        with sql_observation_scope("bad\nlabel"):
            raise AssertionError("invalid label was accepted")


async def test_detached_phase_does_not_count_into_live_parent(tmp_path, caplog):
    from linktools.ai.storage import sql_observation_phase

    state = create_sqlite_storage_context(tmp_path / "detached.db")
    await state.initialize()
    caplog.set_level(logging.DEBUG, logger=_logger.name)

    @sql_observation_phase("runtime.worker", inherit=False)
    async def worker(connection):
        assert current_sql_observation() == ("runtime.worker",)
        await connection.exec_driver_sql("SELECT 1")

    try:
        with sql_observation_scope("runtime.admission"):
            async with state.engine.connect() as connection:
                await asyncio.create_task(worker(connection))
                assert current_sql_observation() == ("runtime.admission",)
                await connection.exec_driver_sql("SELECT 2")
    finally:
        await state.close()

    logs = [record.getMessage() for record in caplog.records if record.name == _logger.name]
    assert any("scope operation=runtime.worker statements=1" in line for line in logs)
    assert any("scope operation=runtime.admission statements=1" in line for line in logs)
    assert all("runtime.admission/runtime.worker" not in line for line in logs)


async def test_debug_disabled_preserves_labels_without_starting_sql_timing(tmp_path, caplog, monkeypatch):
    from linktools.ai.storage import _sql_observation

    state = create_sqlite_storage_context(tmp_path / "disabled.db")
    await state.initialize()
    caplog.set_level(logging.INFO, logger=_logger.name)

    def unexpected_timing():
        raise AssertionError("disabled DEBUG must not start cursor timing")

    monkeypatch.setattr(_sql_observation, "perf_counter_ns", unexpected_timing)
    try:
        with sql_observation_scope("timing.disabled"):
            assert current_sql_observation() == ("timing.disabled",)
            async with state.engine.connect() as connection:
                await connection.exec_driver_sql("SELECT 1")
    finally:
        await state.close()


async def test_runtime_warm_turn_has_specific_source_phases(tmp_path, caplog):
    from linktools.ai.storage import InMemoryCacheBackend
    from .test_pending_session_projection import _runtimes

    cache = InMemoryCacheBackend(max_bytes=2**24)
    async with _runtimes(tmp_path, cache=cache, source_authority=True) as (runtimes, entered, finish, block):
        runtime, _state = runtimes[0]
        reader, _reader_state = runtimes[1]
        session = await runtime.agents.get().create_session("debug-session")
        warm = await session.start("never-log-this-prompt-first")
        producer = runtime._execution_service.runtime_backend()
        warm_owner = producer._turn_owners[warm.execution_id]
        await warm.wait()
        await warm_owner.closed.wait()
        caplog.set_level(logging.DEBUG, logger=_logger.name)

        block()
        turn = await runtime.sessions.start(
            "debug-session", "never-log-this-prompt", principal=runtime.default_principal,
        )
        holder = producer._turn_owners[turn.execution_id]
        await asyncio.wait_for(entered.wait(), 5)
        busy = await reader.history.inspect_session("debug-session", principal=reader.default_principal)
        assert busy.active_execution_id == turn.execution_id
        finish.set()
        await turn.wait(timeout_seconds=5)
        await holder.closed.wait()
        await reader.history.session_detail("debug-session", principal=reader.default_principal, limit=1)

    cursor_logs = [record.getMessage() for record in caplog.records
                   if record.name == _logger.name and record.getMessage().startswith("SQL cursor")]
    assert all("never-log-this-prompt" not in line and "debug-session" not in line for line in cursor_logs)
    for label in (
        "runtime.session.start", "storage.turn.claim", "runtime.execution.worker",
        "coordination.session.verify_cached_fence", "persistence.terminal.receipt_lookup",
        "persistence.terminal.checkpoint_prepare", "storage.turn.release", "history.session.detail",
    ):
        assert any(label in line for line in cursor_logs), label
    assert all("runtime.session.start/runtime.execution.worker" not in line for line in cursor_logs)
