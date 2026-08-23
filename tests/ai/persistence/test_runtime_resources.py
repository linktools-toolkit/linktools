"""RuntimeState resource ownership and restart checks."""

import asyncio

import pytest
from linktools.ai.migrate import provision_database
from linktools.ai.runtime.state import (
    RuntimeDomain,
    RuntimeState,
    RuntimeStatePlan,
    RuntimeStateRoute,
)
from sqlalchemy.ext.asyncio import create_async_engine


@pytest.mark.asyncio
async def test_sqlite_runtime_state_owns_and_reopens_database(tmp_path) -> None:
    path = tmp_path / "runtime.db"
    engine = create_async_engine(f"sqlite+aiosqlite:///{path}")
    await provision_database(engine)
    state = RuntimeState.sql(engine)
    await state.initialize(namespace="runtime", tenant_id="tenant")
    assert state.ready is True
    await state.close()

    reopened = RuntimeState.sql(engine)
    await reopened.initialize(namespace="runtime", tenant_id="tenant")
    assert reopened.ready is True
    await reopened.close()
    await engine.dispose()


@pytest.mark.asyncio
async def test_transient_runtime_state_closes_repository_and_object_domains() -> None:
    plan = RuntimeStatePlan(**{domain.value: RuntimeStateRoute.transient() for domain in RuntimeDomain})
    state = RuntimeState.from_plan(plan)
    await state.initialize(namespace="transient", tenant_id="tenant")
    await state.close()
    await state.close()
    assert state.ready is False


@pytest.mark.asyncio
async def test_runtime_state_close_cursor_retries_failed_action_and_survives_cancellation() -> None:
    state = RuntimeState.in_memory()
    await state.initialize(namespace="close-cursor", tenant_id="tenant")
    calls: list[str] = []
    failed = True

    async def first() -> None:
        nonlocal failed
        calls.append("first")
        if failed:
            failed = False
            raise RuntimeError("retry")

    async def second() -> None:
        calls.append("second")

    state._close_actions = (first, second)
    with pytest.raises(RuntimeError, match="retry"):
        await state.close()
    assert calls == ["first"]
    await state.close()
    assert calls == ["first", "first", "second"]

    cancelled = RuntimeState.in_memory()
    await cancelled.initialize(namespace="close-cancellation", tenant_id="tenant")
    started = asyncio.Event()
    release = asyncio.Event()
    cancellation_calls: list[str] = []

    async def blocking() -> None:
        cancellation_calls.append("started")
        started.set()
        await release.wait()
        cancellation_calls.append("done")

    cancelled._close_actions = (blocking,)
    close_task = asyncio.create_task(cancelled.close())
    await started.wait()
    close_task.cancel()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await close_task
    assert cancellation_calls == ["started", "done"]
    await cancelled.close()


def test_mixed_runtime_plan_has_explicit_routes(tmp_path) -> None:
    plan = RuntimeStatePlan(
        conversation=RuntimeStateRoute.filesystem(tmp_path / "conversation"),
        execution=RuntimeStateRoute.transient(),
        memory=RuntimeStateRoute.memory(),
    )
    assert plan.route(RuntimeDomain.CONVERSATION).retention.value == "durable"
    assert plan.route(RuntimeDomain.EXECUTION).retention.value == "transient"
    assert plan.durable_domains == frozenset({RuntimeDomain.CONVERSATION})
