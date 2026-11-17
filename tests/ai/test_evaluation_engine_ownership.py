#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Evaluation readers retain borrowed engines and retry owned cleanup."""

import asyncio
from contextlib import asynccontextmanager

import pytest

from linktools.ai.errors import AIError
from linktools.ai.evaluation.engine import Evaluations
from linktools.ai.history import History
from linktools.ai.persistence import Persistence, PersistenceStorage
from linktools.ai.task.engine import TaskEngine


@pytest.mark.asyncio
async def test_evaluation_borrows_the_definition_view_until_reader_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    async with Persistence.open("evaluation-borrow", storage=PersistenceStorage.in_memory()) as state:
        engine = TaskEngine(persistence=state)
        view = engine.with_definitions()
        evaluations = await Evaluations.create(view)
        close_history = evaluations._history.close

        async def fail_close() -> None:
            raise ValueError("history cleanup failed")

        monkeypatch.setattr(evaluations._history, "close", fail_close)
        with pytest.raises(ValueError, match="history cleanup failed"):
            await evaluations.close()
        for owner in (view, engine):
            with pytest.raises(AIError):
                await owner.close()
        monkeypatch.setattr(evaluations._history, "close", close_history)
        await evaluations.close()
        await view.close()
        await engine.close()
        assert state.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ("history", "coordinator"))
async def test_evaluation_initialization_failure_releases_engine_borrow(
    monkeypatch: pytest.MonkeyPatch, stage: str,
) -> None:
    async with Persistence.open("evaluation-init", storage=PersistenceStorage.in_memory()) as state:
        engine = TaskEngine(persistence=state)
        if stage == "history":
            @asynccontextmanager
            async def fail_history(**kwargs):
                raise ValueError("initialization failed")
                yield

            monkeypatch.setattr(History, "open", fail_history)
            coordinator = Evaluations
        else:
            class FailingEvaluations(Evaluations):
                def __init__(self, *args, **kwargs) -> None:
                    raise ValueError("initialization failed")

            coordinator = FailingEvaluations
        with pytest.raises(ValueError, match="initialization failed"):
            await coordinator.create(engine)
        await engine.close()
        assert state.ready


@pytest.mark.asyncio
@pytest.mark.parametrize("cancel_body", (False, True))
async def test_standalone_evaluation_retries_owned_engine_close_after_context_failure(
    monkeypatch: pytest.MonkeyPatch, cancel_body: bool,
) -> None:
    context = Evaluations.open(storage=PersistenceStorage.in_memory(), namespace="evaluation-owned")
    evaluations = await context.__aenter__()
    engine = evaluations._engine_owner
    close_engine = engine.close
    failed = False

    async def close_once() -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise ValueError("engine cleanup failed")
        await close_engine()

    monkeypatch.setattr(engine, "close", close_once)
    if cancel_body:
        primary = asyncio.CancelledError("body cancelled")
        with pytest.raises(BaseExceptionGroup) as errors:
            await context.__aexit__(type(primary), primary, primary.__traceback__)
        assert errors.value.exceptions[0] is primary
        assert isinstance(errors.value.exceptions[1], ValueError)
    else:
        with pytest.raises(ValueError, match="engine cleanup failed"):
            await context.__aexit__(None, None, None)
    assert engine.persistence.ready
    assert not evaluations._closed
    await evaluations.close()
    assert not engine.persistence.ready
    assert evaluations._closed
