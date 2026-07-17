#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SqlAlchemyEvalStore contract over an in-memory SQLite backend."""

import asyncio
from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.evaluation.models import (
    EvalResult,
    EvalRun,
    EvalRunStatus,
    EvalTarget,
)
from linktools.ai.evaluation.store import (
    EvalResultConflictError,
    EvalRunNotFoundError,
)
from linktools.ai.storage.sqlalchemy.evaluation import SqlAlchemyEvalStore
from linktools.ai.storage.sqlalchemy.models import Base


@pytest.fixture
def eval_store(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path}/eval.db")
    asyncio.run(_create(engine))
    factory = async_sessionmaker(engine, expire_on_commit=False)
    return SqlAlchemyEvalStore(session_factory=factory)


async def _create(engine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def _run(coro):
    return asyncio.run(coro)


def test_sql_eval_store_round_trip(eval_store) -> None:
    now = datetime.now(timezone.utc)

    async def run() -> None:
        er = EvalRun(
            id="er1",
            suite_id="s1",
            target=EvalTarget(kind="agent", id="a1"),
            status=EvalRunStatus.RUNNING,
            baseline_target=None,
            created_at=now,
            started_at=now,
        )
        await eval_store.create_run(er)
        fetched = await eval_store.get_run("er1")
        assert fetched is not None
        assert fetched.status == EvalRunStatus.RUNNING
        assert fetched.target.id == "a1"
        await eval_store.transition_run(
            "er1", status=EvalRunStatus.SUCCEEDED, finished_at=now
        )
        assert (await eval_store.get_run("er1")).status == EvalRunStatus.SUCCEEDED
        result = EvalResult(
            id="er1-c1", eval_run_id="er1", case_id="c1", scores={"q": 0.9}
        )
        await eval_store.append_result(result)
        with pytest.raises(EvalResultConflictError):
            await eval_store.append_result(result)
        stored = await eval_store.list_results("er1")
        assert len(stored) == 1
        assert stored[0].scores == {"q": 0.9}
        with pytest.raises(EvalRunNotFoundError):
            await eval_store.transition_run("missing", status=EvalRunStatus.SUCCEEDED)

    _run(run())
