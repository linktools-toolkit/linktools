from datetime import datetime, timezone

import pytest
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from linktools.ai.execution.models import RunDefinitionSnapshot, RunKind, RunSnapshot, RunStatus, RunUsage
from linktools.ai.execution.persistence.sqlalchemy import SnapshotRow, SqlAlchemyExecutionStore


@pytest.mark.asyncio
async def test_sqlalchemy_execution_pages_in_database(tmp_path):
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'execution.db'}")
    factory = async_sessionmaker(engine, expire_on_commit=False)
    store = SqlAlchemyExecutionStore(factory)
    await store.initialize_storage(engine)
    await store.create_session(session_id="s", user_id="u", tenant_id="t")
    run = await store.start_run(run_id="r", session_id="s", kind=RunKind.USER_TURN, definition=RunDefinitionSnapshot("a"), user_prompt="p")
    claimed = await store.claim_run("r", owner="w", expected_fence=run.execution_fence)
    snapshot = RunSnapshot("run-snapshot.v1", "r", 1, ({"role": "user", "content": "p"},), "done", RunStatus.COMPLETED, RunUsage(), 0, datetime.now(timezone.utc))
    async with factory() as session:
        async with session.begin():
            session.add(SnapshotRow(run_id="r", revision=1, payload={"resume_messages": list(snapshot.resume_messages), "final_output": "done", "status": "completed", "usage": {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}, "trace_end_sequence": 0}, created_at=snapshot.created_at))
    assert (await store.list_session_turns("s", limit=1)).items[0].run_id == "r"
    assert (await store.get_snapshot("r")).final_output == "done"
    await engine.dispose()
