import pytest

from linktools.ai.execution.commands import StartExecution
from linktools.ai.execution.domain import RunDefinition, RunKind, RunnableType
from linktools.ai.execution.persistence.local import LocalExecutionBackend
from linktools.ai.execution.store import ExecutionStore
from linktools.ai.execution.query import ExecutionQueryService
from linktools.ai.governance.identity import ActorRef, PrincipalContext, ScopeSet


def principal(user: str) -> PrincipalContext:
    return PrincipalContext("tenant", user, ActorRef("user", user), ScopeSet.allow_all())


@pytest.mark.asyncio
async def test_query_service_authorizes_before_returning_turns(tmp_path):
    store = ExecutionStore(LocalExecutionBackend(tmp_path))
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = RunDefinition("agent", RunnableType.AGENT, "agent-spec.v1", {"id": "agent"}, "agent")
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "secret"))
    query = ExecutionQueryService(store)
    with pytest.raises(Exception):
        await query.list_session_turns(session_id="s", principal=principal("other"))
    page = await query.list_session_turns(session_id="s", principal=principal("u"))
    assert page.items[0].input == "secret"


@pytest.mark.asyncio
async def test_run_detail_effective_input_comes_from_run_record_input(tmp_path):
    store = ExecutionStore(LocalExecutionBackend(tmp_path))
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    definition = RunDefinition("agent", RunnableType.AGENT, "agent-spec.v1", {"id": "agent"}, "agent")
    await store.start_run(StartExecution("r", "s", RunKind.USER_TURN, definition, "secret"))
    query = ExecutionQueryService(store)
    detail = await query.get_run_detail(run_id="r", principal=principal("u"))
    assert detail.effective_input == "secret"
