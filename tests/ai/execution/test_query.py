import pytest

from linktools.ai.execution.models import RunDefinitionSnapshot, RunKind
from linktools.ai.execution.persistence.local import LocalExecutionStore
from linktools.ai.execution.query import ExecutionQueryService
from linktools.ai.identity.principal import ActorRef, PrincipalContext, ScopeSet


def principal(user: str) -> PrincipalContext:
    return PrincipalContext("tenant", user, ActorRef("user", user), ScopeSet.allow_all())


@pytest.mark.asyncio
async def test_query_service_authorizes_before_returning_turns(tmp_path):
    store = LocalExecutionStore(tmp_path)
    await store.create_session(session_id="s", user_id="u", tenant_id="tenant")
    await store.start_run(run_id="r", session_id="s", kind=RunKind.USER_TURN, definition=RunDefinitionSnapshot("agent"), user_prompt="secret")
    query = ExecutionQueryService(store)
    with pytest.raises(Exception):
        await query.list_session_turns(session_id="s", principal=principal("other"))
    page = await query.list_session_turns(session_id="s", principal=principal("u"))
    assert page.items[0].user_prompt == "secret"
