import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.model.policy import ModelPolicy
from linktools.ai.runtime import LocalDirectoryStorage, build_runtime
from tests.ai.fakes.model import make_router


def spec() -> AgentSpec:
    return AgentSpec("agent", "agent", ModelPolicy(primary="test-model"), PromptSpec("answer"))


@pytest.mark.asyncio
async def test_build_runtime_uses_execution_store_for_local_storage(tmp_path):
    runtime = build_runtime(storage=LocalDirectoryStorage(tmp_path), model_resolver=make_router())
    assert (await runtime.run(spec(), "hello", session_id="s", tenant_id="t")) is not None
    assert (await runtime.run(spec(), "again", session_id="s", tenant_id="t")) is not None
    session = await runtime.execution.store.get_session("s")
    assert session.latest_completed_run_id is not None
    await runtime.aclose()
