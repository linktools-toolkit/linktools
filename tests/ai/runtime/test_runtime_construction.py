import pytest

from linktools.ai.agent.spec import AgentSpec, PromptSpec
from linktools.ai.errors import PrincipalAccessDeniedError
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


@pytest.mark.asyncio
async def test_anonymous_runs_get_isolated_sessions(tmp_path):
    # Omitting session_id must NOT collapse every run onto a shared "session".
    runtime = build_runtime(storage=LocalDirectoryStorage(tmp_path), model_resolver=make_router())
    assert (await runtime.run(spec(), "hello")) is not None
    assert (await runtime.run(spec(), "again")) is not None
    assert await runtime.execution.store.get_session("session") is None
    await runtime.aclose()


@pytest.mark.asyncio
async def test_session_reuse_rejects_principal_mismatch(tmp_path):
    runtime = build_runtime(storage=LocalDirectoryStorage(tmp_path), model_resolver=make_router())
    await runtime.run(spec(), "hi", session_id="s", tenant_id="t1")
    with pytest.raises(PrincipalAccessDeniedError):
        await runtime.run(spec(), "hi", session_id="s", tenant_id="t2")
    await runtime.aclose()
