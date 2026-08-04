import pytest

from linktools.ai.acp.agent import LinktoolsAcpAgent, STANDARD_AGENT_METHODS
from linktools.ai.acp.capabilities import AcpMode
from linktools.ai.governance.identity import trusted_local_principal


def _agent(tmp_path):
    return LinktoolsAcpAgent(
        runtime=object(),
        state_root=str(tmp_path),
        project_root=str(tmp_path),
        principal=trusted_local_principal(),
        spec_resolver=lambda mode: None,
        modes=(AcpMode("default", "Default"),),
    )


def test_stable_agent_methods_have_one_handler_each(tmp_path) -> None:
    agent = _agent(tmp_path)
    registry = agent.handler_registry()

    assert set(registry) == set(STANDARD_AGENT_METHODS)
    assert len({id(handler) for handler in registry.values()}) == len(registry)


@pytest.mark.asyncio
async def test_unexpected_handler_error_is_sanitized(tmp_path) -> None:
    agent = _agent(tmp_path)
    agent._initialized = True

    async def broken(*args, **kwargs):
        raise RuntimeError("do not expose this")

    agent.sessions.list = broken
    with pytest.raises(Exception) as raised:
        await agent.list_sessions()
    assert "do not expose this" not in str(raised.value)
