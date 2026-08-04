import pytest
from types import SimpleNamespace
import acp.schema as schema

from linktools.ai.acp.agent import LinktoolsAcpAgent, STANDARD_AGENT_METHODS
from linktools.ai.acp.capabilities import AcpMode
from linktools.ai.acp.client_services import AcpClientServices
from linktools.ai.acp.persistence import AcpSessionRepository
from linktools.ai.acp.sessions import AcpSessionService
from linktools.ai.execution.domain import RunStatus
from linktools.ai.execution.live_events import AssistantTextDelta, ExecutionCompleted, ExecutionEventHub
from linktools.ai.governance.identity import trusted_local_principal


def _agent(tmp_path):
    hub = ExecutionEventHub()
    runtime = SimpleNamespace(execution_event_hub=hub)
    session_service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path),
        project_root=tmp_path,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=AcpClientServices(project_root=tmp_path),
    )
    return LinktoolsAcpAgent(
        runtime=runtime,
        event_hub=hub,
        session_service=session_service,
        project_root=str(tmp_path),
        spec_resolver=lambda mode: None,
        modes=(AcpMode("default", "Default"),),
    )


def test_stable_agent_methods_have_one_handler_each(tmp_path) -> None:
    agent = _agent(tmp_path)
    registry = agent.handler_registry()

    assert set(registry) == set(STANDARD_AGENT_METHODS)
    assert len({id(handler) for handler in registry.values()}) == len(registry)


def test_agent_requires_shared_event_hub(tmp_path) -> None:
    hub = ExecutionEventHub()
    runtime = SimpleNamespace(execution_event_hub=hub)
    session_service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path),
        project_root=tmp_path,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=AcpClientServices(project_root=tmp_path),
    )

    with pytest.raises(ValueError, match="one ExecutionEventHub"):
        LinktoolsAcpAgent(
            runtime=runtime,
            event_hub=ExecutionEventHub(),
            session_service=session_service,
            project_root=str(tmp_path),
            spec_resolver=lambda mode: None,
            modes=(AcpMode("default", "Default"),),
        )


@pytest.mark.asyncio
async def test_agent_streams_update_before_prompt_response(tmp_path) -> None:
    hub = ExecutionEventHub()
    runtime = SimpleNamespace(execution_event_hub=hub)

    async def run(spec, prompt, *, principal, session_id, execution_id, extra_toolsets):
        await hub.publish(
            execution_id,
            AssistantTextDelta(execution_id=execution_id, text="hello"),
        )
        await hub.close(execution_id, ExecutionCompleted(execution_id=execution_id))

    runtime.run = run
    async def create_session(session_id, *, principal):
        return None

    async def get_execution_record(execution_id, *, principal):
        return SimpleNamespace(status=RunStatus.COMPLETED, id=execution_id)

    runtime.create_session = create_session
    runtime.get_execution_record = get_execution_record

    async def resolve(mode):
        return object()

    session_service = AcpSessionService(
        runtime=runtime,
        repository=AcpSessionRepository(tmp_path),
        project_root=tmp_path,
        principal=trusted_local_principal(),
        default_mode_id="default",
        client_services=AcpClientServices(project_root=tmp_path),
    )
    agent = LinktoolsAcpAgent(
        runtime=runtime,
        event_hub=hub,
        session_service=session_service,
        project_root=str(tmp_path),
        spec_resolver=resolve,
        modes=(AcpMode("default", "Default"),),
    )
    updates = []

    class Connection:
        async def session_update(self, session_id, update):
            updates.append(update)

    agent.on_connect(Connection())
    agent._initialized = True
    active = await session_service.create(cwd=str(tmp_path))
    response = await agent.prompt(
        active.record.session_id,
        [schema.TextContentBlock(type="text", text="hi")],
    )

    assert response.stop_reason == "end_turn"
    assert [update.session_update for update in updates] == ["agent_message_chunk"]


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
